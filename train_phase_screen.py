"""Train RF -> dual-head low-dimensional propagation correction.

V3 predicts two physically separated quantities:

1. a depth-only cumulative mean-delay profile G(z), and
2. zero-mean relative multilayer phase screens tau_l(x).

The formal default is 4 x 96 relative controls plus 8 mean-delay controls.
``--mean-controls`` remains configurable so the 2/4/8/16 oracle can choose the
smallest sufficient mean branch before a full training run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import corr2d, demod_iq, rf_to_D, to_plain
from models.phase_screen import PhaseScreenModel, coherence, heldout_agreement
from physics.phase_screen import project_mean_delay_controls
from scripts.pilot_phase_asp import (DATA_ROOT, corrected_config, embed,
                                     projected_truth_screen)


def sample_ids(split: str, per_case: int):
    if split == "train":
        ranges = ((0, 199), (200, 399))
    elif split == "val":
        ranges = ((0, 24), (25, 49))
    else:
        raise ValueError(split)
    return [f"{split}_{i:03d}" for lo, hi in ranges
            for i in np.linspace(lo, hi, per_case).round().astype(int)]


def build_cache(ids, model, meta, train_idx, hold_idx, device, top_frac,
                need_reference=True):
    cache = []
    for sample_number, sample_id in enumerate(ids, 1):
        sample = torch.load(DATA_ROOT / "shards" / f"{sample_id}.pt",
                            map_location="cpu", weights_only=False)
        md = sample["metadata"]
        if (md["crop_x_start"], md["crop_z_start"]) != (176, 60):
            raise ValueError(f"{sample_id}: unexpected crop geometry")
        if not np.allclose(md["source_tref_s"], meta.t_ref_s, rtol=0, atol=1e-12):
            raise ValueError(f"{sample_id}: different acquisition timing")
        rf = sample["rf"][None].to(device)
        D = rf_to_D(rf, meta) if need_reference else None
        iq = demod_iq(rf, meta)
        ref = (model.reference(D, train_idx, hold_idx, top_frac)
               if need_reference else None)
        truth_abs = sample["m"].abs().to(device) if need_reference else None

        with torch.no_grad():
            uniform_image11 = None
            if need_reference:
                zero = D.real.new_zeros(1, model.born.nz, model.born.nx)
                all_idx = torch.arange(D.shape[1], device=device)
                images0 = model.angle_images(zero, D, all_idx)
                image0 = images0.mean(dim=1)[0]
                if model.pad:
                    image0 = image0[:, model.pad:-model.pad]
                uniform_image11 = float(corr2d(image0.abs(), truth_abs).item())

            known_ds = embed(sample["delta_s"].to(device), model.pad)
            if model.mean_controls:
                teacher_raw, _, phase_info = projected_truth_screen(
                    known_ds, model.layers, model.controls, model.born.dz,
                    model.limit_us, model.pad, model.born.z0,
                    model.bulk_limit_us, False)
                teacher_mean_raw, teacher_mean_target, mean_info = (
                    project_mean_delay_controls(
                        known_ds, model.mean_controls, model.born.dz,
                        model.pad, model.mean_limit_us))
                teacher_bulk = None
            else:
                teacher_raw, teacher_bulk, phase_info = projected_truth_screen(
                    known_ds, model.layers, model.controls, model.born.dz,
                    model.limit_us, model.pad, model.born.z0,
                    model.bulk_limit_us, model.fit_bulk)
                teacher_mean_raw = known_ds.new_zeros(0)
                teacher_mean_target = known_ds.new_zeros(0)
                mean_info = None

            teacher_tau = model.limit_us * torch.tanh(teacher_raw)
            teacher_mean_us = (model.mean_limit_us * torch.tanh(teacher_mean_raw)
                               if model.mean_controls else teacher_mean_raw)
            teacher_bulk_us = (model.bulk_limit_us * torch.tanh(teacher_bulk)
                               if teacher_bulk is not None else None)

        cache.append({
            "id": sample_id,
            "iq": iq,
            "D": D,
            "ref": ref,
            "truth_abs": truth_abs,
            "teacher_tau": teacher_tau[None],
            "teacher_mean_us": teacher_mean_us[None],
            "teacher_mean_target_us": teacher_mean_target[None],
            "teacher_bulk_us": (teacher_bulk_us[None]
                                if teacher_bulk_us is not None else None),
            "phase_projection": phase_info,
            "mean_projection": mean_info,
            "uniform_image11": uniform_image11,
        })
        if sample_number == 1 or sample_number % 20 == 0 or sample_number == len(ids):
            print(json.dumps({"event": "cached", "sample": sample_id,
                              "count": sample_number, "total": len(ids)}), flush=True)
    return cache


def screen_regularizer(phase_controls_us, limit_us):
    """Weak magnitude prior plus lateral curvature."""
    x = phase_controls_us / limit_us
    magnitude = x.square().mean()
    if x.shape[-1] >= 3:
        d2x = x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]
        curvature = d2x.square().mean()
    else:
        curvature = x.new_zeros(())
    return 0.05 * magnitude + 0.1 * curvature


def mean_regularizer(mean_controls_us, limit_us):
    """Very weak curvature prior on cumulative mean delay G(z)."""
    if mean_controls_us.numel() == 0 or mean_controls_us.shape[-1] < 3:
        return mean_controls_us.new_zeros(())
    x = mean_controls_us / limit_us
    d2 = x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]
    return 0.02 * d2.square().mean()


def regularizer(out, model):
    reg = screen_regularizer(out["phase_controls_us"], model.limit_us)
    if model.mean_controls:
        reg = reg + mean_regularizer(
            out["mean_controls_us"], model.mean_limit_us)
    elif model.fit_bulk:
        reg = reg + 0.1 * torch.tanh(out["raw_bulk"]).square().mean()
    return reg


def teacher_loss(phase_controls_us, mean_controls_us, bulk_coeff_us,
                 item, model):
    """Robust supervised loss on relative and mean propagation delays."""
    loss = F.smooth_l1_loss(
        phase_controls_us / model.limit_us,
        item["teacher_tau"] / model.limit_us,
        beta=0.1,
    )
    if model.mean_controls:
        loss = loss + F.smooth_l1_loss(
            mean_controls_us / model.mean_limit_us,
            item["teacher_mean_us"] / model.mean_limit_us,
            beta=0.1,
        )
    elif model.fit_bulk:
        if item["teacher_bulk_us"] is None:
            raise ValueError("bulk teacher target missing for fit_bulk=True")
        loss = loss + F.smooth_l1_loss(
            bulk_coeff_us / model.bulk_limit_us,
            item["teacher_bulk_us"] / model.bulk_limit_us,
            beta=0.1,
        )
    return loss


def random_context(train_idx, min_angles: int):
    """Random non-empty subset for angle-count/domain randomization."""
    n_total = len(train_idx)
    if not (1 <= min_angles <= n_total):
        raise ValueError("min_context_angles must be within available train angles")
    n = int(torch.randint(min_angles, n_total + 1, (),
                          device=train_idx.device).item())
    perm = torch.randperm(n_total, device=train_idx.device)
    return train_idx[perm[:n]]


@torch.no_grad()
def validate(model, cache, train_idx, hold_idx):
    model.eval()
    rows = []
    for item in cache:
        out = model.forward_precomputed(item["iq"], item["D"], train_idx)
        ref = item["ref"]
        tr = out["images_input"]
        ho = model.angle_images(out["effective_ds"], item["D"], hold_idx)
        input_coh = coherence(tr, ref["mask"], ref["train_scales"])
        hold_agree = heldout_agreement(
            tr, ho, ref["mask"], ref["train_scales"], ref["hold_scales"])
        image11 = torch.cat([tr, ho], dim=1).mean(dim=1)[0]
        if model.pad:
            image11 = image11[:, model.pad:-model.pad]
        rows.append({
            "sample": item["id"],
            "uniform_input": float(ref["uniform_input_coherence"][0]),
            "uniform_hold": float(ref["uniform_holdout_agreement"][0]),
            "input": float(input_coh[0]),
            "hold": float(hold_agree[0]),
            "uniform_image11": item["uniform_image11"],
            "image11": float(corr2d(image11.abs(), item["truth_abs"]).item()),
            "max_phase_control_us": float(out["phase_controls_us"].abs().max()),
            "max_mean_control_us": (float(out["mean_controls_us"].abs().max())
                                    if model.mean_controls else None),
            "mean_end_delay_us": (float(out["mean_delay_profile_us"][0, -1])
                                  if model.mean_controls else None),
            "bulk_us": (out["bulk_coeff_us"][0].tolist()
                        if model.fit_bulk else None),
            "teacher_loss": float(teacher_loss(
                out["phase_controls_us"], out["mean_controls_us"],
                out["bulk_coeff_us"], item, model).item()),
        })
    return {
        "mean_uniform_input": float(np.mean([r["uniform_input"] for r in rows])),
        "mean_uniform_hold": float(np.mean([r["uniform_hold"] for r in rows])),
        "mean_input": float(np.mean([r["input"] for r in rows])),
        "mean_hold": float(np.mean([r["hold"] for r in rows])),
        "mean_uniform_image11": float(np.mean([r["uniform_image11"] for r in rows])),
        "mean_image11": float(np.mean([r["image11"] for r in rows])),
        "hold_wins": sum(r["hold"] > r["uniform_hold"] for r in rows),
        "image11_wins": sum(r["image11"] > r["uniform_image11"] for r in rows),
        "mean_teacher_loss": float(np.mean([r["teacher_loss"] for r in rows])),
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--out", type=Path, default=Path("runs/phase_screen_dual_head"))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--train-per-case", type=int, default=16)
    p.add_argument("--val-per-case", type=int, default=4)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--val-every", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--reg", type=float, default=0.01)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--controls", type=int, default=96)
    p.add_argument("--limit-us", type=float, default=0.5)
    p.add_argument("--mean-controls", type=int, default=8)
    p.add_argument("--mean-limit-us", type=float, default=2.0)
    p.add_argument("--min-context-angles", type=int, default=3)
    p.add_argument("--fit-bulk", action="store_true",
                   help="legacy ablation; requires --mean-controls 0")
    p.add_argument("--objective", choices=("teacher", "coherence", "hybrid"),
                   default="teacher")
    p.add_argument("--teacher-weight", type=float, default=1.0)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260916)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not (1 <= args.train_per_case <= 200 and 1 <= args.val_per_case <= 25):
        p.error("sample counts must fit the two source cases")
    if args.mean_controls < 0:
        p.error("--mean-controls must be non-negative")
    if args.mean_controls and args.fit_bulk:
        p.error("--fit-bulk is legacy-only and requires --mean-controls 0")
    if args.limit_us <= 0 or args.mean_limit_us <= 0:
        p.error("delay limits must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    first = torch.load(DATA_ROOT / "shards" / "train_000.pt",
                       map_location="cpu", weights_only=False)
    cfg, meta = corrected_config(args.config, first, args.n_freq)
    cfg.model.normalize_iq = True
    model = PhaseScreenModel(
        cfg, meta, layers=args.layers, controls=args.controls,
        limit_us=args.limit_us, mean_controls=args.mean_controls,
        mean_limit_us=args.mean_limit_us, fit_bulk=args.fit_bulk).to(device)
    train_idx = torch.as_tensor(meta.train_idx, device=device)
    hold_idx = torch.as_tensor(meta.hold_idx, device=device)
    if not (1 <= args.min_context_angles <= len(train_idx)):
        p.error("--min-context-angles must not exceed available training angles")

    train_ids = sample_ids("train", args.train_per_case)
    val_ids = sample_ids("val", args.val_per_case)
    print(json.dumps({
        "event": "setup",
        "train_samples": train_ids,
        "val_samples": val_ids,
        "n_freq": len(meta.freqs),
        "layers": args.layers,
        "controls": args.controls,
        "limit_us": args.limit_us,
        "mean_controls": args.mean_controls,
        "mean_limit_us": args.mean_limit_us,
        "fit_bulk": args.fit_bulk,
    }), flush=True)

    train_cache = build_cache(
        train_ids, model, meta, train_idx, hold_idx, device, args.top_frac,
        need_reference=args.objective != "teacher")
    val_cache = build_cache(
        val_ids, model, meta, train_idx, hold_idx, device, args.top_frac)

    optimizer = torch.optim.Adam(
        [p for name, p in model.named_parameters()
         if not name.startswith("backbone.fno.head")],
        lr=args.lr)
    history = []
    best_score = float("-inf")
    started = time.monotonic()

    def check(step):
        nonlocal best_score
        report = validate(model, val_cache, train_idx, hold_idx)
        report["step"] = step
        report["elapsed_s"] = time.monotonic() - started
        history.append(report)
        print(json.dumps({
            "event": "validation",
            **{k: v for k, v in report.items() if k != "rows"}
        }), flush=True)
        selection = (-report["mean_teacher_loss"] if args.objective == "teacher"
                     else report["mean_hold"])
        if selection > best_score:
            best_score = selection
            torch.save({
                "model": model.state_dict(),
                "step": step,
                "config": to_plain(cfg),
                "args": vars(args) | {"out": str(args.out)},
                "validation": report,
            }, args.out / "best.pt")
        (args.out / "history.json").write_text(
            json.dumps(history, indent=2) + "\n")

    check(0)
    for step in range(1, args.steps + 1):
        model.train()
        item = train_cache[torch.randint(len(train_cache), ()).item()]
        if args.objective == "teacher":
            ctx_idx = random_context(train_idx, args.min_context_angles)
            phase_raw, mean_raw, bulk_raw = model.predict_components(
                item["iq"], ctx_idx)
            phase = model.limit_us * torch.tanh(phase_raw)
            mean = (model.mean_limit_us * torch.tanh(mean_raw)
                    if model.mean_controls else mean_raw)
            bulk = (model.bulk_limit_us * torch.tanh(bulk_raw)
                    if model.fit_bulk else bulk_raw.new_zeros(bulk_raw.shape))
            sup = teacher_loss(phase, mean, bulk, item, model)
            coh = None
            reg = screen_regularizer(phase, model.limit_us)
            if model.mean_controls:
                reg = reg + mean_regularizer(mean, model.mean_limit_us)
            elif model.fit_bulk:
                reg = reg + 0.1 * torch.tanh(bulk_raw).square().mean()
            loss = sup + args.reg * reg
        else:
            out = model.forward_precomputed(item["iq"], item["D"], train_idx)
            ref = item["ref"]
            coh = coherence(out["images_input"], ref["mask"],
                            ref["train_scales"]).mean()
            reg = regularizer(out, model)
            sup = teacher_loss(
                out["phase_controls_us"], out["mean_controls_us"],
                out["bulk_coeff_us"], item, model)
            loss = -coh + args.reg * reg
            if args.objective == "hybrid":
                loss = loss + args.teacher_weight * sup

        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(json.dumps({
                "event": "train",
                "step": step,
                "sample": item["id"],
                "coherence": (float(coh.detach()) if coh is not None else None),
                "teacher_loss": float(sup.detach()),
                "regularizer": float(reg.detach()),
                "loss": float(loss.detach()),
                "elapsed_s": time.monotonic() - started,
            }), flush=True)
        if step % args.val_every == 0 or step == args.steps:
            check(step)

    print(json.dumps({
        "event": "done",
        "best_selection_score": best_score,
        "elapsed_s": time.monotonic() - started,
    }), flush=True)


if __name__ == "__main__":
    main()

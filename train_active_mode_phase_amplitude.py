"""Train V6 low-dimensional active-mode propagation correction.

Stages
------
phase:
    Train mean-delay + K phase active coefficients with cross-angle agreement.
    Amplitude is fixed to zero.

relative-phase:
    Freeze the warm-started backbone + mean-delay branch, reset the active
    phase head/gate, and train only K relative-active coefficients.  This is
    the clean test of residual phase aberration beyond mean delay.

paired-amplitude:
    Freeze the phase predictor and train only K amplitude coefficients using
    paired simulation data from the same phantom with attenuation removed.
    No cross-angle amplitude self-supervision is used.

The active spatial basis is fixed from population_active_subspace.pt.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import demod_iq, rf_to_D, to_plain
from models.active_mode_phase_amplitude import ActiveModePhaseAmplitudeModel
from models.phase_screen import heldout_agreement
from physics.phase_screen import mean_controls_to_ds
from scripts.pilot_phase_asp import DATA_ROOT, corrected_config
from train_phase_screen import build_cache, sample_ids
from train_phase_screen_cross_angle import split_context_target


def plain_args(args):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def coeff_regularizer(coeff, limit):
    x = coeff / float(limit)
    return x.square().mean()


def smooth_log_envelope(image, kernel: int, eps: float):
    env = image.abs().clamp_min(eps)
    if kernel > 1:
        env = F.avg_pool2d(
            env[:, None], kernel_size=kernel, stride=1,
            padding=kernel // 2)[:, 0]
    return torch.log(env.clamp_min(eps))


def depth_log_energy(image, bins: int, eps: float):
    """Depth-binned log mean envelope; preserves absolute common-mode level."""
    env = image.abs().mean(dim=-1)
    B, nz = env.shape
    edges = torch.linspace(0, nz, bins + 1, device=image.device).round().long()
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        if int(b) <= int(a):
            continue
        rows.append(torch.log(env[:, int(a):int(b)].mean(dim=-1).clamp_min(eps)))
    return torch.stack(rows, dim=-1)


def paired_image_loss(current, reference, smooth_kernel, depth_bins, eps,
                      depth_weight):
    cur_log = smooth_log_envelope(current, smooth_kernel, eps)
    ref_log = smooth_log_envelope(reference, smooth_kernel, eps)
    image = (cur_log - ref_log).abs().mean()
    cur_depth = depth_log_energy(current, depth_bins, eps)
    ref_depth = depth_log_energy(reference, depth_bins, eps)
    depth = (cur_depth - ref_depth).abs().mean()
    return image + depth_weight * depth, image, depth


def reset_active_phase_branch(model, phase_gate_init):
    torch.nn.init.zeros_(model.phase_head.weight)
    torch.nn.init.zeros_(model.phase_head.bias)
    logit = np.log(phase_gate_init / (1.0 - phase_gate_init))
    with torch.no_grad():
        model.phase_gate_logit.fill_(float(logit))


def warm_start_v6(model, checkpoint_path, device, *, reset_active_phase=False,
                  phase_gate_init=0.02):
    """Warm start learned predictor while preserving current fixed basis buffers.

    Active templates are always kept from the newly constructed model.  When
    reset_active_phase is true, the old coefficient head/gate are also skipped
    because they belong to a different active basis.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    current = model.state_dict()
    copied, skipped = [], []
    basis_buffers = {
        "phase_active_templates",
        "amplitude_active_templates",
    }
    for key, value in ckpt["model"].items():
        if key.startswith("born.") or key in basis_buffers:
            skipped.append((key, "physics/basis buffer"))
            continue
        if reset_active_phase and (
            key.startswith("phase_head.") or key == "phase_gate_logit"):
            skipped.append((key, "reset for new relative basis"))
            continue
        if key not in current:
            skipped.append((key, "missing in current model"))
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            skipped.append((key, f"shape {tuple(value.shape)} -> {tuple(current[key].shape)}"))
            continue
        current[key] = value
        copied.append(key)
    model.load_state_dict(current, strict=True)
    if reset_active_phase:
        reset_active_phase_branch(model, phase_gate_init)
    print(json.dumps({
        "event": "warm_start_v6",
        "checkpoint": str(checkpoint_path),
        "source_step": int(ckpt.get("step", -1)),
        "copied_tensors": len(copied),
        "skipped_tensors": len(skipped),
        "reset_active_phase": bool(reset_active_phase),
        "skipped_preview": skipped[:10],
    }), flush=True)
    return ckpt


def set_stage_trainable(model, stage):
    for p in model.parameters():
        p.requires_grad_(False)

    if stage == "phase":
        for p in model.backbone.parameters():
            p.requires_grad_(True)
        if model.mean_head is not None:
            for p in model.mean_head.parameters():
                p.requires_grad_(True)
        for p in model.phase_head.parameters():
            p.requires_grad_(True)
        model.phase_gate_logit.requires_grad_(True)
    elif stage == "relative-phase":
        for p in model.phase_head.parameters():
            p.requires_grad_(True)
        model.phase_gate_logit.requires_grad_(True)
    elif stage == "paired-amplitude":
        for p in model.amplitude_head.parameters():
            p.requires_grad_(True)
        model.amplitude_gate_logit.requires_grad_(True)
    else:
        raise ValueError(stage)


def load_paired_reference_cache(ids, root, meta, device):
    root = Path(root)
    out = {}
    for sid in ids:
        path = root / "shards" / f"{sid}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing paired reference shard: {path}")
        sample = torch.load(path, map_location="cpu", weights_only=False)
        rf = sample["rf"][None].to(device)
        out[sid] = {
            "iq": demod_iq(rf, meta),
            "D": rf_to_D(rf, meta),
        }
    return out


def compound(model, ds, D, idx, amp_rate):
    images = model.angle_images(ds, D, idx, amplitude_rate=amp_rate)
    return images.mean(dim=1)


@torch.no_grad()
def validate_phase(model, cache, train_idx, hold_idx):
    model.eval()
    rows = []
    for item in cache:
        phase_raw, mean_raw, amp_raw = model.predict_all_components(
            item["iq"], train_idx)
        ds_full, amp_rate, phase_coeff, _ = model.network_corrections(
            phase_raw, mean_raw, amp_raw)
        ds_mean = mean_controls_to_ds(
            mean_raw, model.born.nz, model.born.nx, model.born.dz,
            model.mean_limit_us)
        zero_amp = torch.zeros_like(amp_rate)
        ref = item["ref"]

        tr_full = model.angle_images(
            ds_full, item["D"], train_idx, amplitude_rate=zero_amp)
        ho_full = model.angle_images(
            ds_full, item["D"], hold_idx, amplitude_rate=zero_amp)
        full_hold = heldout_agreement(
            tr_full, ho_full, ref["mask"],
            ref["train_scales"], ref["hold_scales"])

        tr_mean = model.angle_images(
            ds_mean, item["D"], train_idx, amplitude_rate=zero_amp)
        ho_mean = model.angle_images(
            ds_mean, item["D"], hold_idx, amplitude_rate=zero_amp)
        mean_hold = heldout_agreement(
            tr_mean, ho_mean, ref["mask"],
            ref["train_scales"], ref["hold_scales"])

        uniform = float(ref["uniform_holdout_agreement"][0])
        full_value = float(full_hold[0])
        mean_value = float(mean_hold[0])
        rows.append({
            "sample": item["id"],
            "uniform_hold": uniform,
            "mean_only_hold": mean_value,
            "phase_hold": full_value,
            "gain_active_on_top_of_mean": full_value - mean_value,
            "max_phase_coeff_us": float(phase_coeff.abs().max()),
        })
    return {
        "mean_uniform_hold": float(np.mean([r["uniform_hold"] for r in rows])),
        "mean_mean_only_hold": float(np.mean([r["mean_only_hold"] for r in rows])),
        "mean_phase_hold": float(np.mean([r["phase_hold"] for r in rows])),
        "delta_phase_vs_uniform": float(np.mean([
            r["phase_hold"] - r["uniform_hold"] for r in rows])),
        "delta_active_on_top_of_mean": float(np.mean([
            r["gain_active_on_top_of_mean"] for r in rows])),
        "active_wins_vs_mean_count": int(sum(
            r["phase_hold"] > r["mean_only_hold"] for r in rows)),
        "phase_gate": float(model.phase_gate_value()),
        "rows": rows,
    }


@torch.no_grad()
def validate_paired_amplitude(model, cache, paired, train_idx,
                              smooth_kernel, depth_bins, eps, depth_weight):
    model.eval()
    rows = []
    for item in cache:
        phase_raw, mean_raw, amp_raw = model.predict_all_components(
            item["iq"], train_idx)
        ds, amp_rate, phase_coeff, amp_coeff = model.network_corrections(
            phase_raw, mean_raw, amp_raw)
        zero_amp = torch.zeros_like(amp_rate)
        current = compound(model, ds, item["D"], train_idx, amp_rate)
        phase_only = compound(model, ds, item["D"], train_idx, zero_amp)
        reference = compound(model, ds, paired[item["id"]]["D"], train_idx, zero_amp)
        full_loss, _, _ = paired_image_loss(
            current, reference, smooth_kernel, depth_bins, eps, depth_weight)
        phase_loss, _, _ = paired_image_loss(
            phase_only, reference, smooth_kernel, depth_bins, eps, depth_weight)
        rows.append({
            "sample": item["id"],
            "phase_only_loss": float(phase_loss),
            "full_loss": float(full_loss),
            "gain": float(phase_loss - full_loss),
            "max_amplitude_coeff_np": float(amp_coeff.abs().max()),
        })
    return {
        "mean_phase_only_loss": float(np.mean([r["phase_only_loss"] for r in rows])),
        "mean_full_loss": float(np.mean([r["full_loss"] for r in rows])),
        "mean_gain": float(np.mean([r["gain"] for r in rows])),
        "amplitude_gate": float(model.amplitude_gate_value()),
        "rows": rows,
    }


def checkpoint_payload(model, optimizer, step, cfg, args, report, history, best_score):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": to_plain(cfg),
        "args": plain_args(args),
        "validation": report,
        "history": history,
        "best_score": float(best_score),
        "training_format": "active_mode_phase_amplitude_v1",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--active-basis", type=Path, required=True)
    p.add_argument("--active-rank", type=int, default=6)
    p.add_argument("--active-basis-source", choices=("balanced", "phase", "amplitude"),
                   default="balanced")
    p.add_argument("--stage", choices=("phase", "relative-phase", "paired-amplitude"), default="phase")
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--paired-reference-root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--train-per-case", type=int, default=200)
    p.add_argument("--val-per-case", type=int, default=25)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mean-controls", type=int, default=8)
    p.add_argument("--mean-limit-us", type=float, default=2.0)
    p.add_argument("--phase-coeff-limit-us", type=float, default=0.2)
    p.add_argument("--amplitude-coeff-limit-np", type=float, default=0.2)
    p.add_argument("--phase-gate-init", type=float, default=0.02)
    p.add_argument("--amplitude-gate-init", type=float, default=0.02)
    p.add_argument("--amplitude-freq-power", type=float, default=1.0)
    p.add_argument("--min-context-angles", type=int, default=3)
    p.add_argument("--target-angles", type=int, default=2)
    p.add_argument("--coeff-reg", type=float, default=1e-3)
    p.add_argument("--smooth-kernel", type=int, default=9)
    p.add_argument("--depth-bins", type=int, default=12)
    p.add_argument("--depth-weight", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260920)
    args = p.parse_args()

    if args.stage == "paired-amplitude" and args.paired_reference_root is None:
        p.error("--paired-reference-root is required for paired-amplitude stage")
    if args.active_rank < 1 or args.steps < 1 or args.val_every < 1:
        p.error("rank/steps/val-every must be positive")
    if args.smooth_kernel < 1 or args.smooth_kernel % 2 == 0:
        p.error("--smooth-kernel must be a positive odd integer")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    first = torch.load(DATA_ROOT / "shards" / "train_000.pt",
                       map_location="cpu", weights_only=False)
    cfg, meta = corrected_config(args.config, first, args.n_freq)
    cfg.model.normalize_iq = True
    model = ActiveModePhaseAmplitudeModel(
        cfg, meta,
        active_basis_path=args.active_basis,
        active_rank=args.active_rank,
        active_basis_source=args.active_basis_source,
        mean_controls=args.mean_controls,
        mean_limit_us=args.mean_limit_us,
        phase_coeff_limit_us=args.phase_coeff_limit_us,
        amplitude_coeff_limit_np=args.amplitude_coeff_limit_np,
        phase_gate_init=args.phase_gate_init,
        amplitude_gate_init=args.amplitude_gate_init,
        amplitude_freq_power=args.amplitude_freq_power,
    ).to(device)
    warm_start_v6(
        model, args.init_checkpoint, device,
        reset_active_phase=(args.stage == "relative-phase"),
        phase_gate_init=args.phase_gate_init)
    set_stage_trainable(model, args.stage)

    train_idx = torch.as_tensor(meta.train_idx, device=device)
    hold_idx = torch.as_tensor(meta.hold_idx, device=device)
    train_ids = sample_ids("train", args.train_per_case)
    val_ids = sample_ids("val", args.val_per_case)
    train_cache = build_cache(
        train_ids, model, meta, train_idx, hold_idx, device,
        args.top_frac, need_reference=True)
    val_cache = build_cache(
        val_ids, model, meta, train_idx, hold_idx, device,
        args.top_frac, need_reference=True)

    paired_train = paired_val = None
    if args.stage == "paired-amplitude":
        paired_train = load_paired_reference_cache(
            train_ids, args.paired_reference_root, meta, device)
        paired_val = load_paired_reference_cache(
            val_ids, args.paired_reference_root, meta, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    history = []
    best_score = float("-inf")
    started = time.monotonic()

    def check(step):
        nonlocal best_score
        if args.stage in ("phase", "relative-phase"):
            report = validate_phase(model, val_cache, train_idx, hold_idx)
            score = (
                report["delta_active_on_top_of_mean"]
                if args.stage == "relative-phase"
                else report["delta_phase_vs_uniform"]
            )
        else:
            report = validate_paired_amplitude(
                model, val_cache, paired_val, train_idx,
                args.smooth_kernel, args.depth_bins, args.eps, args.depth_weight)
            score = report["mean_gain"]
        report["step"] = int(step)
        report["elapsed_s"] = time.monotonic() - started
        history.append(report)
        improved = score > best_score
        if improved:
            best_score = float(score)
        payload = checkpoint_payload(
            model, optimizer, step, cfg, args, report, history, best_score)
        torch.save(payload, args.out / "last.pt")
        if improved:
            torch.save(payload, args.out / "best.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({
            "event": "validation", "stage": args.stage, "step": step,
            "score": score, "best_score": best_score, "improved": improved,
            "phase_gate": float(model.phase_gate_value()),
            "amplitude_gate": float(model.amplitude_gate_value()),
        }), flush=True)

    print(json.dumps({
        "event": "setup",
        "stage": args.stage,
        "active_rank": args.active_rank,
        "active_basis_source": args.active_basis_source,
        "candidate_dim": model.active_candidate_dim,
        "active_relative_only": model.active_relative_only,
        "active_depths_mm": model.active_depths_mm,
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
    }), flush=True)
    check(0)

    for step in range(1, args.steps + 1):
        model.train()
        item = train_cache[torch.randint(len(train_cache), ()).item()]

        if args.stage in ("phase", "relative-phase"):
            ctx_pos, tgt_pos, ctx_idx, tgt_idx = split_context_target(
                train_idx, args.min_context_angles, args.target_angles)
            phase_raw, mean_raw, amp_raw = model.predict_all_components(
                item["iq"], ctx_idx)
            ds, amp_rate, phase_coeff, _ = model.network_corrections(
                phase_raw, mean_raw, amp_raw)
            zero_amp = torch.zeros_like(amp_rate)
            ctx = model.angle_images(ds, item["D"], ctx_idx, amplitude_rate=zero_amp)
            tgt = model.angle_images(ds, item["D"], tgt_idx, amplitude_rate=zero_amp)
            ref = item["ref"]
            cross = heldout_agreement(
                ctx, tgt, ref["mask"],
                ref["train_scales"][:, ctx_pos],
                ref["train_scales"][:, tgt_pos]).mean()
            prior = coeff_regularizer(phase_coeff, model.phase_coeff_limit_us)
            loss = -cross + args.coeff_reg * prior
            aux = {"agreement": float(cross.detach()), "prior": float(prior.detach())}
        else:
            phase_raw, mean_raw, amp_raw = model.predict_all_components(
                item["iq"], train_idx)
            ds, amp_rate, _, amp_coeff = model.network_corrections(
                phase_raw, mean_raw, amp_raw)
            current = compound(model, ds, item["D"], train_idx, amp_rate)
            zero_amp = torch.zeros_like(amp_rate)
            reference = compound(
                model, ds, paired_train[item["id"]]["D"], train_idx, zero_amp)
            task, image_loss, depth_loss = paired_image_loss(
                current, reference, args.smooth_kernel, args.depth_bins,
                args.eps, args.depth_weight)
            prior = coeff_regularizer(
                amp_coeff, model.amplitude_coeff_limit_np)
            loss = task + args.coeff_reg * prior
            aux = {
                "image_loss": float(image_loss.detach()),
                "depth_loss": float(depth_loss.detach()),
                "prior": float(prior.detach()),
            }

        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(json.dumps({
                "event": "train", "stage": args.stage, "step": step,
                "sample": item["id"], "loss": float(loss.detach()),
                **aux,
            }), flush=True)
        if step % args.val_every == 0 or step == args.steps:
            check(step)

    print(json.dumps({
        "event": "done", "stage": args.stage,
        "best_score": best_score,
        "elapsed_s": time.monotonic() - started,
    }), flush=True)


if __name__ == "__main__":
    main()

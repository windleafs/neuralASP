"""Train V7 amplitude-aware predictor on matched attenuated/no-att RF pairs.

V7 deliberately ends the relative-phase line.  The frozen V6 backbone predicts
only the depth-mean phase correction.  Mean-phase-corrected attenuated images
are converted into a 48-D amplitude descriptor (8 depth bins x 6 lateral DCT
modes), then a tiny MLP predicts 6 amplitude active-mode coefficients.

No per-sample amplitude normalization is used in the descriptor.  A single
training-set scale and feature standardization are frozen into the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import demod_iq, rf_to_D, to_plain
from models.active_mode_phase_amplitude import ActiveModePhaseAmplitudeModel
from models.v7_amplitude_predictor import V7AmplitudePredictor
from physics.paired_amplitude_loss import paired_image_loss
from physics.phase_screen import mean_controls_to_ds
from scripts.pilot_phase_asp import DATA_ROOT, corrected_config
from train_phase_screen import sample_ids


def plain_args(args):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def physical_crop(image, pad):
    return image[..., pad:-pad] if pad else image


def copy_frozen_mean_phase(model, checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    current = model.state_dict()
    copied, skipped = [], []
    for key, value in ckpt["model"].items():
        keep = key.startswith("backbone.") or key.startswith("mean_head.")
        if not keep:
            skipped.append(key)
            continue
        if key not in current or tuple(current[key].shape) != tuple(value.shape):
            raise RuntimeError(f"cannot copy frozen mean-phase tensor {key}")
        current[key] = value
        copied.append(key)
    model.load_state_dict(current, strict=True)
    if not copied:
        raise RuntimeError("copied zero mean-phase tensors")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return ckpt, copied, skipped


def build_frozen_base(args, first, device):
    ckpt = torch.load(args.phase_checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    cfg, meta = corrected_config(args.config, first, args.n_freq)
    # Must match the phase checkpoint input convention.
    cfg.model.normalize_iq = True
    model = ActiveModePhaseAmplitudeModel(
        cfg, meta,
        active_basis_path=args.active_basis,
        active_rank=args.active_rank,
        active_basis_source="amplitude",
        mean_controls=int(saved.get("mean_controls", 8)),
        mean_limit_us=float(saved.get("mean_limit_us", 2.0)),
        phase_coeff_limit_us=float(saved.get("phase_coeff_limit_us", 0.2)),
        amplitude_coeff_limit_np=args.coeff_limit_np,
        phase_gate_init=float(saved.get("phase_gate_init", 0.02)),
        amplitude_gate_init=float(saved.get("amplitude_gate_init", 0.02)),
        amplitude_freq_power=float(saved.get("amplitude_freq_power", 1.0)),
    ).to(device)
    _, copied, _ = copy_frozen_mean_phase(model, args.phase_checkpoint, device)
    return model, cfg, meta, copied


def require_pairs(ids, noatt_root):
    missing = [sid for sid in ids
               if not (Path(noatt_root) / "shards" / f"{sid}.pt").exists()]
    if missing:
        preview = ", ".join(missing[:12])
        raise FileNotFoundError(
            f"missing {len(missing)} matched no-att shard(s): {preview}. "
            "Generate/pack the paired dataset before V7 training.")


@torch.no_grad()
def prepare_cache(ids, base, meta, noatt_root, phase_idx, image_idx, device, args):
    cache = []
    for n, sid in enumerate(ids, 1):
        att = torch.load(DATA_ROOT / "shards" / f"{sid}.pt",
                         map_location="cpu", weights_only=False)
        noatt = torch.load(Path(noatt_root) / "shards" / f"{sid}.pt",
                           map_location="cpu", weights_only=False)
        rf_att = att["rf"][None].to(device)
        rf_noatt = noatt["rf"][None].to(device)
        rel_rf = float(
            (rf_att - rf_noatt).pow(2).mean().sqrt()
            / rf_att.pow(2).mean().sqrt().clamp_min(1e-20))
        if rel_rf < 1e-7:
            raise RuntimeError(f"{sid}: attenuated/no-att RF are numerically identical")

        iq = demod_iq(rf_att, meta)
        D_att = rf_to_D(rf_att, meta)
        D_noatt = rf_to_D(rf_noatt, meta)
        _, mean_raw, _ = base.predict_all_components(iq, phase_idx)
        ds = mean_controls_to_ds(
            mean_raw, base.born.nz, base.born.nx, base.born.dz,
            base.mean_limit_us)
        zero_amp = torch.zeros_like(ds)
        phase_images = base.angle_images(
            ds, D_att, image_idx, amplitude_rate=zero_amp)
        envelope = physical_crop(phase_images.abs().mean(dim=1), base.pad)
        baseline = physical_crop(
            phase_images.mean(dim=1), base.pad)
        reference_images = base.angle_images(
            ds, D_noatt, image_idx, amplitude_rate=zero_amp)
        reference = physical_crop(reference_images.mean(dim=1), base.pad)
        base_loss, _, _, _ = paired_image_loss(
            baseline, reference, args.smooth_kernel, args.loss_depth_bins,
            args.loss_eps, args.depth_weight)
        cache.append({
            "id": sid,
            "D_att": D_att,
            "ds": ds,
            "envelope": envelope,
            "reference": reference,
            "baseline_loss": float(base_loss),
            "rf_relative_difference": rel_rf,
        })
        print(json.dumps({
            "event": "v7_cache", "sample": sid, "index": n, "total": len(ids),
            "baseline_loss": float(base_loss),
            "envelope_mean": float(envelope.mean()),
            "rf_relative_difference": rel_rf,
        }), flush=True)
    return cache


@torch.no_grad()
def fit_descriptor_stats(predictor, train_cache):
    means = torch.stack([x["envelope"].mean() for x in train_cache])
    scale = float(means.median().clamp_min(1e-20))
    predictor.descriptor_scale.fill_(scale)
    raw = torch.cat([predictor.raw_descriptor(x["envelope"]) for x in train_cache], dim=0)
    mean = raw.mean(dim=0)
    std = raw.std(dim=0, unbiased=False).clamp_min(1e-4)
    predictor.set_descriptor_stats(scale, mean, std)
    for item in train_cache:
        item["descriptor"] = predictor.raw_descriptor(item.pop("envelope")).detach()
    return {
        "scale": scale,
        "mean_abs_feature_mean": float(mean.abs().mean()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
    }


@torch.no_grad()
def attach_val_descriptors(predictor, cache):
    for item in cache:
        item["descriptor"] = predictor.raw_descriptor(item.pop("envelope")).detach()


def amplitude_rate(base, coeff):
    templates = base.amplitude_active_templates.to(coeff)
    return torch.einsum("bk,kzx->bzx", coeff, templates)


def corrected_image(base, item, image_idx, coeff):
    rate = amplitude_rate(base, coeff)
    images = base.angle_images(
        item["ds"], item["D_att"], image_idx, amplitude_rate=rate)
    return physical_crop(images.mean(dim=1), base.pad)


@torch.no_grad()
def validate(predictor, base, cache, image_idx, args):
    predictor.eval()
    rows = []
    for item in cache:
        coeff, _ = predictor.forward_descriptor(item["descriptor"])
        current = corrected_image(base, item, image_idx, coeff)
        loss, image_loss, depth_loss, _ = paired_image_loss(
            current, item["reference"], args.smooth_kernel,
            args.loss_depth_bins, args.loss_eps, args.depth_weight)
        baseline = item["baseline_loss"]
        value = float(loss)
        gain = baseline - value
        rows.append({
            "sample": item["id"],
            "baseline_loss": baseline,
            "corrected_loss": value,
            "gain": gain,
            "relative_reduction": gain / max(baseline, 1e-12),
            "image_loss": float(image_loss),
            "depth_loss": float(depth_loss),
            "coeff_np": coeff[0].cpu().tolist(),
            "max_abs_coeff_np": float(coeff.abs().max()),
        })
    return {
        "n": len(rows),
        "mean_baseline_loss": float(np.mean([r["baseline_loss"] for r in rows])),
        "mean_corrected_loss": float(np.mean([r["corrected_loss"] for r in rows])),
        "mean_gain": float(np.mean([r["gain"] for r in rows])),
        "mean_relative_reduction": float(np.mean([r["relative_reduction"] for r in rows])),
        "wins": int(sum(r["gain"] > 0 for r in rows)),
        "rows": rows,
    }


def checkpoint_payload(predictor, optimizer, step, cfg, args, report, history, best):
    return {
        "model": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": to_plain(cfg),
        "args": plain_args(args),
        "validation": report,
        "history": history,
        "best_score": float(best),
        "training_format": "v7_amplitude_descriptor_predictor_v1",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--phase-checkpoint", type=Path,
                   default=Path("runs/v6_active_mode_phase/best.pt"))
    p.add_argument("--active-basis", type=Path,
                   default=Path("runs/multisample_active_subspace_refined/population_active_subspace.pt"))
    p.add_argument("--paired-reference-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/v7_amplitude_predictor"))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--active-rank", type=int, default=6)
    p.add_argument("--descriptor-depth-bins", type=int, default=8)
    p.add_argument("--descriptor-lateral-modes", type=int, default=6)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--coeff-limit-np", type=float, default=0.5)
    p.add_argument("--descriptor-eps", type=float, default=1e-5)
    p.add_argument("--train-per-case", type=int, default=25,
                   help="25 -> 50 paired training samples total")
    p.add_argument("--val-per-case", type=int, default=5,
                   help="5 -> 10 paired validation samples total")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--coeff-reg", type=float, default=1e-4)
    p.add_argument("--smooth-kernel", type=int, default=9)
    p.add_argument("--loss-depth-bins", type=int, default=12)
    p.add_argument("--depth-weight", type=float, default=0.5)
    p.add_argument("--loss-eps", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=20260920)
    args = p.parse_args()

    if args.active_rank < 1 or args.steps < 1 or args.val_every < 1:
        p.error("rank/steps/val-every must be positive")
    if args.smooth_kernel < 1 or args.smooth_kernel % 2 == 0:
        p.error("--smooth-kernel must be a positive odd integer")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    train_ids = sample_ids("train", args.train_per_case)
    val_ids = sample_ids("val", args.val_per_case)
    require_pairs(train_ids + val_ids, args.paired_reference_root)

    first = torch.load(DATA_ROOT / "shards" / f"{train_ids[0]}.pt",
                       map_location="cpu", weights_only=False)
    base, cfg, meta, copied = build_frozen_base(args, first, device)
    phase_idx = torch.as_tensor(meta.train_idx, device=device)
    image_idx = torch.arange(meta.n_angles if hasattr(meta, "n_angles") else len(meta.angles_deg),
                             device=device)

    train_cache = prepare_cache(
        train_ids, base, meta, args.paired_reference_root,
        phase_idx, image_idx, device, args)
    val_cache = prepare_cache(
        val_ids, base, meta, args.paired_reference_root,
        phase_idx, image_idx, device, args)

    predictor = V7AmplitudePredictor(
        active_rank=args.active_rank,
        depth_bins=args.descriptor_depth_bins,
        lateral_modes=args.descriptor_lateral_modes,
        hidden=args.hidden,
        coeff_limit_np=args.coeff_limit_np,
        descriptor_eps=args.descriptor_eps,
    ).to(device)
    descriptor_stats = fit_descriptor_stats(predictor, train_cache)
    attach_val_descriptors(predictor, val_cache)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    history = []
    best_score = float("-inf")
    started = time.monotonic()

    def check(step):
        nonlocal best_score
        report = validate(predictor, base, val_cache, image_idx, args)
        report["step"] = int(step)
        report["elapsed_s"] = time.monotonic() - started
        history.append(report)
        score = report["mean_relative_reduction"]
        improved = score > best_score
        if improved:
            best_score = float(score)
        payload = checkpoint_payload(
            predictor, optimizer, step, cfg, args, report, history, best_score)
        torch.save(payload, args.out / "last.pt")
        if improved:
            torch.save(payload, args.out / "best.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({
            "event": "v7_validation", "step": step,
            "mean_baseline_loss": report["mean_baseline_loss"],
            "mean_corrected_loss": report["mean_corrected_loss"],
            "mean_relative_reduction": report["mean_relative_reduction"],
            "wins": report["wins"], "best_score": best_score,
            "improved": improved,
        }), flush=True)

    print(json.dumps({
        "event": "v7_setup",
        "train_samples": len(train_cache),
        "val_samples": len(val_cache),
        "active_rank": args.active_rank,
        "descriptor_dim": predictor.descriptor_dim,
        "descriptor_depth_bins": args.descriptor_depth_bins,
        "descriptor_lateral_modes": args.descriptor_lateral_modes,
        "descriptor_stats": descriptor_stats,
        "frozen_mean_phase_tensors": len(copied),
        "trainable_parameters": sum(p.numel() for p in predictor.parameters()),
    }), flush=True)
    check(0)

    for step in range(1, args.steps + 1):
        predictor.train()
        item = train_cache[torch.randint(len(train_cache), ()).item()]
        coeff, _ = predictor.forward_descriptor(item["descriptor"])
        current = corrected_image(base, item, image_idx, coeff)
        task, image_loss, depth_loss, _ = paired_image_loss(
            current, item["reference"], args.smooth_kernel,
            args.loss_depth_bins, args.loss_eps, args.depth_weight)
        prior = (coeff / args.coeff_limit_np).square().mean()
        loss = task + args.coeff_reg * prior
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite V7 loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(json.dumps({
                "event": "v7_train", "step": step, "sample": item["id"],
                "loss": float(loss.detach()),
                "image_loss": float(image_loss.detach()),
                "depth_loss": float(depth_loss.detach()),
                "prior": float(prior.detach()),
                "max_abs_coeff_np": float(coeff.detach().abs().max()),
            }), flush=True)
        if step % args.val_every == 0 or step == args.steps:
            check(step)

    print(json.dumps({
        "event": "v7_done", "best_score": best_score,
        "elapsed_s": time.monotonic() - started,
        "best": str(args.out / "best.pt"),
    }), flush=True)


if __name__ == "__main__":
    main()

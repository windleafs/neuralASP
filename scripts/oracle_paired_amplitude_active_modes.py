"""Paired-simulation oracle for the low-dimensional amplitude active modes.

Purpose
-------
Before training RF -> amplitude coefficients, test whether the fixed K-mode
amplitude parameterization itself can correct attenuation when the optimizer is
allowed to choose the best coefficients independently for each sample.

Phase correction is intentionally mean-delay only.  The previously tested
relative phase branch is not used.

For each sample:
  1. predict/freeze mean-delay from the V6 phase checkpoint;
  2. form paired images from attenuated and no-attenuation RF;
  3. optimize only K amplitude coefficients on a reduced frequency grid;
  4. transfer the coefficients unchanged to a fresh full-band operator;
  5. report baseline vs corrected paired image/depth losses.

The no-attenuation shard must use the same sample id / phantom realization as
the attenuated shard (same c, rho, scatterers, geometry, acquisition).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from common import demod_iq, rf_to_D
from models.active_mode_phase_amplitude import ActiveModePhaseAmplitudeModel
from physics.phase_screen import mean_controls_to_ds
from scripts.pilot_phase_asp import DATA_ROOT, corrected_config


def smooth_log_envelope(image, kernel: int, eps: float):
    env = image.abs().clamp_min(eps)
    if kernel > 1:
        env = F.avg_pool2d(
            env[:, None], kernel_size=kernel, stride=1,
            padding=kernel // 2)[:, 0]
    return torch.log(env.clamp_min(eps))


def depth_log_energy(image, bins: int, eps: float):
    env = image.abs().mean(dim=-1)
    _, nz = env.shape
    edges = torch.linspace(0, nz, bins + 1, device=image.device).round().long()
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        if int(b) <= int(a):
            continue
        rows.append(torch.log(env[:, int(a):int(b)].mean(dim=-1).clamp_min(eps)))
    return torch.stack(rows, dim=-1)


def paired_loss(current, reference, smooth_kernel, depth_bins, eps, depth_weight):
    cur_log = smooth_log_envelope(current, smooth_kernel, eps)
    ref_log = smooth_log_envelope(reference, smooth_kernel, eps)
    image = (cur_log - ref_log).abs().mean()
    cur_depth = depth_log_energy(current, depth_bins, eps)
    ref_depth = depth_log_energy(reference, depth_bins, eps)
    depth = (cur_depth - ref_depth).abs().mean()
    total = image + depth_weight * depth
    return total, image, depth


def physical_crop(image, pad):
    if pad <= 0:
        return image
    return image[..., pad:-pad]


def copy_mean_phase_weights(model, checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    current = model.state_dict()
    copied = []
    for key, value in ckpt["model"].items():
        if key.startswith("born."):
            continue
        if key in ("phase_active_templates", "amplitude_active_templates"):
            continue
        if key.startswith("phase_head.") or key.startswith("amplitude_head."):
            continue
        if key in ("phase_gate_logit", "amplitude_gate_logit"):
            continue
        if key not in current or tuple(current[key].shape) != tuple(value.shape):
            continue
        current[key] = value
        copied.append(key)
    model.load_state_dict(current, strict=True)
    if not copied:
        raise RuntimeError("copied zero mean-phase predictor tensors")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return ckpt, copied


def build_model(checkpoint, active_basis, active_rank, basis_source,
                first_sample, n_freq, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    config_path = saved.get("config", "configs/l11_ultrawave_500_11angle.yaml")
    cfg, meta = corrected_config(config_path, first_sample, n_freq)
    cfg.model.normalize_iq = True
    model = ActiveModePhaseAmplitudeModel(
        cfg, meta,
        active_basis_path=active_basis,
        active_rank=active_rank,
        active_basis_source=basis_source,
        mean_controls=int(saved.get("mean_controls", 8)),
        mean_limit_us=float(saved.get("mean_limit_us", 2.0)),
        phase_coeff_limit_us=float(saved.get("phase_coeff_limit_us", 0.2)),
        amplitude_coeff_limit_np=float(saved.get("amplitude_coeff_limit_np", 0.2)),
        phase_gate_init=float(saved.get("phase_gate_init", 0.02)),
        amplitude_gate_init=float(saved.get("amplitude_gate_init", 0.02)),
        amplitude_freq_power=float(saved.get("amplitude_freq_power", 1.0)),
    ).to(device)
    _, copied = copy_mean_phase_weights(model, checkpoint, device)
    return model, cfg, meta, copied


def load_pair(att_root, noatt_root, sample_id, device):
    att_path = Path(att_root) / "shards" / f"{sample_id}.pt"
    noatt_path = Path(noatt_root) / "shards" / f"{sample_id}.pt"
    if not att_path.exists():
        raise FileNotFoundError(att_path)
    if not noatt_path.exists():
        raise FileNotFoundError(noatt_path)
    att = torch.load(att_path, map_location="cpu", weights_only=False)
    noatt = torch.load(noatt_path, map_location="cpu", weights_only=False)
    if tuple(att["rf"].shape) != tuple(noatt["rf"].shape):
        raise ValueError(f"{sample_id}: paired RF shapes differ")
    return att, noatt


@torch.no_grad()
def mean_phase_ds(model, meta, rf_att, train_idx):
    iq = demod_iq(rf_att[None], meta)
    _, mean_raw, _ = model.predict_all_components(iq, train_idx)
    ds = mean_controls_to_ds(
        mean_raw, model.born.nz, model.born.nx, model.born.dz,
        model.mean_limit_us)
    return ds, mean_raw


def compound(model, D, ds, amp_rate, idx):
    return model.angle_images(ds, D, idx, amplitude_rate=amp_rate).mean(dim=1)


def amplitude_rate_from_raw(model, raw, limit_np):
    coeff = float(limit_np) * torch.tanh(raw)
    templates = model.amplitude_active_templates.to(
        device=raw.device, dtype=raw.dtype)
    rate = torch.einsum("bk,kzx->bzx", coeff, templates)
    return coeff, rate


def evaluate_loss(model, meta, att, noatt, raw, limit_np, args):
    device = next(model.parameters()).device
    rf_att = att["rf"].to(device)
    rf_noatt = noatt["rf"].to(device)
    D_att = rf_to_D(rf_att[None], meta)
    D_noatt = rf_to_D(rf_noatt[None], meta)
    train_idx = torch.as_tensor(meta.train_idx, device=device)
    all_idx = torch.arange(D_att.shape[1], device=device)

    with torch.no_grad():
        ds, mean_raw = mean_phase_ds(model, meta, rf_att, train_idx)
        zero_amp = torch.zeros_like(ds)
        baseline = physical_crop(
            compound(model, D_att, ds, zero_amp, all_idx), model.pad)
        reference = physical_crop(
            compound(model, D_noatt, ds, zero_amp, all_idx), model.pad)

    coeff, amp_rate = amplitude_rate_from_raw(model, raw, limit_np)
    current = physical_crop(
        compound(model, D_att, ds, amp_rate, all_idx), model.pad)
    total, image, depth = paired_loss(
        current, reference, args.smooth_kernel, args.depth_bins,
        args.eps, args.depth_weight)
    with torch.no_grad():
        base_total, base_image, base_depth = paired_loss(
            baseline, reference, args.smooth_kernel, args.depth_bins,
            args.eps, args.depth_weight)
    return {
        "total": total,
        "image": image,
        "depth": depth,
        "baseline_total": base_total,
        "baseline_image": base_image,
        "baseline_depth": base_depth,
        "coeff": coeff,
        "mean_raw": mean_raw,
    }


def optimize_sample(model, meta, att, noatt, args, sample_id):
    device = next(model.parameters()).device
    raw = torch.zeros(
        1, args.active_rank, device=device, dtype=torch.float32,
        requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=args.lr)
    best = None
    history = []

    for step in range(args.steps + 1):
        out = evaluate_loss(
            model, meta, att, noatt, raw, args.amplitude_limit_np, args)
        prior = (out["coeff"] / args.amplitude_limit_np).square().mean()
        objective = out["total"] + args.coeff_reg * prior
        row = {
            "step": int(step),
            "paired_loss": float(out["total"].detach()),
            "image_loss": float(out["image"].detach()),
            "depth_loss": float(out["depth"].detach()),
            "baseline_paired_loss": float(out["baseline_total"].detach()),
            "gain": float((out["baseline_total"] - out["total"]).detach()),
            "prior": float(prior.detach()),
            "max_abs_coeff_np": float(out["coeff"].detach().abs().max()),
        }
        if step == 0 or step % args.log_every == 0 or step == args.steps:
            history.append(row)
            print(json.dumps({
                "event": "paired_amplitude_oracle_step",
                "sample": sample_id,
                **row,
            }), flush=True)

        score = float(objective.detach())
        if best is None or score < best["objective"]:
            best = {
                "objective": score,
                "step": int(step),
                "raw": raw.detach().clone(),
                "coeff_np": out["coeff"].detach().clone(),
                "paired_loss": float(out["total"].detach()),
                "image_loss": float(out["image"].detach()),
                "depth_loss": float(out["depth"].detach()),
                "baseline_paired_loss": float(out["baseline_total"].detach()),
                "baseline_image_loss": float(out["baseline_image"].detach()),
                "baseline_depth_loss": float(out["baseline_depth"].detach()),
            }

        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_([raw], args.grad_clip)
        optimizer.step()

    best["history"] = history
    return best


@torch.no_grad()
def fullband_evaluate(checkpoint, active_basis, basis_source, active_rank,
                      att, noatt, best_raw, args, device):
    model, cfg, meta, copied = build_model(
        checkpoint, active_basis, active_rank, basis_source,
        att, args.eval_n_freq, device)
    out = evaluate_loss(
        model, meta, att, noatt, best_raw.to(device),
        args.amplitude_limit_np, args)
    return {
        "baseline_paired_loss": float(out["baseline_total"]),
        "baseline_image_loss": float(out["baseline_image"]),
        "baseline_depth_loss": float(out["baseline_depth"]),
        "paired_loss": float(out["total"]),
        "image_loss": float(out["image"]),
        "depth_loss": float(out["depth"]),
        "gain": float(out["baseline_total"] - out["total"]),
        "coeff_np": out["coeff"][0].cpu().tolist(),
        "max_abs_coeff_np": float(out["coeff"].abs().max()),
        "copied_mean_phase_tensors": len(copied),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--active-basis", type=Path, required=True)
    p.add_argument("--active-basis-source", choices=("amplitude", "balanced"),
                   default="amplitude")
    p.add_argument("--active-rank", type=int, default=6)
    p.add_argument("--attenuated-root", type=Path, default=DATA_ROOT)
    p.add_argument("--noatt-root", type=Path, required=True)
    p.add_argument("--sample-ids", nargs="+",
                   default=["val_000", "val_025", "val_049"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--opt-n-freq", type=int, default=64)
    p.add_argument("--eval-n-freq", type=int, default=0,
                   help="0 = contiguous full imaging band")
    p.add_argument("--amplitude-limit-np", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--coeff-reg", type=float, default=1e-4)
    p.add_argument("--smooth-kernel", type=int, default=9)
    p.add_argument("--depth-bins", type=int, default=12)
    p.add_argument("--depth-weight", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.active_rank < 1:
        p.error("--active-rank must be positive")
    if args.amplitude_limit_np <= 0:
        p.error("--amplitude-limit-np must be positive")
    if args.smooth_kernel < 1 or args.smooth_kernel % 2 == 0:
        p.error("--smooth-kernel must be a positive odd integer")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    first_att, _ = load_pair(
        args.attenuated_root, args.noatt_root, args.sample_ids[0], device)
    model, cfg, meta, copied = build_model(
        args.checkpoint, args.active_basis, args.active_rank,
        args.active_basis_source, first_att, args.opt_n_freq, device)

    rows = []
    for sample_id in args.sample_ids:
        att, noatt = load_pair(
            args.attenuated_root, args.noatt_root, sample_id, device)
        best = optimize_sample(model, meta, att, noatt, args, sample_id)
        full = fullband_evaluate(
            args.checkpoint, args.active_basis, args.active_basis_source,
            args.active_rank, att, noatt, best["raw"], args, device)
        torch.save({
            "sample": sample_id,
            "raw": best["raw"].cpu(),
            "coeff_np": best["coeff_np"].cpu(),
            "optimization_grid": {
                k: v for k, v in best.items()
                if k not in ("raw", "coeff_np")
            },
            "fullband": full,
        }, args.out / f"{sample_id}_oracle.pt")
        rows.append({
            "sample": sample_id,
            "best_step": best["step"],
            "opt_baseline_loss": best["baseline_paired_loss"],
            "opt_corrected_loss": best["paired_loss"],
            "opt_gain": best["baseline_paired_loss"] - best["paired_loss"],
            "fullband_baseline_loss": full["baseline_paired_loss"],
            "fullband_corrected_loss": full["paired_loss"],
            "fullband_gain": full["gain"],
            "coeff_np": full["coeff_np"],
            "max_abs_coeff_np": full["max_abs_coeff_np"],
        })

    summary = {
        "experiment": "paired amplitude active-mode oracle with mean-only phase",
        "checkpoint": str(args.checkpoint),
        "active_basis": str(args.active_basis),
        "active_basis_source": args.active_basis_source,
        "active_rank": args.active_rank,
        "amplitude_limit_np": args.amplitude_limit_np,
        "opt_n_freq": args.opt_n_freq,
        "eval_n_freq": args.eval_n_freq,
        "copied_mean_phase_tensors": len(copied),
        "n_samples": len(rows),
        "mean_fullband_baseline_loss": float(np.mean([
            r["fullband_baseline_loss"] for r in rows])),
        "mean_fullband_corrected_loss": float(np.mean([
            r["fullband_corrected_loss"] for r in rows])),
        "mean_fullband_gain": float(np.mean([
            r["fullband_gain"] for r in rows])),
        "mean_fullband_relative_reduction": float(np.mean([
            r["fullband_gain"] / max(r["fullband_baseline_loss"], 1e-12)
            for r in rows])),
        "rows": rows,
    }
    out_json = args.out / "paired_amplitude_active_oracle_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "event": "done",
        "mean_fullband_baseline_loss": summary["mean_fullband_baseline_loss"],
        "mean_fullband_corrected_loss": summary["mean_fullband_corrected_loss"],
        "mean_fullband_gain": summary["mean_fullband_gain"],
        "mean_fullband_relative_reduction": summary[
            "mean_fullband_relative_reduction"],
        "out": str(out_json),
    }), flush=True)


if __name__ == "__main__":
    main()

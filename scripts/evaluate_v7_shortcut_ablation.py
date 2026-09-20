"""Ablate whether V7 learns sample-dependent amplitude correction or a shortcut.

Same-checkpoint comparison on the paired validation set:

  Phase-only   : zero amplitude coefficients.
  Constant-A   : one coefficient vector = mean V7 prediction on TRAIN samples.
  Shuffled-A   : validation predictions are cyclically assigned to the wrong
                 validation samples; all non-zero cyclic shifts are averaged.
  Predicted-A  : each validation sample uses its own V7 prediction.

The key quantities are Predicted-A minus Constant-A and Predicted-A minus
Shuffled-A.  They isolate input-dependent prediction beyond a dataset-wide
fixed correction pattern.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from common import demod_iq, rf_to_D
from models.v7_amplitude_predictor import V7AmplitudePredictor
from physics.paired_amplitude_loss import paired_image_loss
from physics.phase_screen import mean_controls_to_ds
from scripts.pilot_phase_asp import DATA_ROOT
from train_phase_screen import sample_ids
from train_v7_amplitude_predictor import (
    build_frozen_base,
    corrected_image,
    physical_crop,
    prepare_cache,
)


def build_predictor(ckpt, device):
    saved = ckpt["args"]
    model = V7AmplitudePredictor(
        active_rank=int(saved.get("active_rank", 6)),
        depth_bins=int(saved.get("descriptor_depth_bins", 8)),
        lateral_modes=int(saved.get("descriptor_lateral_modes", 6)),
        hidden=int(saved.get("hidden", 32)),
        coeff_limit_np=float(saved.get("coeff_limit_np", 0.5)),
        descriptor_eps=float(saved.get("descriptor_eps", 1e-5)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def prepare_train_descriptors(ids, base, meta, predictor, phase_idx, image_idx, device):
    rows = []
    for n, sid in enumerate(ids, 1):
        att = torch.load(
            DATA_ROOT / "shards" / f"{sid}.pt",
            map_location="cpu", weights_only=False)
        rf = att["rf"][None].to(device)
        iq = demod_iq(rf, meta)
        D = rf_to_D(rf, meta)
        _, mean_raw, _ = base.predict_all_components(iq, phase_idx)
        ds = mean_controls_to_ds(
            mean_raw, base.born.nz, base.born.nx, base.born.dz,
            base.mean_limit_us)
        zero_amp = torch.zeros_like(ds)
        phase_images = base.angle_images(
            ds, D, image_idx, amplitude_rate=zero_amp)
        envelope = physical_crop(phase_images.abs().mean(dim=1), base.pad)
        descriptor = predictor.raw_descriptor(envelope)
        coeff, _ = predictor.forward_descriptor(descriptor)
        rows.append({
            "id": sid,
            "descriptor": descriptor.detach(),
            "coeff": coeff.detach(),
        })
        print(json.dumps({
            "event": "v7_ablation_train_descriptor",
            "sample": sid, "index": n, "total": len(ids),
            "coeff_np": coeff[0].cpu().tolist(),
        }), flush=True)
    return rows


@torch.no_grad()
def attach_predictions(predictor, cache):
    for item in cache:
        descriptor = predictor.raw_descriptor(item.pop("envelope"))
        coeff, _ = predictor.forward_descriptor(descriptor)
        item["descriptor"] = descriptor.detach()
        item["predicted_coeff"] = coeff.detach()


def loss_for_coeff(base, item, image_idx, coeff, args):
    current = corrected_image(base, item, image_idx, coeff)
    loss, image_loss, depth_loss, _ = paired_image_loss(
        current, item["reference"], args.smooth_kernel,
        args.loss_depth_bins, args.loss_eps, args.depth_weight)
    return float(loss), float(image_loss), float(depth_loss)


@torch.no_grad()
def evaluate_main_ablation(base, cache, image_idx, constant_coeff, args):
    rows = []
    for item in cache:
        pred = item["predicted_coeff"]
        zero = torch.zeros_like(pred)
        phase_loss, _, _ = loss_for_coeff(base, item, image_idx, zero, args)
        const_loss, _, _ = loss_for_coeff(
            base, item, image_idx, constant_coeff.to(pred), args)
        pred_loss, pred_image, pred_depth = loss_for_coeff(
            base, item, image_idx, pred, args)
        rows.append({
            "sample": item["id"],
            "phase_only_loss": phase_loss,
            "constant_loss": const_loss,
            "predicted_loss": pred_loss,
            "constant_relative_reduction": (phase_loss - const_loss) / max(phase_loss, 1e-12),
            "predicted_relative_reduction": (phase_loss - pred_loss) / max(phase_loss, 1e-12),
            "predicted_gain_over_constant": const_loss - pred_loss,
            "predicted_image_loss": pred_image,
            "predicted_depth_loss": pred_depth,
            "predicted_coeff_np": pred[0].cpu().tolist(),
        })
    return rows


@torch.no_grad()
def evaluate_all_cyclic_shuffles(base, cache, image_idx, args):
    """Average every non-zero cyclic assignment; no sample keeps its own coeff."""
    n = len(cache)
    if n < 2:
        raise ValueError("need at least two validation samples for shuffled ablation")
    by_shift = []
    per_sample_losses = [[] for _ in range(n)]
    for shift in range(1, n):
        losses = []
        for i, item in enumerate(cache):
            coeff = cache[(i + shift) % n]["predicted_coeff"]
            loss, _, _ = loss_for_coeff(base, item, image_idx, coeff, args)
            losses.append(loss)
            per_sample_losses[i].append(loss)
        by_shift.append({
            "shift": shift,
            "mean_loss": float(np.mean(losses)),
        })
    per_sample = [
        {
            "sample": cache[i]["id"],
            "mean_shuffled_loss": float(np.mean(v)),
            "std_shuffled_loss": float(np.std(v)),
        }
        for i, v in enumerate(per_sample_losses)
    ]
    return by_shift, per_sample


def coeff_stats(coeff):
    a = coeff.detach().cpu().numpy()
    return {
        "mean_np": a.mean(axis=0).tolist(),
        "std_np": a.std(axis=0).tolist(),
        "mean_abs_np": np.abs(a).mean(axis=0).tolist(),
        "max_abs_np": np.abs(a).max(axis=0).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("runs/v7_amplitude_predictor/best.pt"))
    p.add_argument("--paired-reference-root", type=Path)
    p.add_argument("--phase-checkpoint", type=Path)
    p.add_argument("--active-basis", type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-freq", type=int)
    p.add_argument("--train-per-case", type=int)
    p.add_argument("--val-per-case", type=int)
    p.add_argument("--smooth-kernel", type=int)
    p.add_argument("--loss-depth-bins", type=int)
    p.add_argument("--depth-weight", type=float)
    p.add_argument("--loss-eps", type=float)
    p.add_argument("--out", type=Path,
                   default=Path("runs/v7_amplitude_shortcut_ablation"))
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt["args"]
    args.paired_reference_root = args.paired_reference_root or Path(saved["paired_reference_root"])
    args.phase_checkpoint = args.phase_checkpoint or Path(saved["phase_checkpoint"])
    args.active_basis = args.active_basis or Path(saved["active_basis"])
    args.n_freq = args.n_freq if args.n_freq is not None else int(saved.get("n_freq", 64))
    args.train_per_case = (
        args.train_per_case if args.train_per_case is not None
        else int(saved.get("train_per_case", 25)))
    args.val_per_case = (
        args.val_per_case if args.val_per_case is not None
        else int(saved.get("val_per_case", 5)))
    args.smooth_kernel = (
        args.smooth_kernel if args.smooth_kernel is not None
        else int(saved.get("smooth_kernel", 9)))
    args.loss_depth_bins = (
        args.loss_depth_bins if args.loss_depth_bins is not None
        else int(saved.get("loss_depth_bins", 12)))
    args.depth_weight = (
        args.depth_weight if args.depth_weight is not None
        else float(saved.get("depth_weight", 0.5)))
    args.loss_eps = (
        args.loss_eps if args.loss_eps is not None
        else float(saved.get("loss_eps", 1e-5)))
    args.config = saved.get("config", "configs/l11_ultrawave_500_11angle.yaml")
    args.active_rank = int(saved.get("active_rank", 6))
    args.coeff_limit_np = float(saved.get("coeff_limit_np", 0.5))

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    train_ids = sample_ids("train", args.train_per_case)
    val_ids = sample_ids("val", args.val_per_case)
    first = torch.load(
        DATA_ROOT / "shards" / f"{train_ids[0]}.pt",
        map_location="cpu", weights_only=False)
    base, cfg, meta, copied = build_frozen_base(args, first, device)
    predictor = build_predictor(ckpt, device)
    phase_idx = torch.as_tensor(meta.train_idx, device=device)
    image_idx = torch.arange(
        meta.n_angles if hasattr(meta, "n_angles") else len(meta.angles_deg),
        device=device)

    train_rows = prepare_train_descriptors(
        train_ids, base, meta, predictor, phase_idx, image_idx, device)
    train_coeff = torch.cat([r["coeff"] for r in train_rows], dim=0)
    constant_coeff = train_coeff.mean(dim=0, keepdim=True)

    # prepare_cache supplies matched references and frozen mean-phase baseline.
    val_cache = prepare_cache(
        val_ids, base, meta, args.paired_reference_root,
        phase_idx, image_idx, device, args)
    attach_predictions(predictor, val_cache)
    val_coeff = torch.cat([r["predicted_coeff"] for r in val_cache], dim=0)

    rows = evaluate_main_ablation(
        base, val_cache, image_idx, constant_coeff, args)
    by_shift, shuffled_rows = evaluate_all_cyclic_shuffles(
        base, val_cache, image_idx, args)
    shuffled_map = {r["sample"]: r for r in shuffled_rows}
    for row in rows:
        sh = shuffled_map[row["sample"]]
        row.update(sh)
        phase = row["phase_only_loss"]
        row["shuffled_relative_reduction"] = (
            phase - row["mean_shuffled_loss"]) / max(phase, 1e-12)
        row["predicted_gain_over_shuffled"] = (
            row["mean_shuffled_loss"] - row["predicted_loss"])

    phase_mean = float(np.mean([r["phase_only_loss"] for r in rows]))
    const_mean = float(np.mean([r["constant_loss"] for r in rows]))
    pred_mean = float(np.mean([r["predicted_loss"] for r in rows]))
    shuffle_mean = float(np.mean([r["mean_shuffled_loss"] for r in rows]))
    summary = {
        "experiment": "V7 sample-dependence shortcut ablation",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "n_train": len(train_rows),
        "n_val": len(rows),
        "phase_only_mean_loss": phase_mean,
        "constant_mean_loss": const_mean,
        "shuffled_mean_loss": shuffle_mean,
        "predicted_mean_loss": pred_mean,
        "constant_mean_relative_reduction": float(np.mean([
            r["constant_relative_reduction"] for r in rows])),
        "shuffled_mean_relative_reduction": float(np.mean([
            r["shuffled_relative_reduction"] for r in rows])),
        "predicted_mean_relative_reduction": float(np.mean([
            r["predicted_relative_reduction"] for r in rows])),
        "predicted_gain_over_constant_loss": const_mean - pred_mean,
        "predicted_gain_over_shuffled_loss": shuffle_mean - pred_mean,
        "predicted_wins_vs_constant": int(sum(
            r["predicted_loss"] < r["constant_loss"] for r in rows)),
        "predicted_wins_vs_shuffled_mean": int(sum(
            r["predicted_loss"] < r["mean_shuffled_loss"] for r in rows)),
        "constant_wins_vs_phase": int(sum(
            r["constant_loss"] < r["phase_only_loss"] for r in rows)),
        "predicted_wins_vs_phase": int(sum(
            r["predicted_loss"] < r["phase_only_loss"] for r in rows)),
    }

    payload = {
        "summary": summary,
        "constant_coeff_np": constant_coeff[0].cpu().tolist(),
        "train_prediction_stats": coeff_stats(train_coeff),
        "validation_prediction_stats": coeff_stats(val_coeff),
        "cyclic_shuffle_results": by_shift,
        "rows": rows,
        "frozen_mean_phase_tensors": len(copied),
    }
    out_json = args.out / "v7_shortcut_ablation_summary.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "event": "v7_shortcut_ablation_done",
        **summary,
        "constant_coeff_np": payload["constant_coeff_np"],
        "out": str(out_json),
    }), flush=True)


if __name__ == "__main__":
    main()

"""Full-band qualitative/quantitative evaluation for phase-screen checkpoints.

This diagnostic complements ``visualize_phase_screen_checkpoint.py``:

* prediction stays on the checkpoint's training frequency grid;
* imaging/metrics use the full contiguous band by default;
* a shared TGC curve is estimated only from the Uniform image and applied to
  every candidate identically;
* the shared-TGC figure includes the true scatterer magnitude |m| as a fifth
  panel with its own normalization (truth and migrated-image amplitudes are not
  in the same units);
* difference maps show local amplitude change relative to Uniform;
* depth-wise corr / holdout agreement / coherence are reported over configurable
  depth bins so shallow high-SNR content cannot dominate the global metric;
* the default validation sweep evaluates 10 samples per case (20 total) and
  reports mean/median delta-hold, win counts, and case-wise summaries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from common import corr2d, demod_iq, rf_to_D  # noqa: E402
from models.phase_screen import coherence, heldout_agreement  # noqa: E402
from train_phase_screen import sample_ids  # noqa: E402
from scripts.pilot_phase_asp import DATA_ROOT, embed  # noqa: E402
from scripts.visualize_phase_screen_checkpoint import (  # noqa: E402
    angle_images,
    axis_coordinates,
    build_imaging_operator,
    compound,
    fixed_reference,
    frequency_sampling_info,
    load_prediction_model,
    make_teacher,
    score_candidate,
)


def smooth_1d(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x.copy()
    if width % 2 == 0:
        width += 1
    pad = width // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(xp, kernel, mode="valid")


def shared_tgc(uniform: torch.Tensor, smooth_rows: int, max_gain_db: float):
    """Uniform-derived depth gain, returned as [nz] torch tensor."""
    amp = uniform.abs().detach().cpu().numpy()
    profile = np.sqrt(np.mean(amp ** 2, axis=1))
    profile = smooth_1d(profile, smooth_rows)
    positive = profile[profile > 0]
    if positive.size == 0:
        gain = np.ones_like(profile)
    else:
        ref = float(np.quantile(positive, 0.9))
        floor = max(ref * 1e-6, 1e-12)
        gain = ref / np.maximum(profile, floor)
        gain = np.clip(gain, 1.0, 10.0 ** (max_gain_db / 20.0))
    return torch.as_tensor(gain, device=uniform.device, dtype=uniform.real.dtype)


def to_db(image: torch.Tensor, ref: float, db_range: float):
    amp = image.abs().detach().cpu().numpy()
    db = 20.0 * np.log10(np.maximum(amp, 1e-12) / max(ref, 1e-12))
    return np.clip(db, -db_range, 0.0)


def save_tgc_figure(path, sample_id, images, truth_abs, metrics,
                    cfg, meta, born, pad, db_range, dpi,
                    smooth_rows, max_gain_db):
    """Four migrated images on one shared scale + independently scaled truth."""
    names = ["uniform", "network", "teacher", "gt_speed_asm"]
    titles = ["Uniform", "Network", "Teacher", "GT-speed ASP"]
    gain = shared_tgc(images["uniform"], smooth_rows, max_gain_db)
    gained = {name: images[name] * gain[:, None] for name in names}
    ref_amp = max(float(gained[name].abs().max()) for name in names)
    truth_ref = max(float(truth_abs.abs().max()), 1e-12)
    x0, x1, z0, z1 = axis_coordinates(cfg, meta, born, pad)

    fig, axes = plt.subplots(1, 5, figsize=(18.7, 5.0), constrained_layout=True)
    im = None
    for ax, name, title in zip(axes[:4], names, titles):
        im = ax.imshow(
            to_db(gained[name], ref_amp, db_range), cmap="gray",
            vmin=-db_range, vmax=0, origin="upper",
            extent=[x0, x1, z1, z0], aspect="auto")
        dh = metrics[name]["holdout_agreement"] - metrics["uniform"]["holdout_agreement"]
        ax.set_title(f"{title}\nΔhold={dh:+.4f}", fontsize=10)
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")

    truth_im = axes[4].imshow(
        to_db(truth_abs, truth_ref, db_range), cmap="gray",
        vmin=-db_range, vmax=0, origin="upper",
        extent=[x0, x1, z1, z0], aspect="auto")
    axes[4].set_title("Truth |m|\nindependent scale", fontsize=10)
    axes[4].set_xlabel("Lateral x [mm]")
    axes[4].set_ylabel("Depth z [mm]")

    fig.colorbar(im, ax=axes[:4], shrink=0.84, pad=0.012,
                 label="Shared-TGC migrated amplitude [dB]")
    fig.colorbar(truth_im, ax=axes[4], shrink=0.84, pad=0.012,
                 label="Truth |m| [dB, own scale]")
    fig.suptitle(
        f"{sample_id}: shared TGC from Uniform only | max gain={max_gain_db:.0f} dB",
        fontsize=13)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def amplitude_delta_db(candidate: torch.Tensor, uniform: torch.Tensor,
                       mask_floor_db: float, clip_db: float):
    ua = uniform.abs().detach().cpu().numpy()
    ca = candidate.abs().detach().cpu().numpy()
    peak = max(float(ua.max()), 1e-12)
    valid = 20.0 * np.log10(np.maximum(ua, 1e-12) / peak) >= mask_floor_db
    delta = 20.0 * np.log10((ca + 1e-12) / (ua + 1e-12))
    delta = np.clip(delta, -clip_db, clip_db)
    return np.ma.array(delta, mask=~valid)


def save_difference_figure(path, sample_id, images, metrics, cfg, meta, born,
                           pad, dpi, mask_floor_db, clip_db):
    names = ["network", "teacher", "gt_speed_asm"]
    titles = ["Network − Uniform", "Teacher − Uniform", "GT-speed − Uniform"]
    x0, x1, z0, z1 = axis_coordinates(cfg, meta, born, pad)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.0), constrained_layout=True)
    im = None
    for ax, name, title in zip(axes, names, titles):
        delta = amplitude_delta_db(
            images[name], images["uniform"], mask_floor_db, clip_db)
        im = ax.imshow(
            delta, cmap="coolwarm", vmin=-clip_db, vmax=clip_db,
            origin="upper", extent=[x0, x1, z1, z0], aspect="auto")
        dh = metrics[name]["holdout_agreement"] - metrics["uniform"]["holdout_agreement"]
        ax.set_title(f"{title}\nΔhold={dh:+.4f}", fontsize=10)
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")
    fig.colorbar(im, ax=axes, shrink=0.84, pad=0.015,
                 label="Amplitude change vs Uniform [dB]")
    fig.suptitle(
        f"{sample_id}: local amplitude change; Uniform < {mask_floor_db:.0f} dB masked",
        fontsize=13)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def validate_depth_edges(edges):
    vals = [float(v) for v in edges]
    if len(vals) < 2:
        raise ValueError("depth bins require at least two edges")
    if any(b <= a for a, b in zip(vals[:-1], vals[1:])):
        raise ValueError("depth-bin edges must be strictly increasing")
    return vals


def depth_windows(edges):
    return [(float(a), float(b), f"{a:g}-{b:g}mm")
            for a, b in zip(edges[:-1], edges[1:])]


def depth_row_mask(born, z_lo_mm, z_hi_mm, device):
    z_mm = (float(born.z0) +
            torch.arange(born.nz, device=device, dtype=torch.float32)
            * float(born.dz)) * 1e3
    return (z_mm >= z_lo_mm) & (z_mm < z_hi_mm)


def depth_reference(uniform_per_angle, base_mask, train_idx, hold_idx,
                    row_mask):
    """Uniform-derived ROI mask/scales recomputed independently per depth bin."""
    roi_mask = base_mask.clone()
    roi_mask[:, ~row_mask, :] = 0
    train = uniform_per_angle[:, train_idx]
    hold = uniform_per_angle[:, hold_idx]
    tr_scale = (train.abs().square() * roi_mask[:, None]).sum(dim=(-2, -1))\
        .sqrt().clamp_min(1e-30)
    ho_scale = (hold.abs().square() * roi_mask[:, None]).sum(dim=(-2, -1))\
        .sqrt().clamp_min(1e-30)
    return {
        "mask": roi_mask,
        "train_scales": tr_scale,
        "hold_scales": ho_scale,
    }


def depth_metric_for_candidate(per_angle, truth_abs, ref, train_idx, hold_idx,
                               row_mask, pad):
    train = per_angle[:, train_idx]
    hold = per_angle[:, hold_idx]
    coh = coherence(train, ref["mask"], ref["train_scales"])[0]
    hold_score = heldout_agreement(
        train, hold, ref["mask"], ref["train_scales"],
        ref["hold_scales"])[0]
    image = compound(per_angle, pad)
    image_roi = image[row_mask]
    truth_roi = truth_abs[row_mask]
    corr = corr2d(image_roi.abs(), truth_roi)
    return {
        "input_coherence": float(coh),
        "holdout_agreement": float(hold_score),
        "image_abs_corr": float(corr),
    }


def compute_depth_metrics(per_angle_images, truth_abs, global_ref,
                          train_idx, hold_idx, born, pad, edges):
    result = {}
    uniform_per_angle = per_angle_images["uniform"]
    for z_lo, z_hi, label in depth_windows(edges):
        row_mask = depth_row_mask(born, z_lo, z_hi, uniform_per_angle.device)
        if int(row_mask.sum()) < 2:
            result[label] = {"error": "fewer than two depth rows"}
            continue
        ref = depth_reference(
            uniform_per_angle, global_ref["mask"], train_idx, hold_idx, row_mask)
        result[label] = {
            name: depth_metric_for_candidate(
                imgs, truth_abs, ref, train_idx, hold_idx, row_mask, pad)
            for name, imgs in per_angle_images.items()
        }
    return result


@torch.no_grad()
def evaluate_one(sample_id, model, pred_meta, imaging_cfg, imaging_meta,
                 imaging_born, device, top_frac, out_dir, args, depth_edges):
    sample = torch.load(DATA_ROOT / "shards" / f"{sample_id}.pt",
                        map_location="cpu", weights_only=False)
    rf = sample["rf"][None].to(device)

    pred_train_idx = torch.as_tensor(pred_meta.train_idx, device=device)
    iq = demod_iq(rf, pred_meta)
    pred_phase, pred_mean, pred_bulk = model.predict_components(iq, pred_train_idx)
    network_ds = model.components_to_slowness(pred_phase, pred_mean, pred_bulk)

    true_ds = embed(sample["delta_s"].to(device), model.pad)
    teacher_phase, teacher_mean, teacher_bulk, _, _, _ = make_teacher(model, true_ds)
    teacher_ds = model.components_to_slowness(
        teacher_phase[None],
        teacher_mean[None] if model.mean_controls else teacher_mean,
        teacher_bulk[None])

    D = rf_to_D(rf, imaging_meta)
    train_idx = torch.as_tensor(imaging_meta.train_idx, device=device)
    hold_idx = torch.as_tensor(imaging_meta.hold_idx, device=device)
    all_idx = torch.arange(D.shape[1], device=device)
    global_ref = fixed_reference(
        imaging_born, D, train_idx, hold_idx, model.pad, top_frac)

    zero_ds = torch.zeros_like(network_ds)
    candidates_ds = {
        "uniform": zero_ds,
        "network": network_ds,
        "teacher": teacher_ds,
        "gt_speed_asm": true_ds[None],
    }
    metrics = {}
    images = {}
    per_angle_images = {}
    truth_abs = sample["m"].abs().to(device)
    for name, ds in candidates_ds.items():
        per_angle = angle_images(imaging_born, ds, D, all_idx)
        per_angle_images[name] = per_angle
        metrics[name], images[name] = score_candidate(
            per_angle, global_ref, train_idx, hold_idx, truth_abs, model.pad)

    depth_metrics = compute_depth_metrics(
        per_angle_images, truth_abs, global_ref,
        train_idx, hold_idx, imaging_born, model.pad, depth_edges)

    save_tgc_figure(
        out_dir / f"{sample_id}_shared_tgc.png", sample_id, images, truth_abs,
        metrics, imaging_cfg, imaging_meta, imaging_born, model.pad,
        args.db_range, args.dpi, args.tgc_smooth_rows, args.tgc_max_gain_db)
    save_difference_figure(
        out_dir / f"{sample_id}_difference.png", sample_id, images, metrics,
        imaging_cfg, imaging_meta, imaging_born, model.pad, args.dpi,
        args.diff_mask_floor_db, args.diff_clip_db)

    uniform_hold = metrics["uniform"]["holdout_agreement"]
    deltas = {
        name: metrics[name]["holdout_agreement"] - uniform_hold
        for name in ("network", "teacher", "gt_speed_asm")
    }
    return {
        "sample": sample_id,
        "case": sample["metadata"].get("case"),
        "metrics": metrics,
        "depth_metrics": depth_metrics,
        "delta_hold_vs_uniform": deltas,
        "figures": {
            "shared_tgc": f"{sample_id}_shared_tgc.png",
            "difference": f"{sample_id}_difference.png",
        },
    }


def summarize(rows):
    methods = ("network", "teacher", "gt_speed_asm")
    summary = {}
    for method in methods:
        vals = np.asarray([r["delta_hold_vs_uniform"][method] for r in rows], dtype=float)
        summary[method] = {
            "mean_delta_hold": float(vals.mean()),
            "median_delta_hold": float(np.median(vals)),
            "min_delta_hold": float(vals.min()),
            "max_delta_hold": float(vals.max()),
            "wins": int((vals > 0).sum()),
            "ties": int((vals == 0).sum()),
            "losses": int((vals < 0).sum()),
            "n": int(len(vals)),
        }
    return summary


def summarize_by_case(rows):
    groups = {}
    for row in rows:
        key = str(row.get("case"))
        groups.setdefault(key, []).append(row)
    return {key: summarize(group) for key, group in groups.items()}


def summarize_depth(rows, depth_edges):
    methods = ("uniform", "network", "teacher", "gt_speed_asm")
    out = {}
    for _, _, label in depth_windows(depth_edges):
        valid = [r for r in rows if "error" not in r["depth_metrics"].get(label, {})]
        if not valid:
            out[label] = {"error": "no valid samples"}
            continue
        block = {}
        for method in methods:
            block[method] = {
                metric: float(np.mean([
                    r["depth_metrics"][label][method][metric] for r in valid
                ]))
                for metric in ("input_coherence", "holdout_agreement", "image_abs_corr")
            }
        for method in ("network", "teacher", "gt_speed_asm"):
            d_hold = np.asarray([
                r["depth_metrics"][label][method]["holdout_agreement"] -
                r["depth_metrics"][label]["uniform"]["holdout_agreement"]
                for r in valid
            ], dtype=float)
            d_corr = np.asarray([
                r["depth_metrics"][label][method]["image_abs_corr"] -
                r["depth_metrics"][label]["uniform"]["image_abs_corr"]
                for r in valid
            ], dtype=float)
            block[method]["mean_delta_hold_vs_uniform"] = float(d_hold.mean())
            block[method]["median_delta_hold_vs_uniform"] = float(np.median(d_hold))
            block[method]["hold_wins"] = int((d_hold > 0).sum())
            block[method]["mean_delta_corr_vs_uniform"] = float(d_corr.mean())
            block[method]["median_delta_corr_vs_uniform"] = float(np.median(d_corr))
            block[method]["corr_wins"] = int((d_corr > 0).sum())
        block["n"] = len(valid)
        out[label] = block
    return out


def ranking(rows, method):
    ordered = sorted(
        rows, key=lambda r: r["delta_hold_vs_uniform"][method], reverse=True)
    return [
        {"sample": r["sample"],
         "case": r.get("case"),
         "delta_hold": r["delta_hold_vs_uniform"][method]}
        for r in ordered
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--per-case", type=int, default=10,
                   help="samples per case; val default 10 => 20 total")
    p.add_argument("--sample-ids", nargs="+")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--imaging-n-freq", type=int, default=0)
    p.add_argument("--db-range", type=float, default=60.0)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--tgc-smooth-rows", type=int, default=11)
    p.add_argument("--tgc-max-gain-db", type=float, default=30.0)
    p.add_argument("--diff-mask-floor-db", type=float, default=-50.0)
    p.add_argument("--diff-clip-db", type=float, default=6.0)
    p.add_argument(
        "--depth-bin-edges-mm", type=float, nargs="+",
        default=[3.0, 10.0, 20.0, 30.0, 35.0],
        help="consecutive depth-bin edges; default excludes <3 mm and >35 mm",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.per_case < 1:
        p.error("--per-case must be >= 1")
    if args.db_range <= 0 or args.diff_clip_db <= 0:
        p.error("display dB ranges must be positive")
    try:
        depth_edges = validate_depth_edges(args.depth_bin_edges_mm)
    except ValueError as exc:
        p.error(str(exc))

    ids = args.sample_ids if args.sample_ids else sample_ids(args.split, args.per_case)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    first = torch.load(DATA_ROOT / "shards" / f"{ids[0]}.pt",
                       map_location="cpu", weights_only=False)
    ckpt, saved_args, pred_cfg, pred_meta, model = load_prediction_model(
        args.checkpoint, first, device)
    imaging_cfg, imaging_meta, imaging_born = build_imaging_operator(
        saved_args, first, model.pad, args.imaging_n_freq, device)
    freq_info = frequency_sampling_info(imaging_meta, imaging_cfg.physics.c0)
    if not freq_info.get("contiguous_fft_bins", False):
        raise RuntimeError(
            "full-band evaluation requires contiguous imaging bins; use --imaging-n-freq 0")

    rows = []
    for i, sample_id in enumerate(ids, 1):
        row = evaluate_one(
            sample_id, model, pred_meta, imaging_cfg, imaging_meta,
            imaging_born, device, args.top_frac, args.out, args, depth_edges)
        rows.append(row)
        print(json.dumps({
            "event": "evaluated",
            "sample": sample_id,
            "completed": i,
            "total": len(ids),
            "delta_hold": row["delta_hold_vs_uniform"],
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {
        "checkpoint": str(args.checkpoint),
        "step": int(ckpt["step"]),
        "frequency_sampling": freq_info,
        "depth_bin_edges_mm": depth_edges,
        "samples": ids,
        "summary": summarize(rows),
        "summary_by_case": summarize_by_case(rows),
        "depth_summary": summarize_depth(rows, depth_edges),
        "ranking": {
            method: ranking(rows, method)
            for method in ("network", "teacher", "gt_speed_asm")
        },
        "rows": rows,
    }
    (args.out / "fullband_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "event": "done",
        "out": str(args.out),
        "summary": payload["summary"],
        "depth_summary": payload["depth_summary"],
    }), flush=True)


if __name__ == "__main__":
    main()

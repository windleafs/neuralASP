"""Truth-vs-incoherent diagnostic for a suspicious dark B-mode region.

This follows the coherent-cancellation diagnostic.  Once destructive
cross-angle cancellation is excluded, compare the suspected dark ROI against
its surrounding reference ring in both:
  1) truth reflectivity |m| stored in the shard,
  2) Uniform incoherent amplitude,
  3) GT-speed ASP incoherent amplitude.

The central ratios are

    R_m  = mean_ROI(|m|) / mean_RING(|m|)
    R_U  = mean_ROI(I_inc_uniform) / mean_RING(I_inc_uniform)
    R_GT = mean_ROI(I_inc_gt) / mean_RING(I_inc_gt)

Interpretation:
- R_m << 1 and R_GT << 1: supports a genuinely weak-scattering region.
- R_m ~ 1 but R_GT << 1: the GT-speed ASP darkening is not explained by
  truth reflectivity and points to amplitude/migration mismatch or removal of
  misplaced clutter.

No new UltraWave simulation is required.
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

from common import rf_to_D
from scripts.pilot_phase_asp import DATA_ROOT, embed
from scripts.visualize_phase_screen_checkpoint import load_prediction_model
from scripts.diagnose_coherent_cancellation import (
    angle_images,
    axis_coords,
    build_imaging_operator,
    diagnostics,
)


def rectangular_mask(x_mm, z_mm, roi):
    x0, x1, z0, z1 = roi
    xx, zz = np.meshgrid(x_mm, z_mm)
    return (
        (xx >= min(x0, x1)) & (xx <= max(x0, x1))
        & (zz >= min(z0, z1)) & (zz <= max(z0, z1))
    )


def ring_mask(x_mm, z_mm, roi, margin_mm):
    """Rectangular reference ring around ROI, clipped to the image domain."""
    x0, x1, z0, z1 = roi
    inner = rectangular_mask(x_mm, z_mm, roi)
    outer = rectangular_mask(
        x_mm, z_mm,
        [min(x0, x1) - margin_mm, max(x0, x1) + margin_mm,
         min(z0, z1) - margin_mm, max(z0, z1) + margin_mm],
    )
    ring = outer & ~inner
    if not ring.any():
        raise ValueError("reference ring is empty; increase image coverage or margin")
    return ring


def mean_median(arr, mask):
    vals = np.asarray(arr)[mask]
    return float(vals.mean()), float(np.median(vals))


def ratio_db(a, b):
    return 20.0 * np.log10(max(a, 1e-30) / max(b, 1e-30))


def summarize_region(truth, uniform_inc, gt_inc, roi_mask, ref_mask):
    stats = {}
    for name, arr in (
        ("truth_abs_m", truth),
        ("uniform_incoherent", uniform_inc),
        ("gt_incoherent", gt_inc),
    ):
        roi_mean, roi_median = mean_median(arr, roi_mask)
        ref_mean, ref_median = mean_median(arr, ref_mask)
        stats[name] = {
            "roi_mean": roi_mean,
            "roi_median": roi_median,
            "reference_mean": ref_mean,
            "reference_median": ref_median,
            "roi_over_reference_mean": roi_mean / max(ref_mean, 1e-30),
            "roi_over_reference_median": roi_median / max(ref_median, 1e-30),
            "roi_vs_reference_db_mean": ratio_db(roi_mean, ref_mean),
        }

    stats["gt_vs_uniform"] = {
        "roi_incoherent_ratio": (
            stats["gt_incoherent"]["roi_mean"]
            / max(stats["uniform_incoherent"]["roi_mean"], 1e-30)
        ),
        "reference_incoherent_ratio": (
            stats["gt_incoherent"]["reference_mean"]
            / max(stats["uniform_incoherent"]["reference_mean"], 1e-30)
        ),
        "roi_gt_vs_uniform_db": ratio_db(
            stats["gt_incoherent"]["roi_mean"],
            stats["uniform_incoherent"]["roi_mean"],
        ),
        "reference_gt_vs_uniform_db": ratio_db(
            stats["gt_incoherent"]["reference_mean"],
            stats["uniform_incoherent"]["reference_mean"],
        ),
    }
    return stats


def db_image(arr, ref, db_range):
    a = np.asarray(arr)
    return np.clip(
        20.0 * np.log10(np.maximum(a, 1e-12) / max(ref, 1e-12)),
        -db_range, 0.0,
    )


def add_regions(ax, roi, margin_mm):
    x0, x1, z0, z1 = roi
    xl, xr = min(x0, x1), max(x0, x1)
    zt, zb = min(z0, z1), max(z0, z1)

    ax.add_patch(plt.Rectangle(
        (xl, zt), xr - xl, zb - zt,
        fill=False, edgecolor="red", linewidth=1.7, label="ROI"))

    ax.add_patch(plt.Rectangle(
        (xl - margin_mm, zt - margin_mm),
        (xr - xl) + 2 * margin_mm,
        (zb - zt) + 2 * margin_mm,
        fill=False, edgecolor="cyan", linestyle="--", linewidth=1.2,
        label="reference outer boundary"))


def save_figure(path, sample_id, truth, uniform_inc, gt_inc,
                x_mm, z_mm, roi, margin_mm, db_range, dpi, stats):
    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]

    truth_ref = float(np.max(truth))
    inc_ref = max(float(np.max(uniform_inc)), float(np.max(gt_inc)))
    ratio = 20.0 * np.log10(
        np.maximum(gt_inc, 1e-12) / np.maximum(uniform_inc, 1e-12))

    fig, axes = plt.subplots(1, 4, figsize=(16, 5.1), constrained_layout=True)

    im0 = axes[0].imshow(
        db_image(truth, truth_ref, db_range),
        cmap="gray", vmin=-db_range, vmax=0,
        extent=extent, aspect="auto")
    axes[0].set_title(
        "Truth |m|\n"
        f"ROI/ref={stats['truth_abs_m']['roi_over_reference_mean']:.3f} "
        f"({stats['truth_abs_m']['roi_vs_reference_db_mean']:+.2f} dB)")

    im1 = axes[1].imshow(
        db_image(uniform_inc, inc_ref, db_range),
        cmap="gray", vmin=-db_range, vmax=0,
        extent=extent, aspect="auto")
    axes[1].set_title(
        "Uniform incoherent\n"
        f"ROI/ref={stats['uniform_incoherent']['roi_over_reference_mean']:.3f} "
        f"({stats['uniform_incoherent']['roi_vs_reference_db_mean']:+.2f} dB)")

    axes[2].imshow(
        db_image(gt_inc, inc_ref, db_range),
        cmap="gray", vmin=-db_range, vmax=0,
        extent=extent, aspect="auto")
    axes[2].set_title(
        "GT-speed incoherent\n"
        f"ROI/ref={stats['gt_incoherent']['roi_over_reference_mean']:.3f} "
        f"({stats['gt_incoherent']['roi_vs_reference_db_mean']:+.2f} dB)")

    im3 = axes[3].imshow(
        ratio, cmap="coolwarm", vmin=-6, vmax=6,
        extent=extent, aspect="auto")
    axes[3].set_title(
        "GT / Uniform incoherent [dB]\n"
        f"ROI={stats['gt_vs_uniform']['roi_gt_vs_uniform_db']:+.2f} dB")

    for ax in axes:
        add_regions(ax, roi, margin_mm)
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")

    axes[0].legend(loc="lower left", fontsize=7)
    fig.colorbar(im0, ax=axes[0], label="Truth |m| [dB, own scale]")
    fig.colorbar(im1, ax=axes[1:3], label="Incoherent amplitude [dB, shared scale]")
    fig.colorbar(im3, ax=axes[3], label="GT / Uniform [dB]")
    fig.suptitle(f"{sample_id}: truth vs GT-speed dark-region diagnostic")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def classify(stats, truth_low_db=-3.0, gt_dark_db=-3.0):
    """Conservative evidence label; raw metrics remain the primary output."""
    truth_db = stats["truth_abs_m"]["roi_vs_reference_db_mean"]
    gt_db = stats["gt_incoherent"]["roi_vs_reference_db_mean"]

    if truth_db <= truth_low_db and gt_db <= gt_dark_db:
        return {
            "label": "supports_genuinely_weak_scattering",
            "reason": (
                "ROI is weak relative to its reference ring in both truth |m| "
                "and GT-speed incoherent amplitude."
            ),
        }
    if truth_db > truth_low_db and gt_db <= gt_dark_db:
        return {
            "label": "supports_imaging_or_migration_mismatch",
            "reason": (
                "GT-speed incoherent image is dark in the ROI, but truth |m| "
                "does not show a comparably weak region."
            ),
        }
    return {
        "label": "inconclusive_or_not_strongly_dark",
        "reason": (
            "ROI/reference contrasts do not satisfy either conservative "
            "decision pattern; inspect the maps and raw ratios."
        ),
    }


@torch.no_grad()
def analyze_one(sample_id, checkpoint, imaging_n_freq, device,
                roi, margin_mm, db_range, dpi, out):
    sample = torch.load(
        DATA_ROOT / "shards" / f"{sample_id}.pt",
        map_location="cpu", weights_only=False)

    ckpt, saved_args, _, _, model = load_prediction_model(
        checkpoint, sample, device)
    imaging_cfg, imaging_meta, born = build_imaging_operator(
        saved_args, sample, model.pad, imaging_n_freq, device)

    rf = sample["rf"][None].to(device)
    D = rf_to_D(rf, imaging_meta)
    all_idx = torch.arange(D.shape[1], device=device)

    true_ds = embed(sample["delta_s"].to(device), model.pad)
    zero_ds = torch.zeros(1, born.nz, born.nx, device=device, dtype=D.real.dtype)

    uniform_imgs = angle_images(born, zero_ds, D, all_idx)
    gt_imgs = angle_images(born, true_ds[None], D, all_idx)

    _, uniform_inc_t, _ = diagnostics(uniform_imgs, model.pad)
    _, gt_inc_t, _ = diagnostics(gt_imgs, model.pad)

    uniform_inc = uniform_inc_t.detach().cpu().numpy()
    gt_inc = gt_inc_t.detach().cpu().numpy()
    truth = sample["m"].abs().cpu().numpy()

    x_mm, z_mm = axis_coords(imaging_cfg, imaging_meta, born)
    if truth.shape != uniform_inc.shape:
        raise ValueError(
            f"truth/image shape mismatch: truth={truth.shape}, image={uniform_inc.shape}")

    roi_m = rectangular_mask(x_mm, z_mm, roi)
    ref_m = ring_mask(x_mm, z_mm, roi, margin_mm)
    if not roi_m.any():
        raise ValueError("ROI contains no image pixels")

    stats = summarize_region(truth, uniform_inc, gt_inc, roi_m, ref_m)
    decision = classify(stats)

    fig_name = f"{sample_id}_truth_vs_gt_dark_region.png"
    save_figure(
        out / fig_name, sample_id,
        truth, uniform_inc, gt_inc,
        x_mm, z_mm, roi, margin_mm, db_range, dpi, stats)

    return {
        "sample": sample_id,
        "roi_mm": list(map(float, roi)),
        "reference_margin_mm": float(margin_mm),
        "screen_gate": float(model.screen_gate_value().detach().cpu()),
        "statistics": stats,
        "decision": decision,
        "figure": fig_name,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sample-ids", nargs="+", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--imaging-n-freq", type=int, default=0)
    p.add_argument("--roi-mm", type=float, nargs=4, required=True,
                   metavar=("X0", "X1", "Z0", "Z1"))
    p.add_argument("--reference-margin-mm", type=float, default=3.0)
    p.add_argument("--db-range", type=float, default=60.0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.reference_margin_mm <= 0:
        p.error("--reference-margin-mm must be positive")
    if args.db_range <= 0:
        p.error("--db-range must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    rows = []
    for sid in args.sample_ids:
        row = analyze_one(
            sid, args.checkpoint, args.imaging_n_freq, device,
            args.roi_mm, args.reference_margin_mm,
            args.db_range, args.dpi, args.out)
        rows.append(row)
        print(json.dumps({
            "event": "dark_region_truth_diagnostic",
            "sample": sid,
            "decision": row["decision"],
            "statistics": row["statistics"],
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {
        "checkpoint": str(args.checkpoint),
        "interpretation": {
            "truth_and_gt_both_dark":
                "supports genuinely weak scattering / reflectivity",
            "truth_not_dark_but_gt_dark":
                "supports ASP amplitude/migration mismatch or removal of misplaced clutter",
            "note":
                "Truth |m| is the repository's reflectivity proxy, not a full-wave scattering potential.",
        },
        "rows": rows,
    }
    (args.out / "dark_region_truth_diagnostic.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "done", "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()

"""Born visibility diagnostic for suspicious dark regions.

This experiment separates acquisition visibility from full-wave/proxy mismatch.

Given the stored truth reflectivity proxy m, synthesize data with the *same*
uniform-background Born operator used for imaging,

    D_born = F_0(m),

then reconstruct it with the exact adjoint.  In parallel, reconstruct the
UltraWave RF with the same uniform adjoint.  Per-angle images are retained so
we can compare the same incoherent quantity

    I_inc = mean_theta |I_theta|

for Born self-data and UltraWave data.

Interpretation for a chosen ROI relative to its surrounding reference ring:
- truth not dark, Born self-reconstruction dark, UltraWave dark:
    acquisition / limited-view visibility is sufficient to explain the dark ROI.
- truth not dark, Born self-reconstruction not dark, UltraWave dark:
    the proxy m is visible to the Born acquisition model, so the mismatch is
    downstream of the proxy definition: full-wave scattering, impedance,
    directionality, attenuation, multiple scattering, etc.
- truth dark:
    weak reflectivity itself can explain the dark ROI.

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
from scripts.diagnose_coherent_cancellation import (
    angle_images,
    axis_coords,
    build_imaging_operator,
    diagnostics,
)
from scripts.diagnose_dark_region_truth import (
    rectangular_mask,
    ring_mask,
)


def mean_median(arr, mask):
    vals = np.asarray(arr)[mask]
    return float(vals.mean()), float(np.median(vals))


def contrast_stats(arr, roi_mask, ref_mask):
    roi_mean, roi_median = mean_median(arr, roi_mask)
    ref_mean, ref_median = mean_median(arr, ref_mask)
    mean_ratio = roi_mean / max(ref_mean, 1e-30)
    median_ratio = roi_median / max(ref_median, 1e-30)
    return {
        "roi_mean": roi_mean,
        "roi_median": roi_median,
        "reference_mean": ref_mean,
        "reference_median": ref_median,
        "roi_over_reference_mean": mean_ratio,
        "roi_over_reference_median": median_ratio,
        "roi_vs_reference_db_mean": float(
            20.0 * np.log10(max(mean_ratio, 1e-30))),
    }


def percentile_ref(arr, q=99.8):
    a = np.asarray(arr)
    return float(np.percentile(a[np.isfinite(a)], q))


def db_map(arr, ref, db_range):
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
        (xr - xl) + 2.0 * margin_mm,
        (zb - zt) + 2.0 * margin_mm,
        fill=False, edgecolor="cyan", linestyle="--",
        linewidth=1.2, label="reference outer boundary"))


def classify(stats, dark_db=-3.0):
    truth_db = stats["truth_abs_m"]["roi_vs_reference_db_mean"]
    born_db = stats["born_incoherent"]["roi_vs_reference_db_mean"]
    uw_db = stats["ultrawave_uniform_incoherent"]["roi_vs_reference_db_mean"]

    if truth_db <= dark_db:
        return {
            "label": "weak_truth_reflectivity_can_explain_dark_region",
            "reason": (
                "The stored truth |m| is itself weak in the ROI relative to the "
                "surrounding reference ring."
            ),
        }

    if born_db <= dark_db and uw_db <= dark_db:
        return {
            "label": "supports_limited_view_or_acquisition_visibility",
            "reason": (
                "Truth |m| is not weak, but the exact Born self-reconstruction "
                "of that same m is dark in the ROI, as is the UltraWave image. "
                "The acquisition/normal-operator visibility can therefore "
                "explain the dark region without invoking full-wave mismatch."
            ),
        }

    if born_db > dark_db and uw_db <= dark_db:
        return {
            "label": "supports_proxy_vs_full_wave_scattering_mismatch",
            "reason": (
                "The ROI is visible in the Born self-reconstruction of stored m "
                "but remains dark in the UltraWave reconstruction.  The darkening "
                "is therefore not explained by the Born acquisition nullspace."
            ),
        }

    return {
        "label": "inconclusive_or_not_strongly_dark",
        "reason": (
            "The ROI/reference contrasts do not satisfy a conservative decision "
            "pattern; inspect the maps and raw statistics."
        ),
    }


def save_visibility_figure(path, sample_id, truth, born_coh, born_inc, uw_inc,
                           x_mm, z_mm, roi, margin_mm, db_range, dpi, stats):
    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]
    arrays = [truth, born_coh, born_inc, uw_inc]
    keys = [
        "truth_abs_m",
        "born_coherent",
        "born_incoherent",
        "ultrawave_uniform_incoherent",
    ]
    titles = [
        "Truth |m|",
        r"Born self: $|F^H Fm|$",
        "Born self: incoherent",
        "UltraWave: Uniform incoherent",
    ]

    # Units between truth, synthetic Born data and UltraWave data are arbitrary;
    # each panel therefore has its own robust display reference.  ROI/reference
    # ratios in the titles are scale-invariant.
    refs = [percentile_ref(a) for a in arrays]

    fig, axes = plt.subplots(1, 4, figsize=(16.2, 5.2), constrained_layout=True)
    ims = []
    for ax, arr, key, title, ref in zip(axes, arrays, keys, titles, refs):
        ims.append(ax.imshow(
            db_map(arr, ref, db_range),
            cmap="gray", vmin=-db_range, vmax=0,
            extent=extent, aspect="auto"))
        s = stats[key]
        ax.set_title(
            f"{title}\nROI/ref={s['roi_over_reference_mean']:.3f} "
            f"({s['roi_vs_reference_db_mean']:+.2f} dB)")
        add_regions(ax, roi, margin_mm)
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")

    axes[0].legend(loc="lower left", fontsize=7)
    for ax, im in zip(axes, ims):
        fig.colorbar(im, ax=ax, shrink=0.78, label="dB, own robust scale")
    fig.suptitle(f"{sample_id}: truth-m Born visibility test")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_normalized_difference(path, sample_id, born_inc, uw_inc,
                               x_mm, z_mm, roi, margin_mm, dpi):
    """Compare morphology after each image is normalized by its reference ring."""
    roi_m = rectangular_mask(x_mm, z_mm, roi)
    ref_m = ring_mask(x_mm, z_mm, roi, margin_mm)
    b_ref = float(np.asarray(born_inc)[ref_m].mean())
    u_ref = float(np.asarray(uw_inc)[ref_m].mean())

    b = np.asarray(born_inc) / max(b_ref, 1e-30)
    u = np.asarray(uw_inc) / max(u_ref, 1e-30)
    delta = 20.0 * np.log10(np.maximum(u, 1e-12) / np.maximum(b, 1e-12))
    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    im0 = axes[0].imshow(
        np.clip(20*np.log10(np.maximum(b, 1e-12)), -30, 12),
        cmap="gray", vmin=-30, vmax=12, extent=extent, aspect="auto")
    axes[0].set_title("Born incoherent\nnormalized by reference ring")

    axes[1].imshow(
        np.clip(20*np.log10(np.maximum(u, 1e-12)), -30, 12),
        cmap="gray", vmin=-30, vmax=12, extent=extent, aspect="auto")
    axes[1].set_title("UltraWave incoherent\nnormalized by reference ring")

    im2 = axes[2].imshow(
        np.clip(delta, -6, 6), cmap="coolwarm", vmin=-6, vmax=6,
        extent=extent, aspect="auto")
    axes[2].set_title("UltraWave / Born [dB]\n(reference-ring normalized)")

    for ax in axes:
        add_regions(ax, roi, margin_mm)
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")
    fig.colorbar(im0, ax=axes[:2], shrink=0.8, label="relative amplitude [dB]")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="UltraWave / Born [dB]")
    fig.suptitle(f"{sample_id}: Born visibility vs full-wave data morphology")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


@torch.no_grad()
def analyze_one(sample_id, checkpoint, imaging_n_freq, device,
                roi, margin_mm, db_range, dpi, out):
    sample = torch.load(
        DATA_ROOT / "shards" / f"{sample_id}.pt",
        map_location="cpu", weights_only=False)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt["args"]
    pad = int(saved_args.get("pad", 32))

    cfg, meta, born = build_imaging_operator(
        saved_args, sample, pad, imaging_n_freq, device)

    rf = sample["rf"][None].to(device)
    D_uw = rf_to_D(rf, meta)
    all_idx = torch.arange(D_uw.shape[1], device=device)

    zero_ds = D_uw.real.new_zeros(1, born.nz, born.nx)
    u_tx = born.transmit_fields(zero_ds, all_idx)

    m_phys = sample["m"].to(device)
    m_pad = embed(m_phys, pad)[None]
    if m_pad.shape[-2:] != (born.nz, born.nx):
        raise ValueError(
            f"m/grid mismatch: padded m={tuple(m_pad.shape[-2:])}, "
            f"Born grid={(born.nz, born.nx)}")

    # Exact self-data under the current uniform Born acquisition model.
    D_born = born.forward(m_pad, zero_ds, u_tx, all_idx)

    born_imgs = born.adjoint_per_angle(D_born, u_tx, zero_ds)
    uw_imgs = born.adjoint_per_angle(D_uw, u_tx, zero_ds)

    born_coh_t, born_inc_t, born_cf_t = diagnostics(born_imgs, pad)
    uw_coh_t, uw_inc_t, uw_cf_t = diagnostics(uw_imgs, pad)

    truth = sample["m"].abs().cpu().numpy()
    born_coh = born_coh_t.detach().cpu().numpy()
    born_inc = born_inc_t.detach().cpu().numpy()
    uw_inc = uw_inc_t.detach().cpu().numpy()

    x_mm, z_mm = axis_coords(cfg, meta, born)
    if truth.shape != born_inc.shape or truth.shape != uw_inc.shape:
        raise ValueError(
            f"shape mismatch: truth={truth.shape}, Born={born_inc.shape}, "
            f"UltraWave={uw_inc.shape}")

    roi_m = rectangular_mask(x_mm, z_mm, roi)
    ref_m = ring_mask(x_mm, z_mm, roi, margin_mm)
    if not roi_m.any():
        raise ValueError("ROI contains no pixels")

    stats = {
        "truth_abs_m": contrast_stats(truth, roi_m, ref_m),
        "born_coherent": contrast_stats(born_coh, roi_m, ref_m),
        "born_incoherent": contrast_stats(born_inc, roi_m, ref_m),
        "ultrawave_uniform_incoherent": contrast_stats(uw_inc, roi_m, ref_m),
        "born_angle_coherence_factor": contrast_stats(
            born_cf_t.detach().cpu().numpy(), roi_m, ref_m),
        "ultrawave_angle_coherence_factor": contrast_stats(
            uw_cf_t.detach().cpu().numpy(), roi_m, ref_m),
    }
    decision = classify(stats)

    fig1 = f"{sample_id}_born_visibility.png"
    fig2 = f"{sample_id}_born_vs_ultrawave_normalized.png"
    save_visibility_figure(
        out / fig1, sample_id,
        truth, born_coh, born_inc, uw_inc,
        x_mm, z_mm, roi, margin_mm, db_range, dpi, stats)
    save_normalized_difference(
        out / fig2, sample_id,
        born_inc, uw_inc, x_mm, z_mm, roi, margin_mm, dpi)

    return {
        "sample": sample_id,
        "roi_mm": list(map(float, roi)),
        "reference_margin_mm": float(margin_mm),
        "statistics": stats,
        "decision": decision,
        "figures": {
            "visibility": fig1,
            "born_vs_ultrawave_normalized": fig2,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="used for acquisition config / padding only")
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
            "event": "born_visibility",
            "sample": sid,
            "decision": row["decision"],
            "truth_roi_ref": row["statistics"]["truth_abs_m"]["roi_over_reference_mean"],
            "born_roi_ref": row["statistics"]["born_incoherent"]["roi_over_reference_mean"],
            "uw_roi_ref": row["statistics"]["ultrawave_uniform_incoherent"]["roi_over_reference_mean"],
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {
        "checkpoint": str(args.checkpoint),
        "interpretation": {
            "truth_not_dark_born_dark_uw_dark":
                "supports limited-view / acquisition visibility",
            "truth_not_dark_born_not_dark_uw_dark":
                "supports proxy-vs-full-wave scattering mismatch",
            "truth_dark":
                "weak reflectivity can explain the dark region",
            "note":
                "Stored m is a reflectivity proxy, not an exact variable-density full-wave scattering potential.",
        },
        "rows": rows,
    }
    (args.out / "born_visibility_diagnostic.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "done", "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()

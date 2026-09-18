"""Analyze UltraWave physics-path contrast continuation.

For each physics path and contrast scale alpha, reconstruct the scattered RF
with the same uniform-background 0.1 mm oversampled ASP adjoint and measure
ROI/reference contrast in the incoherent image:

    C_ROI(alpha) = 20 log10(mean_ROI / mean_reference_ring).

The script estimates the first zero crossing alpha* for every path by linear
interpolation between neighboring samples with opposite sign.

Interpretation:
- c_only crosses early -> nonlinear propagation/refraction dominates.
- rho_only crosses early -> density/impedance scattering dominates.
- c_plus_rho crosses while each single path does not -> coupling is important.
- only full crosses -> attenuation or multi-physics coupling is important.

No new network training is required.
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
from scripts.pilot_phase_asp import DATA_ROOT
from scripts.diagnose_coherent_cancellation import (
    angle_images,
    axis_coords,
    build_imaging_operator,
    diagnostics,
)
from scripts.diagnose_dark_region_truth import rectangular_mask, ring_mask
from scripts.diagnose_born_visibility import contrast_stats

PATH_ORDER = [
    "c_only",
    "rho_only",
    "attenuation_only",
    "c_plus_rho",
    "full",
]


def load_path_files(root: Path, sample_id: str):
    sample_root = root / sample_id
    if not sample_root.exists():
        raise FileNotFoundError(sample_root)

    out = {}
    for path_name in PATH_ORDER:
        d = sample_root / path_name
        if not d.exists():
            continue
        rows = []
        for path in sorted(d.glob("alpha_*.npz")):
            with np.load(path) as f:
                rf = np.asarray(f["rf"], dtype=np.float32)
                md = json.loads(str(f["metadata_json"].item()))
            if md.get("sample_id") != sample_id:
                raise RuntimeError(f"sample mismatch in {path}")
            if md.get("physics_path") != path_name:
                raise RuntimeError(f"physics path mismatch in {path}")
            alpha = float(md["contrast_alpha"])
            if rf.shape != (11, 192, 2401):
                raise RuntimeError(f"unexpected RF shape in {path}: {rf.shape}")
            rows.append((alpha, path, rf, md))
        if rows:
            rows.sort(key=lambda x: x[0])
            out[path_name] = rows
    if not out:
        raise FileNotFoundError(f"no physics continuation files in {sample_root}")
    return out


def zero_crossing(alphas, contrasts_db):
    """First positive-to-negative zero crossing by linear interpolation."""
    a = np.asarray(alphas, dtype=float)
    y = np.asarray(contrasts_db, dtype=float)
    for i in range(len(a) - 1):
        if y[i] == 0:
            return float(a[i])
        if y[i] > 0 and y[i + 1] < 0:
            t = y[i] / (y[i] - y[i + 1])
            return float(a[i] + t * (a[i + 1] - a[i]))
    if len(a) and y[-1] == 0:
        return float(a[-1])
    return None


def save_curves(path, sample_id, rows_by_path, full_truth_db, dpi):
    fig, ax = plt.subplots(figsize=(8.2, 5.3), constrained_layout=True)
    for path_name in PATH_ORDER:
        if path_name not in rows_by_path:
            continue
        rows = rows_by_path[path_name]
        alpha = [r["alpha"] for r in rows]
        contrast = [r["roi_vs_reference_db"] for r in rows]
        label = path_name
        zc = zero_crossing(alpha, contrast)
        if zc is not None:
            label += f"  (alpha*={zc:.2f})"
        ax.plot(alpha, contrast, marker="o", label=label)

    ax.axhline(0.0, color="black", lw=1.0, ls="--")
    ax.axhline(full_truth_db, color="gray", lw=1.0, ls=":",
               label=f"truth |m| ROI/ref = {full_truth_db:+.2f} dB")
    ax.set(
        xlabel="Contrast scale alpha",
        ylabel="ROI / reference incoherent contrast [dB]",
        title=f"{sample_id}: physics-path contrast continuation",
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_heatmap(path, sample_id, rows_by_path, dpi):
    all_alphas = sorted({
        r["alpha"] for rows in rows_by_path.values() for r in rows
    })
    matrix = np.full((len(PATH_ORDER), len(all_alphas)), np.nan, dtype=float)
    for i, path_name in enumerate(PATH_ORDER):
        if path_name not in rows_by_path:
            continue
        lookup = {r["alpha"]: r["roi_vs_reference_db"] for r in rows_by_path[path_name]}
        for j, a in enumerate(all_alphas):
            if a in lookup:
                matrix[i, j] = lookup[a]

    vmax = max(3.0, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(np.arange(len(all_alphas)))
    ax.set_xticklabels([f"{a:g}" for a in all_alphas])
    ax.set_yticks(np.arange(len(PATH_ORDER)))
    ax.set_yticklabels(PATH_ORDER)
    ax.set_xlabel("Contrast scale alpha")
    ax.set_title(f"{sample_id}: ROI/reference contrast by physics path")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:+.2f}",
                        ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="ROI / reference [dB]")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_selected_images(path, sample_id, path_name, images, x_mm, z_mm,
                         roi, margin_mm, db_range, dpi):
    # Show up to five representative alpha values across the available range.
    alphas = np.asarray(sorted(images))
    if len(alphas) > 5:
        idx = np.unique(np.rint(np.linspace(0, len(alphas)-1, 5)).astype(int))
        alphas = alphas[idx]

    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]
    fig, axes = plt.subplots(1, len(alphas), figsize=(4*len(alphas), 5.0),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, alpha in zip(axes, alphas):
        arr, stats = images[float(alpha)]
        ref = float(np.percentile(arr[np.isfinite(arr)], 99.8))
        db = np.clip(
            20*np.log10(np.maximum(arr, 1e-12)/max(ref, 1e-12)),
            -db_range, 0)
        im = ax.imshow(db, cmap="gray", vmin=-db_range, vmax=0,
                       extent=extent, aspect="auto")
        ax.set_title(
            f"alpha={alpha:g}\n"
            f"ROI/ref={stats['roi_vs_reference_db_mean']:+.2f} dB")
        x0, x1, z0, z1 = roi
        xl, xr = min(x0,x1), max(x0,x1)
        zt, zb = min(z0,z1), max(z0,z1)
        ax.add_patch(plt.Rectangle(
            (xl,zt), xr-xl, zb-zt,
            fill=False, edgecolor="red", linewidth=1.5))
        ax.add_patch(plt.Rectangle(
            (xl-margin_mm,zt-margin_mm),
            (xr-xl)+2*margin_mm,(zb-zt)+2*margin_mm,
            fill=False, edgecolor="cyan", linestyle="--", linewidth=1.0))
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")
        fig.colorbar(im, ax=ax, shrink=0.75, label="dB, own scale")
    fig.suptitle(f"{sample_id}: {path_name} continuation")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def path_interpretation(summary):
    def crossing(name):
        row = summary.get(name)
        return None if row is None else row.get("zero_crossing_alpha")

    c = crossing("c_only")
    rho = crossing("rho_only")
    att = crossing("attenuation_only")
    cr = crossing("c_plus_rho")
    full = crossing("full")

    labels = []
    if c is not None and c <= 0.5:
        labels.append("sound_speed_nonlinearity_is_strong")
    if rho is not None and rho <= 0.5:
        labels.append("density_impedance_effect_is_strong")
    if att is not None and att <= 0.5:
        labels.append("attenuation_path_effect_is_strong")
    if cr is not None and c is None and rho is None:
        labels.append("c_rho_coupling_is_important")
    if full is not None and cr is None:
        labels.append("full_multiphysics_coupling_or_attenuation_is_required")
    if not labels:
        labels.append("no_early_zero_crossing_in_tested_range")
    return labels


@torch.no_grad()
def analyze_one(sample_id, checkpoint, continuation_root, imaging_n_freq,
                device, roi, margin_mm, db_range, dpi, out):
    sample = torch.load(
        DATA_ROOT / "shards" / f"{sample_id}.pt",
        map_location="cpu", weights_only=False)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt["args"]
    pad = int(saved_args.get("pad", 32))

    cfg, meta, born = build_imaging_operator(
        saved_args, sample, pad, imaging_n_freq, device)

    path_files = load_path_files(continuation_root, sample_id)

    # Shared uniform adjoint operator.
    full_rf = sample["rf"][None].to(device)
    D_full = rf_to_D(full_rf, meta)
    all_idx = torch.arange(D_full.shape[1], device=device)
    zero_ds = D_full.real.new_zeros(1, born.nz, born.nx)
    u_tx = born.transmit_fields(zero_ds, all_idx)

    x_mm, z_mm = axis_coords(cfg, meta, born)
    roi_m = rectangular_mask(x_mm, z_mm, roi)
    ref_m = ring_mask(x_mm, z_mm, roi, margin_mm)

    truth = sample["m"].abs().cpu().numpy()
    truth_stats = contrast_stats(truth, roi_m, ref_m)

    rows_by_path = {}
    image_cache = {}

    for path_name in PATH_ORDER:
        if path_name not in path_files:
            continue
        rows = []
        imgs_for_path = {}
        for alpha, source_path, rf_np, md in path_files[path_name]:
            rf = torch.from_numpy(rf_np)[None].to(device)
            D = rf_to_D(rf, meta)
            imgs = born.adjoint_per_angle(D, u_tx, zero_ds)
            _, inc_t, _ = diagnostics(imgs, pad)
            inc = inc_t.detach().cpu().numpy()
            stats = contrast_stats(inc, roi_m, ref_m)

            row = {
                "alpha": float(alpha),
                "file": str(source_path),
                "roi_over_reference": stats["roi_over_reference_mean"],
                "roi_vs_reference_db": stats["roi_vs_reference_db_mean"],
                "rf_rms": float(md["raw_scattered_rf_rms"]),
            }
            rows.append(row)
            imgs_for_path[float(alpha)] = (inc, stats)
        rows_by_path[path_name] = rows
        image_cache[path_name] = imgs_for_path

    # Actual full-dataset endpoint, useful even if continuation did not include
    # full alpha=1 due to a partial run.
    full_imgs = born.adjoint_per_angle(D_full, u_tx, zero_ds)
    _, full_inc_t, _ = diagnostics(full_imgs, pad)
    full_inc = full_inc_t.detach().cpu().numpy()
    full_actual_stats = contrast_stats(full_inc, roi_m, ref_m)

    path_summary = {}
    for path_name, rows in rows_by_path.items():
        alpha = [r["alpha"] for r in rows]
        y = [r["roi_vs_reference_db"] for r in rows]
        path_summary[path_name] = {
            "zero_crossing_alpha": zero_crossing(alpha, y),
            "min_contrast_db": float(np.min(y)),
            "max_contrast_db": float(np.max(y)),
            "alpha_at_min": float(alpha[int(np.argmin(y))]),
            "rows": rows,
        }

    interpretation = path_interpretation(path_summary)

    curve_fig = f"{sample_id}_physics_continuation_curves.png"
    heat_fig = f"{sample_id}_physics_continuation_heatmap.png"
    save_curves(
        out / curve_fig, sample_id, rows_by_path,
        truth_stats["roi_vs_reference_db_mean"], dpi)
    save_heatmap(out / heat_fig, sample_id, rows_by_path, dpi)

    image_figs = {}
    for path_name, images in image_cache.items():
        name = f"{sample_id}_{path_name}_continuation_images.png"
        save_selected_images(
            out / name, sample_id, path_name, images,
            x_mm, z_mm, roi, margin_mm, db_range, dpi)
        image_figs[path_name] = name

    return {
        "sample": sample_id,
        "roi_mm": list(map(float, roi)),
        "reference_margin_mm": float(margin_mm),
        "truth": truth_stats,
        "full_dataset_endpoint": full_actual_stats,
        "paths": path_summary,
        "interpretation": interpretation,
        "figures": {
            "curves": curve_fig,
            "heatmap": heat_fig,
            "path_images": image_figs,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="used for acquisition config / padding only")
    p.add_argument(
        "--continuation-root", type=Path,
        default=Path("/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle/physics_continuation"))
    p.add_argument("--sample-ids", nargs="+", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--imaging-n-freq", type=int, default=0)
    p.add_argument("--roi-mm", type=float, nargs=4, required=True,
                   metavar=("X0","X1","Z0","Z1"))
    p.add_argument("--reference-margin-mm", type=float, default=3.0)
    p.add_argument("--db-range", type=float, default=60.0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.reference_margin_mm <= 0:
        p.error("--reference-margin-mm must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    rows = []
    for sid in args.sample_ids:
        row = analyze_one(
            sid, args.checkpoint, args.continuation_root,
            args.imaging_n_freq, device, args.roi_mm,
            args.reference_margin_mm, args.db_range, args.dpi, args.out)
        rows.append(row)
        print(json.dumps({
            "event": "physics_continuation_analysis",
            "sample": sid,
            "interpretation": row["interpretation"],
            "zero_crossings": {
                k: v["zero_crossing_alpha"]
                for k, v in row["paths"].items()
            },
            "full_dataset_endpoint_db":
                row["full_dataset_endpoint"]["roi_vs_reference_db_mean"],
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {
        "checkpoint": str(args.checkpoint),
        "continuation_root": str(args.continuation_root),
        "path_meanings": {
            "c_only": "sound-speed contrast only",
            "rho_only": "density contrast only",
            "attenuation_only": "attenuation contrast only",
            "c_plus_rho": "sound speed + density",
            "full": "sound speed + density + attenuation",
        },
        "rows": rows,
    }
    (args.out / "physics_continuation_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event":"done","out":str(args.out)}), flush=True)


if __name__ == "__main__":
    main()

"""Analyze UltraWave numerical Frechet/JVP convergence and dark-region visibility.

Input files are produced by generate_ultrawave_frechet_jvp.py and contain
scattered RF D(eps) along the real-medium path.  The numerical directional
Jacobian estimate is

    J_eps = D(eps) / eps.

The script first checks whether J_eps converges as eps -> 0 in the RF/data
domain.  It then reconstructs each J_eps with the same uniform-background
oversampled Born adjoint used elsewhere and compares the suspicious ROI against
its surrounding reference ring.

The decisive comparison is:
- JVP image dark like full UltraWave image -> dark region already exists at
  first order in the full-wave solver; stored proxy m is the mismatch.
- JVP image not dark but full UltraWave image dark -> higher-order/full-contrast
  effects are required to explain the dark region.
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


def load_jvp_files(root: Path, sample_id: str):
    sample_dir = root / sample_id
    if not sample_dir.exists():
        raise FileNotFoundError(sample_dir)
    rows = []
    for path in sorted(sample_dir.glob("eps_*.npz")):
        with np.load(path) as f:
            rf = np.asarray(f["rf"], dtype=np.float32)
            md = json.loads(str(f["metadata_json"].item()))
        if md.get("sample_id") != sample_id:
            raise RuntimeError(f"sample mismatch in {path}")
        eps = float(md["frechet_epsilon"])
        if rf.shape != (11, 192, 2401):
            raise RuntimeError(f"unexpected RF shape in {path}: {rf.shape}")
        rows.append((eps, path, rf, md))
    if not rows:
        raise FileNotFoundError(f"no eps_*.npz files in {sample_dir}")
    rows.sort(key=lambda x: x[0])
    return rows


def complex_coherence_np(a, b):
    aa = np.asarray(a).astype(np.complex128, copy=False).ravel()
    bb = np.asarray(b).astype(np.complex128, copy=False).ravel()
    num = abs(np.vdot(aa, bb))
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(num / max(den, 1e-30))


def direct_rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-30))


def roi_ratio(arr, roi_m, ref_m):
    return contrast_stats(np.asarray(arr), roi_m, ref_m)


def percentile_ref(arr, q=99.8):
    a = np.asarray(arr)
    return float(np.percentile(a[np.isfinite(a)], q))


def db_map(arr, ref, db_range):
    a = np.asarray(arr)
    return np.clip(
        20.0 * np.log10(np.maximum(a, 1e-12) / max(ref, 1e-12)),
        -db_range, 0.0)


def add_regions(ax, roi, margin_mm):
    x0, x1, z0, z1 = roi
    xl, xr = min(x0, x1), max(x0, x1)
    zt, zb = min(z0, z1), max(z0, z1)
    ax.add_patch(plt.Rectangle(
        (xl, zt), xr-xl, zb-zt,
        fill=False, edgecolor="red", linewidth=1.7))
    ax.add_patch(plt.Rectangle(
        (xl-margin_mm, zt-margin_mm),
        (xr-xl)+2*margin_mm, (zb-zt)+2*margin_mm,
        fill=False, edgecolor="cyan", linestyle="--", linewidth=1.2))


def save_convergence(path, eps_rows, dpi):
    eps = np.asarray([r["epsilon"] for r in eps_rows])
    rel = np.asarray([r["rel_l2_to_smallest"] for r in eps_rows])
    coh = np.asarray([r["coherence_to_smallest"] for r in eps_rows])
    rms = np.asarray([r["jvp_rf_rms"] for r in eps_rows])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    axes[0].plot(eps, rel, marker="o")
    axes[0].set(xlabel="epsilon", ylabel="relative L2 to smallest-eps JVP",
                title="JVP convergence: relative difference")
    axes[1].plot(eps, coh, marker="o")
    axes[1].set(xlabel="epsilon", ylabel="complex coherence",
                ylim=(0, 1.01), title="JVP convergence: coherence")
    axes[2].plot(eps, rms, marker="o")
    axes[2].set(xlabel="epsilon", ylabel="RMS of D(eps)/eps",
                title="JVP amplitude stability")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_image_comparison(path, sample_id, truth, jvp_images, full_inc,
                          x_mm, z_mm, roi, margin_mm, db_range, dpi, stats):
    n = 2 + len(jvp_images)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5.2), constrained_layout=True)
    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]

    panels = [("Truth |m|", truth, "truth_abs_m")]
    for eps, arr in jvp_images:
        panels.append((f"UltraWave JVP incoherent\neps={eps:g}", arr, f"jvp_eps_{eps:g}"))
    panels.append(("UltraWave full incoherent", full_inc, "full_ultrawave"))

    for ax, (title, arr, key) in zip(axes, panels):
        ref = percentile_ref(arr)
        im = ax.imshow(
            db_map(arr, ref, db_range), cmap="gray", vmin=-db_range, vmax=0,
            extent=extent, aspect="auto")
        s = stats[key]
        ax.set_title(
            f"{title}\nROI/ref={s['roi_over_reference_mean']:.3f} "
            f"({s['roi_vs_reference_db_mean']:+.2f} dB)")
        add_regions(ax, roi, margin_mm)
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")
        fig.colorbar(im, ax=ax, shrink=0.76, label="dB, own robust scale")

    fig.suptitle(f"{sample_id}: numerical UltraWave first-order vs full-contrast")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def classify(stats, smallest_eps, dark_db=-3.0):
    jkey = f"jvp_eps_{smallest_eps:g}"
    jvp_db = stats[jkey]["roi_vs_reference_db_mean"]
    full_db = stats["full_ultrawave"]["roi_vs_reference_db_mean"]
    truth_db = stats["truth_abs_m"]["roi_vs_reference_db_mean"]

    if truth_db <= dark_db:
        return {
            "label": "truth_reflectivity_already_dark",
            "reason": "Stored truth |m| is itself dark relative to the reference ring.",
        }
    if jvp_db <= dark_db and full_db <= dark_db:
        return {
            "label": "dark_region_present_at_first_order_full_wave",
            "reason": (
                "The numerical UltraWave Jacobian image is already dark in the ROI, "
                "so higher-order/multiple-scattering effects are not required to "
                "create the dark region.  The stored proxy m is not the correct "
                "first-order observable for this acquisition."
            ),
        }
    if jvp_db > dark_db and full_db <= dark_db:
        return {
            "label": "dark_region_requires_full_contrast_higher_order_effects",
            "reason": (
                "The numerical first-order UltraWave response does not reproduce "
                "the dark ROI, but the full-contrast UltraWave data do."
            ),
        }
    return {
        "label": "inconclusive_or_not_strongly_dark",
        "reason": "ROI/reference contrasts do not match a conservative decision pattern.",
    }


@torch.no_grad()
def analyze_one(sample_id, checkpoint, frechet_root, imaging_n_freq, device,
                roi, margin_mm, db_range, dpi, out):
    sample = torch.load(
        DATA_ROOT / "shards" / f"{sample_id}.pt",
        map_location="cpu", weights_only=False)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt["args"]
    pad = int(saved_args.get("pad", 32))

    cfg, meta, born = build_imaging_operator(
        saved_args, sample, pad, imaging_n_freq, device)

    eps_files = load_jvp_files(frechet_root, sample_id)
    smallest_eps, _, smallest_rf, _ = eps_files[0]
    smallest_jvp_rf = smallest_rf.astype(np.float64) / smallest_eps

    conv_rows = []
    jvp_images = []
    stats = {}

    x_mm, z_mm = axis_coords(cfg, meta, born)
    roi_m = rectangular_mask(x_mm, z_mm, roi)
    ref_m = ring_mask(x_mm, z_mm, roi, margin_mm)

    truth = sample["m"].abs().cpu().numpy()
    stats["truth_abs_m"] = roi_ratio(truth, roi_m, ref_m)

    # Uniform imaging operator shared by all JVP/full data.
    full_rf = sample["rf"][None].to(device)
    D_full = rf_to_D(full_rf, meta)
    all_idx = torch.arange(D_full.shape[1], device=device)
    zero_ds = D_full.real.new_zeros(1, born.nz, born.nx)
    u_tx = born.transmit_fields(zero_ds, all_idx)

    for eps, path, rf, md in eps_files:
        jvp_rf = rf.astype(np.float64) / eps
        rel = direct_rel_l2(jvp_rf, smallest_jvp_rf)
        coh = complex_coherence_np(jvp_rf, smallest_jvp_rf)
        rms = float(np.sqrt(np.mean(jvp_rf**2)))

        rf_t = torch.from_numpy(jvp_rf.astype(np.float32))[None].to(device)
        D = rf_to_D(rf_t, meta)
        imgs = born.adjoint_per_angle(D, u_tx, zero_ds)
        _, inc_t, _ = diagnostics(imgs, pad)
        inc = inc_t.detach().cpu().numpy()

        key = f"jvp_eps_{eps:g}"
        stats[key] = roi_ratio(inc, roi_m, ref_m)
        jvp_images.append((eps, inc))
        conv_rows.append({
            "epsilon": float(eps),
            "file": str(path),
            "rel_l2_to_smallest": rel,
            "coherence_to_smallest": coh,
            "jvp_rf_rms": rms,
            "stored_estimated_jvp_rf_rms": float(md["estimated_jvp_rf_rms"]),
            "roi_over_reference": stats[key]["roi_over_reference_mean"],
            "roi_vs_reference_db": stats[key]["roi_vs_reference_db_mean"],
        })

    full_imgs = born.adjoint_per_angle(D_full, u_tx, zero_ds)
    _, full_inc_t, _ = diagnostics(full_imgs, pad)
    full_inc = full_inc_t.detach().cpu().numpy()
    stats["full_ultrawave"] = roi_ratio(full_inc, roi_m, ref_m)

    decision = classify(stats, smallest_eps)

    conv_fig = f"{sample_id}_frechet_convergence.png"
    image_fig = f"{sample_id}_frechet_vs_full.png"
    save_convergence(out / conv_fig, conv_rows, dpi)
    save_image_comparison(
        out / image_fig, sample_id, truth, jvp_images, full_inc,
        x_mm, z_mm, roi, margin_mm, db_range, dpi, stats)

    return {
        "sample": sample_id,
        "smallest_epsilon": float(smallest_eps),
        "roi_mm": list(map(float, roi)),
        "reference_margin_mm": float(margin_mm),
        "convergence": conv_rows,
        "statistics": stats,
        "decision": decision,
        "figures": {
            "convergence": conv_fig,
            "jvp_vs_full": image_fig,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="used for acquisition config / padding only")
    p.add_argument("--frechet-root", type=Path,
                   default=Path("/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle/frechet_jvp"))
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

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    rows = []
    for sid in args.sample_ids:
        row = analyze_one(
            sid, args.checkpoint, args.frechet_root,
            args.imaging_n_freq, device, args.roi_mm,
            args.reference_margin_mm, args.db_range, args.dpi, args.out)
        rows.append(row)
        print(json.dumps({
            "event": "frechet_jvp_analysis",
            "sample": sid,
            "smallest_epsilon": row["smallest_epsilon"],
            "decision": row["decision"],
            "convergence": row["convergence"],
            "full_roi_ref_db": row["statistics"]["full_ultrawave"]["roi_vs_reference_db_mean"],
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {
        "checkpoint": str(args.checkpoint),
        "frechet_root": str(args.frechet_root),
        "interpretation": {
            "jvp_dark_and_full_dark":
                "dark region exists at first order in UltraWave; proxy m is mismatched",
            "jvp_not_dark_but_full_dark":
                "higher-order/full-contrast effects are required",
            "caution":
                "Use the decision only after JVP convergence across epsilons is satisfactory.",
        },
        "rows": rows,
    }
    (args.out / "ultrawave_frechet_jvp_diagnostic.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "done", "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()

"""Overlay the ASP lateral-Nyquist receive limit on point-scatterer residuals.

This is a follow-up to ``analyze_point_scatterer_green.py``.  It reuses the
same weak UltraWave point-scatterer files and current BornModel prediction, but
asks a more specific question: does the element/frequency region with large
Born <-> UltraWave residual phase coincide with the lateral spatial-frequency
limit of the ASP propagation grid?

For a propagation grid with lateral spacing dx, the FFT can represent

    |kx| <= pi / dx.

For a homogeneous propagating wave, kx = k sin(theta), hence

    sin(theta_max(f)) = min(1, c0 / (2 f dx)).

A point at (xp, zp) can therefore reach only surface positions satisfying

    |xe - xp| <= zp tan(theta_max(f))

within the propagating spectrum represented by that grid.  The script overlays
that theoretical boundary on the residual phase maps and separately measures
Born/UltraWave coherence inside and outside the predicted valid aperture.

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

from common import rf_to_D  # noqa: E402
from physics.imaging import BornModel  # noqa: E402
from scripts.analyze_point_scatterer_green import (  # noqa: E402
    C0,
    MODEL_D,
    born_point_data,
    build_point_meta,
    phase_residuals,
    point_files,
)
from scripts.diagnose_born_ultrawave_mismatch import data_fit_metrics  # noqa: E402
from scripts.pilot_phase_asp import padded_meta  # noqa: E402


def nyquist_bounds(meta, md: dict, dx_prop: float, c0: float = C0):
    """Return left/right element-index limits versus frequency.

    The limits describe the receive locations whose straight homogeneous ray
    from the calibration point has |kx| below the FFT Nyquist limit pi/dx.
    Values are fractional element indices so they can be overlaid directly on
    imshow axes.
    """
    freqs = np.asarray(meta.freqs, dtype=np.float64)
    xe = np.asarray(meta.xe_coords, dtype=np.float64)
    if xe.ndim != 1 or len(xe) != meta.n_elements or not np.all(np.diff(xe) > 0):
        raise RuntimeError("expected strictly increasing element coordinates")

    xp = float(md["model_x_m"])
    zp = float(md["model_z_m"])
    ratio = c0 / (2.0 * freqs * dx_prop)
    full = ratio >= 1.0
    clipped = np.clip(ratio, 0.0, 1.0)
    theta_max = np.arcsin(clipped)

    reach = np.empty_like(freqs)
    reach[full] = np.inf
    reach[~full] = zp * np.tan(theta_max[~full])

    elem_index = np.arange(meta.n_elements, dtype=np.float64)
    left = np.zeros_like(freqs)
    right = np.full_like(freqs, meta.n_elements - 1, dtype=np.float64)
    finite = ~full
    left[finite] = np.interp(
        xp - reach[finite], xe, elem_index, left=0.0,
        right=float(meta.n_elements - 1))
    right[finite] = np.interp(
        xp + reach[finite], xe, elem_index, left=0.0,
        right=float(meta.n_elements - 1))

    return {
        "left_element": left,
        "right_element": right,
        "theta_max_rad": theta_max,
        "reach_m": reach,
        "ratio": ratio,
    }


def masked_coherence(pred_f: torch.Tensor, target_f: torch.Tensor,
                     element_mask: np.ndarray):
    """Normalized complex coherence over batch, angle and selected elements."""
    mask = torch.as_tensor(element_mask, dtype=torch.bool, device=pred_f.device)
    if int(mask.sum()) < 1:
        return float("nan")
    p = pred_f[..., mask]
    t = target_f[..., mask]
    num = (t * p.conj()).sum().abs()
    den = (t.abs().square().sum() * p.abs().square().sum()).sqrt()
    return float((num / den.clamp_min(1e-30)).detach().cpu())


def aperture_coherence_curves(pred: torch.Tensor, target: torch.Tensor,
                              bounds: dict):
    n_freq = pred.shape[2]
    n_elem = pred.shape[3]
    inside = np.full(n_freq, np.nan, dtype=np.float64)
    outside = np.full(n_freq, np.nan, dtype=np.float64)
    frac = np.zeros(n_freq, dtype=np.float64)
    elem = np.arange(n_elem)

    for fi in range(n_freq):
        lo = bounds["left_element"][fi]
        hi = bounds["right_element"][fi]
        in_mask = (elem >= np.ceil(lo)) & (elem <= np.floor(hi))
        out_mask = ~in_mask
        frac[fi] = float(in_mask.mean())
        inside[fi] = masked_coherence(pred[:, :, fi, :], target[:, :, fi, :], in_mask)
        if out_mask.any():
            outside[fi] = masked_coherence(
                pred[:, :, fi, :], target[:, :, fi, :], out_mask)

    return inside, outside, frac


def plot_phase_overlay(path: Path, md: dict, meta, phase: np.ndarray,
                       bounds: dict, dx_prop: float, dpi: int):
    fi = int(np.argmin(np.abs(np.asarray(meta.freqs) - 5.75e6)))
    ai = int(np.argmin(np.abs(np.asarray(meta.angles_deg))))
    fmhz = float(meta.freqs[fi] * 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5), constrained_layout=True)
    im = axes[0].imshow(
        phase[:, fi, :], cmap="twilight", vmin=-np.pi, vmax=np.pi,
        aspect="auto", origin="lower",
        extent=[0, meta.n_elements - 1,
                meta.angles_deg[0], meta.angles_deg[-1]])
    lo = float(bounds["left_element"][fi])
    hi = float(bounds["right_element"][fi])
    axes[0].axvline(lo, ls="--", lw=1.6, color="white",
                    label="ASP lateral-Nyquist limit")
    axes[0].axvline(hi, ls="--", lw=1.6, color="white")
    axes[0].set(
        title=f"Residual phase @ {fmhz:.2f} MHz | predicted valid e=[{lo:.1f},{hi:.1f}]",
        xlabel="Element index", ylabel="Transmit angle [deg]")
    axes[0].legend(loc="lower center", fontsize=8)
    fig.colorbar(im, ax=axes[0], label="Phase [rad]")

    im2 = axes[1].imshow(
        phase[ai], cmap="twilight", vmin=-np.pi, vmax=np.pi,
        aspect="auto", origin="lower",
        extent=[0, meta.n_elements - 1,
                meta.freqs[0] * 1e-6, meta.freqs[-1] * 1e-6])
    fmhz_all = np.asarray(meta.freqs) * 1e-6
    axes[1].plot(bounds["left_element"], fmhz_all, "w--", lw=1.6,
                 label="ASP lateral-Nyquist limit")
    axes[1].plot(bounds["right_element"], fmhz_all, "w--", lw=1.6)
    axes[1].set(
        title=f"Residual phase @ {meta.angles_deg[ai]:+.1f} deg",
        xlabel="Element index", ylabel="Frequency [MHz]")
    axes[1].legend(loc="upper center", fontsize=8)
    fig.colorbar(im2, ax=axes[1], label="Phase [rad]")

    theta_deg = np.degrees(bounds["theta_max_rad"][fi])
    fig.suptitle(
        f"Nyquist overlay | point ({md['model_x_m']*1e3:.2f},"
        f"{md['model_z_m']*1e3:.2f}) mm | dx_prop={dx_prop*1e3:.3f} mm | "
        f"theta_max({fmhz:.2f} MHz)={theta_deg:.1f} deg")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_aperture_coherence(path: Path, meta, inside: np.ndarray,
                            outside: np.ndarray, frac: np.ndarray,
                            dpi: int):
    fmhz = np.asarray(meta.freqs) * 1e-6
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    ax.plot(fmhz, inside, label="Inside predicted ASP-valid aperture")
    ax.plot(fmhz, outside, label="Outside predicted ASP-valid aperture")
    ax.set(xlabel="Frequency [MHz]", ylabel="Born ↔ UltraWave coherence",
           ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax2 = ax.twinx()
    ax2.plot(fmhz, frac, ls="--", alpha=0.65,
             label="Valid aperture fraction")
    ax2.set_ylabel("Predicted valid aperture fraction")
    ax2.set_ylim(0, 1.02)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def analyze_file(path: Path, args, device):
    with np.load(path) as f:
        rf_np = np.asarray(f["rf"], dtype=np.float32)
        md = json.loads(str(f["metadata_json"].item()))

    rf = torch.from_numpy(rf_np)[None].to(device)
    cfg, meta = build_point_meta(args.config, md, args.imaging_n_freq)
    pmeta = padded_meta(meta, args.pad, cfg.grid.dx)
    born = BornModel(
        pmeta, cfg.grid.nx + 2 * args.pad, cfg.grid.nz,
        cfg.grid.dx, cfg.grid.dz, cfg.physics.c0,
        eps=cfg.physics.eps_evanescent,
        spreading=cfg.physics.spreading,
    ).to(device)

    target = rf_to_D(rf, meta)
    pred = born_point_data(born, md, args.pad, device)
    fit, gain, _ = data_fit_metrics(pred, target)
    aligned = pred * gain[None, None, :, None]
    phase = phase_residuals(aligned, target)

    dx_prop = float(cfg.grid.dx)
    bounds = nyquist_bounds(meta, md, dx_prop, cfg.physics.c0)
    inside, outside, frac = aperture_coherence_curves(aligned, target, bounds)

    stem = path.stem
    phase_name = f"{stem}_nyquist_phase.png"
    coh_name = f"{stem}_nyquist_aperture_coherence.png"
    plot_phase_overlay(args.out / phase_name, md, meta, phase, bounds,
                       dx_prop, args.dpi)
    plot_aperture_coherence(args.out / coh_name, meta, inside, outside,
                            frac, args.dpi)

    fi = int(np.argmin(np.abs(np.asarray(meta.freqs) - 5.75e6)))
    target_power = target.abs().square().sum(dim=(0, 1, 3)).detach().cpu().numpy()
    valid_f = target_power >= target_power.max() * args.summary_power_frac

    def valid_mean(x):
        x = np.asarray(x)
        mask = valid_f & np.isfinite(x)
        return float(np.mean(x[mask])) if mask.any() else float("nan")

    return {
        "file": str(path),
        "point_id": md["point_id"],
        "model_x_mm": float(md["model_x_m"] * 1e3),
        "model_z_mm": float(md["model_z_m"] * 1e3),
        "dx_prop_mm": dx_prop * 1e3,
        "fit": fit,
        "reference_frequency_mhz": float(meta.freqs[fi] * 1e-6),
        "reference_theta_max_deg": float(np.degrees(bounds["theta_max_rad"][fi])),
        "reference_left_element": float(bounds["left_element"][fi]),
        "reference_right_element": float(bounds["right_element"][fi]),
        "mean_inside_coherence": valid_mean(inside),
        "mean_outside_coherence": valid_mean(outside),
        "mean_valid_aperture_fraction": valid_mean(frac),
        "summary_frequency_power_fraction": args.summary_power_frac,
        "figures": {
            "phase_overlay": phase_name,
            "aperture_coherence": coh_name,
        },
    }


def aggregate(rows):
    keys = ["mean_inside_coherence", "mean_outside_coherence",
            "mean_valid_aperture_fraction"]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path,
                   default=Path("/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle/point_scatterer_green"))
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--imaging-n-freq", type=int, default=0)
    p.add_argument("--summary-power-frac", type=float, default=1e-4,
                   help="ignore very weak band-edge frequencies in scalar summaries")
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not (0 < args.summary_power_frac < 1):
        p.error("--summary-power-frac must be in (0,1)")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    rows = []
    for path in point_files(args.input_root):
        row = analyze_file(path, args, device)
        rows.append(row)
        print(json.dumps({
            "event": "nyquist_point_analyzed",
            "point": row["point_id"],
            "inside": row["mean_inside_coherence"],
            "outside": row["mean_outside_coherence"],
            "valid_fraction": row["mean_valid_aperture_fraction"],
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {"aggregate": aggregate(rows), "rows": rows}
    (args.out / "point_scatterer_nyquist_diagnostic.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "done", "aggregate": payload["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()

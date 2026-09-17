"""Compare the current 0.2 mm ASP grid with a 0.1 mm lateral propagation grid.

This diagnostic reuses the weak UltraWave point-scatterer data. The parameter
/ source voxel remains 0.2 mm x 0.2 mm, but the homogeneous angular-spectrum
propagation grid is oversampled laterally from 0.2 mm to 0.1 mm. The fine grid
uses two lateral samples for the same 0.2 mm-wide point voxel and keeps the
0.2 mm depth sampling unchanged.

The construction preserves the physical lateral cell-edge span. If the coarse
unpadded grid has cell centres x0 + i*dx, the fine grid uses

    dx_fine = dx / 2
    x0_fine = x0 - dx_fine / 2
    nx_fine = 2 * nx

so both grids have the same lateral cell edges. Padding is doubled in pixels,
keeping the same physical padding width.

No new UltraWave simulation is required. A large improvement of full-aperture
angle/element coherence at 0.1 mm would directly support lateral spatial-
frequency truncation as the dominant point-response mismatch.
"""
from __future__ import annotations

import argparse
import copy
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
    MODEL_D,
    MODEL_NX,
    MODEL_NZ,
    born_point_data,
    build_point_meta,
    grouped_curves,
    phase_residuals,
    point_files,
    source_to_data,
)
from scripts.diagnose_born_ultrawave_mismatch import data_fit_metrics  # noqa: E402
from scripts.diagnose_two_parameter_born import grouped_coherence  # noqa: E402
from scripts.pilot_phase_asp import padded_meta  # noqa: E402

FINE_FACTOR = 2


def build_born(meta, nx: int, nz: int, dx: float, dz: float, pad: int,
               c0: float, eps: float, spreading: str, device):
    pmeta = padded_meta(meta, pad, dx)
    return BornModel(
        pmeta, nx + 2 * pad, nz, dx, dz, c0,
        eps=eps, spreading=spreading,
    ).to(device)


def fine_meta_from_coarse(meta, coarse_dx: float):
    result = copy.deepcopy(meta)
    dx_fine = coarse_dx / FINE_FACTOR
    # Preserve physical cell-edge span: first fine centre is half a fine cell
    # to the left of the first coarse centre.
    result.x0 = float(meta.x0) - dx_fine / 2.0
    return result


def fine_point_data(born: BornModel, md: dict, pad_fine: int, device):
    """Born data for the same 0.2 mm-wide c-only voxel on the 0.1 mm x grid."""
    nx_fine = MODEL_NX * FINE_FACTOR
    zero = torch.zeros(
        1, MODEL_NZ, nx_fine + 2 * pad_fine,
        dtype=torch.float32, device=device,
    )
    u0 = born.transmit_fields(zero)

    # Unpadded fine-grid cell centres. The 0.2 mm UltraWave target occupies
    # exactly two 0.1 mm lateral cells and one 0.2 mm depth slab.
    x0_unpadded = born.x0 + pad_fine * born.dx
    x = x0_unpadded + torch.arange(
        nx_fine, device=device, dtype=torch.float32) * born.dx
    xp = float(md["model_x_m"])
    selected = (x >= xp - MODEL_D / 2.0) & (x < xp + MODEL_D / 2.0)
    if int(selected.sum()) != FINE_FACTOR:
        raise RuntimeError(
            f"expected {FINE_FACTOR} fine lateral cells for point voxel, got "
            f"{int(selected.sum())}")

    chi = torch.zeros(1, MODEL_NZ, nx_fine, dtype=torch.float32, device=device)
    iz = int(md["model_iz"])
    dc = float(md["delta_c_frac"])
    chi_value = (1.0 / (1.0 + dc)) ** 2 - 1.0
    chi[:, iz, selected] = chi_value
    chi = torch.nn.functional.pad(chi, (pad_fine, pad_fine))

    q = chi[:, None, None] * u0 * born.w_z
    return source_to_data(born, q, zero)


def target_power_mask(target: torch.Tensor, frac: float):
    power = target.abs().square().sum(dim=(0, 1, 3))
    keep = power >= power.max() * frac
    if int(keep.sum()) < 2:
        raise RuntimeError("frequency power mask retained fewer than two bins")
    return keep


def masked_scalar_summary(aligned: torch.Tensor, target: torch.Tensor,
                          valid_f: torch.Tensor):
    a = aligned[:, :, valid_f, :]
    t = target[:, :, valid_f, :]

    fcoh = grouped_coherence(a, t, 2)
    acoh = grouped_coherence(a, t, 1)
    ecoh = grouped_coherence(a, t, 3)
    residual = ((a - t).abs().square().sum()
                / t.abs().square().sum().clamp_min(1e-30)).sqrt()
    return {
        "mean_frequency_coherence": float(fcoh.mean().detach().cpu()),
        "mean_angle_coherence": float(acoh.mean().detach().cpu()),
        "mean_element_coherence": float(ecoh.mean().detach().cpu()),
        "relative_residual": float(residual.detach().cpu()),
        "n_valid_frequencies": int(valid_f.sum().item()),
    }


def evaluate_prediction(pred: torch.Tensor, target: torch.Tensor,
                        valid_f: torch.Tensor):
    fit, gain, _ = data_fit_metrics(pred, target)
    aligned = pred * gain[None, None, :, None]
    curves = grouped_curves(aligned, target)
    phase = phase_residuals(aligned, target)
    summary = masked_scalar_summary(aligned, target, valid_f)
    return {
        "fit": fit,
        "curves": curves,
        "phase": phase,
        "summary": summary,
    }


def plot_coherence_comparison(path: Path, md: dict, meta,
                              coarse: dict, fine: dict, valid_f: torch.Tensor,
                              dpi: int):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
    xs = [
        np.asarray(meta.freqs) * 1e-6,
        np.asarray(meta.angles_deg),
        np.arange(meta.n_elements),
    ]
    keys = ["frequency", "angle", "element"]
    xlabels = ["Frequency [MHz]", "Transmit angle [deg]", "Element index"]
    titles = ["Frequency coherence", "Angle coherence", "Element coherence"]

    for ax, x, key, xlabel, title in zip(axes, xs, keys, xlabels, titles):
        ax.plot(x, coarse["curves"][key], label="ASP dx=0.2 mm")
        ax.plot(x, fine["curves"][key], label="ASP dx=0.1 mm")
        ax.set(xlabel=xlabel, ylabel="Born ↔ UltraWave coherence",
               title=title, ylim=(0, 1.02))
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    valid_np = valid_f.detach().cpu().numpy()
    if valid_np.any():
        fmhz = np.asarray(meta.freqs) * 1e-6
        axes[0].axvspan(fmhz[valid_np][0], fmhz[valid_np][-1],
                        alpha=0.08, color="gray")

    fig.suptitle(
        f"Point ({md['model_x_m']*1e3:.2f}, {md['model_z_m']*1e3:.2f}) mm | "
        f"same 0.2 mm voxel, lateral propagation oversampling only")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_phase_comparison(path: Path, md: dict, meta,
                          coarse_phase: np.ndarray, fine_phase: np.ndarray,
                          dpi: int):
    ai = int(np.argmin(np.abs(np.asarray(meta.angles_deg))))
    extent = [0, meta.n_elements - 1,
              meta.freqs[0] * 1e-6, meta.freqs[-1] * 1e-6]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    for ax, phase, title in zip(
            axes, [coarse_phase, fine_phase],
            ["Residual phase: ASP dx=0.2 mm", "Residual phase: ASP dx=0.1 mm"]):
        im = ax.imshow(
            phase[ai], cmap="twilight", vmin=-np.pi, vmax=np.pi,
            aspect="auto", origin="lower", extent=extent)
        ax.set(title=title, xlabel="Element index", ylabel="Frequency [MHz]")
        fig.colorbar(im, ax=ax, label="Phase [rad]")
    fig.suptitle(
        f"After per-frequency g(f) | point "
        f"({md['model_x_m']*1e3:.2f},{md['model_z_m']*1e3:.2f}) mm")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def analyze_file(path: Path, args, device):
    with np.load(path) as f:
        rf_np = np.asarray(f["rf"], dtype=np.float32)
        md = json.loads(str(f["metadata_json"].item()))

    rf = torch.from_numpy(rf_np)[None].to(device)
    cfg, meta = build_point_meta(args.config, md, args.imaging_n_freq)
    target = rf_to_D(rf, meta)
    valid_f = target_power_mask(target, args.summary_power_frac)

    # Current grid: 0.2 mm lateral propagation.
    coarse_born = build_born(
        meta, cfg.grid.nx, cfg.grid.nz, cfg.grid.dx, cfg.grid.dz,
        args.pad, cfg.physics.c0, cfg.physics.eps_evanescent,
        cfg.physics.spreading, device)
    coarse_pred = born_point_data(coarse_born, md, args.pad, device)
    coarse = evaluate_prediction(coarse_pred, target, valid_f)

    # Oversampled grid: same physical aperture/padding, dx=0.1 mm only.
    fine_dx = cfg.grid.dx / FINE_FACTOR
    fine_pad = args.pad * FINE_FACTOR
    fine_meta = fine_meta_from_coarse(meta, cfg.grid.dx)
    fine_born = build_born(
        fine_meta, cfg.grid.nx * FINE_FACTOR, cfg.grid.nz,
        fine_dx, cfg.grid.dz, fine_pad, cfg.physics.c0,
        cfg.physics.eps_evanescent, cfg.physics.spreading, device)
    fine_pred = fine_point_data(fine_born, md, fine_pad, device)
    fine = evaluate_prediction(fine_pred, target, valid_f)

    stem = path.stem
    coh_name = f"{stem}_oversampled_coherence.png"
    phase_name = f"{stem}_oversampled_phase.png"
    plot_coherence_comparison(
        args.out / coh_name, md, meta, coarse, fine, valid_f, args.dpi)
    plot_phase_comparison(
        args.out / phase_name, md, meta, coarse["phase"], fine["phase"], args.dpi)

    return {
        "file": str(path),
        "point_id": md["point_id"],
        "model_x_mm": float(md["model_x_m"] * 1e3),
        "model_z_mm": float(md["model_z_m"] * 1e3),
        "delta_c_frac": float(md["delta_c_frac"]),
        "summary_frequency_power_fraction": args.summary_power_frac,
        "coarse_dx_mm": float(cfg.grid.dx * 1e3),
        "fine_dx_mm": float(fine_dx * 1e3),
        "coarse": {
            "fit": coarse["fit"],
            "summary": coarse["summary"],
        },
        "fine": {
            "fit": fine["fit"],
            "summary": fine["summary"],
        },
        "delta_fine_minus_coarse": {
            k: fine["summary"][k] - coarse["summary"][k]
            for k in ("mean_frequency_coherence", "mean_angle_coherence",
                      "mean_element_coherence", "relative_residual")
        },
        "figures": {
            "coherence": coh_name,
            "phase": phase_name,
        },
    }


def aggregate(rows):
    metrics = ["mean_frequency_coherence", "mean_angle_coherence",
               "mean_element_coherence", "relative_residual"]
    out = {}
    for label in ("coarse", "fine"):
        out[label] = {
            k: float(np.mean([r[label]["summary"][k] for r in rows]))
            for k in metrics
        }
    out["delta_fine_minus_coarse"] = {
        k: out["fine"][k] - out["coarse"][k] for k in metrics
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path,
                   default=Path("/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle/point_scatterer_green"))
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--pad", type=int, default=32,
                   help="coarse-grid lateral padding; fine-grid pixels are doubled")
    p.add_argument("--imaging-n-freq", type=int, default=0)
    p.add_argument("--summary-power-frac", type=float, default=1e-4,
                   help="exclude target band-edge bins below this fraction of peak power")
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
            "event": "oversampled_point_analyzed",
            "point": row["point_id"],
            "coarse": row["coarse"]["summary"],
            "fine": row["fine"]["summary"],
            "delta": row["delta_fine_minus_coarse"],
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {"aggregate": aggregate(rows), "rows": rows}
    (args.out / "point_scatterer_oversampled_asp.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "done", "aggregate": payload["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()

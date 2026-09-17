"""Compare weak UltraWave point scatterers with the current Born Green model.

The companion generator creates one weak c-only 0.2 mm voxel in an otherwise
homogeneous medium. This analyzer constructs the same voxel in the current
BornModel and asks whether the measured angle/frequency/element field pattern
matches after removing an arbitrary shared per-frequency complex response.

This is intentionally a Green/operator calibration rather than an imaging
experiment. If the aligned point response is poor, the mismatch lies in the
Tx/Rx Green model, aperture/element sampling, surface transfer, or acquisition
convention. If it is good, complex-phantom mismatch should be sought in the
medium/source discretization instead.
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

from common import build_meta, load_config, rf_to_D  # noqa: E402
from physics.imaging import BornModel  # noqa: E402
from scripts.diagnose_born_ultrawave_mismatch import data_fit_metrics  # noqa: E402
from scripts.diagnose_two_parameter_born import grouped_coherence  # noqa: E402
from scripts.pilot_phase_asp import padded_meta  # noqa: E402

C0 = 1540.0
RHO0 = 1000.0
MODEL_D = 0.2e-3
MODEL_X0 = -0.019125
MODEL_Z0 = 0.000075
MODEL_NX = 192
MODEL_NZ = 216


def point_files(root: Path):
    files = sorted(root.glob("ix*_iz*_dc*.npz"))
    if not files:
        raise FileNotFoundError(f"no point-scatterer files under {root}")
    return files


def build_point_meta(config: str, md: dict, n_freq: int):
    cfg = load_config(config)
    cfg.array.center_x = 0.0
    cfg.grid.x0 = MODEL_X0
    cfg.grid.z0 = MODEL_Z0
    cfg.acq.t_ref_s = md["source_tref_s"]
    cfg.acq.n_freq = n_freq
    cfg.physics.response_mode = "calibrated"
    cfg.physics.pop("response_path", None)
    meta = build_meta(cfg)
    if not np.allclose(meta.angles_deg, md["angles_deg"], atol=1e-10):
        raise RuntimeError("angle grid mismatch")
    if len(meta.band_idx) > 1 and not np.all(np.diff(meta.band_idx) == 1):
        raise RuntimeError("point calibration requires contiguous full-band frequencies")
    return cfg, meta


def point_c_map(md: dict, device):
    c = torch.full((1, MODEL_NZ, MODEL_NX), C0,
                   dtype=torch.float32, device=device)
    iz = int(md["model_iz"])
    ix = int(md["model_ix"])
    c[:, iz, ix] = C0 * (1.0 + float(md["delta_c_frac"]))
    return c


def source_to_data(born: BornModel, source: torch.Tensor,
                   delta_s: torch.Tensor):
    d = born.asp.march_up(source, delta_s, born.omega_)
    d = born.asp._ifft(born.asp._fft(d) * born.surface_transfer)
    return born.sample(d)


def born_point_data(born: BornModel, md: dict, pad: int, device):
    zero = torch.zeros(1, MODEL_NZ, MODEL_NX + 2 * pad,
                       dtype=torch.float32, device=device)
    # All calibration acquisitions use every transmit angle. BornModel does
    # not retain the original metadata object; calling without angles_idx
    # correctly uses its internally registered full angle grid.
    u0 = born.transmit_fields(zero)

    c = point_c_map(md, device)
    cpad = torch.nn.functional.pad(c, (pad, pad), value=C0)
    chi = (C0 / cpad).square() - 1.0
    q = chi[:, None, None] * u0 * born.w_z
    return source_to_data(born, q, zero)


def grouped_curves(pred: torch.Tensor, target: torch.Tensor):
    return {
        "frequency": grouped_coherence(pred, target, 2).detach().cpu().numpy(),
        "angle": grouped_coherence(pred, target, 1).detach().cpu().numpy(),
        "element": grouped_coherence(pred, target, 3).detach().cpu().numpy(),
    }


def summarize_curves(curves: dict):
    return {f"mean_{k}_coherence": float(np.mean(v)) for k, v in curves.items()} | {
        f"median_{k}_coherence": float(np.median(v)) for k, v in curves.items()
    }


def phase_residuals(aligned: torch.Tensor, target: torch.Tensor):
    cross = target * aligned.conj()
    # Collapse batch only; preserve theta,freq,element.
    cross = cross.sum(dim=0)
    phase = torch.angle(cross)
    return phase.detach().cpu().numpy()


def plot_coherence(path: Path, md: dict, meta, raw: dict, aligned: dict, dpi: int):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.5), constrained_layout=True)
    xs = [np.asarray(meta.freqs) * 1e-6,
          np.asarray(meta.angles_deg),
          np.arange(meta.n_elements)]
    labels = [("Frequency [MHz]", "frequency"),
              ("Transmit angle [deg]", "angle"),
              ("Element index", "element")]
    for col, ((xlabel, key), x) in enumerate(zip(labels, xs)):
        axes[0, col].plot(x, raw[key])
        axes[1, col].plot(x, aligned[key])
        axes[0, col].set_title(f"Raw {key} coherence")
        axes[1, col].set_title(f"After per-frequency g(f): {key}")
        for row in (0, 1):
            axes[row, col].set(xlabel=xlabel, ylabel="Coherence", ylim=(0, 1.02))
            axes[row, col].grid(alpha=0.25)
    fig.suptitle(
        f"point ({md['model_x_m']*1e3:.2f}, {md['model_z_m']*1e3:.2f}) mm | "
        f"dc/c0={md['delta_c_frac']:.1e}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_phase(path: Path, md: dict, meta, phase: np.ndarray, dpi: int):
    # phase [theta,freq,element]
    fi = int(np.argmin(np.abs(np.asarray(meta.freqs) - 5.75e6)))
    fmhz = float(meta.freqs[fi] * 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), constrained_layout=True)
    im = axes[0].imshow(
        phase[:, fi, :], cmap="twilight", vmin=-np.pi, vmax=np.pi,
        aspect="auto", origin="lower",
        extent=[0, meta.n_elements - 1, meta.angles_deg[0], meta.angles_deg[-1]])
    axes[0].set(title=f"Residual phase @ {fmhz:.2f} MHz",
                xlabel="Element index", ylabel="Transmit angle [deg]")
    fig.colorbar(im, ax=axes[0], label="Phase [rad]")

    # Central angle: phase residual as function of frequency and element.
    ai = int(np.argmin(np.abs(np.asarray(meta.angles_deg))))
    im2 = axes[1].imshow(
        phase[ai], cmap="twilight", vmin=-np.pi, vmax=np.pi,
        aspect="auto", origin="lower",
        extent=[0, meta.n_elements - 1,
                meta.freqs[0] * 1e-6, meta.freqs[-1] * 1e-6])
    axes[1].set(title=f"Residual phase @ {meta.angles_deg[ai]:+.1f} deg",
                xlabel="Element index", ylabel="Frequency [MHz]")
    fig.colorbar(im2, ax=axes[1], label="Phase [rad]")
    fig.suptitle(
        f"Born ↔ UltraWave residual phase after g(f) | "
        f"point ({md['model_x_m']*1e3:.2f},{md['model_z_m']*1e3:.2f}) mm")
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

    # Reuse the exact robust per-frequency gain returned by data_fit_metrics.
    # That helper applies an energy-relative denominator floor at weak band
    # edges; recomputing the gain with only machine-epsilon clamping can
    # legitimately differ there and previously triggered a false assertion.
    if gain.ndim != 1 or gain.numel() != pred.shape[2]:
        raise RuntimeError("unexpected per-frequency gain shape")
    aligned_pred = pred * gain[None, None, :, None]

    raw_curves = grouped_curves(pred, target)
    aligned_curves = grouped_curves(aligned_pred, target)
    phase = phase_residuals(aligned_pred, target)

    stem = path.stem
    plot_coherence(args.out / f"{stem}_green_coherence.png", md, meta,
                   raw_curves, aligned_curves, args.dpi)
    plot_phase(args.out / f"{stem}_green_phase.png", md, meta, phase, args.dpi)

    mag = gain.abs().detach().cpu().numpy()
    ph = np.unwrap(np.angle(gain.detach().cpu().numpy()))
    return {
        "file": str(path),
        "point_id": md["point_id"],
        "model_x_mm": float(md["model_x_m"] * 1e3),
        "model_z_mm": float(md["model_z_m"] * 1e3),
        "delta_c_frac": float(md["delta_c_frac"]),
        "rf_rms": float(md["rf_rms"]),
        "fit": fit,
        "raw": summarize_curves(raw_curves),
        "after_per_frequency_gain": summarize_curves(aligned_curves),
        "system_response": {
            "gain_magnitude": mag.tolist(),
            "gain_phase_unwrapped_rad": ph.tolist(),
        },
        "phase_residual": {
            "mean_abs_rad": float(np.mean(np.abs(phase))),
            "median_abs_rad": float(np.median(np.abs(phase))),
        },
        "figures": {
            "coherence": f"{stem}_green_coherence.png",
            "phase": f"{stem}_green_phase.png",
        },
    }


def aggregate(rows):
    keys = ["mean_frequency_coherence", "mean_angle_coherence", "mean_element_coherence"]
    return {
        "raw": {k: float(np.mean([r["raw"][k] for r in rows])) for k in keys},
        "after_per_frequency_gain": {
            k: float(np.mean([r["after_per_frequency_gain"][k] for r in rows])) for k in keys
        },
        "mean_per_frequency_relative_residual": float(np.mean([
            r["fit"]["per_frequency_relative_residual"] for r in rows])),
        "mean_abs_phase_residual_rad": float(np.mean([
            r["phase_residual"]["mean_abs_rad"] for r in rows])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path,
                   default=Path("/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle/point_scatterer_green"))
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--imaging-n-freq", type=int, default=0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    rows = []
    for path in point_files(args.input_root):
        row = analyze_file(path, args, device)
        rows.append(row)
        print(json.dumps({
            "event": "point_analyzed",
            "point": row["point_id"],
            "aligned": row["after_per_frequency_gain"],
            "residual": row["fit"]["per_frequency_relative_residual"],
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {"aggregate": aggregate(rows), "rows": rows}
    (args.out / "point_scatterer_green_diagnostic.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "done", "aggregate": payload["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()

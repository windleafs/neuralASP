"""Closed-loop diagnostic for Born/ASP imaging versus UltraWave full-wave RF.

This script is deliberately independent of the learned phase-screen network.
It asks three progressively harder questions on the same sample:

1. Is the current Born/ASP operator self-consistent?
       m_truth --F_GT--> D_Born --F_GT^H--> image
2. How much does the true speed matter when the data are perfectly matched?
       D_Born --F_uniform^H--> image
3. With true speed fixed, how far is UltraWave RF from the Born model?
       D_UltraWave --F_GT^H--> image

A conventional straight-ray DAS image from the same UltraWave RF is included
as an independent imaging baseline.  The main figure is therefore

    Truth |m|
    | F_GT^H F_GT m
    | F_uniform^H F_GT m
    | F_GT^H D_UltraWave
    | DAS(D_UltraWave)

The script also fits the best *shared per-frequency complex response* g(w)
from the Born prediction to the UltraWave data.  The residual after this fit
separates a simple source/system-response mismatch from deeper spatial/model
mismatch.  This fit is an oracle diagnostic because it uses the truth m and
true slowness; it is not an inference-time calibration method.

Important dataset caveat
------------------------
For the L11 UltraWave dataset, ``m`` is a proxy reflectivity label formed from
a high-pass filtered log acoustic impedance and RMS-normalized.  UltraWave RF
is full-wave scattered pressure (heterogeneous total field minus a homogeneous
reference) and depends on sound speed, density and absorption.  Therefore
``m`` is not assumed to be the exact scattering potential of the full-wave
solver; this script is intended to quantify that mismatch explicitly.
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
from physics.imaging import BornModel, DelaySumBaseline  # noqa: E402
from scripts.oracle_phase_screen_decomposition import sample_ids  # noqa: E402
from scripts.pilot_phase_asp import (  # noqa: E402
    DATA_ROOT,
    corrected_config,
    embed,
    padded_meta,
)


def as_complex(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return x.to(torch.complex64)
    return torch.complex(x.to(torch.float32), torch.zeros_like(x, dtype=torch.float32))


def crop_x(x: torch.Tensor, pad: int) -> torch.Tensor:
    return x[..., pad:-pad] if pad else x


def fixed_tgc(image: torch.Tensor, z_m: np.ndarray,
              db_per_mm: float, max_db: float) -> torch.Tensor:
    gain_db = np.minimum(max_db, db_per_mm * z_m * 1e3)
    gain = torch.as_tensor(10.0 ** (gain_db / 20.0),
                           device=image.device, dtype=image.real.dtype)
    return image * gain[:, None]


def independent_db(image: torch.Tensor, db_range: float,
                   quantile: float = 0.995) -> np.ndarray:
    amp = image.abs().detach().cpu().numpy()
    positive = amp[amp > 0]
    ref = float(np.quantile(positive, quantile)) if positive.size else 1.0
    ref = max(ref, 1e-30)
    return np.clip(20.0 * np.log10(np.maximum(amp, 1e-30) / ref),
                   -db_range, 0.0)


def roi_corr(image: torch.Tensor, truth_abs: torch.Tensor,
             z_m: np.ndarray, z_min_mm: float, z_max_mm: float) -> float:
    keep = torch.as_tensor(
        (z_m >= z_min_mm * 1e-3) & (z_m < z_max_mm * 1e-3),
        device=image.device)
    if int(keep.sum()) < 2:
        return float("nan")
    return float(corr2d(image.abs()[keep], truth_abs[keep]))


def data_fit_metrics(pred: torch.Tensor, target: torch.Tensor):
    """Fit global and per-frequency complex gains pred -> target."""
    eps = torch.finfo(pred.real.dtype).eps

    global_num = (target * pred.conj()).sum()
    global_den = pred.abs().square().sum().clamp_min(eps)
    global_gain = global_num / global_den
    global_fit = pred * global_gain
    global_rel = ((target - global_fit).abs().square().sum()
                  / target.abs().square().sum().clamp_min(eps)).sqrt()

    num = (target * pred.conj()).sum(dim=(0, 1, 3))
    pred_power = pred.abs().square().sum(dim=(0, 1, 3))
    target_power = target.abs().square().sum(dim=(0, 1, 3))
    floor = pred_power.max().clamp_min(eps) * 1e-12
    gain = num / pred_power.clamp_min(floor)
    fitted = pred * gain[None, None, :, None]
    freq_rel = ((target - fitted).abs().square().sum()
                / target.abs().square().sum().clamp_min(eps)).sqrt()
    coherence = num.abs() / torch.sqrt(
        pred_power.clamp_min(floor) * target_power.clamp_min(eps))

    return {
        "global_complex_gain": [float(global_gain.real), float(global_gain.imag)],
        "global_gain_relative_residual": float(global_rel),
        "per_frequency_relative_residual": float(freq_rel),
        "mean_frequency_coherence": float(coherence.mean()),
        "median_frequency_coherence": float(coherence.median()),
    }, gain.detach(), coherence.detach()


def save_data_fit(path: Path, freqs_hz: np.ndarray,
                  gain: torch.Tensor, coherence: torch.Tensor,
                  metrics: dict, dpi: int):
    g = gain.detach().cpu().numpy()
    coh = coherence.detach().cpu().numpy()
    mag = np.abs(g)
    positive = mag[mag > 0]
    ref = float(np.median(positive)) if positive.size else 1.0
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-30) / max(ref, 1e-30))
    phase = np.unwrap(np.angle(g))
    f_mhz = np.asarray(freqs_hz) * 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    axes[0].plot(f_mhz, mag_db)
    axes[0].set(title="Best g(f): relative magnitude",
                xlabel="Frequency [MHz]", ylabel="Magnitude [dB rel. median]")
    axes[1].plot(f_mhz, phase)
    axes[1].set(title="Best g(f): unwrapped phase",
                xlabel="Frequency [MHz]", ylabel="Phase [rad]")
    axes[2].plot(f_mhz, coh)
    axes[2].set(title="Born ↔ UltraWave data coherence",
                xlabel="Frequency [MHz]", ylabel="Normalized coherence",
                ylim=(0.0, 1.02))
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Data-domain fit | global residual={:.3f}, per-f residual={:.3f}, "
        "mean coh={:.3f}".format(
            metrics["global_gain_relative_residual"],
            metrics["per_frequency_relative_residual"],
            metrics["mean_frequency_coherence"]),
        fontsize=12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_main_figure(path: Path, sample_id: str, truth_abs: torch.Tensor,
                     self_gt: torch.Tensor, wrong_speed: torch.Tensor,
                     ultrawave_gt: torch.Tensor, das: torch.Tensor,
                     cfg, meta, db_range: float, dpi: int,
                     tgc_db_per_mm: float, tgc_max_db: float,
                     metrics: dict):
    z_m = float(meta.z0) + np.arange(cfg.grid.nz) * float(cfg.grid.dz)
    x_m = float(meta.x0) + np.arange(cfg.grid.nx) * float(cfg.grid.dx)
    extent = [x_m[0] * 1e3, x_m[-1] * 1e3,
              z_m[-1] * 1e3, z_m[0] * 1e3]

    panels = [
        (truth_abs, "Truth |m|", False, None),
        (self_gt, r"$F_{GT}^H F_{GT}m$", True, "born_self"),
        (wrong_speed, r"$F_{uniform}^H F_{GT}m$", True, "born_wrong_speed"),
        (ultrawave_gt, r"$F_{GT}^H D_{UltraWave}$", True, "ultrawave_gt_speed"),
        (das, "DAS(UltraWave)", True, "das_ultrawave"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18.5, 5.0), constrained_layout=True)
    im = None
    for ax, (image, title, use_tgc, key) in zip(axes, panels):
        shown = fixed_tgc(image, z_m, tgc_db_per_mm, tgc_max_db) if use_tgc else image
        db = independent_db(shown, db_range)
        im = ax.imshow(db, cmap="gray", vmin=-db_range, vmax=0,
                       origin="upper", extent=extent, aspect="auto")
        subtitle = "independent scale"
        if key is not None:
            subtitle = "corr={:.3f}".format(metrics[key]["corr_truth_3_35mm"])
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")
    fig.colorbar(im, ax=axes, shrink=0.82, pad=0.012,
                 label="Envelope [dB, independent panel scale]")
    fig.suptitle(
        f"{sample_id}: Born/ASP closed-loop versus UltraWave | "
        f"fixed display TGC {tgc_db_per_mm:.2f} dB/mm, cap {tgc_max_db:.0f} dB",
        fontsize=13)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


@torch.no_grad()
def evaluate_sample(sample_id: str, args, device):
    sample = torch.load(DATA_ROOT / "shards" / f"{sample_id}.pt",
                        map_location="cpu", weights_only=False)
    cfg, meta = corrected_config(args.config, sample, args.imaging_n_freq)
    if args.imaging_n_freq != 0:
        bins = np.asarray(meta.band_idx)
        if len(bins) > 1 and not np.all(np.diff(bins) == 1):
            raise RuntimeError("diagnostic requires contiguous frequency bins; use --imaging-n-freq 0")

    padded = padded_meta(meta, args.pad, cfg.grid.dx)
    born = BornModel(
        padded,
        cfg.grid.nx + 2 * args.pad,
        cfg.grid.nz,
        cfg.grid.dx,
        cfg.grid.dz,
        cfg.physics.c0,
        eps=cfg.physics.eps_evanescent,
        spreading=cfg.physics.spreading,
    ).to(device)
    das_model = DelaySumBaseline(
        meta, cfg.grid.nx, cfg.grid.nz,
        cfg.grid.dx, cfg.grid.dz, cfg.physics.c0).to(device)

    rf = sample["rf"][None].to(device)
    D_uw = rf_to_D(rf, meta)
    iq = demod_iq(rf, meta)

    truth_abs = sample["m"].abs().to(device)
    m = as_complex(embed(sample["m"].to(device), args.pad))[None]
    ds_gt = embed(sample["delta_s"].to(device), args.pad)[None]
    ds_zero = torch.zeros_like(ds_gt)
    all_idx = torch.arange(len(meta.angles_deg), device=device)

    u_gt = born.transmit_fields(ds_gt, all_idx)
    D_born = born.forward(m, ds_gt, u_gt, all_idx)
    self_gt = born.adjoint(D_born, u_gt, ds_gt, all_idx)[0]

    u_zero = born.transmit_fields(ds_zero, all_idx)
    wrong_speed = born.adjoint(D_born, u_zero, ds_zero, all_idx)[0]
    ultrawave_gt = born.adjoint(D_uw, u_gt, ds_gt, all_idx)[0]
    das = das_model(iq)[0]

    self_gt = crop_x(self_gt, args.pad)
    wrong_speed = crop_x(wrong_speed, args.pad)
    ultrawave_gt = crop_x(ultrawave_gt, args.pad)

    z_m = float(meta.z0) + np.arange(cfg.grid.nz) * float(cfg.grid.dz)
    image_metrics = {
        "born_self": {
            "corr_truth_full": float(corr2d(self_gt.abs(), truth_abs)),
            "corr_truth_3_35mm": roi_corr(self_gt, truth_abs, z_m, 3.0, 35.0),
        },
        "born_wrong_speed": {
            "corr_truth_full": float(corr2d(wrong_speed.abs(), truth_abs)),
            "corr_truth_3_35mm": roi_corr(wrong_speed, truth_abs, z_m, 3.0, 35.0),
        },
        "ultrawave_gt_speed": {
            "corr_truth_full": float(corr2d(ultrawave_gt.abs(), truth_abs)),
            "corr_truth_3_35mm": roi_corr(ultrawave_gt, truth_abs, z_m, 3.0, 35.0),
        },
        "das_ultrawave": {
            "corr_truth_full": float(corr2d(das.abs(), truth_abs)),
            "corr_truth_3_35mm": roi_corr(das, truth_abs, z_m, 3.0, 35.0),
        },
    }

    fit_metrics, gain, freq_coherence = data_fit_metrics(D_born, D_uw)

    save_main_figure(
        args.out / f"{sample_id}_closed_loop.png", sample_id,
        truth_abs, self_gt, wrong_speed, ultrawave_gt, das,
        cfg, meta, args.db_range, args.dpi,
        args.tgc_db_per_mm, args.tgc_max_db, image_metrics)
    save_data_fit(
        args.out / f"{sample_id}_data_fit.png",
        np.asarray(meta.freqs), gain, freq_coherence, fit_metrics, args.dpi)

    md = sample.get("metadata", {})
    return {
        "sample": sample_id,
        "case": md.get("case"),
        "truth_label": {
            "definition": "high-pass log acoustic impedance proxy, RMS-normalized",
            "impedance_highpass_sigma_mm": md.get("impedance_highpass_sigma_mm"),
            "m_pre_normalization_rms": md.get("m_pre_normalization_rms"),
        },
        "frequency_sampling": {
            "n_freq": int(len(meta.freqs)),
            "f_min_hz": float(np.min(meta.freqs)),
            "f_max_hz": float(np.max(meta.freqs)),
            "median_df_hz": float(np.median(np.diff(meta.freqs))) if len(meta.freqs) > 1 else None,
            "contiguous_fft_bins": bool(len(meta.band_idx) < 2 or np.all(np.diff(meta.band_idx) == 1)),
        },
        "image_metrics": image_metrics,
        "data_fit": fit_metrics,
        "diagnostic_deltas": {
            "true_speed_gain_with_matched_born_data": (
                image_metrics["born_self"]["corr_truth_3_35mm"] -
                image_metrics["born_wrong_speed"]["corr_truth_3_35mm"]),
            "ultrawave_model_gap_at_true_speed": (
                image_metrics["born_self"]["corr_truth_3_35mm"] -
                image_metrics["ultrawave_gt_speed"]["corr_truth_3_35mm"]),
        },
        "figures": {
            "closed_loop": f"{sample_id}_closed_loop.png",
            "data_fit": f"{sample_id}_data_fit.png",
        },
    }


def aggregate(rows):
    keys = ("born_self", "born_wrong_speed", "ultrawave_gt_speed", "das_ultrawave")
    image = {}
    for key in keys:
        vals = np.asarray([r["image_metrics"][key]["corr_truth_3_35mm"] for r in rows], float)
        image[key] = {
            "mean_corr_truth_3_35mm": float(np.nanmean(vals)),
            "median_corr_truth_3_35mm": float(np.nanmedian(vals)),
        }
    return {
        "image": image,
        "data_fit": {
            "mean_global_gain_relative_residual": float(np.mean([
                r["data_fit"]["global_gain_relative_residual"] for r in rows])),
            "mean_per_frequency_relative_residual": float(np.mean([
                r["data_fit"]["per_frequency_relative_residual"] for r in rows])),
            "mean_frequency_coherence": float(np.mean([
                r["data_fit"]["mean_frequency_coherence"] for r in rows])),
        },
        "diagnostic_deltas": {
            "mean_true_speed_gain_with_matched_born_data": float(np.mean([
                r["diagnostic_deltas"]["true_speed_gain_with_matched_born_data"] for r in rows])),
            "mean_ultrawave_model_gap_at_true_speed": float(np.mean([
                r["diagnostic_deltas"]["ultrawave_model_gap_at_true_speed"] for r in rows])),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--sample-ids", nargs="+")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--imaging-n-freq", type=int, default=0,
                   help="0 keeps the full contiguous RF band")
    p.add_argument("--db-range", type=float, default=55.0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--tgc-db-per-mm", type=float, default=0.42)
    p.add_argument("--tgc-max-db", type=float, default=18.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.count < 1:
        p.error("--count must be >= 1")
    if args.pad < 0:
        p.error("--pad must be >= 0")
    if args.db_range <= 0 or args.tgc_max_db < 0:
        p.error("invalid display range")

    ids = args.sample_ids if args.sample_ids else sample_ids(args.split, args.count)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    rows = []
    for i, sid in enumerate(ids, 1):
        row = evaluate_sample(sid, args, device)
        rows.append(row)
        print(json.dumps({
            "event": "diagnosed",
            "sample": sid,
            "completed": i,
            "total": len(ids),
            "corr": {k: v["corr_truth_3_35mm"]
                     for k, v in row["image_metrics"].items()},
            "data_fit": row["data_fit"],
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {
        "config": args.config,
        "samples": ids,
        "display": {
            "db_range": args.db_range,
            "tgc_db_per_mm": args.tgc_db_per_mm,
            "tgc_max_db": args.tgc_max_db,
        },
        "aggregate": aggregate(rows),
        "rows": rows,
    }
    (args.out / "born_ultrawave_diagnostic.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "done", "aggregate": payload["aggregate"]}),
          flush=True)


if __name__ == "__main__":
    main()

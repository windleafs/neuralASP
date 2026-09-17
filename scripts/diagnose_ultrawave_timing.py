"""Diagnose time-zero / source-reference convention before blaming Born mismatch.

For each sample this script compares:

- DAS(Born GT-speed synthetic data): validates the DAS/time convention itself.
- DAS(UltraWave): current baseline.
- DAS(UltraWave, global dt*): one global timing correction estimated from
  normalized Born<->UltraWave cross-spectrum phase.
- DAS(UltraWave, per-angle dt*_theta): one timing correction per transmit angle.
- F_GT^H D_UltraWave before/after the same timing corrections.

The timing search is amplitude-insensitive: each frequency cross-spectrum is
normalized by Born/UltraWave power before scanning dt.  Under the repository's
D(w) ~ exp(+i w t0) convention, compensating an UltraWave delay dt uses
D_corr(w) = D_UW(w) exp(-i w dt).
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

from common import corr2d, D_to_rf, demod_iq, rf_to_D  # noqa: E402
from physics.imaging import BornModel, DelaySumBaseline  # noqa: E402
from scripts.diagnose_born_ultrawave_mismatch import (  # noqa: E402
    as_complex, crop_x, independent_db, fixed_tgc, roi_corr,
)
from scripts.oracle_phase_screen_decomposition import sample_ids  # noqa: E402
from scripts.pilot_phase_asp import DATA_ROOT, corrected_config, embed, padded_meta  # noqa: E402


def normalized_cross(pred: torch.Tensor, target: torch.Tensor):
    """Return normalized cross spectrum over elements: [theta,freq]."""
    eps = torch.finfo(pred.real.dtype).eps
    num = (target * pred.conj()).sum(dim=-1)[0]
    pp = pred.abs().square().sum(dim=-1)[0]
    tt = target.abs().square().sum(dim=-1)[0]
    return num / torch.sqrt(pp.clamp_min(eps) * tt.clamp_min(eps))


def estimate_offsets(cross: torch.Tensor, omega: torch.Tensor,
                     lo_us: float, hi_us: float, step_ns: float):
    """Grid-search global and per-angle timing offsets from normalized phase."""
    if step_ns <= 0 or hi_us <= lo_us:
        raise ValueError("invalid timing sweep")
    dt = torch.arange(lo_us * 1e-6, hi_us * 1e-6 + step_ns * 0.5e-9,
                      step_ns * 1e-9, device=cross.device, dtype=omega.dtype)
    phase = torch.exp(-1j * dt[:, None] * omega[None, :])

    # Equal-frequency weighting after cross-spectrum normalization.
    global_curve = torch.abs((phase[:, None, :] * cross[None]).sum(dim=(1, 2)))
    gi = int(global_curve.argmax())
    global_dt = dt[gi]

    # Per-angle offsets.
    curves = torch.abs((phase[:, None, :] * cross[None]).sum(dim=-1))  # [ndt,theta]
    ai = curves.argmax(dim=0)
    angle_dt = dt[ai]
    return dt, global_curve, curves, global_dt, angle_dt


def apply_time_offsets(D: torch.Tensor, omega: torch.Tensor, dt: torch.Tensor):
    """Compensate target delays under D(w)~exp(+iwt): multiply exp(-iw dt)."""
    if dt.ndim == 0:
        phase = torch.exp(-1j * omega * dt)[None, None, :, None]
    elif dt.ndim == 1:
        phase = torch.exp(-1j * dt[:, None] * omega[None, :])[None, :, :, None]
    else:
        raise ValueError("dt must be scalar or [theta]")
    return D * phase.to(D.dtype)


def das_from_D(D: torch.Tensor, meta, model: DelaySumBaseline):
    rf = D_to_rf(D, meta)
    iq = demod_iq(rf, meta)
    return model(iq)[0]


def metric(image, truth_abs, z_m):
    return {
        "corr_truth_full": float(corr2d(image.abs(), truth_abs)),
        "corr_truth_3_35mm": roi_corr(image, truth_abs, z_m, 3.0, 35.0),
    }


def save_offset_figure(path, sample_id, dt_grid, global_curve, angle_curves,
                       global_dt, angle_dt, angles, dpi):
    x = dt_grid.detach().cpu().numpy() * 1e6
    g = global_curve.detach().cpu().numpy()
    curves = angle_curves.detach().cpu().numpy()
    a_dt = angle_dt.detach().cpu().numpy() * 1e6
    angles = np.asarray(angles)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    axes[0].plot(x, g / max(g.max(), 1e-30))
    axes[0].axvline(float(global_dt) * 1e6, linestyle="--")
    axes[0].set(title=f"Global timing sweep: dt*={float(global_dt)*1e6:+.3f} us",
                xlabel="Compensated delay dt [us]", ylabel="Normalized phase score")
    for i in range(curves.shape[1]):
        axes[1].plot(x, curves[:, i] / max(curves[:, i].max(), 1e-30), alpha=.55)
    axes[1].set(title="Per-angle timing sweeps", xlabel="dt [us]", ylabel="Normalized score")
    axes[2].plot(angles, a_dt, marker="o")
    axes[2].axhline(float(global_dt) * 1e6, linestyle="--")
    axes[2].set(title="Best per-angle offsets", xlabel="Transmit angle [deg]", ylabel="dt_theta* [us]")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle(sample_id)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_image_figure(path, sample_id, truth_abs, images, metrics,
                      cfg, meta, args):
    z_m = float(meta.z0) + np.arange(cfg.grid.nz) * float(cfg.grid.dz)
    x_m = float(meta.x0) + np.arange(cfg.grid.nx) * float(cfg.grid.dx)
    extent = [x_m[0]*1e3, x_m[-1]*1e3, z_m[-1]*1e3, z_m[0]*1e3]
    names = [
        "truth", "das_born", "das_uw", "das_global", "das_angle",
        "mig_uw", "mig_global", "mig_angle",
    ]
    titles = [
        "Truth |m|", "DAS(Born GT)", "DAS(UltraWave)", "DAS(UW + global dt*)",
        "DAS(UW + per-angle dt*)", "F_GT^H D_UW", "F_GT^H D_UW(global)",
        "F_GT^H D_UW(per-angle)",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 8.3), constrained_layout=True)
    im = None
    for ax, name, title in zip(axes.flat, names, titles):
        image = truth_abs if name == "truth" else images[name]
        shown = image if name == "truth" else fixed_tgc(image, z_m, args.tgc_db_per_mm, args.tgc_max_db)
        im = ax.imshow(independent_db(shown, args.db_range), cmap="gray",
                       vmin=-args.db_range, vmax=0, origin="upper",
                       extent=extent, aspect="auto")
        subtitle = "independent scale" if name == "truth" else f"corr={metrics[name]['corr_truth_3_35mm']:.3f}"
        ax.set_title(f"{title}\n{subtitle}", fontsize=9)
        ax.set_xlabel("Lateral x [mm]"); ax.set_ylabel("Depth z [mm]")
    fig.colorbar(im, ax=axes, shrink=.82, pad=.01,
                 label="Envelope [dB, independent panel scale]")
    fig.suptitle(f"{sample_id}: timing convention diagnostic", fontsize=13)
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


@torch.no_grad()
def evaluate(sample_id, args, device):
    sample = torch.load(DATA_ROOT / "shards" / f"{sample_id}.pt",
                        map_location="cpu", weights_only=False)
    cfg, meta = corrected_config(args.config, sample, args.imaging_n_freq)
    bins = np.asarray(meta.band_idx)
    if len(bins) > 1 and not np.all(np.diff(bins) == 1):
        raise RuntimeError("use --imaging-n-freq 0 for contiguous-band timing diagnosis")

    born = BornModel(
        padded_meta(meta, args.pad, cfg.grid.dx),
        cfg.grid.nx + 2*args.pad, cfg.grid.nz, cfg.grid.dx, cfg.grid.dz,
        cfg.physics.c0, eps=cfg.physics.eps_evanescent,
        spreading=cfg.physics.spreading).to(device)
    das_model = DelaySumBaseline(meta, cfg.grid.nx, cfg.grid.nz,
                                 cfg.grid.dx, cfg.grid.dz, cfg.physics.c0).to(device)

    rf = sample["rf"][None].to(device)
    D_uw = rf_to_D(rf, meta)
    truth_abs = sample["m"].abs().to(device)
    m = as_complex(embed(sample["m"].to(device), args.pad))[None]
    ds_gt = embed(sample["delta_s"].to(device), args.pad)[None]
    all_idx = torch.arange(len(meta.angles_deg), device=device)
    u_gt = born.transmit_fields(ds_gt, all_idx)
    D_born = born.forward(m, ds_gt, u_gt, all_idx)

    cross = normalized_cross(D_born, D_uw)
    omega = born.omega_.to(cross.real.dtype)
    dt_grid, global_curve, angle_curves, global_dt, angle_dt = estimate_offsets(
        cross, omega, args.dt_min_us, args.dt_max_us, args.dt_step_ns)
    D_global = apply_time_offsets(D_uw, omega, global_dt)
    D_angle = apply_time_offsets(D_uw, omega, angle_dt)

    das_born = das_from_D(D_born, meta, das_model)
    das_uw = das_from_D(D_uw, meta, das_model)
    das_global = das_from_D(D_global, meta, das_model)
    das_angle = das_from_D(D_angle, meta, das_model)

    mig_uw = crop_x(born.adjoint(D_uw, u_gt, ds_gt, all_idx)[0], args.pad)
    mig_global = crop_x(born.adjoint(D_global, u_gt, ds_gt, all_idx)[0], args.pad)
    mig_angle = crop_x(born.adjoint(D_angle, u_gt, ds_gt, all_idx)[0], args.pad)

    z_m = float(meta.z0) + np.arange(cfg.grid.nz) * float(cfg.grid.dz)
    images = {
        "das_born": das_born,
        "das_uw": das_uw,
        "das_global": das_global,
        "das_angle": das_angle,
        "mig_uw": mig_uw,
        "mig_global": mig_global,
        "mig_angle": mig_angle,
    }
    metrics = {k: metric(v, truth_abs, z_m) for k, v in images.items()}

    save_offset_figure(args.out / f"{sample_id}_timing_sweep.png", sample_id,
                       dt_grid, global_curve, angle_curves, global_dt, angle_dt,
                       meta.angles_deg, args.dpi)
    save_image_figure(args.out / f"{sample_id}_timing_images.png", sample_id,
                      truth_abs, images, metrics, cfg, meta, args)

    return {
        "sample": sample_id,
        "case": sample["metadata"].get("case"),
        "timing": {
            "global_dt_us": float(global_dt) * 1e6,
            "per_angle_dt_us": (angle_dt.detach().cpu().numpy() * 1e6).tolist(),
            "angles_deg": np.asarray(meta.angles_deg).tolist(),
            "per_angle_dt_std_us": float(angle_dt.std()) * 1e6,
            "per_angle_dt_range_us": float(angle_dt.max() - angle_dt.min()) * 1e6,
        },
        "metrics": metrics,
        "deltas": {
            "das_global_vs_raw": metrics["das_global"]["corr_truth_3_35mm"] - metrics["das_uw"]["corr_truth_3_35mm"],
            "das_angle_vs_raw": metrics["das_angle"]["corr_truth_3_35mm"] - metrics["das_uw"]["corr_truth_3_35mm"],
            "mig_global_vs_raw": metrics["mig_global"]["corr_truth_3_35mm"] - metrics["mig_uw"]["corr_truth_3_35mm"],
            "mig_angle_vs_raw": metrics["mig_angle"]["corr_truth_3_35mm"] - metrics["mig_uw"]["corr_truth_3_35mm"],
        },
        "figures": {
            "timing_sweep": f"{sample_id}_timing_sweep.png",
            "timing_images": f"{sample_id}_timing_images.png",
        },
    }


def aggregate(rows):
    keys = ("das_born", "das_uw", "das_global", "das_angle",
            "mig_uw", "mig_global", "mig_angle")
    out = {k: float(np.mean([r["metrics"][k]["corr_truth_3_35mm"] for r in rows])) for k in keys}
    out.update({
        "mean_global_dt_us": float(np.mean([r["timing"]["global_dt_us"] for r in rows])),
        "mean_abs_global_dt_us": float(np.mean([abs(r["timing"]["global_dt_us"]) for r in rows])),
        "mean_per_angle_dt_std_us": float(np.mean([r["timing"]["per_angle_dt_std_us"] for r in rows])),
    })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--sample-ids", nargs="+")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--imaging-n-freq", type=int, default=0)
    p.add_argument("--dt-min-us", type=float, default=-1.0)
    p.add_argument("--dt-max-us", type=float, default=1.0)
    p.add_argument("--dt-step-ns", type=float, default=5.0)
    p.add_argument("--db-range", type=float, default=55.0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--tgc-db-per-mm", type=float, default=0.42)
    p.add_argument("--tgc-max-db", type=float, default=18.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    ids = args.sample_ids if args.sample_ids else sample_ids(args.split, args.count)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    rows = []
    for i, sid in enumerate(ids, 1):
        row = evaluate(sid, args, device)
        rows.append(row)
        print(json.dumps({"event": "timing_diagnosed", "sample": sid,
                          "completed": i, "total": len(ids),
                          "timing": row["timing"], "deltas": row["deltas"]}),
              flush=True)
        torch.cuda.empty_cache()

    payload = {"samples": ids, "aggregate": aggregate(rows), "rows": rows}
    (args.out / "timing_diagnostic.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "done", "aggregate": payload["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()

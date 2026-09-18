"""Diagnose whether dark regions are low-energy or coherent-cancellation artifacts.

For each requested sample and correction method (Uniform / Network / Teacher /
GT-speed), compute per-angle complex images and derive

    I_coh = |mean_theta I_theta|
    I_inc = mean_theta |I_theta|
    C_theta = |sum_theta I_theta| / (sum_theta |I_theta| + eps)

The key test is:
- low I_inc + low I_coh -> genuinely weak returned/scattered energy;
- normal I_inc + low I_coh + low C_theta -> destructive cross-angle interference.

Optionally pass --roi-mm x0 x1 z0 z1 to quantify a suspected dark region.
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

from common import demod_iq, rf_to_D
from scripts.pilot_phase_asp import DATA_ROOT, corrected_config, embed, padded_meta
from scripts.visualize_phase_screen_checkpoint import (
    load_prediction_model,
    make_teacher,
)
from physics.oversampled_imaging import LateralOversampledBornModel


def build_imaging_operator(saved_args, first_sample, pad, imaging_n_freq, device):
    cfg, meta = corrected_config(saved_args["config"], first_sample, imaging_n_freq)
    factor = int(cfg.physics.get("lateral_oversample", 1))
    born = LateralOversampledBornModel(
        padded_meta(meta, pad, cfg.grid.dx),
        cfg.grid.nx + 2 * pad,
        cfg.grid.nz,
        cfg.grid.dx,
        cfg.grid.dz,
        cfg.physics.c0,
        eps=cfg.physics.eps_evanescent,
        spreading=cfg.physics.spreading,
        lateral_oversample=factor,
    ).to(device)
    return cfg, meta, born


def angle_images(born, ds, D, idx):
    u = born.transmit_fields(ds, idx)
    return born.adjoint_per_angle(D[:, idx], u, ds)


def crop_physical(x, pad):
    return x[..., pad:-pad] if pad else x


def diagnostics(images, pad):
    """images: [B, angle, z, x]. Return cropped [z, x] maps for B=1."""
    images = crop_physical(images, pad)[0]
    coherent = images.mean(dim=0).abs()
    incoherent = images.abs().mean(dim=0)
    coherence_factor = images.sum(dim=0).abs() / images.abs().sum(dim=0).clamp_min(1e-30)
    return coherent, incoherent, coherence_factor


def db_map(x, ref, db_range):
    arr = x.detach().cpu().numpy()
    db = 20.0 * np.log10(np.maximum(arr, 1e-12) / max(ref, 1e-12))
    return np.clip(db, -db_range, 0.0)


def axis_coords(cfg, meta, born):
    x = (float(meta.x0) + np.arange(cfg.grid.nx) * float(cfg.grid.dx)) * 1e3
    z = (float(born.z0) + np.arange(born.nz) * float(born.dz)) * 1e3
    return x, z


def roi_mask(x_mm, z_mm, roi):
    if roi is None:
        return None
    x0, x1, z0, z1 = roi
    xx, zz = np.meshgrid(x_mm, z_mm)
    return (xx >= min(x0, x1)) & (xx <= max(x0, x1)) & (zz >= min(z0, z1)) & (zz <= max(z0, z1))


def summarize_roi(maps, mask, truth_abs=None):
    if mask is None:
        return None
    out = {}
    for name, (coh, inc, cf) in maps.items():
        coh_np = coh.detach().cpu().numpy()
        inc_np = inc.detach().cpu().numpy()
        cf_np = cf.detach().cpu().numpy()
        out[name] = {
            "mean_coherent_amplitude": float(coh_np[mask].mean()),
            "mean_incoherent_amplitude": float(inc_np[mask].mean()),
            "mean_angle_coherence_factor": float(cf_np[mask].mean()),
            "coherent_over_incoherent": float(
                coh_np[mask].mean() / max(inc_np[mask].mean(), 1e-30)
            ),
        }
    if truth_abs is not None:
        t = truth_abs.detach().cpu().numpy()
        out["truth"] = {
            "mean_abs_m": float(t[mask].mean()),
            "median_abs_m": float(np.median(t[mask])),
        }
    if "uniform" in out and "gt_speed_asm" in out:
        out["gt_vs_uniform"] = {
            "coherent_ratio": out["gt_speed_asm"]["mean_coherent_amplitude"] /
                              max(out["uniform"]["mean_coherent_amplitude"], 1e-30),
            "incoherent_ratio": out["gt_speed_asm"]["mean_incoherent_amplitude"] /
                                max(out["uniform"]["mean_incoherent_amplitude"], 1e-30),
            "coherence_factor_delta": out["gt_speed_asm"]["mean_angle_coherence_factor"] -
                                      out["uniform"]["mean_angle_coherence_factor"],
        }
    return out


def save_main_figure(path, sample_id, maps, x_mm, z_mm, db_range, roi, dpi):
    methods = ["uniform", "network", "teacher", "gt_speed_asm"]
    titles = ["Uniform", "Network", "Teacher", "GT-speed ASP"]
    ref = max(float(maps[m][1].max()) for m in methods)
    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]

    fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
    im_db = None
    im_cf = None
    for j, (m, title) in enumerate(zip(methods, titles)):
        coh, inc, cf = maps[m]
        im_db = axes[0, j].imshow(
            db_map(coh, ref, db_range), cmap="gray", vmin=-db_range, vmax=0,
            extent=extent, aspect="auto")
        axes[0, j].set_title(f"{title}\ncoherent")
        axes[1, j].imshow(
            db_map(inc, ref, db_range), cmap="gray", vmin=-db_range, vmax=0,
            extent=extent, aspect="auto")
        axes[1, j].set_title("incoherent")
        im_cf = axes[2, j].imshow(
            cf.detach().cpu().numpy(), cmap="viridis", vmin=0, vmax=1,
            extent=extent, aspect="auto")
        axes[2, j].set_title(r"$C_\theta$")
        for i in range(3):
            axes[i, j].set_xlabel("Lateral x [mm]")
            axes[i, j].set_ylabel("Depth z [mm]")
            if roi is not None:
                x0, x1, z0, z1 = roi
                rect = plt.Rectangle(
                    (min(x0, x1), min(z0, z1)), abs(x1 - x0), abs(z1 - z0),
                    fill=False, edgecolor="red", linewidth=1.5)
                axes[i, j].add_patch(rect)

    fig.colorbar(im_db, ax=axes[:2, :], shrink=0.72, label="Amplitude [dB], shared scale")
    fig.colorbar(im_cf, ax=axes[2, :], shrink=0.72, label="Angle coherence factor")
    fig.suptitle(f"{sample_id}: coherent-cancellation diagnostic")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_difference_figure(path, sample_id, maps, x_mm, z_mm, roi, dpi):
    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]
    u_coh, u_inc, u_cf = maps["uniform"]
    g_coh, g_inc, g_cf = maps["gt_speed_asm"]

    eps = 1e-30
    d_coh = 20 * torch.log10(g_coh.clamp_min(eps) / u_coh.clamp_min(eps))
    d_inc = 20 * torch.log10(g_inc.clamp_min(eps) / u_inc.clamp_min(eps))
    d_cf = g_cf - u_cf

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    a = axes[0].imshow(d_coh.detach().cpu().numpy(), cmap="coolwarm", vmin=-6, vmax=6,
                       extent=extent, aspect="auto")
    axes[0].set_title("GT - Uniform coherent [dB]")
    b = axes[1].imshow(d_inc.detach().cpu().numpy(), cmap="coolwarm", vmin=-6, vmax=6,
                       extent=extent, aspect="auto")
    axes[1].set_title("GT - Uniform incoherent [dB]")
    c = axes[2].imshow(d_cf.detach().cpu().numpy(), cmap="coolwarm", vmin=-0.5, vmax=0.5,
                       extent=extent, aspect="auto")
    axes[2].set_title(r"GT - Uniform $C_\theta$")
    for ax in axes:
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")
        if roi is not None:
            x0, x1, z0, z1 = roi
            ax.add_patch(plt.Rectangle(
                (min(x0, x1), min(z0, z1)), abs(x1-x0), abs(z1-z0),
                fill=False, edgecolor="black", linewidth=1.5))
    fig.colorbar(a, ax=axes[0], label="dB")
    fig.colorbar(b, ax=axes[1], label="dB")
    fig.colorbar(c, ax=axes[2], label="Δ coherence factor")
    fig.suptitle(f"{sample_id}: does GT-speed create a cancellation null?")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


@torch.no_grad()
def analyze_one(sample_id, checkpoint, imaging_n_freq, device, roi, db_range, dpi, out):
    sample = torch.load(DATA_ROOT / "shards" / f"{sample_id}.pt",
                        map_location="cpu", weights_only=False)
    ckpt, saved_args, pred_cfg, pred_meta, model = load_prediction_model(
        checkpoint, sample, device)
    imaging_cfg, imaging_meta, born = build_imaging_operator(
        saved_args, sample, model.pad, imaging_n_freq, device)

    rf = sample["rf"][None].to(device)
    iq = demod_iq(rf, pred_meta)
    pred_train_idx = torch.as_tensor(pred_meta.train_idx, device=device)
    phase_raw, mean_raw, bulk_raw = model.predict_components(iq, pred_train_idx)
    network_ds = model.network_components_to_slowness(phase_raw, mean_raw, bulk_raw)

    true_ds = embed(sample["delta_s"].to(device), model.pad)
    t_phase, t_mean, t_bulk, _, _, _ = make_teacher(model, true_ds)
    teacher_ds = model.components_to_slowness(
        t_phase[None],
        t_mean[None] if model.mean_controls else t_mean,
        t_bulk[None])

    D = rf_to_D(rf, imaging_meta)
    all_idx = torch.arange(D.shape[1], device=device)
    zero_ds = torch.zeros_like(network_ds)

    candidates = {
        "uniform": zero_ds,
        "network": network_ds,
        "teacher": teacher_ds,
        "gt_speed_asm": true_ds[None],
    }

    maps = {}
    for name, ds in candidates.items():
        imgs = angle_images(born, ds, D, all_idx)
        maps[name] = diagnostics(imgs, model.pad)

    x_mm, z_mm = axis_coords(imaging_cfg, imaging_meta, born)
    mask = roi_mask(x_mm, z_mm, roi)
    truth_abs = sample["m"].abs().to(device)
    roi_summary = summarize_roi(maps, mask, truth_abs)

    save_main_figure(
        out / f"{sample_id}_coherent_cancellation.png",
        sample_id, maps, x_mm, z_mm, db_range, roi, dpi)
    save_difference_figure(
        out / f"{sample_id}_gt_vs_uniform_cancellation.png",
        sample_id, maps, x_mm, z_mm, roi, dpi)

    return {
        "sample": sample_id,
        "screen_gate": float(model.screen_gate_value().detach().cpu()),
        "roi_mm": roi,
        "roi_summary": roi_summary,
        "figures": {
            "diagnostic": f"{sample_id}_coherent_cancellation.png",
            "gt_vs_uniform": f"{sample_id}_gt_vs_uniform_cancellation.png",
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sample-ids", nargs="+", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--imaging-n-freq", type=int, default=0)
    p.add_argument("--roi-mm", type=float, nargs=4,
                   metavar=("X0", "X1", "Z0", "Z1"),
                   help="optional suspected-dark-region ROI in mm")
    p.add_argument("--db-range", type=float, default=60.0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    rows = []
    for sid in args.sample_ids:
        row = analyze_one(
            sid, args.checkpoint, args.imaging_n_freq, device,
            args.roi_mm, args.db_range, args.dpi, args.out)
        rows.append(row)
        print(json.dumps({
            "event": "cancellation_diagnostic",
            "sample": sid,
            "screen_gate": row["screen_gate"],
            "roi_summary": row["roi_summary"],
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {
        "checkpoint": str(args.checkpoint),
        "criterion": {
            "coherent_low_and_incoherent_low": "weak returned/scattered energy",
            "coherent_low_incoherent_preserved_and_Ctheta_low": "destructive cross-angle interference",
        },
        "rows": rows,
    }
    (args.out / "coherent_cancellation_diagnostic.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "done", "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()

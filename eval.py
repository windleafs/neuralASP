"""Evaluation: metrics + key figures for the demo.

Compares on the test set:
  * ours      : full pipeline (neural operator + heterogeneous propagation)
  * uniform   : same pipeline with delta_s forced to 0 (ablation)
  * das       : classic straight-ray delay-and-sum baseline

Protocols:
  * RF residual on training angles and on held-out angles (m and eta are
    estimated from training angles only, holdout RF is *predicted*).
  * image correlation of envelopes with the ground-truth scattering map.
  * background sound-speed RMSE.

Usage:
  python eval.py --ckpt runs/demo/joint.pt --out runs/demo/eval
  python eval.py --ckpt runs/demo/joint.pt --source fullwave   # independent sim
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (build_meta, corr2d, demod_iq, get_device, load_config,
                    logcompress, relative_residual, set_seed, D_to_rf, iq_delay_sum, _to_C)
from data.fullwave import simulate_fullwave
from data.synthetic import SyntheticUSDataset
from models.pipeline import ImagingPipeline, load_pipeline_checkpoint
from physics.imaging import DelaySumBaseline
from depth_metrics import depth_metrics


def calibrate_gain(pred, target):
    """Per-frequency complex scalar gain fitted between pred and target
    (models unknown system response); returns calibrated prediction."""
    num = (target * pred.conj()).sum(dim=(0, 1, 3))  # [n_w]
    den = pred.abs().pow(2).sum(dim=(0, 1, 3)) + 1e-12
    g = (num / den)[None, None, :, None]
    return pred * g


@torch.no_grad()
def evaluate(pipe, das, batch, meta, tr_idx, ho_idx, refine_steps=0,
             refine_lr=1e-6):
    rf, D, dsv = batch["rf"], batch["D"], batch["delta_s"]
    m_true = batch.get("m", batch.get("m_ref"))
    c_true = batch["c"]
    out = {}
    # ours = neural-operator init + per-sample RF-consistency refinement of
    # delta_s on the training angles (the proposed closed loop)
    out["ours"] = pipe(rf, delta_s_true=dsv, eta_mode="net", train_idx=tr_idx,
                       pred_idx=ho_idx, return_all=True,
                       refine_steps=refine_steps, refine_lr=refine_lr)
    # ablations: no refinement / uniform background
    out["ours_noref"] = pipe(rf, delta_s_true=dsv, eta_mode="net",
                             train_idx=tr_idx, pred_idx=ho_idx,
                             return_all=False)
    out["uniform"] = pipe(rf, delta_s_true=dsv, eta_mode="zero",
                          train_idx=tr_idx, pred_idx=ho_idx, return_all=True)
    das_img = iq_delay_sum(demod_iq(rf, meta)[:, tr_idx], das.idx[tr_idx], das.apod, meta).mean(1)

    res = {}
    for tag in ("ours", "ours_noref", "uniform"):
        res[f"resid_{tag}_train"] = relative_residual(
            out[tag]["d_hat"], D[:, tr_idx]).item()
        res[f"resid_{tag}_hold"] = relative_residual(
            out[tag]["d_hat_pred"], D[:, ho_idx]).item()
    for tag in ("ours", "uniform"):
        res[f"resid_{tag}_hold_target_fit_diagnostic"] = relative_residual(
            calibrate_gain(out[tag]["d_hat_pred"], D[:, ho_idx]),
            D[:, ho_idx]).item()

    m_ref = batch.get("m_ref", m_true)
    ref_env = m_ref.abs()
    for tag, img in (("ours", out["ours"]["env"]),
                     ("uniform", out["uniform"]["env"]),
                     ("das", das_img.abs())):
        cs = [corr2d(img[b], ref_env[b]).item() for b in range(img.shape[0])]
        res[f"img_corr_{tag}"] = float(np.mean(cs))
    for tag in ("ours", "uniform"):
        res[f"m_abs_corr_{tag}"] = float(np.mean([corr2d(out[tag]["m_hat"][b].abs(), ref_env[b]).item() for b in range(len(rf))]))
        res[f"img_corr_{tag}_all_angles_diagnostic"] = float(np.mean([corr2d(out[tag]["I"][b].abs(), ref_env[b]).item() for b in range(len(rf))]))
    res["c_rmse_ours_m"] = float((out["ours"]["c_hat"] - c_true)
                                 .pow(2).mean().sqrt().item())
    res["c_rmse_uniform_m"] = float((out["uniform"]["c_hat"] - c_true)
                                    .pow(2).mean().sqrt().item())
    z_mm = (meta.z0 + torch.arange(c_true.shape[-2], device=c_true.device)
            * pipe.cfg.grid.dz) * 1e3
    depth_range = pipe.cfg.get("eval", {}).get("depth_roi_mm", [5., 40.])
    res.update(depth_metrics(out, m_ref, c_true, z_mm, depth_range))
    return res, out, das_img


def figures(out, das_img, batch, res, out_dir, meta, ho_idx, cfg):
    os.makedirs(out_dir, exist_ok=True)
    b = 0  # first test sample
    m_ref = batch.get("m_ref", batch.get("m"))
    g = batch["c"].shape[-2:]
    x0, z0 = meta.get("x0", 0.0) * 1e3, meta.get("z0", 0.0) * 1e3
    ext = [x0, x0 + g[1] * cfg.grid.dx * 1e3,
           z0 + g[0] * cfg.grid.dz * 1e3, z0]

    # --- sound speed ------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    im0 = axes[0].imshow(batch["c"][b].cpu().numpy(), cmap="turbo",
                         extent=ext, aspect="auto")
    axes[0].set_title("c true [m/s]")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(out["ours"]["c_hat"][b].cpu().numpy(), cmap="turbo",
                         extent=ext, aspect="auto")
    axes[1].set_title(f"c_hat (ours), RMSE={res['c_rmse_ours_m']:.1f} m/s")
    plt.colorbar(im1, ax=axes[1])
    err = (out["ours"]["c_hat"][b] - batch["c"][b]).cpu().numpy()
    im2 = axes[2].imshow(err, cmap="RdBu_r", vmin=-40, vmax=40,
                         extent=ext, aspect="auto")
    axes[2].set_title("c_hat error")
    plt.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_c.png", dpi=130)
    plt.close(fig)

    # --- scattering image -------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (img, title) in zip(axes, [
            (m_ref[b].abs().cpu().numpy(), "|m| true"),
            (out["ours"]["m_hat"][b].abs().cpu().numpy(), "|m_hat| (ours)"),
            (out["ours"]["I_input"][b].abs().cpu().numpy(), "|I| input-angle adjoint")]):
        im = ax.imshow(img, cmap="magma", extent=ext, aspect="auto")
        ax.set_title(title); plt.colorbar(im, ax=ax)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_m.png", dpi=130)
    plt.close(fig)

    # --- envelope comparison ----------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [(logcompress(das_img[b]), "DAS (uniform c0)", "das"),
              (logcompress(out["uniform"]["I_input"][b]), "input-angle adjoint, uniform",
               "uniform"),
              (logcompress(out["ours"]["I_input"][b]), "input-angle adjoint, ours",
               "ours")]
    for ax, (img, title, tag) in zip(axes, panels):
        ax.imshow(img.cpu().numpy(), cmap="gray", extent=ext, aspect="auto",
                  vmin=-55, vmax=0)
        ax.set_title(f"{title}\ncorr={res.get(f'img_corr_{tag}', float('nan')):.3f}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_env.png", dpi=130)
    plt.close(fig)

    # --- RF trace overlay (held-out angle) --------------------------------
    th = int(ho_idx[-1]); e = batch["rf"].shape[2] // 2
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    t = np.arange(batch["rf"].shape[-1]) / meta.fs * 1e6
    measured_band_rf = D_to_rf(batch["D"], meta)
    for ax, (pred, name) in zip(axes, [
            (out["uniform"]["rf_pred"], "uniform"),
            (out["ours"]["rf_pred"], "ours")]):
        ax.plot(t, measured_band_rf[b, th, e].cpu().numpy(), "k", lw=0.8,
                label="measured RF (same frequency band)")
        ax.plot(t, pred[b, th, e].cpu().numpy(), "r--", lw=0.8,
                label=f"predicted RF ({name})")
        ax.legend(fontsize=8); ax.set_ylabel("amplitude")
        ax.set_title(f"held-out angle #{th}, element #{e}")
    axes[1].set_xlabel("t [us]")
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_rf.png", dpi=130)
    plt.close(fig)

    # --- residual bar chart ------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    keys = ["resid_uniform_train", "resid_ours_noref_train",
            "resid_ours_train",
            "resid_uniform_hold", "resid_ours_noref_hold",
            "resid_ours_hold", "resid_ours_hold_target_fit_diagnostic"]
    vals = [res[k] for k in keys]
    ax.bar(range(len(keys)), vals, color=["C0", "C2", "C1"] * 2 + ["C3"])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("resid_", "").replace("_", "\n")
                        for k in keys], fontsize=7)
    ax.set_ylabel("relative RF residual")
    ax.set_title("RF data consistency (train angles | held-out angles)")
    fig.tight_layout(); fig.savefig(f"{out_dir}/fig_resid.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", default="auto",
                    choices=["auto", "synthetic", "fullwave", "l11_kwave", "l11_fullwave"])
    args = ap.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.config:
        cfg = load_config(args.config)
    elif "config" in checkpoint:
        cfg = _to_C(checkpoint["config"])
        # The exact calibrated response is restored from checkpoint buffers.
        cfg.physics.pop("response_path", None)
    else:
        raise ValueError("legacy checkpoint has no v2 operator/config; retrain with repaired config")
    source = cfg.get("data", {}).get("source", "synthetic") \
        if args.source == "auto" else args.source
    set_seed(cfg.seed + 7)
    device = get_device(cfg.device)
    meta = build_meta(cfg)
    tr_idx = torch.tensor(meta.train_idx, device=device)
    ho_idx = torch.tensor(meta.hold_idx, device=device)

    pipe = ImagingPipeline(cfg, meta).to(device)
    load_pipeline_checkpoint(pipe, checkpoint)
    pipe.eval()
    g = cfg.grid
    das = DelaySumBaseline(meta, g.nx, g.nz, g.dx, g.dz,
                           cfg.physics.c0).to(device)
    refine_steps = int(cfg.eval.refine_steps) if "eval" in cfg else 0
    refine_lr = float(cfg.eval.refine_lr) if "eval" in cfg else 3e-6

    out_dir = args.out or os.path.join(os.path.dirname(args.ckpt), "eval")
    os.makedirs(out_dir, exist_ok=True)

    if source == "synthetic":
        test_set = SyntheticUSDataset(cfg, "test", device)
        batches = [{"rf": test_set.rf, "D": test_set.D,
                    "delta_s": test_set.delta_s, "m": test_set.m,
                    "c": test_set.c}]
    elif source == "fullwave":
        samples = simulate_fullwave(cfg, cfg.fullwave.n_test)
        win = torch.tensor(np.asarray(meta.win)).view(1, -1, 1).to(device)
        batches = [{k: torch.stack([s[k] for s in samples]).to(device) * win
                    if k == "D" else
                    torch.stack([s[k] for s in samples]).to(device)
                    for k in ("rf", "D", "delta_s", "m_ref", "c")}]
    else:
        from data.l11_kwave import L11KWaveDataset
        test_set = L11KWaveDataset(cfg, "test", device)
        batches = [{"rf": test_set.rf, "D": test_set.D,
                    "delta_s": test_set.delta_s, "m": test_set.m,
                    "c": test_set.c}]

    cap = int(cfg.get("eval", {}).get("batch_size", cfg.train.batch_size))
    if cap < 1:
        raise ValueError("eval batch_size must be positive")
    batches = [{key: value[start:start + cap] for key, value in batch.items()}
               for batch in batches
               for start in range(0, batch["rf"].shape[0], cap)]
    all_res, first = [], None
    for bi, batch in enumerate(batches):
        res, out, das_img = evaluate(pipe, das, batch, meta, tr_idx, ho_idx,
                                     refine_steps, refine_lr)
        all_res.append(res)
        if bi == 0:
            first = (out, das_img, batch, res)
    mean_res = {k: float(np.mean([r[k] for r in all_res]))
                for k in all_res[0]}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"per_sample": all_res, "mean": mean_res}, f, indent=1)
    print(json.dumps(mean_res, indent=1))

    out, das_img, batch, res = first
    figures(out, das_img, batch, res, out_dir, meta, meta.hold_idx, cfg)

    np.savez(os.path.join(out_dir, "outputs.npz"),
             c_hat=out["ours"]["c_hat"][0].cpu().numpy(),
             c_true=batch["c"][0].cpu().numpy(),
             m_hat_real=out["ours"]["m_hat"][0].real.cpu().numpy(),
             m_hat_imag=out["ours"]["m_hat"][0].imag.cpu().numpy(),
             I_real=out["ours"]["I"][0].real.cpu().numpy(),
             I_imag=out["ours"]["I"][0].imag.cpu().numpy(),
             I_input_real=out["ours"]["I_input"][0].real.cpu().numpy(),
             I_input_imag=out["ours"]["I_input"][0].imag.cpu().numpy(),
             m_true=batch.get("m", batch.get("m_ref"))[0].cpu().numpy(),
             rf_pred=out["ours"]["rf_pred"][0].cpu().numpy(),
             rf_meas=batch["rf"][0].cpu().numpy())
    print(f"figures + outputs.npz written to {out_dir}")


if __name__ == "__main__":
    main()

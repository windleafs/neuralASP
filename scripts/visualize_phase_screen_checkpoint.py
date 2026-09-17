"""Visual evaluation for V3 dual-head phase-screen checkpoints.

For each selected sample this script writes three figures:

1. ``*_bmode.png``
   Uniform | Network | Teacher | GT-speed ASP, all with one shared dB scale.
2. ``*_components.png``
   Uniform | predicted mean-only | predicted screen-only | full Network.
3. ``*_parameters.png``
   Predicted vs teacher cumulative mean delay G(z) and every relative
   phase-screen delay curve tau_l(x).

A ``metrics.json`` file stores per-sample holdout agreement, input coherence,
image absolute correlation, and parameter errors.

Example
-------
python scripts/visualize_phase_screen_checkpoint.py \
    --checkpoint runs/phase_screen_dual_head_C96_M8/best.pt \
    --sample-ids val_013 val_041 val_049 \
    --gpu 0 \
    --db-range 60 \
    --out runs/phase_screen_dual_head_C96_M8/visual

Or choose evenly spaced samples automatically:

python scripts/visualize_phase_screen_checkpoint.py \
    --checkpoint runs/phase_screen_dual_head_C96_M8/best.pt \
    --split val --count 8 --gpu 0 \
    --out runs/phase_screen_dual_head_C96_M8/visual
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
from models.phase_screen import (  # noqa: E402
    PhaseScreenModel,
    coherence,
    heldout_agreement,
)
from physics.phase_screen import (  # noqa: E402
    mean_delay_profile_us,
    phase_control_curves_us,
    project_mean_delay_controls,
)
from scripts.oracle_phase_screen_decomposition import sample_ids  # noqa: E402
from scripts.pilot_phase_asp import (  # noqa: E402
    DATA_ROOT,
    corrected_config,
    embed,
    projected_truth_screen,
)


def load_model(checkpoint: Path, first_sample: dict, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt["args"]
    cfg, meta = corrected_config(
        saved_args["config"], first_sample, saved_args["n_freq"])
    cfg.model.normalize_iq = True
    model = PhaseScreenModel(
        cfg,
        meta,
        layers=int(saved_args.get("layers", 4)),
        controls=int(saved_args.get("controls", 24)),
        pad=int(saved_args.get("pad", 32)),
        limit_us=float(saved_args.get("limit_us", 0.2)),
        mean_controls=int(saved_args.get("mean_controls", 0)),
        mean_limit_us=float(saved_args.get("mean_limit_us", 2.0)),
        bulk_limit_us=float(saved_args.get("bulk_limit_us", 2.0)),
        fit_bulk=bool(saved_args.get("fit_bulk", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return ckpt, saved_args, cfg, meta, model


def crop_physical(image: torch.Tensor, pad: int):
    return image[..., pad:-pad] if pad else image


def compound(images: torch.Tensor, pad: int):
    """[1,angle,z,x] -> cropped [z,x] complex compound image."""
    return crop_physical(images.mean(dim=1)[0], pad)


def bmode_db(image: torch.Tensor, ref: float, db_range: float):
    amp = image.abs().detach().cpu().numpy()
    db = 20.0 * np.log10(np.maximum(amp, 1e-12) / max(ref, 1e-12))
    return np.clip(db, -db_range, 0.0)


def score_candidate(images, ref, train_idx, hold_idx, truth_abs, pad):
    train = images[:, train_idx]
    hold = images[:, hold_idx]
    coh = coherence(train, ref["mask"], ref["train_scales"])[0]
    agreement = heldout_agreement(
        train, hold, ref["mask"], ref["train_scales"],
        ref["hold_scales"])[0]
    image = compound(images, pad)
    return {
        "input_coherence": float(coh),
        "holdout_agreement": float(agreement),
        "image11_abs_corr": float(corr2d(image.abs(), truth_abs)),
    }, image


def make_teacher(model, true_ds):
    if model.mean_controls:
        phase_raw, _, phase_info = projected_truth_screen(
            true_ds,
            model.layers,
            model.controls,
            model.born.dz,
            model.limit_us,
            model.pad,
            model.born.z0,
            model.bulk_limit_us,
            False,
        )
        mean_raw, mean_target, mean_info = project_mean_delay_controls(
            true_ds,
            model.mean_controls,
            model.born.dz,
            model.pad,
            model.mean_limit_us,
        )
        bulk_raw = true_ds.new_zeros(2)
    else:
        phase_raw, bulk_raw, phase_info = projected_truth_screen(
            true_ds,
            model.layers,
            model.controls,
            model.born.dz,
            model.limit_us,
            model.pad,
            model.born.z0,
            model.bulk_limit_us,
            model.fit_bulk,
        )
        mean_raw = true_ds.new_zeros(0)
        mean_target = true_ds.new_zeros(0)
        mean_info = None
        if bulk_raw is None:
            bulk_raw = true_ds.new_zeros(2)
    return phase_raw, mean_raw, bulk_raw, phase_info, mean_info, mean_target


def axis_coordinates(cfg, meta, model):
    x0 = float(meta.x0) * 1e3
    x1 = float(meta.x0 + (cfg.grid.nx - 1) * cfg.grid.dx) * 1e3
    z0 = float(model.born.z0) * 1e3
    z1 = float(model.born.z0 + (model.born.nz - 1) * model.born.dz) * 1e3
    return x0, x1, z0, z1


def save_bmode_figure(path, sample_id, candidates, metrics,
                      cfg, meta, model, db_range, dpi):
    names = ["uniform", "network", "teacher", "gt_speed_asm"]
    titles = ["Uniform", "Network", "Teacher", "GT-speed ASP"]
    ref_amp = max(float(candidates[name].abs().max()) for name in candidates)
    x0, x1, z0, z1 = axis_coordinates(cfg, meta, model)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 5.0), constrained_layout=True)
    im = None
    for ax, name, title in zip(axes, names, titles):
        im = ax.imshow(
            bmode_db(candidates[name], ref_amp, db_range),
            cmap="gray",
            vmin=-db_range,
            vmax=0,
            origin="upper",
            extent=[x0, x1, z1, z0],
            aspect="auto",
        )
        m = metrics[name]
        ax.set_title(
            f"{title}\nhold={m['holdout_agreement']:.4f}, "
            f"corr={m['image11_abs_corr']:.4f}", fontsize=10)
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")
    fig.colorbar(im, ax=axes, shrink=0.84, pad=0.015, label="Shared amplitude [dB]")
    fig.suptitle(f"{sample_id}: full correction comparison", fontsize=13)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_component_figure(path, sample_id, candidates, metrics,
                          cfg, meta, model, db_range, dpi):
    names = ["uniform", "network_mean_only", "network_screen_only", "network"]
    titles = ["Uniform", "Predicted mean only", "Predicted screens only", "Full network"]
    ref_amp = max(float(candidates[name].abs().max()) for name in candidates)
    x0, x1, z0, z1 = axis_coordinates(cfg, meta, model)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 5.0), constrained_layout=True)
    im = None
    for ax, name, title in zip(axes, names, titles):
        im = ax.imshow(
            bmode_db(candidates[name], ref_amp, db_range),
            cmap="gray",
            vmin=-db_range,
            vmax=0,
            origin="upper",
            extent=[x0, x1, z1, z0],
            aspect="auto",
        )
        m = metrics[name]
        ax.set_title(f"{title}\nhold={m['holdout_agreement']:.4f}", fontsize=10)
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")
    fig.colorbar(im, ax=axes, shrink=0.84, pad=0.015, label="Shared amplitude [dB]")
    fig.suptitle(f"{sample_id}: network component ablation", fontsize=13)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_parameter_figure(path, sample_id, model,
                          pred_phase_raw, pred_mean_raw,
                          teacher_phase_raw, teacher_mean_raw,
                          cfg, meta, dpi):
    if not model.mean_controls:
        raise ValueError("parameter figure currently requires the V3 mean-profile head")

    pred_G = mean_delay_profile_us(
        pred_mean_raw, model.born.nz, model.mean_limit_us)[0].detach().cpu().numpy()
    teach_G = mean_delay_profile_us(
        teacher_mean_raw[None], model.born.nz,
        model.mean_limit_us)[0].detach().cpu().numpy()

    pred_tau = phase_control_curves_us(
        pred_phase_raw[0], model.born.nx, model.limit_us,
        model.pad).detach().cpu().numpy()
    teach_tau = phase_control_curves_us(
        teacher_phase_raw, model.born.nx, model.limit_us,
        model.pad).detach().cpu().numpy()

    if model.pad:
        pred_tau = pred_tau[:, model.pad:-model.pad]
        teach_tau = teach_tau[:, model.pad:-model.pad]

    z_mm = (float(model.born.z0) +
            np.arange(model.born.nz) * float(model.born.dz)) * 1e3
    x_mm = (float(meta.x0) +
            np.arange(cfg.grid.nx) * float(cfg.grid.dx)) * 1e3

    fig = plt.figure(figsize=(15.5, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, model.layers)
    ax_mean = fig.add_subplot(gs[0, :])
    ax_mean.plot(z_mm, teach_G, "--", linewidth=2, label="Teacher")
    ax_mean.plot(z_mm, pred_G, linewidth=2, label="Network")
    knot_z = np.linspace(z_mm[0], z_mm[-1], model.mean_controls + 1)[1:]
    pred_mean_us = (model.mean_limit_us * torch.tanh(pred_mean_raw[0]))\
        .detach().cpu().numpy()
    teacher_mean_us = (model.mean_limit_us * torch.tanh(teacher_mean_raw))\
        .detach().cpu().numpy()
    ax_mean.scatter(knot_z, teacher_mean_us, marker="x", s=35)
    ax_mean.scatter(knot_z, pred_mean_us, marker="o", s=24)
    ax_mean.set_title("Cumulative lateral-mean delay G(z)")
    ax_mean.set_xlabel("Depth z [mm]")
    ax_mean.set_ylabel("Delay [us]")
    ax_mean.grid(alpha=0.25)
    ax_mean.legend()

    phase_rmse = float(np.sqrt(np.mean((pred_tau - teach_tau) ** 2)))
    for layer in range(model.layers):
        ax = fig.add_subplot(gs[1, layer])
        ax.plot(x_mm, teach_tau[layer], "--", linewidth=1.8, label="Teacher")
        ax.plot(x_mm, pred_tau[layer], linewidth=1.8, label="Network")
        rmse = float(np.sqrt(np.mean(
            (pred_tau[layer] - teach_tau[layer]) ** 2)))
        ax.set_title(f"Layer {layer + 1}, RMSE={rmse:.3f} us")
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Relative delay [us]")
        ax.grid(alpha=0.25)
        if layer == 0:
            ax.legend()

    mean_rmse = float(np.sqrt(np.mean((pred_G - teach_G) ** 2)))
    fig.suptitle(
        f"{sample_id}: parameter diagnosis | mean-profile RMSE={mean_rmse:.3f} us, "
        f"phase RMSE={phase_rmse:.3f} us",
        fontsize=13,
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return mean_rmse, phase_rmse


@torch.no_grad()
def evaluate_one(sample_id, model, cfg, meta, device,
                 top_frac, db_range, dpi, out_dir):
    sample = torch.load(DATA_ROOT / "shards" / f"{sample_id}.pt",
                        map_location="cpu", weights_only=False)
    rf = sample["rf"][None].to(device)
    D = rf_to_D(rf, meta)
    iq = demod_iq(rf, meta)
    truth_abs = sample["m"].abs().to(device)
    true_ds = embed(sample["delta_s"].to(device), model.pad)

    train_idx = torch.as_tensor(meta.train_idx, device=device)
    hold_idx = torch.as_tensor(meta.hold_idx, device=device)
    all_idx = torch.arange(D.shape[1], device=device)
    ref = model.reference(D, train_idx, hold_idx, top_frac)

    pred_phase_raw, pred_mean_raw, pred_bulk_raw = model.predict_components(iq, train_idx)
    network_ds = model.components_to_slowness(
        pred_phase_raw, pred_mean_raw, pred_bulk_raw)

    teacher_phase_raw, teacher_mean_raw, teacher_bulk_raw, phase_info, mean_info, _ = (
        make_teacher(model, true_ds))
    teacher_ds = model.components_to_slowness(
        teacher_phase_raw[None],
        teacher_mean_raw[None] if model.mean_controls else teacher_mean_raw,
        teacher_bulk_raw[None],
    )

    zero_ds = torch.zeros_like(network_ds)
    gt_ds = true_ds[None]

    if model.mean_controls:
        zero_phase = torch.zeros_like(pred_phase_raw)
        zero_mean = torch.zeros_like(pred_mean_raw)
        mean_only_ds = model.components_to_slowness(
            zero_phase, pred_mean_raw, pred_bulk_raw)
        screen_only_ds = model.components_to_slowness(
            pred_phase_raw, zero_mean, pred_bulk_raw)
    else:
        mean_only_ds = zero_ds
        screen_only_ds = network_ds

    ds_candidates = {
        "uniform": zero_ds,
        "network": network_ds,
        "teacher": teacher_ds,
        "gt_speed_asm": gt_ds,
        "network_mean_only": mean_only_ds,
        "network_screen_only": screen_only_ds,
    }

    images = {
        name: model.angle_images(ds, D, all_idx)
        for name, ds in ds_candidates.items()
    }
    metrics = {}
    compound_images = {}
    for name, angle_images in images.items():
        metrics[name], compound_images[name] = score_candidate(
            angle_images, ref, train_idx, hold_idx,
            truth_abs, model.pad)

    mean_rmse = None
    phase_rmse = None
    save_bmode_figure(
        out_dir / f"{sample_id}_bmode.png",
        sample_id,
        compound_images,
        metrics,
        cfg,
        meta,
        model,
        db_range,
        dpi,
    )
    save_component_figure(
        out_dir / f"{sample_id}_components.png",
        sample_id,
        compound_images,
        metrics,
        cfg,
        meta,
        model,
        db_range,
        dpi,
    )
    if model.mean_controls:
        mean_rmse, phase_rmse = save_parameter_figure(
            out_dir / f"{sample_id}_parameters.png",
            sample_id,
            model,
            pred_phase_raw,
            pred_mean_raw,
            teacher_phase_raw,
            teacher_mean_raw,
            cfg,
            meta,
            dpi,
        )

    pred_phase_us = (model.limit_us * torch.tanh(pred_phase_raw[0]))\
        .detach().cpu().tolist()
    teacher_phase_us = (model.limit_us * torch.tanh(teacher_phase_raw))\
        .detach().cpu().tolist()
    pred_mean_us = ((model.mean_limit_us * torch.tanh(pred_mean_raw[0]))
                    .detach().cpu().tolist() if model.mean_controls else [])
    teacher_mean_us = ((model.mean_limit_us * torch.tanh(teacher_mean_raw))
                       .detach().cpu().tolist() if model.mean_controls else [])

    return {
        "sample": sample_id,
        "case": sample["metadata"].get("case"),
        "metrics": metrics,
        "parameter_error": {
            "mean_profile_rmse_us": mean_rmse,
            "phase_curve_rmse_us": phase_rmse,
        },
        "predicted": {
            "phase_controls_us": pred_phase_us,
            "mean_controls_us": pred_mean_us,
        },
        "teacher": {
            "phase_controls_us": teacher_phase_us,
            "mean_controls_us": teacher_mean_us,
            "phase_projection": phase_info,
            "mean_projection": mean_info,
        },
        "figures": {
            "bmode": f"{sample_id}_bmode.png",
            "components": f"{sample_id}_components.png",
            "parameters": (f"{sample_id}_parameters.png"
                           if model.mean_controls else None),
        },
    }


def aggregate(rows):
    methods = list(rows[0]["metrics"].keys()) if rows else []
    result = {}
    for method in methods:
        result[method] = {
            key: float(np.mean([r["metrics"][method][key] for r in rows]))
            for key in ("input_coherence", "holdout_agreement", "image11_abs_corr")
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--sample-ids", nargs="+")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--db-range", type=float, default=60.0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.db_range <= 0:
        p.error("--db-range must be positive")
    if args.dpi < 72:
        p.error("--dpi must be >= 72")

    ids = args.sample_ids if args.sample_ids else sample_ids(args.split, args.count)
    args.out.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    first = torch.load(DATA_ROOT / "shards" / f"{ids[0]}.pt",
                       map_location="cpu", weights_only=False)
    ckpt, saved_args, cfg, meta, model = load_model(
        args.checkpoint, first, device)

    rows = []
    for i, sample_id in enumerate(ids, 1):
        row = evaluate_one(
            sample_id,
            model,
            cfg,
            meta,
            device,
            args.top_frac,
            args.db_range,
            args.dpi,
            args.out,
        )
        rows.append(row)
        print(json.dumps({
            "event": "visualized",
            "sample": sample_id,
            "completed": i,
            "total": len(ids),
            "uniform_hold": row["metrics"]["uniform"]["holdout_agreement"],
            "network_hold": row["metrics"]["network"]["holdout_agreement"],
            "teacher_hold": row["metrics"]["teacher"]["holdout_agreement"],
            "gt_hold": row["metrics"]["gt_speed_asm"]["holdout_agreement"],
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {
        "checkpoint": str(args.checkpoint),
        "step": int(ckpt["step"]),
        "model": {
            "layers": model.layers,
            "controls": model.controls,
            "limit_us": model.limit_us,
            "mean_controls": model.mean_controls,
            "mean_limit_us": model.mean_limit_us,
            "fit_bulk": model.fit_bulk,
        },
        "visualization": {
            "shared_db_range": args.db_range,
            "dpi": args.dpi,
            "top_frac": args.top_frac,
        },
        "samples": ids,
        "aggregate": aggregate(rows),
        "rows": rows,
    }
    (args.out / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "event": "done",
        "out": str(args.out),
        "aggregate": payload["aggregate"],
    }), flush=True)


if __name__ == "__main__":
    main()

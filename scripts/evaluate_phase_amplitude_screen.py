"""Evaluate V5 phase + propagation-amplitude correction checkpoints.

Primary ablation:
    Uniform | Phase-only | Amplitude-only | Phase+Amplitude

The script reports holdout agreement using the same fixed Uniform reference
mask/scales used during training and produces shared-display full-band B-mode
figures plus predicted correction-parameter plots.
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
from models.phase_amplitude_screen import PhaseAmplitudeScreenModel
from models.phase_screen import heldout_agreement
from physics.amplitude_screen import amplitude_control_curves_np
from physics.amplitude_consistency import amplitude_pattern_consistency
from physics.phase_screen import phase_control_curves_us
from scripts.pilot_phase_asp import DATA_ROOT, corrected_config


def load_learned_state_preserve_physics(model, checkpoint_state):
    """Load learned tensors while keeping the eval-time physics operator.

    Training checkpoints persist frequency-dependent born.* buffers such as
    source_response. Full-band evaluation can intentionally rebuild the Born/
    ASP operator with a different number of frequencies, so those buffers must
    come from the freshly constructed eval model rather than the checkpoint.
    """
    current = model.state_dict()
    copied = []
    skipped_physics = []

    for key, value in checkpoint_state.items():
        if key.startswith("born."):
            skipped_physics.append(key)
            continue
        if key not in current:
            raise RuntimeError(
                f"checkpoint contains unexpected learned tensor: {key}")
        if tuple(current[key].shape) != tuple(value.shape):
            raise RuntimeError(
                "learned tensor shape mismatch for "
                f"{key}: checkpoint={tuple(value.shape)} "
                f"eval={tuple(current[key].shape)}")
        current[key] = value
        copied.append(key)

    missing_learned = [
        key for key in current
        if not key.startswith("born.") and key not in checkpoint_state
    ]
    if missing_learned:
        raise RuntimeError(
            "checkpoint is missing learned tensors required by eval model: "
            + ", ".join(missing_learned[:8]))

    model.load_state_dict(current, strict=True)
    return {
        "copied_tensors": len(copied),
        "skipped_physics_buffers": len(skipped_physics),
        "skipped_physics_preview": skipped_physics[:8],
    }


def build_model(checkpoint, first_sample, imaging_n_freq, device):
    ckpt = torch.load(
        checkpoint, map_location=device, weights_only=False)
    args = ckpt["args"]
    cfg, meta = corrected_config(
        args["config"], first_sample, imaging_n_freq)
    cfg.model.normalize_iq = True

    model = PhaseAmplitudeScreenModel(
        cfg, meta,
        layers=int(args.get("layers", 4)),
        controls=int(args.get("controls", 96)),
        limit_us=float(args.get("limit_us", 0.5)),
        mean_controls=int(args.get("mean_controls", 8)),
        mean_limit_us=float(args.get("mean_limit_us", 2.0)),
        screen_gate=True,
        screen_gate_init=float(args.get("screen_gate_init", 0.02)),
        amplitude_layers=int(args.get("amplitude_layers", 4)),
        amplitude_controls=int(args.get("amplitude_controls", 48)),
        amplitude_limit_np=float(args.get("amplitude_limit_np", 0.5)),
        amplitude_gate_init=float(args.get("amplitude_gate_init", 0.02)),
        amplitude_freq_power=float(args.get("amplitude_freq_power", 1.0)),
    ).to(device)

    load_report = load_learned_state_preserve_physics(
        model, ckpt["model"])
    print(json.dumps({
        "event": "checkpoint_load",
        "checkpoint": str(checkpoint),
        "checkpoint_n_freq": int(
            ckpt.get("args", {}).get("n_freq", -1)),
        "eval_n_freq": int(len(meta.freqs)),
        **load_report,
    }), flush=True)

    model.eval()
    return ckpt, args, cfg, meta, model

def fixed_tgc(nz, dz):
    z_mm = np.arange(nz) * dz * 1e3
    return np.minimum(18.0, 0.42 * z_mm)


def bmode_db(image, tgc_db, reference, db_range):
    env = np.abs(image)
    shown = env * 10.0 ** (tgc_db[:, None] / 20.0)
    db = 20.0 * np.log10(
        np.maximum(shown / max(reference, 1e-30), 1e-9))
    return np.clip(db, -db_range, 0.0)


def physical_crop(x, pad):
    return x[..., pad:-pad] if pad else x


@torch.no_grad()
def evaluate_one(sample_id, model, meta, cfg, train_idx, hold_idx,
                 db_range, dpi, out, amp_smooth, amp_eps):
    sample = torch.load(
        DATA_ROOT / "shards" / f"{sample_id}.pt",
        map_location="cpu", weights_only=False)
    device = next(model.parameters()).device
    rf = sample["rf"][None].to(device)
    iq = demod_iq(rf, meta)
    D = rf_to_D(rf, meta)

    phase_raw, mean_raw, amp_raw, _ = model.predict_all_components(
        iq, train_idx)
    ds_full, amp_full = model.network_corrections(
        phase_raw, mean_raw, amp_raw)

    zero_ds = torch.zeros_like(ds_full)
    zero_amp = torch.zeros_like(amp_full)
    ref = model.reference(D, train_idx, hold_idx, top_frac=0.2)

    methods = {
        "uniform": (zero_ds, zero_amp),
        "phase_only": (ds_full, zero_amp),
        "amplitude_only": (zero_ds, amp_full),
        "full": (ds_full, amp_full),
    }

    rows = {}
    all_images = {}
    all_idx = torch.arange(D.shape[1], device=device)
    for name, (ds, amp) in methods.items():
        tr = model.angle_images(
            ds, D, train_idx, amplitude_rate=amp)
        ho = model.angle_images(
            ds, D, hold_idx, amplitude_rate=amp)
        hold = heldout_agreement(
            tr, ho, ref["mask"],
            ref["train_scales"], ref["hold_scales"])
        amp_consistency = amplitude_pattern_consistency(
            tr, ho, ref["mask"],
            smooth_kernel=amp_smooth, eps=amp_eps)
        imgs = model.angle_images(
            ds, D, all_idx, amplitude_rate=amp)
        comp = physical_crop(imgs.mean(dim=1)[0], model.pad)
        all_images[name] = comp.detach().cpu().numpy()
        rows[name] = {
            "holdout_agreement": float(hold[0]),
            "amplitude_pattern_consistency": float(amp_consistency),
        }

    # Shared fixed TGC and shared display reference from all methods.
    tgc = fixed_tgc(cfg.grid.nz, cfg.grid.dz)
    shown_peaks = []
    for img in all_images.values():
        shown = np.abs(img) * 10.0 ** (tgc[:, None] / 20.0)
        shown_peaks.append(np.percentile(shown, 99.5))
    display_ref = max(shown_peaks)

    x = (
        float(meta.x0) + np.arange(cfg.grid.nx) * cfg.grid.dx
    ) * 1e3
    z = (
        float(meta.get("z0", 0.0))
        + np.arange(cfg.grid.nz) * cfg.grid.dz
    ) * 1e3
    extent = [x[0], x[-1], z[-1], z[0]]

    order = ["uniform", "phase_only", "amplitude_only", "full"]
    titles = ["Uniform", "Phase-only", "Amplitude-only", "Phase + Amplitude"]
    fig, axes = plt.subplots(
        1, 4, figsize=(15.8, 5.0), constrained_layout=True)
    im = None
    for ax, key, title in zip(axes, order, titles):
        db = bmode_db(
            all_images[key], tgc, display_ref, db_range)
        im = ax.imshow(
            db, cmap="gray", vmin=-db_range, vmax=0,
            extent=extent, aspect="auto")
        ax.set_title(
            f"{title}\nhold={rows[key]['holdout_agreement']:.4f} | "
            f"amp={rows[key]['amplitude_pattern_consistency']:.4f}")
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")
    fig.colorbar(im, ax=axes, shrink=0.78, label="Amplitude [dB], shared TGC")
    fig.suptitle(
        f"{sample_id}: phase-amplitude correction ablation")
    bmode_name = f"{sample_id}_phase_amplitude_ablation.png"
    fig.savefig(out / bmode_name, dpi=dpi)
    plt.close(fig)

    phase_curves = phase_control_curves_us(
        phase_raw[0], model.born.nx,
        model.limit_us, model.pad)
    phase_curves = (
        model.screen_gate_value() * phase_curves)
    amp_curves = amplitude_control_curves_np(
        amp_raw[0], model.born.nx,
        model.amplitude_limit_np, model.pad)
    amp_curves = (
        model.amplitude_gate_value() * amp_curves)
    mean_profile = model.mean_limit_us * torch.tanh(mean_raw[0])

    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    axes[0].plot(
        np.arange(len(mean_profile.detach().cpu())) + 1,
        mean_profile.detach().cpu().numpy(), marker="o")
    axes[0].set(
        title="Mean-delay controls",
        xlabel="Control index", ylabel="Delay [us]")

    xp = (
        float(model.born.x0)
        + np.arange(model.born.nx) * model.born.dx
    ) * 1e3
    for layer in range(phase_curves.shape[0]):
        axes[1].plot(
            xp, phase_curves[layer].detach().cpu().numpy(),
            label=f"L{layer+1}")
    axes[1].set(
        title=f"Effective phase screens | gate={float(model.screen_gate_value()):.3f}",
        xlabel="Lateral x [mm]", ylabel="Integrated delay [us]")
    axes[1].legend(fontsize=8)

    for layer in range(amp_curves.shape[0]):
        axes[2].plot(
            xp, amp_curves[layer].detach().cpu().numpy(),
            label=f"L{layer+1}")
    axes[2].set(
        title=f"Effective amplitude screens | gate={float(model.amplitude_gate_value()):.3f}",
        xlabel="Lateral x [mm]", ylabel="Integrated log amplitude [Np]")
    axes[2].legend(fontsize=8)
    param_name = f"{sample_id}_phase_amplitude_parameters.png"
    fig.savefig(out / param_name, dpi=dpi)
    plt.close(fig)

    return {
        "sample": sample_id,
        "screen_gate": float(model.screen_gate_value().detach().cpu()),
        "amplitude_gate": float(model.amplitude_gate_value().detach().cpu()),
        "max_effective_phase_us": float(phase_curves.abs().max()),
        "max_effective_amplitude_np": float(amp_curves.abs().max()),
        "methods": rows,
        "delta_full_vs_phase": (
            rows["full"]["holdout_agreement"]
            - rows["phase_only"]["holdout_agreement"]),
        "delta_full_vs_uniform": (
            rows["full"]["holdout_agreement"]
            - rows["uniform"]["holdout_agreement"]),
        "delta_amplitude_consistency_vs_phase": (
            rows["phase_only"]["amplitude_pattern_consistency"]
            - rows["full"]["amplitude_pattern_consistency"]),
        "figures": {
            "ablation": bmode_name,
            "parameters": param_name,
        },
    }


def aggregate(rows):
    methods = ["uniform", "phase_only", "amplitude_only", "full"]
    out = {
        key: float(np.mean([
            r["methods"][key]["holdout_agreement"] for r in rows
        ]))
        for key in methods
    }
    out["delta_full_vs_phase"] = (
        out["full"] - out["phase_only"])
    out["delta_full_vs_uniform"] = (
        out["full"] - out["uniform"])
    out["full_wins_vs_phase"] = sum(
        r["methods"]["full"]["holdout_agreement"]
        > r["methods"]["phase_only"]["holdout_agreement"]
        for r in rows)
    out["full_wins_vs_uniform"] = sum(
        r["methods"]["full"]["holdout_agreement"]
        > r["methods"]["uniform"]["holdout_agreement"]
        for r in rows)
    out["phase_only_amplitude_consistency"] = float(np.mean([
        r["methods"]["phase_only"]["amplitude_pattern_consistency"]
        for r in rows
    ]))
    out["full_amplitude_consistency"] = float(np.mean([
        r["methods"]["full"]["amplitude_pattern_consistency"]
        for r in rows
    ]))
    out["delta_amplitude_consistency_vs_phase"] = (
        out["phase_only_amplitude_consistency"]
        - out["full_amplitude_consistency"])
    out["full_amplitude_consistency_wins_vs_phase"] = sum(
        r["methods"]["full"]["amplitude_pattern_consistency"]
        < r["methods"]["phase_only"]["amplitude_pattern_consistency"]
        for r in rows)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sample-ids", nargs="+", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--imaging-n-freq", type=int, default=0)
    p.add_argument("--db-range", type=float, default=60.0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--amplitude-consistency-smooth", type=int, default=9)
    p.add_argument("--amplitude-consistency-eps", type=float, default=1e-4)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if (args.amplitude_consistency_smooth < 1
            or args.amplitude_consistency_smooth % 2 == 0):
        p.error("--amplitude-consistency-smooth must be a positive odd integer")
    if args.amplitude_consistency_eps <= 0:
        p.error("--amplitude-consistency-eps must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    first = torch.load(
        DATA_ROOT / "shards" / f"{args.sample_ids[0]}.pt",
        map_location="cpu", weights_only=False)
    ckpt, saved_args, cfg, meta, model = build_model(
        args.checkpoint, first, args.imaging_n_freq, device)

    train_idx = torch.as_tensor(meta.train_idx, device=device)
    hold_idx = torch.as_tensor(meta.hold_idx, device=device)

    rows = []
    for sid in args.sample_ids:
        row = evaluate_one(
            sid, model, meta, cfg,
            train_idx, hold_idx,
            args.db_range, args.dpi, args.out,
            args.amplitude_consistency_smooth,
            args.amplitude_consistency_eps)
        rows.append(row)
        print(json.dumps({
            "event": "phase_amplitude_eval",
            "sample": sid,
            "screen_gate": row["screen_gate"],
            "amplitude_gate": row["amplitude_gate"],
            "delta_full_vs_phase": row["delta_full_vs_phase"],
            "delta_full_vs_uniform": row["delta_full_vs_uniform"],
            "delta_amplitude_consistency_vs_phase":
                row["delta_amplitude_consistency_vs_phase"],
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {
        "checkpoint": str(args.checkpoint),
        "operator": {
            "parameter_dx_mm": float(cfg.grid.dx * 1e3),
            "lateral_oversample": int(
                cfg.physics.get("lateral_oversample", 1)),
            "propagation_dx_mm": float(
                cfg.grid.dx * 1e3 /
                int(cfg.physics.get("lateral_oversample", 1))),
            "amplitude_freq_power": model.amplitude_freq_power,
            "amplitude_f0_hz": model.amplitude_f0_hz,
            "amplitude_consistency_smooth":
                args.amplitude_consistency_smooth,
            "amplitude_consistency_eps":
                args.amplitude_consistency_eps,
        },
        "aggregate": aggregate(rows),
        "rows": rows,
    }
    (args.out / "phase_amplitude_eval.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "event": "done",
        "aggregate": payload["aggregate"],
    }), flush=True)


if __name__ == "__main__":
    main()

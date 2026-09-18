"""Per-sample amplitude-screen oracle upper-bound experiment.

This is intentionally *not* a trainable/generalizing model.  For each sample,
the phase/mean correction predicted by a frozen checkpoint is held fixed, while
one independent low-dimensional amplitude screen tensor is optimized directly
against that sample's true holdout-agreement metric.

Purpose
-------
Measure whether the current multiplicative amplitude-screen parameterization
has enough expressive power to matter at all.

If even this direct metric oracle yields only ~1e-4 holdout gain over phase-only,
then the bottleneck is the amplitude-screen parameterization / imaging model,
not the RF-to-amplitude network or its supervision.

If the oracle yields a much larger gain, the parameterization is viable and the
remaining problem is learning/supervision.

The oracle is deliberately allowed to use the holdout angles during
optimization.  Therefore it is an upper-bound diagnostic and must not be
interpreted as a generalization result.
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
from models.phase_screen import heldout_agreement
from physics.amplitude_consistency import amplitude_pattern_consistency
from physics.amplitude_screen import (
    amplitude_control_curves_np,
    controls_to_discrete_amplitude_rate,
)
from scripts.evaluate_phase_amplitude_screen import (
    DATA_ROOT,
    bmode_db,
    build_model,
    fixed_tgc,
    physical_crop,
)


def oracle_regularizer(curves_np: torch.Tensor, limit_np: float) -> torch.Tensor:
    """Very light smoothness prior; zero weight by default in the script."""
    x = curves_np / limit_np
    mag = x.square().mean()
    if x.shape[-1] >= 3:
        d2x = x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]
        lateral = d2x.square().mean()
    else:
        lateral = x.new_zeros(())
    if x.shape[-2] >= 2:
        depth = (x[..., 1:, :] - x[..., :-1, :]).square().mean()
    else:
        depth = x.new_zeros(())
    return 0.02 * mag + 0.05 * lateral + 0.01 * depth


@torch.no_grad()
def fixed_phase_correction(model, iq, train_idx):
    phase_raw, mean_raw, _, _ = model.predict_all_components(iq, train_idx)
    ds = model._components_to_slowness_scaled(
        phase_raw,
        mean_raw=mean_raw,
        screen_scale=model.screen_gate_value(),
    )
    return {
        "phase_raw": phase_raw.detach(),
        "mean_raw": mean_raw.detach(),
        "ds": ds.detach(),
    }


def make_amplitude_rate(raw_amp, model, limit_np):
    return controls_to_discrete_amplitude_rate(
        raw_amp,
        model.born.nz,
        model.born.nx,
        model.born.dz,
        limit_np,
        model.pad,
    )


def metrics(model, D, ds, amp_rate, train_idx, hold_idx, ref,
            amp_smooth, amp_eps):
    train_images = model.angle_images(
        ds, D, train_idx, amplitude_rate=amp_rate)
    hold_images = model.angle_images(
        ds, D, hold_idx, amplitude_rate=amp_rate)

    hold = heldout_agreement(
        train_images,
        hold_images,
        ref["mask"],
        ref["train_scales"],
        ref["hold_scales"],
    ).mean()

    amp_cons = amplitude_pattern_consistency(
        train_images,
        hold_images,
        ref["mask"],
        smooth_kernel=amp_smooth,
        eps=amp_eps,
    )
    return hold, amp_cons, train_images, hold_images


def save_optimization_plot(path, history, dpi):
    step = np.asarray([r["step"] for r in history])
    hold = np.asarray([r["hold"] for r in history])
    amp = np.asarray([r["amplitude_consistency"] for r in history])
    max_np = np.asarray([r["max_abs_np"] for r in history])

    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    ax.plot(step, hold, label="holdout agreement")
    ax.set_xlabel("Oracle optimization step")
    ax.set_ylabel("Holdout agreement")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(step, amp, linestyle="--", label="amplitude consistency")
    ax2.set_ylabel("Amplitude consistency (lower is better)")

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.13))
    ax3.plot(step, max_np, linestyle=":", label="max |A| [Np]")
    ax3.set_ylabel("max |A| [Np]")

    handles = (
        ax.get_lines() + ax2.get_lines() + ax3.get_lines()
    )
    ax.legend(handles, [h.get_label() for h in handles], fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_controls_plot(path, curves, x_mm, limit_np, dpi):
    fig, ax = plt.subplots(figsize=(7.8, 4.5), constrained_layout=True)
    for layer in range(curves.shape[0]):
        ax.plot(x_mm, curves[layer], label=f"L{layer+1}")
    ax.axhline(0.0, linewidth=0.8)
    ax.set(
        xlabel="Lateral x [mm]",
        ylabel="Integrated log amplitude [Np]",
        title=f"Oracle amplitude screens | bound ±{limit_np:.3f} Np",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


@torch.no_grad()
def save_bmode_comparison(path, sample_id, model, meta, cfg, D,
                          ds, baseline_amp, oracle_amp, all_idx,
                          baseline_hold, oracle_hold, db_range, dpi):
    imgs_base = model.angle_images(
        ds, D, all_idx, amplitude_rate=baseline_amp)
    imgs_oracle = model.angle_images(
        ds, D, all_idx, amplitude_rate=oracle_amp)

    comp_base = physical_crop(
        imgs_base.mean(dim=1)[0], model.pad).detach().cpu().numpy()
    comp_oracle = physical_crop(
        imgs_oracle.mean(dim=1)[0], model.pad).detach().cpu().numpy()

    tgc = fixed_tgc(cfg.grid.nz, cfg.grid.dz)
    shown_base = np.abs(comp_base) * 10.0 ** (tgc[:, None] / 20.0)
    shown_oracle = np.abs(comp_oracle) * 10.0 ** (tgc[:, None] / 20.0)
    display_ref = max(
        np.percentile(shown_base, 99.5),
        np.percentile(shown_oracle, 99.5),
    )

    x = (
        float(meta.x0) + np.arange(cfg.grid.nx) * cfg.grid.dx
    ) * 1e3
    z = (
        float(meta.get("z0", 0.0))
        + np.arange(cfg.grid.nz) * cfg.grid.dz
    ) * 1e3
    extent = [x[0], x[-1], z[-1], z[0]]

    fig, axes = plt.subplots(
        1, 2, figsize=(8.3, 5.1), constrained_layout=True)
    im = None
    for ax, image, title, hold in (
        (axes[0], comp_base, "Fixed phase-only", baseline_hold),
        (axes[1], comp_oracle, "Fixed phase + amplitude oracle", oracle_hold),
    ):
        db = bmode_db(image, tgc, display_ref, db_range)
        im = ax.imshow(
            db, cmap="gray", vmin=-db_range, vmax=0,
            extent=extent, aspect="auto")
        ax.set_title(f"{title}\nhold={hold:.5f}")
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")

    fig.colorbar(
        im, ax=axes, shrink=0.8,
        label="Amplitude [dB], shared fixed TGC")
    fig.suptitle(
        f"{sample_id}: amplitude-screen oracle upper bound")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def optimize_one(sample_id, model, meta, cfg, train_idx, hold_idx,
                 args, out_dir):
    device = next(model.parameters()).device
    sample = torch.load(
        DATA_ROOT / "shards" / f"{sample_id}.pt",
        map_location="cpu", weights_only=False)

    rf = sample["rf"][None].to(device)
    iq = demod_iq(rf, meta)
    D = rf_to_D(rf, meta)

    phase = fixed_phase_correction(model, iq, train_idx)
    ds = phase["ds"]
    ref = model.reference(
        D, train_idx, hold_idx, top_frac=args.top_frac)

    zero_amp = torch.zeros_like(ds)
    with torch.no_grad():
        base_hold_t, base_amp_t, _, _ = metrics(
            model, D, ds, zero_amp,
            train_idx, hold_idx, ref,
            args.amplitude_consistency_smooth,
            args.amplitude_consistency_eps,
        )
    baseline_hold = float(base_hold_t)
    baseline_amp = float(base_amp_t)

    raw_amp = torch.zeros(
        1,
        args.amplitude_layers,
        args.amplitude_controls,
        device=device,
        dtype=ds.dtype,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([raw_amp], lr=args.lr)

    history = [{
        "step": 0,
        "hold": baseline_hold,
        "amplitude_consistency": baseline_amp,
        "max_abs_np": 0.0,
        "regularizer": 0.0,
        "loss": -baseline_hold,
    }]

    best_hold = baseline_hold
    best_step = 0
    best_raw = raw_amp.detach().clone()
    best_amp_cons = baseline_amp

    for step in range(1, args.steps + 1):
        amp_rate = make_amplitude_rate(
            raw_amp, model, args.amplitude_limit_np)
        hold, amp_cons, _, _ = metrics(
            model, D, ds, amp_rate,
            train_idx, hold_idx, ref,
            args.amplitude_consistency_smooth,
            args.amplitude_consistency_eps,
        )
        curves = amplitude_control_curves_np(
            raw_amp,
            model.born.nx,
            args.amplitude_limit_np,
            model.pad,
        )
        reg = oracle_regularizer(
            curves, args.amplitude_limit_np)
        loss = -hold + args.reg_weight * reg

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"nonfinite oracle loss for {sample_id} at step {step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_amp], args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            current_hold = float(hold)
            current_amp = float(amp_cons)
            max_abs = float(curves.abs().max())
            row = {
                "step": step,
                "hold": current_hold,
                "amplitude_consistency": current_amp,
                "max_abs_np": max_abs,
                "regularizer": float(reg),
                "loss": float(loss),
            }
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                history.append(row)
                print(json.dumps({
                    "event": "amplitude_oracle_step",
                    "sample": sample_id,
                    **row,
                    "gain_vs_phase": current_hold - baseline_hold,
                }), flush=True)

            if current_hold > best_hold:
                best_hold = current_hold
                best_step = step
                best_amp_cons = current_amp
                best_raw = raw_amp.detach().clone()

    with torch.no_grad():
        best_rate = make_amplitude_rate(
            best_raw, model, args.amplitude_limit_np)
        best_curves_t = amplitude_control_curves_np(
            best_raw[0],
            model.born.nx,
            args.amplitude_limit_np,
            model.pad,
        )
        best_curves = best_curves_t.detach().cpu().numpy()
        _, best_amp_cons_t, _, _ = metrics(
            model, D, ds, best_rate,
            train_idx, hold_idx, ref,
            args.amplitude_consistency_smooth,
            args.amplitude_consistency_eps,
        )
        best_amp_cons = float(best_amp_cons_t)

    # Physical-domain x coordinate on the padded parameter grid.
    x_mm = (
        float(model.born.x0)
        + np.arange(model.born.nx) * model.born.dx
    ) * 1e3

    opt_fig = f"{sample_id}_amplitude_oracle_optimization.png"
    controls_fig = f"{sample_id}_amplitude_oracle_controls.png"
    bmode_fig = f"{sample_id}_amplitude_oracle_bmode.png"

    save_optimization_plot(
        out_dir / opt_fig, history, args.dpi)
    save_controls_plot(
        out_dir / controls_fig,
        best_curves, x_mm,
        args.amplitude_limit_np, args.dpi)

    all_idx = torch.arange(D.shape[1], device=device)
    save_bmode_comparison(
        out_dir / bmode_fig,
        sample_id, model, meta, cfg, D,
        ds, zero_amp, best_rate, all_idx,
        baseline_hold, best_hold,
        args.db_range, args.dpi)

    saturation_fraction = float(
        (np.abs(best_curves) >=
         0.98 * args.amplitude_limit_np).mean()
    )

    artifact = {
        "raw_amplitude": best_raw.detach().cpu(),
        "effective_curves_np": torch.from_numpy(best_curves),
        "fixed_ds": ds.detach().cpu(),
        "baseline_hold": baseline_hold,
        "best_hold": best_hold,
        "best_step": best_step,
        "amplitude_limit_np": args.amplitude_limit_np,
    }
    artifact_path = out_dir / f"{sample_id}_amplitude_oracle.pt"
    torch.save(artifact, artifact_path)

    return {
        "sample": sample_id,
        "baseline_phase_hold": baseline_hold,
        "oracle_hold": best_hold,
        "delta_hold_vs_phase": best_hold - baseline_hold,
        "baseline_amplitude_consistency": baseline_amp,
        "oracle_amplitude_consistency": best_amp_cons,
        "delta_amplitude_consistency": (
            baseline_amp - best_amp_cons),
        "best_step": best_step,
        "max_abs_amplitude_np": float(np.abs(best_curves).max()),
        "mean_abs_amplitude_np": float(np.abs(best_curves).mean()),
        "saturation_fraction": saturation_fraction,
        "amplitude_limit_np": args.amplitude_limit_np,
        "amplitude_shape": [
            args.amplitude_layers,
            args.amplitude_controls,
        ],
        "oracle_uses_holdout_metric": True,
        "figures": {
            "optimization": opt_fig,
            "controls": controls_fig,
            "bmode": bmode_fig,
        },
        "artifact": artifact_path.name,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sample-ids", nargs="+", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--n-freq", type=int, default=64,
        help="frequency count used for both oracle optimization and evaluation")
    p.add_argument("--amplitude-layers", type=int, default=4)
    p.add_argument("--amplitude-controls", type=int, default=48)
    p.add_argument("--amplitude-limit-np", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--reg-weight", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--amplitude-consistency-smooth", type=int, default=9)
    p.add_argument("--amplitude-consistency-eps", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--db-range", type=float, default=60.0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.amplitude_layers < 1 or args.amplitude_controls < 2:
        p.error("invalid amplitude control shape")
    if args.amplitude_limit_np <= 0:
        p.error("--amplitude-limit-np must be positive")
    if args.steps < 1 or args.lr <= 0:
        p.error("--steps and --lr must be positive")
    if args.reg_weight < 0:
        p.error("--reg-weight must be non-negative")
    if (args.amplitude_consistency_smooth < 1
            or args.amplitude_consistency_smooth % 2 == 0):
        p.error(
            "--amplitude-consistency-smooth must be a positive odd integer")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    first = torch.load(
        DATA_ROOT / "shards" / f"{args.sample_ids[0]}.pt",
        map_location="cpu", weights_only=False)
    ckpt, saved_args, cfg, meta, model = build_model(
        args.checkpoint, first, args.n_freq, device)

    # Freeze the entire network.  Only per-sample oracle controls are optimized.
    for p_model in model.parameters():
        p_model.requires_grad_(False)
    model.eval()

    train_idx = torch.as_tensor(
        meta.train_idx, device=device)
    hold_idx = torch.as_tensor(
        meta.hold_idx, device=device)

    print(json.dumps({
        "event": "amplitude_oracle_setup",
        "checkpoint": str(args.checkpoint),
        "sample_ids": args.sample_ids,
        "n_freq": len(meta.freqs),
        "amplitude_shape": [
            args.amplitude_layers,
            args.amplitude_controls,
        ],
        "amplitude_limit_np": args.amplitude_limit_np,
        "steps": args.steps,
        "lr": args.lr,
        "reg_weight": args.reg_weight,
        "oracle_uses_holdout_metric": True,
        "warning": (
            "This is a per-sample metric oracle / upper-bound diagnostic, "
            "not a generalization experiment."
        ),
    }), flush=True)

    rows = []
    for sid in args.sample_ids:
        row = optimize_one(
            sid, model, meta, cfg,
            train_idx, hold_idx,
            args, args.out)
        rows.append(row)
        print(json.dumps({
            "event": "amplitude_oracle_result",
            **{
                k: row[k] for k in (
                    "sample",
                    "baseline_phase_hold",
                    "oracle_hold",
                    "delta_hold_vs_phase",
                    "baseline_amplitude_consistency",
                    "oracle_amplitude_consistency",
                    "best_step",
                    "max_abs_amplitude_np",
                    "saturation_fraction",
                )
            },
        }), flush=True)
        torch.cuda.empty_cache()

    aggregate = {
        "n": len(rows),
        "mean_baseline_phase_hold": float(np.mean([
            r["baseline_phase_hold"] for r in rows])),
        "mean_oracle_hold": float(np.mean([
            r["oracle_hold"] for r in rows])),
        "mean_delta_hold_vs_phase": float(np.mean([
            r["delta_hold_vs_phase"] for r in rows])),
        "median_delta_hold_vs_phase": float(np.median([
            r["delta_hold_vs_phase"] for r in rows])),
        "oracle_wins_vs_phase": int(sum(
            r["delta_hold_vs_phase"] > 0 for r in rows)),
        "mean_delta_amplitude_consistency": float(np.mean([
            r["delta_amplitude_consistency"] for r in rows])),
        "mean_max_abs_amplitude_np": float(np.mean([
            r["max_abs_amplitude_np"] for r in rows])),
        "mean_saturation_fraction": float(np.mean([
            r["saturation_fraction"] for r in rows])),
    }

    payload = {
        "checkpoint": str(args.checkpoint),
        "experiment": "A1 fixed-phase 4x48 amplitude metric oracle",
        "oracle_uses_holdout_metric": True,
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "aggregate": aggregate,
        "rows": rows,
    }
    (args.out / "amplitude_oracle_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n")

    print(json.dumps({
        "event": "done",
        "aggregate": aggregate,
        "out": str(args.out),
    }), flush=True)


if __name__ == "__main__":
    main()

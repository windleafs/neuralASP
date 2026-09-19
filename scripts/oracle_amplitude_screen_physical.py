"""Compare metric-only and energy-constrained amplitude-screen oracles.

Two per-sample upper-bound optimizations are run with the phase/mean correction
frozen:

Oracle-H
    minimize -H_hold

Oracle-Physical
    minimize -H_hold
             + lambda_E * L_global_energy
             + lambda_z * L_depth_energy
             + lambda_s * R(A)

The energy terms preserve the phase-only image energy globally and in depth
bins, preventing the amplitude screen from improving normalized holdout
agreement simply by suppressing difficult regions.

Optimization can use a reduced frequency set for speed.  The selected 4x48
controls are then transferred unchanged to a freshly rebuilt contiguous
full-band operator (default imaging_n_freq=0) for the final comparison.

This remains an oracle experiment because the holdout angles are explicitly
used during optimization.  It measures parameterization behavior, not
generalization.
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
from physics.amplitude_screen import amplitude_control_curves_np
from scripts.evaluate_phase_amplitude_screen import (
    DATA_ROOT,
    bmode_db,
    build_model,
    fixed_tgc,
    physical_crop,
)
from scripts.oracle_amplitude_screen import (
    fixed_phase_correction,
    make_amplitude_rate,
    oracle_regularizer,
)


def all_angle_images(model, D, ds, amp_rate, train_idx, hold_idx):
    tr = model.angle_images(ds, D, train_idx, amplitude_rate=amp_rate)
    ho = model.angle_images(ds, D, hold_idx, amplitude_rate=amp_rate)
    return tr, ho


def hold_metric(train_images, hold_images, ref):
    return heldout_agreement(
        train_images,
        hold_images,
        ref["mask"],
        ref["train_scales"],
        ref["hold_scales"],
    ).mean()


def _masked_mean_envelope(images, mask):
    weight = mask.to(images.real.dtype)[:, None]
    denom = (weight.sum() * images.shape[1]).clamp_min(1.0)
    return (images.abs() * weight).sum() / denom


def _depth_bin_means(images, mask, n_bins):
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    B, A, Z, X = images.shape
    if mask.shape != (B, Z, X):
        raise ValueError("mask/image shape mismatch")

    weight = mask.to(images.real.dtype)[:, None]
    env = images.abs()
    edges = torch.linspace(
        0, Z, n_bins + 1, device=images.device
    ).round().long()

    values = []
    valid = []
    for i in range(n_bins):
        z0 = int(edges[i])
        z1 = int(edges[i + 1])
        if z1 <= z0:
            values.append(env.new_tensor(0.0))
            valid.append(False)
            continue
        w = weight[..., z0:z1, :]
        denom = (w.sum() * A)
        if float(denom.detach()) <= 0:
            values.append(env.new_tensor(0.0))
            valid.append(False)
            continue
        values.append(
            (env[..., z0:z1, :] * w).sum() / denom
        )
        valid.append(True)
    return torch.stack(values), torch.tensor(
        valid, device=images.device, dtype=torch.bool)


def energy_preservation_losses(train_images, hold_images,
                               base_train, base_hold,
                               mask, n_depth_bins,
                               eps=1e-8):
    current = torch.cat([train_images, hold_images], dim=1)
    baseline = torch.cat([base_train, base_hold], dim=1)

    cur_global = _masked_mean_envelope(current, mask)
    base_global = _masked_mean_envelope(
        baseline.detach(), mask).clamp_min(eps)
    global_log_ratio = torch.log(
        cur_global.clamp_min(eps) / base_global)
    global_loss = global_log_ratio.square()

    cur_depth, valid_cur = _depth_bin_means(
        current, mask, n_depth_bins)
    base_depth, valid_base = _depth_bin_means(
        baseline.detach(), mask, n_depth_bins)
    valid = valid_cur & valid_base & (base_depth > eps)
    if bool(valid.any()):
        depth_log_ratio = torch.log(
            cur_depth[valid].clamp_min(eps)
            / base_depth[valid].clamp_min(eps)
        )
        depth_loss = depth_log_ratio.square().mean()
    else:
        depth_loss = global_loss.new_zeros(())

    return {
        "global_loss": global_loss,
        "depth_loss": depth_loss,
        "global_ratio": cur_global / base_global,
        "depth_current": cur_depth,
        "depth_baseline": base_depth,
        "depth_valid": valid,
    }


def control_curves(raw_amp, model, limit_np):
    return amplitude_control_curves_np(
        raw_amp,
        model.born.nx,
        limit_np,
        model.pad,
    )


def optimize_variant(name, model, D, ds, train_idx, hold_idx, ref,
                     base_train, base_hold, args, physical):
    device = D.device
    raw_amp = torch.zeros(
        1,
        args.amplitude_layers,
        args.amplitude_controls,
        device=device,
        dtype=ds.dtype,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([raw_amp], lr=args.lr)

    with torch.no_grad():
        baseline_hold = float(
            hold_metric(base_train, base_hold, ref))

    history = []
    best_raw = raw_amp.detach().clone()
    best_objective = float("inf")
    best_hold = baseline_hold
    best_step = 0

    for step in range(args.steps + 1):
        amp_rate = make_amplitude_rate(
            raw_amp, model, args.amplitude_limit_np)
        tr, ho = all_angle_images(
            model, D, ds, amp_rate, train_idx, hold_idx)
        hold = hold_metric(tr, ho, ref)
        energy = energy_preservation_losses(
            tr, ho, base_train, base_hold,
            ref["mask"], args.depth_bins,
            eps=args.energy_eps,
        )
        curves = control_curves(
            raw_amp, model, args.amplitude_limit_np)
        smooth = oracle_regularizer(
            curves, args.amplitude_limit_np)

        if physical:
            objective = (
                -hold
                + args.global_energy_weight * energy["global_loss"]
                + args.depth_energy_weight * energy["depth_loss"]
                + args.smooth_reg_weight * smooth
            )
        else:
            objective = -hold + args.metric_reg_weight * smooth

        if not torch.isfinite(objective):
            raise RuntimeError(
                f"nonfinite {name} objective at step {step}")

        row = {
            "step": int(step),
            "hold": float(hold.detach()),
            "objective": float(objective.detach()),
            "global_energy_ratio": float(
                energy["global_ratio"].detach()),
            "global_energy_loss": float(
                energy["global_loss"].detach()),
            "depth_energy_loss": float(
                energy["depth_loss"].detach()),
            "smooth_regularizer": float(smooth.detach()),
            "max_abs_np": float(curves.detach().abs().max()),
        }
        if step == 0 or step % args.log_every == 0 or step == args.steps:
            history.append(row)
            print(json.dumps({
                "event": "physical_oracle_step",
                "variant": name,
                **row,
                "gain_vs_phase": row["hold"] - baseline_hold,
            }), flush=True)

        candidate_raw = raw_amp.detach().clone()
        score = float(objective.detach())
        if score < best_objective:
            best_objective = score
            best_raw = candidate_raw
            best_hold = float(hold.detach())
            best_step = step

        if step == args.steps:
            break

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            [raw_amp], args.grad_clip)
        optimizer.step()

    return {
        "name": name,
        "best_raw": best_raw,
        "best_step": best_step,
        "best_objective": best_objective,
        "best_hold_optimization_grid": best_hold,
        "baseline_hold_optimization_grid": baseline_hold,
        "history": history,
    }


@torch.no_grad()
def evaluate_controls(model, meta, cfg, sample, raw_amp,
                      train_idx, hold_idx, top_frac,
                      amplitude_limit_np):
    device = next(model.parameters()).device
    rf = sample["rf"][None].to(device)
    iq = demod_iq(rf, meta)
    D = rf_to_D(rf, meta)

    phase = fixed_phase_correction(model, iq, train_idx)
    ds = phase["ds"]
    ref = model.reference(
        D, train_idx, hold_idx, top_frac=top_frac)

    zero_amp = torch.zeros_like(ds)
    amp_rate = make_amplitude_rate(
        raw_amp.to(device), model, amplitude_limit_np)

    base_tr, base_ho = all_angle_images(
        model, D, ds, zero_amp, train_idx, hold_idx)
    cur_tr, cur_ho = all_angle_images(
        model, D, ds, amp_rate, train_idx, hold_idx)

    base_hold = float(hold_metric(base_tr, base_ho, ref))
    cur_hold = float(hold_metric(cur_tr, cur_ho, ref))

    energy = energy_preservation_losses(
        cur_tr, cur_ho, base_tr, base_ho,
        ref["mask"], n_depth_bins=8, eps=1e-8)

    all_idx = torch.arange(D.shape[1], device=device)
    base_all = model.angle_images(
        ds, D, all_idx, amplitude_rate=zero_amp)
    cur_all = model.angle_images(
        ds, D, all_idx, amplitude_rate=amp_rate)

    base_comp = physical_crop(
        base_all.mean(dim=1)[0], model.pad)
    cur_comp = physical_crop(
        cur_all.mean(dim=1)[0], model.pad)

    curves = control_curves(
        raw_amp.to(device), model, amplitude_limit_np)[0]

    return {
        "base_hold": base_hold,
        "hold": cur_hold,
        "delta_hold": cur_hold - base_hold,
        "global_energy_ratio": float(
            energy["global_ratio"].detach()),
        "depth_energy_loss": float(
            energy["depth_loss"].detach()),
        "base_comp": base_comp.detach().cpu().numpy(),
        "comp": cur_comp.detach().cpu().numpy(),
        "curves": curves.detach().cpu().numpy(),
        "depth_current": energy[
            "depth_current"].detach().cpu().numpy(),
        "depth_baseline": energy[
            "depth_baseline"].detach().cpu().numpy(),
        "depth_valid": energy[
            "depth_valid"].detach().cpu().numpy(),
    }


def save_fullband_bmode(path, sample_id, cfg, meta,
                        base, metric, physical,
                        db_range, dpi):
    tgc = fixed_tgc(cfg.grid.nz, cfg.grid.dz)
    images = [base, metric["comp"], physical["comp"]]
    shown = [
        np.abs(x) * 10.0 ** (tgc[:, None] / 20.0)
        for x in images
    ]
    ref = max(np.percentile(x, 99.5) for x in shown)

    x_mm = (
        float(meta.x0)
        + np.arange(cfg.grid.nx) * cfg.grid.dx
    ) * 1e3
    z_mm = (
        float(meta.get("z0", 0.0))
        + np.arange(cfg.grid.nz) * cfg.grid.dz
    ) * 1e3
    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]

    fig, axes = plt.subplots(
        1, 3, figsize=(12.4, 5.1), constrained_layout=True)
    panels = [
        ("Phase-only", base, metric["base_hold"]),
        ("Oracle-H", metric["comp"], metric["hold"]),
        ("Physical Oracle", physical["comp"], physical["hold"]),
    ]
    im = None
    for ax, (title, image, hold) in zip(axes, panels):
        db = bmode_db(image, tgc, ref, db_range)
        im = ax.imshow(
            db, cmap="gray", vmin=-db_range, vmax=0,
            extent=extent, aspect="auto")
        ax.set_title(f"{title}\nfull-band hold={hold:.5f}")
        ax.set_xlabel("Lateral x [mm]")
        ax.set_ylabel("Depth z [mm]")
    fig.colorbar(
        im, ax=axes, shrink=0.8,
        label="Amplitude [dB], shared full-band TGC")
    fig.suptitle(
        f"{sample_id}: metric oracle vs energy-constrained oracle")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_controls(path, x_mm, metric, physical,
                  amplitude_limit_np, dpi):
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 4.5), constrained_layout=True)
    for ax, title, result in (
        (axes[0], "Oracle-H", metric),
        (axes[1], "Physical Oracle", physical),
    ):
        for layer in range(result["curves"].shape[0]):
            ax.plot(
                x_mm, result["curves"][layer],
                label=f"L{layer+1}")
        ax.axhline(0.0, linewidth=0.8)
        ax.axhline(
            amplitude_limit_np, linewidth=0.7,
            linestyle="--")
        ax.axhline(
            -amplitude_limit_np, linewidth=0.7,
            linestyle="--")
        ax.set(
            title=title,
            xlabel="Lateral x [mm]",
            ylabel="Integrated log amplitude [Np]",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_depth_energy(path, metric, physical, dpi):
    fig, ax = plt.subplots(
        figsize=(7.4, 4.6), constrained_layout=True)

    for label, result in (
        ("Oracle-H", metric),
        ("Physical Oracle", physical),
    ):
        valid = result["depth_valid"]
        base = result["depth_baseline"][valid]
        cur = result["depth_current"][valid]
        ratio_db = 20.0 * np.log10(
            np.maximum(cur, 1e-12)
            / np.maximum(base, 1e-12)
        )
        ax.plot(
            np.arange(len(ratio_db)) + 1,
            ratio_db, marker="o", label=label)

    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.set(
        xlabel="Depth-energy bin",
        ylabel="Current / phase-only energy [dB]",
        title="Full-band depth-energy preservation",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def saturation_stats(curves, limit_np):
    abs_curves = np.abs(curves)
    return {
        "max_abs_np": float(abs_curves.max()),
        "mean_abs_np": float(abs_curves.mean()),
        "positive_fraction": float((curves > 0).mean()),
        "negative_fraction": float((curves < 0).mean()),
        "saturation_fraction": float(
            (abs_curves >= 0.98 * limit_np).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sample-ids", nargs="+", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--opt-n-freq", type=int, default=64)
    p.add_argument(
        "--eval-n-freq", type=int, default=0,
        help="0 means contiguous full band")
    p.add_argument("--amplitude-layers", type=int, default=4)
    p.add_argument("--amplitude-controls", type=int, default=48)
    p.add_argument("--amplitude-limit-np", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--global-energy-weight", type=float, default=0.5)
    p.add_argument("--depth-energy-weight", type=float, default=1.0)
    p.add_argument("--smooth-reg-weight", type=float, default=1e-3)
    p.add_argument("--metric-reg-weight", type=float, default=0.0)
    p.add_argument("--depth-bins", type=int, default=8)
    p.add_argument("--energy-eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--top-frac", type=float, default=0.2)
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
    if min(
        args.global_energy_weight,
        args.depth_energy_weight,
        args.smooth_reg_weight,
        args.metric_reg_weight,
    ) < 0:
        p.error("loss weights must be non-negative")
    if args.depth_bins < 1:
        p.error("--depth-bins must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    first = torch.load(
        DATA_ROOT / "shards" / f"{args.sample_ids[0]}.pt",
        map_location="cpu", weights_only=False)

    _, _, cfg_opt, meta_opt, model_opt = build_model(
        args.checkpoint, first, args.opt_n_freq, device)
    for parameter in model_opt.parameters():
        parameter.requires_grad_(False)
    model_opt.eval()

    _, _, cfg_eval, meta_eval, model_eval = build_model(
        args.checkpoint, first, args.eval_n_freq, device)
    for parameter in model_eval.parameters():
        parameter.requires_grad_(False)
    model_eval.eval()

    train_opt = torch.as_tensor(
        meta_opt.train_idx, device=device)
    hold_opt = torch.as_tensor(
        meta_opt.hold_idx, device=device)
    train_eval = torch.as_tensor(
        meta_eval.train_idx, device=device)
    hold_eval = torch.as_tensor(
        meta_eval.hold_idx, device=device)

    print(json.dumps({
        "event": "physical_oracle_setup",
        "checkpoint": str(args.checkpoint),
        "opt_n_freq": len(meta_opt.freqs),
        "eval_n_freq": len(meta_eval.freqs),
        "amplitude_shape": [
            args.amplitude_layers,
            args.amplitude_controls,
        ],
        "global_energy_weight": args.global_energy_weight,
        "depth_energy_weight": args.depth_energy_weight,
        "smooth_reg_weight": args.smooth_reg_weight,
        "warning": (
            "Both variants use holdout angles during per-sample optimization; "
            "they are oracle diagnostics, not generalization results."
        ),
    }), flush=True)

    rows = []
    for sample_id in args.sample_ids:
        sample = torch.load(
            DATA_ROOT / "shards" / f"{sample_id}.pt",
            map_location="cpu", weights_only=False)

        rf_opt = sample["rf"][None].to(device)
        iq_opt = demod_iq(rf_opt, meta_opt)
        D_opt = rf_to_D(rf_opt, meta_opt)
        ds_opt = fixed_phase_correction(
            model_opt, iq_opt, train_opt)["ds"]
        ref_opt = model_opt.reference(
            D_opt, train_opt, hold_opt,
            top_frac=args.top_frac)
        zero_opt = torch.zeros_like(ds_opt)
        base_tr, base_ho = all_angle_images(
            model_opt, D_opt, ds_opt, zero_opt,
            train_opt, hold_opt)

        metric_opt = optimize_variant(
            "Oracle-H",
            model_opt, D_opt, ds_opt,
            train_opt, hold_opt, ref_opt,
            base_tr, base_ho, args,
            physical=False,
        )
        physical_opt = optimize_variant(
            "Physical-Oracle",
            model_opt, D_opt, ds_opt,
            train_opt, hold_opt, ref_opt,
            base_tr, base_ho, args,
            physical=True,
        )

        metric_eval = evaluate_controls(
            model_eval, meta_eval, cfg_eval,
            sample, metric_opt["best_raw"],
            train_eval, hold_eval,
            args.top_frac,
            args.amplitude_limit_np,
        )
        physical_eval = evaluate_controls(
            model_eval, meta_eval, cfg_eval,
            sample, physical_opt["best_raw"],
            train_eval, hold_eval,
            args.top_frac,
            args.amplitude_limit_np,
        )

        metric_stats = saturation_stats(
            metric_eval["curves"],
            args.amplitude_limit_np)
        physical_stats = saturation_stats(
            physical_eval["curves"],
            args.amplitude_limit_np)

        # Both eval calls use the same phase-only baseline.
        base_comp = metric_eval["base_comp"]

        bmode_name = (
            f"{sample_id}_physical_oracle_fullband_bmode.png")
        controls_name = (
            f"{sample_id}_physical_oracle_controls.png")
        depth_name = (
            f"{sample_id}_physical_oracle_depth_energy.png")

        save_fullband_bmode(
            args.out / bmode_name,
            sample_id, cfg_eval, meta_eval,
            base_comp, metric_eval, physical_eval,
            args.db_range, args.dpi)

        x_mm = (
            float(model_eval.born.x0)
            + np.arange(model_eval.born.nx)
            * model_eval.born.dx
        ) * 1e3
        save_controls(
            args.out / controls_name,
            x_mm, metric_eval, physical_eval,
            args.amplitude_limit_np, args.dpi)
        save_depth_energy(
            args.out / depth_name,
            metric_eval, physical_eval, args.dpi)

        torch.save({
            "metric_oracle_raw": metric_opt["best_raw"].cpu(),
            "physical_oracle_raw": physical_opt["best_raw"].cpu(),
            "metric_curves_np": torch.from_numpy(
                metric_eval["curves"]),
            "physical_curves_np": torch.from_numpy(
                physical_eval["curves"]),
            "opt_n_freq": len(meta_opt.freqs),
            "eval_n_freq": len(meta_eval.freqs),
            "args": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
        }, args.out / f"{sample_id}_physical_oracle.pt")

        row = {
            "sample": sample_id,
            "fullband_phase_hold": metric_eval["base_hold"],
            "metric_oracle": {
                "optimization_grid_hold":
                    metric_opt["best_hold_optimization_grid"],
                "fullband_hold": metric_eval["hold"],
                "fullband_delta_hold_vs_phase":
                    metric_eval["delta_hold"],
                "fullband_global_energy_ratio":
                    metric_eval["global_energy_ratio"],
                "fullband_depth_energy_loss":
                    metric_eval["depth_energy_loss"],
                **metric_stats,
            },
            "physical_oracle": {
                "optimization_grid_hold":
                    physical_opt["best_hold_optimization_grid"],
                "fullband_hold": physical_eval["hold"],
                "fullband_delta_hold_vs_phase":
                    physical_eval["delta_hold"],
                "fullband_global_energy_ratio":
                    physical_eval["global_energy_ratio"],
                "fullband_depth_energy_loss":
                    physical_eval["depth_energy_loss"],
                **physical_stats,
            },
            "figures": {
                "fullband_bmode": bmode_name,
                "controls": controls_name,
                "depth_energy": depth_name,
            },
        }
        rows.append(row)

        print(json.dumps({
            "event": "physical_oracle_result",
            "sample": sample_id,
            "phase_hold_fullband": row["fullband_phase_hold"],
            "metric_oracle_delta_fullband":
                row["metric_oracle"][
                    "fullband_delta_hold_vs_phase"],
            "physical_oracle_delta_fullband":
                row["physical_oracle"][
                    "fullband_delta_hold_vs_phase"],
            "metric_global_energy_ratio":
                row["metric_oracle"][
                    "fullband_global_energy_ratio"],
            "physical_global_energy_ratio":
                row["physical_oracle"][
                    "fullband_global_energy_ratio"],
            "metric_positive_fraction":
                row["metric_oracle"]["positive_fraction"],
            "physical_positive_fraction":
                row["physical_oracle"]["positive_fraction"],
        }), flush=True)
        torch.cuda.empty_cache()

    def mean_nested(section, key):
        return float(np.mean([
            row[section][key] for row in rows
        ]))

    aggregate = {
        "n": len(rows),
        "mean_fullband_phase_hold": float(np.mean([
            row["fullband_phase_hold"] for row in rows
        ])),
        "metric_oracle": {
            "mean_delta_hold_vs_phase":
                mean_nested(
                    "metric_oracle",
                    "fullband_delta_hold_vs_phase"),
            "mean_global_energy_ratio":
                mean_nested(
                    "metric_oracle",
                    "fullband_global_energy_ratio"),
            "mean_depth_energy_loss":
                mean_nested(
                    "metric_oracle",
                    "fullband_depth_energy_loss"),
            "mean_positive_fraction":
                mean_nested(
                    "metric_oracle",
                    "positive_fraction"),
            "mean_saturation_fraction":
                mean_nested(
                    "metric_oracle",
                    "saturation_fraction"),
        },
        "physical_oracle": {
            "mean_delta_hold_vs_phase":
                mean_nested(
                    "physical_oracle",
                    "fullband_delta_hold_vs_phase"),
            "mean_global_energy_ratio":
                mean_nested(
                    "physical_oracle",
                    "fullband_global_energy_ratio"),
            "mean_depth_energy_loss":
                mean_nested(
                    "physical_oracle",
                    "fullband_depth_energy_loss"),
            "mean_positive_fraction":
                mean_nested(
                    "physical_oracle",
                    "positive_fraction"),
            "mean_saturation_fraction":
                mean_nested(
                    "physical_oracle",
                    "saturation_fraction"),
        },
    }

    payload = {
        "checkpoint": str(args.checkpoint),
        "experiment": (
            "metric-only versus energy-constrained amplitude oracle"),
        "oracle_uses_holdout_metric": True,
        "optimization_frequency_count": len(meta_opt.freqs),
        "evaluation_frequency_count": len(meta_eval.freqs),
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "aggregate": aggregate,
        "rows": rows,
    }
    (args.out / "physical_amplitude_oracle_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n")

    print(json.dumps({
        "event": "done",
        "aggregate": aggregate,
        "out": str(args.out),
    }), flush=True)


if __name__ == "__main__":
    main()

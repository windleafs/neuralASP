"""Train the V5 joint phase + propagation-amplitude correction model.

Recommended workflow
--------------------
1) Warm-start from the best gated phase checkpoint and train only the new
   amplitude branch/gate.  This is the clean amplitude ablation.
2) If validation improves, optionally run a short joint fine-tune.

Both stages use context/target angle splitting.  The correction is predicted
from context angles only and scored on disjoint target angles.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import to_plain
from models.phase_amplitude_screen import PhaseAmplitudeScreenModel
from models.phase_screen import heldout_agreement
from physics.amplitude_consistency import amplitude_pattern_consistency
from scripts.pilot_phase_asp import DATA_ROOT, corrected_config
from train_phase_screen import build_cache, sample_ids
from train_phase_screen_cross_angle import split_context_target
from train_phase_screen_staged import _warm_start_learned_weights


def plain_args(args):
    return {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }


def amplitude_regularizer(effective_curves_np, limit_np):
    """Bounded low-dimensional prior; preserves common-mode amplitude."""
    x = effective_curves_np / limit_np
    magnitude = x.square().mean()
    if x.shape[-1] >= 3:
        d2x = x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]
        lateral = d2x.square().mean()
    else:
        lateral = x.new_zeros(())
    if x.shape[-2] >= 2:
        depth = (x[..., 1:, :] - x[..., :-1, :]).square().mean()
    else:
        depth = x.new_zeros(())
    return 0.05 * magnitude + 0.05 * lateral + 0.02 * depth


def screen_regularizer(effective_phase_us, limit_us):
    x = effective_phase_us / limit_us
    magnitude = x.square().mean()
    if x.shape[-1] >= 3:
        d2 = x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]
        curvature = d2.square().mean()
    else:
        curvature = x.new_zeros(())
    return 0.05 * magnitude + 0.1 * curvature


@torch.no_grad()
def validate(model, cache, train_idx, hold_idx, amp_smooth, amp_eps):
    model.eval()
    rows = []
    for item in cache:
        out = model.forward_precomputed(item["iq"], item["D"], train_idx)
        ref = item["ref"]
        tr = out["images_input"]
        ho = model.angle_images(
            out["effective_ds"], item["D"], hold_idx,
            amplitude_rate=out["amplitude_rate"])

        hold = heldout_agreement(
            tr, ho, ref["mask"],
            ref["train_scales"], ref["hold_scales"])
        amp_cons_full = amplitude_pattern_consistency(
            tr, ho, ref["mask"],
            smooth_kernel=amp_smooth, eps=amp_eps)

        # Phase-only ablation using the exact same predicted phase/mean.
        zero_amp = torch.zeros_like(out["amplitude_rate"])
        tr_phase = model.angle_images(
            out["effective_ds"], item["D"], train_idx,
            amplitude_rate=zero_amp)
        ho_phase = model.angle_images(
            out["effective_ds"], item["D"], hold_idx,
            amplitude_rate=zero_amp)
        hold_phase = heldout_agreement(
            tr_phase, ho_phase, ref["mask"],
            ref["train_scales"], ref["hold_scales"])
        amp_cons_phase = amplitude_pattern_consistency(
            tr_phase, ho_phase, ref["mask"],
            smooth_kernel=amp_smooth, eps=amp_eps)

        # Amplitude-only residual on top of zero slowness.
        zero_ds = torch.zeros_like(out["effective_ds"])
        tr_amp = model.angle_images(
            zero_ds, item["D"], train_idx,
            amplitude_rate=out["amplitude_rate"])
        ho_amp = model.angle_images(
            zero_ds, item["D"], hold_idx,
            amplitude_rate=out["amplitude_rate"])
        hold_amp = heldout_agreement(
            tr_amp, ho_amp, ref["mask"],
            ref["train_scales"], ref["hold_scales"])
        amp_cons_amp = amplitude_pattern_consistency(
            tr_amp, ho_amp, ref["mask"],
            smooth_kernel=amp_smooth, eps=amp_eps)

        rows.append({
            "sample": item["id"],
            "uniform_hold": float(ref["uniform_holdout_agreement"][0]),
            "phase_only_hold": float(hold_phase[0]),
            "amplitude_only_hold": float(hold_amp[0]),
            "full_hold": float(hold[0]),
            "phase_only_amplitude_consistency": float(amp_cons_phase),
            "amplitude_only_amplitude_consistency": float(amp_cons_amp),
            "full_amplitude_consistency": float(amp_cons_full),
            "screen_gate": float(out["screen_gate"]),
            "amplitude_gate": float(out["amplitude_gate"]),
            "max_effective_phase_us": float(
                out["effective_phase_controls_us"].abs().max()),
            "max_effective_amplitude_np": float(
                out["effective_amplitude_controls_np"].abs().max()),
        })

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    return {
        "mean_uniform_hold": mean("uniform_hold"),
        "mean_phase_only_hold": mean("phase_only_hold"),
        "mean_amplitude_only_hold": mean("amplitude_only_hold"),
        "mean_full_hold": mean("full_hold"),
        "delta_phase_vs_uniform": (
            mean("phase_only_hold") - mean("uniform_hold")),
        "delta_amplitude_vs_uniform": (
            mean("amplitude_only_hold") - mean("uniform_hold")),
        "delta_full_vs_phase": (
            mean("full_hold") - mean("phase_only_hold")),
        "delta_full_vs_uniform": (
            mean("full_hold") - mean("uniform_hold")),
        "mean_phase_only_amplitude_consistency": mean(
            "phase_only_amplitude_consistency"),
        "mean_amplitude_only_amplitude_consistency": mean(
            "amplitude_only_amplitude_consistency"),
        "mean_full_amplitude_consistency": mean(
            "full_amplitude_consistency"),
        "delta_amplitude_consistency_vs_phase": (
            mean("phase_only_amplitude_consistency")
            - mean("full_amplitude_consistency")),
        "full_amplitude_consistency_wins_vs_phase": sum(
            r["full_amplitude_consistency"]
            < r["phase_only_amplitude_consistency"]
            for r in rows),
        "full_hold_wins_vs_phase": sum(
            r["full_hold"] > r["phase_only_hold"] for r in rows),
        "full_hold_wins_vs_uniform": sum(
            r["full_hold"] > r["uniform_hold"] for r in rows),
        "screen_gate": float(model.screen_gate_value().detach().cpu()),
        "amplitude_gate": float(model.amplitude_gate_value().detach().cpu()),
        "rows": rows,
    }


def set_trainable_stage(model, stage):
    for p in model.parameters():
        p.requires_grad_(False)

    if stage == "amplitude":
        for p in model.amplitude_head.parameters():
            p.requires_grad_(True)
        model.amplitude_gate_logit.requires_grad_(True)
    elif stage == "joint":
        for name, p in model.named_parameters():
            if name.startswith("born."):
                continue
            if name.startswith("backbone.fno.head"):
                continue
            p.requires_grad_(True)
    else:
        raise ValueError(stage)


def checkpoint_payload(model, optimizer, step, cfg, args, report,
                       history, best_score):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": to_plain(cfg),
        "args": plain_args(args),
        "validation": report,
        "history": history,
        "best_score": float(best_score),
        "training_format": "phase_amplitude_complex_screen_v1",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--train-per-case", type=int, default=200)
    p.add_argument("--val-per-case", type=int, default=25)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--controls", type=int, default=96)
    p.add_argument("--limit-us", type=float, default=0.5)
    p.add_argument("--mean-controls", type=int, default=8)
    p.add_argument("--mean-limit-us", type=float, default=2.0)
    p.add_argument("--screen-gate-init", type=float, default=0.02)

    p.add_argument("--amplitude-layers", type=int, default=4)
    p.add_argument("--amplitude-controls", type=int, default=48)
    p.add_argument("--amplitude-limit-np", type=float, default=0.5)
    p.add_argument("--amplitude-gate-init", type=float, default=0.02)
    p.add_argument("--amplitude-freq-power", type=float, default=1.0)

    p.add_argument("--stage", choices=("amplitude", "joint"),
                   default="amplitude")
    p.add_argument("--min-context-angles", type=int, default=3)
    p.add_argument("--target-angles", type=int, default=2)
    p.add_argument("--phase-agreement-weight", type=float, default=1.0)
    p.add_argument("--amplitude-consistency-weight", type=float, default=0.25)
    p.add_argument("--amplitude-consistency-smooth", type=int, default=9)
    p.add_argument("--amplitude-consistency-eps", type=float, default=1e-4)
    p.add_argument("--amplitude-reg", type=float, default=0.02)
    p.add_argument("--screen-reg", type=float, default=0.01)
    p.add_argument("--amplitude-gate-reg", type=float, default=1e-3)
    p.add_argument("--screen-gate-reg", type=float, default=1e-3)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    args = p.parse_args()

    if args.steps < 1 or args.val_every < 1:
        p.error("steps and val-every must be positive")
    if args.mean_controls < 1:
        p.error("mean-controls must be positive")
    if args.amplitude_freq_power < 0:
        p.error("amplitude-freq-power must be non-negative")
    if not (0 < args.amplitude_gate_init < 1):
        p.error("amplitude-gate-init must lie in (0,1)")
    if args.phase_agreement_weight < 0 or args.amplitude_consistency_weight < 0:
        p.error("objective weights must be non-negative")
    if (args.amplitude_consistency_smooth < 1
            or args.amplitude_consistency_smooth % 2 == 0):
        p.error("amplitude-consistency-smooth must be a positive odd integer")
    if args.amplitude_consistency_eps <= 0:
        p.error("amplitude-consistency-eps must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    first = torch.load(
        DATA_ROOT / "shards" / "train_000.pt",
        map_location="cpu", weights_only=False)
    cfg, meta = corrected_config(args.config, first, args.n_freq)
    cfg.model.normalize_iq = True

    model = PhaseAmplitudeScreenModel(
        cfg, meta,
        layers=args.layers,
        controls=args.controls,
        limit_us=args.limit_us,
        mean_controls=args.mean_controls,
        mean_limit_us=args.mean_limit_us,
        screen_gate=True,
        screen_gate_init=args.screen_gate_init,
        amplitude_layers=args.amplitude_layers,
        amplitude_controls=args.amplitude_controls,
        amplitude_limit_np=args.amplitude_limit_np,
        amplitude_gate_init=args.amplitude_gate_init,
        amplitude_freq_power=args.amplitude_freq_power,
    ).to(device)

    _warm_start_learned_weights(
        model, args.init_checkpoint, device)
    set_trainable_stage(model, args.stage)

    train_idx = torch.as_tensor(meta.train_idx, device=device)
    hold_idx = torch.as_tensor(meta.hold_idx, device=device)
    if args.min_context_angles + args.target_angles > len(train_idx):
        p.error("context + target exceeds available train angles")

    train_ids = sample_ids("train", args.train_per_case)
    val_ids = sample_ids("val", args.val_per_case)

    print(json.dumps({
        "event": "setup",
        "stage": args.stage,
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "n_freq": len(meta.freqs),
        "parameter_dx_mm": float(cfg.grid.dx * 1e3),
        "propagation_dx_mm": float(
            cfg.grid.dx * 1e3 /
            int(cfg.physics.get("lateral_oversample", 1))),
        "phase_controls": [args.layers, args.controls],
        "amplitude_controls": [
            args.amplitude_layers, args.amplitude_controls],
        "amplitude_limit_np": args.amplitude_limit_np,
        "amplitude_freq_power": args.amplitude_freq_power,
        "phase_agreement_weight": args.phase_agreement_weight,
        "amplitude_consistency_weight": args.amplitude_consistency_weight,
        "amplitude_consistency_smooth": args.amplitude_consistency_smooth,
        "amplitude_consistency_eps": args.amplitude_consistency_eps,
        "initial_screen_gate": float(model.screen_gate_value().detach()),
        "initial_amplitude_gate": float(model.amplitude_gate_value().detach()),
    }), flush=True)

    train_cache = build_cache(
        train_ids, model, meta, train_idx, hold_idx,
        device, args.top_frac, need_reference=True)
    val_cache = build_cache(
        val_ids, model, meta, train_idx, hold_idx,
        device, args.top_frac, need_reference=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters")
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    history = []
    best_score = float("-inf")
    best_amplitude_score = float("-inf")
    started = time.monotonic()

    def check(step):
        nonlocal best_score, best_amplitude_score
        report = validate(
            model, val_cache, train_idx, hold_idx,
            args.amplitude_consistency_smooth,
            args.amplitude_consistency_eps)
        report["step"] = int(step)
        report["elapsed_s"] = time.monotonic() - started
        history.append(report)

        # Select explicitly by gain over the warm-start phase-only model.
        selection = report["delta_full_vs_phase"]
        amplitude_selection = report[
            "delta_amplitude_consistency_vs_phase"]
        improved = selection > best_score
        amplitude_improved = amplitude_selection > best_amplitude_score
        if improved:
            best_score = float(selection)
        if amplitude_improved:
            best_amplitude_score = float(amplitude_selection)

        payload = checkpoint_payload(
            model, optimizer, step, cfg, args,
            report, history, best_score)
        payload["best_amplitude_score"] = float(best_amplitude_score)
        torch.save(payload, args.out / "last.pt")
        if improved:
            torch.save(payload, args.out / "best.pt")
        if amplitude_improved:
            torch.save(payload, args.out / "best_amplitude.pt")
        (args.out / "history.json").write_text(
            json.dumps(history, indent=2) + "\n")

        print(json.dumps({
            "event": "validation",
            "step": int(step),
            "selection_delta_full_vs_phase": selection,
            "best_score": best_score,
            "improved": improved,
            "amplitude_selection_gain": amplitude_selection,
            "best_amplitude_score": best_amplitude_score,
            "amplitude_improved": amplitude_improved,
            "mean_uniform_hold": report["mean_uniform_hold"],
            "mean_phase_only_hold": report["mean_phase_only_hold"],
            "mean_full_hold": report["mean_full_hold"],
            "delta_full_vs_phase": report["delta_full_vs_phase"],
            "full_hold_wins_vs_phase": report["full_hold_wins_vs_phase"],
            "phase_only_amplitude_consistency":
                report["mean_phase_only_amplitude_consistency"],
            "full_amplitude_consistency":
                report["mean_full_amplitude_consistency"],
            "delta_amplitude_consistency_vs_phase":
                report["delta_amplitude_consistency_vs_phase"],
            "full_amplitude_consistency_wins_vs_phase":
                report["full_amplitude_consistency_wins_vs_phase"],
            "screen_gate": report["screen_gate"],
            "amplitude_gate": report["amplitude_gate"],
        }), flush=True)

    check(0)

    for step in range(1, args.steps + 1):
        model.train()
        item = train_cache[
            torch.randint(len(train_cache), ()).item()]

        ctx_pos, tgt_pos, ctx_idx, tgt_idx = split_context_target(
            train_idx, args.min_context_angles, args.target_angles)

        phase_raw, mean_raw, amp_raw, _ = (
            model.predict_all_components(item["iq"], ctx_idx))
        ds, amp_rate = model.network_corrections(
            phase_raw, mean_raw, amp_raw)

        ctx_img = model.angle_images(
            ds, item["D"], ctx_idx, amplitude_rate=amp_rate)
        tgt_img = model.angle_images(
            ds, item["D"], tgt_idx, amplitude_rate=amp_rate)

        ref = item["ref"]
        cross = heldout_agreement(
            ctx_img, tgt_img, ref["mask"],
            ref["train_scales"][:, ctx_pos],
            ref["train_scales"][:, tgt_pos],
        ).mean()
        amp_consistency = amplitude_pattern_consistency(
            ctx_img, tgt_img, ref["mask"],
            smooth_kernel=args.amplitude_consistency_smooth,
            eps=args.amplitude_consistency_eps)

        phase_eff = (
            model.screen_gate_value()
            * model.limit_us * torch.tanh(phase_raw))
        amp_curves = (
            model.amplitude_gate_value()
            * model.amplitude_limit_np * torch.tanh(amp_raw))

        amp_prior = amplitude_regularizer(
            amp_curves, model.amplitude_limit_np)
        phase_prior = screen_regularizer(
            phase_eff, model.limit_us)

        loss = (
            -args.phase_agreement_weight * cross
            + args.amplitude_consistency_weight * amp_consistency
            + args.amplitude_reg * amp_prior
            + args.screen_reg * phase_prior
            + args.amplitude_gate_reg
              * model.amplitude_gate_value().square()
            + args.screen_gate_reg
              * model.screen_gate_value().square()
        )
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(json.dumps({
                "event": "train",
                "stage": args.stage,
                "step": int(step),
                "sample": item["id"],
                "cross_angle_agreement": float(cross.detach()),
                "amplitude_pattern_consistency": float(
                    amp_consistency.detach()),
                "screen_gate": float(
                    model.screen_gate_value().detach()),
                "amplitude_gate": float(
                    model.amplitude_gate_value().detach()),
                "amplitude_prior": float(amp_prior.detach()),
                "phase_prior": float(phase_prior.detach()),
                "loss": float(loss.detach()),
                "elapsed_s": time.monotonic() - started,
            }), flush=True)

        if step % args.val_every == 0 or step == args.steps:
            check(step)

    print(json.dumps({
        "event": "done",
        "stage": args.stage,
        "best_delta_full_vs_phase": best_score,
        "best_delta_amplitude_consistency_vs_phase":
            best_amplitude_score,
        "final_screen_gate": float(
            model.screen_gate_value().detach().cpu()),
        "final_amplitude_gate": float(
            model.amplitude_gate_value().detach().cpu()),
        "elapsed_s": time.monotonic() - started,
    }), flush=True)


if __name__ == "__main__":
    main()

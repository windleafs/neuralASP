"""Mean-first + gated relative-screen training with cross-angle self-validation.

Stage A (recommended): train only the mean-delay branch against its teacher.
Stage B: warm-start Stage A, enable a learnable relative-screen gate initialized
near zero, and optimize held-out cross-angle agreement on the 0.1 mm ASP chain.

The important distinction from the legacy hybrid objective is that the
correction is predicted only from context angles, while the physical loss is
measured on disjoint target angles that were not used by the predictor.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import to_plain
from models.phase_screen import PhaseScreenModel, heldout_agreement
from scripts.pilot_phase_asp import DATA_ROOT, corrected_config
from train_phase_screen import (
    build_cache,
    mean_regularizer,
    sample_ids,
    screen_regularizer,
    validate,
)
from train_phase_screen_staged import _warm_start_learned_weights


def plain_args(args):
    return {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }


def split_context_target(train_idx, min_context: int, target_angles: int):
    n = len(train_idx)
    if target_angles < 1 or min_context < 1:
        raise ValueError("context/target counts must be positive")
    if min_context + target_angles > n:
        raise ValueError("min_context + target_angles exceeds available train angles")
    perm = torch.randperm(n, device=train_idx.device)
    target_pos = perm[:target_angles]
    remaining = perm[target_angles:]
    if len(remaining) == min_context:
        context_pos = remaining
    else:
        n_ctx = int(torch.randint(
            min_context, len(remaining) + 1, (), device=train_idx.device).item())
        context_pos = remaining[:n_ctx]
    return context_pos, target_pos, train_idx[context_pos], train_idx[target_pos]


def mean_teacher_loss(mean_controls_us, item, model):
    if not model.mean_controls:
        return mean_controls_us.new_zeros(())
    return F.smooth_l1_loss(
        mean_controls_us / model.mean_limit_us,
        item["teacher_mean_us"] / model.mean_limit_us,
        beta=0.1,
    )


def phase_teacher_loss(phase_controls_us, item, model):
    return F.smooth_l1_loss(
        phase_controls_us / model.limit_us,
        item["teacher_tau"] / model.limit_us,
        beta=0.1,
    )


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
        "training_format": "phase_screen_cross_angle_v1",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--train-per-case", type=int, default=200)
    p.add_argument("--val-per-case", type=int, default=25)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--controls", type=int, default=96)
    p.add_argument("--limit-us", type=float, default=0.5)
    p.add_argument("--mean-controls", type=int, default=8)
    p.add_argument("--mean-limit-us", type=float, default=2.0)
    p.add_argument("--stage", choices=("mean", "cross_angle"), required=True)
    p.add_argument("--min-context-angles", type=int, default=3)
    p.add_argument("--target-angles", type=int, default=2)
    p.add_argument("--mean-teacher-weight", type=float, default=1.0)
    p.add_argument("--phase-teacher-weight", type=float, default=0.0)
    p.add_argument("--screen-reg", type=float, default=0.01)
    p.add_argument("--mean-reg", type=float, default=0.01)
    p.add_argument("--gate-reg", type=float, default=1e-3)
    p.add_argument("--screen-gate-init", type=float, default=0.02)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--init-checkpoint", type=Path)
    args = p.parse_args()

    if not (1 <= args.train_per_case <= 200 and 1 <= args.val_per_case <= 25):
        p.error("sample counts must fit the two source cases")
    if args.steps < 1 or args.val_every < 1:
        p.error("steps and val-every must be positive")
    if args.mean_controls < 1:
        p.error("this trainer requires --mean-controls >= 1")
    if not (0.0 < args.screen_gate_init < 1.0):
        p.error("--screen-gate-init must lie in (0,1)")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    first = torch.load(DATA_ROOT / "shards" / "train_000.pt",
                       map_location="cpu", weights_only=False)
    cfg, meta = corrected_config(args.config, first, args.n_freq)
    cfg.model.normalize_iq = True

    use_gate = args.stage == "cross_angle"
    model = PhaseScreenModel(
        cfg, meta,
        layers=args.layers,
        controls=args.controls,
        limit_us=args.limit_us,
        mean_controls=args.mean_controls,
        mean_limit_us=args.mean_limit_us,
        screen_gate=use_gate,
        screen_gate_init=args.screen_gate_init,
    ).to(device)

    if args.init_checkpoint is not None:
        _warm_start_learned_weights(model, args.init_checkpoint, device)

    train_idx = torch.as_tensor(meta.train_idx, device=device)
    hold_idx = torch.as_tensor(meta.hold_idx, device=device)
    if args.min_context_angles + args.target_angles > len(train_idx):
        p.error("context + target angle counts exceed available train angles")

    train_ids = sample_ids("train", args.train_per_case)
    val_ids = sample_ids("val", args.val_per_case)
    need_reference = args.stage == "cross_angle"

    print(json.dumps({
        "event": "setup",
        "stage": args.stage,
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "n_freq": len(meta.freqs),
        "parameter_dx_mm": float(cfg.grid.dx * 1e3),
        "lateral_oversample": int(cfg.physics.get("lateral_oversample", 1)),
        "propagation_dx_mm": float(
            cfg.grid.dx * 1e3 / int(cfg.physics.get("lateral_oversample", 1))),
        "screen_gate_enabled": model.screen_gate_enabled,
        "screen_gate_init": float(model.screen_gate_value().detach().cpu()),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
    }), flush=True)

    train_cache = build_cache(
        train_ids, model, meta, train_idx, hold_idx, device,
        args.top_frac, need_reference=need_reference)
    val_cache = build_cache(
        val_ids, model, meta, train_idx, hold_idx, device,
        args.top_frac, need_reference=True)

    if args.stage == "mean":
        # Mean-first stage: keep the screen head exactly fixed.  The shared
        # backbone is still trained because it supplies the mean branch.
        for p_ in model.phase_head.parameters():
            p_.requires_grad_(False)

    trainable = [
        p_ for name, p_ in model.named_parameters()
        if p_.requires_grad and not name.startswith("backbone.fno.head")
    ]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    history = []
    best_score = float("-inf")
    started = time.monotonic()

    def check(step):
        nonlocal best_score
        report = validate(model, val_cache, train_idx, hold_idx)
        report["step"] = int(step)
        report["screen_gate"] = float(model.screen_gate_value().detach().cpu())
        report["elapsed_s"] = time.monotonic() - started
        history.append(report)

        # Mean warm-up is selected by teacher accuracy; cross-angle is selected
        # by the actual fixed validation holdout metric.
        selection = (
            -report["mean_teacher_loss"]
            if args.stage == "mean"
            else report["mean_hold"]
        )
        improved = selection > best_score
        if improved:
            best_score = float(selection)

        payload = checkpoint_payload(
            model, optimizer, step, cfg, args, report, history, best_score)
        torch.save(payload, args.out / "last.pt")
        if improved:
            torch.save(payload, args.out / "best.pt")
        (args.out / "history.json").write_text(
            json.dumps(history, indent=2) + "\n")

        print(json.dumps({
            "event": "validation",
            "step": int(step),
            "selection": float(selection),
            "best_score": float(best_score),
            "improved": bool(improved),
            "screen_gate": report["screen_gate"],
            "mean_hold": report["mean_hold"],
            "mean_uniform_hold": report["mean_uniform_hold"],
            "mean_teacher_loss": report["mean_teacher_loss"],
        }), flush=True)

    check(0)

    for step in range(1, args.steps + 1):
        model.train()
        item = train_cache[torch.randint(len(train_cache), ()).item()]

        if args.stage == "mean":
            # Angle-count randomization remains in the predictor, but the
            # relative-screen head contributes neither to ds nor to the loss.
            n_total = len(train_idx)
            n_ctx = int(torch.randint(
                args.min_context_angles, n_total + 1, (),
                device=device).item())
            ctx_idx = train_idx[
                torch.randperm(n_total, device=device)[:n_ctx]
            ]
            phase_raw, mean_raw, bulk_raw = model.predict_components(
                item["iq"], ctx_idx)
            mean_us = model.mean_limit_us * torch.tanh(mean_raw)
            mean_sup = mean_teacher_loss(mean_us, item, model)
            mean_prior = mean_regularizer(mean_us, model.mean_limit_us)
            loss = (args.mean_teacher_weight * mean_sup
                    + args.mean_reg * mean_prior)
            cross = None
            phase_sup = phase_raw.new_zeros(())
            gate = model.screen_gate_value()

        else:
            ctx_pos, tgt_pos, ctx_idx, tgt_idx = split_context_target(
                train_idx, args.min_context_angles, args.target_angles)

            phase_raw, mean_raw, bulk_raw = model.predict_components(
                item["iq"], ctx_idx)
            ds = model.network_components_to_slowness(
                phase_raw, mean_raw, bulk_raw)

            ctx_imgs = model.angle_images(ds, item["D"], ctx_idx)
            tgt_imgs = model.angle_images(ds, item["D"], tgt_idx)
            ref = item["ref"]
            cross = heldout_agreement(
                ctx_imgs,
                tgt_imgs,
                ref["mask"],
                ref["train_scales"][:, ctx_pos],
                ref["train_scales"][:, tgt_pos],
            ).mean()

            phase_us = model.limit_us * torch.tanh(phase_raw)
            mean_us = model.mean_limit_us * torch.tanh(mean_raw)
            gate = model.screen_gate_value()
            effective_phase_us = gate * phase_us

            mean_sup = mean_teacher_loss(mean_us, item, model)
            phase_sup = phase_teacher_loss(phase_us, item, model)
            screen_prior = screen_regularizer(
                effective_phase_us, model.limit_us)
            mean_prior = mean_regularizer(mean_us, model.mean_limit_us)

            loss = (
                -cross
                + args.mean_teacher_weight * mean_sup
                + args.phase_teacher_weight * phase_sup
                + args.screen_reg * screen_prior
                + args.mean_reg * mean_prior
                + args.gate_reg * gate.square()
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
                "cross_angle_agreement": (
                    float(cross.detach()) if cross is not None else None),
                "mean_teacher_loss": float(mean_sup.detach()),
                "phase_teacher_loss": float(phase_sup.detach()),
                "screen_gate": float(gate.detach()),
                "loss": float(loss.detach()),
                "elapsed_s": time.monotonic() - started,
            }), flush=True)

        if step % args.val_every == 0 or step == args.steps:
            check(step)

    print(json.dumps({
        "event": "done",
        "stage": args.stage,
        "best_selection_score": best_score,
        "final_screen_gate": float(model.screen_gate_value().detach().cpu()),
        "elapsed_s": time.monotonic() - started,
    }), flush=True)


if __name__ == "__main__":
    main()

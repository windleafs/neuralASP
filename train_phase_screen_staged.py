"""Staged/resumable training for the dual-head phase-screen model.

Typical workflow
----------------
1. Teacher warm-up (cheap; no differentiable imaging in the training loss):

   python train_phase_screen_staged.py \
       --config configs/l11_ultrawave_500_11angle.yaml \
       --out runs/phase_screen_teacher \
       --objective teacher --steps 1500 \
       --train-per-case 200 --val-per-case 25

2. Hybrid fine-tuning through the configured oversampled ASP operator:

   python train_phase_screen_staged.py \
       --config configs/l11_ultrawave_500_11angle.yaml \
       --out runs/phase_screen_hybrid \
       --objective hybrid --steps 800 \
       --init-checkpoint runs/phase_screen_teacher/best.pt \
       --train-per-case 200 --val-per-case 25

3. Resume an interrupted staged run exactly from ``last.pt``:

   python train_phase_screen_staged.py ... \
       --resume runs/phase_screen_hybrid/last.pt --steps 1200

``--init-checkpoint`` is a weight-only warm start: optimizer/history/step reset.
It intentionally ignores Born/ASP buffers so checkpoints trained before the
0.1 mm propagation-grid change can still seed the learned predictor when the
learned tensor shapes match.  ``--resume`` is stricter and requires a checkpoint
written by this script with optimizer/history state.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import to_plain
from models.phase_screen import PhaseScreenModel, coherence
from scripts.pilot_phase_asp import DATA_ROOT, corrected_config
from train_phase_screen import (
    build_cache,
    mean_regularizer,
    random_context,
    regularizer,
    sample_ids,
    screen_regularizer,
    teacher_loss,
    validate,
)


def _plain_args(args):
    return {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }


def _warm_start_learned_weights(model, checkpoint_path: Path, device):
    """Load matching learned tensors while leaving current physics buffers intact."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise ValueError(f"{checkpoint_path}: checkpoint has no 'model' state")
    current = model.state_dict()
    copied = []
    skipped = []
    for key, value in ckpt["model"].items():
        if key.startswith("born."):
            skipped.append((key, "physics buffer"))
            continue
        if key not in current:
            skipped.append((key, "missing in current model"))
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            skipped.append((key, f"shape {tuple(value.shape)} -> {tuple(current[key].shape)}"))
            continue
        current[key] = value
        copied.append(key)
    model.load_state_dict(current, strict=True)
    if not copied:
        raise RuntimeError("warm start copied zero learned tensors")
    print(json.dumps({
        "event": "warm_start",
        "checkpoint": str(checkpoint_path),
        "source_step": int(ckpt.get("step", -1)),
        "copied_tensors": len(copied),
        "skipped_tensors": len(skipped),
        "skipped_preview": skipped[:8],
    }), flush=True)
    return ckpt


def _assert_resume_architecture(args, ckpt):
    saved = ckpt.get("args", {})
    for key in ("layers", "controls", "mean_controls", "fit_bulk"):
        if key in saved and saved[key] != getattr(args, key):
            raise ValueError(
                f"resume architecture mismatch: {key}: checkpoint={saved[key]} "
                f"current={getattr(args, key)}")


def _checkpoint_payload(model, optimizer, step, cfg, args, report,
                        history, best_score):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": to_plain(cfg),
        "args": _plain_args(args),
        "validation": report,
        "history": history,
        "best_score": float(best_score),
        "training_format": "phase_screen_staged_v1",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--out", type=Path, default=Path("runs/phase_screen_staged"))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--train-per-case", type=int, default=200)
    p.add_argument("--val-per-case", type=int, default=25)
    p.add_argument("--steps", type=int, default=1000,
                   help="total target optimization step; on resume this is not an increment")
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--reg", type=float, default=0.01)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--controls", type=int, default=96)
    p.add_argument("--limit-us", type=float, default=0.5)
    p.add_argument("--mean-controls", type=int, default=8)
    p.add_argument("--mean-limit-us", type=float, default=2.0)
    p.add_argument("--min-context-angles", type=int, default=3)
    p.add_argument("--fit-bulk", action="store_true")
    p.add_argument("--objective", choices=("teacher", "coherence", "hybrid"),
                   default="teacher")
    p.add_argument("--teacher-weight", type=float, default=1.0)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260916)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--init-checkpoint", type=Path,
                       help="weight-only warm start; optimizer/step/history reset")
    group.add_argument("--resume", type=Path,
                       help="resume model+optimizer+history from staged last.pt")
    args = p.parse_args()

    if not (1 <= args.train_per_case <= 200 and 1 <= args.val_per_case <= 25):
        p.error("sample counts must fit the two source cases")
    if args.steps < 0 or args.val_every < 1:
        p.error("--steps must be non-negative and --val-every >= 1")
    if args.mean_controls < 0:
        p.error("--mean-controls must be non-negative")
    if args.mean_controls and args.fit_bulk:
        p.error("--fit-bulk requires --mean-controls 0")
    if args.limit_us <= 0 or args.mean_limit_us <= 0:
        p.error("delay limits must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    first = torch.load(DATA_ROOT / "shards" / "train_000.pt",
                       map_location="cpu", weights_only=False)
    cfg, meta = corrected_config(args.config, first, args.n_freq)
    cfg.model.normalize_iq = True
    model = PhaseScreenModel(
        cfg, meta, layers=args.layers, controls=args.controls,
        limit_us=args.limit_us, mean_controls=args.mean_controls,
        mean_limit_us=args.mean_limit_us, fit_bulk=args.fit_bulk).to(device)

    init_source = None
    if args.init_checkpoint is not None:
        init_source = _warm_start_learned_weights(
            model, args.init_checkpoint, device)

    train_idx = torch.as_tensor(meta.train_idx, device=device)
    hold_idx = torch.as_tensor(meta.hold_idx, device=device)
    if not (1 <= args.min_context_angles <= len(train_idx)):
        p.error("--min-context-angles must not exceed available training angles")

    train_ids = sample_ids("train", args.train_per_case)
    val_ids = sample_ids("val", args.val_per_case)
    need_train_reference = args.objective != "teacher"
    print(json.dumps({
        "event": "setup",
        "objective": args.objective,
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "n_freq": len(meta.freqs),
        "layers": args.layers,
        "controls": args.controls,
        "mean_controls": args.mean_controls,
        "lateral_oversample": int(cfg.physics.get("lateral_oversample", 1)),
        "parameter_dx_mm": float(cfg.grid.dx * 1e3),
        "propagation_dx_mm": float(cfg.grid.dx * 1e3 /
                                    int(cfg.physics.get("lateral_oversample", 1))),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "resume": str(args.resume) if args.resume else None,
    }), flush=True)

    # Teacher-only training does not need the expensive train-set image reference.
    train_cache = build_cache(
        train_ids, model, meta, train_idx, hold_idx, device, args.top_frac,
        need_reference=need_train_reference)
    # Validation always uses the physical imaging chain.
    val_cache = build_cache(
        val_ids, model, meta, train_idx, hold_idx, device, args.top_frac,
        need_reference=True)

    trainable = [p_ for name, p_ in model.named_parameters()
                 if not name.startswith("backbone.fno.head")]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    history = []
    best_score = float("-inf")
    start_step = 0

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt.get("training_format") != "phase_screen_staged_v1":
            raise ValueError(
                "--resume requires a checkpoint written by train_phase_screen_staged.py; "
                "use --init-checkpoint for legacy/best checkpoints")
        _assert_resume_architecture(args, ckpt)
        if "optimizer" not in ckpt:
            raise ValueError("resume checkpoint has no optimizer state")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        history = list(ckpt.get("history", []))
        best_score = float(ckpt.get("best_score", float("-inf")))
        start_step = int(ckpt["step"])
        print(json.dumps({
            "event": "resume_loaded",
            "checkpoint": str(args.resume),
            "start_step": start_step,
            "history_entries": len(history),
            "best_score": best_score,
        }), flush=True)

    if start_step > args.steps:
        raise ValueError(
            f"resume step {start_step} exceeds requested total --steps {args.steps}")

    started = time.monotonic()

    def check(step):
        nonlocal best_score
        report = validate(model, val_cache, train_idx, hold_idx)
        report["step"] = int(step)
        report["elapsed_s"] = time.monotonic() - started
        history.append(report)
        selection = (-report["mean_teacher_loss"] if args.objective == "teacher"
                     else report["mean_hold"])
        improved = selection > best_score
        if improved:
            best_score = float(selection)
        payload = _checkpoint_payload(
            model, optimizer, step, cfg, args, report, history, best_score)
        # last.pt is the exact continuation point; best.pt is the selected model.
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
            **{k: v for k, v in report.items() if k not in ("rows", "step")},
        }), flush=True)

    if args.resume is None:
        check(0)

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        item = train_cache[torch.randint(len(train_cache), ()).item()]

        if args.objective == "teacher":
            ctx_idx = random_context(train_idx, args.min_context_angles)
            phase_raw, mean_raw, bulk_raw = model.predict_components(
                item["iq"], ctx_idx)
            phase = model.limit_us * torch.tanh(phase_raw)
            mean = (model.mean_limit_us * torch.tanh(mean_raw)
                    if model.mean_controls else mean_raw)
            bulk = (model.bulk_limit_us * torch.tanh(bulk_raw)
                    if model.fit_bulk else bulk_raw.new_zeros(bulk_raw.shape))
            sup = teacher_loss(phase, mean, bulk, item, model)
            coh = None
            reg = screen_regularizer(phase, model.limit_us)
            if model.mean_controls:
                reg = reg + mean_regularizer(mean, model.mean_limit_us)
            elif model.fit_bulk:
                reg = reg + 0.1 * torch.tanh(bulk_raw).square().mean()
            loss = sup + args.reg * reg
        else:
            out = model.forward_precomputed(item["iq"], item["D"], train_idx)
            ref = item["ref"]
            coh = coherence(
                out["images_input"], ref["mask"], ref["train_scales"]).mean()
            reg = regularizer(out, model)
            sup = teacher_loss(
                out["phase_controls_us"], out["mean_controls_us"],
                out["bulk_coeff_us"], item, model)
            loss = -coh + args.reg * reg
            if args.objective == "hybrid":
                loss = loss + args.teacher_weight * sup

        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == start_step + 1 or step % 10 == 0:
            print(json.dumps({
                "event": "train",
                "step": int(step),
                "sample": item["id"],
                "coherence": float(coh.detach()) if coh is not None else None,
                "teacher_loss": float(sup.detach()),
                "regularizer": float(reg.detach()),
                "loss": float(loss.detach()),
                "elapsed_s": time.monotonic() - started,
            }), flush=True)
        if step % args.val_every == 0 or step == args.steps:
            check(step)

    if start_step == args.steps and args.resume is not None:
        print(json.dumps({
            "event": "already_complete",
            "step": start_step,
            "requested_steps": args.steps,
        }), flush=True)

    print(json.dumps({
        "event": "done",
        "start_step": start_step,
        "final_step": args.steps,
        "best_selection_score": best_score,
        "elapsed_s": time.monotonic() - started,
        "warm_start_source_step": (int(init_source.get("step", -1))
                                   if init_source is not None else None),
    }), flush=True)


if __name__ == "__main__":
    main()

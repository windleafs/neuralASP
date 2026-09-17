"""Follow-up oracle experiments for phase-screen bottleneck diagnosis.

This script answers two questions left open by the first oracle decomposition:

1. How much of the GT correction comes from the lateral-mean (bulk) slowness
   profile versus zero-mean lateral aberration screens?
2. For a fixed K-layer representation, is the teacher bottleneck caused mainly
   by the delay bound or by lateral control compression?

The first decomposition compares

    uniform
    mean_only
    screen_only
    mean_plus_screen
    gt_speed_asm

The second decomposition sweeps teacher parameterizations at fixed K.  The
recommended default variants are exactly the diagnostic set discussed after the
first oracle run:

    24 controls, 0.2 us
    24 controls, 20 us
    48 controls, 20 us
    96 controls, 20 us
    full lateral controls, 20 us

For every teacher variant, both screen-only and mean+screen candidates are
reported so clipping/compression can be separated from the missing-bulk gauge.

Example
-------
python scripts/experiment_phase_screen_bottlenecks.py \
    --split val --count 20 --layers 4 \
    --teacher-sweep 24:0.2 24:20 48:20 96:20 full:20 \
    --gpu 0 --out runs/phase_screen_bottlenecks
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from common import rf_to_D  # noqa: E402
from physics.imaging import BornModel  # noqa: E402
from scripts.oracle_phase_screen_decomposition import (  # noqa: E402
    fixed_reference,
    ideal_to_ds,
    images_and_field,
    print_table,
    projection_to_ds,
    sample_ids,
    score,
    summarize,
)
from scripts.pilot_phase_asp import (  # noqa: E402
    DATA_ROOT,
    corrected_config,
    embed,
    padded_meta,
)


def parse_teacher_sweep(values: list[str]):
    result = []
    for value in values:
        try:
            controls_text, limit_text = value.split(":", 1)
            controls = None if controls_text.lower() == "full" else int(controls_text)
            limit_us = float(limit_text)
        except Exception as exc:
            raise ValueError(
                f"invalid teacher sweep item {value!r}; expected C:limit_us or full:limit_us"
            ) from exc
        if controls is not None and controls < 2:
            raise ValueError("teacher controls must be >= 2")
        if limit_us <= 0:
            raise ValueError("teacher limit_us must be positive")
        result.append((controls, limit_us))
    return result


def lateral_mean_ds(true_ds: torch.Tensor, pad: int):
    """Depth-resolved lateral mean, broadcast over the whole padded grid.

    The mean is estimated only over the physical aperture.  It is then applied
    across the padded propagation grid because it represents the common
    background/bulk propagation term rather than a localized aberration.
    """
    physical = true_ds[:, pad:-pad] if pad else true_ds
    mean_profile = physical.mean(dim=-1, keepdim=True)
    return mean_profile.expand_as(true_ds).clone()


def projection_stats(info: dict):
    return {
        "target_max_abs_us": float(info["target_max_abs_us"]),
        "saturated_control_fraction": float(info["saturated_control_fraction"]),
        "control_count": int(info.get("control_count", -1)),
        "control_max_abs_us": float(info.get("control_max_abs_us", float("nan"))),
        "control_frac_gt_95pct_limit": float(
            info.get("control_frac_gt_95pct_limit", float("nan"))
        ),
    }


@torch.no_grad()
def run_one(sample_id: str, args, teacher_sweep, device):
    sample = torch.load(DATA_ROOT / "shards" / f"{sample_id}.pt",
                        map_location="cpu", weights_only=False)
    cfg, meta = corrected_config(args.config, sample, args.n_freq)
    born = BornModel(
        padded_meta(meta, args.pad, cfg.grid.dx),
        cfg.grid.nx + 2 * args.pad,
        cfg.grid.nz,
        cfg.grid.dx,
        cfg.grid.dz,
        cfg.physics.c0,
        eps=cfg.physics.eps_evanescent,
        spreading=cfg.physics.spreading,
    ).to(device)

    D = rf_to_D(sample["rf"].to(device), meta)
    true_ds = embed(sample["delta_s"].to(device), args.pad)
    if args.pad:
        true_ds[:, :args.pad] = 0
        true_ds[:, -args.pad:] = 0
    truth_abs = sample["m"].abs().to(device)

    all_idx = torch.arange(cfg.acq.n_angles, device=device)
    train_idx = torch.as_tensor(meta.train_idx, device=device)
    hold_idx = torch.as_tensor(meta.hold_idx, device=device)

    gt_images, gt_u = images_and_field(born, true_ds, D, all_idx)
    zero = torch.zeros_like(true_ds)
    uniform_images, uniform_u = images_and_field(born, zero, D, all_idx)
    mask, train_scales, hold_scales = fixed_reference(
        uniform_images, train_idx, hold_idx, args.pad, args.top_frac
    )

    rows = []

    def record(method: str, ds: torch.Tensor, extra=None):
        images, u = images_and_field(born, ds, D, all_idx)
        row = {
            "sample": sample_id,
            "case": sample["metadata"].get("case"),
            "method": method,
            **score(images, u, gt_u, train_idx, hold_idx, mask,
                    train_scales, hold_scales, args.pad, truth_abs, born),
        }
        if extra:
            row.update(extra)
        rows.append(row)

    # Shared baselines.
    rows.append({
        "sample": sample_id,
        "case": sample["metadata"].get("case"),
        "method": "uniform",
        **score(uniform_images, uniform_u, gt_u, train_idx, hold_idx, mask,
                train_scales, hold_scales, args.pad, truth_abs, born),
    })
    rows.append({
        "sample": sample_id,
        "case": sample["metadata"].get("case"),
        "method": "gt_speed_asm",
        **score(gt_images, gt_u, gt_u, train_idx, hold_idx, mask,
                train_scales, hold_scales, args.pad, truth_abs, born),
    })

    # Experiment A: separate common mean propagation from relative screens.
    mean_ds = lateral_mean_ds(true_ds, args.pad)
    record("mean_only", mean_ds)

    ideal_ds, ideal_info = ideal_to_ds(
        true_ds, args.layers, born, args.pad, args.oracle_limit_us
    )
    record(f"screen_only_ideal_K{args.layers}", ideal_ds,
           {"projection": projection_stats(ideal_info)})
    record(f"mean_plus_ideal_K{args.layers}", mean_ds + ideal_ds,
           {"projection": projection_stats(ideal_info)})

    # Experiment B: fixed-K teacher bottleneck sweep.  A full-control variant
    # uses born.nx controls, removing lateral downsampling while preserving the
    # same projected_truth_screen -> V2 discrete-screen path.
    for requested_controls, limit_us in teacher_sweep:
        controls = born.nx if requested_controls is None else requested_controls
        teacher_ds, info = projection_to_ds(
            true_ds, args.layers, controls, born, args.pad, limit_us
        )
        label_controls = "full" if requested_controls is None else str(controls)
        suffix = f"K{args.layers}_C{label_controls}_L{limit_us:g}us"
        extra = {
            "projection": projection_stats(info),
            "teacher_controls": controls,
            "teacher_limit_us": limit_us,
            "teacher_full_lateral": requested_controls is None,
        }
        record(f"teacher_screen_{suffix}", teacher_ds, extra)
        record(f"mean_plus_teacher_{suffix}", mean_ds + teacher_ds, extra)

    return rows


def aggregate_projection(rows: list[dict]):
    """Summarize clipping/compression diagnostics for methods that have them."""
    result = {}
    for method in dict.fromkeys(r["method"] for r in rows):
        subset = [r for r in rows if r["method"] == method and "projection" in r]
        if not subset:
            continue
        keys = (
            "target_max_abs_us",
            "saturated_control_fraction",
            "control_max_abs_us",
            "control_frac_gt_95pct_limit",
        )
        result[method] = {
            key: float(np.mean([r["projection"][key] for r in subset]))
            for key in keys
        }
        result[method]["n"] = len(subset)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--layers", type=int, default=4,
                   help="fixed layer count for both decompositions")
    p.add_argument(
        "--teacher-sweep", nargs="+",
        default=["24:0.2", "24:20", "48:20", "96:20", "full:20"],
        help="teacher variants as controls:limit_us; use full for born.nx controls",
    )
    p.add_argument("--oracle-limit-us", type=float, default=20.0)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    teacher_sweep = parse_teacher_sweep(args.teacher_sweep)
    ids = sample_ids(args.split, args.count)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.monotonic()
    result_path = args.out / "results.jsonl"
    with result_path.open("w") as f:
        for i, sid in enumerate(ids, 1):
            sample_rows = run_one(sid, args, teacher_sweep, device)
            rows.extend(sample_rows)
            for row in sample_rows:
                f.write(json.dumps(row) + "\n")
            f.flush()

            summary = summarize(rows)
            projection = aggregate_projection(rows)
            payload = {
                "args": vars(args) | {"out": str(args.out)},
                "teacher_sweep": [
                    {"controls": ("full" if c is None else c), "limit_us": limit}
                    for c, limit in teacher_sweep
                ],
                "ids": ids[:i],
                "summary": summary,
                "projection_summary": projection,
            }
            (args.out / "summary.json").write_text(
                json.dumps(payload, indent=2) + "\n"
            )
            print(json.dumps({
                "event": "sample_complete",
                "sample": sid,
                "completed": i,
                "total": len(ids),
                "elapsed_s": time.monotonic() - started,
                "scores": {
                    r["method"]: {
                        "hold": r["holdout_agreement"],
                        "image": r["image11_abs_corr"],
                        "field": r["field_rel_l2"],
                    }
                    for r in sample_rows
                },
            }), flush=True)
            torch.cuda.empty_cache()

    final = summarize(rows)
    projection = aggregate_projection(rows)
    print_table(final)
    print("\nprojection diagnostics")
    print("-" * 91)
    for method, item in projection.items():
        print(
            f"{method:42s} target_max={item['target_max_abs_us']:.4f} us  "
            f"sat={item['saturated_control_fraction']:.4f}  "
            f"near_limit={item['control_frac_gt_95pct_limit']:.4f}"
        )
    print(json.dumps({
        "event": "done",
        "elapsed_s": time.monotonic() - started,
        "summary": final,
        "projection_summary": projection,
    }), flush=True)


if __name__ == "__main__":
    main()

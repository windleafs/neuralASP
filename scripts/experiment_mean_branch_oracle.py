"""Oracle sweep for the mean-delay branch degrees of freedom.

The relative aberration branch is fixed to the empirically supported
4 x 96-control representation. Only the depth-only mean branch is varied:
2/4/8/16 cumulative-delay controls. This answers how many learned G(z)
controls are needed before training the V3 dual-head network.

Example
-------
python scripts/experiment_mean_branch_oracle.py \
    --split val --count 20 \
    --mean-controls 2 4 8 16 \
    --screen-layers 4 --screen-controls 96 \
    --gpu 0 --out runs/mean_branch_oracle
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
from physics.phase_screen import (  # noqa: E402
    mean_controls_to_ds,
    project_mean_delay_controls,
)
from scripts.experiment_phase_screen_bottlenecks import lateral_mean_ds  # noqa: E402
from scripts.oracle_phase_screen_decomposition import (  # noqa: E402
    fixed_reference,
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


@torch.no_grad()
def run_one(sample_id: str, args, device):
    sample = torch.load(
        DATA_ROOT / "shards" / f"{sample_id}.pt",
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
        uniform_images, train_idx, hold_idx, args.pad, args.top_frac)

    rows = []

    def record(method, ds, extra=None):
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

    full_mean_ds = lateral_mean_ds(true_ds, args.pad)
    record("full_mean_only", full_mean_ds)

    screen_ds, screen_info = projection_to_ds(
        true_ds, args.screen_layers, args.screen_controls, born,
        args.pad, args.screen_limit_us)
    screen_name = f"screen_K{args.screen_layers}_C{args.screen_controls}"
    record(screen_name, screen_ds, {"screen_projection": screen_info})
    record("full_mean_plus_" + screen_name, full_mean_ds + screen_ds,
           {"screen_projection": screen_info})

    for controls in args.mean_controls:
        mean_raw, target_us, info = project_mean_delay_controls(
            true_ds, controls, born.dz, args.pad, args.mean_limit_us)
        if info["saturated_control_fraction"] > 0:
            raise RuntimeError(
                f"mean limit {args.mean_limit_us} us clipped C={controls}; "
                "increase --mean-limit-us")
        mean_ds = mean_controls_to_ds(
            mean_raw, born.nz, born.nx, born.dz, args.mean_limit_us)
        extra = {
            "mean_projection": info,
            "mean_target_controls_us": target_us.tolist(),
        }
        record(f"mean_C{controls}_only", mean_ds, extra)
        record(f"mean_C{controls}_plus_{screen_name}",
               mean_ds + screen_ds, extra)

    return rows


def aggregate_mean_projection(rows):
    result = {}
    for method in dict.fromkeys(r["method"] for r in rows):
        subset = [
            r for r in rows
            if r["method"] == method and "mean_projection" in r]
        if not subset:
            continue
        result[method] = {
            "target_max_abs_us": float(np.mean([
                r["mean_projection"]["target_max_abs_us"] for r in subset])),
            "saturated_control_fraction": float(np.mean([
                r["mean_projection"]["saturated_control_fraction"]
                for r in subset])),
            "end_delay_us_mean": float(np.mean([
                r["mean_projection"]["end_delay_us"] for r in subset])),
            "end_delay_us_std": float(np.std([
                r["mean_projection"]["end_delay_us"] for r in subset])),
            "n": len(subset),
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--mean-controls", type=int, nargs="+",
                   default=[2, 4, 8, 16])
    p.add_argument("--mean-limit-us", type=float, default=20.0)
    p.add_argument("--screen-layers", type=int, default=4)
    p.add_argument("--screen-controls", type=int, default=96)
    p.add_argument("--screen-limit-us", type=float, default=20.0)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if any(c < 1 for c in args.mean_controls):
        p.error("all --mean-controls values must be positive")
    if args.mean_limit_us <= 0 or args.screen_limit_us <= 0:
        p.error("delay limits must be positive")

    ids = sample_ids(args.split, args.count)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.monotonic()
    with (args.out / "results.jsonl").open("w") as f:
        for i, sid in enumerate(ids, 1):
            sample_rows = run_one(sid, args, device)
            rows.extend(sample_rows)
            for row in sample_rows:
                f.write(json.dumps(row) + "\n")
            f.flush()

            payload = {
                "args": vars(args) | {"out": str(args.out)},
                "ids": ids[:i],
                "summary": summarize(rows),
                "mean_projection_summary": aggregate_mean_projection(rows),
            }
            (args.out / "summary.json").write_text(
                json.dumps(payload, indent=2) + "\n")
            print(json.dumps({
                "event": "sample_complete",
                "sample": sid,
                "completed": i,
                "total": len(ids),
                "elapsed_s": time.monotonic() - started,
            }), flush=True)
            torch.cuda.empty_cache()

    final = summarize(rows)
    print_table(final)
    print("\nmean projection diagnostics")
    print("-" * 91)
    for method, item in aggregate_mean_projection(rows).items():
        print(
            f"{method:42s} target_max={item['target_max_abs_us']:.4f} us  "
            f"end={item['end_delay_us_mean']:+.4f}+-"
            f"{item['end_delay_us_std']:.4f} us"
        )
    print(json.dumps({
        "event": "done",
        "elapsed_s": time.monotonic() - started,
        "summary": final,
        "mean_projection_summary": aggregate_mean_projection(rows),
    }), flush=True)


if __name__ == "__main__":
    main()

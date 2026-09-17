"""Unified visual + quantitative evaluation for oversampled phase-screen models.

This is the recommended evaluation entry point after the main imaging chain
moved to

    parameter grid   dx = 0.2 mm
    propagation grid dx = 0.1 mm

It deliberately reuses the two already-tested oversampled-aware evaluators
instead of reimplementing physics a third time:

* ``visualize_phase_screen_checkpoint_oversampled.py`` produces
  - Uniform / Network / Teacher / GT-speed B-mode
  - mean-only / screen-only / full-network component ablation
  - predicted-vs-teacher mean profile and phase-screen parameter plots
  - ``metrics.json``

* ``evaluate_phase_screen_fullband_oversampled.py`` produces
  - shared-TGC comparison including Truth |m|
  - local amplitude-difference maps
  - depth-wise hold/coherence/truth-correlation metrics
  - aggregate/ranking ``fullband_summary.json``

The same sample ids are passed to both.  Imaging uses the contiguous full band
by default.  The two evaluators currently run sequentially, so the selected
samples are imaged twice; this trades some runtime for a single, robust entry
point with no duplicated Born/ASP implementation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def run(cmd):
    print(json.dumps({"event": "launch", "command": cmd}), flush=True)
    subprocess.run(cmd, cwd=PROJECT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sample-ids", nargs="+", required=True,
                   help="explicit sample ids, e.g. val_000 val_025 val_049")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--imaging-n-freq", type=int, default=0,
                   help="0 = full contiguous imaging band (recommended)")
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--db-range", type=float, default=60.0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--tgc-smooth-rows", type=int, default=11)
    p.add_argument("--tgc-max-gain-db", type=float, default=30.0)
    p.add_argument("--diff-mask-floor-db", type=float, default=-50.0)
    p.add_argument("--diff-clip-db", type=float, default=6.0)
    p.add_argument("--depth-bin-edges-mm", type=float, nargs="+",
                   default=[3.0, 10.0, 20.0, 30.0, 35.0])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.imaging_n_freq != 0:
        print(json.dumps({
            "warning": "nonzero --imaging-n-freq may create a sparse frequency comb; "
                       "0 is the recommended full-band setting"
        }), flush=True)

    visual_out = args.out / "visual"
    fullband_out = args.out / "fullband"
    visual_out.mkdir(parents=True, exist_ok=True)
    fullband_out.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    checkpoint = str(args.checkpoint)
    sample_ids = list(args.sample_ids)

    visual_cmd = [
        python,
        "scripts/visualize_phase_screen_checkpoint_oversampled.py",
        "--checkpoint", checkpoint,
        "--sample-ids", *sample_ids,
        "--gpu", str(args.gpu),
        "--imaging-n-freq", str(args.imaging_n_freq),
        "--top-frac", str(args.top_frac),
        "--db-range", str(args.db_range),
        "--dpi", str(args.dpi),
        "--out", str(visual_out),
    ]
    run(visual_cmd)

    fullband_cmd = [
        python,
        "scripts/evaluate_phase_screen_fullband_oversampled.py",
        "--checkpoint", checkpoint,
        "--sample-ids", *sample_ids,
        "--gpu", str(args.gpu),
        "--imaging-n-freq", str(args.imaging_n_freq),
        "--top-frac", str(args.top_frac),
        "--db-range", str(args.db_range),
        "--dpi", str(args.dpi),
        "--tgc-smooth-rows", str(args.tgc_smooth_rows),
        "--tgc-max-gain-db", str(args.tgc_max_gain_db),
        "--diff-mask-floor-db", str(args.diff_mask_floor_db),
        "--diff-clip-db", str(args.diff_clip_db),
        "--depth-bin-edges-mm", *[str(v) for v in args.depth_bin_edges_mm],
        "--out", str(fullband_out),
    ]
    run(fullband_cmd)

    summary = {
        "checkpoint": checkpoint,
        "sample_ids": sample_ids,
        "parameter_grid_dx_mm": 0.2,
        "configured_propagation_grid": "read from checkpoint/config; L11 default 0.1 mm",
        "visual_metrics": str(visual_out / "metrics.json"),
        "fullband_summary": str(fullband_out / "fullband_summary.json"),
        "visual_dir": str(visual_out),
        "fullband_dir": str(fullband_out),
    }
    (args.out / "evaluation_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "done", **summary}), flush=True)


if __name__ == "__main__":
    main()

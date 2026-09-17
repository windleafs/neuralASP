"""Write a compact provenance and numeric-range report for the L11 dataset."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from common import build_meta, load_config
from data.l11_kwave import validate_l11_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PROJECT / "configs/l11_kwave.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg.data.root)
    manifest = json.loads((root / "index.json").read_text())
    meta = build_meta(cfg)
    rf_rms, m_rms, c_min, c_max, ds_max = [], [], [], [], []
    for record in manifest["samples"]:
        sample = torch.load(root / record["path"], map_location="cpu", weights_only=False)
        validate_l11_sample(sample, cfg, meta, check_transform=False)
        rf_rms.append(float(sample["rf"].square().mean().sqrt()))
        m_rms.append(float(sample["m"].abs().square().mean().sqrt()))
        c_min.append(float(sample["c"].min()))
        c_max.append(float(sample["c"].max()))
        ds_max.append(float(sample["delta_s"].abs().max()))
    separation = {}
    cases = sorted({r["case"] for r in manifest["samples"]})
    for case in cases:
        train = [r["z_index"] for r in manifest["samples"]
                 if r["case"] == case and r["split"] == "train"]
        val = [r["z_index"] for r in manifest["samples"]
               if r["case"] == case and r["split"] == "val"]
        if train and val:
            separation[case] = min(abs(a - b) for a in train for b in val)
    train_cases = {r["case"] for r in manifest["samples"] if r["split"] == "train"}
    test_cases = {r["case"] for r in manifest["samples"] if r["split"] == "test"}
    eval_metrics = PROJECT / "runs/l11_kwave_smoke/eval/metrics.json"
    counts = Counter((r["split"], r["case"]) for r in manifest["samples"])
    report = {
        "dataset_root": str(root), "config": str(args.config),
        "counts": {f"{split}/{case}": count for (split, case), count in counts.items()},
        "all_status_complete": all(r["status"] == "complete" for r in manifest["samples"]),
        "test_case_disjoint": not bool(train_cases & test_cases),
        "min_train_val_native_slice_separation": separation,
        "acquisition": manifest["acquisition"], "model_grid": manifest["model_grid"],
        "frequency_bins": len(meta.freqs),
        "frequency_range_hz": [float(meta.freqs[0]), float(meta.freqs[-1])],
        "numeric_ranges": {
            "rf_rms_min_max": [min(rf_rms), max(rf_rms)],
            "m_rms_min_max": [min(m_rms), max(m_rms)],
            "c_min_max_m_s": [min(c_min), max(c_max)],
            "delta_s_abs_max_s_m": max(ds_max),
        },
        "normalization": json.loads((root / "normalization.json").read_text()),
        "validation": json.loads((root / "validation_summary.json").read_text()),
        "pilot_smoke": json.loads((root / "pilot_smoke.json").read_text()),
        "formal_training_smoke": "m/eta/joint each 1 step; not full training",
        "eval_smoke_batches": len(json.loads(eval_metrics.read_text())["per_sample"]),
        "limitations": [
            "m is a normalized high-pass log-impedance surrogate, not an exact Born inversion truth.",
            "Full-wave k-Wave RF and the neural_asp Born forward model have model mismatch.",
            "Smoke results demonstrate runnable plumbing, not trained reconstruction accuracy.",
        ],
    }
    output = root / "dataset_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

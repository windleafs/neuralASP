"""Pack raw k-Wave NPZ files into neural_asp PyTorch shards.

The first pilot establishes one fixed RF gain. Every subsequent sample uses
that same gain; no per-sample normalization is performed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from common import build_meta, load_config, rf_to_D  # noqa: E402
from data.l11_kwave import validate_l11_sample  # noqa: E402


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_json(path: Path):
    return json.loads(path.read_text())


def rf_rms(raw_path: Path):
    with np.load(raw_path) as raw:
        rf = np.asarray(raw["rf"], dtype=np.float64)
        return float(np.sqrt(np.mean(rf * rf)))


def establish_normalization(root: Path, cfg, pilot_id="train_000"):
    path = root / "normalization.json"
    if path.exists():
        value = load_json(path)
        if value.get("method") != "fixed_pilot_rms_gain":
            raise RuntimeError(f"unsupported normalization in {path}")
        return value
    raw_path = root / "raw" / f"{pilot_id}.npz"
    if not raw_path.exists():
        raise FileNotFoundError(f"pilot raw shard is missing: {raw_path}")
    source_rms = rf_rms(raw_path)
    if not np.isfinite(source_rms) or source_rms <= 0:
        raise RuntimeError(f"invalid pilot RF RMS {source_rms}")
    target_rms = float(cfg.data.rf_rms_ref)
    value = {
        "method": "fixed_pilot_rms_gain",
        "pilot_id": pilot_id,
        "pilot_raw_rf_rms": source_rms,
        "target_rf_rms": target_rms,
        "gain": target_rms / source_rms,
        "per_sample_normalization": False,
    }
    atomic_json(path, value)
    return value


def pack_record(root: Path, record, cfg, meta, normalization, overwrite=False):
    raw_path = root / record["raw_path"]
    out_path = root / record["path"]
    if out_path.exists() and not overwrite:
        sample = torch.load(out_path, map_location="cpu", weights_only=False)
        validate_l11_sample(sample, cfg, meta, check_transform=True)
        return "existing"
    if not raw_path.exists():
        return "missing"
    with np.load(raw_path) as raw:
        rf_raw = np.asarray(raw["rf"], dtype=np.float32)
        c_np = np.asarray(raw["c"], dtype=np.float32)
        m_np = np.asarray(raw["m"], dtype=np.float32)
        raw_meta = json.loads(str(raw["metadata_json"].item()))
    gain = np.float32(normalization["gain"])
    rf = torch.from_numpy(np.ascontiguousarray(rf_raw * gain))
    c = torch.from_numpy(np.ascontiguousarray(c_np))
    delta_s = (1.0 / c - 1.0 / float(cfg.physics.c0)).to(torch.float32)
    m_real = torch.from_numpy(np.ascontiguousarray(m_np))
    m = torch.complex(m_real, torch.zeros_like(m_real)).to(torch.complex64)
    D = rf_to_D(rf, meta).to(torch.complex64)
    sample = {
        "rf": rf.to(torch.float32),
        "D": D,
        "delta_s": delta_s,
        "m": m,
        "c": c.to(torch.float32),
        "metadata": {
            **raw_meta,
            "rf_gain": float(normalization["gain"]),
            "raw_rf_rms": float(torch.from_numpy(rf_raw).pow(2).mean().sqrt()),
            "packed_rf_rms": float(rf.pow(2).mean().sqrt()),
            "D_definition": "conj(FFT(rf))[band_idx]",
        },
    }
    validate_l11_sample(sample, cfg, meta, check_transform=True)
    atomic_torch_save(out_path, sample)
    return "packed"


def update_status(root: Path, manifest):
    for record in manifest["samples"]:
        record["status"] = "complete" if (root / record["path"]).exists() else "pending"
    manifest["normalization_path"] = "normalization.json"
    atomic_json(root / "index.json", manifest)


def pack(root: Path, cfg, pilot_only=False, pilot_id="train_000",
         overwrite=False):
    manifest = load_json(root / "index.json")
    normalization = establish_normalization(root, cfg, pilot_id)
    meta = build_meta(cfg)
    records = manifest["samples"]
    if pilot_only:
        records = [r for r in records if r["id"] == pilot_id]
        if not records:
            raise KeyError(pilot_id)
    counts = {"packed": 0, "existing": 0, "missing": 0, "failed": 0}
    for record in records:
        try:
            result = pack_record(root, record, cfg, meta, normalization, overwrite)
            counts[result] += 1
            print(f"{record['id']}: {result}", flush=True)
        except Exception as exc:
            counts["failed"] += 1
            print(f"{record['id']}: FAILED: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            if pilot_only:
                raise
    update_status(root, manifest)
    print(json.dumps({"counts": counts, "normalization": normalization}, indent=2))
    if counts["failed"]:
        raise RuntimeError(f"{counts['failed']} shard(s) failed validation")


def validate_all(root: Path, cfg):
    manifest = load_json(root / "index.json")
    meta = build_meta(cfg)
    by_split = {"train": 0, "val": 0, "test": 0}
    failures = []
    for record in manifest["samples"]:
        path = root / record["path"]
        if not path.exists():
            failures.append(f"{record['id']}: missing")
            continue
        try:
            sample = torch.load(path, map_location="cpu", weights_only=False)
            validate_l11_sample(sample, cfg, meta, check_transform=True)
            by_split[record["split"]] += 1
        except Exception as exc:
            failures.append(f"{record['id']}: {type(exc).__name__}: {exc}")
    expected = {"train": int(cfg.train.n_train), "val": int(cfg.train.n_val),
                "test": int(cfg.train.n_test)}
    summary = {"valid_by_split": by_split, "expected": expected,
               "failures": failures, "ok": by_split == expected and not failures}
    atomic_json(root / "validation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not summary["ok"]:
        raise RuntimeError("dataset validation failed")


def status(root: Path):
    manifest = load_json(root / "index.json")
    values = {}
    for record in manifest["samples"]:
        raw = (root / record["raw_path"]).exists()
        packed = (root / record["path"]).exists()
        key = f"{record['split']}_{'packed' if packed else 'raw' if raw else 'pending'}"
        values[key] = values.get(key, 0) + 1
    print(json.dumps(values, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PROJECT / "configs/l11_kwave.yaml")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--mode", choices=("pilot", "pack", "validate", "status"),
                    required=True)
    ap.add_argument("--pilot-id", default="train_000")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = args.root or Path(cfg.data.root)
    if args.mode == "pilot":
        pack(root, cfg, pilot_only=True, pilot_id=args.pilot_id,
             overwrite=args.overwrite)
    elif args.mode == "pack":
        pack(root, cfg, pilot_only=False, pilot_id=args.pilot_id,
             overwrite=args.overwrite)
    elif args.mode == "validate":
        validate_all(root, cfg)
    else:
        status(root)


if __name__ == "__main__":
    main()

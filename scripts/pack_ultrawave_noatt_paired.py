"""Pack matched no-attenuation UltraWave raw RF into neuralASP shards.

This packer preserves the *source attenuated dataset normalization gain*.
It never estimates a new gain from the no-att data, because absolute amplitude
is the supervision signal in paired attenuation correction.

Truth tensors (c, delta_s, m) are copied from the matching attenuated shard.
Only rf/D are replaced by the no-attenuation simulation.
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

from common import build_meta, load_config, rf_to_D
from data.l11_kwave import validate_l11_sample


DEFAULT_BASE_ROOT = Path(
    "/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle")
DEFAULT_NOATT_ROOT = Path(
    "/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle_noatt")
DEFAULT_CONFIG = PROJECT / "configs/l11_ultrawave_500_11angle.yaml"


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


def source_gain(base_root: Path, noatt_root: Path):
    src = load_json(base_root / "normalization.json")
    paired = load_json(noatt_root / "normalization.json")
    if src.get("method") != "fixed_pilot_rms_gain":
        raise RuntimeError("unsupported source normalization method")
    if paired != src:
        raise RuntimeError(
            "no-att normalization.json differs from source dataset; "
            "paired amplitude supervision requires identical fixed gain")
    gain = float(src["gain"])
    if not np.isfinite(gain) or gain <= 0:
        raise RuntimeError("invalid source normalization gain")
    return gain


def raw_metadata(path: Path):
    with np.load(path) as raw:
        return json.loads(str(raw["metadata_json"].item()))


def pack_record(base_root, noatt_root, record, cfg, meta, gain, overwrite=False):
    raw_path = noatt_root / record["raw_path"]
    out_path = noatt_root / record["path"]
    src_path = base_root / record["path"]

    if out_path.exists() and not overwrite:
        sample = torch.load(out_path, map_location="cpu", weights_only=False)
        validate_l11_sample(sample, cfg, meta, check_transform=True)
        if sample["metadata"].get("paired_variant") != "matched_no_attenuation_v1":
            raise RuntimeError(f"{record['id']}: existing shard is not matched no-att")
        return "existing"
    if not raw_path.exists():
        return "missing_raw"
    if not src_path.exists():
        return "missing_source"

    with np.load(raw_path) as raw:
        rf_raw = np.asarray(raw["rf"], dtype=np.float32)
        noatt_md = json.loads(str(raw["metadata_json"].item()))
    src = torch.load(src_path, map_location="cpu", weights_only=False)

    expected_id = record["id"]
    src_id = src.get("metadata", {}).get("id", src.get("metadata", {}).get("sample_id"))
    if src_id is not None and src_id != expected_id:
        raise RuntimeError(f"{expected_id}: source shard id mismatch: {src_id}")
    if noatt_md.get("sample_id") != expected_id:
        raise RuntimeError(f"{expected_id}: raw no-att sample id mismatch")

    rf = torch.from_numpy(np.ascontiguousarray(rf_raw * np.float32(gain))).float()
    D = rf_to_D(rf, meta).to(torch.complex64)

    sample = {
        "rf": rf,
        "D": D,
        "delta_s": src["delta_s"].clone().to(torch.float32),
        "m": src["m"].clone().to(torch.complex64),
        "c": src["c"].clone().to(torch.float32),
        "metadata": {
            **src["metadata"],
            **noatt_md,
            "id": expected_id,
            "rf_gain": gain,
            "normalization_source_dataset": str(base_root),
            "per_sample_normalization": False,
            "raw_rf_rms": float(torch.from_numpy(rf_raw).pow(2).mean().sqrt()),
            "packed_rf_rms": float(rf.pow(2).mean().sqrt()),
            "D_definition": "conj(FFT(rf))[band_idx]",
            "truth_tensors_copied_from_attenuated_pair": True,
        },
    }
    validate_l11_sample(sample, cfg, meta, check_transform=True)
    atomic_torch_save(out_path, sample)
    return "packed"


def update_manifest(noatt_root: Path, manifest):
    for rec in manifest["samples"]:
        rec["status"] = "complete" if (noatt_root / rec["path"]).exists() else "pending"
    manifest["normalization_path"] = "normalization.json"
    atomic_json(noatt_root / "index.json", manifest)


def pack(base_root, noatt_root, cfg, sample_ids=None, overwrite=False):
    manifest = load_json(noatt_root / "index.json")
    gain = source_gain(base_root, noatt_root)
    meta = build_meta(cfg)
    records = manifest["samples"]
    if sample_ids:
        wanted = set(sample_ids)
        records = [r for r in records if r["id"] in wanted]
        missing = wanted - {r["id"] for r in records}
        if missing:
            raise KeyError(f"sample ids not found: {sorted(missing)}")

    counts = {"packed": 0, "existing": 0, "missing_raw": 0, "missing_source": 0, "failed": 0}
    for rec in records:
        try:
            result = pack_record(
                base_root, noatt_root, rec, cfg, meta, gain, overwrite)
            counts[result] += 1
            print(f"{rec['id']}: {result}", flush=True)
        except Exception as exc:
            counts["failed"] += 1
            print(
                f"{rec['id']}: FAILED: {type(exc).__name__}: {exc}",
                file=sys.stderr, flush=True)
            if sample_ids:
                raise

    update_manifest(noatt_root, manifest)
    print(json.dumps({"counts": counts, "source_gain": gain}, indent=2))
    if counts["failed"]:
        raise RuntimeError(f"{counts['failed']} no-att shard(s) failed")


def validate(base_root, noatt_root, cfg, sample_ids=None):
    manifest = load_json(noatt_root / "index.json")
    meta = build_meta(cfg)
    records = manifest["samples"]
    if sample_ids:
        wanted = set(sample_ids)
        records = [r for r in records if r["id"] in wanted]
    rows = []
    failures = []
    gain = source_gain(base_root, noatt_root)

    for rec in records:
        path = noatt_root / rec["path"]
        src_path = base_root / rec["path"]
        if not path.exists():
            failures.append(f"{rec['id']}: missing no-att shard")
            continue
        try:
            sample = torch.load(path, map_location="cpu", weights_only=False)
            src = torch.load(src_path, map_location="cpu", weights_only=False)
            validate_l11_sample(sample, cfg, meta, check_transform=True)
            if not torch.equal(sample["c"], src["c"]):
                raise RuntimeError("c truth differs from attenuated pair")
            if not torch.equal(sample["delta_s"], src["delta_s"]):
                raise RuntimeError("delta_s truth differs from attenuated pair")
            if not torch.equal(sample["m"], src["m"]):
                raise RuntimeError("m truth differs from attenuated pair")
            if abs(float(sample["metadata"]["rf_gain"]) - gain) > 1e-12:
                raise RuntimeError("RF gain differs from source dataset")
            rows.append({
                "sample": rec["id"],
                "att_rms": float(src["rf"].pow(2).mean().sqrt()),
                "noatt_rms": float(sample["rf"].pow(2).mean().sqrt()),
                "rms_ratio_noatt_over_att": float(
                    sample["rf"].pow(2).mean().sqrt()
                    / src["rf"].pow(2).mean().sqrt().clamp_min(1e-12)),
            })
        except Exception as exc:
            failures.append(f"{rec['id']}: {type(exc).__name__}: {exc}")

    summary = {
        "ok": not failures,
        "n_valid": len(rows),
        "n_requested": len(records),
        "source_gain": gain,
        "mean_rms_ratio_noatt_over_att": float(np.mean([
            r["rms_ratio_noatt_over_att"] for r in rows])) if rows else None,
        "rows": rows,
        "failures": failures,
    }
    atomic_json(noatt_root / "paired_validation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError("paired no-att validation failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    p.add_argument("--noatt-root", type=Path, default=DEFAULT_NOATT_ROOT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--mode", choices=("pack", "validate"), required=True)
    p.add_argument("--sample-ids", nargs="+")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.mode == "pack":
        pack(args.base_root, args.noatt_root, cfg, args.sample_ids, args.overwrite)
    else:
        validate(args.base_root, args.noatt_root, cfg, args.sample_ids)


if __name__ == "__main__":
    main()

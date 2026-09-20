"""Generate strictly matched no-attenuation UltraWave RF pairs.

This script intentionally does NOT import torch. Run it in the same py310 +
NVIDIA HPC SDK environment used by generate_l11_ultrawave_raw.py.

For each source-dataset sample, the anatomy, sound-speed map, density map,
scatterer realization, geometry, transmit angles, source waveform and receive
processing are kept identical. Only attenuation is removed:

    alpha_coeff(x,z) = 0
    BonA(x,z)        = 0

Crucially, the stored scattered RF is referenced to a *matching zero-
attenuation homogeneous simulation*:

    RF_noatt = p(c,rho,alpha=0) - p(c0,rho0,alpha=0)

not to the original alpha=0.002 homogeneous reference.

Raw NPZ files are packed separately by pack_ultrawave_noatt_paired.py so the
UltraWave process never imports torch/OpenMP runtimes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.generate_l11_ultrawave_raw import (  # noqa: E402
    ANGLES,
    DT,
    NT,
    DEFAULT_ROOT as DEFAULT_BASE_ROOT,
    absorption_model,
    configure_gpu,
    geometry_case,
    set_angle,
    sim,
    solver_for,
)
from scripts.generate_ultrawave_contrast_continuation import (  # noqa: E402
    build_base_maps,
    load_manifest,
    record_for,
)

DEFAULT_OUT_ROOT = Path(
    "/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle_noatt")
NOATT_ALPHA = 0.0


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def zero_attenuation_maps(base_maps: dict):
    out = {k: np.array(v, copy=True) for k, v in base_maps.items()}
    c = np.asarray(base_maps["sound_speed"], dtype=np.float32)
    out["alpha_coeff"] = np.zeros_like(c, dtype=np.float32)
    out["BonA"] = np.zeros_like(c, dtype=np.float32)
    if np.any(np.asarray(out["sound_speed"]) <= 0):
        raise RuntimeError("non-positive sound speed")
    if np.any(np.asarray(out["density"]) <= 0):
        raise RuntimeError("non-positive density")
    return out


def homogeneous_noatt_maps():
    case = geometry_case()
    shape = np.asarray(case["maps"]["sound_speed"]).shape
    maps = {
        "sound_speed": np.full(shape, 1540.0, dtype=np.float32),
        "density": np.full(shape, 1000.0, dtype=np.float32),
        "alpha_coeff": np.zeros(shape, dtype=np.float32),
        "BonA": np.zeros(shape, dtype=np.float32),
    }
    return maps


def noatt_reference_fingerprint(case, fit):
    h = hashlib.sha256()
    h.update(np.asarray(case["x"]).tobytes())
    h.update(np.asarray(case["z"]).tobytes())
    h.update(json.dumps({
        "angles_deg": ANGLES.tolist(),
        "dt_s": DT,
        "nt": NT,
        "f0_hz": float(case["f0"]),
        "c0_m_per_s": 1540.0,
        "rho0_kg_per_m3": 1000.0,
        "alpha_coeff": NOATT_ALPHA,
        "BonA": 0.0,
        "fit": fit,
    }, sort_keys=True).encode())
    return h.hexdigest()


def prepare(base_root: Path, out_root: Path):
    source = load_manifest(base_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "raw").mkdir(exist_ok=True)
    (out_root / "shards").mkdir(exist_ok=True)

    records = []
    for src in source["samples"]:
        rec = dict(src)
        rec["status"] = "pending"
        rec["raw_path"] = f"raw/{src['id']}.npz"
        rec["path"] = f"shards/{src['id']}.pt"
        records.append(rec)

    manifest = {
        **{k: v for k, v in source.items() if k != "samples"},
        "version": int(source.get("version", 1)),
        "paired_variant": "matched_no_attenuation_v1",
        "source_dataset_root": str(base_root),
        "attenuation_definition": "alpha_coeff(x,z)=0 everywhere",
        "reference_definition": "homogeneous c0/rho0 with alpha_coeff=0",
        "samples": records,
    }
    atomic_json(out_root / "index.json", manifest)

    src_norm = base_root / "normalization.json"
    if not src_norm.exists():
        raise FileNotFoundError(
            f"missing source normalization {src_norm}; paired amplitude requires identical gain")
    shutil.copy2(src_norm, out_root / "normalization.json")
    print(json.dumps({
        "event": "prepared_noatt_pair_root",
        "base_root": str(base_root),
        "out_root": str(out_root),
        "samples": len(records),
        "normalization_copied": str(src_norm),
    }, indent=2), flush=True)


def reference_path(out_root: Path):
    return out_root / "reference_native_noatt.npz"


def generate_reference(out_root: Path):
    out = reference_path(out_root)
    if out.exists():
        with np.load(out) as f:
            rf = np.asarray(f["rf_native"])
            if rf.shape != (NT, 192, 11) or not np.isfinite(rf).all():
                raise RuntimeError(f"invalid existing no-att reference {out}")
        print(f"valid no-att reference exists: {out}", flush=True)
        return

    maps = homogeneous_noatt_maps()
    case = geometry_case(maps)
    case["maps"] = maps
    fit = absorption_model(case)
    solver = solver_for(case, maps, fit)
    values, trefs, timings = [], [], []
    start = time.monotonic()
    for angle in ANGLES:
        tref = set_angle(solver, case, float(angle))
        total, timing = solver.run()
        values.append(total)
        trefs.append(float(tref))
        timings.append(timing)
        print(
            f"noatt reference angle={angle:+.1f} "
            f"solve={timing['solve_readback_s']:.2f}s "
            f"elapsed={time.monotonic()-start:.1f}s",
            flush=True,
        )
    rf_native = np.stack(values, axis=-1)
    fp = noatt_reference_fingerprint(case, fit)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        rf_native=rf_native,
        dt_s=np.float64(DT),
        nt=np.int64(NT),
        angles_deg=ANGLES,
        source_tref_s=np.asarray(trefs),
        attenuation_coeff=np.float64(NOATT_ALPHA),
        fingerprint=np.asarray(fp),
        timings_json=np.asarray(json.dumps(timings)),
    )
    os.replace(tmp, out)
    del solver
    gc.collect()
    print(f"saved {out}", flush=True)


def load_reference(out_root: Path):
    path = reference_path(out_root)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run --mode reference before generating samples")
    with np.load(path) as f:
        refs = np.array(f["rf_native"], copy=True)
        fp = str(f["fingerprint"].item())
    if refs.shape != (NT, 192, 11):
        raise RuntimeError(f"unexpected no-att reference shape {refs.shape}")
    return refs, fp


def validate_existing(path: Path, sample_id: str):
    with np.load(path) as f:
        rf = np.asarray(f["rf"])
        md = json.loads(str(f["metadata_json"].item()))
    if rf.shape != (11, 192, 2401) or not np.isfinite(rf).all():
        raise RuntimeError(f"invalid existing no-att raw file {path}")
    if md.get("sample_id") != sample_id:
        raise RuntimeError(f"sample mismatch in {path}")
    if md.get("attenuation_mode") != "zero_everywhere":
        raise RuntimeError(f"attenuation metadata mismatch in {path}")


def simulate_one(sample_id: str, record: dict, refs: np.ndarray, ref_fp: str,
                 out_root: Path):
    out = out_root / "raw" / f"{sample_id}.npz"
    if out.exists():
        validate_existing(out, sample_id)
        print(f"skip {sample_id}", flush=True)
        return

    _, base_maps, report = build_base_maps(record)
    maps = zero_attenuation_maps(base_maps)
    case = geometry_case(maps)
    case["maps"] = maps
    fit = absorption_model(case)
    solver = solver_for(case, maps, fit)

    start = time.monotonic()
    values, trefs, timings = [], [], []
    for ai, angle in enumerate(ANGLES):
        tref = set_angle(solver, case, float(angle))
        total, timing = solver.run()
        rf, _ = sim.analytic_channels(
            total - refs[:, :, ai], DT, band=[4e6, 7.5e6])
        if rf.shape != (2401, 192):
            raise RuntimeError(f"unexpected RF shape {rf.shape}")
        values.append(rf.T)
        trefs.append(float(tref))
        timings.append(timing)
        print(
            f"{sample_id} noatt angle={angle:+.1f} "
            f"solve={timing['solve_readback_s']:.2f}s "
            f"elapsed={time.monotonic()-start:.1f}s",
            flush=True,
        )

    rf = np.stack(values).astype(np.float32)
    rms = float(np.sqrt(np.mean(rf.astype(np.float64) ** 2)))
    if not np.isfinite(rf).all() or rms <= 0:
        raise RuntimeError("invalid no-att RF")

    md = {
        "sample_id": sample_id,
        "split": record["split"],
        "case": record["case"],
        "h5": record["h5"],
        "z_index": int(record["z_index"]),
        "scatter_seed": int(record["scatter_seed"]),
        "base_anatomy_id": record.get("base_anatomy_id"),
        "phantom_variant": record.get("phantom_variant", "oa_breast_original"),
        "backend": "ultrawave",
        "paired_variant": "matched_no_attenuation_v1",
        "attenuation_mode": "zero_everywhere",
        "alpha_coeff_value": NOATT_ALPHA,
        "BonA_applied": False,
        "angles_deg": ANGLES.tolist(),
        "source_tref_s": trefs,
        "native_dt_s": DT,
        "native_nt": NT,
        "fs_hz": 40e6,
        "band_hz": [4e6, 7.5e6],
        "raw_scattered_rf_rms": rms,
        "reference_fingerprint": ref_fp,
        "reference_definition": "c0=1540,rho0=1000,alpha_coeff=0",
        "elapsed_s": time.monotonic() - start,
        "timings": timings,
        "base_tissue_voxel_report": report,
    }
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, rf=rf, metadata_json=np.asarray(json.dumps(md)))
    os.replace(tmp, out)
    del solver
    gc.collect()
    print(f"saved {out} rms={rms:.6g}", flush=True)


def select_records(manifest: dict, sample_ids):
    if sample_ids:
        return [record_for(manifest, sid) for sid in sample_ids]
    return list(manifest["samples"])


def status(out_root: Path):
    manifest = load_manifest(out_root)
    counts = {}
    for rec in manifest["samples"]:
        raw = (out_root / rec["raw_path"]).exists()
        packed = (out_root / rec["path"]).exists()
        key = f"{rec['split']}_{'packed' if packed else 'raw' if raw else 'pending'}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({
        "counts": counts,
        "reference_exists": reference_path(out_root).exists(),
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--mode", required=True,
                   choices=("prepare", "reference", "generate", "status"))
    p.add_argument("--sample-ids", nargs="+")
    args = p.parse_args()

    if args.mode == "prepare":
        prepare(args.base_root, args.out_root)
        return
    if args.mode == "status":
        status(args.out_root)
        return
    if not (args.out_root / "index.json").exists():
        raise FileNotFoundError(
            f"missing {args.out_root / 'index.json'}; run --mode prepare first")

    configure_gpu()
    if args.mode == "reference":
        generate_reference(args.out_root)
        return

    refs, ref_fp = load_reference(args.out_root)
    source_manifest = load_manifest(args.base_root)
    for rec in select_records(source_manifest, args.sample_ids):
        simulate_one(rec["id"], rec, refs, ref_fp, args.out_root)


if __name__ == "__main__":
    main()

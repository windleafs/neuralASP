"""Generate weak-to-full contrast UltraWave RF for Born validity tests.

This script intentionally does NOT import torch. It reuses the validated
UltraWave/OpenACC acquisition path from ``generate_l11_ultrawave_raw.py`` and
changes only the medium contrast:

    c_alpha   = c0   + alpha * (c - c0)
    rho_alpha = rho0 + alpha * (rho - rho0)

with c0=1540 m/s and rho0=1000 kg/m^3.

To isolate c/rho scattering, attenuation is held at the homogeneous reference
value (alpha_coeff=0.002 everywhere) and BonA=0 for every alpha. Therefore the
full-wave medium approaches the *same* homogeneous reference as alpha -> 0.
The stored RF is the same scattered-pressure observable used by the main
UltraWave dataset: heterogeneous total pressure minus ``reference_native``
followed by the same 4--7.5 MHz analytic-channel resampling.

Example
-------
python scripts/generate_ultrawave_contrast_continuation.py \
  --sample-ids val_000 val_025 val_049 \
  --alphas 0.05 0.1 0.25 0.5 1.0

Run in the py310 + NVIDIA HPC SDK environment used for UltraWave generation.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.generate_l11_ultrawave_raw import (  # noqa: E402
    ANGLES,
    DT,
    NT,
    DEFAULT_ROOT,
    absorption_model,
    configure_gpu,
    geometry_case,
    medium_builder,
    set_angle,
    sim,
    solver_for,
)

C0 = 1540.0
RHO0 = 1000.0
ALPHA0 = 0.002


def alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:.3f}".replace(".", "p")


def load_manifest(root: Path):
    return json.loads((root / "index.json").read_text())


def record_for(manifest: dict, sample_id: str):
    matches = [r for r in manifest["samples"] if r["id"] == sample_id]
    if len(matches) != 1:
        raise KeyError(f"sample {sample_id!r} not found exactly once")
    record = matches[0]
    if record.get("phantom_variant", "oa_breast_original") != "oa_breast_original":
        raise ValueError("contrast continuation currently supports oa_breast_original only")
    return record


def build_base_maps(record: dict):
    case = geometry_case()
    with h5py.File(record["h5"], "r") as f:
        plane = np.asarray(f["phan"][int(record["z_index"])])
    maps, _, report, *_ = medium_builder.build_medium(
        plane,
        case["x"],
        case["z"],
        seed=int(record["scatter_seed"]),
        preset="dual_scale",
    )
    return case, maps, report


def scale_maps(base_maps: dict, alpha: float):
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    c = np.asarray(base_maps["sound_speed"], dtype=np.float32)
    rho = np.asarray(base_maps["density"], dtype=np.float32)
    out = {k: np.array(v, copy=True) for k, v in base_maps.items()}
    out["sound_speed"] = (C0 + alpha * (c - C0)).astype(np.float32)
    out["density"] = (RHO0 + alpha * (rho - RHO0)).astype(np.float32)
    out["alpha_coeff"] = np.full_like(c, ALPHA0, dtype=np.float32)
    out["BonA"] = np.zeros_like(c, dtype=np.float32)
    return out


def load_reference(base_root: Path):
    path = base_root / "reference_native.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; generate the original UltraWave reference first")
    with np.load(path) as f:
        refs = np.array(f["rf_native"], copy=True)
    if refs.shape != (NT, 192, 11):
        raise RuntimeError(f"unexpected reference shape {refs.shape}")
    return refs


def validate_existing(path: Path, sample_id: str, alpha: float):
    with np.load(path) as f:
        rf = np.asarray(f["rf"])
        md = json.loads(str(f["metadata_json"].item()))
    if rf.shape != (11, 192, 2401) or not np.isfinite(rf).all():
        raise RuntimeError(f"invalid existing continuation file {path}")
    if md.get("sample_id") != sample_id or abs(float(md.get("contrast_alpha")) - alpha) > 1e-9:
        raise RuntimeError(f"metadata mismatch in existing {path}")


def simulate_alpha(sample_id: str, record: dict, base_maps: dict, report: dict,
                   refs: np.ndarray, alpha: float, out_root: Path):
    sample_dir = out_root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    out = sample_dir / f"{alpha_tag(alpha)}.npz"
    if out.exists():
        validate_existing(out, sample_id, alpha)
        print(f"skip {sample_id} alpha={alpha:g}", flush=True)
        return

    start = time.monotonic()
    maps = scale_maps(base_maps, alpha)
    case = geometry_case(maps)
    case["maps"] = maps
    fit = absorption_model(case)
    solver = solver_for(case, maps, fit)

    values = []
    trefs = []
    timings = []
    for ai, angle in enumerate(ANGLES):
        tref = set_angle(solver, case, float(angle))
        total, timing = solver.run()
        rf, _ = sim.analytic_channels(total - refs[:, :, ai], DT, band=[4e6, 7.5e6])
        if rf.shape != (2401, 192):
            raise RuntimeError(f"unexpected resampled RF shape {rf.shape}")
        values.append(rf.T)
        trefs.append(tref)
        timings.append(timing)
        print(
            f"{sample_id} alpha={alpha:g} angle={angle:+.1f} "
            f"solve={timing['solve_readback_s']:.2f}s "
            f"elapsed={time.monotonic()-start:.1f}s",
            flush=True,
        )

    rf = np.stack(values).astype(np.float32)
    rms = float(np.sqrt(np.mean(rf.astype(np.float64) ** 2)))
    if not np.isfinite(rf).all() or rms <= 0:
        raise RuntimeError("invalid contrast-continuation RF")

    md = {
        "sample_id": sample_id,
        "case": record["case"],
        "h5": record["h5"],
        "z_index": int(record["z_index"]),
        "scatter_seed": int(record["scatter_seed"]),
        "contrast_alpha": float(alpha),
        "contrast_definition": "c=c0+a(c-c0); rho=rho0+a(rho-rho0)",
        "c0_m_per_s": C0,
        "rho0_kg_per_m3": RHO0,
        "attenuation_mode": "uniform_reference",
        "alpha_coeff_background": ALPHA0,
        "BonA_applied": False,
        "angles_deg": ANGLES.tolist(),
        "source_tref_s": trefs,
        "native_dt_s": DT,
        "native_nt": NT,
        "fs_hz": 40e6,
        "band_hz": [4e6, 7.5e6],
        "raw_scattered_rf_rms": rms,
        "elapsed_s": time.monotonic() - start,
        "timings": timings,
        "base_tissue_voxel_report": report,
    }

    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, rf=rf, metadata_json=np.asarray(json.dumps(md)))
    os.replace(tmp, out)
    del solver
    gc.collect()
    print(f"saved {out} rms={rms:.6g}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--sample-ids", nargs="+", default=["val_000"])
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.05, 0.1, 0.25, 0.5, 1.0])
    args = p.parse_args()

    alphas = sorted(set(float(a) for a in args.alphas))
    if not alphas or any(a <= 0 or a > 1 for a in alphas):
        p.error("all --alphas must lie in (0, 1]")
    out_root = args.out_root or (args.base_root / "contrast_continuation")
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.base_root)
    refs = load_reference(args.base_root)
    configure_gpu()

    run_meta = {
        "base_root": str(args.base_root),
        "sample_ids": args.sample_ids,
        "alphas": alphas,
        "c0_m_per_s": C0,
        "rho0_kg_per_m3": RHO0,
        "attenuation_mode": "uniform_reference",
        "alpha_coeff_background": ALPHA0,
    }
    (out_root / "run_config.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    for sid in args.sample_ids:
        record = record_for(manifest, sid)
        _, base_maps, report = build_base_maps(record)
        for alpha in alphas:
            simulate_alpha(sid, record, base_maps, report, refs, alpha, out_root)


if __name__ == "__main__":
    main()

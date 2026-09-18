"""Generate UltraWave contrast-continuation physics decomposition.

Five paths are supported:
  c_only            : vary sound speed only
  rho_only          : vary density only
  attenuation_only  : vary alpha_coeff only
  c_plus_rho        : vary sound speed and density
  full              : vary sound speed, density and attenuation

For each path and contrast scale alpha,

    p_alpha = p0 + alpha * (p_full - p0)

for the selected parameters; all unselected parameters remain at homogeneous
reference values. BonA remains zero, matching the linear-acoustic dataset.

The stored RF is the same scattered-pressure observable used by the main
UltraWave dataset: heterogeneous total pressure minus homogeneous reference,
then the same 4--7.5 MHz analytic-channel resampling.

Run only in the py310 + NVIDIA HPC SDK UltraWave environment.

Example
-------
python scripts/generate_ultrawave_physics_continuation.py \
  --sample-ids val_049 \
  --paths c_only rho_only attenuation_only c_plus_rho full \
  --alphas 0.02 0.05 0.1 0.2 0.3 0.5 0.75 1.0
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.generate_l11_ultrawave_raw import (  # noqa: E402
    ANGLES,
    DT,
    DEFAULT_ROOT,
    absorption_model,
    configure_gpu,
    geometry_case,
    set_angle,
    sim,
    solver_for,
)
from scripts.generate_ultrawave_contrast_continuation import (  # noqa: E402
    ALPHA0,
    C0,
    RHO0,
    build_base_maps,
    load_manifest,
    load_reference,
    record_for,
)

PATHS = {
    "c_only": ("sound_speed",),
    "rho_only": ("density",),
    "attenuation_only": ("alpha_coeff",),
    "c_plus_rho": ("sound_speed", "density"),
    "full": ("sound_speed", "density", "alpha_coeff"),
}


def alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:.3f}".replace(".", "p")


def make_maps(base_maps: dict, path_name: str, alpha: float):
    if path_name not in PATHS:
        raise KeyError(path_name)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0,1]")

    c = np.asarray(base_maps["sound_speed"], dtype=np.float32)
    rho = np.asarray(base_maps["density"], dtype=np.float32)
    att = np.asarray(base_maps["alpha_coeff"], dtype=np.float32)

    out = {k: np.array(v, copy=True) for k, v in base_maps.items()}
    out["sound_speed"] = np.full_like(c, C0, dtype=np.float32)
    out["density"] = np.full_like(rho, RHO0, dtype=np.float32)
    out["alpha_coeff"] = np.full_like(att, ALPHA0, dtype=np.float32)
    out["BonA"] = np.zeros_like(c, dtype=np.float32)

    selected = PATHS[path_name]
    if "sound_speed" in selected:
        out["sound_speed"] = (C0 + alpha * (c - C0)).astype(np.float32)
    if "density" in selected:
        out["density"] = (RHO0 + alpha * (rho - RHO0)).astype(np.float32)
    if "alpha_coeff" in selected:
        out["alpha_coeff"] = (ALPHA0 + alpha * (att - ALPHA0)).astype(np.float32)

    if np.any(out["sound_speed"] <= 0):
        raise RuntimeError("non-positive sound speed")
    if np.any(out["density"] <= 0):
        raise RuntimeError("non-positive density")
    if np.any(out["alpha_coeff"] < 0):
        raise RuntimeError("negative attenuation")
    return out


def validate_existing(path: Path, sample_id: str, path_name: str, alpha: float):
    with np.load(path) as f:
        rf = np.asarray(f["rf"])
        md = json.loads(str(f["metadata_json"].item()))
    if rf.shape != (11, 192, 2401) or not np.isfinite(rf).all():
        raise RuntimeError(f"invalid existing file {path}")
    if md.get("sample_id") != sample_id:
        raise RuntimeError(f"sample mismatch in {path}")
    if md.get("physics_path") != path_name:
        raise RuntimeError(f"path mismatch in {path}")
    if abs(float(md.get("contrast_alpha")) - alpha) > 1e-12:
        raise RuntimeError(f"alpha mismatch in {path}")


def simulate_one(sample_id: str, record: dict, base_maps: dict, report: dict,
                 refs: np.ndarray, path_name: str, alpha: float, out_root: Path):
    out_dir = out_root / sample_id / path_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{alpha_tag(alpha)}.npz"

    if out.exists():
        validate_existing(out, sample_id, path_name, alpha)
        print(f"skip {sample_id} path={path_name} alpha={alpha:g}", flush=True)
        return

    maps = make_maps(base_maps, path_name, alpha)
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
            f"{sample_id} path={path_name} alpha={alpha:g} "
            f"angle={angle:+.1f} solve={timing['solve_readback_s']:.2f}s "
            f"elapsed={time.monotonic()-start:.1f}s",
            flush=True,
        )

    rf = np.stack(values).astype(np.float32)
    rms = float(np.sqrt(np.mean(rf.astype(np.float64) ** 2)))
    if not np.isfinite(rf).all() or rms <= 0:
        raise RuntimeError("invalid continuation RF")

    md = {
        "sample_id": sample_id,
        "case": record["case"],
        "h5": record["h5"],
        "z_index": int(record["z_index"]),
        "scatter_seed": int(record["scatter_seed"]),
        "physics_path": path_name,
        "selected_parameters": list(PATHS[path_name]),
        "contrast_alpha": float(alpha),
        "path_definition": "selected p(alpha)=p0+alpha*(p_full-p0)",
        "c0_m_per_s": C0,
        "rho0_kg_per_m3": RHO0,
        "alpha0": ALPHA0,
        "BonA_applied": False,
        "angles_deg": ANGLES.tolist(),
        "source_tref_s": trefs,
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
    p.add_argument("--sample-ids", nargs="+", default=["val_049"])
    p.add_argument("--paths", nargs="+", choices=sorted(PATHS),
                   default=list(PATHS))
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    args = p.parse_args()

    alphas = sorted(set(float(v) for v in args.alphas))
    paths = list(dict.fromkeys(args.paths))
    if not alphas or any(v <= 0 or v > 1 for v in alphas):
        p.error("all --alphas must lie in (0,1]")

    out_root = args.out_root or (args.base_root / "physics_continuation")
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.base_root)
    refs = load_reference(args.base_root)
    configure_gpu()

    (out_root / "run_config.json").write_text(json.dumps({
        "base_root": str(args.base_root),
        "sample_ids": args.sample_ids,
        "paths": paths,
        "alphas": alphas,
        "path_definitions": {k: list(v) for k, v in PATHS.items()},
        "c0_m_per_s": C0,
        "rho0_kg_per_m3": RHO0,
        "alpha0": ALPHA0,
        "BonA_applied": False,
    }, indent=2) + "\n")

    for sid in args.sample_ids:
        record = record_for(manifest, sid)
        _, base_maps, report = build_base_maps(record)
        for path_name in paths:
            for alpha in alphas:
                simulate_one(
                    sid, record, base_maps, report, refs,
                    path_name, alpha, out_root)


if __name__ == "__main__":
    main()

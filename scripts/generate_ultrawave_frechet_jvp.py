"""Generate UltraWave numerical Frechet/JVP data along the real phantom path.

This script intentionally does NOT import torch.  It reuses the validated
UltraWave/OpenACC acquisition chain and evaluates small positive perturbations

    p_eps = p0 + eps * (p_full - p0)

for the actual medium parameters carried by the dataset:
    sound speed c,
    density rho,
    attenuation coefficient alpha_coeff.

BonA remains zero, matching the linear-acoustic dataset.

Because every stored RF observable is already
    total_pressure(p_eps) - homogeneous_reference
followed by the same linear analytic-channel resampling, the directional
Frechet derivative is approximated directly by

    D^(1) ~= D(eps) / eps,

with no analytic Born source-term assumption.

Run this file only in the py310 + NVIDIA HPC SDK environment used for the
UltraWave generator.

Example
-------
python scripts/generate_ultrawave_frechet_jvp.py \
  --sample-ids val_049 \
  --epsilons 0.02 0.05 0.1
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
    NT,
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


def eps_tag(eps: float) -> str:
    return f"eps_{eps:.4f}".replace(".", "p")


def scale_full_path(base_maps: dict, eps: float):
    if not 0.0 < eps <= 0.25:
        raise ValueError("eps must lie in (0, 0.25] for the Frechet pilot")

    c = np.asarray(base_maps["sound_speed"], dtype=np.float32)
    rho = np.asarray(base_maps["density"], dtype=np.float32)
    att = np.asarray(base_maps["alpha_coeff"], dtype=np.float32)

    out = {k: np.array(v, copy=True) for k, v in base_maps.items()}
    out["sound_speed"] = (C0 + eps * (c - C0)).astype(np.float32)
    out["density"] = (RHO0 + eps * (rho - RHO0)).astype(np.float32)
    out["alpha_coeff"] = (ALPHA0 + eps * (att - ALPHA0)).astype(np.float32)
    out["BonA"] = np.zeros_like(c, dtype=np.float32)

    if np.any(out["density"] <= 0):
        raise RuntimeError("non-positive density along Frechet path")
    if np.any(out["sound_speed"] <= 0):
        raise RuntimeError("non-positive sound speed along Frechet path")
    if np.any(out["alpha_coeff"] < 0):
        raise RuntimeError("negative attenuation along Frechet path")
    return out


def validate_existing(path: Path, sample_id: str, eps: float):
    with np.load(path) as f:
        rf = np.asarray(f["rf"])
        md = json.loads(str(f["metadata_json"].item()))
    if rf.shape != (11, 192, 2401) or not np.isfinite(rf).all():
        raise RuntimeError(f"invalid existing Frechet file {path}")
    if md.get("sample_id") != sample_id:
        raise RuntimeError(f"sample id mismatch in {path}")
    if abs(float(md.get("frechet_epsilon")) - eps) > 1e-12:
        raise RuntimeError(f"epsilon mismatch in {path}")
    if md.get("path_parameters") != ["sound_speed", "density", "alpha_coeff"]:
        raise RuntimeError(f"parameter-path mismatch in {path}")


def simulate_eps(sample_id: str, record: dict, base_maps: dict, report: dict,
                 refs: np.ndarray, eps: float, out_root: Path):
    sample_dir = out_root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    out = sample_dir / f"{eps_tag(eps)}.npz"

    if out.exists():
        validate_existing(out, sample_id, eps)
        print(f"skip {sample_id} eps={eps:g}", flush=True)
        return

    maps = scale_full_path(base_maps, eps)
    case = geometry_case(maps)
    case["maps"] = maps
    fit = absorption_model(case)
    solver = solver_for(case, maps, fit)

    start = time.monotonic()
    values = []
    trefs = []
    timings = []

    for ai, angle in enumerate(ANGLES):
        tref = set_angle(solver, case, float(angle))
        total, timing = solver.run()

        # Same observable and preprocessing as the primary dataset.
        rf, _ = sim.analytic_channels(
            total - refs[:, :, ai], DT, band=[4e6, 7.5e6])
        if rf.shape != (2401, 192):
            raise RuntimeError(f"unexpected RF shape {rf.shape}")

        values.append(rf.T)
        trefs.append(float(tref))
        timings.append(timing)
        print(
            f"{sample_id} eps={eps:g} angle={angle:+.1f} "
            f"solve={timing['solve_readback_s']:.2f}s "
            f"elapsed={time.monotonic()-start:.1f}s",
            flush=True,
        )

    rf = np.stack(values).astype(np.float32)
    rms = float(np.sqrt(np.mean(rf.astype(np.float64) ** 2)))
    jvp_rms = rms / eps
    if not np.isfinite(rf).all() or rms <= 0:
        raise RuntimeError("invalid Frechet RF")

    md = {
        "sample_id": sample_id,
        "case": record["case"],
        "h5": record["h5"],
        "z_index": int(record["z_index"]),
        "scatter_seed": int(record["scatter_seed"]),
        "frechet_epsilon": float(eps),
        "path_definition": "p(eps)=p0+eps*(p_full-p0)",
        "path_parameters": ["sound_speed", "density", "alpha_coeff"],
        "c0_m_per_s": C0,
        "rho0_kg_per_m3": RHO0,
        "alpha0": ALPHA0,
        "BonA_applied": False,
        "angles_deg": ANGLES.tolist(),
        "source_tref_s": trefs,
        "native_dt_s": DT,
        "native_nt": NT,
        "fs_hz": 40e6,
        "band_hz": [4e6, 7.5e6],
        "raw_scattered_rf_rms": rms,
        "estimated_jvp_rf_rms": jvp_rms,
        "elapsed_s": time.monotonic() - start,
        "timings": timings,
        "base_tissue_voxel_report": report,
    }

    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        rf=rf,
        metadata_json=np.asarray(json.dumps(md)),
    )
    os.replace(tmp, out)

    del solver
    gc.collect()
    print(
        f"saved {out} rf_rms={rms:.6g} jvp_rms={jvp_rms:.6g}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--sample-ids", nargs="+", default=["val_049"])
    p.add_argument(
        "--epsilons", nargs="+", type=float,
        default=[0.02, 0.05, 0.1],
        help="small positive path scales; use several to verify JVP convergence",
    )
    args = p.parse_args()

    epsilons = sorted(set(float(v) for v in args.epsilons))
    if not epsilons or any(v <= 0 or v > 0.25 for v in epsilons):
        p.error("all --epsilons must lie in (0, 0.25]")

    out_root = args.out_root or (args.base_root / "frechet_jvp")
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.base_root)
    refs = load_reference(args.base_root)
    configure_gpu()

    run_meta = {
        "base_root": str(args.base_root),
        "sample_ids": args.sample_ids,
        "epsilons": epsilons,
        "path_definition": "p(eps)=p0+eps*(p_full-p0)",
        "path_parameters": ["sound_speed", "density", "alpha_coeff"],
        "c0_m_per_s": C0,
        "rho0_kg_per_m3": RHO0,
        "alpha0": ALPHA0,
        "BonA_applied": False,
    }
    (out_root / "run_config.json").write_text(
        json.dumps(run_meta, indent=2) + "\n")

    for sid in args.sample_ids:
        record = record_for(manifest, sid)
        _, base_maps, report = build_base_maps(record)
        for eps in epsilons:
            simulate_eps(
                sid, record, base_maps, report, refs, eps, out_root)


if __name__ == "__main__":
    main()

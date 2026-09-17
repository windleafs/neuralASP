"""Generate weak single-voxel UltraWave scatterers for Green calibration.

The medium is homogeneous except for one 0.2 mm model-grid voxel represented
by the corresponding 4x4 UltraWave cells (50 um spacing).  This deliberately
matches the current BornModel discretization so that anatomy downsampling and
complex reflectivity labels are removed from the comparison.

Default calibration points are approximately

    (x,z) = (0,10) mm, (0,20) mm, (8,20) mm

with a sound-speed-only perturbation dc/c0 = 1e-3. Density and attenuation stay
at the homogeneous reference values and BonA=0.  RF uses the same 11-angle
source, reference subtraction, 4--7.5 MHz band, and 40 MHz resampling as the
main UltraWave dataset.

Run in the py310 + NVIDIA HPC SDK environment used for UltraWave generation.
This script intentionally does not import torch.
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
    DEFAULT_ROOT,
    DT,
    NT,
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
ATTEN0 = 0.002
MODEL_D = 0.2e-3
FINE_D = 50e-6
MODEL_NX = 192
MODEL_NZ = 216
BLOCK = 4
MODEL_X0 = -0.019125
MODEL_Z0 = 0.000075

DEFAULT_POINTS_MM = [(0.0, 10.0), (0.0, 20.0), (8.0, 20.0)]


def point_id(ix: int, iz: int) -> str:
    return f"ix{ix:03d}_iz{iz:03d}"


def nearest_model_index(x_mm: float, z_mm: float):
    ix = int(np.rint((x_mm * 1e-3 - MODEL_X0) / MODEL_D))
    iz = int(np.rint((z_mm * 1e-3 - MODEL_Z0) / MODEL_D))
    if not (0 <= ix < MODEL_NX and 0 <= iz < MODEL_NZ):
        raise ValueError(f"requested point {(x_mm, z_mm)} mm is outside model grid")
    x_actual = MODEL_X0 + ix * MODEL_D
    z_actual = MODEL_Z0 + iz * MODEL_D
    return ix, iz, x_actual, z_actual


def load_reference(base_root: Path):
    path = base_root / "reference_native.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    with np.load(path) as f:
        refs = np.array(f["rf_native"], copy=True)
    if refs.shape != (NT, 192, 11):
        raise RuntimeError(f"unexpected reference shape {refs.shape}")
    return refs


def homogeneous_maps(case):
    shape = (len(case["z"]), len(case["x"]))
    return {
        "sound_speed": np.full(shape, C0, np.float32),
        "density": np.full(shape, RHO0, np.float32),
        "alpha_coeff": np.full(shape, ATTEN0, np.float32),
        "BonA": np.zeros(shape, np.float32),
    }


def implant_model_voxel(maps: dict, case: dict, ix: int, iz: int,
                        delta_c_frac: float):
    width = MODEL_NX * BLOCK
    crop_x_start = (len(case["x"]) - width) // 2
    crop_z_start = medium_builder.FACE
    xs = crop_x_start + ix * BLOCK
    zs = crop_z_start + iz * BLOCK
    if xs < 0 or xs + BLOCK > maps["sound_speed"].shape[1]:
        raise RuntimeError("point x block outside fine grid")
    if zs < 0 or zs + BLOCK > maps["sound_speed"].shape[0]:
        raise RuntimeError("point z block outside fine grid")
    maps["sound_speed"][zs:zs + BLOCK, xs:xs + BLOCK] = C0 * (1.0 + delta_c_frac)
    return crop_x_start, crop_z_start, xs, zs


def validate_existing(path: Path, expected: dict):
    with np.load(path) as f:
        rf = np.asarray(f["rf"])
        md = json.loads(str(f["metadata_json"].item()))
    if rf.shape != (11, 192, 2401) or not np.isfinite(rf).all():
        raise RuntimeError(f"invalid existing point file {path}")
    for key in ("model_ix", "model_iz"):
        if int(md[key]) != int(expected[key]):
            raise RuntimeError(f"existing point metadata mismatch for {key}")
    if abs(float(md["delta_c_frac"]) - float(expected["delta_c_frac"])) > 1e-12:
        raise RuntimeError("existing point contrast mismatch")


def simulate_point(base_root: Path, out_root: Path, refs: np.ndarray,
                   x_mm: float, z_mm: float, delta_c_frac: float):
    ix, iz, x_actual, z_actual = nearest_model_index(x_mm, z_mm)
    pid = point_id(ix, iz)
    out_root.mkdir(parents=True, exist_ok=True)
    out = out_root / f"{pid}_dc{delta_c_frac:.1e}.npz"
    expected = dict(model_ix=ix, model_iz=iz, delta_c_frac=delta_c_frac)
    if out.exists():
        validate_existing(out, expected)
        print(f"skip {pid}", flush=True)
        return

    case = geometry_case()
    maps = homogeneous_maps(case)
    crop_x_start, crop_z_start, fine_x_start, fine_z_start = implant_model_voxel(
        maps, case, ix, iz, delta_c_frac)
    case["maps"] = maps
    fit = absorption_model(case)
    solver = solver_for(case, maps, fit)

    values = []
    trefs = []
    native_scattered_rms = []
    start = time.monotonic()
    for ai, angle in enumerate(ANGLES):
        tref = set_angle(solver, case, float(angle))
        total, timing = solver.run()
        scattered = total - refs[:, :, ai]
        native_scattered_rms.append(float(np.sqrt(np.mean(scattered.astype(np.float64) ** 2))))
        rf, _ = sim.analytic_channels(scattered, DT, band=[4e6, 7.5e6])
        if rf.shape != (2401, 192):
            raise RuntimeError(f"unexpected RF shape {rf.shape}")
        values.append(rf.T)
        trefs.append(tref)
        print(
            f"{pid} angle={angle:+.1f} solve={timing['solve_readback_s']:.2f}s "
            f"elapsed={time.monotonic()-start:.1f}s",
            flush=True,
        )

    rf = np.stack(values).astype(np.float32)
    rf_rms = float(np.sqrt(np.mean(rf.astype(np.float64) ** 2)))
    if not np.isfinite(rf).all() or rf_rms <= 0:
        raise RuntimeError("invalid point-scatterer RF")

    md = {
        "point_id": pid,
        "requested_x_mm": float(x_mm),
        "requested_z_mm": float(z_mm),
        "model_ix": ix,
        "model_iz": iz,
        "model_x_m": float(x_actual),
        "model_z_m": float(z_actual),
        "model_x0_m": MODEL_X0,
        "model_z0_m": MODEL_Z0,
        "model_dx_m": MODEL_D,
        "model_dz_m": MODEL_D,
        "voxel_shape_model": [1, 1],
        "voxel_shape_fine": [BLOCK, BLOCK],
        "fine_dx_m": FINE_D,
        "fine_x_start": int(fine_x_start),
        "fine_z_start": int(fine_z_start),
        "crop_x_start": int(crop_x_start),
        "crop_z_start": int(crop_z_start),
        "delta_c_frac": float(delta_c_frac),
        "scatterer_sound_speed_m_per_s": float(C0 * (1.0 + delta_c_frac)),
        "background_sound_speed_m_per_s": C0,
        "background_density_kg_per_m3": RHO0,
        "background_alpha_coeff": ATTEN0,
        "BonA_applied": False,
        "angles_deg": ANGLES.tolist(),
        "source_tref_s": trefs,
        "native_dt_s": DT,
        "native_nt": NT,
        "fs_hz": 40e6,
        "band_hz": [4e6, 7.5e6],
        "rf_rms": rf_rms,
        "native_scattered_rms_by_angle": native_scattered_rms,
        "elapsed_s": time.monotonic() - start,
    }

    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, rf=rf, metadata_json=np.asarray(json.dumps(md)))
    os.replace(tmp, out)
    del solver
    gc.collect()
    print(f"saved {out} rf_rms={rf_rms:.6g}", flush=True)


def parse_points(values):
    if not values:
        return list(DEFAULT_POINTS_MM)
    if len(values) % 2:
        raise ValueError("--points-mm expects x z pairs")
    return [(float(values[i]), float(values[i + 1]))
            for i in range(0, len(values), 2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--points-mm", nargs="*", type=float,
                   help="x z pairs; default: 0 10, 0 20, 8 20")
    p.add_argument("--delta-c-frac", type=float, default=1e-3)
    args = p.parse_args()
    if not 0 < args.delta_c_frac <= 1e-2:
        p.error("--delta-c-frac must be in (0, 1e-2]")

    points = parse_points(args.points_mm)
    out_root = args.out_root or (args.base_root / "point_scatterer_green")
    refs = load_reference(args.base_root)
    configure_gpu()

    (out_root / "run_config.json").parent.mkdir(parents=True, exist_ok=True)
    (out_root / "run_config.json").write_text(json.dumps({
        "points_mm": points,
        "delta_c_frac": args.delta_c_frac,
        "definition": "one 0.2mm model voxel = 4x4 UltraWave cells; c-only perturbation",
    }, indent=2) + "\n")

    for x_mm, z_mm in points:
        simulate_point(args.base_root, out_root, refs, x_mm, z_mm,
                       args.delta_c_frac)


if __name__ == "__main__":
    main()

"""Generate resumable raw L11 k-Wave shards from OA-Breast anatomy.

Run this script with the Python environment that provides k-wave-python.
Packing into the neural_asp tensor contract is handled separately in the
PyTorch environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

NUMERICAL_ROOT = Path("/data/zhuangyang/NumerialBreastPhantoms")
SIM_ROOT = Path("/data/zhuangyang/kwave_breast_phantom_3d")
DEFAULT_ROOT = NUMERICAL_ROOT / "l11_neural_asp_64"
H5_ROOT = NUMERICAL_ROOT / "NumerialBreastPhantoms" / "hdf5"
ANGLES = np.asarray([-8.0, 0.0, 8.0], dtype=np.float64)
BAND_HZ = (4.0e6, 7.5e6)
MODEL_NZ, MODEL_NX, BLOCK = 216, 192, 4
MODEL_DZ = MODEL_DX = 0.2e-3
SEED0 = 2026091500

sys.path.insert(0, str(NUMERICAL_ROOT))
sys.path.insert(0, str(SIM_ROOT))
import build_acoustic_l11 as medium_builder  # noqa: E402
import run_fixed_bmode as sim  # noqa: E402


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def scan_candidates(path: Path):
    """Find central slices containing both fat and gland in the probe FOV."""
    candidates = []
    with h5py.File(path, "r") as f:
        ds = f["phan"]
        for zi in range(20, ds.shape[0] - 20, 2):
            plane = np.asarray(ds[zi])
            width = min(192, plane.shape[0])
            j0 = (plane.shape[0] - width) // 2
            roi = plane[j0:j0 + width]
            area = float(np.mean(roi > 0))
            gland = float(np.mean(roi == 2))
            fat = float(np.mean(roi == 3))
            skin = int(np.count_nonzero(roi == 4))
            if area >= 0.20 and gland >= 0.05 and fat >= 0.05 and skin >= 20:
                balance = min(gland, fat)
                candidates.append(dict(z_index=zi, area=area, gland=gland,
                                       fat=fat, score=balance + 0.05 * area))
    if not candidates:
        raise RuntimeError(f"no eligible slices in {path}")
    return candidates


def select_spread(candidates, count, exclude=(), min_separation=10):
    """Select high-quality candidates spread across the elevation range."""
    exclude = list(exclude)
    pool = [c for c in candidates
            if all(abs(c["z_index"] - z) >= min_separation for z in exclude)]
    selected = []
    zlo, zhi = pool[0]["z_index"], pool[-1]["z_index"]
    targets = np.linspace(zlo, zhi, count + 2)[1:-1]
    span = max(1.0, zhi - zlo)
    for target in targets:
        valid = [c for c in pool
                 if all(abs(c["z_index"] - s["z_index"]) >= min_separation
                        for s in selected)]
        if not valid:
            break
        best = max(valid, key=lambda c: c["score"] - 0.25 * abs(c["z_index"] - target) / span)
        selected.append(best)
    if len(selected) < count:
        valid = sorted(pool, key=lambda c: c["score"], reverse=True)
        for c in valid:
            if all(abs(c["z_index"] - s["z_index"]) >= min_separation for s in selected):
                selected.append(c)
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise RuntimeError(f"could select only {len(selected)}/{count} slices")
    return sorted(selected, key=lambda c: c["z_index"])


def prepare(root: Path):
    index_path = root / "index.json"
    if index_path.exists():
        print(f"manifest already exists: {index_path}")
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(exist_ok=True)
    plan = [
        ("train", "Neg_07_Left", 24),
        ("train", "Neg_35_Left", 24),
        ("val", "Neg_07_Left", 4),
        ("val", "Neg_35_Left", 4),
        ("test", "Neg_47_Left", 8),
    ]
    scans = {}
    selected_by_case = {}
    records = []
    split_counter = {"train": 0, "val": 0, "test": 0}
    for split, case, count in plan:
        path = H5_ROOT / f"{case}.h5"
        if case not in scans:
            scans[case] = scan_candidates(path)
        exclude = [x["z_index"] for x in selected_by_case.get(case, [])]
        chosen = select_spread(scans[case], count, exclude=exclude,
                               min_separation=10)
        selected_by_case.setdefault(case, []).extend(chosen)
        for candidate in chosen:
            sid = f"{split}_{split_counter[split]:03d}"
            split_counter[split] += 1
            records.append(dict(
                id=sid, split=split, case=case,
                h5=str(path), z_index=int(candidate["z_index"]),
                scatter_seed=SEED0 + len(records), status="pending",
                raw_path=f"raw/{sid}.npz", path=f"shards/{sid}.pt",
                selection={k: float(v) if isinstance(v, float) else int(v)
                           for k, v in candidate.items()},
            ))
    backup = {}
    for case, candidates in scans.items():
        used = [x["z_index"] for x in selected_by_case.get(case, [])]
        pool = [c for c in candidates
                if all(abs(c["z_index"] - z) >= 6 for z in used)]
        backup[case] = pool[:16]
    manifest = dict(
        version=1,
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        acquisition=dict(angles_deg=ANGLES.tolist(), elements=sim.NE,
                         pitch_m=sim.PITCH, fs_hz=sim.FS,
                         band_hz=list(BAND_HZ), rf_samples=2401),
        model_grid=dict(nz=MODEL_NZ, nx=MODEL_NX,
                        dz_m=MODEL_DZ, dx_m=MODEL_DX),
        samples=records, backup_candidates=backup,
    )
    atomic_json(index_path, manifest)
    print(f"wrote {index_path} with {len(records)} samples")
    print(json.dumps(split_counter, indent=2))


def geometry_and_reference_maps():
    x, z = medium_builder.geometry(depth_mm=45.0, lateral_mm=28.0)
    vals = {"sound_speed": sim.C0, "density": 1000.0,
            "alpha_coeff": 0.002, "BonA": 5.0}
    refs = {k: np.full((len(z), len(x)), v, dtype="float32")
            for k, v in vals.items()}
    return x, z, refs


def make_reference(root: Path):
    out = root / "reference_native.npz"
    if out.exists():
        print(f"reference already exists: {out}")
        return
    x, z, refs = geometry_and_reference_maps()
    kg = sim.grid_for(x, z)
    values = []
    t0 = time.monotonic()
    for angle in ANGLES:
        source, sensor, _, _, _ = sim.source_for(
            kg, x, float(angle), face=medium_builder.FACE)
        values.append(sim.propagate(kg, source, sensor, refs))
        print(f"reference angle={angle:+g} done in {time.monotonic()-t0:.1f}s",
              flush=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, rf_native=np.stack(values, axis=-1),
                        angles_deg=ANGLES, x_m=x, z_m=z,
                        face=np.int32(medium_builder.FACE),
                        dt_s=np.float64(kg.dt))
    os.replace(tmp, out)
    print(f"saved {out}")


def block_mean(a, factor=BLOCK):
    nz = MODEL_NZ * factor
    nx = MODEL_NX * factor
    return a[:nz, :nx].reshape(MODEL_NZ, factor, MODEL_NX, factor).mean((1, 3))


def truth_maps(maps, x, face):
    width = MODEL_NX * BLOCK
    i0 = (len(x) - width) // 2
    slz = slice(face, face + MODEL_NZ * BLOCK)
    slx = slice(i0, i0 + width)
    c_fine = maps["sound_speed"][slz, slx].astype(np.float64)
    rho_fine = maps["density"][slz, slx].astype(np.float64)
    c = block_mean(c_fine).astype("float32")
    log_impedance = np.log(np.maximum(c_fine * rho_fine, 1.0))
    smooth = gaussian_filter(log_impedance, sigma=5.0, mode="nearest")
    m = block_mean(log_impedance - smooth).astype("float32")
    m -= m.mean(dtype=np.float64)
    rms = float(np.sqrt(np.mean(m.astype(np.float64) ** 2)))
    if rms <= 1e-12:
        raise RuntimeError("zero reflectivity RMS")
    m *= np.float32(0.2 / rms)
    return c, m, dict(crop_x_start=i0, crop_z_start=face,
                      impedance_highpass_sigma_mm=0.25,
                      m_pre_normalization_rms=rms)


def load_manifest(root: Path):
    return json.loads((root / "index.json").read_text())


def simulate_record(root: Path, record):
    out = root / record["raw_path"]
    if out.exists():
        print(f"skip complete raw shard {record['id']}")
        return
    ref_path = root / "reference_native.npz"
    if not ref_path.exists():
        raise FileNotFoundError("run --mode reference first")
    ref_data = np.load(ref_path)
    refs = np.asarray(ref_data["rf_native"])
    x = np.asarray(ref_data["x_m"])
    z = np.asarray(ref_data["z_m"])
    face = int(ref_data["face"])
    with h5py.File(record["h5"], "r") as f:
        plane = np.asarray(f["phan"][record["z_index"]])
    maps, codes, report, scat_c, scat_rho, table = medium_builder.build_medium(
        plane, x, z, seed=int(record["scatter_seed"]), preset="dual_scale")
    c, m, truth_meta = truth_maps(maps, x, face)
    kg = sim.grid_for(x, z)
    dt = float(kg.dt)
    if not np.isclose(dt, float(ref_data["dt_s"]), rtol=0, atol=1e-15):
        raise RuntimeError("reference time step mismatch")
    rf_angles = []
    t_refs = []
    t0 = time.monotonic()
    for ai, angle in enumerate(ANGLES):
        source, sensor, t_ref, _, _ = sim.source_for(kg, x, float(angle), face=face)
        t_refs.append(float(t_ref))
        total = sim.propagate(kg, source, sensor, maps)
        rf, _ = sim.analytic_channels(total - refs[:, :, ai], dt, band=BAND_HZ)
        rf_angles.append(rf.T)
        print(f"{record['id']} angle={angle:+g} elapsed={time.monotonic()-t0:.1f}s",
              flush=True)
    rf = np.stack(rf_angles).astype("float32")
    if rf.shape != (3, sim.NE, 2401):
        raise RuntimeError(f"unexpected RF shape {rf.shape}")
    if c.shape != (MODEL_NZ, MODEL_NX) or m.shape != c.shape:
        raise RuntimeError(f"unexpected truth shapes {c.shape}, {m.shape}")
    if not all(np.isfinite(a).all() for a in (rf, c, m)):
        raise RuntimeError("non-finite raw sample")
    metadata = dict(
        id=record["id"], split=record["split"], case=record["case"],
        h5=record["h5"], z_index=record["z_index"],
        scatter_seed=record["scatter_seed"], elapsed_s=time.monotonic() - t0,
        tissue_voxel_report=report, preset="dual_scale",
        angles_deg=ANGLES.tolist(), band_hz=list(BAND_HZ), fs_hz=sim.FS,
        t_ref_s=t_refs, source_f0_hz=float(sim.F0),
        truth_x0_m=float(x[truth_meta["crop_x_start"]] + (BLOCK - 1) * sim.DX / 2),
        truth_z0_m=float(z[face] + (BLOCK - 1) * sim.DX / 2),
        density_scatter_relative={"gland": 0.015, "fat": 0.003,
                                  "skin": 0.008, "vessel": 0.0015},
        **truth_meta,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, rf=rf, c=c, m=m,
                        metadata_json=np.asarray(json.dumps(metadata)))
    os.replace(tmp, out)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


def run_worker(root: Path, worker_id: int, workers: int, only_id=None,
               raise_on_error=False):
    manifest = load_manifest(root)
    records = manifest["samples"]
    if only_id is not None:
        records = [r for r in records if r["id"] == only_id]
        if not records:
            raise KeyError(only_id)
    else:
        records = [r for i, r in enumerate(records) if i % workers == worker_id]
    for record in records:
        try:
            simulate_record(root, record)
            err = root / "raw" / f"{record['id']}.error.txt"
            if err.exists():
                err.unlink()
        except Exception as exc:
            err = root / "raw" / f"{record['id']}.error.txt"
            err.write_text(f"{type(exc).__name__}: {exc}\n")
            print(f"FAILED {record['id']}: {exc}", file=sys.stderr, flush=True)
            if raise_on_error:
                raise


def status(root: Path):
    manifest = load_manifest(root)
    counts = {}
    missing = []
    for record in manifest["samples"]:
        complete = (root / record["raw_path"]).exists()
        key = f"{record['split']}_{'complete' if complete else 'pending'}"
        counts[key] = counts.get(key, 0) + 1
        if not complete:
            missing.append(record["id"])
    print(json.dumps(dict(counts=counts, missing=missing), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--mode", required=True,
                    choices=("prepare", "reference", "pilot", "worker", "status"))
    ap.add_argument("--sample-id", default="train_000")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "reference":
        make_reference(args.root)
    elif args.mode == "pilot":
        run_worker(args.root, 0, 1, only_id=args.sample_id,
                   raise_on_error=True)
    elif args.mode == "worker":
        if not (0 <= args.worker_id < args.workers):
            ap.error("worker-id must satisfy 0 <= worker-id < workers")
        run_worker(args.root, args.worker_id, args.workers)
    else:
        status(args.root)


if __name__ == "__main__":
    main()

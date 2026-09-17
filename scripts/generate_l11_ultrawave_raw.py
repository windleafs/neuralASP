"""Independent 500-sample, 11-angle UltraWave linear acoustic full-wave RF.

Run with py310 and NVIDIA HPC SDK on PATH. Do not import torch here: the
validated OpenACC setup must load its own runtime before any PyTorch libgomp.
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

import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, "/home/zhuangyang/fmmodel/UltraWave/benchmarks")
from scripts.generate_l11_kwave_raw import (H5_ROOT, atomic_json, scan_candidates,
                                            truth_maps, medium_builder, sim)
import benchmark_calibration as bench

DEFAULT_ROOT = Path("/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle")
ANGLES = np.linspace(-8., 8., 11)
ORDER, PML, DT, NT = 8, 24, 2.5e-9, 24001
SEED0 = 202609150000


def choose_anatomy(candidates, n_train=200, n_val=25):
    """Reserve two contiguous validation blocks; >=10 slices from training."""
    z = np.asarray([c["z_index"] for c in candidates])
    q = np.asarray([c["score"] for c in candidates])
    sizes = (n_val // 2, n_val - n_val // 2)
    targets = np.quantile(z, [.32, .70])
    windows = []
    for size, target in zip(sizes, targets):
        group = []
        for start in range(len(z) - size + 1):
            ix = np.arange(start, start + size)
            forbidden = (z >= z[ix[0]] - 9) & (z <= z[ix[-1]] + 9)
            score = float(q[ix].mean()) - .25 * abs(z[ix].mean() - target) / np.ptp(z)
            group.append((ix, forbidden, score))
        windows.append(group)
    best = None
    for a in windows[0]:
        for b in windows[1]:
            if z[a[0][-1]] + 10 > z[b[0][0]]:
                continue
            allowed = ~(a[1] | b[1])
            if allowed.sum() < n_train:
                continue
            score = a[2] + b[2]
            if best is None or score > best[0]:
                best = (score, np.flatnonzero(allowed), np.concatenate((a[0], b[0])))
    if best is None:
        raise RuntimeError("cannot partition 200 unique train/25 unique val slices with 2mm guard")
    train_pool = best[1]
    indices = np.rint(np.linspace(0, len(train_pool) - 1, n_train)).astype(int)
    train = [candidates[i] for i in train_pool[indices]]
    val = [candidates[i] for i in best[2]]
    # Strongest anatomical slice first is a representative visibility pilot.
    train.sort(key=lambda c: c["score"], reverse=True)
    return train, val


def geometry_case(maps=None):
    x, z = medium_builder.geometry(depth_mm=45., lateral_mm=28.)
    if maps is None:
        vals = dict(sound_speed=1540., density=1000., alpha_coeff=.002, BonA=0.)
        maps = {k: np.full((len(z), len(x)), v, np.float32) for k, v in vals.items()}
    cols = np.flatnonzero((x >= sim.XE[0]) & (x <= sim.XE[-1]))
    case = dict(maps=maps, x=x, z=z, xe=sim.XE.copy(), dx=50e-6,
                face=medium_builder.FACE, dt=DT, nt=NT, end=(NT-1)*DT,
                f0=7.5e6, c0=1540., pml=PML, delays=np.zeros(len(cols)),
                weights=np.interp(x[cols], sim.XE, np.hanning(sim.NE)),
                tref=4/7.5e6, input_dir="OA-Breast dual_scale")
    bench.validate_case(case)
    return case


def absorption_model(case):
    fit = bench.fit_absorption(case)
    if DT > min(fit["taus_s"]) / 8 * 1.001:
        raise RuntimeError("relaxation timestep stability check failed")
    return fit


def fingerprint(case, fit):
    h = hashlib.sha256()
    h.update(case["x"].tobytes()); h.update(case["z"].tobytes())
    h.update(json.dumps(dict(angles=ANGLES.tolist(), dt=DT, nt=NT, order=ORDER,
                            fit=fit, f0=case["f0"], pml=PML,
                            source="1e5 Pa Gaussian sine sigma=1/f0, continuous Hann",
                            receiver="end-step", reference="1540m/s,1000kg/m3,alpha.002"),
                        sort_keys=True).encode())
    return h.hexdigest()


def prepare(root):
    if (root / "index.json").exists():
        raise FileExistsError(f"manifest already exists: {root / 'index.json'}")
    if shutil.disk_usage(root.parent).free < 35 * 1024**3:
        raise RuntimeError("need at least35GiB free before starting")
    samples, all_selected, stats = [], {}, {}
    for case in ("Neg_07_Left", "Neg_35_Left"):
        pool = scan_candidates(H5_ROOT / f"{case}.h5")
        train, val = choose_anatomy(pool)
        all_selected[case] = {"train": train, "val": val}
        stats[case] = {"eligible": len(pool), "unique_train": len(train), "unique_val": len(val),
                       "min_train_val_layer_gap": min(abs(a["z_index"]-b["z_index"])
                                                      for a in train for b in val)}
    test_case = "Neg_47_Left"
    pool = scan_candidates(H5_ROOT / f"{test_case}.h5")
    indices = np.rint(np.linspace(0, len(pool)-1, 50)).astype(int)
    all_selected[test_case] = {"test": [pool[i] for i in indices]}
    stats[test_case] = {"eligible": len(pool), "unique_test": 50}
    counters = {"train": 0, "val": 0, "test": 0}
    for split in counters:
        for case, chosen in all_selected.items():
            for item in chosen.get(split, []):
                sid = f"{split}_{counters[split]:03d}"
                counters[split] += 1
                samples.append(dict(id=sid, split=split, case=case, h5=str(H5_ROOT/f"{case}.h5"),
                                    z_index=int(item["z_index"]), backend="ultrawave",
                                    base_anatomy_id=f"{case}/z{item['z_index']}",
                                    anatomy_repeated=False, scatter_seed=SEED0+len(samples),
                                    raw_path=f"raw/{sid}.npz", path=f"shards/{sid}.pt",
                                    status="pending", selection=item))
    if counters != dict(train=400, val=50, test=50):
        raise RuntimeError(counters)
    if len({r["base_anatomy_id"] for r in samples}) != 500:
        raise RuntimeError("duplicate anatomy selection")
    case = geometry_case(); fit = absorption_model(case)
    manifest = dict(version=1, backend="ultrawave", simulation="2D linear acoustic full-wave",
                    acquisition=dict(angles_deg=ANGLES.tolist(), elements=192, pitch_m=.2e-3,
                                     fs_hz=40e6, band_hz=[4e6,7.5e6], rf_samples=2401,
                                     source_f0_hz=7.5e6, native_dt_s=DT, native_nt=NT),
                    model_grid=dict(nz=216,nx=192,dz_m=.2e-3,dx_m=.2e-3),
                    physics=dict(space_order=ORDER, dx_m=50e-6, absorption_fit=fit,
                                 BonA_applied=False, boundary="aligned external PML24; graded edge attenuation"),
                    reference_fingerprint=fingerprint(case,fit), anatomy_stats=stats, samples=samples)
    (root/"raw").mkdir(parents=True,exist_ok=True)
    atomic_json(root/"index.json",manifest)
    print(json.dumps(dict(counts=counters,anatomy=stats,unique_anatomies=500),indent=2),flush=True)


def configure_gpu():
    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported in UltraWave GPU process")
    bench.ultra_environment(False)


def solver_for(case, maps, fit):
    solver = bench.UltraSolver(case, maps, ORDER, False, fit)
    solver.source = next(p for p in solver.op.parameters if p.name == "src")
    return solver


def set_angle(solver, case, angle):
    cols = np.flatnonzero((case["x"] >= case["xe"][0]) & (case["x"] <= case["xe"][-1]))
    raw = case["x"][cols] * np.sin(np.deg2rad(angle))/case["c0"]
    case["delays"] = raw - raw.min()
    case["tref"] = 4/case["f0"] - raw.min()
    _, _, wave = bench.source_arrays(case)
    solver.source.data[:] = wave.T
    return float(case["tref"])


def reference(root):
    manifest = json.loads((root/"index.json").read_text())
    out = root/"reference_native.npz"
    if out.exists():
        with np.load(out) as f:
            if str(f["fingerprint"].item()) != manifest["reference_fingerprint"]:
                raise RuntimeError("reference fingerprint mismatch")
        print("valid reference exists",flush=True); return
    configure_gpu(); case=geometry_case(); fit=absorption_model(case)
    solver=solver_for(case,case["maps"],fit)
    values=[]; start=time.monotonic()
    for angle in ANGLES:
        set_angle(solver,case,float(angle))
        rf, timing=solver.run(); values.append(rf)
        print(f"reference angle={angle:+.1f} solve={timing['solve_readback_s']:.2f}s elapsed={time.monotonic()-start:.1f}s",flush=True)
    tmp=out.with_suffix(".tmp.npz")
    np.savez(tmp, rf_native=np.stack(values,axis=-1), dt_s=DT, nt=NT,
             angles_deg=ANGLES, fingerprint=np.asarray(fingerprint(case,fit)))
    os.replace(tmp,out); print(f"saved {out}",flush=True)


def show_pilot(root, images, c, codes, x, z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ix=(np.arange(192)-(192-1)/2)*.2e-3
    iz=np.arange(216)*.2e-3
    image=np.mean(images,axis=0); env=abs(image)
    # One depth-only display gain, never an anatomy-dependent mask/outline.
    tgc=np.minimum(18., .42*iz*1e3)
    shown=env*10**(tgc[:,None]/20)
    display=float(np.percentile(shown[iz>=.002],98))
    db=np.clip(20*np.log10(np.maximum(shown/max(display,1e-30),1e-9)),-45,0)
    if not np.isfinite(db).all() or display<=0:
        raise RuntimeError("invalid pilot B-mode")
    np.savez_compressed(root/"pilot_beamformed.npz",image=image,envelope=env,bmode_db=db,
                        x_m=ix,z_m=iz,tgc_db=tgc,display_reference=display)
    fig,ax=plt.subplots(figsize=(7,7))
    im=ax.imshow(db,cmap="gray",vmin=-45,vmax=0,extent=[ix[0]*1e3,ix[-1]*1e3,iz[-1]*1e3,0],aspect="equal")
    ax.set(xlabel="Lateral [mm]",ylabel="Depth [mm]",title="UltraWave / 11-angle RF B-mode / real sound speed")
    fig.colorbar(im,ax=ax,label="dB"); fig.tight_layout()
    fig.savefig(root/"pilot_bmode_only.png",dpi=160); plt.close(fig)
    # Separate truth panel for QA only; no labels are superimposed on B-mode.
    fig,ax=plt.subplots(figsize=(7,7))
    im=ax.imshow(c,cmap="turbo",extent=[ix[0]*1e3,ix[-1]*1e3,iz[-1]*1e3,0],aspect="equal")
    fig.colorbar(im,ax=ax,label="m/s"); ax.set(title="Pilot sound-speed truth",xlabel="Lateral [mm]",ylabel="Depth [mm]")
    fig.tight_layout(); fig.savefig(root/"pilot_c_truth.png",dpi=160); plt.close(fig)
    # Label-only QA statistic, computed after RF propagation/beamforming.
    from scipy.ndimage import map_coordinates, binary_erosion
    zi=(iz-z[0])/50e-6; xi=(ix-x[0])/50e-6
    ZZ,XX=np.meshgrid(zi,xi,indexing="ij")
    lab=map_coordinates(codes,[ZZ,XX],order=0,mode="nearest")
    results=[]
    for lo in range(5,40,5):
        band=(iz[:,None]*1e3>=lo)&(iz[:,None]*1e3<lo+5)
        gland=binary_erosion(lab==2,iterations=2)&band
        fat=binary_erosion(lab==3,iterations=2)&band
        if gland.sum()>20 and fat.sum()>20:
            a=float(np.mean(env[gland]**2)); b=float(np.mean(env[fat]**2))
            results.append(dict(depth_mm=[lo,lo+5],gland_pixels=int(gland.sum()),fat_pixels=int(fat.sum()),
                                gland_fat_power_contrast_db=10*np.log10(max(a,1e-30)/max(b,1e-30))))
    atomic_json(root/"pilot_bmode_metrics.json",dict(bins=results,tgc="0.42dB/mm capped18dB",display="98th percentile,45dB",anatomy_overlay=False))


def simulate(root, record, refs, pilot=False):
    out=root/record["raw_path"]
    if out.exists():
        with np.load(out) as raw:
            if raw["rf"].shape!=(11,192,2401) or not np.isfinite(raw["rf"]).all():
                raise RuntimeError(f"invalid existing {out}")
        print(f"skip {record['id']}",flush=True); return
    start=time.monotonic()
    with h5py.File(record["h5"],"r") as f:
        plane=np.asarray(f["phan"][record["z_index"]])
    case=geometry_case()
    maps,codes,report,*_=medium_builder.build_medium(plane,case["x"],case["z"],seed=record["scatter_seed"],preset="dual_scale")
    case["maps"]=maps; bench.validate_case(case)
    c,m,truth_meta=truth_maps(maps,case["x"],case["face"])
    fit=absorption_model(case); solver=solver_for(case,maps,fit)
    values,images,trefs,timings=[],[],[],[]; repeat_error=None
    image_x=(np.arange(192)-95.5)*.2e-3; image_z=np.arange(216)*.2e-3
    for ai,angle in enumerate(ANGLES):
        tref=set_angle(solver,case,float(angle)); total,timing=solver.run()
        if pilot and ai==0:
            repeat,_=solver.run()
            repeat_error=float(np.max(abs(total-repeat))/max(float(abs(total).max()),1e-30))
            if repeat_error>1e-5:
                raise RuntimeError(f"state reset mismatch {repeat_error}")
        rf,analytic=sim.analytic_channels(total-refs[:,:,ai],DT,band=[4e6,7.5e6])
        if rf.shape!=(2401,192):
            raise RuntimeError(f"resampledRF shape {rf.shape}")
        values.append(rf.T); trefs.append(tref); timings.append(timing)
        if pilot:
            images.append(sim.beamform(analytic,image_x,image_z,float(angle),tref))
        print(f"{record['id']} angle={angle:+.1f} solve={timing['solve_readback_s']:.2f}s elapsed={time.monotonic()-start:.1f}s",flush=True)
    rf=np.stack(values).astype(np.float32)
    if not all(np.isfinite(a).all() for a in (rf,c,m)) or np.sqrt(np.mean(rf.astype(float)**2))<=0:
        raise RuntimeError("invalid raw sample")
    metadata={k:record[k] for k in ("id","split","case","h5","z_index","scatter_seed","backend","base_anatomy_id","anatomy_repeated")}
    metadata.update(dict(angles_deg=ANGLES.tolist(),fs_hz=40e6,band_hz=[4e6,7.5e6],source_f0_hz=7.5e6,
                         native_dt_s=DT,native_nt=NT,space_order=ORDER,dx_m=50e-6,
                         preset="dual_scale",simulation="2D linear acoustic full-wave",BonA_applied=False,
                         absorption_fit=fit,source_tref_s=trefs,receiver_sampling="pressure after update",
                         elapsed_s=time.monotonic()-start,timings=timings,state_reset_peak_relative_error=repeat_error,
                         tissue_voxel_report=report,reference_fingerprint=fingerprint(case,fit),**truth_meta))
    if pilot:
        show_pilot(root,images,c,codes,case["x"],case["z"])
    tmp=out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,rf=rf,c=c,m=m,metadata_json=np.asarray(json.dumps(metadata)))
    os.replace(tmp,out)
    del solver; gc.collect()
    print(f"saved {record['id']} wall={time.monotonic()-start:.1f}s",flush=True)


def worker(root,worker_id,workers,only=None):
    configure_gpu(); manifest=json.loads((root/"index.json").read_text())
    with np.load(root/"reference_native.npz") as f:
        if str(f["fingerprint"].item())!=manifest["reference_fingerprint"]:
            raise RuntimeError("reference fingerprint mismatch")
        refs=np.array(f["rf_native"],copy=True)
    if refs.shape!=(NT,192,11):
        raise RuntimeError(refs.shape)
    records=[r for i,r in enumerate(manifest["samples"]) if r["id"]==only] if only else [r for i,r in enumerate(manifest["samples"]) if i%workers==worker_id]
    if not records:
        raise RuntimeError("empty worker selection")
    for record in records:
        try:
            simulate(root,record,refs,pilot=only is not None)
            err=root/"raw"/f"{record['id']}.error.txt"
            if err.exists(): err.unlink()
        except Exception as exc:
            (root/"raw"/f"{record['id']}.error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            raise


def status(root):
    manifest=json.loads((root/"index.json").read_text()); counts={}
    for r in manifest["samples"]:
        key=f"{r['split']}_{'raw' if (root/r['raw_path']).exists() else 'pending'}"
        counts[key]=counts.get(key,0)+1
    print(json.dumps(dict(counts=counts,errors=[p.name for p in (root/"raw").glob('*.error.txt')]),indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    ap.add_argument("--mode",required=True,choices=("prepare","reference","pilot","worker","status","gpu-probe"))
    ap.add_argument("--sample-id",default="train_000"); ap.add_argument("--worker-id",type=int,default=0); ap.add_argument("--workers",type=int,default=2)
    args=ap.parse_args()
    if args.mode=="prepare": prepare(args.root)
    elif args.mode=="reference": reference(args.root)
    elif args.mode=="pilot": worker(args.root,0,1,args.sample_id)
    elif args.mode=="worker":
        if not 0<=args.worker_id<args.workers: ap.error("invalid worker-id")
        worker(args.root,args.worker_id,args.workers)
    elif args.mode=="gpu-probe": bench.gpu_probe()
    else: status(args.root)


if __name__=="__main__": main()

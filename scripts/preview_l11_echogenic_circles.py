"""Preview deterministic high-echo circles without running UltraWave or changing RF."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.generate_l11_kwave_raw import medium_builder
from data.echogenic_circles import add_echogenic_circles

DEFAULT_ROOT = Path("/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle")


def raw_reflectivity(maps, x, face):
    """Downsample the physical high-pass log impedance without per-image scaling."""
    i0 = (len(x) - 768) // 2
    c = maps["sound_speed"][face:face+864, i0:i0+768].astype(np.float64)
    rho = maps["density"][face:face+864, i0:i0+768].astype(np.float64)
    log_impedance = np.log(np.maximum(c*rho, 1.0))
    highpass = log_impedance - gaussian_filter(log_impedance, 5.0, mode="nearest")
    return highpass.reshape(216, 4, 192, 4).mean((1, 3)), i0


def preview(root, output, ids):
    index = json.loads((root / "index.json").read_text())
    records = {r["id"]: r for r in index["samples"]}
    x, z = medium_builder.geometry(depth_mm=45.0, lateral_mm=28.0)
    rows, stats = [], []
    for sid in ids:
        rec = records[sid]
        with h5py.File(rec["h5"], "r") as f:
            plane = np.asarray(f["phan"][rec["z_index"]])
        maps, codes, _, *_ = medium_builder.build_medium(
            plane, x, z, seed=rec["scatter_seed"], preset="dual_scale")
        augmented, mask, circles = add_echogenic_circles(
            maps, codes, x, z, seed=rec["scatter_seed"]+404)
        repeat, repeat_mask, repeat_circles = add_echogenic_circles(
            maps, codes, x, z, seed=rec["scatter_seed"]+404)
        assert np.array_equal(repeat["density"], augmented["density"])
        assert np.array_equal(repeat_mask, mask) and repeat_circles == circles
        assert all(np.array_equal(augmented[k], maps[k]) for k in
                   ("sound_speed", "alpha_coeff", "BonA"))
        assert np.count_nonzero(augmented["density"] != maps["density"]) > 0
        before, i0 = raw_reflectivity(maps, x, medium_builder.FACE)
        after, _ = raw_reflectivity(augmented, x, medium_builder.FACE)
        coarse_mask = mask[medium_builder.FACE:medium_builder.FACE+864,
                           i0:i0+768].reshape(216, 4, 192, 4).mean((1, 3)) > 0.5
        rms_before = np.sqrt(gaussian_filter(before**2, 2.5))
        rms_after = np.sqrt(gaussian_filter(after**2, 2.5))
        inside_before = float(np.sqrt(np.mean(before[coarse_mask]**2)))
        inside_after = float(np.sqrt(np.mean(after[coarse_mask]**2)))
        stat = dict(id=sid, case=rec["case"], z_index=rec["z_index"],
                    circles=circles, inside_rms_before=inside_before,
                    inside_rms_after=inside_after,
                    inside_power_gain_db=float(20*np.log10(inside_after/inside_before)),
                    sound_speed_max_abs_diff=0.0,
                    density_changed_pixels=int(np.count_nonzero(augmented["density"] != maps["density"])))
        stats.append(stat)
        rows.append((sid, maps, rms_before, rms_after, circles))
        print(json.dumps(stat), flush=True)

    all_rms = np.concatenate([v.ravel() for _, _, a, b, _ in rows for v in (a, b)])
    floor = max(float(np.median(all_rms)), 1e-12)
    extent = (x[i0]*1e3, x[i0+767]*1e3,
              z[medium_builder.FACE+863]*1e3, z[medium_builder.FACE]*1e3)
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 3.1*len(rows)),
                             sharex=True, sharey=True, constrained_layout=True)
    if len(rows) == 1:
        axes = axes[None, :]
    for row, (sid, maps, before, after, circles) in enumerate(rows):
        c = maps["sound_speed"][medium_builder.FACE:medium_builder.FACE+864,
                                i0:i0+768].reshape(216, 4, 192, 4).mean((1, 3))
        axes[row, 0].imshow(c, extent=extent, cmap="turbo", vmin=1450, vmax=1600)
        for col, img in ((1, before), (2, after)):
            db = 20*np.log10(np.maximum(img/floor, 1e-6))
            axes[row, col].imshow(db, extent=extent, cmap="gray", vmin=-14, vmax=15)
            for circle in circles:
                axes[row, col].add_patch(Circle(
                    (circle["x_mm"], circle["z_mm"]), circle["radius_mm"],
                    fill=False, edgecolor="cyan", linewidth=1.1))
        axes[row, 0].set_ylabel(f"{sid}\nDepth (mm)")
    for col, title in enumerate(("Sound speed (m/s)", "Before: echo proxy (dB)",
                                 "After: echo proxy (dB)")):
        axes[0, col].set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("Lateral (mm)")
    fig.suptitle("Circular high-echo phantom preview | NOT simulated RF or B-mode", fontsize=13)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "high_echo_circles_preview.png", dpi=160)
    plt.close(fig)
    (output / "preview_metrics.json").write_text(json.dumps(stats, indent=2)+"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ids", nargs="+", default=["train_000", "train_200", "val_000", "test_000"])
    args = ap.parse_args()
    preview(args.root, args.output, args.ids)


if __name__ == "__main__":
    main()

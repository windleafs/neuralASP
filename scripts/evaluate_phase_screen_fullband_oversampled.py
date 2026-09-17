"""Full-band phase-screen evaluation using the configured oversampled ASP grid.

This is the main-chain counterpart of ``evaluate_phase_screen_fullband.py``.
It reuses that script's plotting/metrics/CLI, but replaces its legacy direct
BornModel construction and manual fine-grid adjoint with the same
LateralOversampledBornModel used by PhaseScreenModel.

For configs/l11_ultrawave_500_11angle.yaml the parameter grid remains 0.2 mm
while Tx/Rx propagation runs at 0.1 mm laterally.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import scripts.evaluate_phase_screen_fullband as fullband  # noqa: E402
import scripts.visualize_phase_screen_checkpoint as vis  # noqa: E402
from physics.oversampled_imaging import LateralOversampledBornModel  # noqa: E402
from scripts.pilot_phase_asp import corrected_config, padded_meta  # noqa: E402


def build_imaging_operator(saved_args, first_sample, pad, imaging_n_freq, device):
    cfg, meta = corrected_config(
        saved_args["config"], first_sample, imaging_n_freq)
    factor = int(cfg.physics.get("lateral_oversample", 1))
    born = LateralOversampledBornModel(
        padded_meta(meta, pad, cfg.grid.dx),
        cfg.grid.nx + 2 * pad,
        cfg.grid.nz,
        cfg.grid.dx,
        cfg.grid.dz,
        cfg.physics.c0,
        eps=cfg.physics.eps_evanescent,
        spreading=cfg.physics.spreading,
        lateral_oversample=factor,
    ).to(device)
    return cfg, meta, born


def angle_images(born, ds, D, idx):
    """Oversampled-aware per-angle adjoint returned on the parameter grid."""
    u = born.transmit_fields(ds, idx)
    return born.adjoint_per_angle(D[:, idx], u, ds)


def main():
    # ``evaluate_phase_screen_fullband`` imported several helper functions from
    # ``visualize_phase_screen_checkpoint``.  Functions such as fixed_reference
    # retain the *visualize module's* globals, so patching fullband.angle_images
    # alone is insufficient: fixed_reference would still resolve the legacy
    # visualize.angle_images and call born.asp.adjoint directly with a coarse
    # 0.2 mm ds map against the 0.1 mm propagation carrier (512 vs 256).
    # Patch both module namespaces so every direct and nested helper uses the
    # oversampled operator consistently.
    fullband.build_imaging_operator = build_imaging_operator
    fullband.angle_images = angle_images
    vis.build_imaging_operator = build_imaging_operator
    vis.angle_images = angle_images
    fullband.main()


if __name__ == "__main__":
    main()

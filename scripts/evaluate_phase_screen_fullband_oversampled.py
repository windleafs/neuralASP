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
    u = born.transmit_fields(ds, idx)
    return born.adjoint_per_angle(D[:, idx], u, ds)


def main():
    # evaluate_phase_screen_fullband imported these symbols into its module
    # namespace at import time; replacing them here keeps every downstream
    # reference/mask/score/figure on the same oversampled operator.
    fullband.build_imaging_operator = build_imaging_operator
    fullband.angle_images = angle_images
    fullband.main()


if __name__ == "__main__":
    main()

"""Oversampled-aware visual evaluation for phase-screen checkpoints.

This wrapper keeps the existing visualization CLI/plots but replaces the
legacy manual Born/ASP imaging path with the same lateral-oversampled operator
used by the main chain.  It avoids the 512-vs-256 mismatch caused by calling
``born.asp.adjoint`` directly with a 0.2 mm parameter-grid slowness map after
ASP propagation moved to a 0.1 mm lateral grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

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
    u = born.transmit_fields(ds, idx)
    return born.adjoint_per_angle(D[:, idx], u, ds)


def main():
    # The legacy module imported these helpers into its own namespace, so
    # replace them before entering its existing main()/plotting flow.
    vis.build_imaging_operator = build_imaging_operator
    vis.angle_images = angle_images
    vis.main()


if __name__ == "__main__":
    main()

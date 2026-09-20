"""Candidate low-dimensional propagation modes for sensitivity analysis.

The goal is not to parameterize the full medium. These helpers construct a
small dictionary of smooth lateral screen modes and inject them at selected
propagation depths so that their effect on the final image can be measured.

The default lateral dictionary is a DCT-like basis on the physical aperture:
  k=0 : lateral piston / common mode
  k=1 : first smooth lateral variation
  k=2 : second smooth lateral variation
  ...

Each mode is normalized to unit RMS on the physical (unpadded) aperture.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "lateral_dct_modes",
    "select_depth_screen_indices",
    "phase_screen_perturbation",
    "amplitude_screen_perturbation",
]


def lateral_dct_modes(nx: int, pad: int, n_modes: int, *,
                      device=None, dtype=torch.float32) -> torch.Tensor:
    """Return [K,nx] smooth DCT-like modes with unit physical-aperture RMS."""
    if nx < 2:
        raise ValueError("nx must be at least 2")
    if pad < 0 or 2 * pad >= nx:
        raise ValueError("pad must leave a non-empty physical aperture")
    if n_modes < 1:
        raise ValueError("n_modes must be positive")

    n_phys = nx - 2 * pad
    j = torch.arange(n_phys, device=device, dtype=dtype)
    # DCT-II coordinate. k=0 is exactly constant; k>0 have zero discrete mean.
    x = (j + 0.5) / float(n_phys)
    modes_phys = []
    for k in range(n_modes):
        v = torch.cos(math.pi * float(k) * x)
        rms = v.square().mean().sqrt().clamp_min(1e-12)
        modes_phys.append(v / rms)
    physical = torch.stack(modes_phys, dim=0)

    if pad == 0:
        return physical

    out = torch.empty(n_modes, nx, device=device, dtype=dtype)
    out[:, pad:nx - pad] = physical
    # Continue smoothly into numerical padding with edge values.
    out[:, :pad] = physical[:, :1]
    out[:, nx - pad:] = physical[:, -1:]
    return out


def select_depth_screen_indices(nz: int, z0_m: float, dz_m: float,
                                count: int,
                                min_depth_mm: float | None = None,
                                max_depth_mm: float | None = None):
    """Select approximately uniform valid ASP slab indices and depths in mm."""
    if nz < 2:
        raise ValueError("nz must contain propagation slabs")
    if dz_m <= 0:
        raise ValueError("dz_m must be positive")
    if count < 1:
        raise ValueError("count must be positive")

    first_mm = (z0_m + dz_m) * 1e3
    last_mm = (z0_m + (nz - 2) * dz_m) * 1e3
    lo = first_mm if min_depth_mm is None else max(first_mm, min_depth_mm)
    hi = last_mm if max_depth_mm is None else min(last_mm, max_depth_mm)
    if hi < lo:
        raise ValueError("requested depth interval does not overlap the grid")

    target = torch.linspace(lo, hi, count, dtype=torch.float64)
    idx = torch.round(
        (target * 1e-3 - float(z0_m)) / float(dz_m)
    ).long().clamp(0, nz - 2)
    idx = torch.unique_consecutive(idx)
    depths_mm = (
        float(z0_m) + idx.to(torch.float64) * float(dz_m)
    ) * 1e3
    return idx.tolist(), depths_mm.tolist()


def phase_screen_perturbation(mode: torch.Tensor, nz: int, depth_index: int,
                              dz_m: float, phase_step_us: float) -> torch.Tensor:
    """Embed one integrated-delay perturbation as a slowness screen [s/m]."""
    if mode.ndim != 1:
        raise ValueError("mode must have shape [nx]")
    if not (0 <= depth_index < nz - 1):
        raise ValueError("depth_index must address a propagation slab")
    if dz_m <= 0 or phase_step_us <= 0:
        raise ValueError("dz_m and phase_step_us must be positive")

    out = torch.zeros(
        nz, mode.numel(), device=mode.device, dtype=mode.dtype)
    out[depth_index] = mode * (phase_step_us * 1e-6 / dz_m)
    return out


def amplitude_screen_perturbation(mode: torch.Tensor, nz: int,
                                  depth_index: int, dz_m: float,
                                  amplitude_step_np: float) -> torch.Tensor:
    """Embed one integrated log-amplitude perturbation as a rate [Np/m]."""
    if mode.ndim != 1:
        raise ValueError("mode must have shape [nx]")
    if not (0 <= depth_index < nz - 1):
        raise ValueError("depth_index must address a propagation slab")
    if dz_m <= 0 or amplitude_step_np <= 0:
        raise ValueError("dz_m and amplitude_step_np must be positive")

    out = torch.zeros(
        nz, mode.numel(), device=mode.device, dtype=mode.dtype)
    out[depth_index] = mode * (amplitude_step_np / dz_m)
    return out

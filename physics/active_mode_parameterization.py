"""Fixed population active-mode basis for low-dimensional propagation correction.

The basis is learned once from the task-oriented sensitivity experiment and is
then treated as a fixed physical parameterization. Network outputs are only the
coefficients along the leading population active modes.
"""
from __future__ import annotations

from pathlib import Path

import torch

from physics.active_subspace import gram_spectrum
from physics.propagation_modes import (
    amplitude_screen_perturbation,
    lateral_dct_modes,
    phase_screen_perturbation,
)

__all__ = [
    "load_shared_active_basis",
    "build_active_mode_templates",
]


def load_shared_active_basis(path, rank: int = 6, source: str = "balanced"):
    """Load leading parameter-space active modes from a population artifact.

    source:
      phase      - use the population phase eigenspace directly
      amplitude  - use the population amplitude eigenspace directly
      balanced   - eigendecompose the trace-normalized mean of G_phase/G_amp

    The balanced option removes arbitrary unit scaling between phase [us] and
    amplitude [Np] before defining one shared spatial subspace.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if rank < 1:
        raise ValueError("rank must be positive")

    Gp = torch.as_tensor(payload["Gp"], dtype=torch.float32)
    Ga = torch.as_tensor(payload["Ga"], dtype=torch.float32)
    if Gp.ndim != 2 or Gp.shape != Ga.shape or Gp.shape[0] != Gp.shape[1]:
        raise ValueError("population Gram matrices must be matching square tensors")
    if rank > Gp.shape[0]:
        raise ValueError("rank exceeds candidate parameter dimension")

    if source == "phase":
        Vh = torch.as_tensor(payload["phase_Vh"], dtype=torch.float32)
    elif source == "amplitude":
        Vh = torch.as_tensor(payload["amplitude_Vh"], dtype=torch.float32)
    elif source == "balanced":
        Gp_n = Gp / torch.trace(Gp).clamp_min(1e-30)
        Ga_n = Ga / torch.trace(Ga).clamp_min(1e-30)
        Vh = gram_spectrum(0.5 * (Gp_n + Ga_n))["Vh"].to(torch.float32)
    else:
        raise ValueError("source must be phase, amplitude or balanced")

    depths_mm = [float(v) for v in payload["depths_mm"]]
    labels_phase = list(payload.get("labels_phase", []))
    n_depth = len(depths_mm)
    if n_depth < 1 or Gp.shape[0] % n_depth:
        raise ValueError("candidate dimension is incompatible with stored depths")
    n_lateral_modes = Gp.shape[0] // n_depth

    # The sensitivity scripts enumerate depth-major, lateral-mode-minor.
    if labels_phase and len(labels_phase) != Gp.shape[0]:
        raise ValueError("stored labels do not match candidate dimension")

    return {
        "Vh": Vh[:rank].contiguous(),
        "depths_mm": depths_mm,
        "n_lateral_modes": int(n_lateral_modes),
        "candidate_dim": int(Gp.shape[0]),
        "source": source,
        "path": str(path),
    }


def build_active_mode_templates(basis, *, nz: int, nx: int, dz_m: float,
                                z0_m: float, pad: int, device=None,
                                dtype=torch.float32):
    """Convert parameter-space eigenvectors into phase/amplitude screen templates.

    Returns two tensors [K,nz,nx]:
      phase_unit_ds    : slowness perturbation [s/m] per 1 us coefficient
      amplitude_unit_rate : log-amplitude rate [Np/m] per 1 Np coefficient
    """
    Vh = torch.as_tensor(basis["Vh"], device=device, dtype=dtype)
    depths_mm = basis["depths_mm"]
    n_lat = int(basis["n_lateral_modes"])
    if Vh.shape[1] != len(depths_mm) * n_lat:
        raise ValueError("basis shape does not match depth/lateral candidate grid")

    lateral = lateral_dct_modes(
        nx, pad, n_lat, device=device, dtype=dtype)
    phase_candidates = []
    amp_candidates = []
    depth_indices = []
    for z_mm in depths_mm:
        zi = int(round((z_mm * 1e-3 - float(z0_m)) / float(dz_m)))
        zi = max(0, min(nz - 2, zi))
        depth_indices.append(zi)
        for k in range(n_lat):
            phase_candidates.append(phase_screen_perturbation(
                lateral[k], nz, zi, dz_m, 1.0))
            amp_candidates.append(amplitude_screen_perturbation(
                lateral[k], nz, zi, dz_m, 1.0))

    P = torch.stack(phase_candidates, dim=0)
    A = torch.stack(amp_candidates, dim=0)
    phase_templates = torch.einsum("kp,pzx->kzx", Vh, P)
    amp_templates = torch.einsum("kp,pzx->kzx", Vh, A)
    return {
        "phase_unit_ds": phase_templates,
        "amplitude_unit_rate": amp_templates,
        "depth_indices": depth_indices,
    }

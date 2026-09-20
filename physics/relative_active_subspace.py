"""Utilities for removing lateral common/mean modes from active subspaces.

The candidate parameterization is depth-major and lateral-mode-minor.  The
lateral DCT k=0 basis vector is the common (constant-across-aperture) mode at
each candidate depth.  Removing all k=0 coordinates makes the residual active
space orthogonal to the explicit mean-delay branch on the physical aperture.
"""
from __future__ import annotations

import torch

__all__ = [
    "common_mode_indices",
    "relative_mode_indices",
    "project_gram_remove_common",
    "relative_energy_fraction",
]


def common_mode_indices(n_depth: int, n_lateral_modes: int):
    if n_depth < 1 or n_lateral_modes < 1:
        raise ValueError("mode counts must be positive")
    return [d * n_lateral_modes for d in range(n_depth)]


def relative_mode_indices(n_depth: int, n_lateral_modes: int):
    common = set(common_mode_indices(n_depth, n_lateral_modes))
    return [i for i in range(n_depth * n_lateral_modes) if i not in common]


def project_gram_remove_common(G: torch.Tensor, n_depth: int,
                               n_lateral_modes: int) -> torch.Tensor:
    """Return P_perp^T G P_perp by zeroing all lateral k=0 coordinates."""
    n = n_depth * n_lateral_modes
    if G.ndim != 2 or G.shape != (n, n):
        raise ValueError(f"expected Gram shape {(n, n)}, got {tuple(G.shape)}")
    keep = torch.ones(n, dtype=G.dtype, device=G.device)
    keep[common_mode_indices(n_depth, n_lateral_modes)] = 0
    return G * keep[:, None] * keep[None, :]


def relative_energy_fraction(G: torch.Tensor, G_relative: torch.Tensor) -> float:
    """Fraction of trace sensitivity energy retained after common-mode removal."""
    total = torch.trace(G).real
    relative = torch.trace(G_relative).real
    if float(total.abs()) <= 1e-30:
        return 0.0
    return float((relative / total).clamp(min=0.0, max=1.0))

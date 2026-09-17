"""Utilities for discrete multilayer phase-screen parameterizations.

A control curve represents an integrated delay tau_l(x) [us].  To reuse the
existing split-step ASP interface, each curve is embedded in exactly one
propagation slab as

    delta_s(z_l, x) = tau_l(x) / dz,

so the existing screen exp(i * omega * delta_s * dz) is exactly
exp(i * omega * tau_l).  This is deliberately different from spreading the
same integrated delay uniformly through a depth block, which is a coarse
slowness model rather than a discrete phase screen.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["phase_control_curves_us", "controls_to_discrete_ds"]


def phase_control_curves_us(raw: torch.Tensor, nx: int, limit_us: float,
                            pad: int = 0) -> torch.Tensor:
    """Interpolate bounded controls and fix the lateral piston gauge.

    Parameters
    ----------
    raw:
        ``[L, K]`` or ``[B, L, K]`` unconstrained controls.
    nx:
        Fine-grid lateral size, including optional padding.
    limit_us:
        Maximum absolute delay after the safety rescaling.
    pad:
        Number of lateral padding samples on each side.  The zero-mean gauge
        is computed only over the physical field of view.
    """
    if raw.ndim not in (2, 3):
        raise ValueError("raw controls must have shape [L,K] or [B,L,K]")
    if nx <= 0:
        raise ValueError("nx must be positive")
    if pad < 0 or 2 * pad >= nx:
        raise ValueError("pad must leave a non-empty physical aperture")

    squeeze = raw.ndim == 2
    x = raw.unsqueeze(0) if squeeze else raw
    curves = F.interpolate(limit_us * torch.tanh(x), size=nx,
                           mode="linear", align_corners=True)
    physical = curves[..., pad:nx - pad] if pad else curves
    curves = curves - physical.mean(dim=-1, keepdim=True)

    # Mean subtraction can increase the absolute peak beyond limit_us.
    scale = torch.maximum(
        curves.abs().amax(dim=-1, keepdim=True) / limit_us,
        torch.ones_like(curves[..., :1]),
    )
    curves = curves / scale
    return curves[0] if squeeze else curves


def controls_to_discrete_ds(raw: torch.Tensor, nz: int, nx: int, dz: float,
                            limit_us: float = 0.2, pad: int = 0) -> torch.Tensor:
    """Embed integrated-delay controls as discrete ASP phase screens.

    The returned tensor has shape ``[nz,nx]`` or ``[B,nz,nx]`` and a zero
    final row because the ASP uses rows ``0..nz-2`` as propagation slabs.
    Each depth region contributes one non-zero slab located near its centre.
    """
    if nz < 2:
        raise ValueError("nz must contain at least one propagation slab")
    if dz <= 0:
        raise ValueError("dz must be positive")

    squeeze = raw.ndim == 2
    x = raw.unsqueeze(0) if squeeze else raw
    if x.ndim != 3:
        raise ValueError("raw controls must have shape [L,K] or [B,L,K]")

    B, layers, _ = x.shape
    curves = phase_control_curves_us(x, nx, limit_us, pad)
    edges = torch.round(torch.linspace(0, nz - 1, layers + 1,
                                       device=x.device)).long().tolist()
    if any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError("more phase-screen layers than propagation slabs")

    ds = torch.zeros(B, nz, nx, device=x.device, dtype=x.dtype)
    for layer, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        # Propagation slabs are indexed [a, b).  Put one screen at the centre
        # of that interval; ASP applies it between two homogeneous half-steps.
        zi = (a + b - 1) // 2
        ds[:, zi, :] += curves[:, layer, :] * (1e-6 / dz)

    return ds[0] if squeeze else ds

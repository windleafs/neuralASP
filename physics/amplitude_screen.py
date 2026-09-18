"""Low-dimensional integrated log-amplitude screen utilities.

Each amplitude control curve A_l(x) is an integrated signed log-amplitude
correction [Np] applied at one propagation slab:

    S_A,l(x,f) = exp[-A_l(x) (f/f0)^gamma].

Positive A attenuates relative to the reference propagation model; negative A
provides bounded compensating gain.  The controls describe a *relative*
propagation-amplitude correction, not an absolute tissue attenuation map.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "amplitude_control_curves_np",
    "controls_to_discrete_amplitude_rate",
]


def amplitude_control_curves_np(raw: torch.Tensor, nx: int, limit_np: float,
                                pad: int = 0) -> torch.Tensor:
    """Interpolate bounded low-dimensional amplitude controls to x."""
    if raw.ndim not in (2, 3):
        raise ValueError("raw controls must have shape [L,K] or [B,L,K]")
    if nx <= 0:
        raise ValueError("nx must be positive")
    if limit_np <= 0:
        raise ValueError("limit_np must be positive")
    if pad < 0 or 2 * pad >= nx:
        raise ValueError("pad must leave a non-empty physical aperture")

    squeeze = raw.ndim == 2
    x = raw.unsqueeze(0) if squeeze else raw
    curves = F.interpolate(
        limit_np * torch.tanh(x), size=nx,
        mode="linear", align_corners=True)

    # Do not remove the lateral mean: attenuation/transport can have a genuine
    # common-mode component.  Only bound interpolation overshoot numerically.
    scale = torch.maximum(
        curves.abs().amax(dim=-1, keepdim=True) / limit_np,
        torch.ones_like(curves[..., :1]),
    )
    curves = curves / scale
    return curves[0] if squeeze else curves


def controls_to_discrete_amplitude_rate(raw: torch.Tensor, nz: int, nx: int,
                                        dz: float, limit_np: float = 0.5,
                                        pad: int = 0) -> torch.Tensor:
    """Embed integrated log-amplitude controls as discrete Np/m screens."""
    if nz < 2:
        raise ValueError("nz must contain at least one propagation slab")
    if dz <= 0:
        raise ValueError("dz must be positive")

    squeeze = raw.ndim == 2
    x = raw.unsqueeze(0) if squeeze else raw
    if x.ndim != 3:
        raise ValueError("raw controls must have shape [L,K] or [B,L,K]")

    B, layers, _ = x.shape
    curves = amplitude_control_curves_np(x, nx, limit_np, pad)
    edges = torch.round(torch.linspace(
        0, nz - 1, layers + 1, device=x.device)).long().tolist()
    if any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError("more amplitude-screen layers than propagation slabs")

    rate = torch.zeros(B, nz, nx, device=x.device, dtype=x.dtype)
    for layer, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        zi = (a + b - 1) // 2
        rate[:, zi, :] += curves[:, layer, :] / dz

    return rate[0] if squeeze else rate

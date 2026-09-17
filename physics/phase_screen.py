"""Utilities for discrete multilayer phase-screen parameterizations.

A relative-screen control curve represents an integrated delay tau_l(x) [us].
Each curve is embedded in exactly one propagation slab so the existing ASP
screen ``exp(i * omega * delta_s * dz)`` is exactly ``exp(i * omega * tau_l)``.

V3 also supports a lateral-mean propagation branch. That branch predicts a
small number of cumulative mean-delay controls G_k [us] at positive depths.
G(0)=0 is fixed, the controls are interpolated along depth, and finite
differences convert G(z) back to the mean slowness perturbation required by the
ASP.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "phase_control_curves_us",
    "controls_to_discrete_ds",
    "mean_delay_profile_us",
    "mean_controls_to_ds",
    "project_mean_delay_controls",
]


def phase_control_curves_us(raw: torch.Tensor, nx: int, limit_us: float,
                            pad: int = 0) -> torch.Tensor:
    """Interpolate bounded relative-screen controls and fix lateral piston gauge."""
    if raw.ndim not in (2, 3):
        raise ValueError("raw controls must have shape [L,K] or [B,L,K]")
    if nx <= 0:
        raise ValueError("nx must be positive")
    if limit_us <= 0:
        raise ValueError("limit_us must be positive")
    if pad < 0 or 2 * pad >= nx:
        raise ValueError("pad must leave a non-empty physical aperture")

    squeeze = raw.ndim == 2
    x = raw.unsqueeze(0) if squeeze else raw
    curves = F.interpolate(limit_us * torch.tanh(x), size=nx,
                           mode="linear", align_corners=True)
    physical = curves[..., pad:nx - pad] if pad else curves
    curves = curves - physical.mean(dim=-1, keepdim=True)

    scale = torch.maximum(
        curves.abs().amax(dim=-1, keepdim=True) / limit_us,
        torch.ones_like(curves[..., :1]),
    )
    curves = curves / scale
    return curves[0] if squeeze else curves


def controls_to_discrete_ds(raw: torch.Tensor, nz: int, nx: int, dz: float,
                            limit_us: float = 0.2, pad: int = 0) -> torch.Tensor:
    """Embed relative integrated-delay controls as discrete ASP phase screens."""
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
        zi = (a + b - 1) // 2
        ds[:, zi, :] += curves[:, layer, :] * (1e-6 / dz)

    return ds[0] if squeeze else ds


def mean_delay_profile_us(raw: torch.Tensor, nz: int,
                          limit_us: float = 2.0) -> torch.Tensor:
    """Expand cumulative mean-delay controls to a depth profile G(z) [us]."""
    if raw.ndim not in (1, 2):
        raise ValueError("mean raw controls must have shape [K] or [B,K]")
    if raw.shape[-1] < 1:
        raise ValueError("at least one mean-delay control is required")
    if nz < 2:
        raise ValueError("nz must be at least 2")
    if limit_us <= 0:
        raise ValueError("limit_us must be positive")

    squeeze = raw.ndim == 1
    x = raw.unsqueeze(0) if squeeze else raw
    bounded = limit_us * torch.tanh(x)
    zero = torch.zeros(bounded.shape[0], 1, device=x.device, dtype=x.dtype)
    knots = torch.cat([zero, bounded], dim=-1)
    profile = F.interpolate(knots[:, None, :], size=nz,
                            mode="linear", align_corners=True)[:, 0]
    return profile[0] if squeeze else profile


def mean_controls_to_ds(raw: torch.Tensor, nz: int, nx: int, dz: float,
                        limit_us: float = 2.0) -> torch.Tensor:
    """Convert cumulative mean-delay controls to an ASP slowness carrier."""
    if nx <= 0:
        raise ValueError("nx must be positive")
    if dz <= 0:
        raise ValueError("dz must be positive")

    squeeze = raw.ndim == 1
    x = raw.unsqueeze(0) if squeeze else raw
    if x.ndim != 2:
        raise ValueError("mean raw controls must have shape [K] or [B,K]")

    G = mean_delay_profile_us(x, nz, limit_us)
    step_ds = (G[:, 1:] - G[:, :-1]) * (1e-6 / dz)
    ds = torch.zeros(x.shape[0], nz, nx, device=x.device, dtype=x.dtype)
    ds[:, :-1, :] = step_ds[:, :, None]
    return ds[0] if squeeze else ds


def project_mean_delay_controls(true_ds: torch.Tensor, n_controls: int,
                                dz: float, pad: int = 0,
                                limit_us: float = 2.0):
    """Project a known medium onto K cumulative mean-delay controls."""
    if true_ds.ndim != 2:
        raise ValueError("true_ds must have shape [nz,nx]")
    if n_controls < 1:
        raise ValueError("n_controls must be positive")
    if dz <= 0:
        raise ValueError("dz must be positive")
    if limit_us <= 0:
        raise ValueError("limit_us must be positive")
    nz, nx = true_ds.shape
    if nz < 2:
        raise ValueError("true_ds must contain propagation rows")
    if pad < 0 or 2 * pad >= nx:
        raise ValueError("pad must leave a non-empty physical aperture")

    physical = true_ds[:-1, pad:nx - pad] if pad else true_ds[:-1]
    mean_ds = physical.mean(dim=-1)
    cumulative = torch.cat([
        torch.zeros_like(mean_ds[:1]),
        torch.cumsum(mean_ds * dz * 1e6, dim=0),
    ])
    sampled = F.interpolate(cumulative[None, None, :],
                            size=n_controls + 1,
                            mode="linear", align_corners=True)[0, 0, 1:]
    saturated = (sampled.abs() > limit_us).float().mean()
    clipped = sampled.clamp(-0.999 * limit_us, 0.999 * limit_us)
    raw = torch.atanh(clipped / limit_us)
    return raw, sampled, {
        "target_max_abs_us": float(sampled.abs().max()),
        "saturated_control_fraction": float(saturated),
        "end_delay_us": float(cumulative[-1]),
        "n_controls": int(n_controls),
    }

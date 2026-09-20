"""V7 amplitude-aware low-dimensional coefficient predictor.

The phase path is frozen upstream.  V7 consumes the mean-phase-corrected
incoherent envelope and predicts K amplitude active-mode coefficients from a
compact descriptor:

    log envelope -> 8 depth bins x 6 lateral DCT modes = 48 features
    48 -> 32 -> K

The descriptor uses one training-set global amplitude scale, never per-sample
normalization, so the lateral k=0/common-amplitude information is preserved.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics.propagation_modes import lateral_dct_modes

__all__ = ["V7AmplitudePredictor"]


class V7AmplitudePredictor(nn.Module):
    def __init__(self, active_rank: int = 6, depth_bins: int = 8,
                 lateral_modes: int = 6, hidden: int = 32,
                 coeff_limit_np: float = 0.5, descriptor_eps: float = 1e-5):
        super().__init__()
        if active_rank < 1 or depth_bins < 1 or lateral_modes < 1 or hidden < 1:
            raise ValueError("rank/bin/mode/hidden sizes must be positive")
        if coeff_limit_np <= 0 or descriptor_eps <= 0:
            raise ValueError("coefficient limit and descriptor eps must be positive")
        self.active_rank = int(active_rank)
        self.depth_bins = int(depth_bins)
        self.lateral_modes = int(lateral_modes)
        self.hidden = int(hidden)
        self.coeff_limit_np = float(coeff_limit_np)
        self.descriptor_eps = float(descriptor_eps)
        dim = self.depth_bins * self.lateral_modes
        self.descriptor_dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.active_rank),
        )
        # Start exactly at the phase-only baseline.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("descriptor_scale", torch.tensor(1.0))
        self.register_buffer("descriptor_mean", torch.zeros(dim))
        self.register_buffer("descriptor_std", torch.ones(dim))

    @torch.no_grad()
    def set_descriptor_stats(self, scale: float, mean: torch.Tensor,
                             std: torch.Tensor):
        if scale <= 0:
            raise ValueError("descriptor scale must be positive")
        if mean.shape != (self.descriptor_dim,) or std.shape != (self.descriptor_dim,):
            raise ValueError("descriptor statistic shape mismatch")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("non-finite descriptor statistics")
        self.descriptor_scale.fill_(float(scale))
        self.descriptor_mean.copy_(mean.to(self.descriptor_mean))
        self.descriptor_std.copy_(std.to(self.descriptor_std).clamp_min(1e-4))

    def raw_descriptor(self, envelope: torch.Tensor) -> torch.Tensor:
        """Convert physical-aperture envelope [B,Z,X] to [B,depth_bins*modes]."""
        if envelope.ndim != 3:
            raise ValueError("expected envelope [B,Z,X]")
        _, _, nx = envelope.shape
        scale = self.descriptor_scale.to(envelope)
        log_env = torch.log(envelope / scale + self.descriptor_eps)
        pooled = F.adaptive_avg_pool2d(
            log_env[:, None], (self.depth_bins, nx))[:, 0]
        modes = lateral_dct_modes(
            nx, 0, self.lateral_modes, device=envelope.device, dtype=envelope.dtype)
        coeff = torch.einsum("bdx,kx->bdk", pooled, modes) / float(nx)
        return coeff.reshape(envelope.shape[0], -1)

    def standardized_descriptor(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.shape[-1] != self.descriptor_dim:
            raise ValueError("descriptor dimension mismatch")
        return (raw - self.descriptor_mean.to(raw)) / self.descriptor_std.to(raw)

    def forward_descriptor(self, raw: torch.Tensor):
        z = self.standardized_descriptor(raw)
        raw_coeff = self.mlp(z)
        coeff = self.coeff_limit_np * torch.tanh(raw_coeff)
        return coeff, raw_coeff

    def forward(self, envelope: torch.Tensor):
        raw = self.raw_descriptor(envelope)
        coeff, raw_coeff = self.forward_descriptor(raw)
        return {
            "descriptor_raw": raw,
            "descriptor_standardized": self.standardized_descriptor(raw),
            "raw_coeff": raw_coeff,
            "coeff_np": coeff,
        }

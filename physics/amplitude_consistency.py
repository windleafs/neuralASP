"""Amplitude-pattern consistency metrics for cross-angle imaging.

The metric is deliberately insensitive to a per-angle global gain and to
high-frequency speckle fluctuations.  Each complex angle image is converted to
a smoothed log-envelope map, normalized by its masked RMS, and centered by its
masked spatial mean.  The loss then compares the mean context and target
amplitude patterns over the fixed ROI mask.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "centered_log_envelope",
    "amplitude_pattern_consistency",
]


def _check_kernel(kernel: int):
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("smooth_kernel must be a positive odd integer")


def centered_log_envelope(images: torch.Tensor, mask: torch.Tensor,
                          smooth_kernel: int = 9,
                          eps: float = 1e-4) -> torch.Tensor:
    """Return centered, smoothed log-envelope maps.

    Parameters
    ----------
    images:
        Complex tensor [B, angle, z, x].
    mask:
        Fixed real ROI mask [B, z, x].  It must be derived independently of
        the corrected images (the training code uses the Uniform reference).
    smooth_kernel:
        Odd spatial averaging kernel.  This suppresses angle-dependent speckle
        so the metric focuses on larger-scale amplitude transport.
    eps:
        Floor after per-angle masked-RMS normalization.

    Notes
    -----
    A per-angle multiplicative gain cancels because the image is first
    normalized by its masked RMS and then spatially centered in log amplitude.
    """
    _check_kernel(int(smooth_kernel))
    if eps <= 0:
        raise ValueError("eps must be positive")
    if images.ndim != 4:
        raise ValueError("images must have shape [B, angle, z, x]")
    if mask.ndim != 3 or mask.shape[0] != images.shape[0]:
        raise ValueError("mask must have shape [B, z, x]")
    if tuple(mask.shape[-2:]) != tuple(images.shape[-2:]):
        raise ValueError("mask/image spatial shapes must match")

    weight = mask.to(images.real.dtype)[:, None]
    area = weight.sum(dim=(-2, -1)).clamp_min(1.0)

    envelope = images.abs()
    rms = (
        (envelope.square() * weight).sum(dim=(-2, -1))
        / area
    ).sqrt().clamp_min(1e-12)
    envelope = envelope / rms[..., None, None]

    if smooth_kernel > 1:
        B, A, Z, X = envelope.shape
        p = smooth_kernel // 2
        flat = envelope.reshape(B * A, 1, Z, X)
        flat = F.pad(flat, (p, p, p, p), mode="replicate")
        flat = F.avg_pool2d(
            flat, kernel_size=smooth_kernel, stride=1)
        envelope = flat.reshape(B, A, Z, X)

    log_env = torch.log(envelope.clamp_min(float(eps)))
    spatial_mean = (
        (log_env * weight).sum(dim=(-2, -1))
        / area
    )
    centered = log_env - spatial_mean[..., None, None]
    return centered * weight


def amplitude_pattern_consistency(context_images: torch.Tensor,
                                  target_images: torch.Tensor,
                                  mask: torch.Tensor,
                                  smooth_kernel: int = 9,
                                  eps: float = 1e-4) -> torch.Tensor:
    """Cross-angle amplitude-pattern mismatch; lower is better.

    The returned scalar is the masked MSE between the mean centered
    log-envelope pattern of context and target angles.
    """
    if context_images.shape[0] != target_images.shape[0]:
        raise ValueError("context and target batch sizes must match")
    if tuple(context_images.shape[-2:]) != tuple(target_images.shape[-2:]):
        raise ValueError("context and target spatial shapes must match")

    ctx = centered_log_envelope(
        context_images, mask, smooth_kernel=smooth_kernel, eps=eps)
    tgt = centered_log_envelope(
        target_images, mask, smooth_kernel=smooth_kernel, eps=eps)

    ctx_mean = ctx.mean(dim=1)
    tgt_mean = tgt.mean(dim=1)
    weight = mask.to(ctx_mean.dtype)
    area = weight.sum(dim=(-2, -1)).clamp_min(1.0)
    per_batch = (
        (ctx_mean - tgt_mean).square() * weight
    ).sum(dim=(-2, -1)) / area
    return per_batch.mean()

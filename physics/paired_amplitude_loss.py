"""Scale-stable paired image losses for attenuation correction.

The absolute units of ASP adjoint images can be very small.  A fixed absolute
log floor (for example 1e-5) can therefore clamp both paired images to the
same constant and produce an exactly-zero loss/gradient.

We instead divide both current and reference by ONE detached scale derived
from the reference image.  This changes only the arbitrary common numerical
unit; the current/reference amplitude ratio, depth dependence, and common-
mode attenuation information are preserved.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "paired_reference_scale",
    "smooth_log_envelope",
    "depth_log_energy",
    "paired_image_loss",
]


def paired_reference_scale(reference: torch.Tensor, floor: float = 1e-20):
    if reference.ndim != 3:
        raise ValueError("expected paired image shape [B,Z,X]")
    scale = reference.detach().abs().mean(dim=(-2, -1), keepdim=True)
    return scale.clamp_min(float(floor))


def smooth_log_envelope(image: torch.Tensor, kernel: int, eps: float,
                        scale: torch.Tensor):
    env = image.abs()
    if kernel > 1:
        env = F.avg_pool2d(
            env[:, None], kernel_size=kernel, stride=1,
            padding=kernel // 2)[:, 0]
    return torch.log(env / scale + float(eps))


def depth_log_energy(image: torch.Tensor, bins: int, eps: float,
                     scale: torch.Tensor):
    """Depth-binned log mean envelope in the same paired reference units."""
    env = image.abs().mean(dim=-1)
    B, nz = env.shape
    scale_1d = scale.reshape(B, 1)
    edges = torch.linspace(
        0, nz, bins + 1, device=image.device).round().long()
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        if int(b) <= int(a):
            continue
        value = env[:, int(a):int(b)].mean(dim=-1, keepdim=True)
        rows.append(torch.log(value / scale_1d + float(eps))[:, 0])
    return torch.stack(rows, dim=-1)


def paired_image_loss(current: torch.Tensor, reference: torch.Tensor,
                      smooth_kernel: int, depth_bins: int, eps: float,
                      depth_weight: float):
    if current.shape != reference.shape:
        raise ValueError("paired image shapes differ")
    scale = paired_reference_scale(reference)
    cur_log = smooth_log_envelope(current, smooth_kernel, eps, scale)
    ref_log = smooth_log_envelope(reference, smooth_kernel, eps, scale)
    image = (cur_log - ref_log).abs().mean()
    cur_depth = depth_log_energy(current, depth_bins, eps, scale)
    ref_depth = depth_log_energy(reference, depth_bins, eps, scale)
    depth = (cur_depth - ref_depth).abs().mean()
    total = image + float(depth_weight) * depth
    return total, image, depth, scale

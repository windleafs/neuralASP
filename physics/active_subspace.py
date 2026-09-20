"""Small utilities for multi-sample propagation active-subspace analysis."""
from __future__ import annotations

import torch

__all__ = [
    "gram_spectrum",
    "linearity_diagnostics",
    "subspace_overlap",
    "subspace_overlap_curve",
    "below_screen_mask",
]


def gram_spectrum(G: torch.Tensor):
    """Eigen-spectrum of a symmetric positive semidefinite Gram matrix.

    Returns right-singular-vector equivalents in Vh, with rows ordered from
    largest to smallest response energy.
    """
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("G must be square")
    Gs = 0.5 * (G + G.T)
    evals, evecs = torch.linalg.eigh(Gs)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0)
    evecs = evecs[:, order]
    s = evals.sqrt()
    energy = evals
    frac = energy / energy.sum().clamp_min(1e-30)
    cumulative = torch.cumsum(frac, dim=0)

    def rank_at(threshold: float) -> int:
        idx = torch.nonzero(cumulative >= threshold)
        return int(idx[0, 0] + 1) if idx.numel() else int(len(s))

    return {
        "eigenvalues": evals,
        "singular_values": s,
        "energy_fraction": frac,
        "cumulative_energy": cumulative,
        "rank_90": rank_at(0.90),
        "rank_95": rank_at(0.95),
        "rank_99": rank_at(0.99),
        "Vh": evecs.T,
    }


def linearity_diagnostics(reference_J: torch.Tensor, test_J: torch.Tensor,
                          eps: float = 1e-12,
                          active_rel_threshold: float = 1e-6):
    """Columnwise agreement between finite-difference Jacobians.

    Both matrices must have shape [features, parameters]. A perfectly linear
    response gives cosine=1, norm_ratio=1 and relative_error=0.

    Near-null reference columns are marked inactive instead of contributing
    artificial cosine=0 entries. active_rel_threshold is relative to the
    largest reference-column norm.
    """
    if reference_J.shape != test_J.shape or reference_J.ndim != 2:
        raise ValueError("Jacobian shapes must match and be 2D")
    if active_rel_threshold < 0:
        raise ValueError("active_rel_threshold must be non-negative")

    ref_norm_raw = reference_J.norm(dim=0)
    test_norm_raw = test_J.norm(dim=0)
    active_floor = torch.maximum(
        ref_norm_raw.max() * active_rel_threshold,
        ref_norm_raw.new_tensor(eps),
    )
    valid = ref_norm_raw > active_floor

    ref_norm = ref_norm_raw.clamp_min(eps)
    test_norm = test_norm_raw.clamp_min(eps)
    cosine = (reference_J * test_J).sum(dim=0) / (ref_norm * test_norm)
    norm_ratio = test_norm / ref_norm
    relative_error = (test_J - reference_J).norm(dim=0) / ref_norm
    return {
        "cosine": cosine,
        "norm_ratio": norm_ratio,
        "relative_error": relative_error,
        "valid": valid,
        "reference_norm": ref_norm_raw,
        "test_norm": test_norm_raw,
        "active_floor": active_floor,
    }
def subspace_overlap(Vh_a: torch.Tensor, Vh_b: torch.Tensor, k: int) -> torch.Tensor:
    """Mean squared canonical overlap of two k-dimensional row subspaces."""
    if Vh_a.ndim != 2 or Vh_b.ndim != 2:
        raise ValueError("Vh inputs must be 2D")
    if Vh_a.shape[1] != Vh_b.shape[1]:
        raise ValueError("subspaces must live in the same parameter space")
    k = min(int(k), Vh_a.shape[0], Vh_b.shape[0])
    if k < 1:
        raise ValueError("k must be positive")
    A = Vh_a[:k]
    B = Vh_b[:k]
    return (A @ B.T).square().sum() / float(k)


def subspace_overlap_curve(Vh_a: torch.Tensor, Vh_b: torch.Tensor,
                           max_rank: int):
    """Return O(k) for k=1..max_rank for two row-subspace bases."""
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    limit = min(max_rank, Vh_a.shape[0], Vh_b.shape[0])
    ranks = list(range(1, limit + 1))
    values = torch.stack([
        subspace_overlap(Vh_a, Vh_b, k) for k in ranks
    ])
    return ranks, values

def below_screen_mask(mask: torch.Tensor, depth_index: int) -> torch.Tensor:
    """Keep only pixels at or below one screen depth.

    mask is [..., nz, nx].  The returned tensor has the same shape/dtype.
    """
    if mask.ndim < 2:
        raise ValueError("mask must have spatial dimensions")
    nz = mask.shape[-2]
    if not (0 <= depth_index < nz):
        raise ValueError("depth_index out of range")
    out = mask.clone()
    out[..., :depth_index, :] = 0
    return out

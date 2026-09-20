import torch

from physics.active_subspace import (
    below_screen_mask,
    gram_spectrum,
    linearity_diagnostics,
    subspace_overlap,
    subspace_overlap_curve,
)


def test_gram_spectrum_recovers_rank_one_direction():
    v = torch.tensor([1.0, 2.0, -1.0])
    G = torch.outer(v, v)
    rep = gram_spectrum(G)
    assert rep["rank_90"] == 1
    assert rep["rank_95"] == 1
    assert rep["energy_fraction"][0] > 0.999999


def test_linearity_diagnostics_identity():
    torch.manual_seed(0)
    J = torch.randn(20, 5)
    d = linearity_diagnostics(J, J.clone())
    torch.testing.assert_close(d["cosine"], torch.ones(5), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(d["norm_ratio"], torch.ones(5), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(d["relative_error"], torch.zeros(5), atol=1e-6, rtol=1e-6)


def test_subspace_overlap_identical_is_one():
    Q, _ = torch.linalg.qr(torch.randn(8, 4))
    Vh = Q.T
    overlap = subspace_overlap(Vh, Vh, 4)
    torch.testing.assert_close(overlap, torch.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_below_screen_mask_zeros_shallower_rows():
    mask = torch.ones(1, 6, 3)
    out = below_screen_mask(mask, 2)
    assert out[:, :2].sum() == 0
    assert out[:, 2:].sum() == 12

def test_linearity_diagnostics_ignore_near_null_columns():
    ref = torch.zeros(10, 3)
    test = torch.zeros(10, 3)
    ref[:, 0] = torch.arange(10, dtype=torch.float32)
    test[:, 0] = ref[:, 0]
    # Column 1 is exactly null, column 2 is tiny relative to column 0.
    ref[:, 2] = 1e-10
    test[:, 2] = 2e-10
    d = linearity_diagnostics(ref, test, active_rel_threshold=1e-6)
    assert d["valid"].tolist() == [True, False, False]
    torch.testing.assert_close(d["cosine"][0], torch.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_subspace_overlap_curve_identical_is_one():
    Q, _ = torch.linalg.qr(torch.randn(9, 5))
    Vh = Q.T
    ranks, values = subspace_overlap_curve(Vh, Vh, 5)
    assert ranks == [1, 2, 3, 4, 5]
    torch.testing.assert_close(values, torch.ones(5), atol=1e-6, rtol=1e-6)

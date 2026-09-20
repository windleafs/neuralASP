import torch

from physics.relative_active_subspace import (
    common_mode_indices,
    project_gram_remove_common,
    relative_energy_fraction,
    relative_mode_indices,
)


def test_common_and_relative_indices_depth_major_layout():
    assert common_mode_indices(3, 4) == [0, 4, 8]
    assert relative_mode_indices(2, 3) == [1, 2, 4, 5]


def test_project_gram_removes_every_k0_coordinate():
    G = torch.ones(6, 6)
    out = project_gram_remove_common(G, n_depth=2, n_lateral_modes=3)
    for idx in (0, 3):
        assert torch.count_nonzero(out[idx]) == 0
        assert torch.count_nonzero(out[:, idx]) == 0
    keep = torch.tensor([1, 2, 4, 5])
    torch.testing.assert_close(out[keep[:, None], keep[None, :]], torch.ones(4, 4))


def test_relative_energy_fraction_is_trace_fraction():
    G = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    rel = project_gram_remove_common(G, n_depth=2, n_lateral_modes=2)
    # common indices are 0 and 2, so retained trace is 3 + 1 out of 10.
    assert abs(relative_energy_fraction(G, rel) - 0.4) < 1e-7

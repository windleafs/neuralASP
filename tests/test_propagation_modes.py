import torch

from physics.propagation_modes import (
    amplitude_screen_perturbation,
    lateral_dct_modes,
    phase_screen_perturbation,
    select_depth_screen_indices,
)


def test_lateral_modes_have_unit_physical_rms():
    nx, pad, k = 40, 6, 5
    modes = lateral_dct_modes(nx, pad, k)
    physical = modes[:, pad:-pad]
    rms = physical.square().mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        physical[1:].mean(dim=-1),
        torch.zeros(k - 1), atol=1e-6, rtol=1e-6)


def test_phase_screen_integrates_to_requested_delay():
    nx, nz, dz = 24, 16, 2e-4
    mode = torch.ones(nx)
    step_us = 0.03
    ds = phase_screen_perturbation(mode, nz, 5, dz, step_us)
    integrated_us = ds.sum(dim=0) * dz * 1e6
    torch.testing.assert_close(
        integrated_us, torch.full((nx,), step_us), atol=1e-7, rtol=1e-6)


def test_amplitude_screen_integrates_to_requested_np():
    nx, nz, dz = 24, 16, 2e-4
    mode = torch.ones(nx)
    step_np = 0.04
    rate = amplitude_screen_perturbation(mode, nz, 7, dz, step_np)
    integrated = rate.sum(dim=0) * dz
    torch.testing.assert_close(
        integrated, torch.full((nx,), step_np), atol=1e-7, rtol=1e-6)


def test_depth_selection_is_valid_and_monotone():
    idx, depth = select_depth_screen_indices(
        nz=101, z0_m=0.0, dz_m=2e-4, count=6,
        min_depth_mm=2.0, max_depth_mm=18.0)
    assert len(idx) == len(depth)
    assert all(0 <= i < 100 for i in idx)
    assert all(a < b for a, b in zip(idx[:-1], idx[1:]))
    assert depth[0] >= 2.0 - 0.11
    assert depth[-1] <= 18.0 + 0.11

import torch

from physics.phase_screen import (
    mean_controls_to_ds,
    mean_delay_profile_us,
    project_mean_delay_controls,
)


def test_mean_controls_reproduce_constant_mean_slowness():
    nz, nx, dz = 33, 40, 0.2e-3
    value = 1.5e-5
    true_ds = torch.full((nz, nx), value)
    true_ds[-1] = 0

    for controls in (2, 4, 8, 16):
        raw, target, info = project_mean_delay_controls(
            true_ds, controls, dz, pad=4, limit_us=20.0)
        assert info["saturated_control_fraction"] == 0.0
        ds = mean_controls_to_ds(
            raw, nz, nx, dz, limit_us=20.0)
        torch.testing.assert_close(
            ds[:-1], true_ds[:-1], atol=2e-8, rtol=2e-4)
        torch.testing.assert_close(ds[-1], torch.zeros(nx))
        assert target.shape == (controls,)


def test_mean_profile_is_anchored_at_zero_and_ends_at_last_control():
    raw = torch.tensor([0.2, -0.1, 0.35], dtype=torch.float32)
    limit_us = 2.0
    profile = mean_delay_profile_us(raw, nz=65, limit_us=limit_us)
    bounded = limit_us * torch.tanh(raw)
    torch.testing.assert_close(profile[0], torch.tensor(0.0))
    torch.testing.assert_close(profile[-1], bounded[-1])


def test_mean_controls_are_differentiable():
    raw = torch.randn(3, 8, requires_grad=True)
    ds = mean_controls_to_ds(raw, nz=65, nx=48, dz=0.2e-3,
                             limit_us=2.0)
    loss = ds.square().mean()
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()

import math

import torch

from physics.phase_screen import controls_to_discrete_ds, phase_control_curves_us
from physics.angular_spectrum import HeterogeneousAngularSpectrum


def test_discrete_screen_uses_one_row_per_layer_and_zero_mean_gauge():
    raw = torch.tensor([
        [-0.5, -0.1, 0.3, 0.8],
        [0.6, 0.2, -0.2, -0.7],
        [-0.3, 0.1, 0.4, 0.2],
        [0.1, -0.4, -0.1, 0.5],
    ], dtype=torch.float32)
    nz, nx, dz, pad = 33, 40, 0.2e-3, 4
    ds = controls_to_discrete_ds(raw, nz, nx, dz, limit_us=0.2, pad=pad)
    assert ds.shape == (nz, nx)
    assert torch.count_nonzero(ds[-1]) == 0
    active = ds.abs().amax(dim=-1) > 0
    assert int(active.sum()) == raw.shape[0]

    curves = phase_control_curves_us(raw, nx, 0.2, pad)
    torch.testing.assert_close(curves[:, pad:-pad].mean(-1),
                               torch.zeros(raw.shape[0]), atol=2e-7, rtol=0)


def test_sparse_ds_produces_exact_integrated_phase_screen():
    raw = torch.tensor([[[-0.4, 0.0, 0.5, 0.2]]], dtype=torch.float32)
    nz, nx, dz = 9, 16, 0.25e-3
    limit_us = 0.2
    ds = controls_to_discrete_ds(raw, nz, nx, dz,
                                 limit_us=limit_us, pad=0)
    curves = phase_control_curves_us(raw, nx, limit_us, pad=0)

    omega = torch.tensor([2.0 * math.pi * 4.0e6], dtype=torch.float32)
    asp = HeterogeneousAngularSpectrum(nx, 0.2e-3, dz, 1540.0)
    active_row = int((ds[0].abs().amax(dim=-1) > 0).nonzero()[0])
    scr = asp.screen(ds[0, active_row], omega)
    expected = torch.exp(1j * omega[:, None] * curves[0, 0][None] * 1e-6)
    torch.testing.assert_close(scr, expected, atol=2e-6, rtol=2e-6)


def test_discrete_screen_is_differentiable():
    raw = torch.randn(2, 4, 24, requires_grad=True)
    ds = controls_to_discrete_ds(raw, 65, 80, 0.2e-3,
                                 limit_us=0.2, pad=8)
    loss = ds.square().mean()
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()

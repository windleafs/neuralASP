"""Tests for coarse-parameter / fine-lateral-propagation Born imaging."""
import math
import os
import tempfile

import torch

from common import build_meta, load_config
from physics.oversampled_imaging import LateralOversampledBornModel

DT = torch.complex128

CFG = """
seed: 0
physics: {c0: 1540.0, f0: 2.0e6, bandwidth: 0.5, fs: 20.0e6,
          eps_evanescent: 1.0e-9, spreading: farfield2d,
          lateral_oversample: 2}
grid: {nx: 32, nz: 16, dx: 0.3e-3, dz: 0.3e-3}
coarse: {factor: 4}
array: {n_elements: 10, pitch: 0.3e-3}
acq: {n_angles: 3, angle_span_deg: 8.0, n_t: 256, n_freq: 4,
      holdout_stride: 3}
"""


def make_model():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(CFG)
        path = f.name
    cfg = load_config(path)
    os.unlink(path)
    meta = build_meta(cfg)
    g = cfg.grid
    born = LateralOversampledBornModel(
        meta, g.nx, g.nz, g.dx, g.dz, cfg.physics.c0,
        dtype=DT, eps=1e-9, spreading=cfg.physics.spreading,
        lateral_oversample=2,
    )
    return cfg, meta, born


def test_grid_geometry_and_shapes():
    cfg, meta, born = make_model()
    assert born.nx == cfg.grid.nx
    assert born.param_nx == cfg.grid.nx
    assert born.prop_nx == 2 * cfg.grid.nx
    assert born.dx == cfg.grid.dx
    assert born.prop_dx == cfg.grid.dx / 2

    ds = torch.zeros(cfg.grid.nz, cfg.grid.nx, dtype=torch.float64)
    u = born.transmit_fields(ds)
    assert u.shape[-2:] == (cfg.grid.nz, 2 * cfg.grid.nx)

    m = torch.zeros(cfg.grid.nz, cfg.grid.nx, dtype=DT)
    m[8, 16] = 1
    D = born.forward(m, ds, u)
    assert D.shape == (cfg.acq.n_angles, len(meta.freqs), cfg.array.n_elements)
    img = born.adjoint(D, u, ds)
    assert img.shape == m.shape
    per_angle = born.adjoint_per_angle(D, u, ds)
    assert per_angle.shape == (cfg.acq.n_angles, cfg.grid.nz, cfg.grid.nx)


def test_parameter_lift_transpose():
    cfg, _, born = make_model()
    torch.manual_seed(1)
    x = torch.randn(cfg.grid.nz, cfg.grid.nx, dtype=torch.float64)
    y = torch.randn(cfg.grid.nz, 2 * cfg.grid.nx, dtype=torch.float64)
    Rx = born.to_propagation_grid(x)
    RTy = born.from_propagation_adjoint(y)
    lhs = (Rx * y).sum()
    rhs = (x * RTy).sum()
    assert torch.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_oversampled_born_forward_adjoint_dot():
    cfg, meta, born = make_model()
    torch.manual_seed(2)
    ds = torch.randn(cfg.grid.nz, cfg.grid.nx, dtype=torch.float64) * 1e-5
    m = torch.randn(cfg.grid.nz, cfg.grid.nx, dtype=DT) * 0.1
    d = torch.randn(cfg.acq.n_angles, len(meta.freqs),
                    cfg.array.n_elements, dtype=DT)

    u = born.transmit_fields(ds)
    Fm = born.forward(m, ds, u)
    Fhd = born.adjoint(d, u, ds)
    lhs = (Fm * d.conj()).sum()
    rhs = (m * Fhd.conj()).sum()
    rel = (lhs - rhs).abs() / lhs.abs().clamp_min(1e-30)
    assert rel.item() < 1e-8, rel.item()


def test_delta_s_gradient_reaches_parameter_grid():
    cfg, _, born = make_model()
    torch.manual_seed(3)
    ds = (torch.randn(cfg.grid.nz, cfg.grid.nx, dtype=torch.float64)
          * 1e-5).requires_grad_(True)
    m = torch.randn(cfg.grid.nz, cfg.grid.nx, dtype=DT) * 0.1
    u = born.transmit_fields(ds)
    loss = born.forward(m, ds, u).abs().square().sum()
    loss.backward()
    assert ds.grad is not None
    assert ds.grad.shape == ds.shape
    assert torch.isfinite(ds.grad).all()
    assert ds.grad.abs().sum() > 0

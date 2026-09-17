"""Adjoint-consistency tests: <F m, d> == <m, F^H d> (dot product tests)
and autograd-vs-manual-adjoint gradient checks, all in double precision."""

import math
import os
import tempfile

import numpy as np
import pytest
import torch
import yaml

from common import build_meta, load_config
from physics.angular_spectrum import HeterogeneousAngularSpectrum
from physics.imaging import BornModel

DT = torch.complex128

CFG = """
seed: 0
physics: {c0: 1540.0, f0: 2.0e6, bandwidth: 0.5, fs: 20.0e6,
          eps_evanescent: 1.0e-6, spreading: farfield2d}
grid: {nx: 48, nz: 24, dx: 0.3e-3, dz: 0.3e-3}
coarse: {factor: 4}
array: {n_elements: 12, pitch: 0.3e-3}
acq: {n_angles: 4, angle_span_deg: 12.0, n_t: 256, n_freq: 5, holdout_stride: 4}
"""


def _cfg():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(CFG)
        path = f.name
    cfg = load_config(path)
    os.unlink(path)
    return cfg


def _born():
    cfg = _cfg()
    meta = build_meta(cfg)
    g = cfg.grid
    born = BornModel(meta, g.nx, g.nz, g.dx, g.dz, cfg.physics.c0,
                     dtype=DT, eps=1e-9)
    return cfg, born, len(meta.band_idx)


def test_step_adjoint_dot():
    """<A u, v> == <u, A^H v> for one split step."""
    torch.manual_seed(1)
    asp = HeterogeneousAngularSpectrum(32, 0.3e-3, 0.3e-3, 1540.0, eps=1e-9,
                                       dtype=DT)
    omega = torch.tensor([2 * math.pi * 1.8e6, 2 * math.pi * 2.2e6])
    H = asp.half_transfer(omega)
    scr = asp.screen(torch.randn(32, dtype=torch.float64) * 2e-5, omega)
    u = torch.randn(3, 2, 32, dtype=DT)
    v = torch.randn(3, 2, 32, dtype=DT)
    lhs = (asp._step(u, H, scr) * v.conj()).sum()
    rhs = (u * asp._step_adj(v, H, scr).conj()).sum()
    assert (lhs - rhs).abs().item() / lhs.abs().item() < 1e-10


def test_sample_scatter_dot():
    torch.manual_seed(2)
    cfg, born, n_w = _born()
    u = torch.randn(2, 3, 5, cfg.grid.nx, dtype=DT)
    d = torch.randn(2, 3, 5, 12, dtype=DT)
    lhs = (born.sample(u) * d.conj()).sum()
    rhs = (u * born.scatter(d).conj()).sum()
    assert (lhs - rhs).abs().item() / lhs.abs().item() < 1e-10


def test_born_forward_adjoint_dot():
    """The core adjoint-consistency check: <F(m), D> == <m, F^H(D)>."""
    torch.manual_seed(3)
    cfg, born, n_w = _born()
    g = cfg.grid
    ds = torch.randn(g.nz, g.nx, dtype=torch.float64) * 2e-5
    m = torch.randn(g.nz, g.nx, dtype=DT) * 0.2
    D = torch.randn(4, n_w, 12, dtype=DT)

    u_tx = born.transmit_fields(ds)
    Fm = born.forward(m, ds, u_tx)
    FhD = born.adjoint(D, u_tx, ds)
    lhs = (Fm * D.conj()).sum()
    rhs = (m * FhD.conj()).sum()
    rel = (lhs - rhs).abs().item() / (lhs.abs().item() + 1e-30)
    assert rel < 1e-8, rel


def test_autograd_matches_manual_adjoint():
    """grad ||F(m) - D||^2 wrt (Re, Im) of m equals Re/Im of 2 F^H(Fm - D);
    also finite-difference direction check on the real parametrization."""
    torch.manual_seed(4)
    cfg, born, n_w = _born()
    g = cfg.grid
    ds = torch.randn(g.nz, g.nx, dtype=torch.float64) * 2e-5
    mr = (torch.randn(g.nz, g.nx, dtype=torch.float64) * 0.1).detach()
    mr.requires_grad_(True)
    mi = (torch.randn(g.nz, g.nx, dtype=torch.float64) * 0.1).detach()
    mi.requires_grad_(True)
    D = torch.randn(4, n_w, 12, dtype=DT)

    def make_m(mr, mi):
        return torch.complex(mr, mi)

    u_tx = born.transmit_fields(ds)
    Fm = born.forward(make_m(mr, mi), ds, u_tx)
    loss = (Fm - D).abs().pow(2).sum()
    loss.backward()
    manual = born.adjoint(Fm - D, u_tx, ds)
    assert torch.allclose(mr.grad, 2 * manual.real, rtol=1e-6, atol=1e-9)
    assert torch.allclose(mi.grad, 2 * manual.imag, rtol=1e-6, atol=1e-9)

    # finite-difference directional derivative
    dmr = torch.randn_like(mr); dmi = torch.randn_like(mi)
    h = 1e-6
    with torch.no_grad():
        lp = (born.forward(make_m(mr + h * dmr, mi + h * dmi), ds, u_tx)
              - D).abs().pow(2).sum().item()
        lm = (born.forward(make_m(mr - h * dmr, mi - h * dmi), ds, u_tx)
              - D).abs().pow(2).sum().item()
    fd = (lp - lm) / (2 * h)
    ag = (mr.grad * dmr + mi.grad * dmi).sum().item()
    assert abs(fd - ag) / (abs(fd) + 1e-12) < 1e-4


def test_delta_s_gradient_fd():
    """Gradient through the whole forward operator wrt the slowness model."""
    torch.manual_seed(5)
    cfg, born, n_w = _born()
    g = cfg.grid
    ds = torch.randn(g.nz, g.nx, dtype=torch.float64) * 1e-5
    ds.requires_grad_(True)
    m = torch.randn(g.nz, g.nx, dtype=DT) * 0.2
    u_tx = born.transmit_fields(ds)
    loss = born.forward(m, ds, u_tx).abs().pow(2).sum()
    loss.backward()
    g_auto = ds.grad.clone()
    eps = 1e-9
    for (iz, ix) in [(4, 10), (15, 30)]:
        dsp = ds.detach().clone(); dsp[iz, ix] += eps
        dsm = ds.detach().clone(); dsm[iz, ix] -= eps
        lp = born.forward(m, dsp, born.transmit_fields(dsp)) \
            .abs().pow(2).sum().item()
        lm = born.forward(m, dsm, born.transmit_fields(dsm)) \
            .abs().pow(2).sum().item()
        fd = (lp - lm) / (2 * eps)
        assert abs(g_auto[iz, ix].item() - fd) / (abs(fd) + 1e-12) < 1e-3, \
            (iz, ix, g_auto[iz, ix].item(), fd)


def test_adjoint_imaging_focuses_point():
    """Sanity: illumination-compensated adjoint image of data from an
    isolated point scatterer in a homogeneous medium peaks at (within one
    cell of) the scatterer location."""
    torch.manual_seed(6)
    cfg, born, n_w = _born()
    g = cfg.grid
    ds = torch.zeros(g.nz, g.nx, dtype=torch.float64)
    m = torch.zeros(g.nz, g.nx, dtype=DT)
    cz, cx = 14, 24
    zz, xx = torch.meshgrid(torch.arange(g.nz), torch.arange(g.nx),
                            indexing="ij")
    m += (((xx - cx) ** 2 + (zz - cz) ** 2) < 4.0).to(DT) * 0.5
    u_tx = born.transmit_fields(ds)
    D = born.forward(m, ds, u_tx)
    img = born.adjoint(D, u_tx, ds)
    illum = born.illumination(u_tx)
    img = img / (illum + 0.05 * illum.mean()).sqrt()
    pz, px = divmod(img.abs().argmax().item(), g.nx)
    assert abs(pz - cz) <= 2 and abs(px - cx) <= 1, (pz, px)

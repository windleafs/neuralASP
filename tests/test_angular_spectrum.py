"""Angular-spectrum propagator unit tests (double precision)."""

import math

import numpy as np
import pytest
import torch

from physics.angular_spectrum import HeterogeneousAngularSpectrum

DT = torch.complex128
F64 = torch.float64


def _make(nx=96, dx=0.3e-3, dz=0.3e-3, c0=1540.0):
    return HeterogeneousAngularSpectrum(nx, dx, dz, c0, eps=1e-12, dtype=DT)


def _omega(f):
    return torch.tensor([2 * math.pi * f], dtype=F64)


def test_plane_wave_exact_phase():
    """A propagating plane wave (FFT-bin aligned) acquires exactly exp(i kz L)."""
    nx, dx, dz, c0 = 96, 0.3e-3, 0.3e-3, 1540.0
    asp = _make(nx, dx, dz, c0)
    f = 2.0e6
    omega = _omega(f)
    k = (omega / c0).item()
    kx0 = round(0.3 * k / (2 * math.pi / (nx * dx))) * (2 * math.pi / (nx * dx))
    x = torch.arange(nx, dtype=F64) * dx
    u0 = torch.exp(1j * kx0 * x).expand(1, 1, nx).contiguous()
    ds = torch.zeros(40, nx, dtype=F64)
    u_all = asp.forward(u0, ds, omega)
    kz = math.sqrt(k ** 2 - kx0 ** 2)
    expect = torch.exp(1j * (kx0 * x + kz * 39 * dz))
    err = (u_all[0, 0, 39] - expect).abs().max().item()
    assert err < 1e-10, err


def test_evanescent_exact_decay():
    """A pure evanescent plane wave decays as exp(-|kz| L)."""
    nx, dx, dz, c0 = 96, 0.3e-3, 0.3e-3, 1540.0
    asp = _make(nx, dx, dz, c0)
    f = 2.0e6
    omega = _omega(f)
    k = (omega / c0).item()
    kx0 = round(1.05 * k / (2 * math.pi / (nx * dx))) * (2 * math.pi / (nx * dx))
    x = torch.arange(nx, dtype=F64) * dx
    u0 = torch.exp(1j * kx0 * x).expand(1, 1, nx).contiguous()
    ds = torch.zeros(40, nx, dtype=F64)
    u_all = asp.forward(u0, ds, omega)
    gamma = math.sqrt(kx0 ** 2 - k ** 2)
    expect_decay = math.exp(-gamma * 20 * dz)
    got_decay = (u_all[0, 0, 20] / u_all[0, 0, 0]).abs().max().item()
    assert abs(got_decay - expect_decay) / expect_decay < 1e-4
    assert got_decay < 1e-5  # strongly attenuated, not amplified


def test_constant_slowness_phase_exact_on_axis():
    """Constant delta_s: the on-axis plane wave accumulates exactly
    exp(i omega ds L) on top of the homogeneous phase (exact for kx=0)."""
    nx, dx, dz, c0 = 96, 0.3e-3, 0.3e-3, 1540.0
    asp = _make(nx, dx, dz, c0)
    f = 2.0e6
    omega = _omega(f)
    ds_val = 2.0e-5
    nz = 50
    ds = torch.full((nz, nx), ds_val, dtype=F64)
    u0 = torch.ones(1, 1, nx, dtype=DT)
    u_all = asp.forward(u0, ds, omega)
    L = (nz - 1) * dz
    phase = (omega / c0 * L + omega * ds_val * L).item()
    got = u_all[0, 0, -1, nx // 2].item()
    assert abs(got.real - math.cos(phase)) < 1e-9
    assert abs(got.imag - math.sin(phase)) < 1e-9


def test_fresnel_agreement():
    """Gaussian aperture vs analytic paraxial Fresnel integral."""
    nx, dx, dz, c0 = 512, 0.1e-3, 0.5e-3, 1540.0
    asp = _make(nx, dx, dz, c0)
    f = 2.0e6
    lam = c0 / f
    omega = _omega(f)
    x = ((torch.arange(nx, dtype=F64) - nx / 2) * dx)
    w0 = 2.0e-3
    u0 = torch.exp(-(x / w0) ** 2).expand(1, 1, nx).contiguous().to(DT)
    nz = 40  # L = 19.5 mm
    ds = torch.zeros(nz, nx, dtype=F64)
    u_as = asp.forward(u0, ds, omega)[0, 0, -1]
    L = (nz - 1) * dz
    kern = torch.exp(1j * math.pi * (x[:, None] - x[None, :]) ** 2 / (lam * L))
    pref = torch.exp(torch.tensor(1j * 2 * math.pi * L / lam)) \
        / torch.sqrt(torch.tensor(1j * lam * L))
    u_fr = pref * (kern @ u0[0, 0]) * dx  # dx = quadrature measure
    sl = slice(nx // 2 - 80, nx // 2 + 80)  # small-angle region
    num = (u_as[sl] - u_fr[sl]).abs().pow(2).sum().item()
    den = u_fr[sl].abs().pow(2).sum().item()
    assert num / den < 5e-3, num / den


def test_phase_screen_differentiable():
    """Finite-difference check of dL/d(delta_s) through the phase screens."""
    torch.manual_seed(0)
    nx, nz = 32, 12
    asp = _make(nx)
    omega = _omega(2.0e6)
    u0 = torch.randn(1, 1, nx, dtype=DT)
    ds = torch.randn(nz, nx, dtype=F64) * 1e-5
    ds.requires_grad_(True)
    u_all = asp.forward(u0, ds, omega)
    loss = u_all.abs().pow(2).sum()
    loss.backward()
    g_auto = ds.grad.clone()
    eps = 1e-9
    for (iz, ix) in [(3, 5), (7, 20), (10, 31 % nx)]:
        ds_p = ds.detach().clone(); ds_p[iz, ix] += eps
        ds_m = ds.detach().clone(); ds_m[iz, ix] -= eps
        lp = asp.forward(u0, ds_p, omega).abs().pow(2).sum().item()
        lm = asp.forward(u0, ds_m, omega).abs().pow(2).sum().item()
        g_fd = (lp - lm) / (2 * eps)
        rel = abs(g_auto[iz, ix].item() - g_fd) / (abs(g_fd) + 1e-12)
        assert rel < 1e-4, (iz, ix, g_auto[iz, ix].item(), g_fd)

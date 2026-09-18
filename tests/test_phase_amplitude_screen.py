import torch

from physics.amplitude_screen import (
    amplitude_control_curves_np,
    controls_to_discrete_amplitude_rate,
)
from physics.angular_spectrum import HeterogeneousAngularSpectrum


def test_amplitude_control_embedding_preserves_integrated_screen():
    raw = torch.zeros(2, 3, dtype=torch.float64)
    raw[0, 1] = torch.atanh(torch.tensor(0.5, dtype=torch.float64))
    dz = 0.2e-3
    rate = controls_to_discrete_amplitude_rate(
        raw, nz=9, nx=12, dz=dz, limit_np=0.4, pad=0)
    curves = amplitude_control_curves_np(
        raw, nx=12, limit_np=0.4, pad=0)

    # Each layer is embedded in exactly one slab, so its depth integral equals
    # the requested integrated log-amplitude curve.
    nonzero_rows = (rate.abs().sum(dim=-1) > 0).nonzero().flatten()
    assert len(nonzero_rows) == 1
    zi = int(nonzero_rows[0])
    assert torch.allclose(rate[zi] * dz, curves[0], atol=1e-12, rtol=1e-12)


def test_complex_screen_magnitude_and_phase():
    asp = HeterogeneousAngularSpectrum(
        nx=8, dx=0.1e-3, dz=0.2e-3, c0=1540.0,
        dtype=torch.complex128)
    omega = 2 * torch.pi * torch.tensor(
        [4e6, 6e6], dtype=torch.float64)
    ds = torch.full((8,), 20e-9, dtype=torch.float64)
    amp_rate = torch.full((8,), 0.2 / asp.dz, dtype=torch.float64)
    omega_ref = float(2 * torch.pi * 4e6)

    scr = asp.complex_screen(
        ds, amp_rate, omega,
        amplitude_omega_ref=omega_ref,
        amplitude_freq_power=1.0)

    expected_mag = torch.exp(
        -0.2 * (omega / omega_ref))[:, None].expand_as(scr.real)
    assert torch.allclose(
        scr.abs(), expected_mag, atol=1e-12, rtol=1e-12)

    expected_phase = (
        asp.dz * omega[:, None] * ds[None, :])
    phase_error = torch.angle(
        scr * torch.exp(-1j * expected_phase))
    assert torch.allclose(
        phase_error, torch.zeros_like(phase_error),
        atol=1e-12, rtol=1e-12)


def test_complex_screen_step_adjoint_identity():
    torch.manual_seed(3)
    asp = HeterogeneousAngularSpectrum(
        nx=16, dx=0.1e-3, dz=0.2e-3, c0=1540.0,
        dtype=torch.complex128)
    omega = 2 * torch.pi * torch.tensor(
        [4.5e6, 6.5e6], dtype=torch.float64)
    H = asp.half_transfer(omega)
    ds = 50e-9 * torch.randn(16, dtype=torch.float64)
    amp_rate = (0.15 / asp.dz) * torch.randn(
        16, dtype=torch.float64)
    scr = asp.complex_screen(
        ds, amp_rate, omega,
        amplitude_omega_ref=float(2 * torch.pi * 6e6),
        amplitude_freq_power=1.0)

    u = torch.randn(2, 16, dtype=torch.complex128)
    v = torch.randn(2, 16, dtype=torch.complex128)
    Au = asp._step(u, H, scr)
    AHv = asp._step_adj(v, H, scr)

    lhs = torch.vdot(Au, v)
    rhs = torch.vdot(u, AHv)
    assert torch.allclose(lhs, rhs, atol=1e-11, rtol=1e-11)


def test_complex_screen_imager_accepts_explicit_amplitude_f0():
    from types import SimpleNamespace
    from physics.complex_screen_imaging import (
        LateralOversampledComplexScreenBornModel,
    )

    meta = SimpleNamespace(
        angles_deg=[-1.0, 0.0, 1.0],
        xe_coords=[-0.0002, 0.0, 0.0002],
        freqs=[4.5e6, 5.5e6, 6.5e6],
        f0=5.5e6,
        x0=-0.0004,
        z0=0.0,
        t_ref_s=[0.0, 0.0, 0.0],
        system_response=[1.0, 1.0, 1.0],
    )
    model = LateralOversampledComplexScreenBornModel(
        meta,
        nx=4,
        nz=5,
        dx=0.2e-3,
        dz=0.2e-3,
        c0=1540.0,
        lateral_oversample=2,
        amplitude_freq_power=1.0,
        amplitude_f0_hz=6.0e6,
    )
    assert model.amplitude_f0_hz == 6.0e6
    assert abs(model.amplitude_omega_ref / (2.0 * torch.pi) - 6.0e6) < 1e-6

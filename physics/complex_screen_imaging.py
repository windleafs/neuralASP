"""Oversampled Born imaging with joint phase and log-amplitude screens."""
from __future__ import annotations

import torch

from physics.oversampled_imaging import LateralOversampledBornModel

__all__ = ["LateralOversampledComplexScreenBornModel"]


class LateralOversampledComplexScreenBornModel(LateralOversampledBornModel):
    """Lateral-oversampled Born model with optional propagation amplitude rate.

    Public parameter grids remain coarse.  Both slowness perturbation and
    amplitude rate are lifted laterally to the internal ASP grid.
    """

    def __init__(self, *args, amplitude_freq_power=1.0,
                 amplitude_f0_hz=None, **kwargs):
        # amplitude_f0_hz belongs to this derived complex-screen operator and
        # must not leak into LateralOversampledBornModel.__init__().
        super().__init__(*args, **kwargs)
        if amplitude_freq_power < 0:
            raise ValueError("amplitude_freq_power must be non-negative")
        self.amplitude_freq_power = float(amplitude_freq_power)

        if amplitude_f0_hz is None:
            amplitude_f0_hz = (
                float(torch.as_tensor(self.omega_).mean().item())
                / (2.0 * torch.pi)
            )
        if float(amplitude_f0_hz) <= 0:
            raise ValueError("amplitude_f0_hz must be positive")

        self.amplitude_f0_hz = float(amplitude_f0_hz)
        self.amplitude_omega_ref = (
            2.0 * torch.pi * self.amplitude_f0_hz
        )

    def _amp_prop(self, amplitude_rate):
        if amplitude_rate is None:
            return None
        return self.to_propagation_grid(amplitude_rate)

    def transmit_fields(self, delta_s, angles_idx=None, amplitude_rate=None):
        ds_prop = self.to_propagation_grid(delta_s)
        amp_prop = self._amp_prop(amplitude_rate)

        if angles_idx is None:
            ang = self.angles_
        else:
            ang = self.angles_[angles_idx]
        k0 = self.omega_ / self.c0
        phase = (
            ang[:, None, None].sin() * k0[None, :, None]
            * self.x[None, None, :]
        )
        u0 = self.apod[None, None, :] * torch.exp(
            1j * phase.to(self.cdtype))
        response = (
            self.source_response if angles_idx is None
            else self.source_response[angles_idx]
        )
        u0 = u0 * response[..., None]
        u0 = self.asp._ifft(
            self.asp._fft(u0) * self.surface_transfer)
        if ds_prop.dim() == 3:
            u0 = u0.unsqueeze(0).expand(ds_prop.shape[0], *u0.shape)

        return self.asp.forward(
            u0, ds_prop, self.omega_,
            amplitude_rate=amp_prop,
            amplitude_omega_ref=self.amplitude_omega_ref,
            amplitude_freq_power=self.amplitude_freq_power,
        )

    def forward(self, m, delta_s, u_tx, angles_idx=None, amplitude_rate=None):
        m_prop = self.to_propagation_grid(m)
        ds_prop = self.to_propagation_grid(delta_s)
        amp_prop = self._amp_prop(amplitude_rate)

        m_b = m_prop.reshape(
            m_prop.shape[:-2]
            + (1,) * (u_tx.dim() - m_prop.dim())
            + m_prop.shape[-2:]
        )
        q = u_tx * self.w_z * m_b
        d = self.asp.march_up(
            q, ds_prop, self.omega_,
            amplitude_rate=amp_prop,
            amplitude_omega_ref=self.amplitude_omega_ref,
            amplitude_freq_power=self.amplitude_freq_power,
        )
        d = self.asp._ifft(
            self.asp._fft(d) * self.surface_transfer)
        return self.sample(d)

    def adjoint_per_angle(self, D, u_tx, delta_s, amplitude_rate=None):
        ds_prop = self.to_propagation_grid(delta_s)
        amp_prop = self._amp_prop(amplitude_rate)
        b0 = self.scatter(D)
        b0 = self.asp._ifft(
            self.asp._fft(b0) * self.surface_transfer.conj())
        b = self.asp.adjoint(
            b0, ds_prop, self.omega_,
            amplitude_rate=amp_prop,
            amplitude_omega_ref=self.amplitude_omega_ref,
            amplitude_freq_power=self.amplitude_freq_power,
        )
        img_prop = (b * (u_tx * self.w_z).conj()).sum(dim=-3)
        return self.from_propagation_adjoint(img_prop)

    def adjoint(self, D, u_tx, delta_s, angles_idx=None, amplitude_rate=None):
        per_angle = self.adjoint_per_angle(
            D, u_tx, delta_s, amplitude_rate=amplitude_rate)
        return per_angle.sum(dim=-3)

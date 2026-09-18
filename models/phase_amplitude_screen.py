"""Joint phase + propagation-amplitude correction model.

The model keeps the successful mean-delay branch and augments it with two
bounded residual branches:

    delta_s_net = delta_s_mean + alpha_phi * delta_s_relative
    a_net       = alpha_A * a_relative

where a_net is a signed log-amplitude rate [Np/m].  During ASP propagation the
combined screen is

    exp(-a dz (f/f0)^gamma) * exp(i omega delta_s dz).

Both residual gates are learnable scalars initialized near zero.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.phase_screen import PhaseScreenModel
from physics.amplitude_screen import (
    amplitude_control_curves_np,
    controls_to_discrete_amplitude_rate,
)
from physics.complex_screen_imaging import (
    LateralOversampledComplexScreenBornModel,
)
from physics.phase_screen import mean_delay_profile_us


class PhaseAmplitudeScreenModel(PhaseScreenModel):
    def __init__(self, cfg, meta, layers=4, controls=96, pad=32,
                 limit_us=0.5, mean_controls=8, mean_limit_us=2.0,
                 screen_gate=True, screen_gate_init=0.02,
                 amplitude_layers=4, amplitude_controls=48,
                 amplitude_limit_np=0.5,
                 amplitude_gate_init=0.02,
                 amplitude_freq_power=1.0,
                 amplitude_f0_hz=None):
        super().__init__(
            cfg, meta,
            layers=layers,
            controls=controls,
            pad=pad,
            limit_us=limit_us,
            mean_controls=mean_controls,
            mean_limit_us=mean_limit_us,
            fit_bulk=False,
            screen_gate=screen_gate,
            screen_gate_init=screen_gate_init,
        )
        if amplitude_layers < 1 or amplitude_controls < 2:
            raise ValueError("amplitude branch requires positive low-dimensional controls")
        if amplitude_limit_np <= 0:
            raise ValueError("amplitude_limit_np must be positive")
        if not (0.0 < amplitude_gate_init < 1.0):
            raise ValueError("amplitude_gate_init must lie in (0,1)")

        self.amplitude_layers = int(amplitude_layers)
        self.amplitude_controls = int(amplitude_controls)
        self.amplitude_limit_np = float(amplitude_limit_np)
        self.amplitude_freq_power = float(amplitude_freq_power)
        self.amplitude_f0_hz = (
            float(amplitude_f0_hz)
            if amplitude_f0_hz is not None
            else float(meta.f0)
        )

        width = cfg.model.fno_width
        self.amplitude_head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.amplitude_head.weight)
        nn.init.zeros_(self.amplitude_head.bias)

        amp_logit = math.log(
            amplitude_gate_init / (1.0 - amplitude_gate_init))
        self.amplitude_gate_logit = nn.Parameter(
            torch.tensor(float(amp_logit)))

        padded_meta = copy.deepcopy(meta)
        padded_meta.x0 -= pad * cfg.grid.dx
        self.born = LateralOversampledComplexScreenBornModel(
            padded_meta,
            cfg.grid.nx + 2 * pad,
            cfg.grid.nz,
            cfg.grid.dx,
            cfg.grid.dz,
            cfg.physics.c0,
            eps=cfg.physics.eps_evanescent,
            spreading=cfg.physics.spreading,
            lateral_oversample=int(
                cfg.physics.get("lateral_oversample", 1)),
            amplitude_freq_power=self.amplitude_freq_power,
            amplitude_f0_hz=self.amplitude_f0_hz,
        )

    def amplitude_gate_value(self):
        return torch.sigmoid(self.amplitude_gate_logit)

    def predict_all_components(self, iq, train_idx):
        latent = self._latent(iq, train_idx)

        phase_latent = F.adaptive_avg_pool2d(
            latent, (self.layers, self.controls))
        phase_raw = self.phase_head(phase_latent)[:, 0]

        mean_latent = F.adaptive_avg_pool2d(
            latent, (self.mean_controls, 1))
        mean_raw = self.mean_head(mean_latent)[:, 0, :, 0]

        amp_latent = F.adaptive_avg_pool2d(
            latent, (self.amplitude_layers, self.amplitude_controls))
        amplitude_raw = self.amplitude_head(amp_latent)[:, 0]

        bulk_raw = latent.new_zeros(latent.shape[0], 2)
        return phase_raw, mean_raw, amplitude_raw, bulk_raw

    def network_corrections(self, phase_raw, mean_raw, amplitude_raw):
        ds = self._components_to_slowness_scaled(
            phase_raw,
            mean_raw=mean_raw,
            screen_scale=self.screen_gate_value(),
        )
        amplitude_rate = controls_to_discrete_amplitude_rate(
            amplitude_raw,
            self.born.nz,
            self.born.nx,
            self.born.dz,
            self.amplitude_limit_np,
            self.pad,
        )
        amplitude_rate = (
            self.amplitude_gate_value() * amplitude_rate)
        return ds, amplitude_rate

    def angle_images(self, ds, D, idx, amplitude_rate=None):
        u = self.born.transmit_fields(
            ds, idx, amplitude_rate=amplitude_rate)
        return self.born.adjoint_per_angle(
            D[:, idx], u, ds, amplitude_rate=amplitude_rate)

    def forward_precomputed(self, iq, D, train_idx):
        phase_raw, mean_raw, amplitude_raw, bulk_raw = (
            self.predict_all_components(iq, train_idx))
        ds, amplitude_rate = self.network_corrections(
            phase_raw, mean_raw, amplitude_raw)
        images = self.angle_images(
            ds, D, train_idx, amplitude_rate=amplitude_rate)

        phase_controls_us = self.limit_us * torch.tanh(phase_raw)
        amp_controls_np = amplitude_control_curves_np(
            amplitude_raw,
            self.born.nx,
            self.amplitude_limit_np,
            self.pad,
        )
        effective_amp_controls_np = (
            self.amplitude_gate_value() * amp_controls_np)

        mean_controls_us = (
            self.mean_limit_us * torch.tanh(mean_raw))
        mean_profile_us = mean_delay_profile_us(
            mean_raw, self.born.nz, self.mean_limit_us)

        return {
            "raw_controls": phase_raw,
            "phase_controls_us": phase_controls_us,
            "effective_phase_controls_us": (
                self.screen_gate_value() * phase_controls_us),
            "screen_gate": self.screen_gate_value(),
            "raw_mean": mean_raw,
            "mean_controls_us": mean_controls_us,
            "mean_delay_profile_us": mean_profile_us,
            "raw_amplitude": amplitude_raw,
            "amplitude_controls_np": amp_controls_np,
            "effective_amplitude_controls_np": (
                effective_amp_controls_np),
            "amplitude_gate": self.amplitude_gate_value(),
            "amplitude_rate": amplitude_rate,
            "effective_ds": ds,
            "images_input": images,
            "raw_bulk": bulk_raw,
            "bulk_coeff_us": phase_raw.new_zeros(
                phase_raw.shape[0], 2),
        }

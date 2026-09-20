"""V6 active-mode phase + amplitude propagation correction network.

The model keeps the empirically useful mean-delay branch and replaces large
free multilayer phase/amplitude screens with a fixed K-dimensional population
active basis learned from the sensitivity analysis.

    delta_s = delta_s_mean + sum_k a_phi[k] Psi_k^phi
    amp     =                sum_k a_amp[k] Psi_k^A

The spatial basis is fixed; only the coefficients are predicted from RF/IQ.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.phase_screen import PhaseScreenModel
from physics.active_mode_parameterization import (
    build_active_mode_templates,
    load_shared_active_basis,
)
from physics.complex_screen_imaging import LateralOversampledComplexScreenBornModel
from physics.phase_screen import mean_controls_to_ds, mean_delay_profile_us


class ActiveModePhaseAmplitudeModel(PhaseScreenModel):
    def __init__(self, cfg, meta, active_basis_path: str | Path,
                 active_rank: int = 6, active_basis_source: str = "balanced",
                 pad: int = 32, mean_controls: int = 8,
                 mean_limit_us: float = 2.0,
                 phase_coeff_limit_us: float = 0.2,
                 amplitude_coeff_limit_np: float = 0.2,
                 phase_gate_init: float = 0.02,
                 amplitude_gate_init: float = 0.02,
                 amplitude_freq_power: float = 1.0,
                 amplitude_f0_hz: float | None = None):
        super().__init__(
            cfg, meta,
            layers=1, controls=2,
            pad=pad, limit_us=1.0,
            mean_controls=mean_controls,
            mean_limit_us=mean_limit_us,
            fit_bulk=False,
            screen_gate=False,
        )
        if active_rank < 1:
            raise ValueError("active_rank must be positive")
        if phase_coeff_limit_us <= 0 or amplitude_coeff_limit_np <= 0:
            raise ValueError("coefficient limits must be positive")
        if not (0 < phase_gate_init < 1 and 0 < amplitude_gate_init < 1):
            raise ValueError("gate initializations must lie in (0,1)")

        self.active_rank = int(active_rank)
        self.active_basis_source = str(active_basis_source)
        self.phase_coeff_limit_us = float(phase_coeff_limit_us)
        self.amplitude_coeff_limit_np = float(amplitude_coeff_limit_np)
        self.amplitude_freq_power = float(amplitude_freq_power)
        self.amplitude_f0_hz = (
            float(amplitude_f0_hz) if amplitude_f0_hz is not None
            else float(meta.f0)
        )

        # Replace legacy free-screen head with compact coefficient heads.
        width = cfg.model.fno_width
        self.phase_head = nn.Linear(width, self.active_rank)
        self.amplitude_head = nn.Linear(width, self.active_rank)
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)
        nn.init.zeros_(self.amplitude_head.weight)
        nn.init.zeros_(self.amplitude_head.bias)

        phase_logit = math.log(phase_gate_init / (1.0 - phase_gate_init))
        amp_logit = math.log(amplitude_gate_init / (1.0 - amplitude_gate_init))
        self.phase_gate_logit = nn.Parameter(torch.tensor(float(phase_logit)))
        self.amplitude_gate_logit = nn.Parameter(torch.tensor(float(amp_logit)))

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
            lateral_oversample=int(cfg.physics.get("lateral_oversample", 1)),
            amplitude_freq_power=self.amplitude_freq_power,
            amplitude_f0_hz=self.amplitude_f0_hz,
        )

        basis = load_shared_active_basis(
            active_basis_path, rank=self.active_rank, source=self.active_basis_source)
        templates = build_active_mode_templates(
            basis, nz=self.born.nz, nx=self.born.nx, dz_m=self.born.dz,
            z0_m=self.born.z0, pad=self.pad, device=torch.device("cpu"),
            dtype=torch.float32)
        self.register_buffer(
            "phase_active_templates", templates["phase_unit_ds"], persistent=True)
        self.register_buffer(
            "amplitude_active_templates", templates["amplitude_unit_rate"],
            persistent=True)
        self.active_depth_indices = list(templates["depth_indices"])
        self.active_depths_mm = list(basis["depths_mm"])
        self.active_candidate_dim = int(basis["candidate_dim"])

    def phase_gate_value(self):
        return torch.sigmoid(self.phase_gate_logit)

    def amplitude_gate_value(self):
        return torch.sigmoid(self.amplitude_gate_logit)

    def predict_all_components(self, iq, train_idx):
        latent = self._latent(iq, train_idx)
        pooled = latent.mean(dim=(-2, -1))
        phase_raw = self.phase_head(pooled)
        amp_raw = self.amplitude_head(pooled)

        if self.mean_head is not None:
            mean_latent = F.adaptive_avg_pool2d(
                latent, (self.mean_controls, 1))
            mean_raw = self.mean_head(mean_latent)[:, 0, :, 0]
        else:
            mean_raw = latent.new_zeros(latent.shape[0], 0)
        return phase_raw, mean_raw, amp_raw

    def active_coefficients(self, phase_raw, amp_raw):
        phase_coeff = (
            self.phase_gate_value() * self.phase_coeff_limit_us * torch.tanh(phase_raw))
        amp_coeff = (
            self.amplitude_gate_value() * self.amplitude_coeff_limit_np * torch.tanh(amp_raw))
        return phase_coeff, amp_coeff

    def network_corrections(self, phase_raw, mean_raw, amp_raw):
        phase_coeff, amp_coeff = self.active_coefficients(phase_raw, amp_raw)
        phase_templates = self.phase_active_templates.to(
            device=phase_coeff.device, dtype=phase_coeff.dtype)
        amp_templates = self.amplitude_active_templates.to(
            device=amp_coeff.device, dtype=amp_coeff.dtype)

        ds = torch.einsum("bk,kzx->bzx", phase_coeff, phase_templates)
        amp_rate = torch.einsum("bk,kzx->bzx", amp_coeff, amp_templates)
        if self.mean_controls:
            ds = ds + mean_controls_to_ds(
                mean_raw, self.born.nz, self.born.nx, self.born.dz,
                self.mean_limit_us)
        return ds, amp_rate, phase_coeff, amp_coeff

    def angle_images(self, ds, D, idx, amplitude_rate=None):
        u = self.born.transmit_fields(ds, idx, amplitude_rate=amplitude_rate)
        return self.born.adjoint_per_angle(
            D[:, idx], u, ds, amplitude_rate=amplitude_rate)

    def forward_precomputed(self, iq, D, train_idx):
        phase_raw, mean_raw, amp_raw = self.predict_all_components(iq, train_idx)
        ds, amp_rate, phase_coeff, amp_coeff = self.network_corrections(
            phase_raw, mean_raw, amp_raw)
        images = self.angle_images(
            ds, D, train_idx, amplitude_rate=amp_rate)
        mean_controls_us = self.mean_limit_us * torch.tanh(mean_raw)
        mean_profile_us = mean_delay_profile_us(
            mean_raw, self.born.nz, self.mean_limit_us)
        return {
            "raw_phase_coeff": phase_raw,
            "raw_amplitude_coeff": amp_raw,
            "phase_coeff_us": phase_coeff,
            "amplitude_coeff_np": amp_coeff,
            "phase_gate": self.phase_gate_value(),
            "amplitude_gate": self.amplitude_gate_value(),
            "raw_mean": mean_raw,
            "mean_controls_us": mean_controls_us,
            "mean_delay_profile_us": mean_profile_us,
            "effective_ds": ds,
            "amplitude_rate": amp_rate,
            "images_input": images,
        }

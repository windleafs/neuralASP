"""RF-to-low-dimensional propagation-correction model.

V2 predicts zero-mean multilayer phase screens. V3 adds a separate lateral-mean
propagation branch so the model represents

    delta_s(z, x) ~= mean_delay_profile(z) + relative_phase_screens(z, x).

V4 optionally gates the relative-screen contribution,

    delta_s_net = delta_s_mean + alpha * delta_s_relative,

with a learnable scalar alpha in [0, 1].  This keeps teacher/physics
parameterization ungated while allowing the network to start near the empirically
strong mean-only solution and add relative screens only when imaging loss supports
them.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import demod_iq, rf_to_D
from models.neural_operator import NeuralOperator
from physics.oversampled_imaging import LateralOversampledBornModel
from physics.phase_screen import (
    controls_to_discrete_ds,
    mean_controls_to_ds,
    mean_delay_profile_us,
)


class PhaseScreenModel(nn.Module):
    def __init__(self, cfg, meta, layers=4, controls=24, pad=32,
                 limit_us=0.2, mean_controls=0, mean_limit_us=2.0,
                 bulk_limit_us=2.0, fit_bulk=False,
                 screen_gate=False, screen_gate_init=0.02):
        super().__init__()
        if mean_controls < 0:
            raise ValueError("mean_controls must be non-negative")
        if mean_controls and fit_bulk:
            raise ValueError("mean profile head and legacy fit_bulk are mutually exclusive")
        if screen_gate and not (0.0 < screen_gate_init < 1.0):
            raise ValueError("screen_gate_init must lie strictly inside (0, 1)")
        self.meta = meta
        self.layers = layers
        self.controls = controls
        self.pad = pad
        self.limit_us = limit_us
        self.mean_controls = mean_controls
        self.mean_limit_us = mean_limit_us
        self.bulk_limit_us = bulk_limit_us
        self.fit_bulk = fit_bulk
        self.screen_gate_enabled = bool(screen_gate)
        self.screen_gate_init = float(screen_gate_init)

        self.backbone = NeuralOperator(cfg, meta, cfg.grid.nx, cfg.grid.nz,
                                       cfg.grid.dx, cfg.grid.dz)
        width = cfg.model.fno_width
        self.phase_head = nn.Conv2d(width, 1, 1)
        self.mean_head = nn.Conv2d(width, 1, 1) if mean_controls else None
        self.bulk_head = nn.Linear(width, 2) if fit_bulk else None
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)
        if self.mean_head is not None:
            nn.init.zeros_(self.mean_head.weight)
            nn.init.zeros_(self.mean_head.bias)
        if self.bulk_head is not None:
            nn.init.zeros_(self.bulk_head.weight)
            nn.init.zeros_(self.bulk_head.bias)

        if self.screen_gate_enabled:
            logit = math.log(screen_gate_init / (1.0 - screen_gate_init))
            self.screen_gate_logit = nn.Parameter(torch.tensor(float(logit)))
        else:
            self.register_parameter("screen_gate_logit", None)

        padded_meta = copy.deepcopy(meta)
        padded_meta.x0 -= pad * cfg.grid.dx
        self.born = LateralOversampledBornModel(
            padded_meta, cfg.grid.nx + 2 * pad,
            cfg.grid.nz, cfg.grid.dx, cfg.grid.dz,
            cfg.physics.c0,
            eps=cfg.physics.eps_evanescent,
            spreading=cfg.physics.spreading,
            lateral_oversample=int(cfg.physics.get("lateral_oversample", 1)),
        )

    def screen_gate_value(self):
        """Scalar network screen gate; legacy models return exactly one."""
        if self.screen_gate_logit is None:
            return self.phase_head.weight.new_ones(())
        return torch.sigmoid(self.screen_gate_logit)

    def _latent(self, iq, train_idx):
        features = self.backbone.extract_features(iq[:, train_idx], train_idx)
        return self.backbone.fno.forward_features(self.backbone.pre_fno(features))

    def predict_components(self, iq, train_idx):
        """Return raw relative-screen, mean-profile and legacy-bulk controls."""
        latent = self._latent(iq, train_idx)
        pooled = F.adaptive_avg_pool2d(latent, (self.layers, self.controls))
        phase_raw = self.phase_head(pooled)[:, 0]

        if self.mean_head is not None:
            mean_latent = F.adaptive_avg_pool2d(
                latent, (self.mean_controls, 1))
            mean_raw = self.mean_head(mean_latent)[:, 0, :, 0]
        else:
            mean_raw = latent.new_zeros(latent.shape[0], 0)

        if self.bulk_head is not None:
            bulk_raw = self.bulk_head(latent.mean(dim=(-2, -1)))
        else:
            bulk_raw = latent.new_zeros(latent.shape[0], 2)
        return phase_raw, mean_raw, bulk_raw

    def predict_controls(self, iq, train_idx):
        """Backward-compatible two-output API."""
        phase_raw, mean_raw, bulk_raw = self.predict_components(iq, train_idx)
        return phase_raw, (mean_raw if self.mean_controls else bulk_raw)

    def _components_to_slowness_scaled(self, phase_raw, mean_raw=None,
                                       bulk_raw=None, screen_scale=1.0):
        if phase_raw.shape[-2:] != (self.layers, self.controls):
            raise ValueError("incorrect phase-control shape")
        ds_rel = controls_to_discrete_ds(
            phase_raw, self.born.nz, self.born.nx, self.born.dz,
            self.limit_us, self.pad)
        ds = ds_rel * screen_scale

        if self.mean_controls:
            if mean_raw is None or mean_raw.shape[-1] != self.mean_controls:
                raise ValueError("incorrect mean-delay control shape")
            ds = ds + mean_controls_to_ds(
                mean_raw, self.born.nz, self.born.nx, self.born.dz,
                self.mean_limit_us)
        elif self.fit_bulk:
            if bulk_raw is None:
                raise ValueError("legacy bulk controls are required")
            coeff = self.bulk_limit_us * torch.tanh(bulk_raw)
            z = self.born.z0 + torch.arange(
                self.born.nz, device=phase_raw.device,
                dtype=phase_raw.dtype) * self.born.dz
            t = z / z[-1]
            G = coeff[:, :1] * t[None] + coeff[:, 1:] * t.square()[None]
            ds = ds.clone()
            ds[:, :-1] += ((G[:, 1:] - G[:, :-1])
                            * (1e-6 / self.born.dz))[:, :, None]
        return ds

    def components_to_slowness(self, phase_raw, mean_raw=None, bulk_raw=None):
        """Ungated physical/teacher parameterization (legacy semantics)."""
        return self._components_to_slowness_scaled(
            phase_raw, mean_raw, bulk_raw, screen_scale=1.0)

    def network_components_to_slowness(self, phase_raw, mean_raw=None,
                                       bulk_raw=None):
        """Network correction using the optional learnable relative-screen gate."""
        return self._components_to_slowness_scaled(
            phase_raw, mean_raw, bulk_raw,
            screen_scale=self.screen_gate_value())

    def screen_to_slowness(self, raw, aux_raw=None):
        """Backward-compatible ungated two-argument conversion API."""
        if self.mean_controls:
            return self.components_to_slowness(raw, mean_raw=aux_raw)
        return self.components_to_slowness(
            raw, bulk_raw=(aux_raw if self.fit_bulk else None))

    def angle_images(self, ds, D, idx):
        """Per-angle complex adjoint images, with no iterative m estimator."""
        born = self.born
        u = born.transmit_fields(ds, idx)
        return born.adjoint_per_angle(D[:, idx], u, ds)

    def forward_precomputed(self, iq, D, train_idx):
        phase_raw, mean_raw, bulk_raw = self.predict_components(iq, train_idx)
        ds = self.network_components_to_slowness(
            phase_raw, mean_raw, bulk_raw)
        images = self.angle_images(ds, D, train_idx)

        phase_controls_us = self.limit_us * torch.tanh(phase_raw)
        gate = self.screen_gate_value()
        effective_phase_controls_us = gate * phase_controls_us
        mean_controls_us = (self.mean_limit_us * torch.tanh(mean_raw)
                            if self.mean_controls else mean_raw)
        mean_profile_us = (mean_delay_profile_us(
            mean_raw, self.born.nz, self.mean_limit_us)
            if self.mean_controls else
            phase_raw.new_zeros(phase_raw.shape[0], self.born.nz))
        bulk_coeff_us = (self.bulk_limit_us * torch.tanh(bulk_raw)
                         if self.fit_bulk else
                         phase_raw.new_zeros(phase_raw.shape[0], 2))
        return {
            "raw_controls": phase_raw,
            "phase_controls_us": phase_controls_us,
            "effective_phase_controls_us": effective_phase_controls_us,
            "screen_gate": gate,
            "raw_mean": mean_raw,
            "mean_controls_us": mean_controls_us,
            "mean_delay_profile_us": mean_profile_us,
            "raw_bulk": bulk_raw,
            "bulk_coeff_us": bulk_coeff_us,
            "effective_ds": ds,
            "images_input": images,
        }

    def forward(self, rf, train_idx):
        iq = demod_iq(rf, self.meta)
        D = rf_to_D(rf, self.meta)
        return self.forward_precomputed(iq, D, train_idx)

    @torch.no_grad()
    def reference(self, D, train_idx, hold_idx, top_frac=0.2):
        """Fixed uniform-propagation mask and per-angle norm for each sample."""
        B = D.shape[0]
        zero = D.real.new_zeros(B, self.born.nz, self.born.nx)
        train = self.angle_images(zero, D, train_idx)
        hold = self.angle_images(zero, D, hold_idx)
        region = train[..., self.pad:-self.pad] if self.pad else train
        power = region.abs().square().mean(dim=1).sqrt()
        threshold = torch.quantile(power.flatten(1), 1.0 - top_frac,
                                   dim=1).view(B, 1, 1)
        mask = zero.clone()
        binary = (power >= threshold).to(mask.dtype)
        if self.pad:
            mask[..., self.pad:-self.pad] = binary
        else:
            mask[:] = binary
        tr_scale = (train.abs().square() * mask[:, None]).sum(dim=(-2, -1))\
            .sqrt().clamp_min(1e-30)
        ho_scale = (hold.abs().square() * mask[:, None]).sum(dim=(-2, -1))\
            .sqrt().clamp_min(1e-30)
        return {"mask": mask, "train_scales": tr_scale,
                "hold_scales": ho_scale,
                "uniform_input_coherence": coherence(train, mask, tr_scale),
                "uniform_holdout_agreement": heldout_agreement(
                    train, hold, mask, tr_scale, ho_scale)}


def coherence(images, mask, scales):
    """Normalized coherent energy for [B,angle,z,x] images."""
    v = images / scales[:, :, None, None]
    numerator = (v.sum(dim=1).abs().square() * mask).sum(dim=(-2, -1))
    denominator = (images.shape[1] * v.abs().square().sum(dim=1)
                   * mask).sum(dim=(-2, -1))
    return numerator / denominator.clamp_min(1e-30)


def heldout_agreement(train, hold, mask, train_scales, hold_scales):
    """Real normalized agreement of held-out images with input composite."""
    reference = (train / train_scales[:, :, None, None]).mean(dim=1)
    test = hold / hold_scales[:, :, None, None]
    inner = (test * reference.conj()[:, None] * mask[:, None])\
        .sum(dim=(-2, -1)).real
    test_norm = (test.abs().square() * mask[:, None]).sum(dim=(-2, -1)).sqrt()
    ref_norm = (reference.abs().square() * mask).sum(dim=(-2, -1)).sqrt()
    return (inner / (test_norm * ref_norm[:, None]).clamp_min(1e-30)).mean(dim=1)

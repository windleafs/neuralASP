"""RF-to-multilayer-phase-screen model coupled to angular-spectrum imaging.

The learned output is a small set of relative layer delays. Optional bulk
coefficients are retained for ablation/backward compatibility, but the default
V2 model estimates only zero-mean relative aberration. No sound-speed map or
scattering image is predicted or solved.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import demod_iq, rf_to_D
from models.neural_operator import NeuralOperator
from physics.imaging import BornModel
from physics.phase_screen import controls_to_discrete_ds


class PhaseScreenModel(nn.Module):
    def __init__(self, cfg, meta, layers=4, controls=24, pad=32,
                 limit_us=0.2, bulk_limit_us=2.0, fit_bulk=False):
        super().__init__()
        self.meta = meta
        self.layers = layers
        self.controls = controls
        self.pad = pad
        self.limit_us = limit_us
        self.bulk_limit_us = bulk_limit_us
        self.fit_bulk = fit_bulk
        self.backbone = NeuralOperator(cfg, meta, cfg.grid.nx, cfg.grid.nz,
                                       cfg.grid.dx, cfg.grid.dz)
        width = cfg.model.fno_width
        self.phase_head = nn.Conv2d(width, 1, 1)
        self.bulk_head = nn.Linear(width, 2) if fit_bulk else None
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)
        if self.bulk_head is not None:
            nn.init.zeros_(self.bulk_head.weight)
            nn.init.zeros_(self.bulk_head.bias)

        padded_meta = copy.deepcopy(meta)
        padded_meta.x0 -= pad * cfg.grid.dx
        self.born = BornModel(padded_meta, cfg.grid.nx + 2 * pad,
                              cfg.grid.nz, cfg.grid.dx, cfg.grid.dz,
                              cfg.physics.c0,
                              eps=cfg.physics.eps_evanescent,
                              spreading=cfg.physics.spreading)

    def predict_controls(self, iq, train_idx):
        features = self.backbone.extract_features(iq[:, train_idx], train_idx)
        latent = self.backbone.fno.forward_features(
            self.backbone.pre_fno(features))
        pooled = F.adaptive_avg_pool2d(latent, (self.layers, self.controls))
        raw = self.phase_head(pooled)[:, 0]
        if self.bulk_head is not None:
            bulk_raw = self.bulk_head(latent.mean(dim=(-2, -1)))
        else:
            # Keep a stable output/API shape for diagnostics and old tooling.
            bulk_raw = latent.new_zeros(latent.shape[0], 2)
        return raw, bulk_raw

    def screen_to_slowness(self, raw, bulk_raw=None):
        """[B,L,K] integrated delays -> sparse [B,nz,nx] ASP parameter.

        Each layer is a true discrete phase screen. Its integrated delay is
        placed in one propagation slab so that ASP evaluates
        ``exp(i * omega * tau_l(x))`` exactly. ``delta_s`` here is therefore
        only an interface carrier for the ASP, not a predicted physical
        sound-speed/slowness field.
        """
        if raw.shape[-2:] != (self.layers, self.controls):
            raise ValueError("incorrect phase-control shape")
        ds = controls_to_discrete_ds(raw, self.born.nz, self.born.nx,
                                     self.born.dz, self.limit_us, self.pad)
        if self.fit_bulk and bulk_raw is not None:
            coeff = self.bulk_limit_us * torch.tanh(bulk_raw)
            z = self.born.z0 + torch.arange(self.born.nz, device=raw.device,
                                             dtype=raw.dtype) * self.born.dz
            t = z / z[-1]
            G = coeff[:, :1] * t[None] + coeff[:, 1:] * t.square()[None]
            ds = ds.clone()
            ds[:, :-1] += ((G[:, 1:] - G[:, :-1])
                            * (1e-6 / self.born.dz))[:, :, None]
        return ds

    def angle_images(self, ds, D, idx):
        """Per-angle complex adjoint images, with no iterative m estimator."""
        born = self.born
        u = born.transmit_fields(ds, idx)
        b0 = born.scatter(D[:, idx])
        b0 = born.asp._ifft(born.asp._fft(b0)
                            * born.surface_transfer.conj())
        b = born.asp.adjoint(b0, ds, born.omega_)
        return (b * (u * born.w_z).conj()).sum(dim=2)

    def forward_precomputed(self, iq, D, train_idx):
        raw, bulk_raw = self.predict_controls(iq, train_idx)
        ds = self.screen_to_slowness(raw, bulk_raw)
        images = self.angle_images(ds, D, train_idx)
        return {"raw_controls": raw, "raw_bulk": bulk_raw,
                "phase_controls_us": self.limit_us * torch.tanh(raw),
                "bulk_coeff_us": self.bulk_limit_us * torch.tanh(bulk_raw),
                "effective_ds": ds, "images_input": images}

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

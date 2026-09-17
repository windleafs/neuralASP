"""End-to-end imaging pipeline.

RF -> (band-pass FFT) D, (IQ demod) rf_iq
  -> NeuralOperator -> coarse delta_s -> upsample -> c_hat
  -> differentiable heterogeneous angular-spectrum transmit fields
  -> 1..3 unfolded data-consistency iterations for the fine complex
     scattering image m (prox P_psi, learned steps gamma_k)
  -> predicted RF (data consistency), adjoint complex image I, envelope.

Everything is differentiable end to end, so the RF-consistency loss trains
both the neural operator (propagation parameters) and the prox (image prior).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import D_to_rf, demod_iq, rf_to_D
from physics.oversampled_imaging import LateralOversampledBornModel
from models.neural_operator import NeuralOperator

__all__ = ["ImagingPipeline", "ComplexProx"]


class ComplexProx(nn.Module):
    """Learned image-domain regularizer P_psi for the complex m:
    complex soft-shrinkage + a small residual CNN on (Re, Im)."""

    def __init__(self, hidden=16):
        super().__init__()
        self.thresh = nn.Parameter(torch.tensor(0.02))
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, m):
        mod = m.abs() + 1e-9
        m = m * torch.clamp(1.0 - self.thresh / mod, min=0.0)
        d = self.net(torch.stack([m.real, m.imag], dim=1))
        return m + torch.complex(d[:, 0], d[:, 1])


class ImagingPipeline(nn.Module):
    """Implements the closed loop of the spec (section 7 interface):

        forward(rf) -> {"c_hat", "delta_s", "m_hat", "I", "env",
                        "d_hat", "rf_pred", "D", "iq"}
    """

    def __init__(self, cfg, meta, dtype=torch.complex64):
        super().__init__()
        self.cfg = cfg
        self.meta = meta
        g = cfg.grid
        self.born = LateralOversampledBornModel(
            meta, g.nx, g.nz, g.dx, g.dz, cfg.physics.c0,
            dtype=dtype, eps=cfg.physics.eps_evanescent,
            spreading=cfg.physics.spreading,
            lateral_oversample=int(cfg.physics.get("lateral_oversample", 1)),
        )
        self.neural_operator = NeuralOperator(cfg, meta, g.nx, g.nz, g.dx, g.dz)
        m = cfg.model
        self.prox = nn.ModuleList(
            [ComplexProx(m.prox_hidden) for _ in range(m.n_unroll)])
        self.gamma = nn.Parameter(torch.tensor(
            [1.0] + [0.3] * m.n_unroll))
        self.m_rms_ref = m.m_rms_ref
        self.s0 = 1.0 / cfg.physics.c0
        self.illum_compensate = bool(cfg.model.get("illum_compensate", True))
        self.operator_version = "l11_consistency_v3_oversampled_asp"
        weight = meta.win if meta.get("response_mode", "legacy_window") == "legacy_window" else np.ones(len(meta.freqs))
        self.register_buffer("measurement_weight", torch.as_tensor(weight, dtype=dtype).view(1, -1, 1))

    def _normalize(self, img):
        rms = img.abs().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().detach()
        return img * (self.m_rms_ref / (rms + 1e-9))

    def _fadj(self, D, u_tx, ds_fine):
        img = self.born.adjoint(D * self.measurement_weight.conj(), u_tx, ds_fine)
        if not self.illum_compensate:
            return img
        illum = self.born.illumination(u_tx * self.measurement_weight.unsqueeze(-1))
        den = (illum + 0.05 * illum.mean(dim=(-2, -1), keepdim=True)).sqrt()
        return img / den

    def _estimate_m(self, D_tr, ds_fine, u_tx):
        m = self.gamma[0] * self._normalize(self._fadj(D_tr, u_tx, ds_fine))
        for k, prox in enumerate(self.prox):
            resid = D_tr - self._forward_m(m, ds_fine, u_tx)
            m = prox(m + self.gamma[k + 1]
                     * self._normalize(self._fadj(resid, u_tx, ds_fine)))
        return m

    def _forward_m(self, m, ds_fine, u_tx):
        return self.born.forward(m, ds_fine, u_tx) * self.measurement_weight

    @torch.enable_grad()
    def refine_delta_s(self, D_tr, ds_init, train_idx, steps=25, lr=1e-6):
        g = self.cfg.grid
        f = self.cfg.coarse.factor
        p0 = torch.nn.functional.adaptive_avg_pool2d(
            ds_init[:, None], (g.nz // f, g.nx // f))
        p = p0.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)
        ds_max = self.cfg.model.ds_max
        for _ in range(steps):
            opt.zero_grad()
            ds = self._upsample(p)
            u_tx = self.born.transmit_fields(ds, train_idx)
            m = self._estimate_m(D_tr, ds, u_tx)
            d_hat = self._forward_m(m, ds, u_tx)
            loss = ((d_hat - D_tr).abs() ** 2).mean() \
                / D_tr.abs().pow(2).mean().detach()
            p.grad, = torch.autograd.grad(loss, p)
            opt.step()
            with torch.no_grad():
                p.clamp_(-ds_max, ds_max)
        return self._upsample(p.detach())

    def forward(self, rf, delta_s_true=None, eta_mode="net", train_idx=None,
                pred_idx=None, return_all=True, refine_steps=0,
                refine_lr=1e-6):
        meta = self.meta
        if train_idx is None:
            train_idx = torch.arange(rf.shape[1], device=rf.device)

        D = rf_to_D(rf, meta)
        iq = demod_iq(rf, meta)

        ds_coarse = None
        if eta_mode == "truth":
            ds_fine = delta_s_true
        elif eta_mode == "zero":
            ds_fine = torch.zeros_like(delta_s_true) if delta_s_true is not None \
                else torch.zeros(rf.shape[0], self.cfg.grid.nz, self.cfg.grid.nx,
                                 device=rf.device)
        else:
            no_out = self.neural_operator(iq[:, train_idx], train_idx)
            ds_coarse = no_out["delta_s"]
            ds_fine = self._upsample(ds_coarse)
            if refine_steps > 0:
                ds_fine = self.refine_delta_s(D[:, train_idx], ds_fine,
                                              train_idx, refine_steps,
                                              refine_lr)

        u_tx = self.born.transmit_fields(ds_fine, train_idx)
        D_tr = D[:, train_idx]
        m = self._estimate_m(D_tr, ds_fine, u_tx)
        d_hat = self._forward_m(m, ds_fine, u_tx)

        out = {
            "delta_s": ds_fine,
            "delta_s_coarse": ds_coarse,
            "c_hat": 1.0 / (self.s0 + ds_fine),
            "m_hat": m,
            "d_hat": d_hat,
            "D": D,
            "iq": iq,
        }
        if pred_idx is not None:
            u_tx_p = self.born.transmit_fields(ds_fine, pred_idx)
            out["d_hat_pred"] = self._forward_m(m, ds_fine, u_tx_p)
        if not return_all:
            return out

        u_tx_all = self.born.transmit_fields(ds_fine)
        I = self._fadj(D, u_tx_all, ds_fine)
        d_hat_all = self._forward_m(m, ds_fine, u_tx_all)
        out["I"] = I
        out["I_input"] = self._fadj(D_tr, u_tx, ds_fine)
        out["env"] = out["I_input"].abs()
        out["rf_pred"] = D_to_rf(d_hat_all, meta)
        return out

    def _upsample(self, ds_coarse):
        g = self.cfg.grid
        return F.interpolate(ds_coarse, size=(g.nz, g.nx), mode="bilinear",
                             align_corners=False).squeeze(1)


def load_pipeline_checkpoint(pipe, checkpoint):
    """Never silently reinterpret legacy weights with a changed operator."""
    if checkpoint.get("operator_version") != pipe.operator_version:
        raise ValueError(
            "checkpoint uses an incompatible measurement operator; "
            "oversampled-ASP v3 checkpoints must be retrained")
    if "config" not in checkpoint:
        raise ValueError("v3 checkpoint must include acquisition config")
    from common import _to_C, build_meta
    saved_cfg = _to_C(checkpoint["config"])
    for section, keys in (("grid", ("nx", "nz", "dx", "dz")),
                          ("coarse", ("factor",)),
                          ("physics", ("c0", "eps_evanescent", "spreading",
                                       "lateral_oversample")),
                          ("model", ("ds_max", "m_rms_ref", "n_unroll"))):
        for key in keys:
            if saved_cfg[section].get(key) != pipe.cfg[section].get(key):
                raise ValueError(f"checkpoint model mismatch: {section}.{key}")
    if bool(saved_cfg.model.get("illum_compensate", True)) != pipe.illum_compensate:
        raise ValueError("checkpoint model mismatch: illum_compensate")
    frozen_m_init = checkpoint.get("stage") == "m" and saved_cfg.train.get("freeze_stages", False)
    if not frozen_m_init and bool(saved_cfg.model.get("normalize_iq", False)) != pipe.neural_operator.normalize_iq:
        raise ValueError("checkpoint encoder mismatch: normalize_iq")
    saved_cfg.physics.pop("response_path", None)
    saved = build_meta(saved_cfg)
    for key in ("n_t", "fs", "f0", "x0", "z0", "response_mode", "band_idx", "xe_coords", "t_ref_s", "angles_deg", "train_idx", "hold_idx"):
        if not np.array_equal(np.asarray(saved[key]), np.asarray(pipe.meta[key])):
            raise ValueError(f"checkpoint acquisition mismatch: {key}")
    pipe.load_state_dict(checkpoint["pipeline"])

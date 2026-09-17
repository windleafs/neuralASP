"""End-to-end pipeline tests: shapes, full differentiability (gradients reach
the neural operator, prox and steps), and synthetic-data round trips."""

import math
import os
import tempfile

import pytest
import torch
import yaml

from common import D_to_rf, build_meta, demod_iq, load_config, rf_to_D
from data.synthetic import make_phantom, simulate_sample
from models.pipeline import ImagingPipeline
from physics.imaging import BornModel

CFG = """
seed: 0
physics: {c0: 1540.0, f0: 2.0e6, bandwidth: 0.5, fs: 20.0e6,
          eps_evanescent: 1.0e-6, spreading: farfield2d}
grid: {nx: 48, nz: 24, dx: 0.3e-3, dz: 0.3e-3}
coarse: {factor: 4}
array: {n_elements: 12, pitch: 0.3e-3}
acq: {n_angles: 4, angle_span_deg: 12.0, n_t: 256, n_freq: 5, holdout_stride: 4}
model: {enc_channels: [8, 12, 8], fno_width: 16, fno_modes: [6, 5],
        fno_layers: 2, ds_max: 3.0e-5, n_unroll: 2, prox_hidden: 8,
        m_rms_ref: 0.2}
train: {noise_snr_db: 25.0}
synth: {c_perturb_percent: 4.0, c_corr_len_mm: 10.0, n_points: 5,
        n_extended: 1, point_sigma_mm: 0.6}
"""


def _cfg():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(CFG)
        path = f.name
    cfg = load_config(path)
    os.unlink(path)
    return cfg


def _batch(cfg, n=2, device="cpu"):
    meta = build_meta(cfg)
    g = cfg.grid
    born = BornModel(meta, g.nx, g.nz, g.dx, g.dz, cfg.physics.c0)
    import numpy as np
    rng = np.random.default_rng(0)
    gen = torch.Generator().manual_seed(0)
    rf, D, ds, m, c = [], [], [], [], []
    for _ in range(n):
        ph = make_phantom(rng, cfg, meta, g.nz, g.nx, g.dz, g.dx)
        out = simulate_sample(born, ph, meta, cfg.train.noise_snr_db, gen)
        rf.append(out[0]); D.append(out[1]); ds.append(out[2])
        m.append(out[3]); c.append(out[4])
    return (torch.stack(rf).to(device), torch.stack(D).to(device),
            torch.stack(ds).to(device), torch.stack(m).to(device),
            torch.stack(c).to(device))


def test_rf_D_roundtrip():
    cfg = _cfg()
    meta = build_meta(cfg)
    rf, D, *_ = _batch(cfg, n=1)
    D2 = rf_to_D(rf, meta)
    assert D2.shape == D.shape
    assert (D2 - D).abs().max().item() / D.abs().max().item() < 1e-4
    rf2 = D_to_rf(D, meta)
    err = (rf2 - rf).abs().max().item() / (rf.abs().max().item() + 1e-12)
    assert err < 1e-4


def test_pipeline_shapes_and_grads():
    cfg = _cfg()
    meta = build_meta(cfg)
    pipe = ImagingPipeline(cfg, meta)
    rf, D, ds, m_true, c_true = _batch(cfg)
    tr_idx = torch.tensor(meta.train_idx)

    out = pipe(rf, delta_s_true=ds, eta_mode="net", train_idx=tr_idx,
               return_all=True)
    assert out["c_hat"].shape == ds.shape
    assert out["m_hat"].shape == ds.shape
    assert out["I"].shape == ds.shape
    assert out["rf_pred"].shape == rf.shape
    assert out["d_hat"].shape == D[:, tr_idx].shape
    assert (out["delta_s"].abs() <= cfg.model.ds_max + 1e-12).all()

    # data-consistency + supervision loss reaches every branch
    loss = ((out["d_hat"] - D[:, tr_idx]).abs() ** 2).mean() \
        + (out["m_hat"].abs()).mean() \
        + (out["delta_s_coarse"].abs()).mean()
    loss.backward()
    n_with_grad = 0
    n_total = 0
    for name, p in pipe.named_parameters():
        n_total += 1
        if p.grad is not None and torch.isfinite(p.grad).all():
            n_with_grad += 1
        else:
            assert p.grad is None or not torch.isfinite(p.grad).all(), name
    assert n_with_grad == n_total  # every parameter got a finite gradient
    # specifically the neural operator and the unrolled prox; the prox
    # first conv may legitimately have zero grad at init (its output layer
    # is zero-initialized so P = identity + shrinkage at step 0)
    for name, p in pipe.named_parameters():
        if name.startswith("gamma") or name.startswith("prox.0.thresh"):
            assert p.grad is not None and p.grad.abs().sum() > 0, name
        elif name == "neural_operator.attn.bias":
            # A shared shift of all angle logits cancels in softmax; its
            # gradient is mathematically zero, up to rounding error.
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().max() < 1e-5, name
        elif name.startswith("neural_operator") and "blocks" not in name:
            assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_holdout_prediction_protocol():
    """delta_s/m from train angles must give finite predictions on holdout."""
    cfg = _cfg()
    meta = build_meta(cfg)
    pipe = ImagingPipeline(cfg, meta)
    rf, D, ds, m_true, c_true = _batch(cfg)
    tr = torch.tensor(meta.train_idx)
    ho = torch.tensor(meta.hold_idx)
    out = pipe(rf, delta_s_true=ds, eta_mode="net", train_idx=tr,
               pred_idx=ho, return_all=False)
    assert out["d_hat_pred"].shape == D[:, ho].shape
    assert torch.isfinite(out["d_hat_pred"]).all()

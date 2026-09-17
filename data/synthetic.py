"""Synthetic ultrasound dataset for the closed loop.

Each sample: a random smooth background sound-speed map (=> delta_s true on
the fine grid), a random complex scattering image m_true (points + extended
structures), and RF data simulated with the *Born forward model* through the
true background plus complex Gaussian noise.

The simulation uses the same physics family as the pipeline, which is why an
independent full-wave test set is provided in data/fullwave.py (used by
eval.py --source fullwave) to avoid validating the propagator against itself.
"""

import hashlib
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, zoom
from torch.utils.data import Dataset

from common import D_to_rf, build_meta
from physics.imaging import BornModel

__all__ = ["make_phantom", "simulate_sample", "SyntheticUSDataset"]


def _smooth_random_field(shape, corr_len_px, rng, coarsen=16):
    """Random field on a very coarse grid, upsampled and Gaussian-smoothed;
    values in [-1, 1]."""
    h, w = shape
    coarse = rng.standard_normal((max(2, int(np.ceil(h / coarsen))),
                                  max(2, int(np.ceil(w / coarsen)))))
    f = zoom(coarse, (h / coarse.shape[0], w / coarse.shape[1]), order=3)
    f = gaussian_filter(f, sigma=max(1.0, corr_len_px / 2.0))
    f = f / (np.abs(f).max() + 1e-12)
    return f[:h, :w]


def make_phantom(rng, cfg, meta, nz, nx, dz, dx):
    """Random phantom -> (delta_s [nz,nx], m [nz,nx] complex, c [nz,nx])."""
    s = cfg.synth
    c0 = cfg.physics.c0

    corr_px = s.c_corr_len_mm * 1e-3 / dz
    pert = _smooth_random_field((nz, nx), corr_px, rng)
    delta_s = pert * ((1.0 / (c0 * (1 - s.c_perturb_percent / 100.0))
                       - 1.0 / c0))
    c = 1.0 / (1.0 / c0 + delta_s)

    m = np.zeros((nz, nx), dtype=np.complex64)
    sig_px = max(1.0, s.point_sigma_mm * 1e-3 / dz)
    zz, xx = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")

    def blob(cx_px, cz_px, radius, amp):
        g = np.exp(-((xx - cx_px) ** 2 + (zz - cz_px) ** 2)
                   / (2 * radius ** 2))
        m_out = amp * g
        return m_out

    for _ in range(s.n_points):
        cx = rng.uniform(0.2, 0.8) * nx
        cz = rng.uniform(0.08, 0.92) * nz
        amp = rng.uniform(0.4, 1.5) * rng.choice([-1, 1])
        m += blob(cx, cz, sig_px * rng.uniform(0.8, 1.3), amp)
    for _ in range(s.n_extended):
        cx, cz = rng.uniform(0.3, 0.7) * nx, rng.uniform(0.2, 0.8) * nz
        r = rng.uniform(3.0, 6.0) * 1e-3 / dz
        mask = ((xx - cx) ** 2 + (zz - cz) ** 2) < r ** 2
        m += (rng.uniform(0.4, 0.8) * rng.choice([-1, 1])) * mask

    m = m * np.exp(1j * rng.uniform(-0.3, 0.3, size=m.shape))
    m = torch.tensor(m.astype(np.complex64))
    m = m * (cfg.model.m_rms_ref / (m.abs().pow(2).mean().sqrt() + 1e-12))
    return (torch.tensor(delta_s, dtype=torch.float32), m,
            torch.tensor(c, dtype=torch.float32))


def simulate_sample(born, phantom, meta, snr_db, generator=None):
    """Born forward simulation with complex Gaussian noise on D.
    The band window is part of the forward model: D = (F(m) + noise) * W."""
    delta_s, m, c = phantom
    with torch.no_grad():
        u_tx = born.transmit_fields(delta_s)
        D = born.forward(m[None], delta_s, u_tx)          # [n_th, n_w, ne]
    sigma = D.abs().pow(2).mean().sqrt() * (10 ** (-snr_db / 20.0))
    noise = torch.randn(D.shape, generator=generator,
                        dtype=torch.complex64) * sigma
    weight = meta.win if meta.get("response_mode", "legacy_window") == "legacy_window" else np.ones(len(meta.freqs))
    win = torch.tensor(np.asarray(weight), dtype=D.dtype) \
        .view((1,) * max(D.dim() - 2, 0) + (-1, 1))
    D = (D + noise) * win
    rf = D_to_rf(D, meta)
    return rf, D, delta_s, m, c


class SyntheticUSDataset(Dataset):
    """Precomputed in memory; cached on disk keyed by config hash *and* a
    code version salt (physics changes invalidate the cache)."""

    CODE_VER = "4_l11_consistency"  # invalidate data made with old geometry

    def __init__(self, cfg, split, device="cpu", cache_dir="data_cache"):
        super().__init__()
        n = {"train": cfg.train.n_train, "val": cfg.train.n_val,
             "test": cfg.train.n_test}[split]
        seed = cfg.seed + {"train": 0, "val": 1000, "test": 2000}[split]
        os.makedirs(cache_dir, exist_ok=True)
        h = hashlib.md5((self.CODE_VER
                         + repr(sorted(_flatten(cfg)))).encode()).hexdigest()[:10]
        path = os.path.join(cache_dir, f"synth_{split}_{n}_{seed}_{h}.pt")
        if os.path.exists(path):
            blob = torch.load(path, map_location="cpu")
        else:
            meta = build_meta(cfg)
            g = cfg.grid
            born = BornModel(meta, g.nx, g.nz, g.dx, g.dz, cfg.physics.c0)
            rng = np.random.default_rng(seed)
            gen = torch.Generator().manual_seed(seed)
            samples = []
            for _ in range(n):
                ph = make_phantom(rng, cfg, meta, g.nz, g.nx, g.dz, g.dx)
                samples.append(simulate_sample(born, ph, meta,
                                               cfg.train.noise_snr_db, gen))
            blob = {
                "rf": torch.stack([s[0] for s in samples]),
                "D": torch.stack([s[1] for s in samples]),
                "delta_s": torch.stack([s[2] for s in samples]),
                "m": torch.stack([s[3] for s in samples]),
                "c": torch.stack([s[4] for s in samples]),
            }
            torch.save(blob, path)
        self.rf = blob["rf"].to(device)
        self.D = blob["D"].to(device)
        self.delta_s = blob["delta_s"].to(device)
        self.m = blob["m"].to(device)
        self.c = blob["c"].to(device)

    def __len__(self):
        return self.rf.shape[0]

    def __getitem__(self, i):
        return {"rf": self.rf[i], "D": self.D[i], "delta_s": self.delta_s[i],
                "m": self.m[i], "c": self.c[i]}


def _flatten(cfg, prefix=""):
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return {k: str(v) for k, v in out.items() if not isinstance(v, dict)}

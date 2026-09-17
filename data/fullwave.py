"""Independent full-wave test-set generator (frequency-domain finite
difference Helmholtz solver with a matched absorbing layer).

This simulator shares no code path with the angular-spectrum propagator: it
solves the two-way wave equation on a finer grid with sparse LU, so the
resulting data contain all multiples and true two-way physics.  It is used
only for *evaluation* (eval.py --source fullwave), never for training.

For each frequency and transmit angle we solve

    (Lap + k^2 (1 + i sigma)^2) u = -q

once for the background medium (no scatterers) and once for the full medium;
the recorded data is u_full - u_bg at the receiver row (i.e. the scattered
field, with the direct arrival removed exactly by subtraction).
"""

import hashlib
import math
import os
import pickle

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy.ndimage import gaussian_filter, zoom

from common import build_meta

__all__ = ["make_fullwave_phantom", "simulate_fullwave"]


def _pml_sigma(n, pml, ramp=3.0, smax=None):
    """Graded absorption profile for one axis (0 in the interior)."""
    sig = np.zeros(n)
    if smax is None:
        smax = 0.6
    for i in range(n):
        d = min(i, n - 1 - i)
        if d < pml:
            sig[i] = smax * ((pml - d) / pml) ** ramp
    return sig


def _helmholtz_matrix(k, c2d, dx, dz, pml):
    """Sparse operator for (Lap + k^2 (1+i sigma)^2), matched-layer damped.

    4th-order accurate Laplacian in the interior (essential at ~5 points per
    wavelength: a 2nd-order stencil has percent-level phase dispersion),
    falling back to 2nd order within two cells of each boundary.  Edge-based
    assembly keeps the matrix symmetric.
    """
    nz, nx = c2d.shape
    sig_x = _pml_sigma(nx, pml)[None, :]
    sig_z = _pml_sigma(nz, pml)[:, None]
    sig = np.maximum(sig_x, sig_z)
    ksqr = (k ** 2) * (1.0 / c2d ** 2) * (1.0 + 1j * sig) ** 2

    idx = np.arange(nz * nx).reshape(nz, nx)
    rows, cols, vals = [], [], []

    def add_edges(r, c, v):
        rows.append(np.asarray(r).reshape(-1))
        cols.append(np.asarray(c).reshape(-1))
        vals.append(np.broadcast_to(v, np.asarray(r).shape).reshape(-1))

    ix = np.arange(nx)
    iz = np.arange(nz)
    hi_x = (ix >= 2) & (ix <= nx - 3)
    hi_z = (iz >= 2) & (iz <= nz - 3)
    main = ksqr \
        + np.where(hi_x[None, :], -30.0 / (12 * dx ** 2), -2.0 / dx ** 2) \
        + np.where(hi_z[:, None], -30.0 / (12 * dz ** 2), -2.0 / dz ** 2)

    # ---- x-axis stencils ----
    for d in (1, -1):
        c1 = np.where(hi_x[:-1] & hi_x[1:], 16.0 / (12 * dx ** 2),
                      1.0 / dx ** 2)                                  # [nx-1]
        r_ = idx[:, 0:nx - 1] if d == 1 else idx[:, 1:nx]
        c_ = idx[:, 1:nx] if d == 1 else idx[:, 0:nx - 1]
        add_edges(r_, c_, c1[None, :])
        m2 = hi_x[0:nx - 2] & hi_x[2:nx]
        c2v = np.where(m2, -1.0 / (12 * dx ** 2), 0.0)                 # [nx-2]
        r_ = idx[:, 0:nx - 2] if d == 1 else idx[:, 2:nx]
        c_ = idx[:, 2:nx] if d == 1 else idx[:, 0:nx - 2]
        if (c2v != 0).any():
            sel = np.broadcast_to((c2v != 0)[None, :], r_.shape)
            add_edges(r_[sel], c_[sel],
                      np.broadcast_to(c2v[None, :], r_.shape)[sel])
    # ---- z-axis stencils ----
    for d in (1, -1):
        c1 = np.where(hi_z[:-1] & hi_z[1:], 16.0 / (12 * dz ** 2),
                      1.0 / dz ** 2)                                  # [nz-1]
        r_ = idx[0:nz - 1, :] if d == 1 else idx[1:nz, :]
        c_ = idx[1:nz, :] if d == 1 else idx[0:nz - 1, :]
        add_edges(r_, c_, c1[:, None])
        m2 = hi_z[0:nz - 2] & hi_z[2:nz]
        c2v = np.where(m2, -1.0 / (12 * dz ** 2), 0.0)                 # [nz-2]
        r_ = idx[0:nz - 2, :] if d == 1 else idx[2:nz, :]
        c_ = idx[2:nz, :] if d == 1 else idx[0:nz - 2, :]
        if (c2v != 0).any():
            sel = np.broadcast_to((c2v != 0)[:, None], r_.shape)
            add_edges(r_[sel], c_[sel],
                      np.broadcast_to(c2v[:, None], r_.shape)[sel])

    a = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nz * nx, nz * nx)).tocsr()
    a = a + sp.diags(main.reshape(-1))
    return a.tocsc()


def make_fullwave_phantom(rng, cfg, nz, nx, dz, dx):
    """Background +/- perturbation and weak scatterers (as delta-c)."""
    s, fs_ = cfg.synth, cfg.fullwave
    c0 = cfg.physics.c0
    corr_px = s.c_corr_len_mm * 1e-3 / dz
    pert = _smooth_rf((nz, nx), corr_px, rng)
    dc_bg = -pert * c0 * (s.c_perturb_percent / 100.0)
    dc_sc = np.zeros((nz, nx))
    sig_px = max(1.0, s.point_sigma_mm * 1e-3 / dz)
    zz, xx = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    for _ in range(s.n_points):
        cx = rng.uniform(0.25, 0.75) * nx
        cz = rng.uniform(0.15, 0.9) * nz
        g = np.exp(-((xx - cx) ** 2 + (zz - cz) ** 2) / (2 * sig_px ** 2))
        dc_sc += rng.uniform(0.5, 1.0) * rng.choice([-1, 1]) \
            * c0 * fs_.scatter_contrast_percent / 100.0 * g
    c_bg = c0 + dc_bg
    c_full = c_bg + dc_sc
    return c_bg, c_full, dc_sc


def _smooth_rf(shape, corr_len_px, rng, coarsen=16):
    coarse = rng.standard_normal((max(2, int(np.ceil(shape[0] / coarsen))),
                                  max(2, int(np.ceil(shape[1] / coarsen)))))
    f = zoom(coarse, (shape[0] / coarse.shape[0], shape[1] / coarse.shape[1]),
             order=3)
    f = gaussian_filter(f, sigma=max(1.0, corr_len_px / 2.0))
    return f[:shape[0], :shape[1]] / (np.abs(f).max() + 1e-12)


def simulate_fullwave(cfg, n_samples, seed=1234):
    """Returns list of samples: rf [n_th, ne, n_t], D [n_th, n_w, ne],
    c_bg / dc_sc downsampled to the model fine grid, meta."""
    meta = build_meta(cfg)
    g = cfg.grid
    ov = cfg.fullwave.oversample
    nx_fw, nz_fw = g.nx * ov, g.nz * ov
    dxf, dzf = g.dx / ov, g.dz / ov
    pml = cfg.fullwave.pml_cells

    freqs = np.asarray(meta.freqs)
    angles = np.asarray(meta.angles_deg) * np.pi / 180.0
    xe = np.asarray(meta.xe_coords)
    x_fw = np.arange(nx_fw) * dxf

    src_row = pml + 2
    rec_row = pml + 2
    apod = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(len(xe)) + 1)
                              / (len(xe) + 1))
    src_window = np.interp(x_fw, xe, apod, left=0.0, right=0.0)
    rec_cols = np.round(xe / dxf).astype(int)

    rng = np.random.default_rng(seed)
    samples = []
    os.makedirs("data_cache", exist_ok=True)
    cache_key = hashlib.md5((str(seed) + str(n_samples)
                             + str(sorted(cfg.items()))).encode()).hexdigest()[:10]
    cache_path = os.path.join("data_cache", f"fullwave_{cache_key}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            raw_D = pickle.load(f)
    else:
        raw_D = []
        for si in range(n_samples):
            c_bg, c_full, dc_sc = make_fullwave_phantom(rng, cfg, nz_fw, nx_fw,
                                                        dzf, dxf)
            D = np.zeros((len(angles), len(freqs), len(xe)), dtype=np.complex64)
            for iw, f in enumerate(freqs):
                k = 2 * np.pi * f
                Abg = _helmholtz_matrix(k, c_bg, dxf, dzf, pml)
                Afl = _helmholtz_matrix(k, c_full, dxf, dzf, pml)
                lu_bg = spla.splu(Abg)
                lu_fl = spla.splu(Afl)
                for it, th in enumerate(angles):
                    q = np.zeros((nz_fw, nx_fw), dtype=np.complex128)
                    q[src_row, :] = (src_window
                                     * np.exp(1j * k * np.sin(th) * x_fw))
                    ub = lu_bg.solve(-q.reshape(-1)).reshape(nz_fw, nx_fw)
                    uf = lu_fl.solve(-q.reshape(-1)).reshape(nz_fw, nx_fw)
                    scat = uf[rec_row, rec_cols] - ub[rec_row, rec_cols]
                    D[it, iw] = scat
            raw_D.append({"D": D, "c_bg": c_bg, "dc_sc": dc_sc})
            print(f"[fullwave] sample {si + 1}/{n_samples} solved", flush=True)
        with open(cache_path, "wb") as f:
            pickle.dump(raw_D, f)

    from common import D_to_rf
    for raw in raw_D:
        D, c_bg, dc_sc = raw["D"], raw["c_bg"], raw["dc_sc"]
        D = torch.tensor(D)
        sigma = D.abs().pow(2).mean().sqrt() * 10 ** (-cfg.train.noise_snr_db
                                                      / 20.0)
        D = D + torch.randn(D.shape) * sigma
        rf = D_to_rf(D[None], meta)[0]
        # downsample maps to the model fine grid
        c_bg_m = torch.tensor(zoom(c_bg, (g.nz / nz_fw, g.nx / nx_fw),
                                   order=1)[:g.nz, :g.nx], dtype=torch.float32)
        dc_m = torch.tensor(zoom(dc_sc, (g.nz / nz_fw, g.nx / nx_fw),
                                 order=1)[:g.nz, :g.nx], dtype=torch.float32)
        ds_m = (1.0 / c_bg_m - 1.0 / cfg.physics.c0)
        samples.append({"rf": rf, "D": D, "c": c_bg_m, "delta_s": ds_m,
                        "m_ref": dc_m})
    return samples

"""Shared utilities: config handling, acquisition metadata, RF <-> frequency
domain transforms, IQ demodulation and small numeric helpers.

Conventions
-----------
* RF tensors: ``[..., N_theta, N_e, N_t]`` real.
* Frequency-domain data D: ``[..., N_theta, N_freq, N_e]`` complex, positive
  band only (the band selected from the real RF spectrum), windowed.
* Complex images (m, I): ``[..., nz, nx]`` complex; x is the FFT (last) axis.
* Physics marching fields: ``[..., n_theta, n_freq, nz, nx]`` complex.
"""

import copy
import math
import os
import random
import re

import numpy as np
import torch
import yaml

__all__ = [
    "C", "load_config", "set_seed", "get_device", "build_meta",
    "rf_to_D", "D_to_rf", "demod_iq", "envelope", "logcompress",
    "rel_complex_dtype", "relative_residual", "corr2d",
]


# --------------------------------------------------------------------- config
class C(dict):
    """Attribute-style dict for configs. Nested dicts are converted to C at
    construction so that attribute access returns *shared* children (writes
    through cfg.train.x = y persist)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:  # pragma: no cover - misuse guard
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value


def _to_C(d):
    return C({k: (_to_C(v) if isinstance(v, dict) else v)
              for k, v in d.items()})


def to_plain(d):
    """Recursively convert C back to plain dicts (yaml-serializable)."""
    return {k: (to_plain(v) if isinstance(v, dict) else v) for k, v in d.items()}


def _deep_update(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


_NUM_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _coerce_numbers(d):
    """YAML 1.1 parses '2.0e6' (unsigned exponent) as a string; coerce such
    numeric-looking strings back to floats."""
    for k, v in d.items():
        if isinstance(v, dict):
            _coerce_numbers(v)
        elif isinstance(v, str) and _NUM_RE.match(v.strip()):
            d[k] = float(v)
    return d


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _coerce_numbers(cfg)
    if overrides:
        _deep_update(cfg, overrides)
    return _to_C(copy.deepcopy(cfg))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def rel_complex_dtype(cdtype):
    return torch.float64 if cdtype == torch.complex128 else torch.float32


# ----------------------------------------------------------------------- meta
def band_indices(n_t, fs, f0, bandwidth):
    """Indices of positive FFT bins inside the band, ordered by frequency."""
    f = np.fft.fftfreq(n_t, 1.0 / fs)
    lo, hi = f0 * (1.0 - bandwidth / 2.0), f0 * (1.0 + bandwidth / 2.0)
    idx = np.where((f > lo) & (f < hi))[0]
    return np.sort(idx)


def select_evenly(idx, n_freq):
    """Sub-select band bins. n_freq <= 0 (or >= available) keeps the full
    contiguous band - required for clean time-domain pulses. Otherwise bins
    are sub-sampled with an INTEGER stride (uniform spacing); a fractional
    stride would break the time-domain waveform (non-harmonic tones)."""
    if n_freq <= 0 or n_freq >= len(idx):
        return idx
    step = int(np.ceil(len(idx) / n_freq))
    return idx[::step]


def band_window(freqs, f0, bandwidth):
    """Smooth raised-cosine window over the band (1 at f0, ->0 at edges)."""
    half = f0 * bandwidth / 2.0
    x = np.clip((freqs - f0) / half, -1.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def build_meta(cfg, device=None):
    """Acquisition metadata shared by data / physics / models."""
    p, a, g = cfg.physics, cfg.acq, cfg.grid
    n_t = a.n_t
    band = band_indices(n_t, p.fs, p.f0, p.bandwidth)
    band = select_evenly(band, a.n_freq)
    freqs = np.fft.fftfreq(n_t, 1.0 / p.fs)[band]
    win = band_window(freqs, p.f0, p.bandwidth)

    angles_deg = np.linspace(-a.angle_span_deg, a.angle_span_deg, a.n_angles)
    arr = cfg.array
    xe = (np.arange(arr.n_elements) - (arr.n_elements - 1) / 2.0) * arr.pitch
    xe = xe + arr.get("center_x", g.nx * g.dx / 2.0)
    x0, z0 = float(g.get("x0", 0.0)), float(g.get("z0", 0.0))
    tref = np.asarray(a.get("t_ref_s", [0.0] * a.n_angles), dtype=np.float64)
    if tref.shape != (a.n_angles,) or not np.isfinite(tref).all():
        raise ValueError("acq.t_ref_s must contain one finite time per angle")
    response = np.ones(len(freqs), dtype=np.complex128)
    response_path = p.get("response_path")
    if response_path:
        with np.load(response_path, allow_pickle=False) as source:
            if source["freqs"].shape != freqs.shape or not np.allclose(source["freqs"], freqs, rtol=0, atol=1e-5):
                raise ValueError("system response frequency grid does not match config")
            response = source["response"].astype(np.complex128)
        if response.shape != freqs.shape or not np.isfinite(response).all():
            raise ValueError("invalid system response")

    hold_idx = np.arange(a.n_angles)[np.arange(a.n_angles) % a.holdout_stride
                                     == a.holdout_stride - 1]
    train_idx = np.setdiff1d(np.arange(a.n_angles), hold_idx)

    dec = max(1, int(p.fs / (p.f0 * 3)))  # IQ decimation (>=3x oversample)
    meta = C(
        n_t=n_t, fs=p.fs, f0=p.f0, bandwidth=p.bandwidth, dec=dec,
        fs_iq=p.fs / dec,
        band_idx=band, freqs=freqs, win=win,
        angles_deg=angles_deg, xe_coords=xe,
        train_idx=train_idx, hold_idx=hold_idx,
        n_elements=cfg.array.n_elements, x0=x0, z0=z0,
        t_ref_s=tref, system_response=response,
        response_mode=p.get("response_mode", "legacy_window"),
    )
    if device is not None:
        for k in ("band_idx",):
            meta[k] = torch.as_tensor(meta[k], dtype=torch.long, device=device)
    return meta


# ------------------------------------------------------------ RF <-> frequency
def rf_to_D(rf, meta, win=False):
    """``[..., N_theta, N_e, N_t]`` real RF -> ``[..., N_theta, N_freq, N_e]``
    complex band data (the conjugate of the positive-bin DFT, matching the
    e^{-i omega t} physical convention; exact inverse of :func:`D_to_rf`).
    The band window is *not* applied by default: it is part of the forward
    model and applied identically to predictions and targets elsewhere.
    Differentiable."""
    idx = torch.as_tensor(meta.band_idx, device=rf.device)
    spec = torch.fft.fft(rf, dim=-1)[..., idx].conj()
    if win:
        w = torch.as_tensor(np.asarray(meta.win), dtype=spec.dtype,
                            device=rf.device)
        spec = spec * w
    return spec.transpose(-1, -2).contiguous()


def D_to_rf(D, meta):
    """``[..., N_theta, N_freq, N_e]`` -> real band-limited RF
    ``[..., N_theta, N_e, N_t]``.

    With the e^{-i omega t} convention a scatterer with arrival t0 has
    spectrum ~ e^{+i w t0}; the real time signal is
    x(t) = Re{ integral D(w) e^{-i w t} dw }, i.e. the DFT places conj(D)
    at the positive bins (otherwise the pulse lands at T - t0: the time
    axis comes out reversed).
    """
    idx = torch.as_tensor(meta.band_idx, device=D.device)
    n_t = meta.n_t
    spec = D.transpose(-1, -2).conj()        # [..., N_e, N_freq]
    full = torch.zeros(*spec.shape[:-1], n_t, dtype=spec.dtype,
                       device=spec.device)
    full[..., idx] = spec
    full[..., (-idx) % n_t] = spec.conj()
    return torch.fft.ifft(full, dim=-1).real


def demod_iq(rf, meta):
    """Complex baseband IQ: demodulate at f0, low-pass, decimate.
    ``[B, N_theta, N_e, N_t]`` -> ``[B, N_theta, N_e, N_t_iq]`` complex."""
    n_t = rf.shape[-1]
    t = torch.arange(n_t, device=rf.device, dtype=rf.dtype) / meta.fs
    carrier = torch.exp(-2j * math.pi * meta.f0 * t)
    base = rf * carrier
    spec = torch.fft.fft(base, dim=-1)
    f = torch.fft.fftfreq(n_t, 1.0 / meta.fs, device=rf.device)
    half = meta.f0 * meta.bandwidth / 2.0 * 1.2
    mask = (f.abs() <= half).to(spec.dtype)
    base = torch.fft.ifft(spec * mask, dim=-1)
    return base[..., ::meta.dec].contiguous()


def iq_delay_sum(iq, idx, apod, meta):
    """Interpolate baseband IQ, restore carrier phase, then sum receivers."""
    B, n_th, ne, T = iq.shape
    valid = (idx >= 0) & (idx <= T - 1)
    safe = idx.clamp(0, T - 2)
    i0 = safe.floor().long(); fr = safe - i0
    elements = torch.arange(ne, device=iq.device)[:, None, None]
    images = []
    for it in range(n_th):
        val = iq[:, it, elements, i0[it]] * (1 - fr[it]) + iq[:, it, elements, i0[it] + 1] * fr[it]
        phase = torch.exp(2j * math.pi * meta.f0 * idx[it] / meta.fs_iq)
        images.append((val * phase * valid[it] * apod[None, :, None, None]).sum(1))
    return torch.stack(images, dim=1)


# ---------------------------------------------------------------- misc imaging
def envelope(x):
    return x.abs()


def logcompress(x, dr=60.0):
    """20*log10 envelope, normalized to peak, clipped at -dr dB."""
    a = x.abs()
    a = a / (a.amax(dim=(-2, -1), keepdim=True) + 1e-12)
    db = 20.0 * torch.log10(a + 1e-12)
    return torch.clamp(db, min=-dr, max=0.0)


def relative_residual(pred, target):
    """||pred - target|| / ||target|| over all entries."""
    num = (pred - target).abs().pow(2).sum()
    den = target.abs().pow(2).sum()
    return (num / (den + 1e-20)).sqrt()


def corr2d(a, b):
    """Pearson correlation between two (possibly complex) images."""
    a = a - a.mean()
    b = b - b.mean()
    num = (a * b.conj()).real.sum()
    den = (a.abs().pow(2).sum() * b.abs().pow(2).sum()).sqrt()
    return (num / (den + 1e-20)).real

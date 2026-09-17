"""Neural operator estimating the coarse-grid background slowness
perturbation delta_s from RF/IQ data.

Pipeline (section 4.2 of the spec):

1. shared RF/IQ encoder (per-angle features over the receive aperture);
2. angle aggregation with a geometry-aware *integration layer*: RF features
   are queried at the straight-ray two-way travel time t(theta, e; x, z)
   (coordinate query - the time axis is mapped to space through the actual
   acquisition geometry, never by equating t with z);
3. a complex delay-and-sum image in the homogeneous background is added as
   extra channels (initial physical image);
4. attention pooling over transmit angles using the true angles (sin/cos
   positional encoding);
5. a small FNO refines the coarse feature map;
6. a parameter head outputs bounded, smoothed delta_s via tanh.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from physics.imaging import delay_integrate, straight_ray_delays
from models.encoder import RFEncoder

__all__ = ["NeuralOperator", "FNO2d"]


class SpectralConv2d(nn.Module):
    """Standard Fourier neural-operator layer (2D, truncated modes)."""

    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * modes[0] * modes[1])
        self.w_pos = nn.Parameter(
            torch.randn(in_ch, out_ch, *modes, dtype=torch.cfloat)
            * math.sqrt(scale))
        self.w_neg = nn.Parameter(
            torch.randn(in_ch, out_ch, *modes, dtype=torch.cfloat)
            * math.sqrt(scale))

    def forward(self, x):
        B, _, H, W = x.shape
        xft = torch.fft.rfft2(x)
        # keep positive/negative mode blocks disjoint: at most H//2 rows
        m1 = min(self.modes[0], H // 2)
        m2 = min(self.modes[1], xft.shape[-1])
        out = torch.zeros(B, self.w_pos.shape[1], H, xft.shape[-1],
                          dtype=xft.dtype, device=x.device)
        spec_pos = torch.einsum("bixy,ioxy->boxy",
                                xft[:, :, :m1, :m2], self.w_pos[:, :, :m1, :m2])
        spec_neg = torch.einsum("bixy,ioxy->boxy",
                                xft[:, :, -m1:, :m2], self.w_neg[:, :, :m1, :m2])
        out[:, :, :m1, :m2] = spec_pos
        out[:, :, -m1:, :m2] = spec_neg
        return torch.fft.irfft2(out, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, in_ch, width, modes, layers, out_ch=1):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(SpectralConv2d(width, width, modes))
            self.skips.append(nn.Conv2d(width, width, 1))
        self.head = nn.Conv2d(width, out_ch, 3, padding=1)

    def forward(self, x):
        return self.head(self.forward_features(x))

    def forward_features(self, x):
        """Return the spatial FNO latent map before the task-specific head."""
        x = self.lift(x)
        for spec, skip in zip(self.blocks, self.skips):
            x = F.gelu(spec(x) + skip(x))
        return x


class NeuralOperator(nn.Module):
    """RF/IQ -> coarse-grid delta_s (bounded, smooth)."""

    def __init__(self, cfg, meta, nx, nz, dx, dz):
        super().__init__()
        m = cfg.model
        self.encoder = RFEncoder(2, tuple(m.enc_channels))
        c_enc = self.encoder.out_ch
        self.ds_max = m.ds_max
        self.nx, self.nz, self.dx, self.dz = nx, nz, dx, dz
        self.meta = meta
        self.normalize_iq = bool(cfg.model.get("normalize_iq", False))

        f = cfg.coarse.factor
        self.nsx, self.nsz = nx // f, nz // f
        X = meta.get("x0", 0.0) + (np.arange(self.nsx) * f + (f - 1) / 2) * dx
        Z = meta.get("z0", 0.0) + (np.arange(self.nsz) * f + (f - 1) / 2) * dz
        t = straight_ray_delays(meta.angles_deg, meta.xe_coords, X, Z,
                                cfg.physics.c0, meta.get("t_ref_s"))
        # IQ-rate indices for the complex DAS channel
        idx_iq = torch.tensor(t / (1.0 / meta.fs_iq), dtype=torch.float32)
        # feature-rate indices for the encoder channel (time stride 8)
        feat_fs = meta.fs_iq / self.encoder.TIME_STRIDE
        idx_f = torch.tensor(t / (1.0 / feat_fs), dtype=torch.float32)
        apod = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(meta.n_elements) + 1)
                                  / (meta.n_elements + 1))
        self.register_buffer("idx_iq", idx_iq, persistent=False)
        self.register_buffer("idx_f", idx_f, persistent=False)
        self.register_buffer("apod", torch.tensor(apod, dtype=torch.float32),
                             persistent=False)
        ang = torch.tensor(np.asarray(meta.angles_deg) * np.pi / 180.0,
                           dtype=torch.float32)
        self.register_buffer("ang_pe", torch.stack([ang.sin(), ang.cos()],
                                                   dim=-1), persistent=False)

        # per-angle projection before integration + attention pooling
        self.proj = nn.Conv2d(c_enc, 8, 1)
        self.attn = nn.Conv2d(8 + 2, 1, 1)
        # number of (non-holdout) angles the operator is fed at call time
        hold = np.arange(cfg.acq.n_angles) % cfg.acq.holdout_stride \
            == cfg.acq.holdout_stride - 1
        self.n_call_angles = int(cfg.acq.n_angles - hold.sum())
        n_in = 8 + 8 + 2 + 2 * self.n_call_angles  # pooled enc + mean +
        #      coords + per-angle complex DAS (Re/Im), fixed channel count
        self.pre_fno = nn.Conv2d(n_in, m.fno_width, 1)
        self.fno = FNO2d(m.fno_width, m.fno_width, tuple(m.fno_modes),
                         m.fno_layers, out_ch=1)

        # fixed Gaussian smoothing of the estimated slowness (separable)
        k = 5
        g = torch.exp(-0.5 * ((torch.arange(k) - k // 2) / 1.0) ** 2)
        g = (g / g.sum())
        self.register_buffer("sm_h", g.view(1, 1, 1, k), persistent=False)
        self.register_buffer("sm_v", g.view(1, 1, k, 1), persistent=False)

    def extract_features(self, rf_iq, angles_idx, coords=None, meta=None):
        """rf_iq: [B, n_theta', ne, n_t_iq] complex IQ for the selected angles;
        angles_idx: long tensor indexing these angles into the full angle list
        (used to pick the correct travel-time tables and angle encodings).
        Returns the geometry-aware spatial feature map before the FNO."""
        B, n_th = rf_iq.shape[:2]
        angles_idx = angles_idx.to(rf_iq.device)
        iq2 = torch.stack([rf_iq.real, rf_iq.imag], dim=2)  # [B, th, 2, ne, T]
        if self.normalize_iq:
            rms = iq2.square().mean(dim=(1, 2, 3, 4), keepdim=True).sqrt().detach()
            iq2 = iq2 / rms.clamp_min(1e-12)

        feats = self.encoder(iq2)                            # [B, th, C, ne, T']
        feats = self.proj(feats.reshape(B * n_th, *feats.shape[2:]))
        feats = feats.reshape(B, n_th, -1, *feats.shape[-2:])
        idx_f = self.idx_f[angles_idx]
        fint = delay_integrate(feats, idx_f, self.apod)      # [B, th, 8, nsz, nsx]
        fint = fint / (fint.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True)
                       .sqrt().detach() + 1e-9)

        # per-angle complex DAS images: the residual phase vs the background
        # delay model encodes the slowness perturbation and varies with the
        # angle (different ray paths), so angles are kept as separate
        # channels rather than averaged
        from common import iq_delay_sum
        das = iq_delay_sum(rf_iq, self.idx_iq[angles_idx], self.apod, self.meta)
        # [B, n_th, nsz, nsx] complex -> [B, 2*n_th, nsz, nsx]
        das_ch = torch.cat([das.real, das.imag], dim=1)
        das_ch = das_ch / (das_ch.pow(2).mean(dim=(1, 2, 3), keepdim=True)
                           .sqrt().detach() + 1e-12)

        # attention pooling over angles (true-angle positional encoding)
        pe_all = self.ang_pe[angles_idx]
        logits = []
        for it in range(n_th):
            pe = pe_all[it].view(1, 2, 1, 1).expand(B, 2, *fint.shape[-2:])
            logits.append(self.attn(torch.cat([fint[:, it], pe], dim=1)))
        logits = torch.stack(logits, dim=1)                  # [B, th, 1, H, W]
        w_att = torch.softmax(logits, dim=1)
        pooled = (fint * w_att).sum(dim=1)
        mean_pooled = fint.mean(dim=1)
        # absolute coordinates
        Bz, Bx = torch.meshgrid(
            torch.linspace(0, 1, self.nsz, device=fint.device),
            torch.linspace(0, 1, self.nsx, device=fint.device),
            indexing="ij")
        coord = torch.stack([Bz, Bx], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

        x = torch.cat([pooled, mean_pooled, das_ch, coord], dim=1)
        return x

    def forward(self, rf_iq, angles_idx, coords=None, meta=None):
        """Default slowness path, preserved for existing model checkpoints."""
        features = self.extract_features(rf_iq, angles_idx, coords, meta)
        out = torch.tanh(self.fno(self.pre_fno(features))) * self.ds_max
        out = F.conv2d(F.conv2d(out, self.sm_h, padding=(0, 2)),
                       self.sm_v, padding=(2, 0))
        return {"delta_s": out, "features": features}

"""Neural operator for geometry-aware RF/IQ feature extraction.

Pipeline:

1. shared RF/IQ encoder (per-angle features over the receive aperture);
2. geometry-aware delay integration at straight-ray two-way travel times;
3. complex homogeneous-background DAS features;
4. angle-set invariant attention/mean pooling using true-angle encodings;
5. a small FNO refines the coarse spatial feature map.

The default ``forward`` path still supports the legacy slowness-regression
model.  Phase-screen models reuse ``extract_features`` and the FNO latent map.
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
    """Geometry-aware RF/IQ operator with angle-set invariant features."""

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
        idx_iq = torch.tensor(t / (1.0 / meta.fs_iq), dtype=torch.float32)
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

        self.proj = nn.Conv2d(c_enc, 8, 1)
        self.attn = nn.Conv2d(8 + 2, 1, 1)

        # Fixed channel count independent of the number of transmit angles:
        #   8 attention-pooled encoded features
        #   8 mean-pooled encoded features
        #   2 attention-pooled complex DAS channels (Re/Im)
        #   2 mean-pooled complex DAS channels (Re/Im)
        #   2 normalized coordinates
        n_in = 8 + 8 + 2 + 2 + 2
        self.pre_fno = nn.Conv2d(n_in, m.fno_width, 1)
        self.fno = FNO2d(m.fno_width, m.fno_width, tuple(m.fno_modes),
                         m.fno_layers, out_ch=1)

        # Legacy slowness-output smoothing retained for existing model users.
        k = 5
        g = torch.exp(-0.5 * ((torch.arange(k) - k // 2) / 1.0) ** 2)
        g = g / g.sum()
        self.register_buffer("sm_h", g.view(1, 1, 1, k), persistent=False)
        self.register_buffer("sm_v", g.view(1, 1, k, 1), persistent=False)

    def extract_features(self, rf_iq, angles_idx, coords=None, meta=None):
        """Return an angle-set invariant geometry-aware spatial feature map.

        Parameters
        ----------
        rf_iq:
            ``[B, n_theta, n_elements, n_time]`` complex IQ for selected angles.
        angles_idx:
            Indices of those angles in the full acquisition geometry.

        The output channel count is fixed, so the same trained model can be
        evaluated with different subsets/numbers of transmit angles.
        """
        B, n_th = rf_iq.shape[:2]
        if n_th < 1:
            raise ValueError("at least one transmit angle is required")
        angles_idx = angles_idx.to(rf_iq.device)
        if len(angles_idx) != n_th:
            raise ValueError("angles_idx length must match RF angle dimension")

        iq2 = torch.stack([rf_iq.real, rf_iq.imag], dim=2)
        if self.normalize_iq:
            rms = iq2.square().mean(dim=(1, 2, 3, 4), keepdim=True).sqrt().detach()
            iq2 = iq2 / rms.clamp_min(1e-12)

        feats = self.encoder(iq2)
        feats = self.proj(feats.reshape(B * n_th, *feats.shape[2:]))
        feats = feats.reshape(B, n_th, -1, *feats.shape[-2:])
        fint = delay_integrate(feats, self.idx_f[angles_idx], self.apod)
        fint = fint / (fint.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True)
                       .sqrt().detach() + 1e-9)

        # Angle attention is based on encoded RF features plus true-angle PE.
        pe_all = self.ang_pe[angles_idx]
        logits = []
        for it in range(n_th):
            pe = pe_all[it].view(1, 2, 1, 1).expand(B, 2, *fint.shape[-2:])
            logits.append(self.attn(torch.cat([fint[:, it], pe], dim=1)))
        logits = torch.stack(logits, dim=1)  # [B,theta,1,H,W]
        w_att = torch.softmax(logits, dim=1)
        pooled = (fint * w_att).sum(dim=1)
        mean_pooled = fint.mean(dim=1)

        # Complex homogeneous-background DAS is also pooled as a set rather
        # than concatenated angle-by-angle.  This removes the previous hard
        # dependency of the network input channels on n_theta.
        from common import iq_delay_sum
        das = iq_delay_sum(rf_iq, self.idx_iq[angles_idx], self.apod, self.meta)
        das_ri = torch.stack([das.real, das.imag], dim=2)  # [B,theta,2,H,W]
        das_scale = das_ri.square().mean(dim=(1, 2, 3, 4), keepdim=True)\
            .sqrt().detach().clamp_min(1e-12)
        das_ri = das_ri / das_scale
        das_att = (das_ri * w_att).sum(dim=1)
        das_mean = das_ri.mean(dim=1)

        Bz, Bx = torch.meshgrid(
            torch.linspace(0, 1, self.nsz, device=fint.device),
            torch.linspace(0, 1, self.nsx, device=fint.device),
            indexing="ij")
        coord = torch.stack([Bz, Bx], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

        return torch.cat([pooled, mean_pooled, das_att, das_mean, coord], dim=1)

    def forward(self, rf_iq, angles_idx, coords=None, meta=None):
        """Legacy slowness path, now using angle-set invariant features."""
        features = self.extract_features(rf_iq, angles_idx, coords, meta)
        out = torch.tanh(self.fno(self.pre_fno(features))) * self.ds_max
        out = F.conv2d(F.conv2d(out, self.sm_h, padding=(0, 2)),
                       self.sm_v, padding=(2, 0))
        return {"delta_s": out, "features": features}

"""RF/IQ encoder: convolutions over the (element, time) aperture, shared
across all transmit angles."""

import math

import torch
import torch.nn as nn

__all__ = ["RFEncoder"]


class RFEncoder(nn.Module):
    """Input [B, n_theta, 2, ne, n_t_iq] (IQ real/imag) ->
    features [B, n_theta, C, ne, n_t_iq / 8].

    Three stride-(1,2) convolutions decimate time by 8 (aperture axis kept).
    The same weights are applied to every transmit angle.
    """

    TIME_STRIDE = 8

    def __init__(self, in_ch=2, channels=(32, 48, 24)):
        super().__init__()
        layers = []
        c_in = in_ch
        for c in channels:
            layers += [
                nn.Conv2d(c_in, c, kernel_size=(3, 7), padding=(1, 3),
                          stride=(1, 2)),
                nn.GroupNorm(math.gcd(8, c), c),
                nn.GELU(),
            ]
            c_in = c
        self.net = nn.Sequential(*layers)
        self.out_ch = channels[-1]

    def forward(self, iq):
        B, n_th = iq.shape[:2]
        f = self.net(iq.reshape(B * n_th, *iq.shape[2:]))
        return f.reshape(B, n_th, *f.shape[1:])

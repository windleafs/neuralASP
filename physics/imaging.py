"""Single-scattering (Born) forward operator F_eta, its Hermitian adjoint
F_eta^H, adjoint imaging condition and an analytic delay-and-sum baseline.

Forward model (per transmit angle theta, per frequency omega):

    u_theta^tx(r, omega)              transmit field (angular-spectrum march)
    q(r) = w(z, omega) m(r) u_theta^tx(r, omega)
    d_hat(theta, e, omega) = S_e [ sum_z P_{z->0} q ]        (march_up + sample)

Adjoint (back-march of data, then correlation with the transmit field):

    b = S^H d  at the surface,  b_z = (A_{z-1}^H ... A_0^H b)
    (F_eta^H d)(r) = sum_{theta,omega} conj(w(z,omega) u_theta^tx(r,omega)) b_z(r)

S is linear interpolation of the surface field at the element positions
(its exact transpose is used for S^H).  w(z,omega) is an optional 2D
far-field spreading weight.  F and F^H are built from the same primitives,
so <F m, d> = <m, F^H d> exactly (tested in tests/test_adjoint.py).
"""

import math

import numpy as np
import torch
import torch.nn as nn

from physics.angular_spectrum import HeterogeneousAngularSpectrum, rel_real

__all__ = ["BornModel", "DelaySumBaseline", "straight_ray_delays",
           "delay_integrate"]


def straight_ray_delays(angles, xe, X, Z, c0, t_ref_s=None):
    """Two-way straight-ray travel times t[theta, e, iz, ix] in seconds.

    Plane-wave transmit at angle theta (from the array plane, t=0 at z=0)
    plus straight receiving path from (x, z) to element xe, both at c0.
    This is only used for *feature extraction* (coordinate query of the RF
    data) and the DAS baseline, never as an imaging physics assumption.
    """
    ath = np.deg2rad(np.asarray(angles, dtype=np.float64).reshape(-1))
    xe = np.asarray(xe, dtype=np.float64).reshape(-1)
    Xv = np.asarray(X, dtype=np.float64).reshape(1, 1, 1, -1)  # [1,1,1,nx]
    Zv = np.asarray(Z, dtype=np.float64).reshape(1, 1, -1, 1)  # [1,1,nz,1]
    # Explicit [theta, element, depth, lateral] layout.
    tx = (Xv * np.sin(ath)[:, None, None, None]
          + Zv * np.cos(ath)[:, None, None, None]) / c0
    if t_ref_s is not None:
        tx = tx + np.asarray(t_ref_s).reshape(-1, 1, 1, 1)
    rx = np.sqrt((Xv - xe[None, :, None, None]) ** 2 + Zv ** 2) / c0
    return tx + rx


class BornModel(nn.Module):
    """Differentiable single-scattering transmit-receive model."""

    def __init__(self, meta, nx, nz, dx, dz, c0, dtype=torch.complex64,
                 eps=1e-6, spreading="farfield2d"):
        super().__init__()
        self.nx, self.nz, self.dx, self.dz, self.c0 = nx, nz, dx, dz, c0
        self.cdtype = dtype
        self.asp = HeterogeneousAngularSpectrum(nx, dx, dz, c0, eps, dtype)

        angles = np.asarray(meta.angles_deg, dtype=np.float64) * np.pi / 180.0
        xe = np.asarray(meta.xe_coords, dtype=np.float64)
        omega = 2.0 * math.pi * np.asarray(meta.freqs, dtype=np.float64)
        self.register_buffer("angles", torch.tensor(angles, dtype=torch.float64),
                             persistent=False)
        self.register_buffer("omega", torch.tensor(omega), persistent=False)
        self.register_buffer("xe", torch.tensor(xe), persistent=False)

        rd = rel_real(dtype)
        self.register_buffer("angles_", self.angles.to(rd), persistent=False)
        self.register_buffer("omega_", self.omega.to(rd), persistent=False)

        self.x0 = float(meta.get("x0", 0.0))
        self.z0 = float(meta.get("z0", 0.0))
        if self.z0 < 0:
            raise ValueError("grid.z0 must be at or below the receiver surface")
        x = self.x0 + torch.arange(nx, dtype=rd) * dx
        self.register_buffer("x", x, persistent=False)
        tref = torch.as_tensor(meta.get("t_ref_s", np.zeros(len(angles))), dtype=rd)
        timing = torch.exp(1j * self.omega_[None, :] * tref[:, None])
        response = torch.as_tensor(meta.get("system_response", np.ones(len(omega))), dtype=dtype)
        self.register_buffer("source_response", timing.to(dtype) * response[None, :])
        k = self.omega_[:, None] / c0
        kz = torch.sqrt((k ** 2 - self.asp.kx[None, :] ** 2 + 1j * eps * k ** 2).to(dtype))
        self.register_buffer("surface_transfer", torch.exp(1j * kz * self.z0), persistent=False)

        # transmit apodization: Hann over the physical aperture, 0 outside
        apod_e = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(len(xe)) + 1)
                                    / (len(xe) + 1))
        apod = np.interp(x.numpy(), xe, apod_e, left=0.0, right=0.0)
        self.register_buffer("apod", torch.tensor(apod, dtype=rd),
                             persistent=False)

        # receiver sampling (linear interpolation) and its transpose
        pos = ((torch.tensor(xe, dtype=rd) - self.x0) / dx)
        i0 = pos.floor().long().clamp(0, nx - 2)
        fr = pos - i0
        self.register_buffer("smp_i0", i0, persistent=False)
        self.register_buffer("smp_fr", fr, persistent=False)

        # depth weight w(z, omega): optional 2D far-field spreading,
        # normalized to 1 at reference depth z_ref = 15 dz for f0.  Depths
        # are the slab nodes i*dz (where the marching fields live).
        zc = self.z0 + torch.arange(nz, dtype=rd) * dz
        k = self.omega_.view(-1, 1) / c0                   # [n_w, 1]
        k_ref = (2.0 * math.pi * float(np.asarray(meta.f0))) / c0
        z_ref = 15.0 * dz
        if spreading == "farfield2d":
            w = torch.sqrt(k_ref * z_ref / (k * torch.clamp(zc, min=dz)[None]))
        else:
            w = torch.ones_like(k * zc[None])
        w = w * dz
        # stored right-aligned as [1, n_w, nz, 1] to broadcast against
        # [..., n_theta, n_freq, nz, nx] with or without a batch dim
        self.register_buffer("w_z", w[None, :, :, None], persistent=False)

    # ------------------------------------------------------------------ pieces
    def sample(self, u):
        """u [..., nx] -> [..., n_e]: interpolate at element positions."""
        v0 = u[..., self.smp_i0]
        v1 = u[..., self.smp_i0 + 1]
        return v0 * (1.0 - self.smp_fr) + v1 * self.smp_fr

    def scatter(self, d):
        """Exact transpose of :meth:`sample` (real interpolation weights)."""
        shape = d.shape[:-1]
        out = torch.zeros(*shape, self.nx, dtype=d.dtype, device=d.device)
        flat = out.view(-1, self.nx)
        n_b = flat.shape[0]
        i0 = self.smp_i0.expand(n_b, -1)
        i1 = i0 + 1
        w0 = (1.0 - self.smp_fr).expand(n_b, -1) * d.reshape(n_b, -1)
        w1 = self.smp_fr.expand(n_b, -1) * d.reshape(n_b, -1)
        flat.scatter_add_(1, i0, w0)
        flat.scatter_add_(1, i1, w1)
        return out

    # ------------------------------------------------------------------- main
    def transmit_fields(self, delta_s, angles_idx=None):
        """Downward-marched transmit plane waves.

        delta_s: [..., nz, nx] real slowness perturbation (fine grid).
        Returns [..., n_theta', n_freq, nz, nx] complex fields at every slab.
        """
        if angles_idx is None:
            ang = self.angles_
        else:
            ang = self.angles_[angles_idx]
        k0 = self.omega_ / self.c0                        # [n_w]
        phase = (ang[:, None, None].sin() * k0[None, :, None]
                 * self.x[None, None, :])                 # [n_th, n_w, nx]
        u0 = self.apod[None, None, :] * torch.exp(1j * phase.to(self.cdtype))
        response = self.source_response if angles_idx is None else self.source_response[angles_idx]
        u0 = u0 * response[..., None]
        u0 = self.asp._ifft(self.asp._fft(u0) * self.surface_transfer)
        if delta_s.dim() == 3:                            # batched slowness
            u0 = u0.unsqueeze(0).expand(delta_s.shape[0], *u0.shape)
        return self.asp.forward(u0, delta_s, self.omega_)

    def forward(self, m, delta_s, u_tx, angles_idx=None):
        """Predict data d_hat = F_eta(m): [..., n_theta', n_freq, n_e].

        m: [nz, nx] or [B, nz, nx] complex scattering image.
        u_tx: transmit fields from :meth:`transmit_fields` (same angles).
        """
        m_b = m.reshape(m.shape[:-2]
                        + (1,) * (u_tx.dim() - m.dim())
                        + m.shape[-2:])
        q = u_tx * self.w_z * m_b
        d = self.asp.march_up(q, delta_s, self.omega_)
        d = self.asp._ifft(self.asp._fft(d) * self.surface_transfer)
        return self.sample(d)

    def adjoint(self, D, u_tx, delta_s, angles_idx=None):
        """Adjoint image F_eta^H(D) (== adjoint imaging condition):
        [nz, nx] (or [B, nz, nx]) complex, summed over theta and omega."""
        b0 = self.scatter(D)
        b0 = self.asp._ifft(self.asp._fft(b0) * self.surface_transfer.conj())
        b = self.asp.adjoint(b0, delta_s, self.omega_)     # [..., n_th, n_w, nz, nx]
        img = (b * (u_tx * self.w_z).conj()).sum(dim=(-3, -4))
        return img

    def illumination(self, u_tx):
        """Diagonal of the normal operator: sum_{theta,omega}
        |u_tx w|^2, shape like the adjoint image ([nz, nx] / [B, nz, nx])."""
        return (u_tx * self.w_z).abs().pow(2).sum(dim=(-3, -4))


def delay_integrate(x, idx, apod):
    """Sample ``x`` along its last (time) axis at fractional, element- and
    angle-dependent indices, then sum over the element axis.

    x:   [B, n_th, C, ne, T]  (real or complex; T is time samples)
    idx: [n_th, ne, K1, K2]   fractional sample indices
    apod:[ne]                 element weights
    Returns [B, n_th, C, K1, K2]. Differentiable (linear in x).
    """
    B, n_th, C, ne, T = x.shape
    valid = (idx >= 0) & (idx <= T - 1)
    idx = idx.clamp(0.0, float(T - 2))
    i0 = idx.floor().long()
    fr = (idx - i0).to(x.dtype)
    e_idx = torch.arange(ne, device=x.device).view(ne, 1, 1)
    e_idx = e_idx.expand(ne, *idx.shape[-2:])
    out = x.new_zeros(B, n_th, C, *idx.shape[-2:])
    for it in range(n_th):  # loop over theta to bound peak memory
        xt = x[:, it]                                   # [B, C, ne, T]
        v0 = xt[:, :, e_idx, i0[it]]                    # [B, C, ne, K1, K2]
        v1 = xt[:, :, e_idx, i0[it] + 1]
        val = (v0 * (1.0 - fr[it]) + v1 * fr[it]) * valid[it]
        out[:, it] = (val * apod.to(x.dtype)[None, None, :, None, None]).sum(2)
    return out


class DelaySumBaseline(nn.Module):
    """Classic straight-ray delay-and-sum on IQ data (comparison baseline)."""

    def __init__(self, meta, nx, nz, dx, dz, c0):
        super().__init__()
        self.nx, self.nz = nx, nz
        self.meta = meta
        X = meta.get("x0", 0.0) + np.arange(nx) * dx
        Z = meta.get("z0", 0.0) + np.arange(nz) * dz
        t = straight_ray_delays(meta.angles_deg, meta.xe_coords, X, Z, c0, meta.get("t_ref_s"))
        idx = torch.tensor(t * meta.fs_iq, dtype=torch.float32)
        apod = 0.5 - 0.5 * np.cos(2.0 * np.pi
                                  * (np.arange(meta.n_elements) + 1)
                                  / (meta.n_elements + 1))
        self.register_buffer("idx", idx, persistent=False)
        self.register_buffer("apod", torch.tensor(apod, dtype=torch.float32),
                             persistent=False)

    def forward(self, iq):
        """iq: [B, n_theta, n_e, n_t_iq] complex -> [B, nz, nx] complex DAS."""
        from common import iq_delay_sum
        return iq_delay_sum(iq, self.idx, self.apod, self.meta).mean(dim=1)

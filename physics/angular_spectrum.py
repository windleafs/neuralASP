"""Differentiable heterogeneous angular-spectrum propagation (2D, one-way).

Split-step Fourier marching for the scalar wave field u(x, z, omega) with
time convention exp(-i omega t):

    u(z + dz) = D_{dz/2} [ exp(i omega dS(x,z) dz) . D_{dz/2}[u(z)] ]

with homogeneous half-step transfer function (orthonormal FFT over x)

    H(kx, omega) = exp(i kz dz/2),  kz = sqrt((omega/c0)^2 - kx^2 + i eps)

The +i eps inside the square root makes evanescent components
(|kx| > omega/c0) decay exponentially instead of blowing up.

Three marching primitives are provided:

* ``forward``   : downward march (transmit field), returns the field at every
                  slab; same transfer functions evaluate an up-going field at
                  larger z, so this is the generic "field continuation".
* ``march_up``  : evaluate an up-going (scattered) field stack at the surface:
                  r_z = A_z r_{z+1} + q_z  (single upward sweep).
* ``adjoint``   : conjugated (time-reversed) back-march used for the adjoint
                  operator F^H / adjoint imaging: b_{z+1} = A_z^H b_z.

``A_z^H`` uses conj(H) and conj(screen) with reversed half-step order, which
is the exact Hermitian adjoint of ``A_z`` because the FFT is applied with
norm='ortho' (unitary).  The adjoint identity is verified in
tests/test_adjoint.py.
"""

import math

import torch
import torch.nn as nn

__all__ = ["HeterogeneousAngularSpectrum"]


class HeterogeneousAngularSpectrum(nn.Module):
    """Split-step angular-spectrum propagator over a stack of slabs.

    Parameters
    ----------
    nx : number of lateral grid points (FFT length, dim=-1).
    dx, dz : grid spacing [m].
    c0 : reference sound speed [m/s].
    eps : relative regularization inside the kz square root.
    dtype : complex compute dtype (complex64 default, complex128 in tests).
    """

    def __init__(self, nx, dx, dz, c0, eps=1e-6, dtype=torch.complex64):
        super().__init__()
        self.nx, self.dx, self.dz, self.c0, self.eps = nx, dx, dz, c0, eps
        self.cdtype = dtype
        kx = 2.0 * math.pi * torch.fft.fftfreq(nx, d=dx, dtype=torch.float64)
        self.register_buffer("kx", kx.to(rel_real(dtype)), persistent=False)

    # ------------------------------------------------------------------ parts
    def half_transfer(self, omega):
        """exp(i kz dz/2), shape [n_freq, nx]."""
        k = omega[:, None] / self.c0                       # [n_w, 1]
        arg = k ** 2 - self.kx[None, :] ** 2 + 1j * self.eps * k ** 2
        kz = torch.sqrt(arg.to(self.cdtype))
        return torch.exp(1j * kz * (self.dz / 2.0))

    def screen(self, ds_col, omega):
        """Phase screen exp(i omega dS dz) for one slab.

        ds_col: [..., nx] real slowness perturbation of the slab (leading
        dims are the batch dims of delta_s). Returns [..., 1, n_freq, nx],
        broadcastable against fields [..., n_theta, n_freq, nx].
        """
        L = ds_col.dim() - 1                              # batch dims
        om = omega.view(*([1] * L), -1, 1)                # [1.., n_w, 1]
        arg = self.dz * om * ds_col.unsqueeze(-2)         # [..., n_w, nx]
        scr = torch.exp(1j * arg.to(self.cdtype))
        if L > 0:                   # batched delta_s: add the theta slot
            scr = scr.unsqueeze(-3)
        return scr

    def _fft(self, u):
        return torch.fft.fft(u, dim=-1, norm="ortho")

    def _ifft(self, u):
        return torch.fft.ifft(u, dim=-1, norm="ortho")

    def _step(self, u, H, scr):
        u = self._ifft(self._fft(u) * H)
        u = u * scr
        u = self._ifft(self._fft(u) * H)
        return u

    def _step_adj(self, u, H, scr):
        u = self._ifft(self._fft(u) * H.conj())
        u = u * scr.conj()
        u = self._ifft(self._fft(u) * H.conj())
        return u

    # ----------------------------------------------------------------- marches
    def forward(self, u0, delta_s, omega_grid, dx=None, dz=None):
        """March a field stack downward slab by slab.

        u0: [..., n_theta, n_freq, nx] complex initial field at z = 0.
        delta_s: [..., nz, nx] real slowness perturbation (fine grid).
        omega_grid: [n_freq] real angular frequencies [rad/s].
        Returns [..., n_theta, n_freq, nz, nx]: field at every slab
        (slab 0 is u0 itself).
        """
        omega = omega_grid.to(rel_real(self.cdtype))
        H = self.half_transfer(omega)                       # [n_w, nx]
        fields = [u0]
        u = u0
        nz = delta_s.shape[-2]
        for z in range(nz - 1):
            scr = self.screen(delta_s[..., z, :], omega)
            u = self._step(u, H, scr)
            fields.append(u)
        return torch.stack(fields, dim=-2)

    def march_up(self, q_stack, delta_s, omega_grid, dx=None, dz=None):
        """Evaluate an up-going source distribution at the surface.

        q_stack: [..., n_theta, n_freq, nz, nx] complex slab sources.
        Implements r_z = A_z r_{z+1} + q_z with the *same* step operators as
        :meth:`forward` (field continuation is direction-agnostic), so the
        surface field equals sum_z A_0...A_{z-1} q_z.
        Returns [..., n_theta, n_freq, nx].
        """
        omega = omega_grid.to(rel_real(self.cdtype))
        H = self.half_transfer(omega)
        r = q_stack[..., -1, :]
        for z in range(q_stack.shape[-2] - 2, -1, -1):
            scr = self.screen(delta_s[..., z, :], omega)
            r = self._step(r, H, scr) + q_stack[..., z, :]
        return r

    def adjoint(self, v0, delta_s, omega_grid, dx=None, dz=None):
        """Conjugated back-march (Hermitian adjoint of the continuation).

        v0: [..., n_theta, n_freq, nx] complex field at the surface.
        Returns [..., n_theta, n_freq, nz, nx] with slab z holding
        A_{z-1}^H ... A_0^H v0.
        """
        omega = omega_grid.to(rel_real(self.cdtype))
        H = self.half_transfer(omega)
        b = v0
        out = []
        nz = delta_s.shape[-2]
        for z in range(nz):
            out.append(b)
            if z < nz - 1:
                scr = self.screen(delta_s[..., z, :], omega)
                b = self._step_adj(b, H, scr)
        return torch.stack(out, dim=-2)


def rel_real(cdtype):
    return torch.float64 if cdtype == torch.complex128 else torch.float32

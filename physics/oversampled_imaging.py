"""Born imaging on a finer lateral propagation grid than the parameter grid.

The learned / reconstructed parameters remain on the original image grid, but
Tx/Rx angular-spectrum propagation can be evaluated on an integer-oversampled
lateral grid.  For the L11 UltraWave setup this implements

    parameter grid   dx = 0.2 mm
    propagation grid dx = 0.1 mm

without changing the network output shape or the stored phase-screen controls.

A coarse parameter cell is represented as a piecewise-constant set of fine
propagation cells via ``repeat_interleave``.  The reverse map used by the
manual Born adjoint is the exact transpose of that replication (sum over the
fine sub-cells), so the forward/adjoint dot-product identity is preserved.
"""
from __future__ import annotations

import copy

import torch

from physics.imaging import BornModel

__all__ = ["LateralOversampledBornModel"]


class LateralOversampledBornModel(BornModel):
    """BornModel with coarse parameters and a finer lateral ASP carrier.

    Parameters ``nx`` and ``dx`` describe the *parameter* grid exposed to the
    rest of the network.  ``lateral_oversample`` only changes the internal ASP
    FFT grid.  Depth sampling is unchanged.

    The fine grid preserves the physical coarse-cell edge span.  If coarse
    centres start at ``x0`` with spacing ``dx``, the fine centres start at

        x0_prop = x0 - 0.5 * (dx - dx_prop)

    with ``dx_prop = dx / lateral_oversample`` and
    ``nx_prop = nx * lateral_oversample``.
    """

    def __init__(self, meta, nx, nz, dx, dz, c0, dtype=torch.complex64,
                 eps=1e-6, spreading="farfield2d", lateral_oversample=1):
        factor = int(lateral_oversample)
        if factor < 1 or factor != lateral_oversample:
            raise ValueError("lateral_oversample must be a positive integer")

        param_x0 = float(meta.get("x0", 0.0))
        prop_dx = float(dx) / factor
        prop_nx = int(nx) * factor
        prop_x0 = param_x0 - 0.5 * (float(dx) - prop_dx)

        prop_meta = copy.deepcopy(meta)
        prop_meta.x0 = prop_x0
        super().__init__(
            prop_meta, prop_nx, nz, prop_dx, dz, c0,
            dtype=dtype, eps=eps, spreading=spreading,
        )

        # Keep the public geometry in parameter-grid units.  The inherited
        # ASP, x-coordinate buffer, sampling indices and surface transfer were
        # already built on the propagation grid above.
        self.lateral_oversample = factor
        self.param_nx = int(nx)
        self.param_dx = float(dx)
        self.param_x0 = param_x0
        self.prop_nx = prop_nx
        self.prop_dx = prop_dx
        self.prop_x0 = prop_x0
        self.nx = self.param_nx
        self.dx = self.param_dx
        self.x0 = self.param_x0

    # ----------------------------------------------------------- grid maps
    def to_propagation_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Piecewise-constant lateral lift from parameter to ASP grid."""
        if x.shape[-1] == self.prop_nx:
            return x
        if x.shape[-1] != self.param_nx:
            raise ValueError(
                f"expected lateral size {self.param_nx} (parameter) or "
                f"{self.prop_nx} (propagation), got {x.shape[-1]}")
        if self.lateral_oversample == 1:
            return x
        return x.repeat_interleave(self.lateral_oversample, dim=-1)

    def from_propagation_adjoint(self, x: torch.Tensor) -> torch.Tensor:
        """Exact transpose of :meth:`to_propagation_grid`."""
        if x.shape[-1] == self.param_nx:
            return x
        if x.shape[-1] != self.prop_nx:
            raise ValueError(
                f"expected propagation lateral size {self.prop_nx}, "
                f"got {x.shape[-1]}")
        if self.lateral_oversample == 1:
            return x
        shape = x.shape[:-1] + (self.param_nx, self.lateral_oversample)
        return x.reshape(shape).sum(dim=-1)

    # ------------------------------------------------------- surface maps
    def scatter(self, d):
        """Exact transpose of receiver interpolation on the propagation grid."""
        shape = d.shape[:-1]
        out = torch.zeros(*shape, self.prop_nx, dtype=d.dtype, device=d.device)
        flat = out.view(-1, self.prop_nx)
        n_b = flat.shape[0]
        i0 = self.smp_i0.expand(n_b, -1)
        i1 = i0 + 1
        w0 = (1.0 - self.smp_fr).expand(n_b, -1) * d.reshape(n_b, -1)
        w1 = self.smp_fr.expand(n_b, -1) * d.reshape(n_b, -1)
        flat.scatter_add_(1, i0, w0)
        flat.scatter_add_(1, i1, w1)
        return out

    # ------------------------------------------------------------- Born API
    def transmit_fields(self, delta_s, angles_idx=None):
        return super().transmit_fields(
            self.to_propagation_grid(delta_s), angles_idx)

    def forward(self, m, delta_s, u_tx, angles_idx=None):
        return super().forward(
            self.to_propagation_grid(m),
            self.to_propagation_grid(delta_s),
            u_tx,
            angles_idx,
        )

    def adjoint_per_angle(self, D, u_tx, delta_s):
        """Per-angle adjoint images on the parameter grid.

        ``D`` is ``[..., n_theta, n_freq, n_e]`` and the result is
        ``[..., n_theta, nz, param_nx]``.  Only frequency is summed here.
        """
        ds_prop = self.to_propagation_grid(delta_s)
        b0 = self.scatter(D)
        b0 = self.asp._ifft(
            self.asp._fft(b0) * self.surface_transfer.conj())
        b = self.asp.adjoint(b0, ds_prop, self.omega_)
        img_prop = (b * (u_tx * self.w_z).conj()).sum(dim=-3)
        return self.from_propagation_adjoint(img_prop)

    def adjoint(self, D, u_tx, delta_s, angles_idx=None):
        img_prop = super().adjoint(
            D, u_tx, self.to_propagation_grid(delta_s), angles_idx)
        return self.from_propagation_adjoint(img_prop)

    def illumination(self, u_tx):
        illum_prop = super().illumination(u_tx)
        return self.from_propagation_adjoint(illum_prop)

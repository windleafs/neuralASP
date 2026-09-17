"""Coordinate-aware SoS adapter. The existing imaging backend is untouched."""
import torch
import torch.nn.functional as F


def resample_sos(c_xz, cx, cz, meta, grid):
    """[x,z] m/s -> [z,x] m/s on the existing pixel centres."""
    c = torch.as_tensor(c_xz)
    cx = torch.as_tensor(cx, device=c.device, dtype=c.dtype)
    cz = torch.as_tensor(cz, device=c.device, dtype=c.dtype)
    if c.ndim != 2 or c.shape != (len(cx), len(cz)):
        raise ValueError('SoS map must have shape [len(cx), len(cz)]')
    if not torch.isfinite(c).all() or (c <= 0).any():
        raise ValueError('SoS must be finite and positive, in m/s')
    for axis in (cx, cz):
        if len(axis) < 2 or not (axis.diff() > 0).all():
            raise ValueError('SoS coordinates must be increasing')
        if not torch.allclose(axis.diff(), axis.diff().mean(), rtol=1e-4, atol=1e-9):
            raise ValueError('SoS source coordinates must be uniform')
    x = meta.x0 + torch.arange(grid.nx, device=c.device, dtype=c.dtype)*grid.dx
    z = meta.z0 + torch.arange(grid.nz, device=c.device, dtype=c.dtype)*grid.dz
    if x.min()<cx.min()-1e-7 or x.max()>cx.max()+1e-7 or z.min()<cz.min()-1e-7 or z.max()>cz.max()+1e-7:
        raise ValueError('SoS source grid does not cover imaging grid')
    zz, xx = torch.meshgrid(z, x, indexing='ij')
    query = torch.stack([2*(xx-cx[0])/(cx[-1]-cx[0])-1,
                         2*(zz-cz[0])/(cz[-1]-cz[0])-1], -1)
    return F.grid_sample(c.T[None,None], query[None], mode='bilinear',
                         padding_mode='border', align_corners=True)[0,0]


def sos_to_delta_s(c_xz, cx, cz, meta, grid, c0):
    return resample_sos(c_xz, cx, cz, meta, grid).reciprocal() - 1/float(c0)

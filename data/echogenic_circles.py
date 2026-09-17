"""Deterministic circular high-echogenicity inclusions for OA-Breast media.

Only microscopic density fluctuations are strengthened. Mean density,
sound speed and attenuation stay as supplied by the original anatomy map.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def add_echogenic_circles(maps, codes, x, z, seed, *, count=None,
                          radius_mm=(1.5, 4.5),
                          scatter_rms_kg_m3=(30.0, 45.0),
                          edge_mm=0.25):
    """Return copied maps, a union mask and reproducible circle metadata.

    Centres are chosen inside fat/gland tissue in the physical L11 field of
    view. A soft boundary avoids adding an artificial specular ring.
    """
    if maps['density'].shape != codes.shape or codes.shape != (len(z), len(x)):
        raise ValueError('map, code and coordinate shapes differ')
    if radius_mm[0] <= 0 or radius_mm[1] < radius_mm[0]:
        raise ValueError('invalid radius interval')
    rng = np.random.default_rng(int(seed))
    target = int(rng.integers(1, 4)) if count is None else int(count)
    if not 1 <= target <= 3:
        raise ValueError('circle count must be 1..3')
    dx_mm = float(np.median(np.diff(x))) * 1e3
    dz_mm = float(np.median(np.diff(z))) * 1e3
    if not np.isclose(dx_mm, dz_mm) or dx_mm <= 0:
        raise ValueError('expected isotropic positive grid spacing')

    result = {key: value.copy() for key, value in maps.items()}
    union = np.zeros(codes.shape, dtype=np.bool_)
    circles = []
    for _ in range(target):
        selected = None
        for _attempt in range(300):
            radius = float(rng.uniform(*radius_mm))
            cx = float(rng.uniform(-13.0, 13.0))
            cz = float(rng.uniform(10.0, 35.0))
            if any(np.hypot(cx-c['x_mm'], cz-c['z_mm']) <
                   radius+c['radius_mm']+1.0 for c in circles):
                continue
            ix = int(round((cx-x[0]*1e3)/dx_mm))
            iz = int(round((cz-z[0]*1e3)/dz_mm))
            reach = int(np.ceil((radius+2*edge_mm)/dx_mm))
            x0, x1 = max(0, ix-reach), min(len(x), ix+reach+1)
            z0, z1 = max(0, iz-reach), min(len(z), iz+reach+1)
            xx = x[x0:x1][None, :]*1e3
            zz = z[z0:z1][:, None]*1e3
            rr = np.hypot(xx-cx, zz-cz)
            inside = rr <= radius
            tissue = np.isin(codes[z0:z1, x0:x1], (2, 3))
            if inside.sum() < 50 or np.mean(tissue[inside]) < 0.90:
                continue
            selected = (cx, cz, radius, x0, x1, z0, z1, rr, inside, tissue)
            break
        if selected is None:
            raise RuntimeError('could not place all circles inside fat/gland')
        cx, cz, radius, x0, x1, z0, z1, rr, inside, tissue = selected
        extra_rms = float(rng.uniform(*scatter_rms_kg_m3))
        noise = gaussian_filter(rng.standard_normal(rr.shape), sigma=0.55,
                                mode='reflect')
        noise -= noise[inside & tissue].mean()
        noise /= max(float(noise[inside & tissue].std()), 1e-12)
        noise = np.clip(noise, -2.5, 2.5)
        taper = 0.5*(1.0-np.tanh((rr-radius)/edge_mm))
        taper *= tissue
        noise -= float(np.sum(noise*taper)/np.sum(taper))
        patch = result['density'][z0:z1, x0:x1]
        patch += (extra_rms*noise*taper).astype(np.float32)
        union[z0:z1, x0:x1] |= inside & tissue
        circles.append({'x_mm': cx, 'z_mm': cz, 'radius_mm': radius,
                        'extra_density_scatter_rms_kg_m3': extra_rms,
                        'tissue_fraction': float(np.mean(tissue[inside])),
                        'area_fine_pixels': int(np.count_nonzero(inside & tissue))})
    if not np.isfinite(result['density']).all():
        raise RuntimeError('non-finite augmented density')
    if result['density'].min() <= 800 or result['density'].max() >= 1300:
        raise RuntimeError(f"augmented density outside validated range: "
                           f"original=[{maps['density'].min():.1f},{maps['density'].max():.1f}] "
                           f"augmented=[{result['density'].min():.1f},{result['density'].max():.1f}]")
    return result, union, circles

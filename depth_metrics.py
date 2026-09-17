"""Physical depth ROI metrics, with no target-fitted gain or normalization.

Image energy fractions describe the reconstructed image, not RF information.
Correlation with a scattering proxy does not validate that proxy physically.
"""

import math

import torch

from common import corr2d


@torch.no_grad()
def depth_metrics(outputs, m_ref, c_true, z_mm, depth_range=(5., 40.)):
    """Return sample-mean metrics on [lo, hi) mm; omit an empty ROI.

    ``outputs`` contains ``ours`` and ``uniform`` pipeline outputs. The
    separate I_input and m_hat metrics must not be interpreted interchangeably.
    """
    lo, hi = map(float, depth_range)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi):
        raise ValueError("depth_roi_mm must contain finite 0 <= lo < hi")
    z_mm = torch.as_tensor(z_mm, device=m_ref.device)
    if z_mm.ndim != 1 or len(z_mm) != m_ref.shape[-2]:
        raise ValueError("z_mm must match the image depth dimension")
    roi = (z_mm >= lo) & (z_mm < hi)
    if not roi.any():
        return {}
    shallow = z_mm < lo
    result = {"roi_depth_min_mm": lo, "roi_depth_max_mm": hi}
    ref = m_ref.abs()
    for tag in ("ours", "uniform"):
        out = outputs[tag]
        for key, metric in (("I_input", "img_corr"), ("m_hat", "m_abs_corr")):
            im = out[key].abs()
            corr = [corr2d(im[b, roi], ref[b, roi]) for b in range(len(im))]
            result[f"{metric}_{tag}_deep"] = torch.stack(corr).mean().item()
            energy = im.square()
            frac = (energy[:, shallow].sum((-2, -1)) /
                    energy.sum((-2, -1)).clamp_min(1e-30))
            result[f"{key}_{tag}_shallow_energy_fraction"] = frac.mean().item()
        err = (out["c_hat"][:, roi] - c_true[:, roi]).square()
        result[f"c_rmse_{tag}_deep_m"] = err.mean((-2, -1)).sqrt().mean().item()
    return result

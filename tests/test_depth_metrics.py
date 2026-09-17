import pytest
import torch

from common import corr2d
from depth_metrics import depth_metrics


def test_bright_surface_cannot_hide_wrong_deep_structure():
    ref = torch.tensor([[[100., 200.], [300., 400.], [1., 2.], [3., 4.]]])
    pred = ref.clone()
    pred[:, 2:] = 5 - ref[:, 2:]
    c = torch.full_like(ref, 1500.)
    out = {tag: {"I_input": pred.to(torch.complex64), "m_hat": pred, "c_hat": c}
           for tag in ("ours", "uniform")}
    assert corr2d(pred[0], ref[0]) > .99
    metrics = depth_metrics(out, ref, c, [0., 4., 5., 39.])
    for tag in out:
        assert metrics[f"img_corr_{tag}_deep"] == pytest.approx(-1.)
        assert metrics[f"m_abs_corr_{tag}_deep"] == pytest.approx(-1.)
        assert metrics[f"I_input_{tag}_shallow_energy_fraction"] > .999
        assert metrics[f"c_rmse_{tag}_deep_m"] == 0


def test_roi_bounds_and_sample_mean_rmse():
    ref = torch.arange(16.).reshape(2, 4, 2)
    c = torch.full_like(ref, 1500.)
    pred_c = c.clone()
    pred_c[:, 0] += 1000  # outside ROI below 5 mm
    pred_c[:, 3] += 1000  # upper boundary 40 mm is excluded
    pred_c[1, 1:3] += 20
    out = {tag: {"I_input": ref, "m_hat": ref, "c_hat": pred_c}
           for tag in ("ours", "uniform")}
    metrics = depth_metrics(out, ref, c, [4., 5., 39., 40.])
    assert metrics["c_rmse_ours_deep_m"] == pytest.approx(10.)
    assert depth_metrics(out, ref, c, [0., 1., 2., 3.]) == {}
    with pytest.raises(ValueError, match="depth_roi_mm"):
        depth_metrics(out, ref, c, [4., 5., 39., 40.], [40., 5.])

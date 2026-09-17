import torch

from common import build_meta, load_config, rf_to_D
from data.l11_kwave import validate_l11_sample


def test_l11_config_contract():
    cfg = load_config("configs/l11_kwave.yaml")
    meta = build_meta(cfg)
    assert (cfg.grid.nz, cfg.grid.nx) == (216, 192)
    assert (cfg.acq.n_angles, cfg.array.n_elements, cfg.acq.n_t) == (3, 192, 2401)
    assert len(meta.freqs) == 53
    assert meta.train_idx.tolist() == [0, 1]
    assert meta.hold_idx.tolist() == [2]


def test_l11_sample_validation_and_transform():
    cfg = load_config("configs/l11_kwave.yaml")
    meta = build_meta(cfg)
    gen = torch.Generator().manual_seed(9)
    rf = torch.randn(3, 192, 2401, generator=gen) * 5e-7
    c = torch.full((216, 192), 1540.0)
    m = torch.full((216, 192), 0.2, dtype=torch.complex64)
    sample = {
        "rf": rf.float(), "D": rf_to_D(rf.float(), meta).to(torch.complex64),
        "delta_s": 1.0 / c - 1.0 / 1540.0, "m": m, "c": c,
    }
    validate_l11_sample(sample, cfg, meta, check_transform=True)


def test_l11_rejects_bad_D():
    cfg = load_config("configs/l11_kwave.yaml")
    meta = build_meta(cfg)
    rf = torch.ones(3, 192, 2401, dtype=torch.float32) * 5e-7
    bad_D = rf_to_D(rf, meta).to(torch.complex64)
    bad_D[0, 0, 0] += 1e-3
    sample = {
        "rf": rf, "D": bad_D,
        "delta_s": torch.zeros(216, 192),
        "m": torch.full((216, 192), 0.2, dtype=torch.complex64),
        "c": torch.full((216, 192), 1540.0),
    }
    try:
        validate_l11_sample(sample, cfg, meta, check_transform=True)
    except ValueError as exc:
        assert "rf_to_D" in str(exc)
    else:
        raise AssertionError("bad D was accepted")


def test_eval_calibration_is_per_frequency():
    from eval import calibrate_gain
    pred = torch.ones(2, 1, 53, 192, dtype=torch.complex64)
    gain = torch.linspace(0.5, 1.5, 53).to(torch.complex64)[None, None, :, None]
    target = pred * gain
    calibrated = calibrate_gain(pred, target)
    assert calibrated.shape == pred.shape
    assert torch.allclose(calibrated, target, rtol=1e-5, atol=1e-6)

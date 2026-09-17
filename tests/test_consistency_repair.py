import copy
import numpy as np
import pytest
import torch
from common import C, _to_C, to_plain, build_meta, demod_iq, iq_delay_sum, rf_to_D
from physics.imaging import BornModel, straight_ray_delays, delay_integrate
from models.pipeline import ImagingPipeline, load_pipeline_checkpoint


def config():
    return _to_C({"physics": {"c0": 1540., "f0": 1e6, "bandwidth": .8, "fs": 8e6,
                  "eps_evanescent": 1e-9, "spreading": "none", "response_mode": "calibrated"},
                 "grid": {"nx": 32, "nz": 20, "dx": .0003, "dz": .0003, "x0": -.0048, "z0": .000075},
                 "array": {"n_elements": 12, "pitch": .0003, "center_x": 0.},
                 "coarse": {"factor": 4},
                 "acq": {"n_angles": 3, "n_t": 512, "n_freq": 0, "angle_span_deg": 8.,
                         "holdout_stride": 3, "t_ref_s": [2e-6, 1e-6, 2e-6]},
                 "model": {"enc_channels": [8, 8, 8], "fno_width": 8, "fno_modes": [2, 2],
                           "fno_layers": 1, "ds_max": 5e-5, "n_unroll": 1, "prox_hidden": 4,
                           "m_rms_ref": .2, "illum_compensate": False}})


def test_plane_wave_time_matches_known_geometry():
    c = 1540.; X = np.array([-.002, .003]); Z = np.array([.012])
    ang = np.array([-8., 0., 8.]); tref = np.array([2e-6, 1e-6, 2e-6])
    actual = straight_ray_delays(ang, [0.], X, Z, c, tref)
    expected = (X[None, :] * np.sin(np.deg2rad(ang))[:, None] + Z[0] * np.cos(np.deg2rad(ang))[:, None] + np.hypot(X, Z[0])[None, :]) / c + tref[:, None]
    np.testing.assert_allclose(actual[:, 0, 0], expected, rtol=1e-14)


def test_system_response_adjoint_and_autograd():
    torch.manual_seed(5); cfg = config(); meta = build_meta(cfg)
    meta.system_response = np.linspace(.2, 1., len(meta.freqs)) * np.exp(1j * np.linspace(0., 1., len(meta.freqs)))
    born = BornModel(meta, 32, 20, .0003, .0003, 1540., dtype=torch.complex128, eps=1e-9, spreading="none")
    ds = torch.randn(1, 20, 32, dtype=torch.float64) * 1e-5
    m = torch.randn(1, 20, 32, dtype=torch.complex128, requires_grad=True)
    u = born.transmit_fields(ds)
    D = torch.randn(1, 3, len(meta.freqs), 12, dtype=torch.complex128)
    pred = born(m, ds, u); adj = born.adjoint(D, u, ds)
    torch.testing.assert_close((pred * D.conj()).sum(), (m * adj.conj()).sum(), rtol=1e-10, atol=1e-12)
    grad, = torch.autograd.grad((pred-D).abs().square().sum(), m)
    torch.testing.assert_close(grad, 2 * born.adjoint(pred-D, u, ds), rtol=1e-10, atol=1e-12)


def test_iq_carrier_phase_restores_receiver_coherence():
    # All channels contain the same analytic pulse with different known delays.
    cfg = config(); meta = build_meta(cfg); t = torch.arange(512) / meta.fs
    delays = torch.tensor([10e-6, 10.375e-6, 10.625e-6])
    rf = (torch.exp(-((t[None] - delays[:, None]) / 2e-6)**2) * torch.cos(2*torch.pi*meta.f0*(t[None]-delays[:, None])))[None, None]
    iq = demod_iq(rf, meta)
    idx = (delays * meta.fs_iq)[None, :, None, None]
    coherent = iq_delay_sum(iq, idx, torch.ones(3), meta).abs().item()
    plain = delay_integrate(iq[:, :, None], idx, torch.ones(3)).abs().item()
    assert coherent > 1.3
    assert coherent > 3 * plain


def test_known_rf_point_localizes_with_per_angle_pulse_times():
    cfg = config(); meta = build_meta(cfg)
    X = cfg.grid.x0 + np.arange(32)*cfg.grid.dx
    Z = cfg.grid.z0 + np.arange(20)*cfg.grid.dz
    iz, ix = 13, 16
    times = straight_ray_delays(meta.angles_deg, meta.xe_coords, [X[ix]], [Z[iz]], 1540., meta.t_ref_s)[:, :, 0, 0]
    t = np.arange(512)/meta.fs
    tau = torch.tensor(t[None,None,:]-times[:,:,None], dtype=torch.float32)
    rf = (torch.exp(-(tau/.4e-6)**2)*torch.cos(2*torch.pi*meta.f0*tau))[None]
    idx = torch.tensor(straight_ray_delays(meta.angles_deg, meta.xe_coords, X, Z, 1540., meta.t_ref_s)*meta.fs_iq, dtype=torch.float32)
    images = iq_delay_sum(demod_iq(rf, meta), idx, torch.ones(12), meta)
    for image in images[0]:
        pz, px = divmod(image.abs().argmax().item(), 32)
        assert abs(pz-iz) <= 1 and abs(px-ix) <= 1, (pz,px)


def test_outside_record_queries_return_zero():
    x = torch.ones(1, 1, 1, 2, 16)
    idx = torch.tensor([[[[-2.]], [[20.]]]])
    assert delay_integrate(x, idx, torch.ones(2)).item() == 0


def test_pipeline_weighted_adjoint_is_loss_gradient():
    cfg = config(); cfg.physics.response_mode = "legacy_window"
    meta = build_meta(cfg); pipe = ImagingPipeline(cfg, meta, dtype=torch.complex128)
    ds = torch.zeros(1, 20, 32, dtype=torch.float64); tr = torch.tensor(meta.train_idx)
    u = pipe.born.transmit_fields(ds, tr)
    m = torch.randn(1, 20, 32, dtype=torch.complex128, requires_grad=True)
    D = torch.randn(1, 2, len(meta.freqs), 12, dtype=torch.complex128)
    pred = pipe._forward_m(m, ds, u)
    grad, = torch.autograd.grad((pred-D).abs().square().sum(), m)
    torch.testing.assert_close(grad, 2*pipe._fadj(pred-D, u, ds), rtol=1e-10, atol=1e-12)


def test_legacy_and_mismatched_checkpoint_rejected():
    cfg = config(); pipe = ImagingPipeline(cfg, build_meta(cfg))
    with pytest.raises(ValueError, match="incompatible"):
        load_pipeline_checkpoint(pipe, {"pipeline": pipe.state_dict()})
    ck = {"operator_version": pipe.operator_version, "config": to_plain(cfg), "pipeline": pipe.state_dict()}
    load_pipeline_checkpoint(pipe, ck)
    changed = copy.deepcopy(cfg); changed.acq.t_ref_s[0] += 1e-6
    with pytest.raises(ValueError, match="t_ref_s"):
        load_pipeline_checkpoint(ImagingPipeline(changed, build_meta(changed)), ck)
    changed = copy.deepcopy(cfg); changed.grid.nz = 24
    with pytest.raises(ValueError, match="grid.nz"):
        load_pipeline_checkpoint(ImagingPipeline(changed, build_meta(changed)), ck)


def test_continuous_band_loader_recomputes_archived_D(tmp_path):
    import json
    from data.l11_kwave import L11KWaveDataset
    cfg = config(); cfg.train = _to_C({"n_train": 1, "n_val": 0, "n_test": 0})
    cfg.data = _to_C({"root": str(tmp_path), "recompute_D": True})
    rf = torch.randn(3, 12, 512) * 5e-7; c = torch.full((20, 32), 1540.)
    s = {"rf": rf.float(), "D": torch.zeros(3, 2, 12, dtype=torch.complex64),
         "c": c, "delta_s": torch.zeros_like(c), "m": torch.full((20,32), .2, dtype=torch.complex64)}
    torch.save(s, tmp_path/'old.pt')
    (tmp_path/'index.json').write_text(json.dumps({"samples": [{"id": "train_000", "split": "train", "status": "complete", "path": "old.pt"}]}))
    dataset = L11KWaveDataset(cfg, "train", "cpu")
    torch.testing.assert_close(dataset.D[0], rf_to_D(rf, build_meta(cfg)))
    assert torch.load(tmp_path/'old.pt', weights_only=False)['D'].shape[1] == 2


def test_normalized_encoder_preserves_iq_amplitude_scale_invariance():
    cfg = config(); cfg.model.normalize_iq = True
    meta = build_meta(cfg); operator = ImagingPipeline(cfg, meta).neural_operator.eval()
    iq = torch.randn(1, 2, 12, 256, dtype=torch.complex64)
    tr = torch.tensor(meta.train_idx)
    with torch.no_grad():
        a = operator(iq * 5e-7, tr)['delta_s']
        b = operator(iq * 5e-4, tr)['delta_s']
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-10)

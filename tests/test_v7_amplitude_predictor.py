import torch

from models.v7_amplitude_predictor import V7AmplitudePredictor


def make_model():
    model = V7AmplitudePredictor(
        active_rank=6, depth_bins=4, lateral_modes=3, hidden=8,
        coeff_limit_np=0.5, descriptor_eps=1e-6)
    model.set_descriptor_stats(
        1.0, torch.zeros(model.descriptor_dim),
        torch.ones(model.descriptor_dim))
    return model


def test_descriptor_preserves_common_amplitude_change():
    model = make_model()
    env = torch.ones(1, 8, 16)
    a = model.raw_descriptor(env).reshape(1, 4, 3)
    b = model.raw_descriptor(0.5 * env).reshape(1, 4, 3)
    # k=0 carries the global/common log-amplitude shift.
    assert torch.all((a[..., 0] - b[..., 0]).abs() > 0.5)
    # Pure common gain should not create higher lateral DCT modes.
    torch.testing.assert_close(
        a[..., 1:], torch.zeros_like(a[..., 1:]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        b[..., 1:], torch.zeros_like(b[..., 1:]), atol=1e-6, rtol=1e-6)


def test_descriptor_detects_lateral_structure():
    model = make_model()
    x = torch.linspace(0.5, 1.5, 16)
    env = x[None, None, :].expand(1, 8, 16).clone()
    q = model.raw_descriptor(env).reshape(1, 4, 3)
    assert float(q[..., 1:].abs().max()) > 1e-3


def test_zero_initialized_predictor_starts_at_phase_only_baseline():
    model = make_model()
    env = torch.rand(2, 8, 16) + 0.1
    out = model(env)
    torch.testing.assert_close(
        out["coeff_np"], torch.zeros_like(out["coeff_np"]))


def test_coefficients_are_bounded():
    model = make_model()
    with torch.no_grad():
        model.mlp[-1].bias.fill_(20.0)
    env = torch.ones(1, 8, 16)
    coeff = model(env)["coeff_np"]
    assert float(coeff.max()) <= 0.5 + 1e-6

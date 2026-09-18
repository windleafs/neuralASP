import torch

from physics.amplitude_consistency import (
    amplitude_pattern_consistency,
    centered_log_envelope,
)


def test_amplitude_consistency_is_invariant_to_per_angle_gain():
    torch.manual_seed(4)
    base = torch.randn(1, 3, 24, 20, dtype=torch.complex64)
    target = torch.randn(1, 2, 24, 20, dtype=torch.complex64)
    mask = torch.ones(1, 24, 20)

    ref = amplitude_pattern_consistency(
        base, target, mask, smooth_kernel=5, eps=1e-4)

    context_gain = torch.tensor([0.5, 2.0, 3.0]).view(1, 3, 1, 1)
    target_gain = torch.tensor([4.0, 0.25]).view(1, 2, 1, 1)
    got = amplitude_pattern_consistency(
        base * context_gain, target * target_gain,
        mask, smooth_kernel=5, eps=1e-4)

    assert torch.allclose(ref, got, atol=2e-5, rtol=2e-5)


def test_identical_amplitude_patterns_have_near_zero_loss():
    torch.manual_seed(5)
    phase1 = torch.exp(1j * torch.randn(1, 2, 16, 18))
    phase2 = torch.exp(1j * torch.randn(1, 3, 16, 18))
    envelope = torch.rand(1, 1, 16, 18) + 0.2
    ctx = envelope * phase1
    tgt = envelope * phase2
    mask = torch.ones(1, 16, 18)

    loss = amplitude_pattern_consistency(
        ctx, tgt, mask, smooth_kernel=3, eps=1e-4)
    assert float(loss) < 1e-8


def test_centered_log_envelope_rejects_even_kernel():
    images = torch.ones(1, 2, 8, 8, dtype=torch.complex64)
    mask = torch.ones(1, 8, 8)
    try:
        centered_log_envelope(images, mask, smooth_kernel=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected an even smoothing kernel to fail")

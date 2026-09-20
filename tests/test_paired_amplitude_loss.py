import torch

from physics.paired_amplitude_loss import paired_image_loss


def test_paired_loss_does_not_collapse_for_tiny_absolute_units():
    ref = torch.full((1, 8, 8), 2e-9)
    cur = torch.full((1, 8, 8), 1e-9)
    total, image, depth, scale = paired_image_loss(
        cur, ref, smooth_kernel=1, depth_bins=4, eps=1e-5, depth_weight=0.5)
    assert float(total) > 0.1
    assert float(image) > 0.1
    assert float(depth) > 0.1
    assert float(scale) > 0


def test_paired_loss_invariant_to_shared_numerical_scale():
    z = torch.linspace(1.0, 2.0, 8)[:, None]
    x = torch.linspace(0.8, 1.2, 8)[None, :]
    ref = (z * x)[None]
    cur = 0.7 * ref
    a = paired_image_loss(
        cur, ref, smooth_kernel=3, depth_bins=4, eps=1e-5, depth_weight=0.5)
    b = paired_image_loss(
        cur * 1e-10, ref * 1e-10, smooth_kernel=3, depth_bins=4,
        eps=1e-5, depth_weight=0.5)
    torch.testing.assert_close(a[0], b[0], rtol=1e-5, atol=1e-6)


def test_paired_loss_preserves_common_mode_gain_difference():
    ref = torch.ones((1, 8, 8))
    same = paired_image_loss(
        ref, ref, smooth_kernel=1, depth_bins=4, eps=1e-5, depth_weight=0.5)
    half = paired_image_loss(
        0.5 * ref, ref, smooth_kernel=1, depth_bins=4,
        eps=1e-5, depth_weight=0.5)
    assert float(same[0]) == 0.0
    assert float(half[0]) > 0.5

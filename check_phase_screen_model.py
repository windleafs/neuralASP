"""Focused numerical check of the V2 discrete phase-screen conversion."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.phase_screen import PhaseScreenModel
from physics.phase_screen import controls_to_discrete_ds
from scripts.pilot_phase_asp import (DATA_ROOT, corrected_config, embed,
                                     projected_truth_screen)


def main():
    sample = torch.load(DATA_ROOT / "shards" / "train_000.pt",
                        map_location="cpu", weights_only=False)
    cfg, meta = corrected_config("configs/l11_ultrawave_500_11angle.yaml",
                                 sample, 64)
    model = PhaseScreenModel(cfg, meta)
    raw = (torch.randn(2, model.layers, model.controls) * 0.1).requires_grad_()
    batched = model.screen_to_slowness(raw)
    reference = controls_to_discrete_ds(raw, model.born.nz, model.born.nx,
                                        model.born.dz, model.limit_us,
                                        model.pad)
    torch.testing.assert_close(batched, reference, atol=1e-7, rtol=1e-6)
    assert torch.count_nonzero(batched[:, -1]) == 0
    # Exactly one propagation row per phase layer is nonzero for generic input.
    active_rows = (batched.abs().amax(dim=-1) > 0).sum(dim=-1)
    assert torch.all(active_rows <= model.layers)
    batched.square().mean().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    zero = model.screen_to_slowness(torch.zeros_like(raw.detach()))
    assert torch.count_nonzero(zero) == 0

    gt_ds = embed(sample["delta_s"], model.pad)
    projected, _, _ = projected_truth_screen(
        gt_ds, model.layers, model.controls, model.born.dz, 20.0,
        model.pad, model.born.z0, model.bulk_limit_us, False)
    controls = 20.0 * torch.tanh(projected)
    expanded = F.interpolate(controls[None], size=model.born.nx,
                             mode="linear", align_corners=True)[0]
    gauge_error = expanded[:, model.pad:-model.pad].mean(-1).abs().max()
    assert float(gauge_error) < 1e-5, float(gauge_error)
    print("V2 discrete phase conversion, gauge and gradients pass")


if __name__ == "__main__":
    main()

"""Focused numerical check of the batched phase-screen-to-ASP conversion."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.phase_screen import PhaseScreenModel
from scripts.pilot_phase_asp import (DATA_ROOT, corrected_config, effective_ds,
                                     embed, projected_truth_screen)
from scripts.experiment_gt_phase_screen_k import ideal_layer_screen


def main():
    sample = torch.load(DATA_ROOT / "shards" / "train_000.pt",
                        map_location="cpu", weights_only=False)
    cfg, meta = corrected_config("configs/l11_ultrawave_500_11angle.yaml",
                                 sample, 64)
    model = PhaseScreenModel(cfg, meta)
    raw = (torch.randn(2, 12, 48) * 0.1).requires_grad_()
    bulk = (torch.randn(2, 2) * 0.1).requires_grad_()
    batched = model.screen_to_slowness(raw, bulk)
    reference = torch.stack([
        effective_ds(raw[i], model.born.nz, model.born.nx, model.born.dz,
                     model.limit_us, model.pad, bulk[i], model.born.z0,
                     model.bulk_limit_us)
        for i in range(len(raw))
    ])
    torch.testing.assert_close(batched, reference, atol=1e-7, rtol=1e-6)
    assert torch.count_nonzero(batched[:, -1]) == 0
    batched.square().mean().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert bulk.grad is not None and torch.isfinite(bulk.grad).all()
    zero = model.screen_to_slowness(torch.zeros_like(raw.detach()),
                                    torch.zeros_like(bulk.detach()))
    assert torch.count_nonzero(zero) == 0
    gt_ds = embed(sample['delta_s'], model.pad)
    projected, _, _ = projected_truth_screen(
        gt_ds, 4, model.controls, model.born.dz, 20.0,
        model.pad, model.born.z0, model.bulk_limit_us, False)
    controls = 20.0 * torch.tanh(projected)
    expanded = F.interpolate(controls[None], size=model.born.nx,
                             mode='linear', align_corners=True)[0]
    gauge_error = expanded[:, model.pad:-model.pad].mean(-1).abs().max()
    assert float(gauge_error) < 1e-5, float(gauge_error)
    exact = ideal_layer_screen(gt_ds, model.born.nz - 1)
    torch.testing.assert_close(exact[:-1], gt_ds[:-1], atol=0, rtol=0)
    for k in (1, 2, 4, 8):
        approx = ideal_layer_screen(gt_ds, k)
        assert torch.allclose(approx[:-1].sum(0), gt_ds[:-1].sum(0),
                              rtol=1e-5, atol=1e-7)
    print("phase conversion, gauge, layer integrals and gradients pass")


if __name__ == "__main__":
    main()

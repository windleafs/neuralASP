"""Evaluate saved phase-only screens with all 11 angles; no optimization or CG."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import corr2d, rf_to_D
from physics.imaging import BornModel
from pilot_phase_asp import (DATA_ROOT, corrected_config, crop, effective_ds,
                             embed, padded_meta, projected_truth_screen)
from pilot_phase_only import angle_images


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True)
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--n-controls", type=int, default=48)
    p.add_argument("--limit-us", type=float, default=0.2)
    p.add_argument("--bulk-limit-us", type=float, default=2.0)
    args = p.parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    sample = torch.load(DATA_ROOT / "shards" / f"{args.sample}.pt",
                        map_location="cpu", weights_only=False)
    cfg, meta = corrected_config("configs/l11_ultrawave_500_11angle.yaml", sample,
                                 args.n_freq)
    pm = padded_meta(meta, args.pad, cfg.grid.dx)
    born = BornModel(pm, cfg.grid.nx + 2 * args.pad, cfg.grid.nz,
                     cfg.grid.dx, cfg.grid.dz, cfg.physics.c0,
                     eps=cfg.physics.eps_evanescent,
                     spreading=cfg.physics.spreading).to(device)
    D = rf_to_D(sample["rf"].to(device), meta)
    truth_abs = sample["m"].to(device).abs()
    idx = torch.arange(len(meta.angles_deg), device=device)
    saved = torch.load(args.controls, map_location=device, weights_only=False)
    raw = saved["raw_phase_controls"]
    bulk = saved["raw_bulk"]
    if tuple(raw.shape) != (args.layers, args.n_controls):
        raise ValueError("saved phase controls do not match requested shape")
    zero = torch.zeros(born.nz, born.nx, device=device)
    fitted = effective_ds(raw, born.nz, born.nx, born.dz, args.limit_us,
                          args.pad, bulk, born.z0, args.bulk_limit_us)
    known = embed(sample["delta_s"].to(device), args.pad)
    proj_raw, proj_bulk, _ = projected_truth_screen(
        known, args.layers, args.n_controls, born.dz, args.limit_us,
        args.pad, born.z0, args.bulk_limit_us, bulk is not None)
    projected = effective_ds(proj_raw, born.nz, born.nx, born.dz,
                             args.limit_us, args.pad, proj_bulk,
                             born.z0, args.bulk_limit_us)
    result = {"sample_id": args.sample, "n_freq": len(meta.freqs)}
    for name, ds in (("uniform", zero), ("fitted", fitted),
                     ("projected_c_screen", projected), ("known_c", known)):
        images = angle_images(born, ds, D, idx)
        result[name] = {
            "image_11_abs_corr": float(corr2d(crop(images.mean(0), args.pad).abs(),
                                                truth_abs).item())
        }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

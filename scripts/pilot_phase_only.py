"""Optimize only an effective phase screen using cross-angle adjoint-image coherence.

No scatterer map is estimated. Eight angles define the objective and a fixed
spatial mask. Three held-out angles are used only for evaluation.
Run from the neural_asp project root after placing this file in scripts/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import corr2d, rf_to_D
from physics.imaging import BornModel
from pilot_phase_asp import (DATA_ROOT, corrected_config, crop, effective_ds,
                             embed, padded_meta, projected_truth_screen)


def angle_images(born: BornModel, ds: torch.Tensor, D: torch.Tensor,
                 idx: torch.Tensor) -> torch.Tensor:
    """One complex adjoint image per transmit angle, summed only over frequency."""
    u = born.transmit_fields(ds, idx)
    b0 = born.scatter(D[idx])
    b0 = born.asp._ifft(born.asp._fft(b0) * born.surface_transfer.conj())
    b = born.asp.adjoint(b0, ds, born.omega_)
    return (b * (u * born.w_z).conj()).sum(dim=1)


def coherence(images: torch.Tensor, mask: torch.Tensor,
              scales: torch.Tensor) -> torch.Tensor:
    """Normalized complex coherence in [0, 1] for an image stack."""
    v = images / scales[:, None, None]
    numerator = (v.sum(0).abs().square() * mask).sum()
    denominator = (len(v) * v.abs().square().sum(0) * mask).sum()
    return numerator / denominator.clamp_min(1e-30)


def heldout_agreement(train: torch.Tensor, hold: torch.Tensor,
                      mask: torch.Tensor, train_scales: torch.Tensor,
                      hold_scales: torch.Tensor) -> torch.Tensor:
    """Mean real normalized inner product of held-out and input angle images."""
    reference = (train / train_scales[:, None, None]).mean(0)
    test = hold / hold_scales[:, None, None]
    inner = (test * reference.conj()[None] * mask).sum(dim=(-2, -1)).real
    test_norm = (test.abs().square() * mask).sum(dim=(-2, -1)).sqrt()
    ref_norm = (reference.abs().square() * mask).sum().sqrt()
    return (inner / (test_norm * ref_norm).clamp_min(1e-30)).mean()


@torch.no_grad()
def fixed_reference(born, D, train_idx, hold_idx, pad, top_frac):
    zero = torch.zeros(born.nz, born.nx, device=D.device)
    tr0 = angle_images(born, zero, D, train_idx)
    ho0 = angle_images(born, zero, D, hold_idx)
    region = crop(tr0, pad)
    power = region.abs().square().mean(0).sqrt()
    threshold = torch.quantile(power.flatten(), 1.0 - top_frac)
    mask = torch.zeros(born.nz, born.nx, device=D.device)
    if pad:
        mask[:, pad:-pad] = (power >= threshold).float()
    else:
        mask[:] = (power >= threshold).float()
    tr_scales = (tr0.abs().square() * mask).sum(dim=(-2, -1)).sqrt().clamp_min(1e-30)
    ho_scales = (ho0.abs().square() * mask).sum(dim=(-2, -1)).sqrt().clamp_min(1e-30)
    return zero, tr0, ho0, mask, tr_scales, ho_scales


@torch.no_grad()
def evaluate(born, ds, D, train_idx, hold_idx, mask, tr_scales,
             ho_scales, pad, truth_abs):
    tr = angle_images(born, ds, D, train_idx)
    ho = angle_images(born, ds, D, hold_idx)
    combined = tr.mean(0)
    return {
        "input_coherence": float(coherence(tr, mask, tr_scales).item()),
        "holdout_agreement": float(heldout_agreement(tr, ho, mask, tr_scales,
                                                      ho_scales).item()),
        "image_abs_corr": float(corr2d(crop(combined, pad).abs(), truth_abs).item()),
        "mask_pixels": int(mask.sum().item()),
    }


def optimize(born, D, train_idx, mask, tr_scales, args):
    raw = torch.nn.Parameter(torch.zeros(args.layers, args.controls,
                                         device=D.device))
    bulk = (torch.nn.Parameter(torch.zeros(2, device=D.device))
            if args.fit_bulk else None)
    params = [raw] + ([bulk] if bulk is not None else [])
    optimizer = torch.optim.Adam(params, lr=args.lr)
    best_loss = float("inf")
    best_raw, best_bulk = raw.detach().clone(), (bulk.detach().clone() if bulk is not None else None)
    history = []
    for step in range(args.opt_steps):
        optimizer.zero_grad(set_to_none=True)
        ds = effective_ds(raw, born.nz, born.nx, born.dz, args.limit_us,
                          args.pad, bulk, born.z0, args.bulk_limit_us)
        tr = angle_images(born, ds, D, train_idx)
        coh = coherence(tr, mask, tr_scales)
        curve = args.limit_us * torch.tanh(raw)
        reg = (curve.square().mean() / args.limit_us**2
               + 0.1 * (curve[:, 1:] - curve[:, :-1]).square().mean() / args.limit_us**2
               + 0.1 * (curve[1:] - curve[:-1]).square().mean() / args.limit_us**2)
        if bulk is not None:
            reg = reg + 0.1 * torch.tanh(bulk).square().mean()
        loss = -coh + args.reg * reg
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite coherence objective")
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best_raw = raw.detach().clone()
            best_bulk = bulk.detach().clone() if bulk is not None else None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        history.append({"step": step + 1, "input_coherence": float(coh.detach()),
                        "regularizer": float(reg.detach()), "loss": float(loss.detach())})
    return best_raw, best_bulk, history


def run_one(args, sample_id, device):
    sample = torch.load(DATA_ROOT / "shards" / f"{sample_id}.pt",
                        map_location="cpu", weights_only=False)
    cfg, meta = corrected_config(args.config, sample, args.n_freq)
    pm = padded_meta(meta, args.pad, cfg.grid.dx)
    born = BornModel(pm, cfg.grid.nx + 2 * args.pad, cfg.grid.nz,
                     cfg.grid.dx, cfg.grid.dz, cfg.physics.c0,
                     eps=cfg.physics.eps_evanescent,
                     spreading=cfg.physics.spreading).to(device)
    D = rf_to_D(sample["rf"].to(device), meta)
    tr_idx = torch.as_tensor(meta.train_idx, device=device)
    ho_idx = torch.as_tensor(meta.hold_idx, device=device)
    truth_abs = sample["m"].to(device).abs()  # Evaluation only.
    zero, _, _, mask, tr_scales, ho_scales = fixed_reference(
        born, D, tr_idx, ho_idx, args.pad, args.top_frac)
    t0 = time.monotonic()
    baseline = evaluate(born, zero, D, tr_idx, ho_idx, mask,
                        tr_scales, ho_scales, args.pad, truth_abs)
    print(json.dumps({"sample": sample_id, "stage": "uniform", **baseline}), flush=True)
    c_ds = embed(sample["delta_s"].to(device), args.pad)
    oracle = evaluate(born, c_ds, D, tr_idx, ho_idx, mask,
                      tr_scales, ho_scales, args.pad, truth_abs)
    truth_raw, truth_bulk, proj_meta = projected_truth_screen(
        c_ds, args.layers, args.controls, born.dz, args.limit_us,
        args.pad, born.z0, args.bulk_limit_us, args.fit_bulk)
    proj_ds = effective_ds(truth_raw, born.nz, born.nx, born.dz,
                           args.limit_us, args.pad, truth_bulk,
                           born.z0, args.bulk_limit_us)
    projected = evaluate(born, proj_ds, D, tr_idx, ho_idx, mask,
                         tr_scales, ho_scales, args.pad, truth_abs)
    print(json.dumps({"sample": sample_id, "stage": "known_c",
                      **oracle}), flush=True)
    print(json.dumps({"sample": sample_id, "stage": "projected_c_screen",
                      **projected}), flush=True)
    raw, bulk, history = optimize(born, D, tr_idx, mask, tr_scales, args)
    ds = effective_ds(raw, born.nz, born.nx, born.dz, args.limit_us,
                      args.pad, bulk, born.z0, args.bulk_limit_us)
    fitted = evaluate(born, ds, D, tr_idx, ho_idx, mask,
                      tr_scales, ho_scales, args.pad, truth_abs)
    print(json.dumps({"sample": sample_id, "stage": "fitted",
                      **fitted}), flush=True)
    result = {
        "sample_id": sample_id,
        "metadata": {"train_idx": list(map(int, meta.train_idx)),
                     "hold_idx": list(map(int, meta.hold_idx)),
                     "n_freq": len(meta.freqs)},
        "settings": {k: v for k, v in vars(args).items() if k not in ("samples", "out")},
        "uniform": baseline,
        "known_c": oracle,
        "projected_c_screen": {**projected, **proj_meta},
        "fitted": {**fitted,
                   "max_abs_control_us": float((args.limit_us * torch.tanh(raw)).abs().max()),
                   "bulk_coeff_us": ((args.bulk_limit_us * torch.tanh(bulk)).tolist()
                                     if bulk is not None else None)},
        "optimization": history,
        "elapsed_s": time.monotonic() - t0,
    }
    return result, raw.cpu(), bulk.cpu() if bulk is not None else None


def selftest():
    j = torch.ones(3, 2, 2, dtype=torch.complex64)
    mask = torch.ones(2, 2)
    scales = torch.ones(3)
    assert abs(float(coherence(j, mask, scales)) - 1.0) < 1e-6
    j[1] = -1
    assert 0 <= float(coherence(j, mask, scales)) < 1.0
    print("phase-only coherence selftest passed", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--samples", nargs="+", default=["train_000", "val_000"])
    p.add_argument("--out", type=Path, default=Path("runs/phase_only_pilot"))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--controls", type=int, default=48)
    p.add_argument("--limit-us", type=float, default=0.2)
    p.add_argument("--bulk-limit-us", type=float, default=2.0)
    p.add_argument("--fit-bulk", action="store_true")
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--opt-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--reg", type=float, default=0.01)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    if not 0 < args.top_frac < 1:
        p.error("--top-frac must be between 0 and 1")
    torch.manual_seed(20260916)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_id in args.samples:
        row, raw, bulk = run_one(args, sample_id, device)
        (args.out / f"{sample_id}.json").write_text(json.dumps(row, indent=2) + "\n")
        torch.save({"raw_phase_controls": raw, "raw_bulk": bulk},
                   args.out / f"{sample_id}.pt")
        rows.append(row)
        torch.cuda.empty_cache()
    (args.out / "summary.json").write_text(json.dumps({
        "samples": args.samples,
        "mean": {key: {metric: sum(r[key][metric] for r in rows) / len(rows)
                       for metric in ("input_coherence", "holdout_agreement",
                                      "image_abs_corr")}
                 for key in ("uniform", "known_c", "projected_c_screen", "fitted")},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

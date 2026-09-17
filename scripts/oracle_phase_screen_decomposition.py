"""Oracle decomposition for the V2 multilayer phase-screen model.

This experiment separates three possible failure sources:

1. representation error:
      GT slowness -> ideal K-layer discrete phase screens
2. teacher/projection error:
      ideal K-layer screens -> K x controls bounded teacher screens
3. network error:
      teacher screens -> RF-predicted screens from a checkpoint

All phase-screen variants use the V2 *discrete* screen definition.  The legacy
block-spread/slab ``effective_ds`` path is intentionally not used here.

Example
-------
python scripts/oracle_phase_screen_decomposition.py \
    --checkpoint runs/phase_screen_v2_L4_K24/best.pt \
    --split val --count 20 --layers 1 2 4 8 \
    --out runs/oracle_decomposition --gpu 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from common import corr2d, demod_iq, rf_to_D  # noqa: E402
from models.phase_screen import PhaseScreenModel, coherence, heldout_agreement  # noqa: E402
from physics.imaging import BornModel  # noqa: E402
from physics.phase_screen import controls_to_discrete_ds  # noqa: E402
from scripts.pilot_phase_asp import (  # noqa: E402
    DATA_ROOT,
    corrected_config,
    crop,
    embed,
    padded_meta,
    projected_truth_screen,
)


def sample_ids(split: str, count: int) -> list[str]:
    max_n = {"train": 400, "val": 50, "test": 50}[split]
    if not (1 <= count <= max_n):
        raise ValueError(f"count must be in [1, {max_n}]")
    return [f"{split}_{i:03d}" for i in
            np.linspace(0, max_n - 1, count).round().astype(int)]


def fixed_reference(images: torch.Tensor, train_idx: torch.Tensor,
                    hold_idx: torch.Tensor, pad: int, top_frac: float):
    """Uniform-propagation mask/scales used for every candidate model."""
    train = images[train_idx]
    hold = images[hold_idx]
    region = crop(train, pad)
    power = region.abs().square().mean(0).sqrt()
    threshold = torch.quantile(power.flatten(), 1.0 - top_frac)
    mask = torch.zeros_like(power)
    mask[:] = (power >= threshold).float()
    if pad:
        mask = torch.nn.functional.pad(mask, (pad, pad))
    train_scales = (train.abs().square() * mask).sum((-2, -1)).sqrt().clamp_min(1e-30)
    hold_scales = (hold.abs().square() * mask).sum((-2, -1)).sqrt().clamp_min(1e-30)
    return mask, train_scales, hold_scales


@torch.no_grad()
def images_and_field(born: BornModel, ds: torch.Tensor, D: torch.Tensor,
                     all_idx: torch.Tensor):
    """Per-angle adjoint images and transmit field for one propagation model."""
    u = born.transmit_fields(ds, all_idx)
    b0 = born.scatter(D[all_idx])
    b0 = born.asp._ifft(born.asp._fft(b0) * born.surface_transfer.conj())
    b = born.asp.adjoint(b0, ds, born.omega_)
    images = (b * (u * born.w_z).conj()).sum(dim=1)
    return images, u


def field_error(u: torch.Tensor, gt: torch.Tensor, pad: int,
                z0: float, dz: float, zmin_mm=5.0, zmax_mm=40.0) -> float:
    lo = max(0, int(np.ceil((zmin_mm * 1e-3 - z0) / dz)))
    hi = min(u.shape[-2], int(np.floor((zmax_mm * 1e-3 - z0) / dz)) + 1)
    a = u[..., lo:hi, pad:-pad] if pad else u[..., lo:hi, :]
    b = gt[..., lo:hi, pad:-pad] if pad else gt[..., lo:hi, :]
    return float((a - b).norm() / b.norm().clamp_min(1e-20))


def score(images: torch.Tensor, u: torch.Tensor, gt_u: torch.Tensor,
          train_idx: torch.Tensor, hold_idx: torch.Tensor,
          mask: torch.Tensor, train_scales: torch.Tensor,
          hold_scales: torch.Tensor, pad: int, truth_abs: torch.Tensor,
          born: BornModel):
    # Reuse the batched V2 metrics to keep training/evaluation semantics aligned.
    tr = images[train_idx][None]
    ho = images[hold_idx][None]
    coh = coherence(tr, mask[None], train_scales[None])[0]
    hold = heldout_agreement(tr, ho, mask[None], train_scales[None],
                            hold_scales[None])[0]
    image = crop(images.mean(dim=0), pad)
    return {
        "input_coherence": float(coh),
        "holdout_agreement": float(hold),
        "image11_abs_corr": float(corr2d(image.abs(), truth_abs)),
        "field_rel_l2": field_error(u, gt_u, pad, born.z0, born.dz),
    }


def projection_to_ds(true_ds: torch.Tensor, layers: int, controls: int,
                     born: BornModel, pad: int, limit_us: float):
    """Project GT medium to bounded V2 teacher controls and discrete screens."""
    raw, _, info = projected_truth_screen(
        true_ds, layers, controls, born.dz, limit_us, pad, born.z0,
        bulk_limit_us=2.0, fit_bulk=False,
    )
    ds = controls_to_discrete_ds(raw, born.nz, born.nx, born.dz,
                                 limit_us=limit_us, pad=pad)
    tau = limit_us * torch.tanh(raw)
    info = dict(info)
    info.update({
        "control_count": controls,
        "control_max_abs_us": float(tau.abs().max()),
        "control_frac_gt_95pct_limit": float(
            (tau.abs() > 0.95 * limit_us).float().mean()),
    })
    return ds, info


def ideal_to_ds(true_ds: torch.Tensor, layers: int, born: BornModel,
                pad: int, oracle_limit_us: float):
    """Full-lateral-resolution K-screen oracle with practically no clipping."""
    raw, _, info = projected_truth_screen(
        true_ds, layers, born.nx, born.dz, oracle_limit_us, pad, born.z0,
        bulk_limit_us=2.0, fit_bulk=False,
    )
    if info["saturated_control_fraction"] > 0:
        raise RuntimeError(
            f"oracle_limit_us={oracle_limit_us} clipped ideal K={layers} screen; "
            "increase --oracle-limit-us"
        )
    ds = controls_to_discrete_ds(raw, born.nz, born.nx, born.dz,
                                 limit_us=oracle_limit_us, pad=pad)
    return ds, info


def load_checkpoint(path: Path | None, first_sample: dict, device):
    if path is None:
        return None, None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]
    cfg, meta = corrected_config(saved_args["config"], first_sample,
                                 saved_args["n_freq"])
    cfg.model.normalize_iq = True
    model = PhaseScreenModel(
        cfg,
        meta,
        layers=int(saved_args.get("layers", 4)),
        controls=int(saved_args.get("controls", 24)),
        fit_bulk=bool(saved_args.get("fit_bulk", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, {"checkpoint": str(path), "step": int(ckpt["step"]),
                   "layers": model.layers, "controls": model.controls,
                   "fit_bulk": model.fit_bulk,
                   "config": saved_args["config"],
                   "n_freq": int(saved_args["n_freq"])}


def summarize(rows: list[dict]):
    methods = list(dict.fromkeys(r["method"] for r in rows))
    metrics = ("input_coherence", "holdout_agreement", "image11_abs_corr",
               "field_rel_l2")
    result = {}
    uniform = {k: np.mean([r[k] for r in rows if r["method"] == "uniform"])
               for k in metrics}
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        item = {k: float(np.mean([r[k] for r in subset])) for k in metrics}
        item["n"] = len(subset)
        item["delta_hold_vs_uniform"] = item["holdout_agreement"] - uniform["holdout_agreement"]
        item["delta_image_vs_uniform"] = item["image11_abs_corr"] - uniform["image11_abs_corr"]
        if method != "uniform":
            by_sample_uniform = {r["sample"]: r for r in rows if r["method"] == "uniform"}
            item["hold_wins_vs_uniform"] = sum(
                r["holdout_agreement"] > by_sample_uniform[r["sample"]]["holdout_agreement"]
                for r in subset
            )
            item["image_wins_vs_uniform"] = sum(
                r["image11_abs_corr"] > by_sample_uniform[r["sample"]]["image11_abs_corr"]
                for r in subset
            )
        result[method] = item
    return result


def print_table(summary: dict):
    print("\nmethod                     hold        image_corr   field_rel_l2   d_hold      d_image")
    print("-" * 91)
    for method, r in summary.items():
        print(f"{method:26s} {r['holdout_agreement']:10.6f}  "
              f"{r['image11_abs_corr']:10.6f}  {r['field_rel_l2']:12.6f}  "
              f"{r['delta_hold_vs_uniform']:+10.6f}  "
              f"{r['delta_image_vs_uniform']:+10.6f}")


@torch.no_grad()
def run_one(sample_id: str, args, device, predicted_model):
    sample = torch.load(DATA_ROOT / "shards" / f"{sample_id}.pt",
                        map_location="cpu", weights_only=False)
    cfg, meta = corrected_config(args.config, sample, args.n_freq)
    born = BornModel(padded_meta(meta, args.pad, cfg.grid.dx),
                     cfg.grid.nx + 2 * args.pad, cfg.grid.nz,
                     cfg.grid.dx, cfg.grid.dz, cfg.physics.c0,
                     eps=cfg.physics.eps_evanescent,
                     spreading=cfg.physics.spreading).to(device)
    D = rf_to_D(sample["rf"].to(device), meta)
    true_ds = embed(sample["delta_s"].to(device), args.pad)
    if args.pad:
        true_ds[:, :args.pad] = 0
        true_ds[:, -args.pad:] = 0
    truth_abs = sample["m"].abs().to(device)
    all_idx = torch.arange(cfg.acq.n_angles, device=device)
    train_idx = torch.as_tensor(meta.train_idx, device=device)
    hold_idx = torch.as_tensor(meta.hold_idx, device=device)

    gt_images, gt_u = images_and_field(born, true_ds, D, all_idx)
    zero = torch.zeros_like(true_ds)
    uniform_images, uniform_u = images_and_field(born, zero, D, all_idx)
    mask, train_scales, hold_scales = fixed_reference(
        uniform_images, train_idx, hold_idx, args.pad, args.top_frac)

    rows = []

    def record(method, ds, images, u, extra=None):
        row = {
            "sample": sample_id,
            "case": sample["metadata"].get("case"),
            "method": method,
            **score(images, u, gt_u, train_idx, hold_idx, mask,
                    train_scales, hold_scales, args.pad, truth_abs, born),
        }
        if extra:
            row.update(extra)
        rows.append(row)

    record("uniform", zero, uniform_images, uniform_u)
    record("gt_speed_asm", true_ds, gt_images, gt_u)

    for k in args.layers:
        ideal_ds, ideal_info = ideal_to_ds(true_ds, k, born, args.pad,
                                           args.oracle_limit_us)
        ideal_images, ideal_u = images_and_field(born, ideal_ds, D, all_idx)
        record(f"ideal_K{k}", ideal_ds, ideal_images, ideal_u,
               {"projection": ideal_info})

        teacher_ds, teacher_info = projection_to_ds(
            true_ds, k, args.controls, born, args.pad, args.limit_us)
        teacher_images, teacher_u = images_and_field(born, teacher_ds, D, all_idx)
        record(f"teacher_K{k}_C{args.controls}", teacher_ds,
               teacher_images, teacher_u, {"projection": teacher_info})

    if predicted_model is not None:
        # The checkpoint must use the same acquisition geometry.  It gets only
        # the training/context angles, exactly as in validation during training.
        iq = demod_iq(sample["rf"][None].to(device), predicted_model.meta)
        pred_train_idx = torch.as_tensor(predicted_model.meta.train_idx,
                                         device=device)
        raw, bulk_raw = predicted_model.predict_controls(iq, pred_train_idx)
        pred_ds_b = predicted_model.screen_to_slowness(raw, bulk_raw)
        pred_ds = pred_ds_b[0]
        pred_images, pred_u = images_and_field(born, pred_ds, D, all_idx)
        phase_controls = predicted_model.limit_us * torch.tanh(raw[0])
        record(
            f"predicted_K{predicted_model.layers}_C{predicted_model.controls}",
            pred_ds, pred_images, pred_u,
            {"predicted_max_control_us": float(phase_controls.abs().max()),
             "predicted_frac_gt_95pct_limit": float(
                 (phase_controls.abs() > 0.95 * predicted_model.limit_us)
                 .float().mean())},
        )

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--controls", type=int, default=24)
    p.add_argument("--limit-us", type=float, default=0.2)
    p.add_argument("--oracle-limit-us", type=float, default=20.0,
                   help="large bound used only for full-resolution ideal screens")
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ids = sample_ids(args.split, args.count)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    args.out.mkdir(parents=True, exist_ok=True)

    first = torch.load(DATA_ROOT / "shards" / f"{ids[0]}.pt",
                       map_location="cpu", weights_only=False)
    predicted_model, checkpoint_info = load_checkpoint(args.checkpoint, first, device)
    if checkpoint_info is not None:
        # Use the checkpoint acquisition settings for predicted-screen scoring.
        if checkpoint_info["config"] != args.config or checkpoint_info["n_freq"] != args.n_freq:
            print(json.dumps({
                "warning": "checkpoint acquisition settings differ from CLI",
                "checkpoint_config": checkpoint_info["config"],
                "cli_config": args.config,
                "checkpoint_n_freq": checkpoint_info["n_freq"],
                "cli_n_freq": args.n_freq,
            }), flush=True)

    rows = []
    started = time.monotonic()
    with (args.out / "results.jsonl").open("w") as f:
        for i, sid in enumerate(ids, 1):
            sample_rows = run_one(sid, args, device, predicted_model)
            rows.extend(sample_rows)
            for row in sample_rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            partial = summarize(rows)
            print(json.dumps({
                "event": "sample_complete",
                "sample": sid,
                "completed": i,
                "total": len(ids),
                "elapsed_s": time.monotonic() - started,
                "scores": {r["method"]: {
                    "hold": r["holdout_agreement"],
                    "image": r["image11_abs_corr"],
                    "field": r["field_rel_l2"],
                } for r in sample_rows},
            }), flush=True)
            (args.out / "summary.json").write_text(json.dumps({
                "args": vars(args) | {
                    "out": str(args.out),
                    "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                },
                "checkpoint": checkpoint_info,
                "ids": ids[:i],
                "summary": partial,
            }, indent=2) + "\n")
            torch.cuda.empty_cache()

    final = summarize(rows)
    print_table(final)
    print(json.dumps({"event": "done", "elapsed_s": time.monotonic() - started,
                      "summary": final}), flush=True)


if __name__ == "__main__":
    main()

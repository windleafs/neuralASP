"""Compare a learned V2 phase screen with fixed training-set mean screens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import corr2d
from models.phase_screen import PhaseScreenModel, heldout_agreement
from scripts.pilot_phase_asp import DATA_ROOT, corrected_config
from train_phase_screen import build_cache, sample_ids, teacher_loss


def raw_from_target(tau, bulk, model):
    raw = torch.atanh((tau / model.limit_us).clamp(-0.999, 0.999))
    if model.fit_bulk:
        bulk_raw = torch.atanh((bulk / model.bulk_limit_us).clamp(-0.999, 0.999))
    else:
        bulk_raw = None
    return raw, bulk_raw


@torch.no_grad()
def score_ds(model, ds, item, train_idx, hold_idx):
    tr = model.angle_images(ds, item["D"], train_idx)
    ho = model.angle_images(ds, item["D"], hold_idx)
    ref = item["ref"]
    hold = heldout_agreement(tr, ho, ref["mask"],
                            ref["train_scales"], ref["hold_scales"])[0]
    image = torch.cat([tr, ho], dim=1).mean(dim=1)[0]
    if model.pad:
        image = image[:, model.pad:-model.pad]
    return float(hold), float(corr2d(image.abs(), item["truth_abs"]))


def mean_metrics(rows, key):
    return {
        "hold": float(np.mean([r[key]["hold"] for r in rows])),
        "image11": float(np.mean([r[key]["image11"] for r in rows])),
        "teacher_loss": float(np.mean([r[key]["teacher_loss"] for r in rows])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    first = torch.load(DATA_ROOT / "shards" / "train_000.pt",
                       map_location="cpu", weights_only=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt["args"]
    cfg, meta = corrected_config(saved_args["config"], first, saved_args["n_freq"])
    cfg.model.normalize_iq = True
    model = PhaseScreenModel(
        cfg, meta,
        layers=saved_args.get("layers", 4),
        controls=saved_args.get("controls", 24),
        fit_bulk=saved_args.get("fit_bulk", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tr_idx = torch.as_tensor(meta.train_idx, device=device)
    ho_idx = torch.as_tensor(meta.hold_idx, device=device)
    train_ids = sample_ids("train", saved_args["train_per_case"])
    val_ids = sample_ids("val", saved_args["val_per_case"])
    train = build_cache(train_ids, model, meta, tr_idx, ho_idx,
                        device, saved_args["top_frac"], need_reference=False)
    val = build_cache(val_ids, model, meta, tr_idx, ho_idx,
                      device, saved_args["top_frac"])

    global_tau = torch.stack([x["teacher_tau"] for x in train]).mean(0)
    global_bulk = (torch.stack([x["teacher_bulk_us"] for x in train]).mean(0)
                   if model.fit_bulk else None)

    with torch.no_grad():
        predicted_train = [model.predict_controls(x["iq"], tr_idx) for x in train]
        predicted_tau_stack = torch.stack([
            model.limit_us * torch.tanh(raw) for raw, _ in predicted_train])
        network_mean_tau = predicted_tau_stack.mean(0)
        if model.fit_bulk:
            predicted_bulk_stack = torch.stack([
                model.bulk_limit_us * torch.tanh(bulk)
                for _, bulk in predicted_train])
            network_mean_bulk = predicted_bulk_stack.mean(0)
        else:
            predicted_bulk_stack = None
            network_mean_bulk = None

    mid = saved_args["train_per_case"]
    groups = (train[:mid], train[mid:])
    case_tau = [torch.stack([x["teacher_tau"] for x in group]).mean(0)
                for group in groups]
    case_bulk = ([torch.stack([x["teacher_bulk_us"] for x in group]).mean(0)
                  for group in groups] if model.fit_bulk else [None, None])

    rows = []
    for item in val:
        sample_id = item["id"]
        case_idx = 0 if int(sample_id.split("_")[1]) < 25 else 1
        with torch.no_grad():
            pred_raw, pred_bulk_raw = model.predict_controls(item["iq"], tr_idx)
            pred_tau = model.limit_us * torch.tanh(pred_raw)
            pred_bulk = (model.bulk_limit_us * torch.tanh(pred_bulk_raw)
                         if model.fit_bulk else None)

            def candidate(tau, bulk):
                raw, bulk_raw = raw_from_target(tau, bulk, model)
                return tau, bulk, raw, bulk_raw

            candidates = {
                "network": (pred_tau, pred_bulk, pred_raw,
                            pred_bulk_raw if model.fit_bulk else None),
                "network_mean": candidate(network_mean_tau, network_mean_bulk),
                "global_mean": candidate(global_tau, global_bulk),
                "case_mean": candidate(case_tau[case_idx], case_bulk[case_idx]),
                "teacher": candidate(item["teacher_tau"], item["teacher_bulk_us"]),
            }
            row = {"sample": sample_id,
                   "uniform_hold": float(item["ref"]["uniform_holdout_agreement"][0]),
                   "uniform_image11": item["uniform_image11"]}
            for name, (tau, bulk, raw, bulk_raw) in candidates.items():
                ds = model.screen_to_slowness(raw, bulk_raw)
                hold, image11 = score_ds(model, ds, item, tr_idx, ho_idx)
                bulk_for_loss = (bulk if bulk is not None
                                 else tau.new_zeros(tau.shape[0], 2))
                row[name] = {
                    "hold": hold,
                    "image11": image11,
                    "teacher_loss": float(teacher_loss(
                        tau, bulk_for_loss, item, model)),
                    "max_control_us": float(tau.abs().max()),
                    "bulk_us": (bulk[0].tolist() if bulk is not None else None),
                }
        rows.append(row)
        print(json.dumps(row), flush=True)

    result = {"checkpoint": str(args.checkpoint), "step": ckpt["step"],
              "samples": val_ids,
              "predicted_train_control_std_us":
                  float(predicted_tau_stack.std(dim=0).square().mean().sqrt()),
              "predicted_train_bulk_std_us":
                  (predicted_bulk_stack.std(dim=0)[0].tolist()
                   if predicted_bulk_stack is not None else None),
              "mean_uniform_hold": float(np.mean([r["uniform_hold"] for r in rows])),
              "mean_uniform_image11": float(np.mean([r["uniform_image11"] for r in rows])),
              "mean": {key: mean_metrics(rows, key) for key in candidates},
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "summary", "step": ckpt["step"],
                      "mean": result["mean"]}), flush=True)


if __name__ == "__main__":
    main()

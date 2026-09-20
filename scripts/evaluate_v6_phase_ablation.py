"""Evaluate V6 phase correction with strict same-checkpoint ablations.

For each sample this script predicts one set of V6 latent components, then
reconstructs four propagation states without changing the network:

  Uniform      : no mean delay, no active phase modes
  Mean-only    : mean-delay branch only
  Active-only  : K active phase coefficients only
  Mean+Active  : full V6 phase correction

Amplitude is fixed to zero in every ablation.

This isolates the independent contribution of the K active modes from the
mean-delay branch and also exports the per-sample active coefficients.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from common import demod_iq, rf_to_D
from models.active_mode_phase_amplitude import ActiveModePhaseAmplitudeModel
from models.phase_screen import heldout_agreement
from physics.phase_screen import mean_controls_to_ds
from scripts.pilot_phase_asp import DATA_ROOT, corrected_config
from train_phase_screen import sample_ids


def build_model_from_checkpoint(checkpoint, active_basis, device, n_freq=None):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    first = torch.load(
        DATA_ROOT / "shards" / "val_000.pt",
        map_location="cpu", weights_only=False)
    config_path = saved.get(
        "config", "configs/l11_ultrawave_500_11angle.yaml")
    if n_freq is None:
        n_freq = int(saved.get("n_freq", 64))
    cfg, meta = corrected_config(config_path, first, n_freq)
    cfg.model.normalize_iq = True

    model = ActiveModePhaseAmplitudeModel(
        cfg, meta,
        active_basis_path=active_basis,
        active_rank=int(saved.get("active_rank", 6)),
        active_basis_source=saved.get("active_basis_source", "balanced"),
        mean_controls=int(saved.get("mean_controls", 8)),
        mean_limit_us=float(saved.get("mean_limit_us", 2.0)),
        phase_coeff_limit_us=float(saved.get("phase_coeff_limit_us", 0.2)),
        amplitude_coeff_limit_np=float(saved.get("amplitude_coeff_limit_np", 0.2)),
        phase_gate_init=float(saved.get("phase_gate_init", 0.02)),
        amplitude_gate_init=float(saved.get("amplitude_gate_init", 0.02)),
        amplitude_freq_power=float(saved.get("amplitude_freq_power", 1.0)),
    ).to(device)

    current = model.state_dict()
    copied, skipped = 0, 0
    for key, value in ckpt["model"].items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            current[key] = value
            copied += 1
        else:
            skipped += 1
    model.load_state_dict(current, strict=True)
    model.eval()
    return model, cfg, meta, ckpt, copied, skipped


def reference_metrics(model, D, train_idx, hold_idx, top_frac):
    return model.reference(D, train_idx, hold_idx, top_frac=top_frac)


def score_state(model, D, ds, zero_amp, train_idx, hold_idx, ref):
    tr = model.angle_images(ds, D, train_idx, amplitude_rate=zero_amp)
    ho = model.angle_images(ds, D, hold_idx, amplitude_rate=zero_amp)
    hold = heldout_agreement(
        tr, ho, ref["mask"], ref["train_scales"], ref["hold_scales"])
    return float(hold[0])


@torch.no_grad()
def evaluate_sample(model, meta, sample_id, train_idx, hold_idx, top_frac):
    device = next(model.parameters()).device
    sample = torch.load(
        DATA_ROOT / "shards" / f"{sample_id}.pt",
        map_location="cpu", weights_only=False)
    rf = sample["rf"][None].to(device)
    iq = demod_iq(rf, meta)
    D = rf_to_D(rf, meta)

    phase_raw, mean_raw, amp_raw = model.predict_all_components(iq, train_idx)
    phase_coeff, _ = model.active_coefficients(phase_raw, amp_raw)

    phase_templates = model.phase_active_templates.to(
        device=phase_coeff.device, dtype=phase_coeff.dtype)
    ds_active = torch.einsum("bk,kzx->bzx", phase_coeff, phase_templates)
    ds_mean = mean_controls_to_ds(
        mean_raw, model.born.nz, model.born.nx, model.born.dz,
        model.mean_limit_us)

    zero_ds = torch.zeros_like(ds_mean)
    zero_amp = torch.zeros_like(ds_mean)
    ds_full = ds_mean + ds_active

    ref = reference_metrics(model, D, train_idx, hold_idx, top_frac)
    uniform = float(ref["uniform_holdout_agreement"][0])
    mean_only = score_state(
        model, D, ds_mean, zero_amp, train_idx, hold_idx, ref)
    active_only = score_state(
        model, D, ds_active, zero_amp, train_idx, hold_idx, ref)
    full = score_state(
        model, D, ds_full, zero_amp, train_idx, hold_idx, ref)

    coeff = phase_coeff[0].detach().cpu().tolist()
    mean_controls_us = (
        model.mean_limit_us * torch.tanh(mean_raw[0])).detach().cpu().tolist()

    return {
        "sample": sample_id,
        "uniform_hold": uniform,
        "mean_only_hold": mean_only,
        "active_only_hold": active_only,
        "full_hold": full,
        "gain_mean_vs_uniform": mean_only - uniform,
        "gain_active_vs_uniform": active_only - uniform,
        "gain_full_vs_uniform": full - uniform,
        "gain_active_on_top_of_mean": full - mean_only,
        "gain_mean_on_top_of_active": full - active_only,
        "phase_coeff_us": coeff,
        "max_abs_phase_coeff_us": max(abs(v) for v in coeff) if coeff else 0.0,
        "mean_controls_us": mean_controls_us,
    }


def summarize(rows):
    def mean(key):
        return float(np.mean([r[key] for r in rows]))
    return {
        "n": len(rows),
        "mean_uniform_hold": mean("uniform_hold"),
        "mean_mean_only_hold": mean("mean_only_hold"),
        "mean_active_only_hold": mean("active_only_hold"),
        "mean_full_hold": mean("full_hold"),
        "mean_gain_mean_vs_uniform": mean("gain_mean_vs_uniform"),
        "mean_gain_active_vs_uniform": mean("gain_active_vs_uniform"),
        "mean_gain_full_vs_uniform": mean("gain_full_vs_uniform"),
        "mean_gain_active_on_top_of_mean": mean("gain_active_on_top_of_mean"),
        "mean_gain_mean_on_top_of_active": mean("gain_mean_on_top_of_active"),
        "full_wins_vs_mean_count": int(sum(
            r["full_hold"] > r["mean_only_hold"] for r in rows)),
        "full_wins_vs_active_count": int(sum(
            r["full_hold"] > r["active_only_hold"] for r in rows)),
        "mean_wins_vs_uniform_count": int(sum(
            r["mean_only_hold"] > r["uniform_hold"] for r in rows)),
        "active_wins_vs_uniform_count": int(sum(
            r["active_only_hold"] > r["uniform_hold"] for r in rows)),
    }


def coefficient_stats(rows):
    coeff = np.asarray([r["phase_coeff_us"] for r in rows], dtype=np.float64)
    if coeff.size == 0:
        return {}
    return {
        "mean_us": coeff.mean(axis=0).tolist(),
        "std_us": coeff.std(axis=0).tolist(),
        "mean_abs_us": np.abs(coeff).mean(axis=0).tolist(),
        "max_abs_us": np.abs(coeff).max(axis=0).tolist(),
        "per_mode_sample_std_us": coeff.std(axis=0).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--active-basis", type=Path, required=True)
    p.add_argument("--sample-ids", nargs="+")
    p.add_argument("--val-per-case", type=int, default=25)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-freq", type=int)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not (0 < args.top_frac <= 1):
        p.error("--top-frac must lie in (0,1]")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    model, cfg, meta, ckpt, copied, skipped = build_model_from_checkpoint(
        args.checkpoint, args.active_basis, device, args.n_freq)
    train_idx = torch.as_tensor(meta.train_idx, device=device)
    hold_idx = torch.as_tensor(meta.hold_idx, device=device)

    ids = args.sample_ids or sample_ids("val", args.val_per_case)
    rows = []
    for sid in ids:
        row = evaluate_sample(
            model, meta, sid, train_idx, hold_idx, args.top_frac)
        rows.append(row)
        print(json.dumps({
            "event": "v6_phase_ablation_sample",
            "sample": sid,
            "uniform": row["uniform_hold"],
            "mean_only": row["mean_only_hold"],
            "active_only": row["active_only_hold"],
            "full": row["full_hold"],
            "active_on_top_of_mean": row["gain_active_on_top_of_mean"],
            "phase_coeff_us": row["phase_coeff_us"],
        }), flush=True)

    payload = {
        "experiment": "v6 phase same-checkpoint ablation",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "active_basis": str(args.active_basis),
        "active_rank": int(model.active_rank),
        "active_basis_source": model.active_basis_source,
        "phase_gate": float(model.phase_gate_value().detach().cpu()),
        "loaded_tensors": copied,
        "skipped_tensors": skipped,
        "summary": summarize(rows),
        "coefficient_stats": coefficient_stats(rows),
        "rows": rows,
    }
    out_json = args.out / "v6_phase_ablation_summary.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "event": "done",
        "summary": payload["summary"],
        "out": str(out_json),
    }), flush=True)


if __name__ == "__main__":
    main()

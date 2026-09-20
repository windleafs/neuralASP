"""Cross-sample propagation active-subspace analysis with bias checks.

This script extends the single-sample sensitivity experiment in three ways:

1. Finite-difference linearity check
   Recompute phase/amplitude Jacobians at multiple perturbation magnitudes and
   compare each candidate-mode derivative with a reference step.

2. Depth coverage-bias check
   Report both global sensitivity and a below-screen conditional sensitivity
   normalized only over pixels at or below the perturbed screen depth.

3. Cross-sample active subspace
   Aggregate parameter-space Gram matrices
       G = (1/N) sum_i J_i^T J_i
   and eigendecompose G to identify propagation modes that remain important
   across tissue realizations.

Phase and amplitude are analyzed separately. Combined phase/amplitude ranking
is intentionally deferred until empirical parameter scales are available.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from common import demod_iq, rf_to_D
from physics.active_subspace import (
    below_screen_mask,
    gram_spectrum,
    linearity_diagnostics,
    subspace_overlap,
    subspace_overlap_curve,
)
from physics.propagation_modes import (
    amplitude_screen_perturbation,
    lateral_dct_modes,
    phase_screen_perturbation,
    select_depth_screen_indices,
)
from scripts.evaluate_phase_amplitude_screen import DATA_ROOT, build_model, physical_crop
from scripts.oracle_amplitude_screen import fixed_phase_correction


def compound(model, D, ds, amp, all_idx):
    return model.angle_images(ds, D, all_idx, amplitude_rate=amp).mean(dim=1)[0]


def response_vector(delta, mask, pad, base_norm):
    d = physical_crop(delta, pad)
    m = physical_crop(mask, pad).bool()
    z = d[m]
    return torch.cat([z.real, z.imag], dim=0) / base_norm


def logenv_response(plus, minus, mask, pad, eps):
    p = physical_crop(plus, pad).abs().clamp_min(eps)
    n = physical_crop(minus, pad).abs().clamp_min(eps)
    m = physical_crop(mask, pad).bool()
    return (0.5 * (torch.log(p) - torch.log(n)))[m]


def normalized_conditional_sensitivity(delta, base, mask, pad, depth_index):
    cond = below_screen_mask(mask, depth_index)
    d = physical_crop(delta, pad)
    b = physical_crop(base, pad)
    m = physical_crop(cond, pad).bool()
    if not bool(m.any()):
        return 0.0
    denom = b[m].abs().square().sum().sqrt().clamp_min(1e-12)
    return float(d[m].abs().square().sum().sqrt() / denom)


def build_jacobian_for_step(model, D, ds0, amp0, mask, base, modes,
                            depth_idx, depths_mm, all_idx,
                            phase_step_us, amplitude_step_np, pad, log_eps):
    base_crop = physical_crop(base, pad)
    mask_crop = physical_crop(mask, pad).bool()
    base_norm = base_crop[mask_crop].abs().square().sum().sqrt().clamp_min(1e-12)

    phase_cols, amp_cols = [], []
    phase_log_cols, amp_log_cols = [], []
    phase_global = np.zeros((len(depth_idx), modes.shape[0]), dtype=np.float64)
    amp_global = np.zeros_like(phase_global)
    phase_cond = np.zeros_like(phase_global)
    amp_cond = np.zeros_like(phase_global)
    labels_phase, labels_amp = [], []

    for di, (zi, zmm) in enumerate(zip(depth_idx, depths_mm)):
        for k in range(modes.shape[0]):
            mode = modes[k]

            dp = phase_screen_perturbation(
                mode, model.born.nz, zi, model.born.dz, phase_step_us)
            p_plus = compound(model, D, ds0 + dp[None], amp0, all_idx)
            p_minus = compound(model, D, ds0 - dp[None], amp0, all_idx)
            p_delta = 0.5 * (p_plus - p_minus)
            p_vec = response_vector(p_delta, mask, pad, base_norm)
            phase_cols.append(p_vec / phase_step_us)
            phase_log_cols.append(
                logenv_response(p_plus, p_minus, mask, pad, log_eps) / phase_step_us)
            phase_global[di, k] = float(p_vec.norm() / phase_step_us)
            phase_cond[di, k] = normalized_conditional_sensitivity(
                p_delta, base, mask, pad, zi) / phase_step_us
            labels_phase.append(f"P:z{zmm:.1f}:k{k}")

            da = amplitude_screen_perturbation(
                mode, model.born.nz, zi, model.born.dz, amplitude_step_np)
            a_plus = compound(model, D, ds0, amp0 + da[None], all_idx)
            a_minus = compound(model, D, ds0, amp0 - da[None], all_idx)
            a_delta = 0.5 * (a_plus - a_minus)
            a_vec = response_vector(a_delta, mask, pad, base_norm)
            amp_cols.append(a_vec / amplitude_step_np)
            amp_log_cols.append(
                logenv_response(a_plus, a_minus, mask, pad, log_eps) / amplitude_step_np)
            amp_global[di, k] = float(a_vec.norm() / amplitude_step_np)
            amp_cond[di, k] = normalized_conditional_sensitivity(
                a_delta, base, mask, pad, zi) / amplitude_step_np
            labels_amp.append(f"A:z{zmm:.1f}:k{k}")

    return {
        "J_phase": torch.stack(phase_cols, dim=1),
        "J_amplitude": torch.stack(amp_cols, dim=1),
        "J_phase_log": torch.stack(phase_log_cols, dim=1),
        "J_amplitude_log": torch.stack(amp_log_cols, dim=1),
        "phase_global": phase_global,
        "amplitude_global": amp_global,
        "phase_conditional": phase_cond,
        "amplitude_conditional": amp_cond,
        "labels_phase": labels_phase,
        "labels_amplitude": labels_amp,
    }


def summarize_linearity(reference, test, active_rel_threshold):
    d = linearity_diagnostics(
        reference, test, active_rel_threshold=active_rel_threshold)
    valid = d["valid"]
    n_total = int(valid.numel())
    n_active = int(valid.sum())
    if n_active == 0:
        return {
            "active_columns": 0,
            "total_columns": n_total,
            "active_fraction": 0.0,
            "mean_cosine": float("nan"),
            "min_cosine": float("nan"),
            "mean_norm_ratio": float("nan"),
            "std_norm_ratio": float("nan"),
            "mean_relative_error": float("nan"),
            "max_relative_error": float("nan"),
        }
    cosine = d["cosine"][valid]
    norm_ratio = d["norm_ratio"][valid]
    relative_error = d["relative_error"][valid]
    return {
        "active_columns": n_active,
        "total_columns": n_total,
        "active_fraction": n_active / float(n_total),
        "mean_cosine": float(cosine.mean()),
        "min_cosine": float(cosine.min()),
        "mean_norm_ratio": float(norm_ratio.mean()),
        "std_norm_ratio": float(norm_ratio.std(unbiased=False)),
        "mean_relative_error": float(relative_error.mean()),
        "max_relative_error": float(relative_error.max()),
    }
def compact_spectrum(rep):
    return {
        "rank_90": rep["rank_90"],
        "rank_95": rep["rank_95"],
        "rank_99": rep["rank_99"],
        "leading_singular_values": rep["singular_values"][:12].detach().cpu().tolist(),
        "leading_energy_fraction": rep["energy_fraction"][:12].detach().cpu().tolist(),
    }


def plot_population_spectra(path, phase_rep, amp_rep, phase_log_rep, amp_log_rep, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    for label, rep in (
        ("phase complex", phase_rep),
        ("amplitude complex", amp_rep),
        ("phase log-envelope", phase_log_rep),
        ("amplitude log-envelope", amp_log_rep),
    ):
        s = rep["singular_values"].detach().cpu().numpy()
        c = rep["cumulative_energy"].detach().cpu().numpy()
        axes[0].semilogy(np.arange(1, len(s)+1), s / max(s[0], 1e-30),
                        marker="o", markersize=3, label=label)
        axes[1].plot(np.arange(1, len(c)+1), c, marker="o", markersize=3, label=label)
    axes[0].set(xlabel="Mode rank", ylabel="Normalized singular value",
                title="Population active-subspace spectrum")
    axes[1].set(xlabel="Mode rank", ylabel="Cumulative response energy",
                title="Population cumulative energy", ylim=(0, 1.02))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_depth_bias(path, depths_mm, pg, pc, ag, ac, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    axes[0].plot(depths_mm, np.sqrt(np.mean(pg**2, axis=1)), marker="o", label="global")
    axes[0].plot(depths_mm, np.sqrt(np.mean(pc**2, axis=1)), marker="o", label="below-screen conditional")
    axes[0].set(title="Phase depth sensitivity", xlabel="Screen depth [mm]",
                ylabel="RMS response per unit perturbation")
    axes[1].plot(depths_mm, np.sqrt(np.mean(ag**2, axis=1)), marker="o", label="global")
    axes[1].plot(depths_mm, np.sqrt(np.mean(ac**2, axis=1)), marker="o", label="below-screen conditional")
    axes[1].set(title="Amplitude depth sensitivity", xlabel="Screen depth [mm]",
                ylabel="RMS response per unit perturbation")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_linearity(path, linearity_rows, dpi):
    phase_steps = [r["phase_step_us"] for r in linearity_rows]
    amp_steps = [r["amplitude_step_np"] for r in linearity_rows]
    phase_err = [r["phase"]["mean_relative_error"] for r in linearity_rows]
    amp_err = [r["amplitude"]["mean_relative_error"] for r in linearity_rows]
    phase_one_minus_cos = [1.0 - r["phase"]["mean_cosine"] for r in linearity_rows]
    amp_one_minus_cos = [1.0 - r["amplitude"]["mean_cosine"] for r in linearity_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    axes[0].plot(phase_steps, phase_err, marker="o", label="relative error")
    axes[0].plot(phase_steps, phase_one_minus_cos, marker="o", label="1 - cosine")
    axes[0].set(
        xlabel="Phase perturbation [us]", ylabel="Deviation from reference Jacobian",
        title="Phase finite-difference linearity")

    axes[1].plot(amp_steps, amp_err, marker="o", label="relative error")
    axes[1].plot(amp_steps, amp_one_minus_cos, marker="o", label="1 - cosine")
    axes[1].set(
        xlabel="Amplitude perturbation [Np]", ylabel="Deviation from reference Jacobian",
        title="Amplitude finite-difference linearity")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_cross_family_overlap(path, ranks, complex_overlap, log_overlap, dpi):
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.plot(ranks, complex_overlap, marker="o", label="complex image")
    ax.plot(ranks, log_overlap, marker="o", label="log-envelope")
    ax.set(
        xlabel="Subspace rank K",
        ylabel="Phase-amplitude subspace overlap O(K)",
        title="Shared spatial active modes: phase vs amplitude",
        ylim=(0.0, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


@torch.no_grad()
def analyze_one(sample_id, model, meta, train_idx, hold_idx, args):
    device = next(model.parameters()).device
    sample = torch.load(DATA_ROOT / "shards" / f"{sample_id}.pt",
                        map_location="cpu", weights_only=False)
    rf = sample["rf"][None].to(device)
    iq = demod_iq(rf, meta)
    D = rf_to_D(rf, meta)
    ds0 = fixed_phase_correction(model, iq, train_idx)["ds"]
    amp0 = torch.zeros_like(ds0)
    all_idx = torch.arange(D.shape[1], device=device)
    base = compound(model, D, ds0, amp0, all_idx)
    mask = model.reference(D, train_idx, hold_idx, top_frac=args.top_frac)["mask"][0]

    modes = lateral_dct_modes(model.born.nx, model.pad, args.lateral_modes,
                              device=device, dtype=ds0.dtype)
    depth_idx, depths_mm = select_depth_screen_indices(
        model.born.nz, float(model.born.z0), float(model.born.dz),
        args.depth_screens, args.min_depth_mm, args.max_depth_mm)

    ref_phase = args.phase_steps_us[0]
    ref_amp = args.amplitude_steps_np[0]
    ref = build_jacobian_for_step(
        model, D, ds0, amp0, mask, base, modes, depth_idx, depths_mm, all_idx,
        ref_phase, ref_amp, model.pad, args.log_eps)

    linearity = []
    for ps, aps in zip(args.phase_steps_us, args.amplitude_steps_np):
        if ps == ref_phase and aps == ref_amp:
            cur = ref
        else:
            cur = build_jacobian_for_step(
                model, D, ds0, amp0, mask, base, modes, depth_idx, depths_mm, all_idx,
                ps, aps, model.pad, args.log_eps)
        linearity.append({
            "phase_step_us": float(ps),
            "amplitude_step_np": float(aps),
            "phase": summarize_linearity(
                ref["J_phase"], cur["J_phase"], args.linearity_active_rel_threshold),
            "amplitude": summarize_linearity(
                ref["J_amplitude"], cur["J_amplitude"], args.linearity_active_rel_threshold),
        })

    Gp = ref["J_phase"].T @ ref["J_phase"]
    Ga = ref["J_amplitude"].T @ ref["J_amplitude"]
    Gpl = ref["J_phase_log"].T @ ref["J_phase_log"]
    Gal = ref["J_amplitude_log"].T @ ref["J_amplitude_log"]

    return {
        "sample": sample_id,
        "depths_mm": depths_mm,
        "depth_indices": depth_idx,
        "labels_phase": ref["labels_phase"],
        "labels_amplitude": ref["labels_amplitude"],
        "phase_global": ref["phase_global"],
        "amplitude_global": ref["amplitude_global"],
        "phase_conditional": ref["phase_conditional"],
        "amplitude_conditional": ref["amplitude_conditional"],
        "Gp": Gp, "Ga": Ga, "Gpl": Gpl, "Gal": Gal,
        "phase_rep": gram_spectrum(Gp),
        "amp_rep": gram_spectrum(Ga),
        "linearity": linearity,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sample-ids", nargs="+", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--imaging-n-freq", type=int, default=64)
    p.add_argument("--depth-screens", type=int, default=8)
    p.add_argument("--lateral-modes", type=int, default=6)
    p.add_argument("--min-depth-mm", type=float, default=2.0)
    p.add_argument("--max-depth-mm", type=float, default=38.0)
    p.add_argument("--phase-steps-us", nargs="+", type=float, default=[0.0025, 0.005, 0.01])
    p.add_argument("--amplitude-steps-np", nargs="+", type=float, default=[0.01, 0.02, 0.04])
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--log-eps", type=float, default=1e-8)
    p.add_argument("--subspace-rank", type=int, default=6)
    p.add_argument("--cross-family-max-rank", type=int, default=15)
    p.add_argument("--linearity-active-rel-threshold", type=float, default=1e-6)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if len(args.phase_steps_us) != len(args.amplitude_steps_np):
        p.error("phase and amplitude step lists must have equal length")
    if any(v <= 0 for v in args.phase_steps_us + args.amplitude_steps_np):
        p.error("all perturbation steps must be positive")
    if args.depth_screens < 1 or args.lateral_modes < 1:
        p.error("candidate mode counts must be positive")
    if args.cross_family_max_rank < 1:
        p.error("--cross-family-max-rank must be positive")
    if args.linearity_active_rel_threshold < 0:
        p.error("--linearity-active-rel-threshold must be non-negative")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    first = torch.load(DATA_ROOT / "shards" / f"{args.sample_ids[0]}.pt",
                       map_location="cpu", weights_only=False)
    _, _, cfg, meta, model = build_model(
        args.checkpoint, first, args.imaging_n_freq, device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    train_idx = torch.as_tensor(meta.train_idx, device=device)
    hold_idx = torch.as_tensor(meta.hold_idx, device=device)

    sample_results = []
    Gp = Ga = Gpl = Gal = None
    phase_global_all, amp_global_all = [], []
    phase_cond_all, amp_cond_all = [], []

    for sid in args.sample_ids:
        row = analyze_one(sid, model, meta, train_idx, hold_idx, args)
        sample_results.append(row)
        for name in ("Gp", "Ga", "Gpl", "Gal"):
            val = row[name]
            if name == "Gp": Gp = val.clone() if Gp is None else Gp + val
            elif name == "Ga": Ga = val.clone() if Ga is None else Ga + val
            elif name == "Gpl": Gpl = val.clone() if Gpl is None else Gpl + val
            else: Gal = val.clone() if Gal is None else Gal + val
        phase_global_all.append(row["phase_global"])
        amp_global_all.append(row["amplitude_global"])
        phase_cond_all.append(row["phase_conditional"])
        amp_cond_all.append(row["amplitude_conditional"])
        print(json.dumps({
            "event": "multisample_active_subspace_sample",
            "sample": sid,
            "phase_rank90": row["phase_rep"]["rank_90"],
            "amplitude_rank90": row["amp_rep"]["rank_90"],
            "linearity": row["linearity"],
        }), flush=True)
        torch.cuda.empty_cache()

    n = float(len(sample_results))
    Gp /= n; Ga /= n; Gpl /= n; Gal /= n
    rep_p = gram_spectrum(Gp)
    rep_a = gram_spectrum(Ga)
    rep_pl = gram_spectrum(Gpl)
    rep_al = gram_spectrum(Gal)

    overlap_ranks, phase_amp_overlap = subspace_overlap_curve(
        rep_p["Vh"], rep_a["Vh"], args.cross_family_max_rank)
    overlap_log_ranks, phase_amp_log_overlap = subspace_overlap_curve(
        rep_pl["Vh"], rep_al["Vh"], args.cross_family_max_rank)
    if overlap_ranks != overlap_log_ranks:
        raise RuntimeError("complex/log overlap rank grids do not match")

    pg = np.mean(np.stack(phase_global_all), axis=0)
    ag = np.mean(np.stack(amp_global_all), axis=0)
    pc = np.mean(np.stack(phase_cond_all), axis=0)
    ac = np.mean(np.stack(amp_cond_all), axis=0)
    depths_mm = sample_results[0]["depths_mm"]

    # Stability: compare each sample active subspace against population space.
    phase_overlap = []
    amp_overlap = []
    k = args.subspace_rank
    for row in sample_results:
        phase_overlap.append(float(subspace_overlap(
            row["phase_rep"]["Vh"], rep_p["Vh"], k)))
        amp_overlap.append(float(subspace_overlap(
            row["amp_rep"]["Vh"], rep_a["Vh"], k)))

    # Average linearity diagnostics across samples at each step.
    linearity_mean = []
    for si in range(len(args.phase_steps_us)):
        phase_keys = ["active_columns", "total_columns", "active_fraction",
                      "mean_cosine", "min_cosine", "mean_norm_ratio",
                      "std_norm_ratio", "mean_relative_error", "max_relative_error"]
        amp_keys = phase_keys
        pmean = {key: float(np.mean([r["linearity"][si]["phase"][key]
                                     for r in sample_results])) for key in phase_keys}
        amean = {key: float(np.mean([r["linearity"][si]["amplitude"][key]
                                     for r in sample_results])) for key in amp_keys}
        linearity_mean.append({
            "phase_step_us": float(args.phase_steps_us[si]),
            "amplitude_step_np": float(args.amplitude_steps_np[si]),
            "phase": pmean,
            "amplitude": amean,
        })

    spectra_name = "population_propagation_active_subspace.png"
    depth_name = "population_depth_coverage_bias.png"
    linearity_name = "population_finite_difference_linearity.png"
    overlap_name = "population_phase_amplitude_subspace_overlap.png"
    plot_population_spectra(args.out / spectra_name, rep_p, rep_a, rep_pl, rep_al, args.dpi)
    plot_depth_bias(args.out / depth_name, depths_mm, pg, pc, ag, ac, args.dpi)
    plot_linearity(args.out / linearity_name, linearity_mean, args.dpi)
    plot_cross_family_overlap(
        args.out / overlap_name, overlap_ranks,
        phase_amp_overlap.detach().cpu().numpy(),
        phase_amp_log_overlap.detach().cpu().numpy(), args.dpi)

    torch.save({
        "Gp": Gp.cpu(), "Ga": Ga.cpu(), "Gpl": Gpl.cpu(), "Gal": Gal.cpu(),
        "phase_Vh": rep_p["Vh"].cpu(), "amplitude_Vh": rep_a["Vh"].cpu(),
        "phase_log_Vh": rep_pl["Vh"].cpu(), "amplitude_log_Vh": rep_al["Vh"].cpu(),
        "phase_amplitude_overlap_ranks": overlap_ranks,
        "phase_amplitude_overlap": phase_amp_overlap.cpu(),
        "phase_amplitude_log_overlap": phase_amp_log_overlap.cpu(),
        "labels_phase": sample_results[0]["labels_phase"],
        "labels_amplitude": sample_results[0]["labels_amplitude"],
        "depths_mm": depths_mm,
    }, args.out / "population_active_subspace.pt")

    payload = {
        "experiment": "cross-sample propagation active subspace",
        "checkpoint": str(args.checkpoint),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "population": {
            "phase": compact_spectrum(rep_p),
            "amplitude": compact_spectrum(rep_a),
            "phase_log_envelope": compact_spectrum(rep_pl),
            "amplitude_log_envelope": compact_spectrum(rep_al),
            "mean_phase_subspace_overlap": float(np.mean(phase_overlap)),
            "min_phase_subspace_overlap": float(np.min(phase_overlap)),
            "mean_amplitude_subspace_overlap": float(np.mean(amp_overlap)),
            "min_amplitude_subspace_overlap": float(np.min(amp_overlap)),
            "subspace_rank_for_overlap": int(k),
            "phase_amplitude_overlap": {
                "ranks": overlap_ranks,
                "complex": phase_amp_overlap.detach().cpu().tolist(),
                "log_envelope": phase_amp_log_overlap.detach().cpu().tolist(),
                "complex_at_rank6": float(phase_amp_overlap[min(5, len(phase_amp_overlap)-1)]),
                "log_envelope_at_rank6": float(
                    phase_amp_log_overlap[min(5, len(phase_amp_log_overlap)-1)]),
            },
        },
        "linearity_mean": linearity_mean,
        "depth_bias": {
            "depths_mm": depths_mm,
            "phase_global_rms": np.sqrt(np.mean(pg**2, axis=1)).tolist(),
            "phase_conditional_rms": np.sqrt(np.mean(pc**2, axis=1)).tolist(),
            "amplitude_global_rms": np.sqrt(np.mean(ag**2, axis=1)).tolist(),
            "amplitude_conditional_rms": np.sqrt(np.mean(ac**2, axis=1)).tolist(),
        },
        "per_sample": [
            {
                "sample": r["sample"],
                "phase": compact_spectrum(r["phase_rep"]),
                "amplitude": compact_spectrum(r["amp_rep"]),
                "phase_overlap_with_population": phase_overlap[i],
                "amplitude_overlap_with_population": amp_overlap[i],
                "linearity": r["linearity"],
            }
            for i, r in enumerate(sample_results)
        ],
        "figures": {
            "spectra": spectra_name,
            "depth_bias": depth_name,
            "linearity": linearity_name,
            "phase_amplitude_overlap": overlap_name,
        },
    }
    (args.out / "multisample_active_subspace_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "event": "done",
        "population": payload["population"],
        "out": str(args.out),
    }), flush=True)


if __name__ == "__main__":
    main()

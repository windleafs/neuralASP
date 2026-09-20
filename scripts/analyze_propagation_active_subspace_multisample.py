"""Multi-sample validation of task-oriented propagation active subspaces.

This follow-up addresses three limitations of the single-sample sensitivity
experiment:

1. finite-difference linearity: repeat the Jacobian estimate at several small
   phase/amplitude perturbation sizes;
2. depth-coverage bias: report both whole-image sensitivity and a conditional
   sensitivity normalized only over pixels at/below each screen;
3. cross-sample stability: aggregate per-sample Gram matrices
       G = (1/N) sum_i J_i^T J_i
   and eigendecompose G to obtain propagation modes that are stable across
   different tissue realizations.

Phase and amplitude families are intentionally kept separate here. Their
physical units differ, so a combined cross-family active space should only be
formed after choosing a data-driven distortion scale for each family.
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
)
from physics.propagation_modes import (
    amplitude_screen_perturbation,
    lateral_dct_modes,
    phase_screen_perturbation,
    select_depth_screen_indices,
)
from scripts.evaluate_phase_amplitude_screen import (
    DATA_ROOT,
    build_model,
    physical_crop,
)
from scripts.oracle_amplitude_screen import fixed_phase_correction


def compound(model, D, ds, amp_rate, all_idx):
    images = model.angle_images(
        ds, D, all_idx, amplitude_rate=amp_rate)
    return images.mean(dim=1)[0]


def reference_mask(model, D, train_idx, hold_idx, top_frac):
    return model.reference(
        D, train_idx, hold_idx, top_frac=top_frac)["mask"][0]


def complex_feature(delta, mask, pad, baseline_norm):
    """Fixed-length real feature vector with masked pixels zeroed."""
    d = physical_crop(delta, pad)
    m = physical_crop(mask, pad).to(d.real.dtype)
    z = d * m
    denom = baseline_norm.clamp_min(1e-12)
    z = z / denom
    return torch.cat([z.real.flatten(), z.imag.flatten()], dim=0)


def logenv_feature(plus, minus, step, mask, pad, eps):
    p = physical_crop(plus, pad).abs().clamp_min(eps)
    n = physical_crop(minus, pad).abs().clamp_min(eps)
    m = physical_crop(mask, pad).to(p.dtype)
    response = (torch.log(p) - torch.log(n)) / (2.0 * float(step))
    return (response * m).flatten()


def masked_baseline_norm(base, mask, pad):
    b = physical_crop(base, pad)
    m = physical_crop(mask, pad).to(b.real.dtype)
    return (b.abs().square() * m).sum().sqrt().clamp_min(1e-12)


def nearest_step_index(steps, reference):
    arr = np.asarray(steps, dtype=np.float64)
    return int(np.argmin(np.abs(arr - float(reference))))


def compact_spectrum(rep, n=12):
    return {
        "rank_90": int(rep["rank_90"]),
        "rank_95": int(rep["rank_95"]),
        "rank_99": int(rep["rank_99"]),
        "leading_singular_values": rep["singular_values"][:n].cpu().tolist(),
        "leading_energy_fraction": rep["energy_fraction"][:n].cpu().tolist(),
        "leading_cumulative_energy": rep["cumulative_energy"][:n].cpu().tolist(),
    }


def summarize_linearity(diag):
    cos = diag["cosine"].detach().cpu().numpy()
    ratio = diag["norm_ratio"].detach().cpu().numpy()
    err = diag["relative_error"].detach().cpu().numpy()
    return {
        "median_cosine": float(np.median(cos)),
        "min_cosine": float(np.min(cos)),
        "median_norm_ratio": float(np.median(ratio)),
        "p95_relative_error": float(np.percentile(err, 95)),
        "max_relative_error": float(np.max(err)),
    }


def family_responses(model, D, ds0, amp0, base, mask, modes,
                     depth_idx, depths_mm, all_idx, family, steps,
                     reference_step, args):
    global_norm = masked_baseline_norm(base, mask, model.pad)
    labels = [
        f"{family[0].upper()}:z{zmm:.1f}:k{k}"
        for zmm in depths_mm
        for k in range(args.lateral_modes)
    ]
    ref_index = nearest_step_index(steps, reference_step)
    reference_step_used = float(steps[ref_index])

    global_by_step = {}
    conditional_reference_cols = None
    logenv_reference_cols = None
    global_sensitivity = None
    conditional_sensitivity = None
    coverage_fraction = []

    # Coverage is a property of depth and fixed reference mask, not family.
    mask_count = float(physical_crop(mask, model.pad).sum().clamp_min(1.0))
    conditional_masks = []
    conditional_norms = []
    for zi in depth_idx:
        cmask = below_screen_mask(mask, zi)
        conditional_masks.append(cmask)
        conditional_norms.append(
            masked_baseline_norm(base, cmask, model.pad))
        coverage_fraction.append(
            float(physical_crop(cmask, model.pad).sum() / mask_count))

    for si, step in enumerate(steps):
        global_cols = []
        cond_cols = [] if si == ref_index else None
        log_cols = [] if si == ref_index else None

        for di, zi in enumerate(depth_idx):
            for k in range(args.lateral_modes):
                mode = modes[k]
                if family == "phase":
                    perturb = phase_screen_perturbation(
                        mode, model.born.nz, zi, model.born.dz, float(step))
                    plus = compound(
                        model, D, ds0 + perturb[None], amp0, all_idx)
                    minus = compound(
                        model, D, ds0 - perturb[None], amp0, all_idx)
                elif family == "amplitude":
                    perturb = amplitude_screen_perturbation(
                        mode, model.born.nz, zi, model.born.dz, float(step))
                    plus = compound(
                        model, D, ds0, amp0 + perturb[None], all_idx)
                    minus = compound(
                        model, D, ds0, amp0 - perturb[None], all_idx)
                else:
                    raise ValueError(family)

                derivative = (plus - minus) / (2.0 * float(step))
                gvec = complex_feature(
                    derivative, mask, model.pad, global_norm)
                global_cols.append(gvec)

                if si == ref_index:
                    cvec = complex_feature(
                        derivative, conditional_masks[di], model.pad,
                        conditional_norms[di])
                    cond_cols.append(cvec)
                    log_cols.append(logenv_feature(
                        plus, minus, step, mask, model.pad, args.log_eps))

        Jg = torch.stack(global_cols, dim=1)
        global_by_step[float(step)] = Jg
        if si == ref_index:
            conditional_reference_cols = torch.stack(cond_cols, dim=1)
            logenv_reference_cols = torch.stack(log_cols, dim=1)
            global_sensitivity = Jg.norm(dim=0).reshape(
                len(depth_idx), args.lateral_modes)
            conditional_sensitivity = conditional_reference_cols.norm(dim=0).reshape(
                len(depth_idx), args.lateral_modes)

    Jref = global_by_step[reference_step_used]
    linearity = {}
    for step, J in global_by_step.items():
        diag = linearity_diagnostics(Jref, J)
        linearity[str(step)] = summarize_linearity(diag)

    return {
        "labels": labels,
        "reference_step": reference_step_used,
        "J_global": Jref,
        "J_conditional": conditional_reference_cols,
        "J_logenv": logenv_reference_cols,
        "global_sensitivity": global_sensitivity,
        "conditional_sensitivity": conditional_sensitivity,
        "coverage_fraction": np.asarray(coverage_fraction),
        "linearity": linearity,
    }


@torch.no_grad()
def analyze_sample(sample_id, model, meta, train_idx, hold_idx, args):
    device = next(model.parameters()).device
    sample = torch.load(
        DATA_ROOT / "shards" / f"{sample_id}.pt",
        map_location="cpu", weights_only=False)
    rf = sample["rf"][None].to(device)
    iq = demod_iq(rf, meta)
    D = rf_to_D(rf, meta)

    ds0 = fixed_phase_correction(model, iq, train_idx)["ds"]
    amp0 = torch.zeros_like(ds0)
    all_idx = torch.arange(D.shape[1], device=device)
    base = compound(model, D, ds0, amp0, all_idx)
    mask = reference_mask(model, D, train_idx, hold_idx, args.top_frac)

    modes = lateral_dct_modes(
        model.born.nx, model.pad, args.lateral_modes,
        device=device, dtype=ds0.dtype)
    depth_idx, depths_mm = select_depth_screen_indices(
        model.born.nz, float(model.born.z0), float(model.born.dz),
        args.depth_screens, args.min_depth_mm, args.max_depth_mm)

    phase = family_responses(
        model, D, ds0, amp0, base, mask, modes, depth_idx, depths_mm, all_idx,
        "phase", args.phase_steps_us, args.reference_phase_step_us, args)
    amplitude = family_responses(
        model, D, ds0, amp0, base, mask, modes, depth_idx, depths_mm, all_idx,
        "amplitude", args.amplitude_steps_np, args.reference_amplitude_step_np, args)

    return {
        "sample": sample_id,
        "depth_indices": depth_idx,
        "depths_mm": depths_mm,
        "phase": phase,
        "amplitude": amplitude,
    }


def gram(J):
    return J.T @ J


def plot_linearity(path, rows, args):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    metrics = [
        ("median_cosine", "Median cosine", (0.9, 1.005)),
        ("median_norm_ratio", "Median norm ratio", None),
        ("p95_relative_error", "95th % relative error", None),
    ]
    for ax, (key, ylabel, ylim) in zip(axes, metrics):
        for family, steps in (
            ("phase", args.phase_steps_us),
            ("amplitude", args.amplitude_steps_np),
        ):
            y = []
            for step in steps:
                vals = [r[family]["linearity"][str(float(step))][key] for r in rows]
                y.append(float(np.mean(vals)))
            ax.plot(steps, y, marker="o", label=family)
        ax.set(xlabel="Perturbation step", ylabel=ylabel, title=ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


def plot_depth_bias(path, mean_phase_global, mean_phase_cond,
                    mean_amp_global, mean_amp_cond, depths_mm, args):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    for ax, g, c, title in (
        (axes[0], mean_phase_global, mean_phase_cond, "Phase"),
        (axes[1], mean_amp_global, mean_amp_cond, "Amplitude"),
    ):
        gr = np.sqrt(np.mean(g ** 2, axis=1))
        cr = np.sqrt(np.mean(c ** 2, axis=1))
        grn = gr / max(gr.max(), 1e-30)
        crn = cr / max(cr.max(), 1e-30)
        ax.plot(depths_mm, grn, marker="o", label="global")
        ax.plot(depths_mm, crn, marker="o", label="below-screen conditional")
        ax.set(
            xlabel="Screen depth [mm]",
            ylabel="Within-family normalized RMS sensitivity",
            title=f"{title}: depth coverage bias check")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


def plot_mean_heatmaps(path, pg, pc, ag, ac, depths_mm, args):
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), constrained_layout=True)
    panels = [
        (axes[0, 0], pg, "Phase global"),
        (axes[0, 1], pc, "Phase conditional"),
        (axes[1, 0], ag, "Amplitude global"),
        (axes[1, 1], ac, "Amplitude conditional"),
    ]
    for ax, arr, title in panels:
        im = ax.imshow(
            arr, aspect="auto", origin="lower",
            extent=[-0.5, arr.shape[1]-0.5, depths_mm[0], depths_mm[-1]])
        ax.set(
            xlabel="Lateral DCT mode index",
            ylabel="Screen depth [mm]", title=title)
        fig.colorbar(im, ax=ax, shrink=0.78)
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


def plot_aggregate_spectra(path, spectra, args):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), constrained_layout=True)
    for ax, family in zip(axes, ("phase", "amplitude")):
        for kind in ("global", "conditional", "logenv"):
            rep = spectra[family][kind]
            c = rep["cumulative_energy"].cpu().numpy()
            ax.plot(np.arange(1, len(c)+1), c, marker="o", markersize=3, label=kind)
        ax.set(
            xlabel="Mode rank", ylabel="Cumulative response energy",
            title=f"{family.capitalize()} cross-sample active subspace",
            ylim=(0, 1.02))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


def plot_active_modes(path, spectrum, depths_mm, lateral_modes, title, args):
    Vh = spectrum["Vh"].cpu().numpy()
    n = min(args.top_active_modes, Vh.shape[0])
    fig, axes = plt.subplots(
        1, n, figsize=(3.5*n, 4.2), constrained_layout=True, squeeze=False)
    for i in range(n):
        arr = Vh[i].reshape(len(depths_mm), lateral_modes)
        ax = axes[0, i]
        vmax = max(np.abs(arr).max(), 1e-12)
        im = ax.imshow(
            arr, aspect="auto", origin="lower",
            extent=[-0.5, lateral_modes-0.5, depths_mm[0], depths_mm[-1]],
            vmin=-vmax, vmax=vmax, cmap="coolwarm")
        ax.set(
            xlabel="DCT k", ylabel="Depth [mm]",
            title=f"mode {i+1}")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(title)
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


def aggregate_linearity(rows, family, steps):
    out = {}
    for step in steps:
        key = str(float(step))
        vals = [r[family]["linearity"][key] for r in rows]
        out[key] = {
            metric: float(np.mean([v[metric] for v in vals]))
            for metric in vals[0]
        }
    return out


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
    p.add_argument("--phase-steps-us", nargs="+", type=float, default=[0.01, 0.02, 0.04])
    p.add_argument("--amplitude-steps-np", nargs="+", type=float, default=[0.01, 0.02, 0.04])
    p.add_argument("--reference-phase-step-us", type=float, default=0.02)
    p.add_argument("--reference-amplitude-step-np", type=float, default=0.02)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--log-eps", type=float, default=1e-8)
    p.add_argument("--top-active-modes", type=int, default=5)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.depth_screens < 1 or args.lateral_modes < 1:
        p.error("mode counts must be positive")
    if any(x <= 0 for x in args.phase_steps_us + args.amplitude_steps_np):
        p.error("finite-difference steps must be positive")
    if not (0 < args.top_frac <= 1):
        p.error("--top-frac must lie in (0,1]")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    first = torch.load(
        DATA_ROOT / "shards" / f"{args.sample_ids[0]}.pt",
        map_location="cpu", weights_only=False)
    _, _, cfg, meta, model = build_model(
        args.checkpoint, first, args.imaging_n_freq, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    train_idx = torch.as_tensor(meta.train_idx, device=device)
    hold_idx = torch.as_tensor(meta.hold_idx, device=device)

    print(json.dumps({
        "event": "multisample_active_subspace_setup",
        "sample_ids": args.sample_ids,
        "n_freq": len(meta.freqs),
        "phase_steps_us": args.phase_steps_us,
        "amplitude_steps_np": args.amplitude_steps_np,
        "depth_screens": args.depth_screens,
        "lateral_modes": args.lateral_modes,
    }), flush=True)

    internals = []
    rows = []
    for sid in args.sample_ids:
        result = analyze_sample(
            sid, model, meta, train_idx, hold_idx, args)
        internals.append(result)

        phase_global_rep = gram_spectrum(gram(result["phase"]["J_global"]))
        amp_global_rep = gram_spectrum(gram(result["amplitude"]["J_global"]))
        row = {
            "sample": sid,
            "depths_mm": result["depths_mm"],
            "phase": {
                "reference_step_us": result["phase"]["reference_step"],
                "global_spectrum": compact_spectrum(phase_global_rep),
                "linearity": result["phase"]["linearity"],
            },
            "amplitude": {
                "reference_step_np": result["amplitude"]["reference_step"],
                "global_spectrum": compact_spectrum(amp_global_rep),
                "linearity": result["amplitude"]["linearity"],
            },
        }
        rows.append(row)
        print(json.dumps({
            "event": "multisample_active_subspace_sample",
            "sample": sid,
            "phase_rank90": row["phase"]["global_spectrum"]["rank_90"],
            "amplitude_rank90": row["amplitude"]["global_spectrum"]["rank_90"],
            "phase_linearity": row["phase"]["linearity"],
            "amplitude_linearity": row["amplitude"]["linearity"],
        }), flush=True)
        torch.cuda.empty_cache()

    n = len(internals)
    def avg_gram(family, key):
        G = None
        for r in internals:
            Ji = r[family][key]
            Gi = gram(Ji).detach().cpu()
            G = Gi if G is None else G + Gi
        return G / float(n)

    spectra = {
        "phase": {
            "global": gram_spectrum(avg_gram("phase", "J_global")),
            "conditional": gram_spectrum(avg_gram("phase", "J_conditional")),
            "logenv": gram_spectrum(avg_gram("phase", "J_logenv")),
        },
        "amplitude": {
            "global": gram_spectrum(avg_gram("amplitude", "J_global")),
            "conditional": gram_spectrum(avg_gram("amplitude", "J_conditional")),
            "logenv": gram_spectrum(avg_gram("amplitude", "J_logenv")),
        },
    }

    # Cross-sample stability against the aggregate subspace.
    for row, internal in zip(rows, internals):
        for family in ("phase", "amplitude"):
            sample_rep = gram_spectrum(gram(internal[family]["J_global"]).cpu())
            agg_rep = spectra[family]["global"]
            k = min(sample_rep["rank_90"], agg_rep["rank_90"])
            row[family]["aggregate_subspace_overlap_at_k"] = float(
                subspace_overlap(sample_rep["Vh"], agg_rep["Vh"], k))
            row[family]["overlap_k"] = int(k)

    depths_mm = np.asarray(internals[0]["depths_mm"])
    pg = np.mean([r["phase"]["global_sensitivity"].cpu().numpy() for r in internals], axis=0)
    pc = np.mean([r["phase"]["conditional_sensitivity"].cpu().numpy() for r in internals], axis=0)
    ag = np.mean([r["amplitude"]["global_sensitivity"].cpu().numpy() for r in internals], axis=0)
    ac = np.mean([r["amplitude"]["conditional_sensitivity"].cpu().numpy() for r in internals], axis=0)
    coverage = np.mean([r["phase"]["coverage_fraction"] for r in internals], axis=0)

    linearity_name = "multisample_finite_difference_linearity.png"
    depth_name = "multisample_depth_coverage_bias.png"
    heatmap_name = "multisample_mean_sensitivity.png"
    spectra_name = "multisample_active_subspace_spectra.png"
    phase_modes_name = "multisample_phase_active_modes.png"
    amp_modes_name = "multisample_amplitude_active_modes.png"
    plot_linearity(args.out / linearity_name, internals, args)
    plot_depth_bias(args.out / depth_name, pg, pc, ag, ac, depths_mm, args)
    plot_mean_heatmaps(args.out / heatmap_name, pg, pc, ag, ac, depths_mm, args)
    plot_aggregate_spectra(args.out / spectra_name, spectra, args)
    plot_active_modes(
        args.out / phase_modes_name, spectra["phase"]["global"],
        depths_mm, args.lateral_modes, "Cross-sample phase active modes", args)
    plot_active_modes(
        args.out / amp_modes_name, spectra["amplitude"]["global"],
        depths_mm, args.lateral_modes, "Cross-sample amplitude active modes", args)

    aggregate = {
        "n": n,
        "phase": {
            "global": compact_spectrum(spectra["phase"]["global"]),
            "conditional": compact_spectrum(spectra["phase"]["conditional"]),
            "logenv": compact_spectrum(spectra["phase"]["logenv"]),
            "linearity": aggregate_linearity(
                internals, "phase", args.phase_steps_us),
            "mean_subspace_overlap": float(np.mean([
                r["phase"]["aggregate_subspace_overlap_at_k"] for r in rows])),
        },
        "amplitude": {
            "global": compact_spectrum(spectra["amplitude"]["global"]),
            "conditional": compact_spectrum(spectra["amplitude"]["conditional"]),
            "logenv": compact_spectrum(spectra["amplitude"]["logenv"]),
            "linearity": aggregate_linearity(
                internals, "amplitude", args.amplitude_steps_np),
            "mean_subspace_overlap": float(np.mean([
                r["amplitude"]["aggregate_subspace_overlap_at_k"] for r in rows])),
        },
        "depth_coverage_fraction": coverage.tolist(),
        "mean_phase_global_sensitivity": pg.tolist(),
        "mean_phase_conditional_sensitivity": pc.tolist(),
        "mean_amplitude_global_sensitivity": ag.tolist(),
        "mean_amplitude_conditional_sensitivity": ac.tolist(),
    }

    artifact = {
        "depths_mm": torch.tensor(depths_mm),
        "coverage_fraction": torch.tensor(coverage),
        "phase_global_gram": avg_gram("phase", "J_global"),
        "phase_conditional_gram": avg_gram("phase", "J_conditional"),
        "phase_logenv_gram": avg_gram("phase", "J_logenv"),
        "amplitude_global_gram": avg_gram("amplitude", "J_global"),
        "amplitude_conditional_gram": avg_gram("amplitude", "J_conditional"),
        "amplitude_logenv_gram": avg_gram("amplitude", "J_logenv"),
        "phase_global_Vh": spectra["phase"]["global"]["Vh"],
        "phase_conditional_Vh": spectra["phase"]["conditional"]["Vh"],
        "amplitude_global_Vh": spectra["amplitude"]["global"]["Vh"],
        "amplitude_conditional_Vh": spectra["amplitude"]["conditional"]["Vh"],
    }
    torch.save(artifact, args.out / "multisample_active_subspace.pt")

    payload = {
        "experiment": "multi-sample propagation active-subspace validation",
        "checkpoint": str(args.checkpoint),
        "note": (
            "Phase and amplitude remain separate because their units differ; "
            "no cross-family ranking is implied by these spectra."),
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "aggregate": aggregate,
        "rows": rows,
        "figures": {
            "linearity": linearity_name,
            "depth_coverage_bias": depth_name,
            "mean_sensitivity": heatmap_name,
            "spectra": spectra_name,
            "phase_active_modes": phase_modes_name,
            "amplitude_active_modes": amp_modes_name,
        },
        "artifact": "multisample_active_subspace.pt",
    }
    (args.out / "multisample_active_subspace_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n")

    print(json.dumps({
        "event": "done",
        "phase_global_rank90": aggregate["phase"]["global"]["rank_90"],
        "phase_conditional_rank90": aggregate["phase"]["conditional"]["rank_90"],
        "amplitude_global_rank90": aggregate["amplitude"]["global"]["rank_90"],
        "amplitude_conditional_rank90": aggregate["amplitude"]["conditional"]["rank_90"],
        "phase_mean_subspace_overlap": aggregate["phase"]["mean_subspace_overlap"],
        "amplitude_mean_subspace_overlap": aggregate["amplitude"]["mean_subspace_overlap"],
        "out": str(args.out),
    }), flush=True)


if __name__ == "__main__":
    main()

"""Task-oriented sensitivity decomposition of propagation-screen parameters.

The experiment asks a narrower question than medium reconstruction:

    Which low-dimensional propagation perturbations most strongly change the
    final compounded image around the current phase-corrected operating point?

Candidate parameters are smooth DCT-like lateral modes placed at candidate
depth slabs. Phase and amplitude perturbations are probed with centered finite
differences. The resulting image-response columns form a sensitivity matrix J.

We report:
  * per-depth and per-lateral-mode sensitivities;
  * phase-only and amplitude-only singular spectra;
  * a combined spectrum after scaling both parameter families by canonical
    small perturbations;
  * leading right-singular vectors, i.e. active propagation-mode mixtures.

This is an analysis tool, not a network-training script.
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


def _compound(model, D, ds, amp_rate, all_idx):
    images = model.angle_images(
        ds, D, all_idx, amplitude_rate=amp_rate)
    return images.mean(dim=1)[0]


def _mask_from_reference(model, D, train_idx, hold_idx, top_frac):
    ref = model.reference(D, train_idx, hold_idx, top_frac=top_frac)
    return ref["mask"][0]


def _response_vector(delta_image, mask, pad, response_space, eps=1e-8):
    """Map an image perturbation to a real vector for SVD."""
    delta = physical_crop(delta_image, pad)
    m = physical_crop(mask, pad).bool()
    if response_space == "complex":
        z = delta[m]
        return torch.cat([z.real, z.imag], dim=0)
    if response_space == "envelope":
        return delta.abs()[m]
    raise ValueError(response_space)


def _log_envelope_response(plus, minus, mask, pad, eps):
    """Centered response of log envelope, useful as a B-mode-oriented view."""
    p = physical_crop(plus, pad).abs().clamp_min(eps)
    n = physical_crop(minus, pad).abs().clamp_min(eps)
    m = physical_crop(mask, pad).bool()
    return (0.5 * (torch.log(p) - torch.log(n)))[m]


def _svd_report(J):
    if J.numel() == 0:
        raise ValueError("empty sensitivity matrix")
    # J is [pixels_or_features, candidate_modes].
    U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    e = S.square()
    frac = e / e.sum().clamp_min(1e-30)
    cumulative = torch.cumsum(frac, dim=0)

    def rank_at(threshold):
        idx = torch.nonzero(cumulative >= threshold)
        return int(idx[0, 0] + 1) if idx.numel() else int(len(S))

    return {
        "singular_values": S.detach().cpu().numpy(),
        "energy_fraction": frac.detach().cpu().numpy(),
        "cumulative_energy": cumulative.detach().cpu().numpy(),
        "rank_90": rank_at(0.90),
        "rank_95": rank_at(0.95),
        "rank_99": rank_at(0.99),
        "Vh": Vh.detach().cpu().numpy(),
    }


def _plot_sensitivity_heatmaps(path, phase_sens, amp_sens, depths_mm, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    for ax, arr, title in (
        (axes[0], phase_sens, "Phase sensitivity"),
        (axes[1], amp_sens, "Amplitude sensitivity"),
    ):
        im = ax.imshow(
            arr, aspect="auto", origin="lower",
            extent=[-0.5, arr.shape[1]-0.5, depths_mm[0], depths_mm[-1]])
        ax.set(
            xlabel="Lateral DCT mode index",
            ylabel="Screen depth [mm]",
            title=title)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Normalized image response")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_depth_summary(path, phase_sens, amp_sens, depths_mm, dpi):
    phase_rms = np.sqrt(np.mean(phase_sens ** 2, axis=1))
    amp_rms = np.sqrt(np.mean(amp_sens ** 2, axis=1))
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.plot(depths_mm, phase_rms, marker="o", label="phase")
    ax.plot(depths_mm, amp_rms, marker="o", label="amplitude")
    ax.set(
        xlabel="Screen depth [mm]",
        ylabel="RMS normalized response",
        title="Depth sensitivity of candidate propagation screens")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_singular_spectra(path, phase, amp, combined, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    for label, rep in (("phase", phase), ("amplitude", amp), ("combined", combined)):
        s = rep["singular_values"]
        axes[0].semilogy(np.arange(1, len(s)+1), s / max(s[0], 1e-30),
                        marker="o", markersize=3, label=label)
        c = rep["cumulative_energy"]
        axes[1].plot(np.arange(1, len(c)+1), c, marker="o", markersize=3, label=label)
    axes[0].set(
        xlabel="Mode rank", ylabel="Normalized singular value",
        title="Sensitivity singular spectrum")
    axes[1].set(
        xlabel="Mode rank", ylabel="Cumulative response energy",
        title="Active-subspace cumulative energy", ylim=(0, 1.02))
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend()
    axes[1].legend()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_top_combined_modes(path, Vh, labels, n_show, dpi):
    n_show = min(n_show, Vh.shape[0])
    n_params = Vh.shape[1]
    fig, axes = plt.subplots(
        n_show, 1, figsize=(max(10.0, n_params * 0.18), 2.4 * n_show),
        constrained_layout=True, squeeze=False)
    x = np.arange(n_params)
    for i in range(n_show):
        ax = axes[i, 0]
        ax.bar(x, Vh[i])
        ax.set_ylabel(f"v{i+1}")
        ax.grid(axis="y", alpha=0.2)
        if i == n_show - 1:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
        else:
            ax.set_xticks([])
    axes[0, 0].set_title("Leading active propagation-mode mixtures")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


@torch.no_grad()
def analyze_sample(sample_id, model, meta, cfg, train_idx, hold_idx, args, out):
    device = next(model.parameters()).device
    sample = torch.load(
        DATA_ROOT / "shards" / f"{sample_id}.pt",
        map_location="cpu", weights_only=False)
    rf = sample["rf"][None].to(device)
    iq = demod_iq(rf, meta)
    D = rf_to_D(rf, meta)

    phase = fixed_phase_correction(model, iq, train_idx)
    ds0 = phase["ds"]
    amp0 = torch.zeros_like(ds0)
    all_idx = torch.arange(D.shape[1], device=device)
    base = _compound(model, D, ds0, amp0, all_idx)
    mask = _mask_from_reference(model, D, train_idx, hold_idx, args.top_frac)

    base_crop = physical_crop(base, model.pad)
    mask_crop = physical_crop(mask, model.pad).bool()
    base_norm = base_crop[mask_crop].abs().square().sum().sqrt().clamp_min(1e-12)

    modes = lateral_dct_modes(
        model.born.nx, model.pad, args.lateral_modes,
        device=device, dtype=ds0.dtype)
    depth_idx, depths_mm = select_depth_screen_indices(
        model.born.nz, float(model.born.z0), float(model.born.dz),
        args.depth_screens, args.min_depth_mm, args.max_depth_mm)

    phase_cols = []
    amp_cols = []
    phase_log_cols = []
    amp_log_cols = []
    phase_sens = np.zeros((len(depth_idx), args.lateral_modes), dtype=np.float64)
    amp_sens = np.zeros_like(phase_sens)
    labels_phase = []
    labels_amp = []

    for di, (zi, zmm) in enumerate(zip(depth_idx, depths_mm)):
        for k in range(args.lateral_modes):
            mode = modes[k]

            dphase = phase_screen_perturbation(
                mode, model.born.nz, zi, model.born.dz, args.phase_step_us)
            p_plus = _compound(model, D, ds0 + dphase[None], amp0, all_idx)
            p_minus = _compound(model, D, ds0 - dphase[None], amp0, all_idx)
            p_delta = 0.5 * (p_plus - p_minus)
            p_vec = _response_vector(
                p_delta, mask, model.pad, "complex") / base_norm
            p_log = _log_envelope_response(
                p_plus, p_minus, mask, model.pad, args.log_eps)
            phase_cols.append(p_vec)
            phase_log_cols.append(p_log)
            phase_sens[di, k] = float(p_vec.norm())
            labels_phase.append(f"P:z{zmm:.1f}:k{k}")

            damp = amplitude_screen_perturbation(
                mode, model.born.nz, zi, model.born.dz, args.amplitude_step_np)
            a_plus = _compound(model, D, ds0, amp0 + damp[None], all_idx)
            a_minus = _compound(model, D, ds0, amp0 - damp[None], all_idx)
            a_delta = 0.5 * (a_plus - a_minus)
            a_vec = _response_vector(
                a_delta, mask, model.pad, "complex") / base_norm
            a_log = _log_envelope_response(
                a_plus, a_minus, mask, model.pad, args.log_eps)
            amp_cols.append(a_vec)
            amp_log_cols.append(a_log)
            amp_sens[di, k] = float(a_vec.norm())
            labels_amp.append(f"A:z{zmm:.1f}:k{k}")

    Jp = torch.stack(phase_cols, dim=1)
    Ja = torch.stack(amp_cols, dim=1)
    Jp_log = torch.stack(phase_log_cols, dim=1)
    Ja_log = torch.stack(amp_log_cols, dim=1)

    # Columns already encode one canonical perturbation step. Concatenating
    # therefore compares the two families at physically chosen small scales.
    Jc = torch.cat([Jp, Ja], dim=1)
    Jc_log = torch.cat([Jp_log, Ja_log], dim=1)

    rep_phase = _svd_report(Jp)
    rep_amp = _svd_report(Ja)
    rep_combined = _svd_report(Jc)
    rep_combined_log = _svd_report(Jc_log)

    heatmap_name = f"{sample_id}_propagation_mode_sensitivity.png"
    depth_name = f"{sample_id}_propagation_depth_sensitivity.png"
    svd_name = f"{sample_id}_propagation_svd.png"
    active_name = f"{sample_id}_propagation_active_modes.png"
    _plot_sensitivity_heatmaps(
        out / heatmap_name, phase_sens, amp_sens, np.asarray(depths_mm), args.dpi)
    _plot_depth_summary(
        out / depth_name, phase_sens, amp_sens, np.asarray(depths_mm), args.dpi)
    _plot_singular_spectra(
        out / svd_name, rep_phase, rep_amp, rep_combined, args.dpi)
    _plot_top_combined_modes(
        out / active_name, rep_combined["Vh"], labels_phase + labels_amp,
        args.top_active_modes, args.dpi)

    artifact = {
        "phase_J": Jp.cpu(),
        "amplitude_J": Ja.cpu(),
        "combined_J": Jc.cpu(),
        "phase_logenv_J": Jp_log.cpu(),
        "amplitude_logenv_J": Ja_log.cpu(),
        "combined_logenv_J": Jc_log.cpu(),
        "phase_sensitivity": torch.from_numpy(phase_sens),
        "amplitude_sensitivity": torch.from_numpy(amp_sens),
        "depth_indices": depth_idx,
        "depths_mm": depths_mm,
        "labels_phase": labels_phase,
        "labels_amplitude": labels_amp,
        "phase_svd": {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                      for k, v in rep_phase.items()},
        "amplitude_svd": {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                          for k, v in rep_amp.items()},
        "combined_svd": {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                         for k, v in rep_combined.items()},
        "combined_logenv_svd": {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                                for k, v in rep_combined_log.items()},
    }
    torch.save(artifact, out / f"{sample_id}_propagation_active_modes.pt")

    def compact(rep):
        return {
            "rank_90": rep["rank_90"],
            "rank_95": rep["rank_95"],
            "rank_99": rep["rank_99"],
            "leading_singular_values": rep["singular_values"][:10].tolist(),
            "leading_energy_fraction": rep["energy_fraction"][:10].tolist(),
        }

    flat_phase = phase_sens.reshape(-1)
    flat_amp = amp_sens.reshape(-1)
    p_order = np.argsort(flat_phase)[::-1][:args.top_report_modes]
    a_order = np.argsort(flat_amp)[::-1][:args.top_report_modes]

    return {
        "sample": sample_id,
        "phase_step_us": args.phase_step_us,
        "amplitude_step_np": args.amplitude_step_np,
        "depths_mm": depths_mm,
        "lateral_modes": args.lateral_modes,
        "phase_svd": compact(rep_phase),
        "amplitude_svd": compact(rep_amp),
        "combined_svd": compact(rep_combined),
        "combined_logenv_svd": compact(rep_combined_log),
        "top_phase_candidates": [
            {"label": labels_phase[int(i)], "response": float(flat_phase[int(i)])}
            for i in p_order
        ],
        "top_amplitude_candidates": [
            {"label": labels_amp[int(i)], "response": float(flat_amp[int(i)])}
            for i in a_order
        ],
        "figures": {
            "sensitivity": heatmap_name,
            "depth": depth_name,
            "svd": svd_name,
            "active_modes": active_name,
        },
        "artifact": f"{sample_id}_propagation_active_modes.pt",
    }


def _aggregate(rows):
    def mean_rank(section, key):
        return float(np.mean([r[section][key] for r in rows]))
    return {
        "n": len(rows),
        "mean_phase_rank90": mean_rank("phase_svd", "rank_90"),
        "mean_phase_rank95": mean_rank("phase_svd", "rank_95"),
        "mean_amplitude_rank90": mean_rank("amplitude_svd", "rank_90"),
        "mean_amplitude_rank95": mean_rank("amplitude_svd", "rank_95"),
        "mean_combined_rank90": mean_rank("combined_svd", "rank_90"),
        "mean_combined_rank95": mean_rank("combined_svd", "rank_95"),
        "mean_combined_logenv_rank90": mean_rank("combined_logenv_svd", "rank_90"),
        "mean_combined_logenv_rank95": mean_rank("combined_logenv_svd", "rank_95"),
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
    p.add_argument("--phase-step-us", type=float, default=0.02)
    p.add_argument("--amplitude-step-np", type=float, default=0.02)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--log-eps", type=float, default=1e-8)
    p.add_argument("--top-active-modes", type=int, default=5)
    p.add_argument("--top-report-modes", type=int, default=8)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.depth_screens < 1 or args.lateral_modes < 1:
        p.error("depth and lateral mode counts must be positive")
    if args.phase_step_us <= 0 or args.amplitude_step_np <= 0:
        p.error("perturbation steps must be positive")
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
        "event": "propagation_active_mode_setup",
        "sample_ids": args.sample_ids,
        "n_freq": len(meta.freqs),
        "depth_screens": args.depth_screens,
        "lateral_modes": args.lateral_modes,
        "phase_step_us": args.phase_step_us,
        "amplitude_step_np": args.amplitude_step_np,
        "parameter_dx_mm": float(cfg.grid.dx * 1e3),
        "propagation_dx_mm": float(
            cfg.grid.dx * 1e3 / int(cfg.physics.get("lateral_oversample", 1))),
    }), flush=True)

    rows = []
    for sid in args.sample_ids:
        row = analyze_sample(
            sid, model, meta, cfg, train_idx, hold_idx, args, args.out)
        rows.append(row)
        print(json.dumps({
            "event": "propagation_active_mode_result",
            "sample": sid,
            "phase_rank90": row["phase_svd"]["rank_90"],
            "amplitude_rank90": row["amplitude_svd"]["rank_90"],
            "combined_rank90": row["combined_svd"]["rank_90"],
            "combined_logenv_rank90": row["combined_logenv_svd"]["rank_90"],
            "top_phase": row["top_phase_candidates"][:3],
            "top_amplitude": row["top_amplitude_candidates"][:3],
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {
        "experiment": "task-oriented propagation active-mode sensitivity",
        "checkpoint": str(args.checkpoint),
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "aggregate": _aggregate(rows),
        "rows": rows,
    }
    (args.out / "propagation_active_mode_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "event": "done",
        "aggregate": payload["aggregate"],
        "out": str(args.out),
    }), flush=True)


if __name__ == "__main__":
    main()

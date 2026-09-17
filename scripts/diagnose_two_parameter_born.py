"""Diagnose whether c/rho first-order scattering better explains UltraWave RF.

This script compares four first-order data models against UltraWave RF:

- proxy: existing scalar proxy reflectivity m;
- c_only: sound-speed/compressibility scattering;
- rho_only: density-gradient scattering;
- c_plus_rho: fixed physical sum of the two first-order terms.

It also reports a per-frequency two-term oracle

    D_UW ~= a(f) D_c + b(f) D_rho

which is diagnostic only. A large oracle gain with a smaller fixed-sum gain
suggests the mechanisms are useful but their relative discretization/Green
weighting still needs calibration. If even the two-term oracle remains weak,
model mismatch likely lies deeper (multiple scattering, attenuation, one-way
propagation, etc.).

The linearized source is derived from

    div((1/rho) grad p) + w^2/(rho c^2) p = 0

or equivalently

    (laplacian + k0^2) p_s = grad(log rho).grad(p0)
                               - w^2 (1/c^2 - 1/c0^2) p0.

After factoring a frequency-only -k0^2 term (absorbed by the fitted response),
we use the dimensionless equivalent source

    q_c   = ((c0/c)^2 - 1) p0
    q_rho = - grad(log rho).grad(p0) / k0^2.

Both terms are propagated through the same homogeneous one-way ASP receiver
operator used elsewhere in the repository. This is a first-order source-shape
diagnostic, not yet a fully calibrated two-way Green-function solver.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from common import D_to_rf, corr2d, demod_iq, rf_to_D  # noqa: E402
from physics.imaging import BornModel, DelaySumBaseline  # noqa: E402
from scripts.diagnose_born_ultrawave_mismatch import (  # noqa: E402
    as_complex, crop_x, data_fit_metrics, fixed_tgc, independent_db, roi_corr,
)
from scripts.generate_l11_kwave_raw import block_mean  # noqa: E402
from scripts.generate_l11_ultrawave_raw import geometry_case, medium_builder  # noqa: E402
from scripts.oracle_phase_screen_decomposition import sample_ids  # noqa: E402
from scripts.pilot_phase_asp import DATA_ROOT, corrected_config, embed, padded_meta  # noqa: E402


def central_diff(x: torch.Tensor, spacing: float, dim: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    sl = [slice(None)] * x.ndim
    a = sl.copy(); b = sl.copy(); c = sl.copy()
    a[dim] = slice(1, -1); b[dim] = slice(2, None); c[dim] = slice(None, -2)
    out[tuple(a)] = (x[tuple(b)] - x[tuple(c)]) / (2.0 * spacing)
    a[dim] = 0; b[dim] = 1; c[dim] = 0
    out[tuple(a)] = (x[tuple(b)] - x[tuple(c)]) / spacing
    a[dim] = -1; b[dim] = -1; c[dim] = -2
    out[tuple(a)] = (x[tuple(b)] - x[tuple(c)]) / spacing
    return out


def rebuild_c_rho(sample: dict, c0: float):
    md = sample["metadata"]
    if md.get("phantom_variant", "oa_breast_original") != "oa_breast_original":
        raise ValueError("this diagnostic currently supports oa_breast_original only")
    case = geometry_case()
    with h5py.File(md["h5"], "r") as f:
        plane = np.asarray(f["phan"][int(md["z_index"])])
    maps, _, _, *_ = medium_builder.build_medium(
        plane, case["x"], case["z"], seed=int(md["scatter_seed"]), preset="dual_scale")

    width = 192 * 4
    i0 = (len(case["x"]) - width) // 2
    face = medium_builder.FACE
    slz = slice(face, face + 216 * 4)
    slx = slice(i0, i0 + width)
    c = block_mean(maps["sound_speed"][slz, slx].astype(np.float64)).astype(np.float32)
    rho = block_mean(maps["density"][slz, slx].astype(np.float64)).astype(np.float32)
    stored = sample["c"].detach().cpu().numpy()
    max_c_err = float(np.max(np.abs(c - stored)))
    if max_c_err > 1e-3:
        raise RuntimeError(f"reconstructed c does not match shard: max abs err={max_c_err}")
    if not np.isfinite(rho).all() or np.min(rho) <= 0:
        raise RuntimeError("invalid reconstructed density")
    return c, rho, max_c_err


def source_to_data(born: BornModel, source: torch.Tensor,
                   delta_s: torch.Tensor) -> torch.Tensor:
    d = born.asp.march_up(source, delta_s, born.omega_)
    d = born.asp._ifft(born.asp._fft(d) * born.surface_transfer)
    return born.sample(d)


def build_two_parameter_data(born: BornModel, c: torch.Tensor, rho: torch.Tensor,
                             u0: torch.Tensor, ds_zero: torch.Tensor,
                             c0: float):
    # c,rho [B,nz,nx], u0 [B,theta,freq,nz,nx]
    chi_c = (c0 / c).square() - 1.0
    log_rho = torch.log(rho)
    gx_rho = central_diff(log_rho, born.dx, -1)
    gz_rho = central_diff(log_rho, born.dz, -2)

    du_dx = central_diff(u0, born.dx, -1)
    du_dz = central_diff(u0, born.dz, -2)
    k0 = (born.omega_ / c0).view(1, 1, -1, 1, 1)

    qc = chi_c[:, None, None] * u0
    qrho = -(gx_rho[:, None, None] * du_dx + gz_rho[:, None, None] * du_dz) \
           / k0.square().clamp_min(1e-12)

    # Reuse the same depth/frequency Green weighting as the scalar Born model.
    qc = qc * born.w_z
    qrho = qrho * born.w_z
    Dc = source_to_data(born, qc, ds_zero)
    Drho = source_to_data(born, qrho, ds_zero)
    return Dc, Drho


def grouped_coherence(pred: torch.Tensor, target: torch.Tensor, group_dim: int):
    # tensors [B,theta,freq,element]; preserve group_dim, reduce all others
    dims = tuple(i for i in range(pred.ndim) if i != group_dim)
    num = (target * pred.conj()).sum(dim=dims).abs()
    den = torch.sqrt(pred.abs().square().sum(dim=dims) *
                     target.abs().square().sum(dim=dims)).clamp_min(1e-30)
    return num / den


def fit_two_term_per_frequency(Dc: torch.Tensor, Dr: torch.Tensor, target: torch.Tensor):
    # Solve independent 2x2 complex LS at each frequency across B,theta,element.
    nf = Dc.shape[2]
    out = torch.zeros_like(target)
    coeff = torch.zeros(nf, 2, dtype=target.dtype, device=target.device)
    eye = torch.eye(2, dtype=target.dtype, device=target.device)
    for fi in range(nf):
        a = torch.stack([Dc[:, :, fi, :].reshape(-1),
                         Dr[:, :, fi, :].reshape(-1)], dim=1)
        y = target[:, :, fi, :].reshape(-1, 1)
        gram = a.conj().T @ a
        rhs = a.conj().T @ y
        lam = gram.diag().real.mean().clamp_min(1e-30) * 1e-8
        x = torch.linalg.solve(gram + lam * eye, rhs)[:, 0]
        coeff[fi] = x
        out[:, :, fi, :] = x[0] * Dc[:, :, fi, :] + x[1] * Dr[:, :, fi, :]
    return out, coeff


def das_from_D(D: torch.Tensor, meta, das_model: DelaySumBaseline):
    rf = D_to_rf(D, meta)
    iq = demod_iq(rf, meta)
    return das_model(iq)[0]


def fit_summary(pred: torch.Tensor, target: torch.Tensor):
    base, gain, fcoh = data_fit_metrics(pred, target)
    acoh = grouped_coherence(pred, target, 1)
    ecoh = grouped_coherence(pred, target, 3)
    base.update({
        "mean_angle_coherence": float(acoh.mean()),
        "median_angle_coherence": float(acoh.median()),
        "mean_element_coherence": float(ecoh.mean()),
        "median_element_coherence": float(ecoh.median()),
    })
    return base, fcoh.detach(), acoh.detach(), ecoh.detach()


def plot_coherences(path: Path, meta, results: dict, dpi: int):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    for name, item in results.items():
        axes[0].plot(np.asarray(meta.freqs) * 1e-6, item["fcoh"].cpu(), label=name)
        axes[1].plot(np.asarray(meta.angles_deg), item["acoh"].cpu(), label=name)
        axes[2].plot(np.arange(meta.n_elements), item["ecoh"].cpu(), label=name)
    axes[0].set(xlabel="Frequency [MHz]", ylabel="Coherence", title="Frequency coherence", ylim=(0, 1.02))
    axes[1].set(xlabel="Transmit angle [deg]", ylabel="Coherence", title="Angle coherence", ylim=(0, 1.02))
    axes[2].set(xlabel="Element index", ylabel="Coherence", title="Element coherence", ylim=(0, 1.02))
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_das(path: Path, sample_id: str, truth_abs: torch.Tensor, images: dict,
             metrics: dict, cfg, meta, args):
    z_m = float(meta.z0) + np.arange(cfg.grid.nz) * float(cfg.grid.dz)
    x_m = float(meta.x0) + np.arange(cfg.grid.nx) * float(cfg.grid.dx)
    extent = [x_m[0]*1e3, x_m[-1]*1e3, z_m[-1]*1e3, z_m[0]*1e3]
    names = ["truth", "ultrawave", "proxy", "c_only", "rho_only", "c_plus_rho"]
    titles = ["Truth |m|", "DAS(UltraWave)", "DAS(proxy)", "DAS(c only)",
              "DAS(rho only)", "DAS(c+rho)"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.5), constrained_layout=True)
    for ax, name, title in zip(axes.ravel(), names, titles):
        image = truth_abs if name == "truth" else images[name]
        if name != "truth":
            image = fixed_tgc(image, z_m, args.tgc_db_per_mm, args.tgc_max_db)
        db = independent_db(image, args.db_range)
        ax.imshow(db, cmap="gray", vmin=-args.db_range, vmax=0,
                  origin="upper", extent=extent, aspect="auto")
        sub = "independent scale" if name == "truth" else f"corr={metrics[name]:.3f}"
        ax.set_title(f"{title}\n{sub}")
        ax.set_xlabel("Lateral x [mm]"); ax.set_ylabel("Depth z [mm]")
    fig.suptitle(f"{sample_id}: two-parameter Born source diagnostic")
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


@torch.no_grad()
def evaluate_one(sid: str, args, device):
    sample = torch.load(DATA_ROOT / "shards" / f"{sid}.pt", map_location="cpu", weights_only=False)
    cfg, meta = corrected_config(args.config, sample, args.imaging_n_freq)
    if len(meta.band_idx) > 1 and not np.all(np.diff(meta.band_idx) == 1):
        raise RuntimeError("diagnostic requires contiguous full-band frequencies")

    c_np, rho_np, max_c_err = rebuild_c_rho(sample, cfg.physics.c0)
    c = torch.from_numpy(c_np).to(device)[None]
    rho = torch.from_numpy(rho_np).to(device)[None]
    c_pad = torch.nn.functional.pad(c, (args.pad, args.pad), value=float(cfg.physics.c0))
    rho0 = float(np.median(rho_np[:4]))
    rho_pad = torch.nn.functional.pad(rho, (args.pad, args.pad), value=rho0)

    pmeta = padded_meta(meta, args.pad, cfg.grid.dx)
    born = BornModel(pmeta, cfg.grid.nx + 2*args.pad, cfg.grid.nz,
                     cfg.grid.dx, cfg.grid.dz, cfg.physics.c0,
                     eps=cfg.physics.eps_evanescent,
                     spreading=cfg.physics.spreading).to(device)
    das_model = DelaySumBaseline(meta, cfg.grid.nx, cfg.grid.nz,
                                 cfg.grid.dx, cfg.grid.dz, cfg.physics.c0).to(device)

    rf = sample["rf"][None].to(device)
    Duw = rf_to_D(rf, meta)
    zero = torch.zeros_like(c_pad)
    all_idx = torch.arange(len(meta.angles_deg), device=device)
    u0 = born.transmit_fields(zero, all_idx)

    mproxy = as_complex(embed(sample["m"].to(device), args.pad))[None]
    Dproxy = born.forward(mproxy, zero, u0, all_idx)
    Dc, Dr = build_two_parameter_data(born, c_pad, rho_pad, u0, zero, cfg.physics.c0)
    Dsum = Dc + Dr
    Doracle, coeff = fit_two_term_per_frequency(Dc, Dr, Duw)

    models = {"proxy": Dproxy, "c_only": Dc, "rho_only": Dr,
              "c_plus_rho": Dsum, "two_term_oracle": Doracle}
    fit_results = {}
    curves = {}
    for name, pred in models.items():
        summary, fcoh, acoh, ecoh = fit_summary(pred, Duw)
        fit_results[name] = summary
        curves[name] = {"fcoh": fcoh, "acoh": acoh, "ecoh": ecoh}

    truth_abs = sample["m"].abs().to(device)
    z_m = float(meta.z0) + np.arange(cfg.grid.nz) * float(cfg.grid.dz)
    das_images = {"ultrawave": das_from_D(Duw, meta, das_model)}
    for name in ("proxy", "c_only", "rho_only", "c_plus_rho"):
        das_images[name] = das_from_D(models[name], meta, das_model)
    image_corr = {name: roi_corr(img, truth_abs, z_m, 3.0, 35.0)
                  for name, img in das_images.items()}

    plot_das(args.out / f"{sid}_two_parameter_das.png", sid, truth_abs,
             das_images, image_corr, cfg, meta, args)
    plot_coherences(args.out / f"{sid}_two_parameter_coherence.png", meta,
                    curves, args.dpi)

    coeff_np = coeff.cpu().numpy()
    row = {
        "sample": sid,
        "case": sample["metadata"].get("case"),
        "reconstruction_check": {"max_abs_c_error_m_per_s": max_c_err,
                                 "rho_padding_reference": rho0},
        "fit": fit_results,
        "das_corr_truth_3_35mm": image_corr,
        "two_term_oracle": {
            "a_c_real": coeff_np[:, 0].real.tolist(),
            "a_c_imag": coeff_np[:, 0].imag.tolist(),
            "b_rho_real": coeff_np[:, 1].real.tolist(),
            "b_rho_imag": coeff_np[:, 1].imag.tolist(),
        },
    }
    return row


def aggregate(rows):
    names = ["proxy", "c_only", "rho_only", "c_plus_rho", "two_term_oracle"]
    out = {}
    for name in names:
        out[name] = {
            "mean_per_frequency_relative_residual": float(np.mean([
                r["fit"][name]["per_frequency_relative_residual"] for r in rows])),
            "mean_frequency_coherence": float(np.mean([
                r["fit"][name]["mean_frequency_coherence"] for r in rows])),
            "mean_angle_coherence": float(np.mean([
                r["fit"][name]["mean_angle_coherence"] for r in rows])),
            "mean_element_coherence": float(np.mean([
                r["fit"][name]["mean_element_coherence"] for r in rows])),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--sample-ids", nargs="+")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--imaging-n-freq", type=int, default=0)
    p.add_argument("--db-range", type=float, default=55.0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--tgc-db-per-mm", type=float, default=0.42)
    p.add_argument("--tgc-max-db", type=float, default=18.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ids = args.sample_ids if args.sample_ids else sample_ids(args.split, args.count)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    rows = []
    for i, sid in enumerate(ids, 1):
        row = evaluate_one(sid, args, device)
        rows.append(row)
        print(json.dumps({
            "event": "diagnosed", "sample": sid, "completed": i, "total": len(ids),
            "fit": {k: {"residual": v["per_frequency_relative_residual"],
                        "f_coh": v["mean_frequency_coherence"]}
                    for k, v in row["fit"].items()},
        }), flush=True)
        torch.cuda.empty_cache()

    payload = {"samples": ids, "aggregate": aggregate(rows), "rows": rows}
    (args.out / "two_parameter_born_diagnostic.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "done", "aggregate": payload["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()

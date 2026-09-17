"""No-network phase-screen pilot for the 11-angle UltraWave dataset.

Run in the neural_asp project root. Screen parameters see only eight input
angles; the three held-out angles are used for evaluation after optimization.
The complex scatterer is reconstructed by a fixed ridge-CG solver.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import build_meta, corr2d, load_config, rf_to_D
from physics.imaging import BornModel


DATA_ROOT = Path("/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle")


def corrected_config(path: str, sample: dict, n_freq: int):
    cfg = load_config(path)
    md = sample["metadata"]
    if md["backend"] != "ultrawave":
        raise ValueError("expected UltraWave sample")
    cfg.array.center_x = 0.0
    # block centres of the 50 um source grid, including the 4x4 block offset
    cfg.grid.x0 = -0.028 + md["crop_x_start"] * 50e-6 + 1.5 * 50e-6
    cfg.grid.z0 = (md["crop_z_start"] - 60) * 50e-6 + 1.5 * 50e-6
    cfg.acq.t_ref_s = md["source_tref_s"]
    cfg.acq.n_freq = n_freq
    cfg.physics.response_mode = "calibrated"
    cfg.physics.pop("response_path", None)
    meta = build_meta(cfg)
    if not np.allclose(meta.angles_deg, md["angles_deg"], atol=1e-10):
        raise ValueError("sample and config angle grids differ")
    if len(meta.t_ref_s) != cfg.acq.n_angles:
        raise ValueError("incorrect number of source delays")
    if abs(cfg.grid.x0 + 0.019125) > 1e-12 or abs(cfg.grid.z0 - 0.000075) > 1e-12:
        raise ValueError("unexpected truth-grid coordinates")
    return cfg, meta


def padded_meta(meta, pad: int, dx: float):
    result = copy.deepcopy(meta)
    result.x0 -= pad * dx
    return result


def embed(image: torch.Tensor, pad: int):
    return F.pad(image, (pad, pad)) if pad else image


def crop(image: torch.Tensor, pad: int):
    return image[..., pad:-pad] if pad else image


def effective_ds(raw: torch.Tensor, nz: int, nx: int, dz: float,
                 limit_us: float = 0.2, pad: int = 0,
                 bulk_raw: torch.Tensor | None = None,
                 z0: float = 0.0, bulk_limit_us: float = 2.0):
    """[L,K] raw controls -> [nz,nx] effective slowness, zero last row.

    Each bounded control is an integrated delay across one depth block;
    subtracting its lateral mean fixes the relative-screen gauge.
    """
    n_layers, _ = raw.shape
    curves = F.interpolate((limit_us * torch.tanh(raw))[None],
                           size=nx, mode="linear", align_corners=True)[0]
    physical = curves[:, pad:nx - pad] if pad else curves
    curves = curves - physical.mean(-1, keepdim=True)
    curves = curves / torch.maximum(curves.abs().amax(dim=-1, keepdim=True) / limit_us,
                                    torch.ones_like(curves[:, :1]))
    edges = torch.linspace(0, nz - 1, n_layers + 1, device=raw.device)
    edges = torch.round(edges).to(torch.long).tolist()
    rows = []
    for layer in range(n_layers):
        depth_steps = edges[layer + 1] - edges[layer]
        if depth_steps <= 0:
            raise ValueError("too many phase-screen layers")
        val = curves[layer] * (1e-6 / (depth_steps * dz))
        rows.extend([val] * depth_steps)
    if bulk_raw is not None:
        b1, b2 = (bulk_limit_us * torch.tanh(bulk_raw)).unbind()
        z = z0 + torch.arange(nz, device=raw.device, dtype=raw.dtype) * dz
        t = z / z[-1]
        G = b1 * t + b2 * t.square()
        dg = (G[1:] - G[:-1]) * (1e-6 / dz)
        rows = [v + dg[j] for j, v in enumerate(rows)]
    rows.append(torch.zeros_like(rows[0]))
    return torch.stack(rows)


def projected_truth_screen(true_ds: torch.Tensor, n_layers: int,
                           n_ctrl: int, dz: float, limit_us: float,
                           pad: int, z0: float, bulk_limit_us: float,
                           fit_bulk: bool):
    """Known-medium diagnostic, never an input to the fitted phase screen."""
    nz, nx = true_ds.shape
    edges = torch.round(torch.linspace(0, nz - 1, n_layers + 1,
                                       device=true_ds.device)).long().tolist()
    mean_profile = (true_ds[:-1, pad:nx - pad] if pad else true_ds[:-1]).mean(-1)
    z = z0 + torch.arange(nz, device=true_ds.device, dtype=true_ds.dtype) * dz
    t = z / z[-1]
    accumulated_us = torch.cat([torch.zeros_like(mean_profile[:1]),
                                torch.cumsum(mean_profile * dz * 1e6, 0)])
    bulk_basis = torch.stack([t, t.square()], dim=-1)
    bulk_coeff = torch.linalg.lstsq(bulk_basis, accumulated_us).solution
    bulk_raw = torch.atanh((bulk_coeff / bulk_limit_us).clamp(-0.999, 0.999))
    bulk_profile = (bulk_limit_us * torch.tanh(bulk_raw))[0] * t
    bulk_profile += (bulk_limit_us * torch.tanh(bulk_raw))[1] * t.square()
    bulk_step_ds = (bulk_profile[1:] - bulk_profile[:-1]) * (1e-6 / dz)
    residual = (true_ds[:-1] - bulk_step_ds[:, None]
                if fit_bulk else true_ds[:-1])
    integrated = torch.stack([residual[edges[l]:edges[l + 1]].sum(0) * dz * 1e6
                              for l in range(n_layers)])
    ctrl = F.interpolate(integrated[None], size=n_ctrl,
                         mode="linear", align_corners=True)[0]
    # ``pad`` is measured in fine-grid pixels, while ``ctrl`` has n_ctrl
    # samples. Expand first so the physical-aperture gauge matches
    # effective_ds()/PhaseScreenModel.screen_to_slowness().
    expanded = F.interpolate(ctrl[None], size=nx,
                             mode="linear", align_corners=True)[0]
    physical = expanded[:, pad:nx - pad] if pad else expanded
    ctrl = ctrl - physical.mean(-1, keepdim=True)
    before_clip = float(ctrl.abs().max().item())
    saturated = float((ctrl.abs() > limit_us).float().mean().item())
    raw = torch.atanh((ctrl / limit_us).clamp(-0.999, 0.999))
    return raw, (bulk_raw if fit_bulk else None), {"target_max_abs_us": before_clip,
                           "saturated_control_fraction": saturated,
                           "bulk_target_coeff_us": bulk_coeff.tolist(),
                           "bulk_applied_coeff_us": (bulk_limit_us * torch.tanh(bulk_raw)).tolist()}


def real_inner(a, b):
    return (a.conj() * b).sum().real


def operator_for(born, ds, idx):
    u = born.transmit_fields(ds, idx)

    def forward(m):
        return born.forward(m, ds, u)

    def adjoint(d):
        return born.adjoint(d, u, ds)

    return forward, adjoint, u


@torch.no_grad()
def ridge_cg(born, ds, idx, D, support, iters=8, ridge_frac=1e-3):
    forward, adjoint, _ = operator_for(born, ds, idx)

    def project(x):
        return x * support

    b = project(adjoint(D))
    b_power = real_inner(b, b).clamp_min(1e-30)
    ab = project(adjoint(forward(b)))
    norm_est = (real_inner(b, ab) / b_power).clamp_min(1e-20)
    ridge = norm_est * ridge_frac

    def normal(x):
        return project(adjoint(forward(project(x)))) + ridge * x

    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = real_inner(r, r)
    history = []
    for _ in range(iters):
        ap = normal(p)
        denom = real_inner(p, ap).clamp_min(1e-30)
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * ap
        rs_next = real_inner(r, r)
        history.append(float((rs_next / b_power).sqrt().item()))
        if float(rs_next.item()) < float(b_power.item()) * 1e-10:
            break
        p = r + (rs_next / rs.clamp_min(1e-30)) * p
        rs = rs_next
    return project(x), float(ridge.item()), history


def relative_residual(pred, target):
    return float(((pred - target).abs().square().sum() /
                  target.abs().square().sum().clamp_min(1e-30)).sqrt().item())


@torch.no_grad()
def score(born, ds, m, D, tr, ho, pad, truth_m):
    f_tr, a_tr, u_tr = operator_for(born, ds, tr)
    d_tr = f_tr(m)
    u_ho = born.transmit_fields(ds, ho)
    d_ho = born.forward(m, ds, u_ho)
    image = a_tr(D[tr])
    truth_abs = truth_m.abs()
    return {
        "rf_input": relative_residual(d_tr, D[tr]),
        "rf_holdout": relative_residual(d_ho, D[ho]),
        "image_abs_corr": float(corr2d(crop(image, pad).abs(), truth_abs).item()),
        "m_abs_corr": float(corr2d(crop(m, pad).abs(), truth_abs).item()),
    }


def optimize_phase(born, m_fixed, D, tr, n_layers, n_ctrl, limit_us,
                   iterations, lr, reg, output_log, pad=0,
                   fit_bulk=False, bulk_limit_us=2.0):
    raw = torch.nn.Parameter(torch.zeros(n_layers, n_ctrl, device=D.device))
    bulk_raw = torch.nn.Parameter(torch.zeros(2, device=D.device)) if fit_bulk else None
    opt = torch.optim.Adam([raw, bulk_raw] if fit_bulk else [raw], lr=lr)
    denom = D[tr].abs().square().mean().detach().clamp_min(1e-30)
    best_obj = float("inf")
    best_raw = raw.detach().clone()
    best_bulk = bulk_raw.detach().clone() if fit_bulk else None
    for step in range(iterations):
        opt.zero_grad(set_to_none=True)
        ds = effective_ds(raw, born.nz, born.nx, born.dz, limit_us,
                          pad, bulk_raw, born.z0, bulk_limit_us)
        u = born.transmit_fields(ds, tr)
        pred = born.forward(m_fixed, ds, u)
        data_loss = (pred - D[tr]).abs().square().mean() / denom
        curve = limit_us * torch.tanh(raw)
        regularizer = (curve.square().mean() / limit_us**2
                       + 0.1 * (curve[:, 1:] - curve[:, :-1]).square().mean() / limit_us**2
                       + 0.1 * (curve[1:] - curve[:-1]).square().mean() / limit_us**2)
        if fit_bulk:
            regularizer = regularizer + 0.1 * torch.tanh(bulk_raw).square().mean()
        loss = data_loss + reg * regularizer
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite phase-screen objective")
        if float(loss.detach().item()) < best_obj:
            best_obj = float(loss.detach().item())
            best_raw = raw.detach().clone()
            best_bulk = bulk_raw.detach().clone() if fit_bulk else None
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw, bulk_raw] if fit_bulk else [raw], 1.0)
        opt.step()
        output_log.append({"step": step + 1,
                           "data_loss": float(data_loss.detach().item()),
                           "regularizer": float(regularizer.detach().item()),
                           "max_abs_tau_us": float((limit_us * torch.tanh(raw)).abs().max().item())})
    return best_raw, best_bulk


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
    # One sample at a time. No held-out RF enters phase fitting or m fitting.
    tr = torch.as_tensor(meta.train_idx, device=device)
    ho = torch.as_tensor(meta.hold_idx, device=device)
    truth_m = sample["m"].to(device)
    support = embed(torch.ones_like(truth_m.real), args.pad)
    zero = torch.zeros(born.nz, born.nx, device=device)
    c_ds = embed(sample["delta_s"].to(device), args.pad)
    if args.pad:
        c_ds[:, :args.pad] = 0
        c_ds[:, -args.pad:] = 0

    t0 = time.monotonic()
    m0, lam0, cg0 = ridge_cg(born, zero, tr, D[tr], support,
                             args.cg_iters, args.ridge_frac)
    zero_score = score(born, zero, m0, D, tr, ho, args.pad, truth_m)
    print(json.dumps({"sample": sample_id, "stage": "uniform",
                      "elapsed_s": time.monotonic() - t0, **zero_score}), flush=True)

    opt_log = []
    raw, bulk_raw = optimize_phase(born, m0.detach(), D, tr, args.layers,
                                   args.controls, args.limit_us, args.opt_steps,
                                   args.lr, args.reg, opt_log, args.pad,
                                   args.fit_bulk, args.bulk_limit_us)
    phase_ds = effective_ds(raw, born.nz, born.nx, born.dz, args.limit_us,
                            args.pad, bulk_raw, born.z0, args.bulk_limit_us)
    m1, lam1, cg1 = ridge_cg(born, phase_ds, tr, D[tr], support,
                             args.cg_iters, args.ridge_frac)
    phase_score = score(born, phase_ds, m1, D, tr, ho, args.pad, truth_m)
    print(json.dumps({"sample": sample_id, "stage": "phase_screen",
                      "elapsed_s": time.monotonic() - t0, **phase_score}), flush=True)

    oracle = None
    projected = None
    if args.oracle:
        truth_raw, truth_bulk, truth_projection = projected_truth_screen(
            c_ds, args.layers, args.controls, born.dz, args.limit_us,
            args.pad, born.z0, args.bulk_limit_us, args.fit_bulk)
        truth_screen_ds = effective_ds(truth_raw, born.nz, born.nx,
                                       born.dz, args.limit_us, args.pad,
                                       truth_bulk if args.fit_bulk else None,
                                       born.z0, args.bulk_limit_us)
        mp, lamp, cgp = ridge_cg(born, truth_screen_ds, tr, D[tr], support,
                                 args.cg_iters, args.ridge_frac)
        projected = {"score": score(born, truth_screen_ds, mp, D, tr, ho,
                                     args.pad, truth_m), "ridge": lamp,
                     "cg_history": cgp, **truth_projection}
        print(json.dumps({"sample": sample_id, "stage": "projected_c_screen",
                          "elapsed_s": time.monotonic() - t0,
                          **projected["score"], **truth_projection}), flush=True)
        mo, lamo, cgo = ridge_cg(born, c_ds, tr, D[tr], support,
                                 args.cg_iters, args.ridge_frac)
        oracle = {"score": score(born, c_ds, mo, D, tr, ho,
                                  args.pad, truth_m), "ridge": lamo,
                  "cg_history": cgo}
        print(json.dumps({"sample": sample_id, "stage": "known_c",
                          "elapsed_s": time.monotonic() - t0,
                          **oracle["score"]}), flush=True)

    result = {
        "sample_id": sample_id,
        "metadata": {"angles_deg": list(map(float, meta.angles_deg)),
                     "train_idx": list(map(int, meta.train_idx)),
                     "hold_idx": list(map(int, meta.hold_idx)),
                     "x0_m": float(meta.x0), "z0_m": float(meta.z0),
                     "t_ref_s": list(map(float, meta.t_ref_s)),
                     "n_freq": len(meta.freqs), "fft_pad_elements": args.pad},
        "uniform": {"score": zero_score, "ridge": lam0,
                    "cg_history": cg0},
        "phase_screen": {"score": phase_score, "ridge": lam1,
                         "cg_history": cg1, "optimization": opt_log,
                         "bulk_coeff_us": ((args.bulk_limit_us * torch.tanh(bulk_raw)).tolist()
                                           if bulk_raw is not None else None),
                         "max_abs_control_us": float((args.limit_us * torch.tanh(raw)).abs().max().item()),
                         "rms_screen_ns": float((phase_ds[:-1] * born.dz * 1e9).square().mean().sqrt().item())},
        "known_c": oracle,
        "projected_c_screen": projected,
        "elapsed_s": time.monotonic() - t0,
    }
    return result, raw.cpu(), (bulk_raw.cpu() if bulk_raw is not None else None), crop(m0, args.pad).cpu(), crop(m1, args.pad).cpu()


def selftest():
    p = torch.zeros(12, 48, requires_grad=True)
    ds0 = effective_ds(p, 216, 192, 0.2e-3)
    assert ds0.shape == (216, 192) and torch.count_nonzero(ds0) == 0
    p2 = (torch.randn_like(p) * 0.1).detach().requires_grad_(True)
    ds = effective_ds(p2, 216, 192, 0.2e-3)
    assert ds[-1].abs().max().item() == 0
    torch.testing.assert_close(ds[:-1].mean(-1), torch.zeros(215), atol=1e-9, rtol=0)
    assert torch.isfinite(ds).all()
    ds.square().mean().backward()
    assert p2.grad is not None and torch.isfinite(p2.grad).all()
    print("phase-screen parameterization selftest passed", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/l11_ultrawave_500_11angle.yaml")
    p.add_argument("--samples", nargs="+", default=["train_000", "val_000"])
    p.add_argument("--out", type=Path, default=Path("runs/l11_phase_asp_pilot"))
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--n-freq", type=int, default=64)
    p.add_argument("--pad", type=int, default=32)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--controls", type=int, default=48)
    p.add_argument("--limit-us", type=float, default=0.2)
    p.add_argument("--cg-iters", type=int, default=8)
    p.add_argument("--ridge-frac", type=float, default=1e-3)
    p.add_argument("--opt-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--reg", type=float, default=0.01)
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--fit-bulk", action="store_true")
    p.add_argument("--bulk-limit-us", type=float, default=2.0)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    torch.manual_seed(20260916)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    args.out.mkdir(parents=True, exist_ok=True)
    all_results = []
    for sample_id in args.samples:
        result, raw, bulk, m0, m1 = run_one(args, sample_id, device)
        (args.out / f"{sample_id}.json").write_text(json.dumps(result, indent=2) + "\n")
        torch.save({"raw_phase_controls": raw, "raw_bulk": bulk, "m_uniform": m0,
                    "m_phase": m1}, args.out / f"{sample_id}.pt")
        all_results.append(result)
        torch.cuda.empty_cache()
    summary = {
        "samples": args.samples,
        "settings": vars(args) | {"out": str(args.out)},
        "mean": {key: {metric: float(np.mean([r[key]["score"][metric]
                                         for r in all_results]))
                       for metric in all_results[0]["uniform"]["score"]}
                 for key in ("uniform", "phase_screen")},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"stage": "summary", **summary["mean"]}), flush=True)


if __name__ == "__main__":
    main()

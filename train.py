"""Staged training of the closed loop.

Stages (spec section 6):
  m     : fixed (true) propagation model, train only the scattering-image
          prox P_psi / steps gamma_k  ->  runs/<out>/stage_m.pt
  eta   : train the neural operator with slowness supervision (resumes m)
          ->  runs/<out>/stage_eta.pt
  joint : joint fine-tuning, slowness supervision weight reduced (resumes eta)
          ->  runs/<out>/joint.pt

Loss:  L = lambda_d ||F_eta(m_hat) - d||^2 + lambda_c L_c + lambda_I L_I
           + lambda_s R(eta)
with holdout transmit angles excluded from every training loss.

Usage:
  python train.py --stage m --out demo
  python train.py --stage eta --out demo --resume runs/demo/stage_m.pt
  python train.py --stage joint --out demo --resume runs/demo/stage_eta.pt
  python train.py --stage all --out demo          # runs the three in order
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from common import C, build_meta, get_device, load_config, relative_residual, set_seed, demod_iq, to_plain
from data.synthetic import SyntheticUSDataset
from models.pipeline import ImagingPipeline, load_pipeline_checkpoint


def make_dataset(cfg, split, device):
    source = cfg.get("data", {}).get("source", "synthetic")
    if source == "synthetic":
        return SyntheticUSDataset(cfg, split, device)
    if source in ("l11_kwave", "l11_fullwave"):
        from data.l11_kwave import L11KWaveDataset
        return L11KWaveDataset(cfg, split, device)
    raise ValueError(f"unknown training data source {source!r}")


def tv_norm(x):
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


def compute_losses(out, batch, meta, cfg, stage, tr_idx):
    ds_max = cfg.model.ds_max
    D_tr = batch["D"][:, tr_idx]
    lam = cfg.train

    supervision_only = stage == "eta" and lam.get("eta_supervision_only", False)
    L_d = torch.zeros((), device=D_tr.device) if supervision_only else \
        (out["d_hat"] - D_tr).abs().pow(2).mean() / D_tr.abs().pow(2).mean().detach().clamp_min(1e-20)

    if stage == "m" or cfg.train.force_uniform:
        L_c = torch.zeros((), device=D_tr.device)
        R = torch.zeros((), device=D_tr.device)
    else:
        g = cfg.grid
        f = cfg.coarse.factor
        target = torch.nn.functional.avg_pool2d(
            batch["delta_s"][:, None], f)                    # [B,1,nsz,nsx]
        ds_coarse = out["delta_s_coarse"]
        L_c = (ds_coarse - target).abs().mean() / ds_max
        R = tv_norm(ds_coarse) / ds_max \
            + (ds_coarse / ds_max).pow(2).mean()

    L_I = torch.zeros((), device=D_tr.device) if supervision_only or lam.lambda_I == 0 else \
        (out["m_hat"] - batch["m"]).abs().mean() / batch["m"].abs().mean().detach().clamp_min(1e-12)

    w_c = 0.0 if (stage in ("m",) or cfg.train.force_uniform) else \
        (lam.get("joint_c_weight", 0.2) if stage == "joint" else 1.0)
    w_d = lam.get("joint_d_weight", 1.0) if stage == "joint" else 1.0
    total = (lam.lambda_d * w_d * L_d + lam.lambda_c * w_c * L_c
             + lam.lambda_I * L_I + lam.lambda_s * R)
    parts = {"L_d": L_d.item(), "L_c": L_c.item() if torch.is_tensor(L_c) else 0.0,
             "L_I": L_I.item(), "R": R.item() if torch.is_tensor(R) else 0.0,
             "total": total.item()}
    return total, parts


@torch.no_grad()
def validate_stage(pipe, dataset, cfg, meta, device, stage):
    """Aggregate every validation sample, never use test data for selection."""
    pipe.eval()
    tr = torch.as_tensor(meta.train_idx, device=device)
    ho = torch.as_tensor(meta.hold_idx, device=device)
    rows = []
    for start in range(0, len(dataset), cfg.train.batch_size):
        sl = slice(start, start + cfg.train.batch_size)
        rf, ds, D = dataset.rf[sl], dataset.delta_s[sl], dataset.D[sl]
        mode = "truth" if stage == "m" else ("zero" if cfg.train.force_uniform else "net")
        o = pipe(rf, delta_s_true=ds, eta_mode=mode, train_idx=tr, pred_idx=ho, return_all=False)
        for b in range(len(rf)):
            rows.append({"ours_train": relative_residual(o["d_hat"][b], D[b, tr]).item(),
                         "ours_hold": relative_residual(o["d_hat_pred"][b], D[b, ho]).item(),
                         "c_rmse": (o["c_hat"][b] - dataset.c[sl][b]).square().mean().sqrt().item()})
    pipe.train()
    return {**{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, "n_samples": len(rows)}


def run_stage(stage, cfg, device, out_dir, resume=None):
    meta = build_meta(cfg)
    tr_idx = torch.tensor(meta.train_idx, device=device)
    ho_idx = torch.tensor(meta.hold_idx, device=device)

    train_set = make_dataset(cfg, "train", device)
    val_set = make_dataset(cfg, "val", device)

    pipe = ImagingPipeline(cfg, meta).to(device)
    if resume:
        load_pipeline_checkpoint(pipe, torch.load(resume, map_location=device, weights_only=False))

    no_params = list(pipe.neural_operator.parameters())
    other_params = [p for n, p in pipe.named_parameters()
                    if not n.startswith("neural_operator")]
    freeze = cfg.train.get("freeze_stages", False)
    for p in no_params:
        p.requires_grad_(not (freeze and stage == "m"))
    for p in other_params:
        p.requires_grad_(not (freeze and stage == "eta"))
    lr_scale = cfg.train.get("joint_lr_scale", 1.0) if stage == "joint" else 1.0
    opt = torch.optim.Adam([
        {"params": no_params, "lr": cfg.train.lr * lr_scale},
        {"params": other_params, "lr": cfg.train.lr_prox * lr_scale},
    ], weight_decay=cfg.train.weight_decay)

    steps = cfg.train.steps
    B = cfg.train.batch_size
    n = len(train_set)
    history = []
    val_history = []
    ckpt = os.path.join(out_dir, f"stage_{stage}.pt" if stage != "joint" else "joint.pt")
    best_score = float("inf")
    best_step = -1
    best_res = None
    def check_and_save(step):
        nonlocal best_score, best_step, best_res
        res = validate_stage(pipe, val_set, cfg, meta, device, stage)
        score = res["ours_train"] if stage == "m" or cfg.train.force_uniform else res["c_rmse"]
        val_history.append({"step": step, **res})
        if math.isfinite(score) and score < best_score:
            best_score, best_step, best_res = score, step, res
            torch.save({"pipeline": pipe.state_dict(), "stage": stage,
                        "operator_version": pipe.operator_version, "config": to_plain(cfg),
                        "selected_step": step, "history": history.copy(), "val_residuals": res}, ckpt)
        print(f"[{stage}] val step={step} best={best_step}: {json.dumps(res)}", flush=True)
    check_and_save(-1)
    # lateral-flip augmentation (exact symmetry of the physics): flip x
    # => reverse the angle list, the element order and the maps
    n_a, n_e = cfg.acq.n_angles, cfg.array.n_elements
    ang_flip = torch.arange(n_a - 1, -1, -1, device=device)
    e_flip = torch.arange(n_e - 1, -1, -1, device=device)
    tr_flip = ang_flip[tr_idx]
    t0 = time.time()
    pipe.train()
    for it in range(steps):
        i = torch.randint(0, n, (B,))
        batch = {k: v[i] for k, v in
                 {"rf": train_set.rf, "D": train_set.D,
                  "delta_s": train_set.delta_s, "m": train_set.m}.items()}
        ang_idx = tr_idx
        if it % 2 == 1 and cfg.train.get("lateral_flip", True):
            batch["rf"] = batch["rf"][:, ang_flip][:, :, e_flip, :]
            batch["D"] = batch["D"][:, ang_flip][:, :, :, e_flip]
            batch["delta_s"] = torch.flip(batch["delta_s"], dims=[-1])
            batch["m"] = torch.flip(batch["m"], dims=[-1])
            ang_idx = tr_flip
        eta_mode = "truth" if stage == "m" else \
            ("zero" if cfg.train.force_uniform else "net")
        if stage == "eta" and cfg.train.get("eta_supervision_only", False):
            coarse = pipe.neural_operator(demod_iq(batch["rf"], meta)[:, ang_idx], ang_idx)["delta_s"]
            out = {"delta_s_coarse": coarse}
        else:
            out = pipe(batch["rf"], delta_s_true=batch["delta_s"],
                       eta_mode=eta_mode, train_idx=ang_idx, return_all=False)
        loss, parts = compute_losses(out, batch, meta, cfg, stage, ang_idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pipe.parameters(), cfg.train.grad_clip)
        opt.step()
        with torch.no_grad():
            pipe.gamma.clamp_(min=0)
            for prox in pipe.prox:
                prox.thresh.clamp_(min=0)

        if it % 20 == 0 or it == steps - 1:
            msg = {k: round(v, 5) for k, v in parts.items()}
            print(f"[{stage}] step {it:4d}/{steps} {msg} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            history.append({"step": it, **parts})
        if (it + 1) % cfg.train.get("val_every", 100) == 0 or it == steps - 1:
            check_and_save(it)

    with open(os.path.join(out_dir, f"history_{stage}.json"), "w") as f:
        json.dump(history, f, indent=1)
    with open(os.path.join(out_dir, f"validation_{stage}.json"), "w") as f:
        json.dump({"selected_step": best_step, "best": best_res, "history": val_history}, f, indent=1)
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stage", default="joint",
                    choices=["m", "eta", "joint", "all"])
    ap.add_argument("--out", default=None, help="run directory name")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force-uniform", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.out:
        cfg.output_dir = f"runs/{args.out}"
    if args.steps:
        cfg.train.steps = args.steps
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.train.force_uniform = args.force_uniform
    cfg.train.stage = args.stage
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        import yaml
        from common import to_plain
        yaml.safe_dump(to_plain(cfg), f)

    print(f"device: {device}, out: {out_dir}")
    if args.stage == "all":
        c1 = run_stage("m", cfg, device, out_dir, args.resume)
        c2 = run_stage("eta", cfg, device, out_dir, c1)
        run_stage("joint", cfg, device, out_dir, c2)
    else:
        run_stage(args.stage, cfg, device, out_dir, args.resume)


if __name__ == "__main__":
    main()

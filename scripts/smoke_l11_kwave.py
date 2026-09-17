"""One-sample, one-step forward/backward smoke test for the L11 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from common import build_meta, get_device, load_config, set_seed  # noqa: E402
from data.l11_kwave import validate_l11_sample  # noqa: E402
from models.pipeline import ImagingPipeline  # noqa: E402
from train import compute_losses  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PROJECT / "configs/l11_kwave.yaml")
    ap.add_argument("--sample", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--stage", choices=("m", "eta", "joint"), default="m")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    root = Path(cfg.data.root)
    sample_path = args.sample or root / "shards/train_000.pt"
    output_path = args.output or root / "pilot_smoke.json"
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    meta = build_meta(cfg)
    validate_l11_sample(sample, cfg, meta, check_transform=True)
    batch = {k: sample[k][None].to(device)
             for k in ("rf", "D", "delta_s", "m", "c")}
    train_idx = torch.as_tensor(meta.train_idx, dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pipe = ImagingPipeline(cfg, meta).to(device).train()
    opt = torch.optim.Adam(pipe.parameters(), lr=float(cfg.train.lr_prox))
    t0 = time.monotonic()
    out = pipe(batch["rf"], delta_s_true=batch["delta_s"],
               eta_mode="truth" if args.stage == "m" else "net",
               train_idx=train_idx, return_all=False)
    loss, parts = compute_losses(out, batch, meta, cfg, args.stage, train_idx)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    finite_gradients = all(torch.isfinite(p.grad).all().item()
                           for p in pipe.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(pipe.parameters(), cfg.train.grad_clip)
    opt.step()
    result = {
        "ok": bool(torch.isfinite(loss).item() and finite_gradients),
        "stage": args.stage,
        "device": str(device), "sample": str(sample_path),
        "elapsed_s": time.monotonic() - t0, "losses": parts,
        "finite_gradients": finite_gradients,
        "rf_shape": list(batch["rf"].shape),
        "D_shape": list(batch["D"].shape),
        "frequency_bins": len(meta.freqs),
    }
    if device.type == "cuda":
        result["peak_cuda_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise RuntimeError("L11 smoke test failed")


if __name__ == "__main__":
    main()

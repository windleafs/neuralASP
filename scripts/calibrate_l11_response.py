"""Fit one frozen system response from training samples/input angles only.

This is an approximate least-squares response with the proxy m labels, not
an exact physical identification. No validation/test or held-out RF is fitted.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_config, build_meta, rf_to_D
from physics.imaging import BornModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/l11_kwave_repaired.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.physics.pop("response_path"))
    meta = build_meta(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    g = cfg.grid
    born = BornModel(meta, g.nx, g.nz, g.dx, g.dz, cfg.physics.c0,
                     eps=cfg.physics.eps_evanescent, spreading=cfg.physics.spreading).to(device)
    root = Path(cfg.data.root)
    records = [r for r in json.loads((root / "index.json").read_text())["samples"]
               if r["split"] == "train" and r.get("status") == "complete"]
    if len(records) != cfg.train.n_train:
        raise ValueError("incomplete training set")
    tr = torch.as_tensor(meta.train_idx, device=device)
    num = torch.zeros(len(meta.freqs), device=device, dtype=torch.complex128)
    den = torch.zeros(len(meta.freqs), device=device, dtype=torch.float64)
    with torch.no_grad():
        for r in records:
            s = torch.load(root / r["path"], map_location=device, weights_only=False)
            ds = s["delta_s"][None]
            u = born.transmit_fields(ds, tr)
            pred = born(s["m"][None], ds, u).to(torch.complex128)
            target = rf_to_D(s["rf"][None], meta)[:, tr].to(torch.complex128)
            num += (target * pred.conj()).sum((0, 1, 3))
            den += pred.abs().square().sum((0, 1, 3))
    response = (num / den.clamp_min(den.max() * 1e-10)).cpu().numpy()
    # Smooth numerator and denominator jointly to stabilize low-energy bins.
    from scipy.ndimage import gaussian_filter1d
    response = gaussian_filter1d(num.cpu().numpy(), 1.0) / np.maximum(gaussian_filter1d(den.cpu().numpy(), 1.0), den.max().item() * 1e-10)
    if not np.isfinite(response).all():
        raise ValueError("non-finite fitted response")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, freqs=meta.freqs, response=response)
    provenance = {"method": "training_proxy_least_squares_smoothed", "n_samples": len(records),
                  "sample_ids": [r["id"] for r in records], "fit_angles": meta.train_idx.tolist(),
                  "excluded_angles": meta.hold_idx.tolist(), "config": args.config,
                  "limitation": "m is a proxy label; response is an approximate shared calibration"}
    out.with_suffix(".json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance), flush=True)
    print(f"saved {out}: {len(response)} frequencies", flush=True)


if __name__ == "__main__":
    main()

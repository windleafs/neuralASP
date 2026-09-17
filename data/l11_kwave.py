"""L11 k-Wave shard dataset for neural angular-spectrum training."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from common import build_meta, rf_to_D

__all__ = ["L11KWaveDataset", "validate_l11_sample"]


def validate_l11_sample(sample, cfg, meta, check_transform=True):
    """Validate one persisted shard against the configured tensor contract."""
    backend = cfg.get("data", {}).get("required_backend")
    if backend and sample.get("metadata", {}).get("backend") != backend:
        raise ValueError(f"sample backend is not {backend!r}")
    expected = {
        "rf": (cfg.acq.n_angles, cfg.array.n_elements, cfg.acq.n_t),
        "D": (cfg.acq.n_angles, len(meta.freqs), cfg.array.n_elements),
        "delta_s": (cfg.grid.nz, cfg.grid.nx),
        "m": (cfg.grid.nz, cfg.grid.nx),
        "c": (cfg.grid.nz, cfg.grid.nx),
    }
    for key, shape in expected.items():
        if key not in sample:
            raise ValueError(f"sample is missing {key!r}")
        if tuple(sample[key].shape) != tuple(shape):
            raise ValueError(f"{key} shape {tuple(sample[key].shape)} != {shape}")
        if not torch.isfinite(sample[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    if sample["rf"].dtype != torch.float32:
        raise ValueError(f"rf dtype must be float32, got {sample['rf'].dtype}")
    if sample["D"].dtype != torch.complex64:
        raise ValueError(f"D dtype must be complex64, got {sample['D'].dtype}")
    if sample["delta_s"].abs().max().item() > cfg.model.ds_max * 1.001:
        raise ValueError("delta_s exceeds model.ds_max")
    m_rms = sample["m"].abs().pow(2).mean().sqrt().item()
    if abs(m_rms - cfg.model.m_rms_ref) > 2e-4:
        raise ValueError(f"m RMS {m_rms:.6g} != {cfg.model.m_rms_ref}")
    if sample["rf"].pow(2).mean().sqrt().item() <= 0:
        raise ValueError("RF energy is zero")
    if check_transform:
        fresh = rf_to_D(sample["rf"], meta)
        if not torch.allclose(sample["D"], fresh, rtol=2e-5, atol=1e-8):
            err = (sample["D"] - fresh).abs().max().item()
            raise ValueError(f"D does not match rf_to_D(rf); max error {err:.3g}")


class L11KWaveDataset(Dataset):
    def __init__(self, cfg, split, device):
        self.root = Path(cfg.data.root)
        index_path = self.root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        index = json.loads(index_path.read_text())
        records = [r for r in index["samples"]
                   if r["split"] == split and r.get("status") == "complete"]
        expected_n = {"train": cfg.train.n_train,
                      "val": cfg.train.n_val,
                      "test": cfg.train.n_test}[split]
        if len(records) != expected_n:
            raise RuntimeError(f"{split}: found {len(records)} complete samples, "
                               f"expected {expected_n}")
        records.sort(key=lambda r: r["id"])
        samples = []
        meta = build_meta(cfg)
        for record in records:
            shard = self.root / record["path"]
            sample = torch.load(shard, map_location="cpu", weights_only=False)
            if cfg.data.get("recompute_D", False):
                sample["D"] = rf_to_D(sample["rf"], meta)
            validate_l11_sample(sample, cfg, meta, check_transform=False)
            samples.append(sample)
        self.rf = torch.stack([s["rf"] for s in samples]).to(device)
        self.D = torch.stack([s["D"] for s in samples]).to(device)
        self.delta_s = torch.stack([s["delta_s"] for s in samples]).to(device)
        self.m = torch.stack([s["m"] for s in samples]).to(device)
        self.c = torch.stack([s["c"] for s in samples]).to(device)
        self.records = records

    def __len__(self):
        return len(self.rf)

    def __getitem__(self, i):
        return {"rf": self.rf[i], "D": self.D[i],
                "delta_s": self.delta_s[i], "m": self.m[i], "c": self.c[i]}

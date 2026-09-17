"""Numeric, anatomy, backend and smoke report for the independent500 set."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT))
from common import build_meta,load_config
from data.l11_fullwave import validate_l11_sample


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,default=PROJECT/"configs/l11_ultrawave_500_11angle.yaml")
    ap.add_argument("--smoke-run",type=Path,default=PROJECT/"runs/l11_ultrawave_500_11angle_smoke")
    args=ap.parse_args(); cfg=load_config(args.config); root=Path(cfg.data.root)
    manifest=json.loads((root/"index.json").read_text()); meta=build_meta(cfg)
    stats={"rf_rms":[],"c_min":[],"c_max":[],"ds_max":[],"m_rms":[],"elapsed_s":[]}
    backend=Counter(); unique=set(); byte_count=0
    for r in manifest["samples"]:
        p=root/r["path"]; s=torch.load(p,map_location="cpu",weights_only=False)
        validate_l11_sample(s,cfg,meta,False); md=s["metadata"]
        backend[md["backend"]]+=1; unique.add(md["base_anatomy_id"]); byte_count+=p.stat().st_size
        stats["rf_rms"].append(float(s["rf"].square().mean().sqrt()))
        stats["c_min"].append(float(s["c"].min())); stats["c_max"].append(float(s["c"].max()))
        stats["ds_max"].append(float(s["delta_s"].abs().max()))
        stats["m_rms"].append(float(s["m"].abs().square().mean().sqrt()))
        stats["elapsed_s"].append(md["elapsed_s"])
    eval_metrics=json.loads((args.smoke_run/"eval/metrics.json").read_text())
    result=dict(ok=True,dataset_root=str(root),config=str(args.config),samples=len(manifest["samples"]),
                counts=dict(Counter(r["split"] for r in manifest["samples"])),backend=dict(backend),
                independent_patients=3,unique_anatomical_slices=len(unique),anatomy_repeated_fraction=1-len(unique)/500,
                anatomy_stats=manifest["anatomy_stats"],acquisition=manifest["acquisition"],physics=manifest["physics"],
                model_grid=manifest["model_grid"],frequency_bins=len(meta.freqs),
                numeric_min_max={k:[min(v),max(v)] for k,v in stats.items()},shard_bytes=byte_count,
                normalization=json.loads((root/"normalization.json").read_text()),
                validation=json.loads((root/"validation_summary.json").read_text()),
                pilot_smoke_m=json.loads((root/"pilot_smoke_m.json").read_text()),
                pilot_smoke_joint=json.loads((root/"pilot_smoke_joint.json").read_text()),
                pilot_bmode_metrics=json.loads((root/"pilot_bmode_metrics.json").read_text()),
                formal_smoke="m/eta/joint each1step; no full training",eval_smoke_samples=len(eval_metrics["per_sample"]),
                eval_smoke_mean=eval_metrics["mean"],
                limitations=["2D linear acoustic full-wave; no BonA nonlinearity or out-of-plane propagation",
                             "50um finite-difference grid has discretization error; not a fine-grid convergence claim",
                             "m is normalized high-pass log-impedance proxy; neural_asp Born forward has model mismatch",
                             "Smoke validates interfaces, not trained reconstruction accuracy"])
    if result["samples"]!=500 or result["backend"]!={"ultrawave":500} or result["eval_smoke_samples"]!=50:
        raise RuntimeError("incomplete final report")
    (root/"dataset_report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()

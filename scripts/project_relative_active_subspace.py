"""Project population active subspace away from the lateral common mode.

The original candidate space contains DCT k=0 at every candidate depth. Those
coordinates overlap the explicit mean-delay branch. This script removes them
from the population Gram matrices before eigendecomposition, yielding a clean
relative-aberration active basis.

Input:
  population_active_subspace.pt

Output:
  population_relative_active_subspace.pt
  relative_active_subspace_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from physics.active_subspace import gram_spectrum
from physics.relative_active_subspace import (
    common_mode_indices,
    project_gram_remove_common,
    relative_energy_fraction,
    relative_mode_indices,
)


def compact(rep):
    return {
        "rank_90": rep["rank_90"],
        "rank_95": rep["rank_95"],
        "rank_99": rep["rank_99"],
        "leading_energy_fraction": rep["energy_fraction"][:12].cpu().tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.input, map_location="cpu", weights_only=False)

    Gp = torch.as_tensor(payload["Gp"], dtype=torch.float64)
    Ga = torch.as_tensor(payload["Ga"], dtype=torch.float64)
    depths_mm = [float(v) for v in payload["depths_mm"]]
    n_depth = len(depths_mm)
    if n_depth < 1 or Gp.shape[0] % n_depth:
        raise ValueError("candidate dimension is incompatible with depth list")
    n_lat = Gp.shape[0] // n_depth
    if n_lat < 2:
        raise ValueError("need at least k=0 plus one relative lateral mode")

    Gp_rel = project_gram_remove_common(Gp, n_depth, n_lat)
    Ga_rel = project_gram_remove_common(Ga, n_depth, n_lat)
    rep_p = gram_spectrum(Gp_rel)
    rep_a = gram_spectrum(Ga_rel)

    Gpl = payload.get("Gpl")
    Gal = payload.get("Gal")
    if Gpl is not None and Gal is not None:
        Gpl = torch.as_tensor(Gpl, dtype=torch.float64)
        Gal = torch.as_tensor(Gal, dtype=torch.float64)
        Gpl_rel = project_gram_remove_common(Gpl, n_depth, n_lat)
        Gal_rel = project_gram_remove_common(Gal, n_depth, n_lat)
        rep_pl = gram_spectrum(Gpl_rel)
        rep_al = gram_spectrum(Gal_rel)
    else:
        Gpl_rel = Gal_rel = None
        rep_pl = rep_al = None

    common_idx = common_mode_indices(n_depth, n_lat)
    relative_idx = relative_mode_indices(n_depth, n_lat)

    out_payload = {
        "Gp": Gp_rel.float(),
        "Ga": Ga_rel.float(),
        "phase_Vh": rep_p["Vh"].float(),
        "amplitude_Vh": rep_a["Vh"].float(),
        "depths_mm": depths_mm,
        "labels_phase": list(payload.get("labels_phase", [])),
        "labels_amplitude": list(payload.get("labels_amplitude", [])),
        "relative_only": True,
        "common_mode_removed": "lateral_dct_k0_at_each_depth",
        "common_mode_indices": common_idx,
        "relative_mode_indices": relative_idx,
        "n_lateral_modes": int(n_lat),
        "source_artifact": str(args.input),
    }
    if Gpl_rel is not None:
        out_payload.update({
            "Gpl": Gpl_rel.float(),
            "Gal": Gal_rel.float(),
            "phase_log_Vh": rep_pl["Vh"].float(),
            "amplitude_log_Vh": rep_al["Vh"].float(),
        })

    out_pt = args.out / "population_relative_active_subspace.pt"
    torch.save(out_payload, out_pt)

    summary = {
        "experiment": "remove lateral common mode from population active subspace",
        "input": str(args.input),
        "output": str(out_pt),
        "candidate_dim": int(Gp.shape[0]),
        "n_depth": int(n_depth),
        "n_lateral_modes": int(n_lat),
        "common_dimensions_removed": len(common_idx),
        "relative_candidate_dim": len(relative_idx),
        "common_mode_indices": common_idx,
        "phase_relative_energy_fraction": relative_energy_fraction(Gp, Gp_rel),
        "amplitude_relative_energy_fraction": relative_energy_fraction(Ga, Ga_rel),
        "phase_relative": compact(rep_p),
        "amplitude_relative": compact(rep_a),
    }
    if rep_pl is not None:
        summary["phase_log_relative_energy_fraction"] = relative_energy_fraction(
            torch.as_tensor(Gpl, dtype=torch.float64), Gpl_rel)
        summary["amplitude_log_relative_energy_fraction"] = relative_energy_fraction(
            torch.as_tensor(Gal, dtype=torch.float64), Gal_rel)
        summary["phase_log_relative"] = compact(rep_pl)
        summary["amplitude_log_relative"] = compact(rep_al)

    out_json = args.out / "relative_active_subspace_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

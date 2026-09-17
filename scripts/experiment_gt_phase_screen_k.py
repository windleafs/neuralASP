"""GT slowness -> K-layer screens -> identical ASM transmit/adjoint correction.

Two screen families isolate depth compression from limits of the present
48-control, bounded phase-head representation. No model training or CG.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if not (PROJECT / 'common.py').exists():
    PROJECT = Path('/home/zhuangyang/fmmodel/neural_asp')
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

from common import corr2d, rf_to_D  # noqa: E402
from physics.imaging import BornModel  # noqa: E402
from scripts.pilot_phase_asp import (DATA_ROOT, corrected_config, crop, embed,
                                     effective_ds, padded_meta,
                                     projected_truth_screen)  # noqa: E402
from scripts.pilot_phase_only import (coherence, heldout_agreement)  # noqa: E402


def layer_edges(nz: int, layers: int, device):
    edges = torch.round(torch.linspace(0, nz - 1, layers + 1,
                                       device=device)).long().tolist()
    if any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError('more layers than propagation slabs')
    return edges


def ideal_layer_screen(ds: torch.Tensor, layers: int) -> torch.Tensor:
    """Depth-block integrated delays spread over the original ASM slabs.

    The lateral grid and phase amplitude are unchanged. The integral over
    each depth block exactly matches GT for every lateral pixel.
    """
    nz = ds.shape[-2]
    edges = layer_edges(nz, layers, ds.device)
    result = torch.zeros_like(ds)
    for a, b in zip(edges[:-1], edges[1:]):
        result[a:b] = ds[a:b].mean(dim=0)
    return result


def compare_delays(ds: torch.Tensor, gt: torch.Tensor, dz: float, pad: int):
    a = ds[:-1, pad:-pad] if pad else ds[:-1]
    b = gt[:-1, pad:-pad] if pad else gt[:-1]
    ta = torch.cat([torch.zeros_like(a[:1]), a.cumsum(0)]) * (dz * 1e6)
    tb = torch.cat([torch.zeros_like(b[:1]), b.cumsum(0)]) * (dz * 1e6)
    return {
        'ds_rel_l2': float((a-b).norm() / b.norm().clamp_min(1e-20)),
        'cumulative_delay_rmse_us': float((ta-tb).square().mean().sqrt()),
        'bottom_delay_rmse_us': float((ta[-1]-tb[-1]).square().mean().sqrt()),
    }


def fixed_reference(images, tr_idx, ho_idx, pad, top_frac):
    tr = images[tr_idx]
    ho = images[ho_idx]
    region = crop(tr, pad)
    power = region.abs().square().mean(0).sqrt()
    threshold = torch.quantile(power.flatten(), 1.0-top_frac)
    mask = torch.zeros_like(power)
    mask[:] = (power >= threshold).float()
    if pad:
        mask = torch.nn.functional.pad(mask, (pad, pad))
    tr_scales = (tr.abs().square() * mask).sum((-2, -1)).sqrt().clamp_min(1e-30)
    ho_scales = (ho.abs().square() * mask).sum((-2, -1)).sqrt().clamp_min(1e-30)
    return mask, tr_scales, ho_scales


@torch.no_grad()
def images_and_field(born, ds, D, all_idx):
    u = born.transmit_fields(ds, all_idx)
    b0 = born.scatter(D[all_idx])
    b0 = born.asp._ifft(born.asp._fft(b0) * born.surface_transfer.conj())
    b = born.asp.adjoint(b0, ds, born.omega_)
    images = (b * (u * born.w_z).conj()).sum(dim=1)
    return images, u


def score(images, tr_idx, ho_idx, mask, tr_scales, ho_scales, pad, truth_abs):
    tr, ho = images[tr_idx], images[ho_idx]
    img11 = crop(images.mean(dim=0), pad)
    return {
        'input_coherence': float(coherence(tr, mask, tr_scales)),
        'holdout_agreement': float(heldout_agreement(
            tr, ho, mask, tr_scales, ho_scales)),
        'image11_abs_corr': float(corr2d(img11.abs(), truth_abs)),
    }


def field_error(u, gt, pad, z0, dz, zmin_mm=5.0, zmax_mm=40.0):
    lo = max(0, int(np.ceil((zmin_mm*1e-3-z0)/dz)))
    hi = min(u.shape[-2], int(np.floor((zmax_mm*1e-3-z0)/dz))+1)
    a = u[..., lo:hi, pad:-pad] if pad else u[..., lo:hi, :]
    b = gt[..., lo:hi, pad:-pad] if pad else gt[..., lo:hi, :]
    return float((a-b).norm() / b.norm().clamp_min(1e-20))


@torch.no_grad()
def run_one(sid, args, device):
    sample = torch.load(DATA_ROOT / 'shards' / f'{sid}.pt',
                        map_location='cpu', weights_only=False)
    cfg, meta = corrected_config(args.config, sample, args.n_freq)
    born = BornModel(padded_meta(meta, args.pad, cfg.grid.dx),
                     cfg.grid.nx+2*args.pad, cfg.grid.nz,
                     cfg.grid.dx, cfg.grid.dz, cfg.physics.c0,
                     eps=cfg.physics.eps_evanescent,
                     spreading=cfg.physics.spreading).to(device)
    D = rf_to_D(sample['rf'].to(device), meta)
    true_ds = embed(sample['delta_s'].to(device), args.pad)
    truth_abs = sample['m'].abs().to(device)
    all_idx = torch.arange(cfg.acq.n_angles, device=device)
    tr_idx = torch.as_tensor(meta.train_idx, device=device)
    ho_idx = torch.as_tensor(meta.hold_idx, device=device)
    rows = []

    gt_images, gt_u = images_and_field(born, true_ds, D, all_idx)
    if args.matched_born:
        m_gt = embed(sample['m'].to(device), args.pad)
        D_matched = born.forward(m_gt, true_ds, gt_u)
        matched_norm = D_matched.norm().clamp_min(1e-20)
        gt_matched_image = born.adjoint(D_matched, gt_u, true_ds)
        gt_image_norm = gt_matched_image.norm().clamp_min(1e-20)
    zero = torch.zeros_like(true_ds)
    uniform_images, uniform_u = images_and_field(born, zero, D, all_idx)
    mask, tr_scales, ho_scales = fixed_reference(
        uniform_images, tr_idx, ho_idx, args.pad, args.top_frac)

    def record(name, ds, images, u, extra=None):
        row = {'sample': sid, 'case': sample['metadata']['case'],
               'method': name,
               **score(images, tr_idx, ho_idx, mask, tr_scales,
                       ho_scales, args.pad, truth_abs),
               **compare_delays(ds, true_ds, born.dz, args.pad),
               'field_rel_l2': field_error(u, gt_u, args.pad, born.z0, born.dz)}
        if extra:
            row.update(extra)
        if args.matched_born:
            modeled = born.forward(m_gt, ds, u)
            reconstructed = born.adjoint(D_matched, u, ds)
            row.update({
                'matched_rf_rel_l2': float((modeled-D_matched).norm()/matched_norm),
                'matched_image_overlap': float(
                    (reconstructed*gt_matched_image.conj()).sum().abs()
                    / (reconstructed.norm()*gt_image_norm).clamp_min(1e-20)),
                'matched_image_abs_corr': float(corr2d(
                    crop(reconstructed, args.pad).abs(),
                    crop(gt_matched_image, args.pad).abs())),
            })
        rows.append(row)

    record('gt_speed_asm', true_ds, gt_images, gt_u)
    record('uniform', zero, uniform_images, uniform_u)
    del uniform_images, uniform_u
    for k in args.layers:
        ideal = ideal_layer_screen(true_ds, k)
        ideal_images, ideal_u = images_and_field(born, ideal, D, all_idx)
        record(f'ideal_K{k}', ideal, ideal_images, ideal_u)
        del ideal_images, ideal_u
        if args.controlled:
            raw, bulk, projection = projected_truth_screen(
                true_ds, k, args.controls, born.dz, args.limit_us,
                args.pad, born.z0, args.bulk_limit_us, True)
            ds = effective_ds(raw, born.nz, born.nx, born.dz,
                              args.limit_us, args.pad, bulk, born.z0,
                              args.bulk_limit_us)
            controlled_images, controlled_u = images_and_field(
                born, ds, D, all_idx)
            record(f'controlled_K{k}', ds, controlled_images,
                   controlled_u, projection)
            del controlled_images, controlled_u
    return rows


def summarize(rows):
    methods = list(dict.fromkeys(r['method'] for r in rows))
    keys = ('input_coherence', 'holdout_agreement', 'image11_abs_corr',
            'ds_rel_l2', 'cumulative_delay_rmse_us',
            'bottom_delay_rmse_us', 'field_rel_l2',
            'matched_rf_rel_l2', 'matched_image_overlap',
            'matched_image_abs_corr',
            'saturated_control_fraction')
    return {method: {k: float(np.mean([r[k] for r in rows
                                      if r['method'] == method and k in r]))
                     for k in keys if any(k in r for r in rows if r['method'] == method)}
            for method in methods}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/l11_ultrawave_500_11angle.yaml')
    p.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--n-freq', type=int, default=64)
    p.add_argument('--layers', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--pad', type=int, default=32)
    p.add_argument('--controls', type=int, default=48)
    p.add_argument('--limit-us', type=float, default=.2)
    p.add_argument('--bulk-limit-us', type=float, default=2.)
    p.add_argument('--top-frac', type=float, default=.2)
    p.add_argument('--controlled', action='store_true')
    p.add_argument('--matched-born', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    max_n = {'train': 400, 'val': 50, 'test': 50}[args.split]
    if not (1 <= args.count <= max_n):
        p.error(f'count must be 1..{max_n}')
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    args.out.mkdir(parents=True, exist_ok=True)
    ids = [f'{args.split}_{i:03d}' for i in
           np.linspace(0, max_n-1, args.count).round().astype(int)]
    results = []
    started = time.monotonic()
    with (args.out / 'results.jsonl').open('w') as f:
        for i, sid in enumerate(ids, 1):
            rows = run_one(sid, args, device)
            for row in rows:
                f.write(json.dumps(row) + '\n')
            f.flush()
            results.extend(rows)
            print(json.dumps({'completed': i, 'total': len(ids), 'sample': sid,
                              'elapsed_s': time.monotonic()-started,
                              'scores': {r['method']: {
                                  'hold': r['holdout_agreement'],
                                  'image11': r['image11_abs_corr'],
                                  'field_error': r['field_rel_l2']}
                                  for r in rows}}), flush=True)
            (args.out / 'summary.json').write_text(json.dumps({
                'args': vars(args) | {'out': str(args.out)},
                'ids': ids[:i], 'summary': summarize(results)}, indent=2)+'\n')


if __name__ == '__main__':
    main()

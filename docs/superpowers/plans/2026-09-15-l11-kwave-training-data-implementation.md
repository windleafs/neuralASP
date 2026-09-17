# L11 k-Wave Training Data Implementation Plan

Date: 2026-09-15

1. Add a backward-compatible `dual_scale` preset to the OA-Breast acoustic
   builder: literature mean sound speeds, visibility density scatterers, and no
   random sound-speed scatterers.
2. Add `configs/l11_kwave.yaml`, an L11 shard dataset loader, and train/eval
   source selection while preserving all existing defaults.
3. Add a resumable generator that selects deterministic case/slice manifests,
   caches homogeneous reference RF, simulates heterogeneous RF, creates coarse
   truth maps and `D`, validates samples, and atomically saves shards.
4. Add conversion/loader tests and run the existing test suite.
5. Generate one pilot sample; verify tensors, RF-to-D identity, and one model
   forward/loss/backward/optimizer step on an A6000.
6. Generate the remaining 63 samples on two idle GPUs using disjoint shard
   assignments, then consolidate and validate the exact 48/8/8 split.
7. Copy the final manifest and validation summary to the user-facing outputs and
   report data paths, size, runtime, and smoke-test results.


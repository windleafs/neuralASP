# L11 k-Wave Training Data for neural_asp

Date: 2026-09-15

## Goal

Adapt `neural_asp` to the validated L11 acquisition and generate 64 k-Wave
samples for initial training. The network input remains channel RF rather than
B-mode images. Each sample must contain traceable acoustic truth and satisfy the
existing `rf`, `D`, `delta_s`, `m`, and `c` contract.

## Acquisition Contract

- Three plane waves at -8, 0, and +8 degrees.
- 192 receive elements at 0.2 mm pitch.
- 7.5 MHz transmit pulse.
- 4.0--7.5 MHz receive band.
- 40 MHz real RF with 2401 samples.
- Per-sample RF shape `[3, 192, 2401]` in `[angle, element, time]` order.
- Frequency data `D` uses the existing `e^{-i omega t}` convention and is
  computed only through `common.rf_to_D`.

## Model Grid and Configuration

A new `configs/l11_kwave.yaml` leaves `configs/default.yaml` unchanged.

- Fine model grid: `nz=216`, `nx=192`, `dz=dx=0.2 mm`.
- Coarse factor: 4, giving a `54 x 48` slowness grid.
- Analysis-band center: 5.75 MHz.
- Fractional analysis bandwidth: 0.608695652, representing 4.0--7.5 MHz.
- RF sampling frequency: 40 MHz; `n_t=2401`.
- Requested `n_freq=64`; the existing integer-stride selector yields 53
  uniformly spaced bins from 4.015 to 7.480 MHz.
- `ds_max=5e-5 s/m`, sufficient for the tissue speed range.
- Batch size: 1.
- With three angles and `holdout_stride=3`, -8 and 0 degrees are training
  angles and +8 degrees is held out for angular prediction validation.

## Dual-Scale Acoustic Medium

Each sample uses OA-Breast anatomy on the existing 0.05 mm k-Wave grid.

1. The macroscopic component assigns the literature tissue-average sound speed
   to fat, fibro-glandular tissue, skin, vessels, and gel. Tissue fractions use
   the existing 0.25 mm partial-volume smoothing. No random sound-speed
   scatterers are added.
2. The microscopic component uses the validated weakly smoothed density
   scatterers: relative RMS 1.5% for gland, 0.3% for fat, 0.8% for skin, and
   0.15% for vessels. Each sample has a unique random seed.
3. Tissue attenuation and mean density remain tissue dependent.
4. The actual propagation sound-speed map is the source of the `c` and
   `delta_s = 1/c - 1/1540` labels.

This retains heterogeneous propagation for the slowness branch while avoiding
the high-frequency random sound-speed phase screen that obscured anatomy in the
original B-mode experiment.

## Supervision Maps

The 0.05 mm simulation maps are cropped to the 38.4 mm aperture and 43.2 mm
depth, then reduced by 4 x 4 cells to the 0.2 mm model grid.

- `c`: area-averaged macroscopic sound speed, float32 `[216, 192]`.
- `delta_s`: computed from the reduced `c`, float32 `[216, 192]`.
- `m`: a signed high-frequency acoustic-impedance perturbation derived from the
  fine-grid density and impedance fields, reduced to the model grid and
  RMS-normalized to `model.m_rms_ref=0.2`, complex64 `[216, 192]` with a zero
  imaginary component in this dataset version.

The `m` target is an approximate Born reflectivity label. Tissue labels and
B-mode pixels are never inserted into it. RF-consistency loss remains the final
constraint linking `m` to the k-Wave measurements.

## Dataset Composition

The split contains exactly 64 samples:

- Train 48: 24 slices from `Neg_07_Left` and 24 from `Neg_35_Left`.
- Validation 8: four additional slices from each training patient, separated
  from training slices by at least 2 mm in elevation.
- Test 8: all from the unseen `Neg_47_Left` patient.

Candidate slices must contain sufficient breast area and both gland and fat in
the L11 field of view. Selected samples store case name, native Z index,
scatter seed, acoustic parameters, and source-file identity. Backup candidates
are preselected so a failed simulation can be replaced without changing split
sizes.

## Generator Architecture

The implementation adds an L11 generator and loader without replacing existing
synthetic or independent-Helmholtz datasets.

1. A candidate scanner reads only required HDF5 slices and constructs the
   deterministic 48/8/8 manifest.
2. A medium builder creates the dual-scale medium for one manifest entry.
3. A reference cache computes homogeneous-medium RF once for each of the three
   transmit angles and reuses it for every sample with identical geometry.
4. A sample simulator runs only the heterogeneous propagation, subtracts the
   cached reference, band-limits/resamples to 40 MHz, and writes a temporary
   shard.
5. A converter transposes RF to `[angle, element, time]`, computes `D` through
   `rf_to_D`, constructs truth maps, validates them, and atomically promotes the
   shard to complete status.
6. `L11KWaveDataset` loads the split tensors for `train.py` and `eval.py`.
7. Training and evaluation gain a `l11_kwave` source option; existing defaults
   remain unchanged.

The data root is
`/data/zhuangyang/NumerialBreastPhantoms/l11_neural_asp_64/`. Source code and
configuration remain in `/home/zhuangyang/fmmodel/neural_asp/`.

## Validation and Pilot Gate

Before bulk generation, one pilot sample must pass all checks:

- exact expected shapes and dtypes;
- all values finite and within declared sound-speed/slowness bounds;
- `D` agrees with a fresh `rf_to_D(rf)` calculation;
- RF energy is nonzero in the expected arrival window;
- `m` has the configured RMS normalization;
- one complete `neural_asp` forward pass, loss calculation, backward pass, and
  optimizer step succeeds with finite gradients on an A6000.

Only after the pilot gate passes may the remaining 63 samples run.

## Recovery and Storage

- Every sample is an independent shard with a manifest status.
- Writes use a temporary filename followed by atomic rename.
- Restart skips validated complete shards and resumes incomplete work.
- A failed slice is logged and replaced by a preselected backup from the same
  split and case policy.
- The reference cache includes acquisition parameters in its key and is rejected
  if any geometry, timing, angle, or medium-reference parameter changes.
- Existing acoustic media, B-mode outputs, synthetic caches, and full-wave test
  caches are never overwritten.
- Expected storage is approximately 0.5--1.5 GB; expected generation time is
  approximately 2--3 hours with reference reuse.

## Completion Criteria

- Exactly 48 train, 8 validation, and 8 test samples validate successfully.
- The dataset loader reproduces the declared contract.
- Existing tests continue to pass.
- New loader/conversion tests pass.
- The pilot forward/backward smoke test passes.
- No full training run starts automatically; only the explicit smoke step is
  performed after generation.

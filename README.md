# Neural Angular-Spectrum Propagation Imaging (first-version prototype)

PyTorch prototype validating the closed loop

```
per-angle x per-element RF/IQ
   -> neural operator: coarse, bounded, smooth background slowness perturbation ds
   -> differentiable heterogeneous angular-spectrum propagation (split-step Fourier)
   -> transmit/receive Born imaging before receive summation
   -> complex image I / envelope, predicted RF, trained with RF data consistency
```

The first-version goal is **not** full waveform inversion: it is to check that a
differentiable propagation model can correct refraction/diffraction *before*
receive summation, without pre-segmenting isoplanatic patches.

## Layout

```
configs/default.yaml        all physics / acquisition / model / training knobs
common.py                   config, metadata, RF<->D, IQ demodulation
physics/angular_spectrum.py split-step angular spectrum (forward / march_up / adjoint)
physics/imaging.py          Born forward F_eta, exact adjoint F_eta^H, DAS baseline
models/encoder.py           theta-shared RF/IQ aperture convolutions
models/neural_operator.py   delay-integration (coordinate query) + attention + FNO -> ds
models/pipeline.py          ImagingPipeline: eta branch, unrolled m estimation, outputs
data/synthetic.py           random phantoms + Born simulation (training data)
data/fullwave.py            independent 2-way Helmholtz FD solver (test-only!)
train.py                    staged training (m -> eta -> joint)
eval.py                     metrics + figures, synthetic or full-wave test set
tests/                      unit tests (physics, adjoint identity, gradients)
run_demo.sh                 one-command demo
```

## Dataset sample schema (per-sample contract)

Each sample is a plain dict of tensors (`data/synthetic.py`,
`SyntheticUSDataset.__getitem__`).  Symbolic sizes refer to
`configs/default.yaml` (demo values: `n_theta=16, n_e=64, n_t=1024,
n_freq=51, nz=100, nx=128`; `B` = batch size).

| key | shape (per sample) | shape (batched) | dtype | meaning |
|---|---|---|---|---|
| `rf` | `[n_theta, n_e, n_t]` | `[B, n_theta, n_e, n_t]` | float32 | real band-limited RF, t=0 at transmit; **the only field the pipeline needs at inference** |
| `D` | `[n_theta, n_freq, n_e]` | `[B, n_theta, n_freq, n_e]` | complex64 | band data `(F(m)+noise)*W` on `meta.freqs`; equals `rf_to_D(rf)` exactly (used as the loss target) |
| `delta_s` | `[nz, nx]` | `[B, nz, nx]` | float32 | true slowness perturbation `1/c - 1/c0`, fine grid; supervision for `L_c` (simulation only) |
| `m` | `[nz, nx]` | `[B, nz, nx]` | complex64 | true complex scattering image, RMS-normalized to `model.m_rms_ref`; supervision for `L_I` (simulation only) |
| `c` | `[nz, nx]` | `[B, nz, nx]` | float32 | true sound speed `1/(1/c0 + delta_s)`; evaluation only (`c_rmse`) |

Axis-order conventions: images are `[z, x]` (x is the FFT / last axis);
frequency-domain data are `[theta, freq, element]`; `rf` is
`[theta, element, time]`.  For the full-wave test set the truth scatterers
are weak `dc` maps and the key is `m_ref` (same shape/dtype as `m`); its `D`
is raw scattered data (windowed by `eval.py` before comparison).

Dataset-level shared metadata (`common.build_meta(cfg)`, not per sample):
`angles_deg [n_theta]` (deg, increasing), `xe_coords [n_e]` (m; array
centered on the grid), `freqs [n_freq]` (Hz, positive band), `win [n_freq]`
(raised cosine), `band_idx [n_freq]` (FFT bin indices), `train_idx` /
`hold_idx` (holdout: `i % holdout_stride == holdout_stride - 1`), and
scalars `n_t, fs, f0, bandwidth, dec, fs_iq, n_elements`.

Conventions any generator must respect:

* time convention `e^{-i omega t}`: `D = conj(FFT(rf)[band_idx])`
  (`common.rf_to_D`); a scatterer with two-way time `t0` peaks in `rf` at
  `t = t0` (not `T - t0`).
* the band window `W` is part of the forward model: predictions are compared
  against `(F(m) + noise) * W`, never against unwindowed spectra.
* noise: complex Gaussian with sigma = RMS(F(m)) * 10^(-snr_db/20),
  `snr_db = train.noise_snr_db` (25 dB in the demo).
* scales: `m` RMS = `model.m_rms_ref` (0.2); `|delta_s| <= model.ds_max`
  (3e-5 s/m).
* lateral-flip augmentation (training only): reverse angle order and element
  order and flip `rf`, `D`, `delta_s`, `m` along x (see `train.py`).

## Physics in one page

Time convention e^{-iwt}. The angular-spectrum half-step transfer is
`H(kx) = exp(i kz dz/2)`, `kz = sqrt((w/c0)^2 - kx^2 + i eps)`, with `+i eps`
making evanescent components decay. One slab step is

```
u(z+dz) = H_{dz/2} [ exp(i w ds(x,z) dz) . H_{dz/2}[u(z)] ]
```

with `ds` the *slowness* perturbation (the network estimates ds, not c; c_hat
= 1/(s0 + ds)). The same transfer functions continue an up-going field to the
surface (`march_up`), and the conjugated/reversed steps form the exact
Hermitian adjoint (`adjoint`) used for back-migration and gradients.

Born forward model per transmit angle theta and frequency w:

```
d(theta,e,w) = S_e sum_z P_{z->0}[ w(z,w) m(x,z) u_theta^tx(x,z,w) ]
```

with S linear interpolation at the elements, w(z,w) a 2D far-field spreading
weight, u_theta^tx the downward-marched steered plane wave. The adjoint image
(correlation imaging condition) is

```
(F^H d)(r) = sum_{theta,w} conj(w u_theta^tx(r,w)) b(r,w),   b = back-marched data
```

reported here with an optional diagonal illumination normalization
(`model.illum_compensate`); the plain sum is the default in the tests.

The scattering image m is estimated by 1-3 unfolded data-consistency
iterations (learned complex prox P_psi, learned steps gamma_k).  A per-sample
RF-consistency refinement of the slowness on the *training* angles is
implemented (`pipeline.refine_delta_s`, spec 4.5 constrained eta branch) but
disabled by default at v1 scale (see limitations).

## Install / run

```bash
# python >= 3.10, torch >= 2.0 (CUDA optional), numpy scipy matplotlib pyyaml pytest
PYTHON=/path/to/python ./run_demo.sh          # tests -> staged training -> eval
# or step by step:
python -m pytest tests -q                     # 14 unit tests
python train.py --stage all --out demo        # m -> eta -> joint (~20 min GPU)
python eval.py --ckpt runs/demo/joint.pt      # metrics + figures
python eval.py --ckpt runs/demo/joint.pt --source fullwave   # independent solver
```

## Results (demo run, `runs/demo`)

Unit tests: **14/14 passed** (angular-spectrum physics incl. plane-wave /
evanescent / Fresnel checks, exact adjoint identity <F m, d> = <m, F^H d>,
autograd-vs-manual-adjoint and finite-difference gradient checks, end-to-end
gradient flow through every branch).

Synthetic test set (4 phantoms, SNR 25 dB; full protocol in
`runs/demo/eval/metrics.json`):

| metric | DAS (uniform c0) | adjoint image, uniform | ours (NO + hetero) |
|---|---|---|---|
| image correlation with truth | 0.178 | 0.488 | 0.484 |
| RF residual, training angles | n/a (no forward model) | **0.318** | 0.324 |
| RF residual, held-out angles | n/a | **0.322** | 0.329 |
| background c RMSE [m/s] | 26.2 (= true perturbation rms) | 26.2 | 29.5 |

Key observations (first-version honesty):

* The differentiable wave-propagation imaging condition (correlation of the
  back-migrated receive field with the transmit field, before receive
  summation) improves image correlation over DAS from 0.18 to ~0.49 and
  produces RF predictions that track the measured echoes (`fig_rf.png`).
* An *oracle* experiment (feeding the true slowness) shows that, at the v1
  unrolling depth (3 iterations), correct-vs-uniform background changes the
  holdout RF residual by only ~0.5% (0.3200 vs 0.3216): a free complex
  scattering image absorbs slow-background errors (eta-m crosstalk), so RF
  consistency alone barely separates the two at this depth. With a deep CG
  solver (x40) the oracle gap grows to ~2.4% but remains small.
* Consequently the v1 neural operator's data->ds regression (128 training
  phantoms) does not yet beat the uniform prior on this synthetic task, and
  per-sample RF-consistency refinement of ds overfits the crosstalk (kept
  off by default; see `eval.refine_steps`). This is the main v1 limitation
  and the roadmap item: stronger/sparser m priors, deeper unrolling, and
  orders of magnitude more training phantoms are needed before the
  estimated background pays off in RF prediction.

Independent full-wave test (2-way FD-Helmholtz solver, no shared code with
the angular spectrum; per-frequency gain calibration applied identically to
all methods): `runs/demo/eval_fw/` -

| metric | DAS | uniform | ours |
|---|---|---|---|
| image correlation with true scatterers | 0.022 | 0.172 | 0.173 |
| calibrated holdout RF residual | n/a | 0.9267 | 0.9263 |

The adjoint wavefield imaging transfers to genuinely independent two-way
physics (scatterer positions recovered, DAS fails); the Born model explains
only part of the full-wave amplitude (multiples + source-convention gaps),
and - consistent with the synthetic oracle study - the estimated background
does not yet separate from the uniform prior.

## Staged training

1. `m`: true propagation, train only P_psi / gamma (data consistency + m supervision).
2. `eta`: neural operator trained with slowness supervision + RF consistency.
3. `joint`: everything fine-tuned, slowness supervision weight x0.2.
4. evaluation adds per-sample eta refinement (RF consistency, training angles).

## Validation / tests

* plane wave acquires exactly exp(i kz L); evanescent wave decays as exp(-|kz| L)
* constant-ds screen accumulates exactly exp(i w ds L) (on-axis, exact)
* Gaussian aperture matches the analytic Fresnel integral (<5e-3 rel.)
* adjoint identity <F m, d> = <m, F^H d> to ~1e-12 (double precision)
* autograd gradient of ||F m - d||^2 equals the manual adjoint (and finite
  differences agree); gradients also flow through ds (phase screens)
* pipeline end-to-end backward populates every parameter group

## First-version limitations (explicit)

* No attenuation inversion (fixed absorption-free propagation), no per-element
  gain / system transfer estimation, single-scattering (Born) only.
* **The eta branch does not yet pay off at v1 scale**: the neural operator's
  data->ds map generalizes weakly across phantoms (memorizes ~100 training
  samples), and RF-consistency refinement of ds overfits the eta-m crosstalk.
  The honest oracle measurement above shows the headroom at this unrolling
  depth is <1%; the mechanism is validated, the estimator is the gap.
* eta-m crosstalk: an unconstrained complex m can mimic slow-background
  errors; separating them needs sparse/structured m priors, deeper
  unrolling (spec allows 1-3; v1 uses 3), or joint regularization.
* ds is an *effective* background: not quantitative sound speed, no guarantee
  of uniqueness; strongly scattering / multi-path media are out of scope.
* Phase-screen split step is a one-way (paraxial-ish) approximation: no
  reflections, no mode conversion, lateral wrap-around via FFT (visible as
  edge artifacts in the adjoint images).
* The independent full-wave test uses a 4th-order FD Helmholtz solver with a
  matched layer; it still has ~1-2% numerical dispersion and its own
  source-convention mismatch (handled by per-frequency gain calibration).
* 2D, single-sided linear array, plane-wave transmit only.

## Roadmap (towards the claim actually paying off)

1. Sparser m parameterization (learned complex shrinkage is weak) - e.g.
   spike-conv / learned ISTA-Net+ priors, or m supported on detected point
   tracks; this directly attacks the crosstalk.
2. Deeper unrolling (5-10 steps) or CG inner solver with prox outer loop;
   the CG experiment shows the oracle gap grows with fit depth.
3. 10^3-10^4 training phantoms (generation is cheap) for the NO; plus
   shot-consistency regularization across angles.
4. Then re-enable per-sample eta refinement (already implemented:
   `refine_delta_s`) from the stronger NO initialization.

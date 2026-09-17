# Implementation plan

1. Keep the k-Wave 64 dataset, old config and running GPU0 training unchanged.
2. Add independent 11-angle config and a compatible l11_fullwave dataset alias.
3. Reuse validated UltraWave benchmark operator, receiver timing, absorption fit and state reset; never import torch in the GPU simulation process.
4. Partition eligible anatomy into disjoint train/val elevation blocks with >=10 native slice separation; prioritize 500 unique case/slice pairs and report duplicates explicitly if required.
5. Generate and fingerprint a dedicated 11-angle UltraWave homogeneous reference; use one reference operator and change source data between angles.
6. Generate pilot raw RF, repeat a propagation to verify reset, naturally beamform 11 angles, pack and run truth/net backward smoke tests.
7. Only after pilot passes, start two atomic/resumable raw workers on idle GPUs1/2 and an error-aware finalizer.
8. Finalizer packs and validates all500, runs formal staged one-step training and all50 test eval, produces a provenance report and writes READY only after all gates pass. No full training.
9. Save report and commands as user-facing outputs. Clearly distinguish generated/running/ready states.

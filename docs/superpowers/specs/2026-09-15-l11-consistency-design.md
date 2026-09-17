# L11 consistency repair

The user approved the diagnostic repair sequence by requesting implementation.

Use explicit physical grid origins, element centre and per-angle pulse reference times. Incorporate timing and source response into Born transmit fields so its forward, adjoint, unrolled residuals and refinement share the same measurement operator. Include the small homogeneous gap between the receiver surface and the first truth pixel. Correct plane-wave travel-time queries and compensate the carrier phase before summing IQ data.

Preserve legacy data/config files; provide a repaired configuration using the continuous frequency band. Recompute D from original RF at dataset load, rather than overwriting archived shards. Estimate one fixed frequency response from training samples and input angles only, record provenance, and embed it in checkpoint buffers. This calibration remains an approximation because m is a proxy label.

Train m with fixed true sound speed; train eta with frozen prox and sound-speed supervision; use conservative RF weight and full sound-speed weight in joint. Validate all validation samples at intervals; select best checkpoint by sound speed RMSE for eta/joint and RF residual for m; compare initialization at step -1 so joint can fall back. Store config and operator version in each checkpoint. Keep raw, uncalibrated held-out predictions distinct from diagnostics fitted to held-out targets.

Verify mathematical adjoints, carrier/timing point localization, D compatibility, finite gradients, frozen stage parameters, full validation accounting and checkpoint selection. Run bounded real-data smoke training and evaluate selected checkpoints. Full convergence claims require subsequent longer training and broader independent cases.

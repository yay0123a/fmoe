# Six-stage roadmap

## Stage 1 — Foundation and contracts (completed)

- Package, configuration, registries, logging and reproducibility.
- Typed task, modality, batch, output and diagnostic contracts.
- Typed task-homogeneous dataset adapters and engineering probe batches.
- Reversible arbitrary-size padding and debug-only differentiable forward.
- Checkpoint, CLI and test infrastructure.
- Permanent architecture, interface and third-party attribution documents.

## Stage 2 — Final backbone and frequency foundation (completed)

- RGB, grayscale and infrared stems.
- Shared source encoder and independent fused feature stream.
- Four-scale custom encoder-decoder, maximum 1/8 downsampling.
- Haar DWT/IDWT, Fourier utilities, AFNO2D, FDConv and cross-frequency
  interaction.
- Exact-size interpolation decoder, real spectral diagnostics, odd-size and
  source-swap acceptance coverage.

## Stage 3 — Functional MoE and joint router (completed)

- Common, low-frequency, detail, semantic and infrared-saliency experts.
- Feature/task/spectrum/modality/auxiliary routing evidence.
- Global Top-2 routing, availability masks, spatial modulation and dense/sparse
  execution parity.
- Typed per-scale diagnostics for logits, probabilities, selected experts,
  branch weights, availability, and spatial gates.

## Stage 4 — Focus, semantic and feedback closure (completed)

- Lightweight focus reliability head.
- Pluggable frozen semantic backend and guide adapter.
- Coarse fusion, guidance pyramid, feedback MoE and task-conditioned refinement.
- Source-symmetric lightweight MFIF interaction and exact arbitrary-size final
  reconstruction.

## Stage 5 — Training system (completed)

- Complete task-specific loss selection and alternating schedule.
- Gradient coordination, EMA, AMP/DDP and exact resume.
- Availability-aware load balancing and expert specialization objectives.

## Stage 6 — Evaluation and paper evidence (deferred by request)

- Fusion, focus and downstream semantic metrics.
- Expert usage, frequency response and guidance-routing correlations.
- Coarse/final comparisons, ablations, complexity and reproducibility reports.

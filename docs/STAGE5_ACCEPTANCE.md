# Stage 5 acceptance

Stage 5 keeps the Stage 1–4 network unchanged and makes it trainable as one
checkpoint across VIF, MFIF, and semantic-oriented VIF.

Implemented contracts:

- `LossContext` and structured `LossOutput`, including components, weighted
  components, diagnostics, and explicit skipped reasons.
- VIF intensity/gradient/source-max SSIM/color/coarse losses; MFIF
  reconstruction/gradient/SSIM/coarse and focus losses; SEG CE/Dice/coarse,
  optional improvement, and boundary alignment.
- Expert frequency specialization, availability-aware importance/hard-load
  balance, optional entropy target, IR saliency preservation, cross-task
  low-frequency consistency, residual magnitude, and range penalties.
- Explicit exhaustive parameter groups, task-specific freezing, phase loss
  multipliers, router temperature schedule, AdamW decay/no-decay splitting,
  warmup/cosine/step/constant schedulers, accumulation and gradient clipping.
- FP32/FP16/BF16 execution, EMA, optional EWC, DDP initialization/wrapping,
  gradient and router-collapse diagnostics.
- Atomic checkpoints with exact optimizer/scheduler/scaler/EMA/task sampler,
  data cursor, phase/router/EWC and Python/NumPy/Torch/CUDA RNG restoration.

The deterministic fixture validates engineering behavior only and is not a
scientific dataset. Implement a `FusionDatasetAdapter` yielding the same
`FusionBatch` contract before research training.

Run the compact 20-step acceptance profile:

```bash
python train.py \
  --config configs/experiments/stage5_training_smoke.yaml \
  --device cpu \
  --max-steps 20
```

Resume:

```bash
python train.py \
  --config configs/experiments/stage5_training_smoke.yaml \
  --resume runs/stage5_training_smoke/checkpoints/latest.pt \
  --max-steps 20
```

Stage 6 evaluation metrics, ablations, benchmarks, and paper plots are
intentionally not implemented in this delivery.

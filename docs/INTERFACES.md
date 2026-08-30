# Interface contract

## Input

`FusionBatch` contains two `SourceBatch` values, a homogeneous `TaskType`,
sample identifiers, and optional supervision. Each source carries an explicit
`ModalityType`; source A is never assumed to be visible and source B is never
assumed to be infrared.

Canonical image layout is `float32 [B,C,H,W]` in `[0,1]`. RGB modalities require
three channels and gray/infrared modalities require one channel.

## Output

`FusionOutput` always contains the final fused tensor and explicit task. It may
also contains coarse fusion, spectral statistics, router diagnostics, and
optional focus/semantic outputs. Stage 4 returns distinct `coarse` and final
`fused` images, initial plus feedback router diagnostics, and predicted
auxiliary evidence. Values are never fabricated from supervision targets.

## Configuration

All final modules are represented in the shared schema. Experiment YAML may
override base YAML recursively. Unknown fields are rejected rather than ignored.
The fully resolved configuration is written to each run directory.

## Future adapters

Real datasets implement `FusionDatasetAdapter`; semantic models implement
`SemanticGuideBackend`; focus guidance implements `FocusGuideBackend`; final
backbones implement `FusionBackbone`. Implementations must preserve these
contracts rather than adding task-specific branches to training scripts.

## Stage 2 backbone output

`BackboneOutput` carries named `FeaturePyramid.s1/s2/s3/s4` values for the
source-A, source-B, and fused four-scale pyramids,
the full-resolution decoder feature, exact-size sigmoid image output, spectral
statistics, and non-scientific debug metadata. Pyramids use the reversibly
padded size; the public fused image is cropped to the original height and width.

## Stage 3 routing output

`FunctionalExpertPool` owns the five experts in stable order and exposes validity.
`TaskFrequencyMoEBlock` returns `MoEOutput` (feature, scaled residual, full
`RouterOutput`, spatial gates, optional expert outputs, and diagnostics); it remains
iterable as `(feature, diagnostics)` for earlier callers.

The fixed five encoder MoE blocks each contribute a `RouterDiagnostics` value containing
`[B,E]` logits, normalized probabilities and availability mask; `[B,K]` Top-k
indices and renormalized weights; `[B,5]` evidence-branch weights; and optional
`[B,E,H,W]` spatial gates, entropy, importance, hard load, and auxiliary
availability descriptors. Diagnostics validate normalization and reject any
selection of an unavailable expert. No ground-truth focus or semantic signal is
accepted by the router.

## Stage 4 auxiliary and feedback output

Focus produces distinct sigmoid reliability and normalized selection maps for every
task by default. Semantic probabilities are
`[B,19,H,W]`; normalized uncertainty and boundary are `[B,1,H,W]`. Initial
diagnostic IDs are `s2/s3/s4`, while the independent feedback pass uses
`feedback.s3.moe0/feedback.s2.moe0`. The semantic checkpoint is frozen but
the predictions remain differentiable with respect to `coarse`.
`FusionOutput.aux` exposes the unclamped final image and clamp ratios, while
`FusionOutput.auxiliary` retains typed focus and semantic predictions.

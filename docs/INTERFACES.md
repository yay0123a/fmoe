# Interface contract

## Input

`FusionBatch` contains two `SourceBatch` values, a homogeneous `TaskType`,
sample identifiers, and optional supervision. Each source carries an explicit
`ModalityType`; stems, source evidence and output color handling use this metadata.
The Stage15 three-stage IACF profile uses ordered predictors and expects A=visible,
B=infrared for VIF/SEG. MFIF uses its two RGB sources in dataset order. The legacy
fusion path remains source-symmetric; IACF does not promise swap invariance.

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

`BackboneOutput` carries named `FeaturePyramid.s1/s2/s3` values and an optional
`s4` for the source-A, source-B, and fused three- or four-scale pyramids.
On three-stage models, `s4` is `None`; iteration, length, and dictionary/index
access expose only the three active scales. It also carries
the full-resolution decoder feature, exact-size sigmoid image output, spectral
statistics, and non-scientific debug metadata. Pyramids use the reversibly
padded size; the public fused image is cropped to the original height and width.

IACF only changes the cross-modal fusion slot, before the existing fused stage/MoE.
`source_a`/`source_b` pyramids and router/expert evidence remain unmodified source
encoder outputs. The spatial anchor is corrected by OAF before adding the existing
projected previous-scale feature and applying the existing refine blocks.
`debug["cross_modal"]` contains detached stage scalars. `weight_a`/`weight_b` are
now spatial/batch means (not maps); the existing IR-weight logging still uses them.
Enabled IACF also supplies explicit `weight_a_mean`/`weight_b_mean` and OAF metrics;
relation/cross-update metrics exist only where attention is enabled. No full
relation, attention, candidate or difference feature maps are retained by default.

## Stage 3 routing output

`FunctionalExpertPool` owns the five experts in stable order and exposes validity.
`TaskFrequencyMoEBlock` returns `MoEOutput` (feature, scaled residual, full
`RouterOutput`, spatial gates, optional expert outputs, and diagnostics); it remains
iterable as `(feature, diagnostics)` for earlier callers.

Training resolves one execution policy per optimizer step. It continuously moves from
uniform dense mixing to soft routing and then Top-2 before switching to direct sparse
dispatch. Sparse aggregation writes selected, weighted expert contributions directly
into `[B,C,H,W]`; it does not construct `[B,E,C,H,W]`. Full expert outputs are retained
only when `return_expert_outputs=True` is explicitly requested for debugging.
Run `python tools/benchmark_moe_execution.py --device cuda:0 --batch-size 4` for
the dense-reference versus sparse training microbenchmark.

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

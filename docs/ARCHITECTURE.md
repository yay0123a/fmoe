# Architecture contract

## Research question

Given the same source scene, can a single checkpoint use the requested task,
current frequency state, modality availability, focus reliability, and semantic
importance to select functional processing paths and produce a task-appropriate
fusion result?

## Final data flow

```text
explicit task + source modality identifiers
                    |
modality-specific stems and shared source encoder
                    |
source-A pyramid + source-B pyramid + fused pyramid
                    |
DWT / Fourier frequency foundation
                    |
feature + task + spectrum + modality + auxiliary router
                    |
global Top-2 selection + spatial expert modulation
                    |
common / low / detail / semantic / infrared-saliency experts
                    |
coarse fusion
        +-----------+-----------+
        |                       |
focus reliability       frozen semantic guide
        +-----------+-----------+
                    |
feedback router and feedback MoE
                    |
task-conditioned refinement decoder
                    |
final task-adaptive fused image
```

## Non-negotiable invariants

1. VIF, MFIF, and SEG-oriented fusion do not use three independent backbones.
2. The source encoder is shared after modality-specific stems.
3. Source-specific features and the fused feature pyramid remain available.
4. Experts specialize by function, not by task identity.
5. Only the infrared expert is hard-masked, and only when no IR source exists.
6. Task, feature, spectrum, modality, and auxiliary evidence all participate in
   the final router.
7. Global Top-2 selection precedes sparse expert execution; the spatial gate
   modulates selected experts rather than replacing Top-2 routing.
8. Focus and semantic predictions, never their ground truth, enter the router.
9. Coarse-to-feedback refinement remains a two-pass fusion process.
10. Every scientific diagnostic is returned through typed output contracts.

## Current Stage 4 boundary

`DebugFusionCore` is a small differentiable plumbing fixture. It verifies task
propagation, modality handling, arbitrary-size padding, output contracts, and
checkpointing. It remains isolated and is forbidden in scientific experiments.

`CustomMultiscaleBackbone` is the scientific Stage 2 implementation. Separate
RGB, infrared, and generic-gray stems feed one shared source encoder. At each of
four scales, a shared content scorer fuses the two source features symmetrically;
an independent fused stream propagates and refines that result. Configured
Stage 2 frequency blocks decompose fused features with Haar DWT, refine LL with AFNO,
refine detail bands with FDConv, condition low/high bands bidirectionally, and
reconstruct before an interpolation decoder. Stage 3/4 retain this implementation for
baselines but replace the instantiated processing slots with MoE, avoiding dead
frequency-foundation parameters in their checkpoints.

Stage 3 replaces the processing slots with five independent functional MoE blocks:
S2×1, S3×2, and S4×2. The common expert handles general local
refinement; low-frequency and detail experts operate on complementary Haar
subspaces; the semantic expert uses multi-dilation context; and the IR expert
uses the actual infrared source feature. A joint router combines feature, task,
spectrum, modality, and auxiliary-availability evidence. Its global Top-2
decision is renormalized before spatial gates modulate the selected residuals.
Only the IR expert is masked when neither source is infrared. Dense execution
exists as a numerical reference; sparse execution evaluates each selected
expert only for the samples that routed to it.

Stage 4 keeps that result as `coarse`. For every task by default, a source-symmetric focus head
predicts two reliability maps from the retained source pyramids. A locally
loaded frozen SegFormer-B0 predicts 19-class probabilities, normalized entropy,
and semantic boundaries from the coarse RGB image for all tasks. No supervision
tensor enters either predictor during model forward.

Eight explicit predicted/availability maps are projected to four scales. At S3/S2,
guidance conditions a second set of functional MoE blocks whose parameters do
not alias the initial MoE. Their routers receive real auxiliary predictions.
For MFIF only, fused window queries attend jointly to both shared-encoder source
features before feedback routing; joint key/value attention makes this operation
invariant to source order. This original lightweight module takes architectural
inspiration from GIFNet's shared-feature / interaction / CNN-decoder pattern.

A convolutional top-down decoder applies task-conditioned FiLM at every scale
and predicts a bounded residual over `coarse`, yielding the final `fused` image.
Semantic parameters stay frozen, while gradients through the semantic network
to the coarse image remain available for Stage 5 objectives.

MFIF does not own an expert; it provides task condition and frequency-reliability
evidence. Segmentation does not own an expert; it provides task condition and
semantic-importance evidence. Functional experts remain shared across tasks.

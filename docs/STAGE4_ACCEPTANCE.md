# Stage 4 acceptance record

Stage 4 is a two-pass feed-forward closed loop:

`coarse -> predicted focus/frozen semantic analysis -> S3/S2 feedback MoE ->
task-conditioned residual -> final`.

The shared focus estimator uses S1/S2 features plus locally normalized Sobel and
Laplacian evidence from each original source. It separately returns sigmoid source
reliabilities, source-selection softmax, confidence, and transition boundary, and runs
for all tasks by default. Swapping sources swaps ordered maps without changing
confidence.

The semantic path has generic/null adapter contracts and the supplied local SegFormer
backend. Frozen parameters stay in eval mode and receive no gradients, while the image
path remains differentiable. Probabilities, normalized entropy, and sample-normalized
probability boundaries are available. Coarse guidance runs for all tasks; an optional
final pass defaults to SEG only.

Exactly eight guidance maps form a lightweight four-scale pyramid. Independent
feedback blocks are fixed to `feedback.s3.moe0` and `feedback.s2.moe0`, with a real
encoder-sharing ablation available through configuration. The refinement
decoder uses residual GuidanceConditioners and a TaskFiLM projected from the same
Stage 3 TaskEmbedding. A shared three-convolution residual head uses
`0.25*tanh(.)` with learnable scale initialized to `1e-3`; pre-clamp output and clamp
ratios are retained.

Acceptance evidence: all 12 combinations of four documented sizes and three tasks
passed full-topology forward using the CPU smoke-width profile; the formal
`stage4_feedback_full.yaml` uses `[48,96,192,384]` and also passed VIF/MFIF/SEG CLI
forward plus training backward dry-runs. VIF/MFIF/SEG backward produced finite
nonzero gradients through
the backbone, coarse MoE, focus head, guidance pyramid, feedback MoE, and refinement
decoder; semantic parameter gradients stayed absent; `76 passed` repository-wide.

All Stage 4 switches alter the real graph rather than metadata only: focus operators,
focus/semantic feedback, guidance pyramid, feedback MoE, Task-FiLM, refinement
decoder, final residual, detach behavior, and encoder/feedback MoE sharing.

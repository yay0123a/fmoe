# Stage 3 acceptance record

Stage 3 uses five shared functional experts (common, LL-only AFNO, directional
high-band FDConv, semantic-conditioned FDConv, and real-IR saliency). It does not use
task-owned experts.

Each block combines feature, shared task token, 11-D DWT/FFT spectrum, symmetric
modality/content difference, and learned-null auxiliary evidence. Routing is forced to
FP32, branch weights initialize equally, the IR validity mask is applied before the
full softmax, global Top-2 precedes expert execution, and a source/auxiliary-aware
`2*sigmoid` spatial gate initializes exactly to one. Dense training and sparse
inference paths are numerically equivalent. Diagnostics include entropy, importance,
hard load, availability, full probabilities, selections, and spatial gates.

The optimized encoder placement is fixed to `s2.moe0`, `s3.moe0`, and `s4.moe0`.
All three blocks reuse one TaskEmbedding while retaining independent router
and expert parameters. The public contracts include `FunctionalExpertPool`,
`MoEOutput`, and `TaskFrequencyMoEBlock`; the Stage 2 frequency foundation remains
available in code without becoming an unused resident module in MoE models. The full
repository test suite reports `76 passed`.

The formal Stage 3 config uses the complete `[48,96,192,384]` research profile;
`stage3_functional_moe_smoke.yaml` is reserved for fast regression tests.

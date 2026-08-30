# Stage 2 acceptance record

The scientific backbone follows the supplied Stage 2 contract: modality stem bank,
shared source encoder, independent fusion encoder, contextual symmetric cross-modal
fusion, named four-scale pyramids, 1/8 maximum downsampling, full skip decoder, and
coarse output for arbitrary image sizes.

Frequency support includes orthonormal Haar DWT/IDWT with explicit metadata, unified
orthonormal Fourier helpers, normalized four-band FFT and DWT energies, differentiable
low/mid/high statistics, block-mixing AFNO with FP32 FFT, FDW/KSM/four-band FBM,
bidirectional cross-frequency conditioning, and a structured foundation output.

Acceptance evidence: DWT maximum error `4.77e-7`; normalized FFT/DWT energies sum to
one; Stage 2 VIF/MFIF/SEG forward passes work; input-A/input-B and S2/S3/S4 frequency
diagnostics are retained; the full repository test suite reports `76 passed`.

The formal experiment uses `[48,96,192,384]` with depths `[2,2,4,4]`.
`stage2_frequency_smoke.yaml` is the explicitly named CPU test profile and uses
`[8,16,32,64]`; it changes capacity only, not the implementation path.

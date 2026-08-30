# Design references

The architecture is informed by, but is not a reimplementation of:

- Frequency Dynamic Convolution for Dense Image Prediction.
- Adaptive Fourier Neural Operators for efficient token mixing.
- WF-Diff for explicit wavelet frequency separation and interaction.
- FAPE-IR for semantic/spectral routing and frequency specialization.
- More Than Meets the Eye (DECC) for semantic-pixel coordination.
- Customized Fusion for closed-loop task-aware semantic compensation.
- SegFormer for the frozen semantic guide architecture.
- GIFNet (CVPR 2025) for the high-level pattern of shared low-level features,
  lightweight task interaction, and convolutional fusion decoding.

Project-specific contributions include five functional fusion experts, the
five-source joint router, global Top-2 plus spatial expert modulation, explicit
modality validity masking, focus/semantic feedback, and task-adaptive fusion
within one checkpoint.

# Notices and research references

The frequency operators, functional experts, router, feedback modules, and
training system are original PyTorch implementations informed by the published
descriptions of Frequency Dynamic Convolution, Adaptive Fourier Neural
Operators, WF-Diff, FAPE-IR, DECC, Customized Fusion, SegFormer, and GIFNet.

GIFNet commit `e9c83a66a3f2b634a9d9e30269cde601ce40c524` was reviewed for the
high-level shared-feature, lightweight-interaction, and convolutional-decoder
pattern. No source was copied or ported. BasicSR, Transformers, MMCV, and
MMDetection are not runtime dependencies.

The local `weights/segformer_b0_cityscapes/pytorch_model.bin` file has SHA-256
`027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c`.
Its upstream provenance and redistribution license must be confirmed before
publishing the weight.

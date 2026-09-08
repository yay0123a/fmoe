# TFS-MoE-Fusion

Task-frequency-semantic mixture-of-experts for VIF, MFIF, and
segmentation-oriented image fusion.

## Setup

```bash
conda env create -f environment.yml
conda activate fMoeFusion
```

For an existing CUDA-enabled environment:

```bash
python -m pip install -e '.[dev]'
```

The frozen semantic guide is loaded from
`weights/segformer_b0_cityscapes/`. Pretrained weights and datasets are local
assets and are not tracked by Git.

The SegFormer directory must contain:

```text
weights/segformer_b0_cityscapes/
├── config.json
├── preprocessor_config.json
└── pytorch_model.bin
```

## Local data

```text
data/
├── semantic_rt/
│   ├── rgb/<sample-id>.jpg
│   ├── thermal/<sample-id>.jpg
│   └── labels/<sample-id>.png
├── mfif/semantic_rt/
│   ├── AiF/<sample-id>.jpg
│   ├── dof_stack/<sample-id>/0.jpg
│   ├── dof_stack/<sample-id>/1.jpg
│   └── quantized_depth/<sample-id>.png
└── splits/
    └── semantic_rt_test_uniform_2000_seed3407.txt
```

The manifest contains one sample ID per line. The default manifest contains
2,000 complete IDs sampled uniformly without replacement from the unique union
of `test_day`, `test_night`, `test_hard`, `test_mc`, and `test_mo`, with seed
3407. The default configuration trains the three VIF, MFIF, and SEG tasks
through the real SemanticRT adapter. These IDs must not also be used for an
unbiased test evaluation. MSRS and FMB training adapters are not implemented
yet.

## Train

```bash
python train.py --config configs/default.yaml
```

`training.device` defaults to `cuda:0`. Edit it in `configs/default.yaml`, or
override it for one command with `--device cuda:1` or `--device cpu`.

Engineering smoke test:

```bash
python train.py --config configs/default.yaml --device cpu --dry-run --task vif
```

The dry-run uses one generated engineering probe, disables the external
SegFormer backend, and never enters the training data pipeline. It checks model
forward, loss, backward, and finite gradients without requiring local data or
pretrained weights.

Resume:

```bash
python train.py --resume runs/<experiment>/checkpoints/latest.pt
```

## Fuse images

```bash
python test.py \
  --task vif \
  --checkpoint runs/<experiment>/checkpoints/latest.pt \
  --input-a visible.png \
  --input-b infrared.png \
  --output runs/test/fused.png
```

`--input-a` and `--input-b` may also be directories. Files with matching
stems are fused in pairs. Use `--save-coarse` to save the coarse result beside
the final image.

MFIF defaults to two generic RGB sources. VIF and SEG default to visible RGB
plus infrared grayscale; explicit modalities can be supplied when needed.

`configs/stage6_vif_mfif.yaml` is a fresh 50-epoch MSRS VIF-first training
profile. It schedules only VIF and MFIF, removes SegFormer and the semantic
expert, and uses a soft, visible-anchored directional gradient loss. It uses
one GPU with batch size 2 and two accumulation steps, for an effective batch
size of 4. Start it without `--resume`:

```bash
CUDA_VISIBLE_DEVICES=4 python train.py --config configs/stage6_vif_mfif.yaml
```

The configuration reads `data/msrs/train/{vi,ir}` for VIF and
`data/mfif/msrs/train/{dof_stack,AiF,depth}` for MFIF. Change
`CUDA_VISIBLE_DEVICES` to the GPU ID assigned to this job.

`configs/stage7_adaptive_ir.yaml` preserves positive, thermally hot targets
(such as people and vehicles) in both day and night images while keeping the
non-hot background anchored to visible luminance and chroma. It uses aligned
IR luminance with separate intensity and structure gates: visible highlights
that are both clipped and locally textureless use soft-knee tone compression,
a weak bloom-ring correction, and bounded zero-mean IR detail. Confident IR-only
edges can still enter the signed-gradient and local-SSIM targets. Weak
valid-expert MoE balancing is disabled so functional experts can specialize
without a uniform-usage objective. The Router and MoE forward path remain
enabled. A task-local, physical-evidence-aware starvation floor activates only
after an eligible expert remains below 2% EMA usage for 500 eligible steps; it
turns off again after recovery above 3% and never targets uniform usage. It uses
batch size 4 with one
accumulation step, keeping the effective batch size at 4. Deterministic mode is
disabled to allow cuDNN autotuning; repeat runs may differ numerically even with
the same seed. Set
`experiment.deterministic: true` when strict reproducibility is required:

```bash
CUDA_VISIBLE_DEVICES=4 python train.py --config configs/stage7_adaptive_ir.yaml
```

`configs/stage8_router_ir_evidence.yaml` adds three image-space signals to the
spatial Router for VIF: signed IR advantage, thermal saliency (local contrast
times cross-modal novelty), and IR-only edge confidence. They replace the three
inactive semantic-guide slots in the Stage 7 profile, so the Router remains at
four projected feature groups plus seven scalar maps instead of gaining extra
channels. MFIF receives zero values in these VIF-only slots and continues to use
its focus evidence. At each MoE site, fused, visible, and IR features use the
same canonical projection before comparison. The IR specialist receives signed
`IR - fused` and `IR - visible` feature residuals together with the thermal
saliency and IR-only edge maps. VIF is the primary optimization task. On MFIF
steps, the modality stems, shared backbone, cross-modal fusion, common and IR
experts, shared decoder trunks, site mixture scales, and VIF heads are frozen.
MFIF updates the focus/detail specialists and its output heads; low-frequency,
Router, and site-adapter gradients are reduced to 20%, 25%, and 20%,
respectively. This makes MFIF auxiliary expert-level supervision rather than a
second full-network objective. This is a structural experiment and should be
trained from scratch in its separate run directory:

```bash
CUDA_VISIBLE_DEVICES=4 python train.py --config configs/stage8_router_ir_evidence.yaml
```

With `--task seg`, inference also saves the final fused image's segmentation:

```bash
python test.py \
  --task seg \
  --checkpoint runs/shared_pool_stage5_y_only_feedback/checkpoints/final_ema.pt \
  --input-a data/semantic_rt/rgb/img_02629.jpg \
  --input-b data/semantic_rt/thermal/img_02629.jpg \
  --output runs/test/img_02629.png
```

This produces `img_02629.png`, `img_02629_seg.png` (8-bit grayscale Cityscapes
train IDs 0–18), and `img_02629_seg_color.png` (RGB Cityscapes colors). The ID
image looks almost black in an image viewer; use the color image for inspection.
Directory inputs produce the same three files for each matched pair. Semantic
guidance must be enabled and `model.guidance.semantic.final_pass_policy` must be
`seg_only` or `all`. For mIoU, map SemanticRT ground-truth labels using the
project's `SEMANTIC_RT_TO_CITYSCAPES` mapping and ignore target pixels with ID 255.

## Verify

```bash
ruff check .
pytest -q
python test.py --task vif --dry-run --device cpu
```

Tests that strictly load local pretrained assets are separate:

```bash
pytest -q -m integration
```

Unfinished work is tracked only in [ROADMAP.md](ROADMAP.md). Third-party and
research acknowledgements are in [NOTICE.md](NOTICE.md).

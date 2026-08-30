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

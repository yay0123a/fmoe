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

## Local data

```text
data/
├── msrs/
├── fmb/
├── semantic_rt/
└── mfif/
    ├── msrs/
    ├── fmb/
    └── semantic_rt/
```

Real-data adapters are intentionally deferred. The current trainer uses the
deterministic fixture to verify the training system.

## Train

```bash
python train.py --config configs/default.yaml --device auto
```

Engineering smoke test:

```bash
python train.py --config configs/default.yaml --device cpu --dry-run --task vif
```

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

Unfinished work is tracked only in [ROADMAP.md](ROADMAP.md). Third-party and
research acknowledgements are in [NOTICE.md](NOTICE.md).

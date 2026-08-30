# Future roadmap

Only unfinished work is recorded here. Completed development stages are not
kept as project documentation.

## Real-data training

- Implement dataset adapters for MSRS, FMB, SemanticRT, and the prepared MFIF
  datasets without changing their official internal layouts.
- Validate pair matching, label IDs, train/validation/test splits, augmentations,
  and task-homogeneous batching.
- Allow training configurations to enable or disable individual tasks while
  preserving exact checkpoint resume.

## First training and qualitative review

- Train the initial VIF/SEG checkpoint, then add MFIF to joint training.
- Save visible, infrared/focus-source, coarse, and final images during training.
- Review brightness, color fidelity, infrared saliency, edge artifacts, focus
  transitions, and semantic consistency before adding benchmark automation.

## Evaluation and research evidence

- Add fusion, focus, and downstream semantic metrics after qualitative quality
  is acceptable.
- Add expert-usage, frequency-response, and guidance-routing diagnostics.
- Run coarse-versus-final comparisons, ablations, complexity measurements, and
  reproducibility experiments.

## Release preparation

- Record dataset and pretrained-weight provenance, checksums, and licenses.
- Publish reproducible environment, training configuration, checkpoint, and
  representative fused images.

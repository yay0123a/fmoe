"""TFS-MoE-Fusion consolidated implementation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from tfs_moe_fusion.types import ConfigurationError


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "tfs_moe_fusion"
    seed: int = 3407
    deterministic: bool = True
    output_dir: str = "runs"


@dataclass(slots=True)
class BackboneConfig:
    name: str = "custom_multiscale"
    channels: list[int] = field(default_factory=lambda: [48, 96, 192, 384])
    depths: list[int] = field(default_factory=lambda: [2, 2, 4, 4])
    max_downsample: int = 8
    normalization: str = "group_norm"
    shared_source_encoder: bool = True
    separate_modality_stems: bool = True
    preserve_source_pyramids: bool = True
    preserve_fused_pyramid: bool = True
    dw_kernel_size: int = 3
    mlp_ratio: float = 2.0
    layer_scale_init: float = 1e-6
    drop_path: float = 0.0


@dataclass(slots=True)
class FrequencyConfig:
    enabled: bool = True
    dwt: bool = True
    afno: bool = True
    fdconv: bool = True
    cross_frequency: bool = True
    placements: list[str] = field(default_factory=lambda: ["s2", "s3", "s4"])
    afno_num_blocks: int = 8
    afno_sparsity_threshold: float = 0.01
    afno_hard_thresholding_fraction: float = 1.0
    fdconv_kernel_size: int = 3
    fdconv_kernel_num: int = 8
    residual_scale: float = 0.1
    statistics_detach: bool = False
    force_fp32_fft: bool = True
    frequency_bands: list[float] = field(
        default_factory=lambda: [0.0, 0.0625, 0.125, 0.25, 0.5]
    )


@dataclass(slots=True)
class MoEConfig:
    enabled: bool = True
    experts: list[str] = field(
        default_factory=lambda: [
            "common",
            "low_frequency",
            "detail",
            "semantic",
            "infrared_saliency",
        ]
    )
    top_k: int = 2
    sparse_execution: bool = True
    train_execution: str = "sparse_batch"
    inference_execution: str = "sparse_batch"
    spatial_gating: bool = True
    router_branches: list[str] = field(
        default_factory=lambda: ["feature", "task", "spectrum", "modality", "auxiliary"]
    )
    infrared_requires_ir_modality: bool = True
    placements: list[str] = field(default_factory=lambda: ["s2", "s3", "s4"])
    block_counts: dict[str, int] = field(
        default_factory=lambda: {"s1": 0, "s2": 1, "s3": 1, "s4": 1}
    )
    router_hidden_channels: int = 64
    expert_expansion: int = 2
    residual_scale: float = 0.001
    router_temperature: float = 1.0
    task_embedding_dim: int = 64
    modality_embedding_dim: int = 32
    force_fp32_routing: bool = True
    spatial_identity_init: bool = True
    branch_dropout: float = 0.0
    noisy_topk: bool = False
    noisy_topk_std: float = 0.1


@dataclass(slots=True)
class FocusGuidanceConfig:
    enabled: bool = True
    backend: str | None = "lightweight_focus"
    predicted_guidance_only: bool = True
    hidden_channels: int = 32
    run_policy: str = "all"
    source_stages: list[str] = field(default_factory=lambda: ["s1", "s2"])
    use_sobel: bool = True
    use_laplacian: bool = True
    use_fdconv: bool = True
    detach_backbone_features: bool = False
    gradient_normalization: str = "per_sample"


@dataclass(slots=True)
class SemanticGuidanceConfig:
    enabled: bool = True
    backend: str | None = "segformer_b0_local"
    frozen: bool = True
    predicted_guidance_only: bool = True
    model_dir: str = "weights/segformer_b0_cityscapes"
    input_size: int = 256
    num_classes: int = 19
    strict_loading: bool = True
    detach_guidance_input: bool = True
    guidance_policy: str = "all"
    guidance_tasks: list[str] = field(default_factory=lambda: ["vif", "mfif", "seg"])
    final_pass_policy: str = "seg_only"


@dataclass(slots=True)
class GuidanceConfig:
    focus: FocusGuidanceConfig = field(default_factory=FocusGuidanceConfig)
    semantic: SemanticGuidanceConfig = field(default_factory=SemanticGuidanceConfig)


@dataclass(slots=True)
class FeedbackConfig:
    enabled: bool = True
    vif_seg_refinement_enabled: bool = False
    independent_moe_parameters: bool = True
    task_film: bool = True
    placements: list[str] = field(default_factory=lambda: ["s3", "s2"])
    guide_channels: int = 32
    transformer_heads: int = 4
    transformer_window_size: int = 4
    mfif_interaction: bool = True
    residual_scale: float = 0.001
    max_residual: float = 0.25
    num_iterations: int = 1
    share_with_encoder_moe: bool = False


@dataclass(slots=True)
class AblationConfig:
    disable_focus_head: bool = False
    disable_focus_feedback: bool = False
    disable_semantic_feedback: bool = False
    detach_focus_from_backbone: bool = False
    detach_semantic_input: bool = False
    disable_guidance_pyramid: bool = False
    disable_feedback_moe: bool = False
    disable_task_film: bool = False
    disable_refinement_decoder: bool = False
    disable_final_residual: bool = False
    share_feedback_encoder_moe: bool = False


@dataclass(slots=True)
class ModelConfig:
    name: str = "tfs_moe_fusion"
    output_channels: int = 3
    pad_multiple: int = 8
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)


@dataclass(slots=True)
class DataConfig:
    dataset: str = "semantic_rt"
    num_workers: int = 4
    pin_memory: bool = True
    root: str = "data/semantic_rt"
    mfif_root: str = "data/mfif/semantic_rt"
    manifest: str = "data/splits/semantic_rt_test_uniform_2000_seed3407.txt"
    crop_size: int = 256
    horizontal_flip_probability: float = 0.5
    rotation_degrees: float = 10.0
    rotation_probability: float = 0.5
    segmentation_min_valid_pixels: int = 64
    segmentation_crop_attempts: int = 10


@dataclass(slots=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    epsilon: float = 1e-8
    lr_multipliers: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class SchedulerConfig:
    name: str = "cosine_with_warmup"
    warmup_steps: int = 0
    minimum_learning_rate: float = 0.0
    step_size: int = 10_000
    gamma: float = 0.5


@dataclass(slots=True)
class TaskSamplingConfig:
    strategy: str = "weighted_random"
    weights: dict[str, float] = field(
        default_factory=lambda: {"vif": 1.0, "mfif": 1.0, "seg": 1.0}
    )


@dataclass(slots=True)
class TaskSchedulePhaseConfig:
    name: str
    start_epoch: int
    end_epoch: int
    pattern: list[str]


@dataclass(slots=True)
class EMAConfig:
    enabled: bool = True
    decay: float = 0.999


@dataclass(slots=True)
class VIFFusionLossConfig:
    intensity: float = 1.0
    gradient: float = 1.0
    ssim: float = 0.5
    color: float = 0.2
    coarse_supervision: float = 0.25
    intensity_mode: str = "pixel_max"
    intensity_energy_normalization: str = "per_sample_mean"
    ir_intensity_max_weight: float = 0.3
    intensity_visible_support_kernel: int = 3
    intensity_weight_smoothing_kernel: int = 3
    gradient_mode: str = "magnitude_max"
    ir_gradient_dominance_ratio: float = 1.2
    visible_gradient_support_kernel: int = 3
    ssim_mode: str = "source_max"


@dataclass(slots=True)
class MFIFFusionLossConfig:
    reconstruction: float = 1.0
    gradient: float = 1.0
    ssim: float = 0.5
    coarse_supervision: float = 0.25
    use_charbonnier: bool = False


@dataclass(slots=True)
class FocusLossConfig:
    selection: float = 1.0
    boundary: float = 0.5
    confidence: float = 0.1


@dataclass(slots=True)
class SemanticLossConfig:
    cross_entropy: float = 1.0
    dice: float = 0.5
    coarse_supervision: float = 0.0
    ignore_index: int = 255
    class_weights: list[float] | None = None
    improvement_enabled: bool = False
    improvement_weight: float = 0.0
    improvement_margin: float = 0.0
    boundary_alignment: float = 0.0


@dataclass(slots=True)
class SegFusionAnchorLossConfig:
    enabled: bool = True
    intensity: float = 1.0
    gradient: float = 1.0


@dataclass(slots=True)
class FrequencyLossConfig:
    enabled: bool = False
    weight: float = 0.01
    low_leakage: float = 1.0
    detail_leakage: float = 1.0
    semantic_boundary: float = 0.1


@dataclass(slots=True)
class MoEBalanceLossConfig:
    enabled: bool = True
    weight: float = 0.01
    soft_balance_weight: float = 1.0
    switch_balance_weight: float = 0.1
    hard_load_weight: float | None = None
    entropy_enabled: bool = False
    entropy_weight: float = 0.0
    entropy_target: float = 1.0


@dataclass(slots=True)
class TaskConsistencyLossConfig:
    enabled: bool = False
    weight: float = 0.01
    probability: float = 0.25
    gaussian_kernel_size: int = 9
    gaussian_sigma: float = 2.0


@dataclass(slots=True)
class InfraredPreservationLossConfig:
    enabled: bool = True
    weight: float = 0.1
    detach_saliency: bool = True
    saliency_alignment: float = 0.1


@dataclass(slots=True)
class RegularizationLossConfig:
    residual_magnitude: float = 0.001
    range_penalty: float = 0.001


@dataclass(slots=True)
class LossConfig:
    strict_targets: bool = True
    vif: VIFFusionLossConfig = field(default_factory=VIFFusionLossConfig)
    mfif: MFIFFusionLossConfig = field(default_factory=MFIFFusionLossConfig)
    focus: FocusLossConfig = field(default_factory=FocusLossConfig)
    seg_fusion: SegFusionAnchorLossConfig = field(
        default_factory=SegFusionAnchorLossConfig
    )
    semantic: SemanticLossConfig = field(default_factory=SemanticLossConfig)
    frequency: FrequencyLossConfig = field(default_factory=FrequencyLossConfig)
    moe: MoEBalanceLossConfig = field(default_factory=MoEBalanceLossConfig)
    consistency: TaskConsistencyLossConfig = field(
        default_factory=TaskConsistencyLossConfig
    )
    infrared: InfraredPreservationLossConfig = field(
        default_factory=InfraredPreservationLossConfig
    )
    regularization: RegularizationLossConfig = field(
        default_factory=RegularizationLossConfig
    )


@dataclass(slots=True)
class GradientClipConfig:
    enabled: bool = True
    max_norm: float = 1.0


@dataclass(slots=True)
class TaskUpdatePolicyConfig:
    freeze: dict[str, list[str]] = field(
        default_factory=lambda: {
            "vif": [],
            "mfif": ["ir_experts", "semantic_experts"],
            "seg": ["focus_head"],
        }
    )


@dataclass(slots=True)
class TrainingPhaseConfig:
    name: str
    start: int
    end: int
    loss_multipliers: dict[str, float] = field(default_factory=dict)


def _default_training_phases() -> list[TrainingPhaseConfig]:
    return [
        TrainingPhaseConfig(
            "stabilization",
            0,
            5,
            {"seg_fusion": 1.0, "semantic": 0.25},
        ),
        TrainingPhaseConfig(
            "semantic_ramp",
            5,
            15,
            {"seg_fusion": 1.0, "semantic": 0.5},
        ),
        TrainingPhaseConfig(
            "joint",
            15,
            44,
            {"seg_fusion": 0.75, "semantic": 1.0},
        ),
        TrainingPhaseConfig(
            "routing_finetune",
            44,
            50,
            {"seg_fusion": 0.75, "semantic": 1.0, "moe": 1.5},
        ),
    ]


@dataclass(slots=True)
class TrainingPhaseScheduleConfig:
    phases: list[TrainingPhaseConfig] = field(default_factory=_default_training_phases)


@dataclass(slots=True)
class RouterTemperatureScheduleConfig:
    start: float = 1.5
    end: float = 0.7
    schedule: str = "cosine"
    minimum: float = 0.5


@dataclass(slots=True)
class MoEWarmupConfig:
    uniform_steps: int = 50
    uniform_to_soft_end: int = 150
    soft_to_topk_end: int = 300


@dataclass(slots=True)
class MoERefreshConfig:
    interval: int = 200
    routed_fraction: float = 0.5
    expert_only: bool = True


@dataclass(slots=True)
class MoERoutingScheduleConfig:
    initial_temperature: float = 1.5
    sparse_start_temperature: float = 1.0
    final_temperature: float = 0.7
    final_step: int = 7500
    initial_noise_std: float = 0.05
    noise_end_step: int = 7500


@dataclass(slots=True)
class MoERouterMonitorConfig:
    ema_decay: float = 0.95
    starvation_threshold: float = 0.02
    overload_threshold: float = 0.80
    patience_steps: int = 500


@dataclass(slots=True)
class MoERecoveryConfig:
    steps: int = 200
    temperature_floor: float = 1.0
    noise_std: float = 0.03
    refresh_interval: int = 50


@dataclass(slots=True)
class MoEExecutionScheduleConfig:
    enabled: bool = True
    warmup: MoEWarmupConfig = field(default_factory=MoEWarmupConfig)
    refresh: MoERefreshConfig = field(default_factory=MoERefreshConfig)
    routing: MoERoutingScheduleConfig = field(default_factory=MoERoutingScheduleConfig)
    monitor: MoERouterMonitorConfig = field(default_factory=MoERouterMonitorConfig)
    recovery: MoERecoveryConfig = field(default_factory=MoERecoveryConfig)


@dataclass(slots=True)
class TrainingDiagnosticsConfig:
    interval: int = 100
    gradient_conflict_enabled: bool = False
    gradient_conflict_interval: int = 1000
    router_collapse_threshold: float = 0.95
    collapse_patience: int = 10
    loss_gradient_interval: int = 100
    loss_gradient_until_step: int = 1000


@dataclass(slots=True)
class CheckpointConfig:
    every_epochs: int = 1
    every_steps: int = 1000
    keep_last: int = 3
    resume: str | None = None


@dataclass(slots=True)
class DistributedConfig:
    enabled: bool = False
    find_unused_parameters: bool = True


@dataclass(slots=True)
class TrainingConfig:
    device: str = "cuda:0"
    epochs: int = 50
    steps_per_epoch: int = 750
    max_steps: int | None = None
    batch_size: int = 2
    precision: str = "fp32"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    task_sampling: TaskSamplingConfig = field(default_factory=TaskSamplingConfig)
    task_schedule: list[TaskSchedulePhaseConfig] = field(default_factory=list)
    gradient_strategy: str = "alternating"
    gradient_accumulation_steps: int = 2
    gradient_clip: GradientClipConfig = field(default_factory=GradientClipConfig)
    task_update_policy: TaskUpdatePolicyConfig = field(
        default_factory=TaskUpdatePolicyConfig
    )
    phases: TrainingPhaseScheduleConfig = field(
        default_factory=TrainingPhaseScheduleConfig
    )
    router_temperature: RouterTemperatureScheduleConfig = field(
        default_factory=RouterTemperatureScheduleConfig
    )
    moe_execution: MoEExecutionScheduleConfig = field(
        default_factory=MoEExecutionScheduleConfig
    )
    diagnostics: TrainingDiagnosticsConfig = field(
        default_factory=TrainingDiagnosticsConfig
    )
    log_every_steps: int = 10
    ema: EMAConfig = field(default_factory=EMAConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


@dataclass(slots=True)
class ProjectConfig:
    schema_version: int = 1
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ProjectConfig:
        config = _construct_dataclass(cls, values, path="config")
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ConfigurationError(
                f"Unsupported schema_version={self.schema_version}; expected 1"
            )
        if self.experiment.seed < 0:
            raise ConfigurationError("experiment.seed must be non-negative")
        if not self.experiment.name.strip():
            raise ConfigurationError("experiment.name cannot be empty")

        model = self.model
        if model.output_channels != 3:
            raise ConfigurationError("The semantic fusion model requires RGB output")
        if model.pad_multiple <= 0:
            raise ConfigurationError("model.pad_multiple must be positive")

        backbone = model.backbone
        if len(backbone.channels) != len(backbone.depths) or not backbone.channels:
            raise ConfigurationError(
                "backbone.channels and backbone.depths must be non-empty and equal length"
            )
        if any(value <= 0 for value in backbone.channels + backbone.depths):
            raise ConfigurationError("backbone channels/depths must be positive")
        if backbone.normalization not in {"group_norm", "layer_norm_2d", "identity"}:
            raise ConfigurationError(
                "backbone.normalization must be group_norm, layer_norm_2d, or identity"
            )
        if backbone.max_downsample != 8:
            raise ConfigurationError("backbone.max_downsample must be 8")
        if not (
            backbone.shared_source_encoder
            and backbone.separate_modality_stems
            and backbone.preserve_source_pyramids
            and backbone.preserve_fused_pyramid
        ):
            raise ConfigurationError(
                "Backbone sharing, modality stems, and all feature pyramids are required"
            )

        frequency = model.frequency
        legal_stages = {f"s{index + 1}" for index in range(len(backbone.channels))}
        if not set(frequency.placements) <= legal_stages:
            raise ConfigurationError(
                f"frequency.placements must be a subset of {sorted(legal_stages)}"
            )
        if frequency.afno_num_blocks <= 0:
            raise ConfigurationError("frequency.afno_num_blocks must be positive")
        if not 0.0 < frequency.afno_hard_thresholding_fraction <= 1.0:
            raise ConfigurationError(
                "frequency.afno_hard_thresholding_fraction must be in (0, 1]"
            )
        if frequency.fdconv_kernel_size <= 0 or frequency.fdconv_kernel_size % 2 == 0:
            raise ConfigurationError(
                "frequency.fdconv_kernel_size must be positive and odd"
            )
        if not 1 <= frequency.fdconv_kernel_num <= frequency.fdconv_kernel_size**2:
            raise ConfigurationError(
                "frequency.fdconv_kernel_num must be between 1 and kernel_size squared"
            )
        if frequency.residual_scale <= 0:
            raise ConfigurationError("frequency.residual_scale must be positive")

        required_experts = {
            "common",
            "low_frequency",
            "detail",
            "semantic",
            "infrared_saliency",
        }
        if (
            len(model.moe.experts) != len(required_experts)
            or set(model.moe.experts) != required_experts
        ):
            raise ConfigurationError(
                "model.moe.experts must contain exactly the five functional experts"
            )
        if not 1 <= model.moe.top_k <= len(model.moe.experts) - 1:
            raise ConfigurationError(
                "model.moe.top_k must fit the four experts available without infrared"
            )
        required_router_branches = {
            "feature",
            "task",
            "spectrum",
            "modality",
            "auxiliary",
        }
        if (
            len(model.moe.router_branches) != len(required_router_branches)
            or set(model.moe.router_branches) != required_router_branches
        ):
            raise ConfigurationError(
                "model.moe.router_branches must contain all five evidence branches"
            )
        if not model.moe.infrared_requires_ir_modality:
            raise ConfigurationError("The IR expert must require a real IR modality")
        if not set(model.moe.placements) <= legal_stages:
            raise ConfigurationError(
                f"moe.placements must be a subset of {sorted(legal_stages)}"
            )
        if set(model.moe.block_counts) != legal_stages or any(
            value < 0 for value in model.moe.block_counts.values()
        ):
            raise ConfigurationError(
                "moe.block_counts must define non-negative s1/s2/s3/s4 counts"
            )
        if model.moe.block_counts != {"s1": 0, "s2": 1, "s3": 1, "s4": 1}:
            raise ConfigurationError("The encoder requires MoE counts 0/1/1/1")
        if model.moe.train_execution not in {
            "dense_masked",
            "sparse_batch",
        } or model.moe.inference_execution not in {"dense_masked", "sparse_batch"}:
            raise ConfigurationError(
                "MoE execution must be dense_masked or sparse_batch"
            )
        if model.moe.router_hidden_channels <= 0 or model.moe.expert_expansion <= 0:
            raise ConfigurationError(
                "MoE hidden channels and expert expansion must be positive"
            )
        if model.moe.residual_scale <= 0 or model.moe.router_temperature <= 0:
            raise ConfigurationError(
                "MoE residual scale and router temperature must be positive"
            )
        if not (
            model.guidance.focus.predicted_guidance_only
            and model.guidance.semantic.predicted_guidance_only
        ):
            raise ConfigurationError("Ground-truth guidance cannot enter the router")
        focus = model.guidance.focus
        semantic = model.guidance.semantic
        feedback = model.feedback
        if focus.enabled and focus.backend != "lightweight_focus":
            raise ConfigurationError(
                "Enabled focus guidance requires lightweight_focus"
            )
        if focus.hidden_channels <= 0:
            raise ConfigurationError("focus.hidden_channels must be positive")
        if focus.run_policy not in {"all", "mfif_only"}:
            raise ConfigurationError("focus.run_policy must be all or mfif_only")
        if semantic.enabled:
            if semantic.backend not in {"segformer_b0_local", "null", None}:
                raise ConfigurationError(
                    "Semantic backend must be segformer_b0_local or null"
                )
            if not semantic.frozen:
                raise ConfigurationError("The semantic backend must remain frozen")
            if semantic.input_size < 32 or semantic.num_classes <= 1:
                raise ConfigurationError("Semantic input_size/classes are invalid")
            if (
                semantic.backend == "segformer_b0_local"
                and not semantic.model_dir.strip()
            ):
                raise ConfigurationError("semantic.model_dir cannot be empty")
        if not feedback.enabled:
            raise ConfigurationError("The final model requires feedback refinement")
        if feedback.vif_seg_refinement_enabled:
            raise ConfigurationError(
                "Stage 1 requires VIF/SEG refinement to remain disabled"
            )
        if not set(feedback.placements) <= legal_stages:
            raise ConfigurationError(
                f"feedback.placements must be a subset of {sorted(legal_stages)}"
            )
        if feedback.guide_channels <= 0 or feedback.transformer_heads <= 0:
            raise ConfigurationError("Feedback guide channels/heads must be positive")
        if feedback.transformer_window_size <= 0 or feedback.residual_scale <= 0:
            raise ConfigurationError(
                "Feedback window size/residual scale must be positive"
            )
        if feedback.num_iterations != 1:
            raise ConfigurationError("feedback.num_iterations must be 1")
        if set(feedback.placements) != {"s2", "s3"} or len(feedback.placements) != 2:
            raise ConfigurationError(
                "feedback placement must contain one S3 and one S2 block"
            )
        for stage, channels in enumerate(backbone.channels, start=1):
            if (
                f"s{stage}" in feedback.placements
                and channels % feedback.transformer_heads
            ):
                raise ConfigurationError(
                    f"backbone channel count at s{stage} must divide transformer_heads"
                )

        data = self.data
        if data.dataset != "semantic_rt":
            raise ConfigurationError("data.dataset must be semantic_rt")
        if data.num_workers < 0:
            raise ConfigurationError("data.num_workers cannot be negative")
        if data.crop_size <= 0:
            raise ConfigurationError("data.crop_size must be positive")
        for name, probability in (
            ("horizontal_flip_probability", data.horizontal_flip_probability),
            ("rotation_probability", data.rotation_probability),
        ):
            if not 0 <= probability <= 1:
                raise ConfigurationError(f"data.{name} must be in [0, 1]")
        if not 0 <= data.rotation_degrees < 45:
            raise ConfigurationError("data.rotation_degrees must be in [0, 45)")
        if data.segmentation_min_valid_pixels < 0:
            raise ConfigurationError(
                "data.segmentation_min_valid_pixels cannot be negative"
            )
        if data.segmentation_crop_attempts <= 0:
            raise ConfigurationError("data.segmentation_crop_attempts must be positive")
        if data.dataset == "semantic_rt" and not all(
            value.strip() for value in (data.root, data.mfif_root, data.manifest)
        ):
            raise ConfigurationError(
                "SemanticRT requires data.root, data.mfif_root, and data.manifest"
            )
        if self.training.epochs <= 0 or self.training.batch_size <= 0:
            raise ConfigurationError("training epochs and batch_size must be positive")
        training = self.training
        if training.device not in {"auto", "cpu", "cuda"} and not (
            training.device.startswith("cuda:")
            and training.device.removeprefix("cuda:").isdigit()
        ):
            raise ConfigurationError(
                "training.device must be auto, cpu, cuda, or cuda:<index>"
            )
        if training.steps_per_epoch <= 0:
            raise ConfigurationError("training.steps_per_epoch must be positive")
        if training.max_steps is not None and training.max_steps <= 0:
            raise ConfigurationError("training.max_steps must be positive when set")
        if training.gradient_accumulation_steps <= 0:
            raise ConfigurationError("gradient_accumulation_steps must be positive")
        if training.log_every_steps <= 0:
            raise ConfigurationError("training.log_every_steps must be positive")
        if self.training.precision not in {"fp32", "fp16", "bf16"}:
            raise ConfigurationError("training.precision must be fp32, fp16, or bf16")
        optimizer = training.optimizer
        if optimizer.name != "adamw":
            raise ConfigurationError("Only the adamw optimizer is supported")
        if optimizer.learning_rate <= 0 or optimizer.weight_decay < 0:
            raise ConfigurationError("optimizer learning rate/weight decay are invalid")
        if len(optimizer.betas) != 2 or any(
            not 0 <= value < 1 for value in optimizer.betas
        ):
            raise ConfigurationError(
                "optimizer.betas must contain two values in [0, 1)"
            )
        if optimizer.epsilon <= 0 or any(
            value <= 0 for value in optimizer.lr_multipliers.values()
        ):
            raise ConfigurationError(
                "optimizer epsilon and LR multipliers must be positive"
            )
        scheduler = training.scheduler
        if scheduler.name not in {"cosine_with_warmup", "constant", "step"}:
            raise ConfigurationError(
                "scheduler.name must be cosine_with_warmup, constant, or step"
            )
        if scheduler.warmup_steps < 0 or scheduler.minimum_learning_rate < 0:
            raise ConfigurationError(
                "scheduler warmup/minimum learning rate are invalid"
            )
        if scheduler.step_size <= 0 or not 0 < scheduler.gamma <= 1:
            raise ConfigurationError("scheduler step_size/gamma are invalid")
        if training.task_sampling.strategy not in {
            "alternating",
            "weighted_random",
            "scheduled",
        }:
            raise ConfigurationError(
                "task sampling must be alternating, weighted_random, or scheduled"
            )
        if set(self.training.task_sampling.weights) != {"vif", "mfif", "seg"}:
            raise ConfigurationError(
                "task_sampling.weights must define vif, mfif, and seg"
            )
        if any(weight <= 0 for weight in self.training.task_sampling.weights.values()):
            raise ConfigurationError("task sampling weights must be positive")
        schedule = training.task_schedule
        if training.task_sampling.strategy == "scheduled":
            if not schedule:
                raise ConfigurationError(
                    "task_schedule is required when task sampling is scheduled"
                )
            previous_end = 0
            legal_tasks = {"vif", "mfif", "seg"}
            for phase in schedule:
                if (
                    not phase.name.strip()
                    or phase.start_epoch != previous_end
                    or phase.end_epoch <= phase.start_epoch
                ):
                    raise ConfigurationError(
                        "Task schedule phases must be named, contiguous, and ordered"
                    )
                if not phase.pattern or not set(phase.pattern) <= legal_tasks:
                    raise ConfigurationError(
                        "Task schedule patterns must contain only vif, mfif, and seg"
                    )
                previous_end = phase.end_epoch
            if previous_end != training.epochs:
                raise ConfigurationError(
                    "Task schedule must cover every configured training epoch"
                )
        elif schedule:
            raise ConfigurationError(
                "task_schedule requires task_sampling.strategy=scheduled"
            )
        if training.gradient_strategy != "alternating":
            raise ConfigurationError(
                "Only homogeneous alternating task updates are supported"
            )
        if training.gradient_clip.enabled and training.gradient_clip.max_norm <= 0:
            raise ConfigurationError("gradient_clip.max_norm must be positive")
        legal_groups = {
            "shared_backbone",
            "coarse_decoder",
            "common_experts",
            "low_frequency_experts",
            "detail_experts",
            "semantic_experts",
            "ir_experts",
            "coarse_routers",
            "focus_head",
            "guidance_pyramid",
            "feedback_experts",
            "feedback_routers",
            "refinement_decoder",
            "residual_head",
        }
        if set(training.task_update_policy.freeze) != {"vif", "mfif", "seg"}:
            raise ConfigurationError(
                "task_update_policy.freeze must define vif, mfif, and seg"
            )
        unknown_groups = (
            set().union(*map(set, training.task_update_policy.freeze.values()))
            - legal_groups
        )
        if unknown_groups:
            raise ConfigurationError(
                f"Unknown task update groups: {sorted(unknown_groups)}"
            )
        phases = training.phases.phases
        if not phases:
            raise ConfigurationError("At least one training phase is required")
        previous_end = -1
        for phase in phases:
            if not phase.name.strip() or phase.start < 0 or phase.end <= phase.start:
                raise ConfigurationError(
                    "Training phases require a name and start < end"
                )
            if phase.start < previous_end:
                raise ConfigurationError(
                    "Training phases must be ordered and non-overlapping"
                )
            if any(value < 0 for value in phase.loss_multipliers.values()):
                raise ConfigurationError("Phase loss multipliers cannot be negative")
            previous_end = phase.end
        temperature = training.router_temperature
        if temperature.schedule not in {"cosine", "linear", "constant"}:
            raise ConfigurationError("router temperature schedule is invalid")
        if min(temperature.start, temperature.end, temperature.minimum) <= 0:
            raise ConfigurationError("router temperatures must be positive")
        execution = training.moe_execution
        warmup = execution.warmup
        if not (
            0
            <= warmup.uniform_steps
            <= warmup.uniform_to_soft_end
            <= warmup.soft_to_topk_end
        ):
            raise ConfigurationError(
                "MoE warmup steps must satisfy 0 <= uniform <= soft <= top-k"
            )
        if execution.refresh.interval <= 0 or not (
            0.0 <= execution.refresh.routed_fraction <= 1.0
        ):
            raise ConfigurationError("MoE refresh interval/routed_fraction are invalid")
        routing = execution.routing
        if (
            min(
                routing.initial_temperature,
                routing.sparse_start_temperature,
                routing.final_temperature,
            )
            <= 0
            or routing.final_step < warmup.soft_to_topk_end
        ):
            raise ConfigurationError("MoE routing temperatures/final_step are invalid")
        if (
            routing.initial_noise_std < 0
            or routing.noise_end_step < warmup.soft_to_topk_end
        ):
            raise ConfigurationError("MoE routing noise schedule is invalid")
        monitor = execution.monitor
        if not 0.0 < monitor.ema_decay < 1.0 or not (
            0.0 <= monitor.starvation_threshold < monitor.overload_threshold <= 1.0
        ):
            raise ConfigurationError("MoE router monitor thresholds are invalid")
        if monitor.patience_steps <= 0:
            raise ConfigurationError("MoE monitor patience_steps must be positive")
        recovery = execution.recovery
        if (
            recovery.steps <= 0
            or recovery.temperature_floor <= 0
            or recovery.noise_std < 0
            or recovery.refresh_interval <= 0
        ):
            raise ConfigurationError("MoE recovery settings are invalid")
        if (
            training.diagnostics.interval <= 0
            or training.diagnostics.collapse_patience <= 0
            or training.diagnostics.loss_gradient_interval <= 0
            or training.diagnostics.loss_gradient_until_step < 0
        ):
            raise ConfigurationError("training diagnostic intervals must be positive")
        if not 0 < training.diagnostics.router_collapse_threshold <= 1:
            raise ConfigurationError("router collapse threshold must be in (0, 1]")
        if not 0.0 < self.training.ema.decay < 1.0:
            raise ConfigurationError("training.ema.decay must be between 0 and 1")
        if (
            training.checkpoint.every_epochs <= 0
            or training.checkpoint.every_steps <= 0
        ):
            raise ConfigurationError("checkpoint intervals must be positive")
        if training.checkpoint.keep_last <= 0:
            raise ConfigurationError("checkpoint.keep_last must be positive")
        vif_loss = training.losses.vif
        if (
            min(
                vif_loss.intensity,
                vif_loss.gradient,
                vif_loss.ssim,
                vif_loss.color,
                vif_loss.coarse_supervision,
            )
            < 0
        ):
            raise ConfigurationError("VIF loss weights cannot be negative")
        if vif_loss.gradient_mode not in {
            "magnitude_max",
            "directional_visible_anchor",
        }:
            raise ConfigurationError(
                "VIF gradient_mode must be magnitude_max or directional_visible_anchor"
            )
        if vif_loss.ssim_mode not in {"source_max", "visible_anchor"}:
            raise ConfigurationError(
                "VIF ssim_mode must be source_max or visible_anchor"
            )
        if vif_loss.intensity_mode not in {
            "pixel_max",
            "gradient_weighted_visible_anchor",
        }:
            raise ConfigurationError(
                "VIF intensity_mode must be pixel_max or "
                "gradient_weighted_visible_anchor"
            )
        if vif_loss.intensity_energy_normalization not in {
            "none",
            "per_sample_mean",
        }:
            raise ConfigurationError(
                "VIF intensity_energy_normalization must be none or per_sample_mean"
            )
        if not 0.0 <= vif_loss.ir_intensity_max_weight <= 1.0:
            raise ConfigurationError(
                "VIF ir_intensity_max_weight must be between 0 and 1"
            )
        for name, kernel in (
            (
                "intensity_visible_support_kernel",
                vif_loss.intensity_visible_support_kernel,
            ),
            (
                "intensity_weight_smoothing_kernel",
                vif_loss.intensity_weight_smoothing_kernel,
            ),
        ):
            if kernel <= 0 or kernel % 2 == 0:
                raise ConfigurationError(f"VIF {name} must be positive and odd")
        if vif_loss.ir_gradient_dominance_ratio < 1.0:
            raise ConfigurationError(
                "VIF ir_gradient_dominance_ratio must be at least 1"
            )
        if (
            vif_loss.visible_gradient_support_kernel <= 0
            or vif_loss.visible_gradient_support_kernel % 2 == 0
        ):
            raise ConfigurationError(
                "VIF visible_gradient_support_kernel must be positive and odd"
            )
        if training.losses.focus.selection < 0 or training.losses.focus.boundary < 0:
            raise ConfigurationError("Focus loss weights cannot be negative")
        seg_fusion = training.losses.seg_fusion
        if min(seg_fusion.intensity, seg_fusion.gradient) < 0:
            raise ConfigurationError("SEG fusion anchor weights cannot be negative")
        balance = training.losses.moe
        if balance.hard_load_weight is not None:
            import warnings

            warnings.warn(
                "losses.moe.hard_load_weight is deprecated; use switch_balance_weight",
                DeprecationWarning,
                stacklevel=2,
            )
            balance.switch_balance_weight = balance.hard_load_weight
        if (
            min(
                balance.weight,
                balance.soft_balance_weight,
                balance.switch_balance_weight,
                balance.entropy_weight,
            )
            < 0
        ):
            raise ConfigurationError("MoE balance loss weights cannot be negative")
        if training.losses.infrared.saliency_alignment < 0:
            raise ConfigurationError("infrared.saliency_alignment cannot be negative")
        semantic_loss = training.losses.semantic
        if (
            min(
                semantic_loss.cross_entropy,
                semantic_loss.dice,
                semantic_loss.coarse_supervision,
                semantic_loss.boundary_alignment,
            )
            < 0
        ):
            raise ConfigurationError("Semantic loss weights cannot be negative")
        if (
            semantic_loss.class_weights is not None
            and len(semantic_loss.class_weights) != semantic.num_classes
        ):
            raise ConfigurationError(
                "semantic.class_weights must match semantic num_classes"
            )
        consistency = training.losses.consistency
        if not 0 <= consistency.probability <= 1:
            raise ConfigurationError("consistency.probability must be in [0, 1]")
        if (
            consistency.gaussian_kernel_size <= 0
            or consistency.gaussian_kernel_size % 2 == 0
        ):
            raise ConfigurationError(
                "consistency Gaussian kernel must be positive and odd"
            )

        output_dir = Path(self.experiment.output_dir)
        if output_dir == Path("/"):
            raise ConfigurationError("experiment.output_dir cannot be filesystem root")


def _construct_dataclass(cls: type[Any], values: Any, path: str) -> Any:
    if not isinstance(values, dict):
        raise ConfigurationError(f"{path} must be a mapping")
    legal = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - legal)
    if unknown:
        raise ConfigurationError(
            f"Unknown field(s) at {path}: {', '.join(unknown)}; "
            f"legal fields: {', '.join(sorted(legal))}"
        )

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in values:
            continue
        kwargs[item.name] = _coerce_value(
            hints[item.name], values[item.name], f"{path}.{item.name}"
        )
    return cls(**kwargs)


def _coerce_value(expected: Any, value: Any, path: str) -> Any:
    origin = get_origin(expected)
    args = get_args(expected)

    if origin in {Union, UnionType}:
        if value is None and type(None) in args:
            return None
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return _coerce_value(non_none[0], value, path)

    if isinstance(expected, type) and is_dataclass(expected):
        return _construct_dataclass(expected, value, path)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigurationError(f"{path} must be a list")
        element_type = args[0] if args else Any
        return [_coerce_value(element_type, item, f"{path}[]") for item in value]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigurationError(f"{path} must be a mapping")
        key_type, value_type = args if args else (Any, Any)
        return {
            _coerce_value(key_type, key, f"{path}.<key>"): _coerce_value(
                value_type, item, f"{path}.{key}"
            )
            for key, item in value.items()
        }
    if expected is Any:
        return value
    if (
        expected is float
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return float(value)
    if expected is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if expected is bool and isinstance(value, bool):
        return value
    if expected is str and isinstance(value, str):
        return value
    if isinstance(value, expected):
        return value
    raise ConfigurationError(
        f"{path} has type {type(value).__name__}; expected {expected}"
    )


from dataclasses import asdict


def _yaml_module():
    try:
        import yaml
    except ImportError as error:
        raise ConfigurationError(
            "PyYAML is required to read project configuration. "
            "Install the project dependencies first."
        ) from error
    return yaml


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    merged = _load_mapping(config_path, seen=set())
    merged.pop("_base_", None)
    return ProjectConfig.from_mapping(merged)


def save_resolved_config(config: ProjectConfig, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    yaml = _yaml_module()
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False, allow_unicode=True)
    return output


def _load_mapping(path: Path, seen: set[Path]) -> dict[str, Any]:
    if path in seen:
        chain = " -> ".join(str(item) for item in [*seen, path])
        raise ConfigurationError(f"Configuration inheritance cycle: {chain}")
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    yaml = _yaml_module()
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")

    base_value = values.get("_base_")
    if base_value is None:
        return values
    if not isinstance(base_value, str):
        raise ConfigurationError(f"_base_ must be a path string in {path}")

    base_path = (path.parent / base_value).resolve()
    base = _load_mapping(base_path, seen | {path})
    override = dict(values)
    override.pop("_base_", None)
    return _deep_merge(base, override)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

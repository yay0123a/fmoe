"""TFS-MoE-Fusion consolidated implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from tfs_moe_fusion.backbone import ArbitrarySizePadder, BackboneOutput, FeaturePyramid
from tfs_moe_fusion.config import ProjectConfig
from tfs_moe_fusion.frequency import spectral_statistics
from tfs_moe_fusion.guidance import (
    FrozenSegformerBackend,
    GuidanceConditioner,
    GuidancePyramidAdapter,
    LightweightFocusHead,
    SemanticGuideOutput,
)
from tfs_moe_fusion.moe import (
    ExpertContext,
    FunctionalMoEBlock,
    RouterContext,
    TaskEmbedding,
)
from tfs_moe_fusion.types import (
    AuxiliaryOutputs,
    FusionBatch,
    RouterDiagnostics,
    SpectralStatistics,
    TaskType,
)


class LightweightMFIFInteraction(nn.Module):
    """Optional lightweight shared-feature interaction requested for MFIF."""

    def __init__(self, channels: int, heads: int = 4, window_size: int = 4) -> None:
        super().__init__()
        self.channels, self.window_size = channels, window_size
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.projection = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(
        self, fused: Tensor, source_a: Tensor, source_b: Tensor, task: TaskType
    ) -> Tensor:
        if task is not TaskType.MFIF:
            return fused
        q, info = self._windows(fused)
        a, _ = self._windows(source_a)
        b, _ = self._windows(source_b)
        update, _ = self.attention(
            self.norm(q),
            self.norm(torch.cat((a, b), 1)),
            self.norm(torch.cat((a, b), 1)),
            need_weights=False,
        )
        return fused + self.scale * self.projection(
            self._reverse(update, info) + self.local(fused)
        )

    def _windows(self, tensor: Tensor):
        b, c, h, w = tensor.shape
        size = self.window_size
        pb, pr = (-h) % size, (-w) % size
        tensor = (
            functional.pad(tensor, (0, pr, 0, pb), mode="replicate")
            if pb or pr
            else tensor
        )
        hp, wp = tensor.shape[-2:]
        windows = (
            tensor.view(b, c, hp // size, size, wp // size, size)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, size * size, c)
        )
        return windows, (b, h, w, hp, wp)

    def _reverse(self, windows: Tensor, info):
        b, h, w, hp, wp = info
        size, c = self.window_size, windows.shape[-1]
        value = (
            windows.view(b, hp // size, wp // size, size, size, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, c, hp, wp)
        )
        return value[..., :h, :w]


class TaskFiLM(nn.Module):
    def __init__(
        self,
        channels: int,
        task_embedding: TaskEmbedding | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.task_embedding = task_embedding or TaskEmbedding(64)
        self.projection = nn.Linear(
            self.task_embedding.embedding.embedding_dim, channels * 2
        )
        nn.init.normal_(self.projection.weight, std=1e-3)
        nn.init.zeros_(self.projection.bias)

    def forward(self, tensor: Tensor, task: TaskType) -> Tensor:
        if not self.enabled:
            return tensor
        token = self.task_embedding(task, tensor.shape[0], tensor.device)
        gamma, beta = self.projection(token).chunk(2, 1)
        return tensor * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class RefinementStage(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        task_embedding: TaskEmbedding,
        use_task_film: bool,
    ) -> None:
        super().__init__()
        self.skip = nn.Conv2d(output_channels * 2, output_channels, 1)
        self.up = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.conditioner = GuidanceConditioner(output_channels)
        self.film = TaskFiLM(output_channels, task_embedding, use_task_film)
        self.body = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(1, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(
        self, tensor: Tensor, skip: Tensor, guide: Tensor, task: TaskType
    ) -> Tensor:
        tensor = functional.interpolate(
            tensor, skip.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.skip(torch.cat((self.up(tensor), skip), 1))
        return self.body(self.film(self.conditioner(value, guide), task))


class TaskResidualHead(nn.Module):
    def __init__(
        self,
        channels: int,
        output_channels: int,
        max_residual: float,
        scale_init: float,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, output_channels, 3, padding=1),
        )
        nn.init.normal_(self.body[-1].weight, std=1e-3)
        nn.init.zeros_(self.body[-1].bias)
        self.max_residual = max_residual
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, tensor: Tensor) -> Tensor:
        return self.scale * self.max_residual * torch.tanh(self.body(tensor))


class TaskConditionedRefinementDecoder(nn.Module):
    def __init__(
        self,
        channels: list[int],
        output_channels: int,
        max_residual: float,
        scale_init: float,
        task_embedding: TaskEmbedding,
        use_task_film: bool = True,
    ) -> None:
        super().__init__()
        self.deep_conditioner = GuidanceConditioner(channels[3])
        self.deep_film = TaskFiLM(channels[3], task_embedding, use_task_film)
        self.stages = nn.ModuleList(
            [
                RefinementStage(
                    channels[index], channels[index - 1], task_embedding, use_task_film
                )
                for index in range(3, 0, -1)
            ]
        )
        self.residual_head = TaskResidualHead(
            channels[0], output_channels, max_residual, scale_init
        )

    def forward(
        self, pyramid: FeaturePyramid, guidance: tuple[Tensor, ...], task: TaskType
    ) -> tuple[Tensor, Tensor]:
        features = tuple(pyramid)
        decoded = self.deep_film(self.deep_conditioner(features[3], guidance[3]), task)
        for stage, skip, guide in zip(
            self.stages, reversed(features[:3]), reversed(guidance[:3]), strict=True
        ):
            decoded = stage(decoded, skip, guide, task)
        return decoded, self.residual_head(decoded)


@dataclass(slots=True)
class FeedbackResult:
    final: Tensor
    coarse: Tensor
    refinement: Tensor
    auxiliary: AuxiliaryOutputs
    coarse_semantic: SemanticGuideOutput | None
    final_semantic: SemanticGuideOutput | None
    router_diagnostics: tuple[RouterDiagnostics, ...]
    spectral_statistics: tuple[SpectralStatistics, ...]
    debug: dict[str, object]


class ClosedLoopRefinement(nn.Module):
    def __init__(
        self,
        config: ProjectConfig,
        task_embedding: TaskEmbedding,
        encoder_moe: nn.ModuleDict | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        channels, focus_cfg, semantic_cfg, feedback_cfg = (
            config.model.backbone.channels,
            config.model.guidance.focus,
            config.model.guidance.semantic,
            config.model.feedback,
        )
        self.focus_head = (
            LightweightFocusHead(
                channels,
                focus_cfg.hidden_channels,
                focus_cfg.detach_backbone_features
                or config.model.ablation.detach_focus_from_backbone,
                focus_cfg.gradient_normalization,
                focus_cfg.use_sobel,
                focus_cfg.use_laplacian,
                focus_cfg.use_fdconv,
            )
            if focus_cfg.enabled and not config.model.ablation.disable_focus_head
            else None
        )
        self.semantic_backend = None
        if semantic_cfg.enabled and semantic_cfg.backend == "segformer_b0_local":
            self.semantic_backend = FrozenSegformerBackend(
                semantic_cfg.model_dir,
                semantic_cfg.input_size,
                semantic_cfg.num_classes,
                semantic_cfg.strict_loading,
            )
        self.guidance_builder = GuidancePyramidAdapter(channels)
        self.share_feedback_moe = (
            feedback_cfg.share_with_encoder_moe
            or config.model.ablation.share_feedback_encoder_moe
            or not feedback_cfg.independent_moe_parameters
        )
        if self.share_feedback_moe:
            if encoder_moe is None:
                raise ValueError("Shared feedback MoE requires encoder MoE blocks")
            self.feedback_moe = nn.ModuleDict(
                {stage: encoder_moe[stage][0] for stage in feedback_cfg.placements}
            )
        else:
            self.feedback_moe = nn.ModuleDict(
                {
                    stage: FunctionalMoEBlock(
                        channels[int(stage[1:]) - 1],
                        config.model.moe,
                        f"feedback.{stage}.moe0",
                        task_embedding,
                    )
                    for stage in feedback_cfg.placements
                }
            )
        self.feedback_conditioners = nn.ModuleDict(
            {
                stage: GuidanceConditioner(channels[int(stage[1:]) - 1])
                for stage in feedback_cfg.placements
            }
        )
        self.mfif_interactions = (
            nn.ModuleDict(
                {
                    stage: LightweightMFIFInteraction(
                        channels[int(stage[1:]) - 1],
                        feedback_cfg.transformer_heads,
                        feedback_cfg.transformer_window_size,
                    )
                    for stage in feedback_cfg.placements
                }
            )
            if feedback_cfg.mfif_interaction
            else nn.ModuleDict()
        )
        self.decoder = TaskConditionedRefinementDecoder(
            channels,
            config.model.output_channels,
            feedback_cfg.max_residual,
            feedback_cfg.residual_scale,
            task_embedding,
            feedback_cfg.task_film and not config.model.ablation.disable_task_film,
        )

    def _semantic(self, image: Tensor, task: TaskType, final: bool = False):
        if self.semantic_backend is None:
            return None
        policy = (
            self.config.model.guidance.semantic.final_pass_policy
            if final
            else self.config.model.guidance.semantic.guidance_policy
        )
        if final and (
            policy == "none" or policy == "seg_only" and task is not TaskType.SEG
        ):
            return None
        if (
            not final
            and policy != "all"
            and task.value not in self.config.model.guidance.semantic.guidance_tasks
        ):
            return None
        detach = (
            self.config.model.guidance.semantic.detach_guidance_input
            or self.config.model.ablation.detach_semantic_input
        )
        return self.semantic_backend(image.detach() if detach else image)

    @staticmethod
    def _crop_semantic(value: SemanticGuideOutput | None, height: int, width: int):
        if value is None:
            return None
        return SemanticGuideOutput(
            value.logits[..., :height, :width],
            value.probabilities[..., :height, :width],
            value.uncertainty[..., :height, :width],
            value.boundary[..., :height, :width],
            value.features,
            value.available,
            value.aux,
        )

    def forward(self, batch: FusionBatch, backbone: BackboneOutput) -> FeedbackResult:
        coarse = backbone.padded_fused_image
        image_a, _ = self.configured_pad(batch.source_a.image)
        image_b, _ = self.configured_pad(batch.source_b.image)
        run_focus = self.focus_head is not None and (
            self.config.model.guidance.focus.run_policy == "all"
            or batch.task is TaskType.MFIF
        )
        focus = (
            self.focus_head(image_a, image_b, backbone.source_a, backbone.source_b)
            if run_focus
            else None
        )
        semantic = self._semantic(coarse, batch.task)
        focus_feedback = (
            None if self.config.model.ablation.disable_focus_feedback else focus
        )
        semantic_feedback = (
            None if self.config.model.ablation.disable_semantic_feedback else semantic
        )
        if self.config.model.ablation.disable_guidance_pyramid:
            guidance = tuple(torch.zeros_like(value) for value in backbone.fused)
        else:
            guidance = self.guidance_builder(
                focus_feedback, semantic_feedback, backbone.fused
            )
        refined = list(backbone.fused)
        diagnostics, statistics = [], []
        for stage in self.config.model.feedback.placements:
            index = int(stage[1:]) - 1
            feature = self.feedback_conditioners[stage](refined[index], guidance[index])
            if stage in self.mfif_interactions:
                feature = self.mfif_interactions[stage](
                    feature,
                    backbone.source_a[index],
                    backbone.source_b[index],
                    batch.task,
                )
            stats = spectral_statistics(feature, f"feedback.{stage}.moe0.input")
            context = RouterContext(
                feature,
                batch.task,
                batch.source_a.modality,
                batch.source_b.modality,
                stats,
                backbone.source_a[index],
                backbone.source_b[index],
                focus.reliability_a if focus_feedback else None,
                focus.reliability_b if focus_feedback else None,
                focus.confidence if focus_feedback else None,
                semantic_feedback.boundary if semantic_feedback else None,
                semantic_feedback.uncertainty if semantic_feedback else None,
                f"feedback.{stage}.moe0",
            )
            expert_context = ExpertContext(
                batch.task,
                batch.source_a.modality,
                batch.source_b.modality,
                backbone.source_a[index],
                backbone.source_b[index],
                focus_feedback.reliability if focus_feedback else None,
                focus_feedback.reliability_a if focus_feedback else None,
                focus_feedback.reliability_b if focus_feedback else None,
                focus_feedback.confidence if focus_feedback else None,
                semantic_feedback.uncertainty if semantic_feedback else None,
                semantic_feedback.boundary if semantic_feedback else None,
                f"feedback.{stage}.moe0",
            )
            if not self.config.model.ablation.disable_feedback_moe:
                feature, diag = self.feedback_moe[stage](
                    feature, context, expert_context
                )
                # A shared block retains its coarse-pass module identity, while the
                # diagnostic belongs to this second, guidance-conditioned pass.
                diag.block_id = f"feedback.{stage}.moe0"
                diagnostics.append(diag)
            refined[index] = feature
            statistics.append(stats)
        refined_pyramid = FeaturePyramid(*refined)
        if self.config.model.ablation.disable_refinement_decoder:
            residual = torch.zeros_like(coarse)
        else:
            _, residual = self.decoder(refined_pyramid, guidance, batch.task)
        if self.config.model.ablation.disable_final_residual:
            residual = torch.zeros_like(residual)
        preclamp = coarse + residual
        final_padded = preclamp.clamp(0, 1)
        final_semantic = self._semantic(final_padded, batch.task, final=True)
        h, w = batch.spatial_size
        crop = lambda value: value[..., :h, :w]
        focus_crop = (
            None
            if focus is None
            else type(focus)(
                crop(focus.reliability_a),
                crop(focus.reliability_b),
                crop(focus.selection_logits),
                crop(focus.selection_a),
                crop(focus.selection_b),
                crop(focus.confidence),
                crop(focus.boundary) if focus.boundary is not None else None,
                focus.aux,
            )
        )
        coarse_semantic = self._crop_semantic(semantic, h, w)
        final_semantic = self._crop_semantic(final_semantic, h, w)
        auxiliary = AuxiliaryOutputs(
            torch.cat((focus_crop.reliability_a, focus_crop.reliability_b), 1)
            if focus_crop
            else None,
            focus_crop.selection if focus_crop else None,
            focus_crop.confidence if focus_crop else None,
            focus_crop.boundary if focus_crop else None,
            coarse_semantic.probabilities if coarse_semantic else None,
            coarse_semantic.logits if coarse_semantic else None,
            coarse_semantic.uncertainty if coarse_semantic else None,
            coarse_semantic.boundary if coarse_semantic else None,
        )
        return FeedbackResult(
            crop(final_padded),
            crop(coarse),
            crop(residual),
            auxiliary,
            coarse_semantic,
            final_semantic,
            tuple(diagnostics),
            tuple(statistics),
            {
                "focus": focus_crop,
                "coarse_semantic": coarse_semantic,
                "final_semantic": final_semantic,
                "final_preclamp": crop(preclamp),
                "clamp_low_ratio": (preclamp < 0).float().mean(),
                "clamp_high_ratio": (preclamp > 1).float().mean(),
                "focus_active": focus is not None,
                "semantic_available": semantic is not None,
                "feedback_placements": tuple(self.config.model.feedback.placements),
                "feedback_moe_shared": self.share_feedback_moe,
                "guidance_pyramid_disabled": self.config.model.ablation.disable_guidance_pyramid,
                "task_film_disabled": self.config.model.ablation.disable_task_film,
                "refinement_decoder_disabled": self.config.model.ablation.disable_refinement_decoder,
            },
        )

    def configured_pad(self, image: Tensor):
        return ArbitrarySizePadder(self.config.model.pad_multiple).pad(image)


from torch import nn

from tfs_moe_fusion.backbone import CustomMultiscaleBackbone
from tfs_moe_fusion.types import FusionOutput


class TFSMoEFusion(nn.Module):
    """Final task-frequency-semantic fusion model."""

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        self.config = config
        self.core = CustomMultiscaleBackbone(
            channels=config.model.backbone.channels,
            depths=config.model.backbone.depths,
            frequency=config.model.frequency,
            output_channels=config.model.output_channels,
            pad_multiple=config.model.pad_multiple,
            moe=config.model.moe,
            backbone_config=config.model.backbone,
        )
        assert self.core.task_embedding is not None
        self.feedback = ClosedLoopRefinement(
            config, self.core.task_embedding, self.core.moe_blocks
        )

    def forward(self, batch: FusionBatch) -> FusionOutput:
        backbone = self.core(batch)
        feedback = self.feedback(batch, backbone)
        return FusionOutput(
            fused=feedback.final,
            task=batch.task,
            coarse=feedback.coarse,
            refinement=feedback.refinement,
            spectral_statistics=(
                *backbone.spectral_statistics,
                *feedback.spectral_statistics,
            ),
            router_diagnostics=(
                *backbone.router_diagnostics,
                *feedback.router_diagnostics,
            ),
            auxiliary=feedback.auxiliary,
            focus=feedback.debug.get("focus"),
            coarse_segmentation=feedback.coarse_semantic,
            segmentation=feedback.final_semantic,
            debug={**backbone.debug, **feedback.debug},
        )

    def generate_all_tasks(
        self, batch: FusionBatch, tasks: tuple[TaskType, ...] | None = None
    ) -> dict[str, FusionOutput]:
        if tasks is None:
            tasks = (
                (TaskType.VIF, TaskType.SEG) if batch.has_infrared else (TaskType.MFIF,)
            )
        outputs = {}
        for task in tasks:
            task_batch = FusionBatch(
                batch.source_a,
                batch.source_b,
                task,
                batch.sample_ids,
                batch.target,
                batch.focus_target,
                batch.segmentation_target,
                batch.metadata,
            )
            outputs[task.value] = self(task_batch)
        return outputs


def build_model(config: ProjectConfig) -> TFSMoEFusion:
    return TFSMoEFusion(config)

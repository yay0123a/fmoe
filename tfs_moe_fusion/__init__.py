"""Task-frequency-semantic mixture-of-experts image fusion."""

from .config import ProjectConfig, load_config
from .model import TFSMoEFusion, build_model
from .types import FusionBatch, FusionOutput, ModalityType, TaskType

__all__ = [
    "FusionBatch",
    "FusionOutput",
    "ModalityType",
    "ProjectConfig",
    "TFSMoEFusion",
    "TaskType",
    "build_model",
    "load_config",
]

__version__ = "0.1.0"

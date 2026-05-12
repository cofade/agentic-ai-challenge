"""App configuration loaded from config.yaml (issue #27, ADR-007).

All models carry sensible defaults so ``config.yaml`` is optional — a missing
file returns an ``AppConfig`` with the same values as the shipped YAML.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class RubricWeightsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_quality: float = 0.5
    metadata_completeness: float = 0.5
    penalty_per_field: float = 0.25


class ConfidenceThresholdsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grounding: float = 0.4


class ConfidenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rubric_weights: RubricWeightsConfig = RubricWeightsConfig()
    thresholds: ConfidenceThresholdsConfig = ConfidenceThresholdsConfig()


class ChatConfig(BaseModel):
    """Knobs for the interactive chat REPL (Phase 8 — issue #65)."""

    model_config = ConfigDict(extra="forbid")

    soft_turn_warning_at: int = 8


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence: ConfidenceConfig = ConfidenceConfig()
    chat: ChatConfig = ChatConfig()


def load_app_config(path: Path = Path("config.yaml")) -> AppConfig:
    """Load ``AppConfig`` from *path*.

    Returns an ``AppConfig`` with all defaults when the file does not exist.
    """
    if not path.exists():
        return AppConfig()
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return AppConfig.model_validate(data or {})


__all__ = [
    "AppConfig",
    "ChatConfig",
    "ConfidenceConfig",
    "ConfidenceThresholdsConfig",
    "RubricWeightsConfig",
    "load_app_config",
]

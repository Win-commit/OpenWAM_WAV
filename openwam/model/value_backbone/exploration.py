"""Noise-distribution updates for value-guided three-stream exploration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class ExplorationConfig:
    enabled: bool = False
    steps: int = 3
    candidates: int = 8
    elite_fraction: float = 0.25
    alpha_smooth: float = 0.9
    sigma_decay: float = 0.5
    min_std: float = 0.05

    @classmethod
    def from_mapping(cls, value: Mapping | None) -> "ExplorationConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("exploration must be a mapping")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown exploration options: {sorted(unknown)}")
        if not isinstance(value.get("enabled", False), bool):
            raise ValueError("exploration.enabled must be a boolean")
        for name in ("steps", "candidates"):
            item = value.get(name, getattr(cls, name))
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"exploration.{name} must be an integer")
        config = cls(**value)
        if config.steps < 1 or config.candidates < 2:
            raise ValueError("exploration.steps must be >= 1 and candidates must be >= 2")
        for name in ("elite_fraction", "alpha_smooth", "sigma_decay", "min_std"):
            number = float(getattr(config, name))
            if not math.isfinite(number):
                raise ValueError(f"exploration.{name} must be finite")
        if not 0 < config.elite_fraction <= 1:
            raise ValueError("exploration.elite_fraction must be in (0, 1]")
        if not 0 <= config.alpha_smooth < 1:
            raise ValueError("exploration.alpha_smooth must be in [0, 1)")
        if not 0 < config.sigma_decay <= 1 or config.min_std <= 0:
            raise ValueError("exploration.sigma_decay must be in (0, 1] and min_std must be positive")
        if math.ceil(config.candidates * config.elite_fraction) < 2:
            raise ValueError("exploration requires at least two elite candidates")
        return config


def update_distribution(samples: torch.Tensor, scores: torch.Tensor, mean: torch.Tensor,
                        std: torch.Tensor, config: ExplorationConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth the sampling distribution toward the highest-scoring initial noises."""
    if samples.shape[0] != config.candidates or scores.shape != (config.candidates,):
        raise ValueError("exploration samples/scores do not match candidates")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("exploration scores must be finite")
    count = math.ceil(config.candidates * config.elite_fraction)
    elite = samples[torch.topk(scores, k=count).indices]
    elite_mean = elite.float().mean(dim=0)
    elite_std = elite.float().std(dim=0, unbiased=False) * config.sigma_decay
    updated_mean = config.alpha_smooth * mean.float() + (1 - config.alpha_smooth) * elite_mean
    updated_std = config.alpha_smooth * std.float() + (1 - config.alpha_smooth) * elite_std
    return updated_mean.to(mean.dtype), updated_std.clamp_min(config.min_std).to(std.dtype)

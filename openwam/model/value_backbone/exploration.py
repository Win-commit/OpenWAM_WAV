"""Noise-distribution updates for value-guided three-stream exploration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class ExplorationConfig:
    enabled: bool = False
    explore_steps: int = 3
    dynamic_groups: int = 8
    value_groups: int = 1
    candidate_batch_size: int = 8
    sigma_decay: float = 0.5
    alpha_smooth: float = 0.9
    value_elites: float = 0.25
    dynamic_elites: float = 0.25
    min_std: float = 0.05

    @classmethod
    def from_mapping(cls, value: Mapping | None) -> "ExplorationConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("exploration must be a mapping")
        options = dict(value)
        legacy = {"steps": "explore_steps", "candidates": "dynamic_groups"}
        for old, new in legacy.items():
            if old in options:
                if new in options:
                    raise ValueError(f"exploration.{old} and {new} cannot both be set")
                options[new] = options.pop(old)
        if "elite_fraction" in options:
            elite = options.pop("elite_fraction")
            if "value_elites" in options or "dynamic_elites" in options:
                raise ValueError("exploration.elite_fraction conflicts with value_elites/dynamic_elites")
            options["value_elites"] = elite
            options["dynamic_elites"] = elite
        if "candidates" in value and "value_groups" not in options:
            options["value_groups"] = 1
        unknown = set(options) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown exploration options: {sorted(unknown)}")
        if not isinstance(options.get("enabled", False), bool):
            raise ValueError("exploration.enabled must be a boolean")
        for name in ("explore_steps", "dynamic_groups", "value_groups", "candidate_batch_size"):
            item = options.get(name, getattr(cls, name))
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise ValueError(f"exploration.{name} must be a positive integer")
        config = cls(**options)
        if config.dynamic_groups < 2:
            raise ValueError("exploration.dynamic_groups must be >= 2")
        for name in ("value_elites", "dynamic_elites", "alpha_smooth", "sigma_decay", "min_std"):
            try:
                number = float(getattr(config, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"exploration.{name} must be a finite number") from exc
            if not math.isfinite(number):
                raise ValueError(f"exploration.{name} must be finite")
        if not 0 < config.value_elites <= 1 or not 0 < config.dynamic_elites <= 1:
            raise ValueError("exploration elite fractions must be in (0, 1]")
        if not 0 <= config.alpha_smooth < 1:
            raise ValueError("exploration.alpha_smooth must be in [0, 1)")
        if not 0 < config.sigma_decay <= 1 or config.min_std <= 0:
            raise ValueError("exploration.sigma_decay must be in (0, 1] and min_std must be positive")
        return config


def update_distribution(samples: torch.Tensor, scores: torch.Tensor, mean: torch.Tensor,
                        std: torch.Tensor, config: ExplorationConfig,
                        elite_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth a sampling distribution toward its highest-scoring initial noises."""
    if samples.shape[0] < 1 or scores.shape != (samples.shape[0],):
        raise ValueError("exploration samples/scores have mismatched candidate counts")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("exploration scores must be finite")
    count = math.ceil(samples.shape[0] * elite_fraction)
    elite = samples[torch.topk(scores, k=count).indices]
    elite_mean = elite.float().mean(dim=0)
    elite_std = elite.float().std(dim=0, unbiased=False) * config.sigma_decay
    updated_mean = config.alpha_smooth * mean.float() + (1 - config.alpha_smooth) * elite_mean
    updated_std = config.alpha_smooth * std.float() + (1 - config.alpha_smooth) * elite_std
    return updated_mean.to(mean.dtype), updated_std.clamp_min(config.min_std).to(std.dtype)

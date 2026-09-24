"""Normalization helpers for the optional WAV value stream.

RoboDojo value targets are stored as raw discounted returns in the sidecar.
The reader Z-scores them before the flow model sees them; deployment needs the
inverse transform after ``symexp`` so candidate ranking is expressed in the
same return units as the offline targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ValueNormalizer:
    """Global per-channel Z-score normalizer with NumPy/Tensor support."""

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.ndim != 1 or std.shape != mean.shape or mean.size == 0:
            raise ValueError(
                f"value normalizer mean/std must be equal non-empty 1-D arrays, got {mean.shape}/{std.shape}"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("value normalizer mean/std must be finite and std must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @property
    def value_dim(self) -> int:
        return int(self.mean.size)

    def normalize(self, values):
        if isinstance(values, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
            std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
            return (values - mean) / std
        return (np.asarray(values) - self.mean) / self.std

    def unnormalize(self, values):
        if isinstance(values, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
            std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
            return values * std + mean
        return np.asarray(values) * self.std + self.mean

    @classmethod
    def from_stats_file(cls, path: str | Path, *, expected_dim: int | None = None) -> "ValueNormalizer":
        """Load a validated ``robodojo_value_stats.json`` artifact."""
        stats_path = Path(path)
        try:
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read value normalization stats from {stats_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{stats_path}: value stats payload must be an object")
        normalizer = cls(
            mean=np.asarray(payload.get("mean"), dtype=np.float32),
            std=np.asarray(payload.get("std"), dtype=np.float32),
        )
        saved_dim = payload.get("value_dim", normalizer.value_dim)
        if int(saved_dim) != normalizer.value_dim:
            raise ValueError(
                f"{stats_path}: value_dim={saved_dim} disagrees with mean/std width {normalizer.value_dim}"
            )
        if expected_dim is not None and normalizer.value_dim != int(expected_dim):
            raise ValueError(
                f"{stats_path}: value stats width {normalizer.value_dim} does not match model value_dim={expected_dim}"
            )
        return normalizer


__all__ = ["ValueNormalizer"]

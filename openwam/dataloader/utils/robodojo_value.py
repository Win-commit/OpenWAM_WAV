"""Offline sparse value targets for formal RoboDojo HDF5 episodes.

RoboDojo's released HDF5 files have observations, state and language but no
reward/success signal.  WAV training therefore uses an explicitly labeled
offline demo proxy: each episode gets rewards ``[-1, ..., -1, 0]`` and its
discounted state values.  This module writes those derived tensors to a
*sidecar* tree; it never opens an HDF5 file for writing.

The proxy is intrinsically scalar.  ``value_dim>1`` is nevertheless supported
for an explicitly configured model interface by broadcasting that scalar into
each channel; the metadata records the width, and candidate scoring averages
the channels.  The shipped RoboDojo recipe deliberately uses ``value_dim=1``.

The sidecar format is deliberately small and strict:

* one ``.hdf5.value.npz`` per source episode, mirroring the source-relative
  path below a user-selected sidecar root;
* versioned JSON metadata embedded in the NPZ, including the source path,
  length, size/mtime and SHA-256 fingerprint;
* one global ``robodojo_value_stats.json`` with the exact Z-score transform
  consumed by the reader and copied into a deploy checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


VALUE_SIDECAR_VERSION = 1
VALUE_TARGET_KIND = "sparse_terminal_demo_proxy"
DEFAULT_VALUE_GAMMA = 0.99
DEFAULT_VALUE_DIM = 1
VALUE_STATS_FILENAME = "robodojo_value_stats.json"


def sparse_terminal_rewards(length: int, *, value_dim: int = DEFAULT_VALUE_DIM) -> np.ndarray:
    """Return the documented terminal-demo proxy reward ``[-1,…,-1,0]``."""
    length = int(length)
    value_dim = int(value_dim)
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if value_dim <= 0:
        raise ValueError(f"value_dim must be positive, got {value_dim}")
    rewards = np.full((length, value_dim), -1.0, dtype=np.float32)
    if length:
        rewards[-1] = 0.0
    return rewards


def discounted_returns(rewards: np.ndarray, *, gamma: float = DEFAULT_VALUE_GAMMA) -> np.ndarray:
    """Compute state values by reverse discounted accumulation."""
    values = np.asarray(rewards, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"rewards must have shape [T, D], got {values.shape}")
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be finite and in (0, 1], got {gamma!r}")
    result = np.zeros_like(values)
    for index in range(values.shape[0] - 1, -1, -1):
        if index == values.shape[0] - 1:
            result[index] = values[index]
        else:
            result[index] = values[index] + gamma * result[index + 1]
    return result


def sparse_terminal_values(
    length: int,
    *,
    gamma: float = DEFAULT_VALUE_GAMMA,
    value_dim: int = DEFAULT_VALUE_DIM,
) -> np.ndarray:
    """Build raw ``V(s_t)`` values for one RoboDojo episode."""
    return discounted_returns(sparse_terminal_rewards(length, value_dim=value_dim), gamma=gamma)


def source_fingerprint(path: str | Path, *, include_sha256: bool = True) -> dict[str, Any]:
    """Return source identity used by a sidecar.

    Dataset workers use the cheap size/mtime fields on every reader startup;
    the offline converter additionally verifies SHA-256 before reusing an
    existing sidecar.  That split catches normal source changes without making
    every worker hash multi-GB HDF5 files.
    """
    source = Path(path)
    stat = source.stat()
    fingerprint: dict[str, Any] = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        fingerprint["sha256"] = digest.hexdigest()
    return fingerprint


def value_sidecar_path(
    source_path: str | Path,
    *,
    dataset_root: str | Path,
    sidecar_root: str | Path,
) -> Path:
    """Map a formal source episode to its non-mutating sidecar path."""
    source = Path(source_path).resolve()
    root = Path(dataset_root).resolve()
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source episode {source} is not below dataset_root {root}") from exc
    return Path(sidecar_root).resolve() / relative.with_suffix(relative.suffix + ".value.npz")


def default_value_stats_path(sidecar_root: str | Path) -> Path:
    return Path(sidecar_root).resolve() / VALUE_STATS_FILENAME


def _source_relative_path(source_path: str | Path, dataset_root: str | Path) -> str:
    try:
        return Path(source_path).resolve().relative_to(Path(dataset_root).resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"source episode {source_path} is not below dataset_root {dataset_root}") from exc


def _validate_values(values: np.ndarray, *, expected_dim: int | None = None, label: str = "values") -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError(f"{label} must have non-empty shape [T, D], got {values.shape}")
    if expected_dim is not None and values.shape[1] != int(expected_dim):
        raise ValueError(f"{label} width must be {expected_dim}, got {values.shape[1]}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values")
    return values


def _atomic_write_npz(path: Path, *, values: np.ndarray, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(
                handle,
                values=np.asarray(values, dtype=np.float32),
                metadata_json=np.asarray(json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))),
            )
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_value_sidecar(
    source_path: str | Path,
    *,
    dataset_root: str | Path,
    sidecar_root: str | Path,
    values: np.ndarray,
    gamma: float = DEFAULT_VALUE_GAMMA,
    value_dim: int = DEFAULT_VALUE_DIM,
    include_source_sha256: bool = True,
) -> Path:
    """Atomically write one source-validated value sidecar."""
    source = Path(source_path)
    values = _validate_values(values, expected_dim=value_dim)
    gamma = float(gamma)
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    fingerprint = source_fingerprint(source, include_sha256=include_source_sha256)
    metadata = {
        "format_version": VALUE_SIDECAR_VERSION,
        "target_kind": VALUE_TARGET_KIND,
        "value_dim": int(value_dim),
        "gamma": gamma,
        "episode_length": int(values.shape[0]),
        "source_relative_path": _source_relative_path(source, dataset_root),
        "source_fingerprint": fingerprint,
    }
    output = value_sidecar_path(source, dataset_root=dataset_root, sidecar_root=sidecar_root)
    _atomic_write_npz(output, values=values, metadata=metadata)
    return output


def _read_metadata(npz, path: Path) -> dict[str, Any]:
    if "metadata_json" not in npz:
        raise ValueError(f"{path}: missing metadata_json")
    raw = npz["metadata_json"]
    try:
        text = str(raw.item())
        metadata = json.loads(text)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid metadata_json") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata_json must encode an object")
    return metadata


def read_value_sidecar(
    path: str | Path,
    *,
    source_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    expected_length: int | None = None,
    expected_gamma: float | None = None,
    expected_value_dim: int = DEFAULT_VALUE_DIM,
    verify_source_sha256: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read and validate a value sidecar without touching source HDF5 content."""
    sidecar = Path(path)
    try:
        with np.load(sidecar, allow_pickle=False) as archive:
            if "values" not in archive:
                raise ValueError(f"{sidecar}: missing values")
            values = np.asarray(archive["values"], dtype=np.float32)
            metadata = _read_metadata(archive, sidecar)
    except (OSError, ValueError, EOFError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(str(sidecar)):
            raise
        raise ValueError(f"could not read RoboDojo value sidecar {sidecar}") from exc

    values = _validate_values(values, expected_dim=expected_value_dim, label=f"{sidecar}:values")
    if metadata.get("format_version") != VALUE_SIDECAR_VERSION:
        raise ValueError(
            f"{sidecar}: unsupported format_version={metadata.get('format_version')!r}; "
            f"expected {VALUE_SIDECAR_VERSION}"
        )
    if metadata.get("target_kind") != VALUE_TARGET_KIND:
        raise ValueError(f"{sidecar}: unexpected target_kind={metadata.get('target_kind')!r}")
    if int(metadata.get("value_dim", -1)) != expected_value_dim:
        raise ValueError(
            f"{sidecar}: metadata value_dim={metadata.get('value_dim')!r} does not match {expected_value_dim}"
        )
    if int(metadata.get("episode_length", -1)) != values.shape[0]:
        raise ValueError(f"{sidecar}: metadata episode_length does not match values length")
    if expected_length is not None and values.shape[0] != int(expected_length):
        raise ValueError(f"{sidecar}: values length {values.shape[0]} does not match source length {expected_length}")
    if expected_gamma is not None:
        recorded = metadata.get("gamma")
        if not isinstance(recorded, (int, float)) or not math.isclose(
            float(recorded), float(expected_gamma), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"{sidecar}: gamma={recorded!r} does not match requested value_gamma={expected_gamma!r}"
            )

    if source_path is not None:
        if dataset_root is None:
            raise ValueError("dataset_root is required when validating a sidecar source")
        expected_relative = _source_relative_path(source_path, dataset_root)
        if metadata.get("source_relative_path") != expected_relative:
            raise ValueError(
                f"{sidecar}: source_relative_path={metadata.get('source_relative_path')!r} "
                f"does not match {expected_relative!r}"
            )
        recorded_fp = metadata.get("source_fingerprint")
        if not isinstance(recorded_fp, Mapping):
            raise ValueError(f"{sidecar}: source_fingerprint is required")
        current_fp = source_fingerprint(source_path, include_sha256=verify_source_sha256)
        for key in ("size", "mtime_ns"):
            if int(recorded_fp.get(key, -1)) != int(current_fp[key]):
                raise ValueError(
                    f"{sidecar}: source {key} changed; regenerate value sidecars before training"
                )
        if verify_source_sha256:
            if recorded_fp.get("sha256") != current_fp.get("sha256"):
                raise ValueError(f"{sidecar}: source SHA-256 changed; regenerate value sidecars before training")
    return values, metadata


def compute_value_stats(values_iter: Iterable[np.ndarray]) -> dict[str, Any]:
    """Compute stable global population mean/std without retaining all values."""
    count = 0
    mean = None
    m2 = None
    minimum = None
    maximum = None
    for values in values_iter:
        values = _validate_values(values)
        data = values.astype(np.float64, copy=False)
        n = int(data.shape[0])
        batch_mean = data.mean(axis=0)
        batch_m2 = ((data - batch_mean) ** 2).sum(axis=0)
        if count == 0:
            count, mean, m2 = n, batch_mean, batch_m2
            minimum, maximum = data.min(axis=0), data.max(axis=0)
            continue
        assert mean is not None and m2 is not None and minimum is not None and maximum is not None
        delta = batch_mean - mean
        total = count + n
        mean = mean + delta * (n / total)
        m2 = m2 + batch_m2 + delta * delta * (count * n / total)
        count = total
        minimum = np.minimum(minimum, data.min(axis=0))
        maximum = np.maximum(maximum, data.max(axis=0))
    if count == 0 or mean is None or m2 is None or minimum is None or maximum is None:
        raise ValueError("cannot compute value stats from an empty episode collection")
    std = np.maximum(np.sqrt(m2 / count), 1e-6)
    return {
        "count": int(count),
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "min": minimum.astype(np.float32).tolist(),
        "max": maximum.astype(np.float32).tolist(),
    }


def write_value_stats(
    sidecar_root: str | Path,
    *,
    values_iter: Iterable[np.ndarray],
    gamma: float,
    tasks: Sequence[str],
    dataset_root: str | Path,
    value_dim: int = DEFAULT_VALUE_DIM,
) -> Path:
    """Write the one global Z-score artifact consumed by training/deploy."""
    stats = compute_value_stats(values_iter)
    if len(stats["mean"]) != int(value_dim):
        raise ValueError(f"computed value stats width {len(stats['mean'])} does not match value_dim={value_dim}")
    payload = {
        "format_version": VALUE_SIDECAR_VERSION,
        "target_kind": VALUE_TARGET_KIND,
        "normalization": "z-score",
        "value_dim": int(value_dim),
        "gamma": float(gamma),
        "tasks": sorted(str(task) for task in tasks),
        "dataset_root_name": Path(dataset_root).resolve().name,
        **stats,
    }
    output = default_value_stats_path(sidecar_root)
    _atomic_write_json(output, payload)
    return output


def load_value_stats(
    path: str | Path,
    *,
    expected_gamma: float | None = None,
    expected_value_dim: int = DEFAULT_VALUE_DIM,
    expected_tasks: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load strict global value stats and return NumPy mean/std vectors."""
    stats_path = Path(path)
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read RoboDojo value stats from {stats_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{stats_path}: value stats payload must be an object")
    if payload.get("format_version") != VALUE_SIDECAR_VERSION:
        raise ValueError(f"{stats_path}: unsupported value stats version")
    if payload.get("target_kind") != VALUE_TARGET_KIND:
        raise ValueError(f"{stats_path}: unexpected value target kind")
    if payload.get("normalization") != "z-score":
        raise ValueError(f"{stats_path}: expected z-score normalization")
    if int(payload.get("value_dim", -1)) != int(expected_value_dim):
        raise ValueError(
            f"{stats_path}: value_dim={payload.get('value_dim')!r} does not match {expected_value_dim}"
        )
    if expected_gamma is not None and not math.isclose(
        float(payload.get("gamma", float("nan"))), float(expected_gamma), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{stats_path}: gamma does not match requested value_gamma={expected_gamma}")
    mean = np.asarray(payload.get("mean"), dtype=np.float32)
    std = np.asarray(payload.get("std"), dtype=np.float32)
    if mean.shape != (int(expected_value_dim),) or std.shape != mean.shape:
        raise ValueError(f"{stats_path}: mean/std must have shape ({expected_value_dim},)")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError(f"{stats_path}: mean/std must be finite and std positive")
    if int(payload.get("count", 0)) <= 0:
        raise ValueError(f"{stats_path}: count must be positive")
    if expected_tasks is not None:
        expected = sorted(str(task) for task in expected_tasks)
        if payload.get("tasks") != expected:
            raise ValueError(
                f"{stats_path}: tasks={payload.get('tasks')!r} does not match selected RoboDojo tasks {expected!r}; "
                "regenerate one global sidecar set for this corpus"
            )
    result = dict(payload)
    result["mean"] = mean
    result["std"] = std
    return result


def build_robodojo_value_sidecars(
    dataset_dir: str | Path,
    sidecar_root: str | Path,
    *,
    tasks: Sequence[str] | None = None,
    embodiment: str = "arx_x5",
    variant: str = "sim",
    gamma: float = DEFAULT_VALUE_GAMMA,
    value_dim: int = DEFAULT_VALUE_DIM,
    overwrite: bool = False,
    include_source_sha256: bool = True,
) -> Path:
    """Materialize sparse return sidecars for a formal RoboDojo corpus.

    The reader imports this function only through the standalone CLI; delaying
    the RoboDojo reader import keeps the target math independently auditable
    and avoids a data-loader import cycle.
    """
    from openwam.dataloader.robodojo import (
        discover_episodes,
        resolve_robodojo_tasks,
        validate_robodojo_episode,
    )

    root = Path(dataset_dir).resolve()
    sidecar_root = Path(sidecar_root).resolve()
    gamma = float(gamma)
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    if int(value_dim) <= 0:
        raise ValueError(f"value_dim must be positive, got {value_dim}")
    selected_tasks = resolve_robodojo_tasks(root, tasks=tasks, embodiment=embodiment, variant=variant)
    sidecar_paths: list[Path] = []
    for task in selected_tasks:
        for source in discover_episodes(root, task, embodiment=embodiment, variant=variant):
            metadata = validate_robodojo_episode(source, variant=variant)
            length = int(metadata["length"])
            output = value_sidecar_path(source, dataset_root=root, sidecar_root=sidecar_root)
            values = None
            if output.is_file() and not overwrite:
                try:
                    values, _ = read_value_sidecar(
                        output,
                        source_path=source,
                        dataset_root=root,
                        expected_length=length,
                        expected_gamma=gamma,
                        expected_value_dim=value_dim,
                        verify_source_sha256=include_source_sha256,
                    )
                except ValueError:
                    # A stale/partial sidecar is safe to replace: derived data
                    # only, never source HDF5.
                    values = None
            if values is None:
                values = sparse_terminal_values(length, gamma=gamma, value_dim=value_dim)
                output = write_value_sidecar(
                    source,
                    dataset_root=root,
                    sidecar_root=sidecar_root,
                    values=values,
                    gamma=gamma,
                    value_dim=value_dim,
                    include_source_sha256=include_source_sha256,
                )
            sidecar_paths.append(output)

    def _iter_values():
        # Re-read the compact sidecars instead of retaining an entire corpus
        # of episode returns in RAM just to compute global statistics.
        for sidecar_path in sidecar_paths:
            values, _metadata = read_value_sidecar(
                sidecar_path,
                expected_gamma=gamma,
                expected_value_dim=value_dim,
            )
            yield values

    return write_value_stats(
        sidecar_root,
        values_iter=_iter_values(),
        gamma=gamma,
        tasks=selected_tasks,
        dataset_root=root,
        value_dim=value_dim,
    )


__all__ = [
    "DEFAULT_VALUE_DIM",
    "DEFAULT_VALUE_GAMMA",
    "VALUE_SIDECAR_VERSION",
    "VALUE_STATS_FILENAME",
    "VALUE_TARGET_KIND",
    "build_robodojo_value_sidecars",
    "compute_value_stats",
    "default_value_stats_path",
    "discounted_returns",
    "load_value_stats",
    "read_value_sidecar",
    "source_fingerprint",
    "sparse_terminal_rewards",
    "sparse_terminal_values",
    "value_sidecar_path",
    "write_value_sidecar",
    "write_value_stats",
]

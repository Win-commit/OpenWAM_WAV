"""Regression coverage for non-mutating RoboDojo WAV value sidecars."""

from __future__ import annotations

import numpy as np

from openwam.dataloader.robodojo import RoboDojoDataset
from openwam.dataloader.utils.robodojo_value import (
    build_robodojo_value_sidecars,
    read_value_sidecar,
    sparse_terminal_rewards,
    sparse_terminal_values,
    value_sidecar_path,
)
from tests.test_robodojo_dataloader import write_episode


def test_sparse_terminal_proxy_and_reader_next_state_alignment(tmp_path):
    """Sidecars leave HDF5 untouched and reader targets match action t+1 states."""
    source = write_episode(tmp_path, task="pick_mug", T=5)
    source_before = source.read_bytes()
    sidecar_root = tmp_path / "wav_values"
    stats_path = build_robodojo_value_sidecars(tmp_path, sidecar_root, gamma=0.9)
    assert stats_path.is_file()
    assert source.read_bytes() == source_before

    sidecar = value_sidecar_path(source, dataset_root=tmp_path, sidecar_root=sidecar_root)
    values, metadata = read_value_sidecar(
        sidecar,
        source_path=source,
        dataset_root=tmp_path,
        expected_length=5,
        expected_gamma=0.9,
    )
    assert np.array_equal(sparse_terminal_rewards(5), np.asarray([[-1], [-1], [-1], [-1], [0]], np.float32))
    assert np.allclose(values, sparse_terminal_values(5, gamma=0.9))
    # The RoboDojo proxy is scalar by construction, but a configured model
    # width remains supported by broadcasting the same target per channel.
    assert sparse_terminal_values(3, gamma=0.9, value_dim=10).shape == (3, 10)
    assert metadata["target_kind"] == "sparse_terminal_demo_proxy"

    dataset = RoboDojoDataset(
        tmp_path,
        task_name="pick_mug",
        num_frames=4,
        multiview=False,
        normalize_mode=None,
        unify_action=False,
        include_value_targets=True,
        value_sidecar_root=sidecar_root,
        value_gamma=0.9,
    )
    sample = dataset[0]
    mean = dataset.value_normalization_stats["mean"]
    std = dataset.value_normalization_stats["std"]
    reconstructed_raw = sample["value"].numpy() * std + mean
    # Window start=0: actions are state[1:4], therefore values are V(s_1:s_4),
    # never the conditioned state V(s_0).
    assert np.all(sample["value_mask"].numpy())
    assert np.allclose(reconstructed_raw, values[1:4])

    # Last valid start has exactly one real next-state target; padded values are
    # masked so their zero fill cannot affect the loss.
    tail = dataset[len(dataset) - 1]
    assert tail["value_mask"].shape == (3, 1)
    assert tail["value_mask"][0].item() is True
    assert tail["value_mask"][1:].any().item() is False


def test_fast_fingerprint_uses_source_metadata_without_sha(tmp_path):
    source = write_episode(tmp_path, task="pick_mug", T=5)
    sidecar_root = tmp_path / "fast_values"
    build_robodojo_value_sidecars(tmp_path, sidecar_root, include_source_sha256=False)
    sidecar = value_sidecar_path(source, dataset_root=tmp_path, sidecar_root=sidecar_root)
    values, metadata = read_value_sidecar(
        sidecar,
        source_path=source,
        dataset_root=tmp_path,
        expected_length=5,
    )
    assert values.shape == (5, 1)
    assert "sha256" not in metadata["source_fingerprint"]

"""Checks for grouped value exploration settings and elite updates."""

import pytest
import torch

from openwam.model.value_backbone.exploration import ExplorationConfig, update_distribution


def test_wav_style_defaults_and_legacy_options():
    config = ExplorationConfig.from_mapping({"enabled": True})
    assert (config.explore_steps, config.dynamic_groups, config.value_groups) == (3, 8, 1)
    assert (config.value_elites, config.dynamic_elites, config.candidate_batch_size) == (0.25, 0.25, 8)
    legacy = ExplorationConfig.from_mapping({"steps": 3, "candidates": 8, "elite_fraction": 0.25})
    assert (legacy.explore_steps, legacy.dynamic_groups, legacy.value_groups) == (3, 8, 1)
    assert (legacy.value_elites, legacy.dynamic_elites) == (0.25, 0.25)


@pytest.mark.parametrize("option", [
    {"candidate_batch_size": 0}, {"dynamic_groups": 1}, {"value_groups": -1},
    {"value_elites": 0}, {"dynamic_elites": 1.1}, {"sigma_decay": float("nan")},
])
def test_invalid_grouped_search_options_rejected(option):
    with pytest.raises(ValueError):
        ExplorationConfig.from_mapping(option)


def test_video_and_value_elites_can_use_different_scores_and_fractions():
    config = ExplorationConfig.from_mapping({
        "dynamic_groups": 4, "value_groups": 2, "dynamic_elites": 0.5,
        "value_elites": 0.25, "alpha_smooth": 0.5, "sigma_decay": 0.5,
        "min_std": 0.1,
    })
    video_samples = torch.tensor([[[-2.0]], [[-1.0]], [[2.0]], [[4.0]]])
    value_samples = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1)
    scores = torch.tensor([-10.0, -5.0, 1.0, 2.0, -1.0, -2.0, 4.0, 3.0])
    video_scores = scores.reshape(4, 2).max(dim=1).values
    video_mean, video_std = update_distribution(
        video_samples, video_scores, torch.zeros(1, 1), torch.ones(1, 1),
        config, config.dynamic_elites,
    )
    value_mean, value_std = update_distribution(
        value_samples, scores, torch.zeros(1, 1), torch.ones(1, 1),
        config, config.value_elites,
    )
    assert video_mean.item() == pytest.approx(0.75)
    assert video_std.item() == pytest.approx(1.125)
    assert value_mean.item() == pytest.approx(3.25)
    assert value_std.item() == pytest.approx(0.625)


def test_nonfinite_value_scores_fail_instead_of_corrupting_search():
    config = ExplorationConfig.from_mapping({"dynamic_groups": 4})
    with pytest.raises(ValueError, match="finite"):
        update_distribution(torch.zeros(4, 1), torch.tensor([0.0, 1.0, float("nan"), 2.0]),
                            torch.zeros(1), torch.ones(1), config, config.dynamic_elites)

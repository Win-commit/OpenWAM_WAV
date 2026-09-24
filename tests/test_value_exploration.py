"""Focused checks for the iterative value-guided search distribution."""

import pytest
import torch

from openwam.model.value_backbone.exploration import ExplorationConfig, update_distribution


def test_exploration_config_rejects_unusable_elite_count():
    with pytest.raises(ValueError, match="at least two elite"):
        ExplorationConfig.from_mapping({"enabled": True, "candidates": 4, "elite_fraction": 0.25})


def test_distribution_moves_toward_high_value_initial_noises():
    config = ExplorationConfig.from_mapping({
        "enabled": True, "candidates": 4, "elite_fraction": 0.5,
        "alpha_smooth": 0.5, "sigma_decay": 0.5, "min_std": 0.1,
    })
    samples = torch.tensor([[[-2.0]], [[-1.0]], [[2.0]], [[4.0]]])
    scores = torch.tensor([-10.0, -5.0, 1.0, 2.0])
    mean, std = update_distribution(samples, scores, torch.zeros(1, 1), torch.ones(1, 1), config)
    assert mean.item() == pytest.approx(1.5)
    assert std.item() == pytest.approx(0.75)


def test_nonfinite_value_scores_fail_instead_of_corrupting_search():
    config = ExplorationConfig.from_mapping({"enabled": True, "candidates": 4, "elite_fraction": 0.5})
    with pytest.raises(ValueError, match="finite"):
        update_distribution(torch.zeros(4, 1), torch.tensor([0.0, 1.0, float("nan"), 2.0]),
                            torch.zeros(1), torch.ones(1), config)

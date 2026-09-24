#!/usr/bin/env python
"""Render full-episode RoboDojo GT/prediction curves as JPEG figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ACTION_NAMES = (
    "left.x", "left.y", "left.z", "left.rot6d.0", "left.rot6d.1",
    "left.rot6d.2", "left.rot6d.3", "left.rot6d.4", "left.rot6d.5", "left.gripper",
    "right.x", "right.y", "right.z", "right.rot6d.0", "right.rot6d.1",
    "right.rot6d.2", "right.rot6d.3", "right.rot6d.4", "right.rot6d.5", "right.gripper",
)
GT_COLOR = "#5188de"
PRED_COLOR = "#ff6b55"
START_COLOR = "#a00000"


def _style_axis(axis, *, xmax: int) -> None:
    axis.grid(True, color="#d0d0d0", linestyle=":", linewidth=0.8)
    axis.set_xlim(1, xmax)
    axis.tick_params(labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_actions(
    output: Path,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    start_indices: np.ndarray,
    model_label: str,
) -> None:
    if ground_truth.shape != prediction.shape or ground_truth.ndim != 2 or ground_truth.shape[1] != 20:
        raise ValueError(f"expected matching (T, 20) action arrays, got {ground_truth.shape}, {prediction.shape}")
    x = np.arange(1, ground_truth.shape[0] + 1)
    starts = np.asarray(start_indices, dtype=np.int64)
    fig, axes = plt.subplots(5, 4, figsize=(22, 17), sharex=True)
    for dimension, axis in enumerate(axes.flat):
        axis.plot(x, ground_truth[:, dimension], color=GT_COLOR, linewidth=1.5, label="Ground Truth")
        axis.plot(x, prediction[:, dimension], color=PRED_COLOR, linewidth=1.3, linestyle="--", label="Inferred")
        axis.scatter(x[starts], ground_truth[starts, dimension], color="blue", s=16, zorder=5, label="GT Start")
        axis.scatter(x[starts], prediction[starts, dimension], color=START_COLOR, marker="x", s=28, linewidth=1.2, zorder=6, label="Inferred Start")
        axis.set_title(f"Dimension-{dimension} ({ACTION_NAMES[dimension]})", fontsize=10)
        _style_axis(axis, xmax=int(x[-1]))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.975), ncol=4, fontsize=10)
    fig.suptitle(f"Comparison of Ground Truth and Inferred Actions — {model_label}", fontsize=21, y=0.997)
    fig.supxlabel("Continuous Timestep (across all open-loop windows)", fontsize=12)
    fig.supylabel("Value", fontsize=12)
    fig.tight_layout(rect=(0.02, 0.035, 0.99, 0.955), h_pad=1.1, w_pad=1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, pil_kwargs={"quality": 94})
    plt.close(fig)


def plot_values(
    output: Path,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    start_indices: np.ndarray,
) -> None:
    ground_truth = np.asarray(ground_truth).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    if ground_truth.shape != prediction.shape:
        raise ValueError(f"value lengths disagree: {ground_truth.shape}, {prediction.shape}")
    x = np.arange(1, ground_truth.shape[0] + 1)
    starts = np.asarray(start_indices, dtype=np.int64)
    fig, axis = plt.subplots(figsize=(16, 7))
    axis.plot(x, ground_truth, color=GT_COLOR, linewidth=2, label="Ground Truth")
    axis.plot(x, prediction, color=PRED_COLOR, linewidth=1.8, linestyle="--", label="Inferred")
    axis.scatter(x[starts], ground_truth[starts], color="blue", s=46, zorder=5, label="GT Start")
    axis.scatter(x[starts], prediction[starts], color=START_COLOR, marker="x", s=57, linewidth=1.5, zorder=6, label="Inferred Start")
    _style_axis(axis, xmax=int(x[-1]))
    axis.set_title("Value Prediction Comparison", fontsize=17)
    axis.set_xlabel("Timestep", fontsize=12)
    axis.set_ylabel("Value", fontsize=12)
    axis.legend(loc="upper right", fontsize=11)
    fig.suptitle("Comparison of Ground Truth and Inferred Values", fontsize=22, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=175, pil_kwargs={"quality": 94})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--action-output", type=Path, required=True)
    parser.add_argument("--value-output", type=Path)
    parser.add_argument("--model-label", required=True)
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as data:
        starts = data["start_indices"]
        plot_actions(args.action_output, data["ground_truth_actions"], data["predicted_actions"], starts, args.model_label)
        if args.value_output is not None:
            plot_values(args.value_output, data["ground_truth_values"], data["predicted_values"], starts)


if __name__ == "__main__":
    main()

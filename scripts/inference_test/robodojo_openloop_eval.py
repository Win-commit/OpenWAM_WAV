#!/usr/bin/env python
"""Run one GT-conditioned open-loop prediction window across a full RoboDojo episode.

Every checkpoint is loaded once. At each 32-action window the model receives
the recorded first image/state and the episode instruction. Windows cover all
next-state action/value targets, including the final overlapping window.
WAV can optionally run iterative value-guided exploration in each window.
Visual outputs under ``vis/`` contain JPEG or MP4 files only.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DATASET_ROOT = Path("/zhaohan/lirunze/RoboDojo/.cache/robodojo_data_modelscope_hdf5_repo/data/RoboDojo")
VALUE_SIDECAR_ROOT = Path("/zhaohan/lirunze/RoboDojo/data/openwam_value_sidecars")
DEFAULT_EPISODE = DATASET_ROOT / "sweep_blocks/arx_x5/data/episode_0000000.hdf5"
WAV_CHECKPOINT = PROJECT_ROOT / "outputs/robodojo_wav_value/2026-09-22_23-15-46/checkpoint_step_60000.safetensors"
BASE_CHECKPOINT = PROJECT_ROOT / "outputs/robodojo_openwam_base/2026-09-23_12-19-56/checkpoint_step_60000.safetensors"
WINDOW_ACTIONS = 32
VIDEO_STRIDE = 4
VIDEO_FRAMES = WINDOW_ACTIONS // VIDEO_STRIDE + 1
CANVAS_HEIGHT = 384
CANVAS_WIDTH = 320
CAMERA_LAYOUT = ("cam_head", "cam_left_wrist", "cam_right_wrist")
MODEL_SPECS = (
    ("robodojo_wav_value_step60000", WAV_CHECKPOINT, True),
    ("robodojo_openwam_base_step60000", BASE_CHECKPOINT, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--sidecar-root", type=Path, default=VALUE_SIDECAR_ROOT)
    parser.add_argument("--wav-checkpoint", type=Path, default=WAV_CHECKPOINT)
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "vis")
    parser.add_argument("--run-name", default=None, help="Unique model directory name under output-root.")
    parser.add_argument("--plot-python", type=Path, default=Path("/zhaohan/miniconda3/envs/robodojo/bin/python"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--fps", type=float, default=25.0, help="Raw episode playback FPS.")
    parser.add_argument("--model", choices=("all", "wav", "base"), default="all")
    parser.add_argument("--explore", action="store_true", help="Use iterative value-guided search for WAV.")
    parser.add_argument("--value-selection", choices=("first", "mean", "last"), default="mean")
    parser.add_argument("--explore-steps", type=int, default=3)
    parser.add_argument("--dynamic-groups", type=int, default=8)
    parser.add_argument("--value-groups", type=int, default=1)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--sigma-decay", type=float, default=0.5)
    parser.add_argument("--alpha-smooth", type=float, default=0.9)
    parser.add_argument("--value-elites", type=float, default=0.25)
    parser.add_argument("--dynamic-elites", type=float, default=0.25)
    parser.add_argument("--min-std", type=float, default=0.05)
    return parser.parse_args()


def window_starts(episode_length: int) -> list[int]:
    if episode_length < WINDOW_ACTIONS + 1:
        raise ValueError(f"episode length {episode_length} must be at least {WINDOW_ACTIONS + 1}")
    starts = list(range(0, episode_length - WINDOW_ACTIONS, WINDOW_ACTIONS))
    if starts[-1] + WINDOW_ACTIONS < episode_length - 1:
        starts.append(episode_length - WINDOW_ACTIONS - 1)
    return starts


def decode_canvas(handle: h5py.File, frame_index: int) -> Image.Image:
    from openwam.dataloader.robodojo import _decode_jpeg
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout

    frames = {
        camera: _decode_jpeg(
            handle[f"vision/{camera}/colors"][frame_index],
            source=f"{handle.filename}:vision/{camera}/colors[{frame_index}]",
        )
        for camera in CAMERA_LAYOUT
    }
    return assemble_multiview_layout(frames, list(CAMERA_LAYOUT), CANVAS_HEIGHT, CANVAS_WIDTH)


def load_episode(args: argparse.Namespace) -> dict:
    from openwam.dataloader.robodojo import read_calibrated_eef20
    from openwam.dataloader.robodojo_contract import resolve_robodojo_calibration
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference
    from openwam.dataloader.utils.robodojo_value import read_value_sidecar, value_sidecar_path

    episode = args.episode.resolve()
    dataset_root = args.dataset_root.resolve()
    episode.relative_to(dataset_root)
    with h5py.File(episode, "r") as handle:
        length = int(handle["state/left_ee_poses"].shape[0])
        starts = window_starts(length)
        raw_instruction = handle["instruction"][()]
        if isinstance(raw_instruction, np.ndarray):
            raw_instruction = raw_instruction.item()
        instruction = (
            bytes(raw_instruction).decode("utf-8").strip()
            if isinstance(raw_instruction, (bytes, np.bytes_))
            else str(raw_instruction).strip()
        )
        if not instruction:
            raise ValueError(f"empty instruction in {episode}")
        first_frames = {start: decode_canvas(handle, start) for start in starts}

    states = read_calibrated_eef20(
        episode,
        resolve_robodojo_calibration(),
        0,
        length,
        variant="sim",
        embodiment="arx_x5",
    )
    sidecar = value_sidecar_path(
        episode,
        dataset_root=dataset_root,
        sidecar_root=args.sidecar_root.resolve(),
    )
    values, metadata = read_value_sidecar(
        sidecar,
        source_path=episode,
        dataset_root=dataset_root,
        expected_length=length,
        expected_gamma=0.99,
        expected_value_dim=1,
    )
    if states.shape != (length, 20) or values.shape != (length, 1):
        raise ValueError(f"unexpected RoboDojo state/value shapes: {states.shape}, {values.shape}")
    return {
        "episode": episode,
        "length": length,
        "starts": starts,
        "first_frames": first_frames,
        "states": states.astype(np.float32, copy=False),
        "ground_truth_actions": states[1:].astype(np.float32, copy=False),
        "ground_truth_values": values[1:, 0].astype(np.float32, copy=False),
        "prompt": format_prompt_for_inference(instruction),
        "value_target_kind": metadata["target_kind"],
    }


def output_directories(output_root: Path, model_specs) -> dict[str, dict[str, Path]]:
    result = {}
    for model_name, _checkpoint, _value_enabled in model_specs:
        model_root = output_root / model_name
        paths = {kind: model_root / kind for kind in ("action", "video", "value")}
        if any(path.exists() and any(path.iterdir()) for path in paths.values()):
            raise FileExistsError(f"visualization files already exist under {model_root}; choose a fresh output root")
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        result[model_name] = paths
    return result


def load_engine(checkpoint: Path, args: argparse.Namespace, expect_value: bool):
    from openwam.deploy import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.server import merge_deploy_cfg

    checkpoint = checkpoint.resolve()
    training_cfg, architecture = load_from_checkpoint_dir(
        str(checkpoint.parent),
        device=args.device,
        ckpt_name=checkpoint.name,
    )
    dl = training_cfg.dataloader
    expected = {
        "type": "robodojo",
        "num_frames": WINDOW_ACTIONS + 1,
        "video_stride": VIDEO_STRIDE,
        "height": CANVAS_HEIGHT,
        "width": CANVAS_WIDTH,
        "multiview": True,
    }
    for key, required in expected.items():
        actual = OmegaConf.select(training_cfg, f"dataloader.{key}")
        if actual != required:
            raise ValueError(f"checkpoint dataloader.{key}={actual!r}; expected {required!r}")
    if int(architecture.action_dim) != 80:
        raise ValueError(f"checkpoint action head is {architecture.action_dim}D; expected unified 80D")
    is_value = getattr(architecture, "value_backbone", None) is not None
    if is_value != expect_value:
        raise ValueError(f"checkpoint value backbone enabled={is_value}; expected={expect_value}")

    deploy_cfg = OmegaConf.create(
        {
            "inference": {
                "denoise_steps": int(args.denoise_steps),
                "denoise_mode": "sync",
                "inference_mode": "sync",
                "value_candidates": 1,
                "value_selection": args.value_selection,
                "value_num_frames": WINDOW_ACTIONS,
            },
            "optimization": {
                "decode_video": True,
                "dit_cache": {"enabled": False},
                "compile": {"enabled": False},
                "prompt_embed_cache": {"enabled": True, "maxsize": 2},
            },
        }
    )
    cfg = merge_deploy_cfg(training_cfg, deploy_cfg)
    return JointInferenceEngine(cfg=cfg, architecture=architecture), architecture


def run_full_episode(
    checkpoint: Path,
    sample: dict,
    args: argparse.Namespace,
    *,
    expect_value: bool,
    model_name: str,
) -> dict:
    engine, architecture = load_engine(checkpoint, args, expect_value)
    cuda_device = torch.device(args.device)
    if cuda_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(cuda_device)
    predicted_actions: list[np.ndarray] = []
    predicted_values: list[np.ndarray] = []
    predicted_video: list[Image.Image] = []
    video_timestamps: list[int] = []
    start_indices: list[int] = []
    last_target_state = 0
    last_video_timestamp = -1
    starts = sample["starts"]
    t0 = time.perf_counter()

    for window_index, start in enumerate(starts):
        conditions = {
            "first_frame_image": [sample["first_frames"][start]],
            "prompt": sample["prompt"],
            "proprio": sample["states"][start],
            "num_frames": WINDOW_ACTIONS + 1,
            "video_num_frames": VIDEO_FRAMES,
            "height": CANVAS_HEIGHT,
            "width": CANVAS_WIDTH,
            "denoise_steps": int(args.denoise_steps),
            "denoise_mode": "sync",
            "seed": int(args.seed) + window_index,
            "value_candidates": 1,
            "value_selection": args.value_selection,
            "value_num_frames": WINDOW_ACTIONS,
        }
        if args.explore:
            conditions["exploration"] = {
                "enabled": True,
                "explore_steps": args.explore_steps,
                "dynamic_groups": args.dynamic_groups,
                "value_groups": args.value_groups,
                "candidate_batch_size": args.candidate_batch_size,
                "sigma_decay": args.sigma_decay,
                "alpha_smooth": args.alpha_smooth,
                "value_elites": args.value_elites,
                "dynamic_elites": args.dynamic_elites,
                "min_std": args.min_std,
            }
        with torch.inference_mode():
            prediction = engine.generate(conditions)
        actions = np.asarray(prediction["actions"], dtype=np.float32)
        frames = prediction.get("video")
        if actions.shape != (WINDOW_ACTIONS, 20):
            raise ValueError(f"{model_name} window {window_index} action shape {actions.shape}")
        if frames is None or len(frames) != VIDEO_FRAMES:
            raise ValueError(f"{model_name} window {window_index} video frame count is not {VIDEO_FRAMES}")
        values = None
        if expect_value:
            values = np.asarray(prediction["values"], dtype=np.float32).reshape(-1)
            if values.shape != (WINDOW_ACTIONS,):
                raise ValueError(f"{model_name} window {window_index} value shape {values.shape}")
            if int(prediction["selected_candidate"]) != 0:
                raise ValueError("single-candidate WAV inference must select candidate 0")

        first_target_state = max(last_target_state + 1, start + 1)
        offset = first_target_state - (start + 1)
        start_indices.append(first_target_state - 1)
        predicted_actions.append(actions[offset:])
        if expect_value:
            predicted_values.append(values[offset:])
        last_target_state = start + WINDOW_ACTIONS

        for frame_index, frame in enumerate(frames):
            timestamp = start + VIDEO_STRIDE * frame_index
            if timestamp > last_video_timestamp and timestamp < sample["length"]:
                predicted_video.append(frame.convert("RGB") if isinstance(frame, Image.Image) else Image.fromarray(frame).convert("RGB"))
                video_timestamps.append(timestamp)
                last_video_timestamp = timestamp
        del prediction
        print(
            f"[{model_name}] window {window_index + 1}/{len(starts)} start={start} "
            f"covered_action_state={first_target_state}:{last_target_state}",
            flush=True,
        )

    actions_full = np.concatenate(predicted_actions, axis=0)
    values_full = np.concatenate(predicted_values, axis=0) if expect_value else None
    if last_target_state != sample["length"] - 1 or actions_full.shape != sample["ground_truth_actions"].shape:
        raise ValueError(f"episode action coverage incomplete: last={last_target_state}, shape={actions_full.shape}")
    if expect_value and values_full.shape != sample["ground_truth_values"].shape:
        raise ValueError(f"episode value coverage incomplete: shape={values_full.shape}")
    if video_timestamps[0] != 0 or video_timestamps[-1] != sample["length"] - 1:
        raise ValueError(f"episode video coverage incomplete: {video_timestamps[0]}..{video_timestamps[-1]}")

    elapsed = time.perf_counter() - t0
    if cuda_device.type == "cuda":
        torch.cuda.synchronize(cuda_device)
        peak_allocated_gib = torch.cuda.max_memory_allocated(cuda_device) / 2**30
        peak_reserved_gib = torch.cuda.max_memory_reserved(cuda_device) / 2**30
    else:
        peak_allocated_gib = peak_reserved_gib = None
    del engine, architecture
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return {
        "actions": actions_full,
        "values": values_full,
        "video_frames": predicted_video,
        "video_timestamps": video_timestamps,
        "start_indices": np.asarray(start_indices, dtype=np.int64),
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
    }


def save_gt_video(episode: Path, length: int, output: Path, fps: float) -> None:
    with h5py.File(episode, "r") as handle:
        with imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame_index in range(length):
                writer.append_data(np.asarray(decode_canvas(handle, frame_index)))
                if frame_index == 0 or (frame_index + 1) % 100 == 0 or frame_index + 1 == length:
                    print(f"[ground truth video] {frame_index + 1}/{length} frames", flush=True)


def save_pred_video(result: dict, length: int, output: Path, fps: float) -> None:
    timestamps = result["video_timestamps"]
    frames = result["video_frames"]
    cursor = 0
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for state_index in range(length):
            while cursor + 1 < len(timestamps) and timestamps[cursor + 1] <= state_index:
                cursor += 1
            # The model predicts one image every four source frames. Hold its
            # latest generated frame so this MP4 aligns with the 25-fps GT.
            writer.append_data(np.asarray(frames[cursor]))
    print(f"[predicted video] {length} frames, {len(frames)} distinct model frames", flush=True)


def save_plot(result: dict, sample: dict, args: argparse.Namespace, paths: dict, *, model_name: str, has_value: bool) -> None:
    plot_script = Path(__file__).with_name("robodojo_full_episode_plot.py")
    if not args.plot_python.is_file():
        raise FileNotFoundError(args.plot_python)
    action_path = paths["action"] / f"{sample['episode'].stem}_gt_vs_pred.jpg"
    with tempfile.TemporaryDirectory(prefix="openwam_robodojo_plot_") as temp_dir:
        data_path = Path(temp_dir) / "curves.npz"
        payload = {
            "ground_truth_actions": sample["ground_truth_actions"],
            "predicted_actions": result["actions"],
            "start_indices": result["start_indices"],
        }
        if has_value:
            payload["ground_truth_values"] = sample["ground_truth_values"]
            payload["predicted_values"] = result["values"]
        np.savez_compressed(data_path, **payload)
        command = [
            str(args.plot_python),
            str(plot_script),
            "--data",
            str(data_path),
            "--action-output",
            str(action_path),
            "--model-label",
            model_name,
        ]
        if has_value:
            command.extend(["--value-output", str(paths["value"] / f"{sample['episode'].stem}_gt_vs_pred.jpg")])
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def report_metrics(model_name: str, result: dict, sample: dict) -> dict:
    action_error = result["actions"].astype(np.float64) - sample["ground_truth_actions"].astype(np.float64)
    action_mae = float(np.mean(np.abs(action_error)))
    action_mse = float(np.mean(np.square(action_error)))
    action_rmse = float(np.sqrt(action_mse))
    video_squared_error = 0.0
    video_pixel_count = 0
    with h5py.File(sample["episode"], "r") as handle:
        for timestamp, frame in zip(result["video_timestamps"], result["video_frames"]):
            if timestamp == 0:  # the first image is observed, not predicted
                continue
            truth = np.asarray(decode_canvas(handle, timestamp), dtype=np.float32)
            predicted = np.asarray(frame, dtype=np.float32)
            difference = predicted - truth
            video_squared_error += float(np.square(difference, dtype=np.float64).sum())
            video_pixel_count += difference.size
    video_mse = video_squared_error / video_pixel_count
    video_psnr_db = float(10.0 * np.log10(255.0**2 / video_mse))
    message = (f"[{model_name}] full episode action MSE={action_mse:.6f} "
               f"MAE={action_mae:.6f} RMSE={action_rmse:.6f}; "
               f"video PSNR={video_psnr_db:.3f} dB")
    metrics = {"action_mse": action_mse, "action_mae": action_mae,
               "action_rmse": action_rmse, "video_psnr_db": video_psnr_db,
               "video_mse": video_mse, "video_frames_evaluated": len(result["video_timestamps"]) - 1}
    if result["values"] is not None:
        value_error = result["values"].astype(np.float64) - sample["ground_truth_values"].astype(np.float64)
        value_mae = float(np.mean(np.abs(value_error)))
        value_rmse = float(np.sqrt(np.mean(np.square(value_error))))
        message += f"; value MAE={value_mae:.6f} RMSE={value_rmse:.6f}"
        metrics.update(value_mae=value_mae, value_rmse=value_rmse)
    print(message, flush=True)
    return metrics


def main() -> None:
    wall_start = time.perf_counter()
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for {args.device}")
    if args.denoise_steps < 1 or args.fps <= 0:
        raise ValueError("denoise-steps and fps must be positive")
    if args.explore and args.model != "wav":
        raise ValueError("--explore requires --model wav")
    model_specs = [spec for index, spec in enumerate(MODEL_SPECS)
                   if args.model == "all" or (args.model == "wav" and index == 0)
                   or (args.model == "base" and index == 1)]
    if args.explore:
        model_specs = [(f"{name}_explore", checkpoint, has_value)
                       for name, checkpoint, has_value in model_specs]
    if args.run_name is not None:
        if len(model_specs) != 1 or Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}:
            raise ValueError("--run-name requires one model and a single safe directory name")
        model_specs = [(args.run_name, model_specs[0][1], model_specs[0][2])]
    checkpoints = [args.wav_checkpoint if args.model != "base" else args.base_checkpoint]
    if args.model == "all":
        checkpoints.append(args.base_checkpoint)
    for path in (args.episode, *checkpoints):
        if not path.is_file():
            raise FileNotFoundError(path)
    sample = load_episode(args)
    output_root = args.output_root.resolve()
    paths = output_directories(output_root, model_specs)
    print(
        f"Episode {sample['episode']} has {sample['length']} states, "
        f"{sample['length'] - 1} action/value targets, {len(sample['starts'])} windows; "
        f"exploration={args.explore}, value_candidates=1",
        flush=True,
    )

    video_filename_gt = f"{sample['episode'].stem}_gt.mp4"
    video_filename_pred = f"{sample['episode'].stem}_predict.mp4"
    baseline_gt = output_root / MODEL_SPECS[0][0] / "video" / video_filename_gt
    reuse_baseline_gt = (args.explore and sample["episode"] == DEFAULT_EPISODE.resolve()
                         and args.fps == 25.0 and baseline_gt.is_file())
    first_gt_video = None
    for model_name, _default_checkpoint, has_value in model_specs:
        gt_video = paths[model_name]["video"] / video_filename_gt
        source_video = first_gt_video or (baseline_gt if reuse_baseline_gt else None)
        if source_video is None:
            save_gt_video(sample["episode"], sample["length"], gt_video, args.fps)
        else:
            try:
                os.link(source_video, gt_video)
            except OSError:
                shutil.copyfile(source_video, gt_video)
        first_gt_video = gt_video

    for model_name, _default_checkpoint, has_value in model_specs:
        checkpoint = args.wav_checkpoint if has_value else args.base_checkpoint
        print(f"[{model_name}] loading {checkpoint}", flush=True)
        result = run_full_episode(
            checkpoint,
            sample,
            args,
            expect_value=has_value,
            model_name=model_name,
        )
        save_pred_video(result, sample["length"], paths[model_name]["video"] / video_filename_pred, args.fps)
        save_plot(result, sample, args, paths[model_name], model_name=model_name,
                  has_value=has_value and not args.explore)
        metrics = report_metrics(model_name, result, sample)
        print(f"[{model_name}] generation time {result['elapsed_seconds']:.1f}s", flush=True)
        print("METRICS_JSON " + json.dumps({
            "run_name": model_name, "episode": str(sample["episode"]),
            "episode_length": sample["length"], "windows": len(sample["starts"]),
            "generation_seconds": result["elapsed_seconds"],
            "wall_seconds": time.perf_counter() - wall_start,
            "peak_allocated_gib": result["peak_allocated_gib"],
            "peak_reserved_gib": result["peak_reserved_gib"],
            "exploration": {"steps": args.explore_steps, "dynamic_groups": args.dynamic_groups,
                            "value_groups": args.value_groups, "candidate_batch_size": args.candidate_batch_size},
            **metrics,
        }), flush=True)
        del result
        gc.collect()

    for model_name, _checkpoint, _has_value in model_specs:
        actual_files = [path for path in (output_root / model_name).rglob("*") if path.is_file()]
        if any(path.suffix.lower() not in {".jpg", ".mp4"} for path in actual_files):
            raise RuntimeError(f"unexpected file type under {output_root / model_name}")
        video_files = list(paths[model_name]["video"].glob("*.mp4"))
        if len(video_files) != 2:
            raise RuntimeError(f"expected exactly two MP4 files in {paths[model_name]['video']}")
        print(f"[{model_name}] files: {[str(path) for path in sorted(actual_files)]}", flush=True)


if __name__ == "__main__":
    main()

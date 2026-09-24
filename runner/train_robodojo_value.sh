#!/usr/bin/env bash
# Eight-GPU RoboDojo WAV value fine-tuning from the OpenWAM Alpha foundation checkpoint.
# LR, run length, and per-GPU batch match the released Sim-RoboDojo config;
# the new value expert inherits the same base LR.
# Additional Hydra overrides can be passed as arguments to this script.
# Use training.report_to=none to skip W&B, or override project.wandb.project.
set -euo pipefail

source /zhaohan/miniconda3/etc/profile.d/conda.sh
conda activate openwam

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 bash scripts/train.sh \
  dataloader=robodojo \
  dataloader.dataset_dir=/zhaohan/lirunze/RoboDojo/.cache/robodojo_data_modelscope_hdf5_repo/data/RoboDojo \
  dataloader.include_value_targets=true \
  dataloader.value_sidecar_root=/zhaohan/lirunze/RoboDojo/data/openwam_value_sidecars \
  model.architecture.value_backbone.enabled=true \
  training.lambda_value=1.0 \
  training.finetune_ckpt_path=/zhaohan/lirunze/OpenWAM_WAV/pretrained_model/OpenWAM-Alpha-Pretrain-Foundation-Model \
  training.num_epochs=null \
  training.max_steps=60000 \
  training.batch_size=4 \
  training.dataset_num_workers=8 \
  training.learning_rate=1e-4 \
  training.report_to=wandb \
  project.wandb.project=openwam-robodojo-value \
  training.output_path=outputs/robodojo_wav_value \
  "$@"

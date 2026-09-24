#!/usr/bin/env bash
# Eight-GPU reproduction of the released OpenWAM-Alpha-Sim-RoboDojo fine-tuning recipe.
# No ValueBackbone or value targets; warm-start from the Alpha foundation checkpoint.
# Additional Hydra overrides can be passed as arguments to this script.
set -euo pipefail

source /zhaohan/miniconda3/etc/profile.d/conda.sh
conda activate openwam

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# W&B otherwise prompts on rank 0 when no key is configured, leaving the
# other ranks waiting in a collective until NCCL times out. Preserve online
# logging when authenticated and keep an uploadable local run otherwise.
if [[ -z "${WANDB_MODE:-}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]] || python -c '
import netrc
try:
    authenticated = netrc.netrc().authenticators("api.wandb.ai") is not None
except (FileNotFoundError, netrc.NetrcParseError):
    authenticated = False
raise SystemExit(0 if authenticated else 1)
'; then
    export WANDB_MODE=online
  else
    export WANDB_MODE=offline
  fi
fi
printf '[RoboDojo Base] W&B mode: %s\n' "$WANDB_MODE"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 bash scripts/train.sh \
  dataloader=robodojo \
  dataloader.dataset_dir=/zhaohan/lirunze/RoboDojo/.cache/robodojo_data_modelscope_hdf5_repo/data/RoboDojo \
  dataloader.include_value_targets=false \
  model.architecture.value_backbone.enabled=false \
  training.lambda_value=0.0 \
  training.finetune_ckpt_path=/zhaohan/lirunze/OpenWAM_WAV/pretrained_model/OpenWAM-Alpha-Pretrain-Foundation-Model \
  training.num_epochs=null \
  training.max_steps=60000 \
  training.batch_size=4 \
  training.gradient_accumulation_steps=1 \
  training.learning_rate=1e-4 \
  training.mixed_precision=bf16 \
  training.zero_stage=2 \
  training.use_gradient_checkpointing=false \
  training.dataset_num_workers=8 \
  training.save_steps=10000 \
  training.save_full_states_for_resume=true \
  training.keep_last_k_ckpts=3 \
  training.report_to=wandb \
  project.wandb.project=robodojo_openwam_base \
  training.output_path=outputs/robodojo_openwam_base \
  "$@"

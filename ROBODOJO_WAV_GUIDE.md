# RoboDojo OpenWAM / WAV 训练与测试

本文整理本仓库的 RoboDojo 基础策略和 WAV（价值引导探索）操作命令。除外部环境自身的位置外，以下仓库内文件、模型和输出路径均以仓库根目录为相对位置。

## 目录与前置条件

以下命令均假设当前工作目录是本仓库根目录。

约定 RoboDojo 仓库与本仓库并列，训练数据位于 RoboDojo 下载缓存，Alpha foundation checkpoint 已放在 `pretrained_model/`：

```bash
conda activate openwam
DATASET_DIR="../RoboDojo/.cache/robodojo_data_modelscope_hdf5_repo/data/RoboDojo"
VALUE_SIDECAR_ROOT="../RoboDojo/data/openwam_value_sidecars"
FOUNDATION_CKPT="pretrained_model/OpenWAM-Alpha-Pretrain-Foundation-Model"
```

确认数据集包含 `<task>/arx_x5/data/episode_*.hdf5`，基础模型目录包含模型权重和配置。训练环境需已安装本项目依赖（见 [`README.md`](README.md)），并能使用 8 张 GPU。以下训练命令假设 `openwam` Conda 环境已经激活；W&B 可登录，也可用 `training.report_to=none` 关闭。

RoboDojo 数据配置为仿真 `arx_x5`、EEF 动作、80 维统一动作空间、33 帧窗口和 4 帧视频步长，详见 [`configs/dataloader/robodojo.yaml`](configs/dataloader/robodojo.yaml)。

## 1. 生成 WAV 价值目标

RoboDojo HDF5 没有原生 reward/success 字段。下面的命令从每个 episode 生成稀疏终点演示回报 sidecar 和全局归一化统计，不会修改源 HDF5。`--fast-fingerprint` 使用文件大小、修改时间和 episode 长度校验；省略该参数会计算完整 SHA-256。

```bash
python scripts/prepare_robodojo_value_targets.py \
  --dataset-dir "$DATASET_DIR" \
  --sidecar-root "$VALUE_SIDECAR_ROOT" \
  --variant sim \
  --embodiment arx_x5 \
  --fast-fingerprint
```

Sidecar 目录结构会镜像数据集任务目录，并在根目录生成 `robodojo_value_stats.json`。训练时使用的 sidecar 根目录和 `value_gamma`（默认 `0.99`）必须与生成时一致。

## 2. 训练基础策略（不含价值分支）

此训练作为 WAV 的对照策略：从 Alpha foundation checkpoint 微调 60,000 步，8 卡、每卡 batch size 4。输出在 `outputs/robodojo_openwam_base/` 下的时间戳目录。

```bash
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train.sh \
  dataloader=robodojo \
  dataloader.dataset_dir="$DATASET_DIR" \
  dataloader.include_value_targets=false \
  model=dual_system \
  model.architecture.variant=joint_self_attn \
  model.architecture.attention_mask_mode=mutual \
  model.architecture.value_backbone.enabled=false \
  training.lambda_value=0.0 \
  training.finetune_ckpt_path="$FOUNDATION_CKPT" \
  training.num_epochs=null \
  training.max_steps=60000 \
  training.batch_size=4 \
  training.gradient_accumulation_steps=1 \
  training.learning_rate=1e-4 \
  training.mixed_precision=bf16 \
  training.zero_stage=2 \
  training.dataset_num_workers=8 \
  training.save_steps=10000 \
  training.save_full_states_for_resume=true \
  training.keep_last_k_ckpts=3 \
  training.report_to=wandb \
  project.wandb.project=robodojo_openwam_base \
  training.output_path=outputs/robodojo_openwam_base
```

## 3. 训练 WAV 价值分支

先完成第 1 节的 sidecar 生成，再运行 WAV 微调。WAV 使用与基础策略相同的 foundation checkpoint、数据、卡数和主要优化参数，另外开启 ValueBackbone 与 value loss。输出位于 `outputs/robodojo_wav_value/` 下的时间戳目录。

```bash
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train.sh \
  dataloader=robodojo \
  dataloader.dataset_dir="$DATASET_DIR" \
  dataloader.include_value_targets=true \
  dataloader.value_sidecar_root="$VALUE_SIDECAR_ROOT" \
  dataloader.value_gamma=0.99 \
  model=dual_system \
  model.architecture.variant=joint_self_attn \
  model.architecture.attention_mask_mode=mutual \
  model.architecture.value_backbone.enabled=true \
  training.lambda_value=1.0 \
  training.finetune_ckpt_path="$FOUNDATION_CKPT" \
  training.num_epochs=null \
  training.max_steps=60000 \
  training.batch_size=4 \
  training.gradient_accumulation_steps=1 \
  training.learning_rate=1e-4 \
  training.mixed_precision=bf16 \
  training.zero_stage=2 \
  training.dataset_num_workers=8 \
  training.report_to=wandb \
  project.wandb.project=openwam-robodojo-value \
  training.output_path=outputs/robodojo_wav_value
```

`training.output_path` 是输出根目录；每次新训练会在其下创建时间戳运行目录。权重文件名为 `checkpoint_step_<step>.safetensors`。例如，训练完成后的路径形式为：

```text
outputs/robodojo_wav_value/<run_timestamp>/checkpoint_step_60000.safetensors
outputs/robodojo_openwam_base/<run_timestamp>/checkpoint_step_60000.safetensors
```

将实际时间戳记下来，供后续测试命令使用。若要从已有训练状态续训，需额外设置 `training.resume_ckpt_path` 指向原运行目录；它与 `training.finetune_ckpt_path` 不能同时设置。完整状态保存由 `training.save_full_states_for_resume=true` 控制。

仓库也提供封装脚本 [`runner/train_robodojo_openwam_base.sh`](runner/train_robodojo_openwam_base.sh) 和 [`runner/train_robodojo_value.sh`](runner/train_robodojo_value.sh)。它们针对另一运行环境写有 Conda、数据和 checkpoint 绝对路径；在本机直接使用前需检查并替换这些机器路径。上面的 `scripts/train.sh` 命令不依赖这些封装脚本。

## 4. 整集开环测试（动作/视频/价值预测）

[`scripts/inference_test/robodojo_openloop_eval.py`](scripts/inference_test/robodojo_openloop_eval.py) 会在指定 episode 上按 32 个动作的窗口逐段推理，并输出动作、视频及 WAV value 对照图和误差指标。下面用环境变量指向两个已训练 checkpoint；替换 `<wav_run_timestamp>` 和 `<base_run_timestamp>`：

```bash
WAV_CKPT="outputs/robodojo_wav_value/<wav_run_timestamp>/checkpoint_step_60000.safetensors"
BASE_CKPT="outputs/robodojo_openwam_base/<base_run_timestamp>/checkpoint_step_60000.safetensors"
EPISODE="$DATASET_DIR/sweep_blocks/arx_x5/data/episode_0000000.hdf5"

python scripts/inference_test/robodojo_openloop_eval.py \
  --episode "$EPISODE" \
  --dataset-root "$DATASET_DIR" \
  --sidecar-root "$VALUE_SIDECAR_ROOT" \
  --wav-checkpoint "$WAV_CKPT" \
  --base-checkpoint "$BASE_CKPT" \
  --output-root outputs/robodojo_openloop_eval \
  --plot-python "$(command -v python)" \
  --model all
```

默认会同时测 WAV 和基础策略。结果写入 `outputs/robodojo_openloop_eval/`，包括 `.mp4` 视频、`.jpg` 曲线图和终端指标（action MSE/MAE/RMSE、video PSNR、WAV value MAE/RMSE）。输出根目录下对应模型目录已有文件时，请换一个新的 `--output-root`。

只对 WAV 开启迭代式 value-guided exploration：

```bash
python scripts/inference_test/robodojo_openloop_eval.py \
  --episode "$EPISODE" \
  --dataset-root "$DATASET_DIR" \
  --sidecar-root "$VALUE_SIDECAR_ROOT" \
  --wav-checkpoint "$WAV_CKPT" \
  --output-root outputs/robodojo_openloop_exploration \
  --plot-python "$(command -v python)" \
  --model wav \
  --explore
```

该脚本的默认数据、checkpoint 和绘图 Python 路径针对特定机器，因此命令中显式覆盖了这些参数。探索模式只预测 WAV 的动作/视频；此模式下不绘制预测 value 曲线。

## 5. RoboDojo 仿真对比（6 GPU）

[`runner/eval_robodojo_wav_6gpu.sh`](runner/eval_robodojo_wav_6gpu.sh) 会在五个任务上先运行 `exploration=false`，再运行 `exploration=true`。每轮用 3 张 GPU 跑策略服务器，另 3 张 GPU 跑仿真。默认任务是 `play_tic_tac_toe`、`fold_clothes`、`stack_bowls`、`build_tower` 和 `put_bottles_into_dustbin`。

```bash
ROBODOJO_ROOT=../RoboDojo \
OPENWAM_CKPT_DIR="outputs/robodojo_wav_value/<wav_run_timestamp>" \
bash runner/eval_robodojo_wav_6gpu.sh \
  --gpus 0,1,2,3,4,5 \
  --seed 0 \
  --eval-num native \
  --num-envs 25 \
  --candidate-batch-size 8 \
  --run-id wav_compare_001
```

checkpoint 目录须含 `config.yaml` 和 `checkpoint_step_60000.safetensors`。`native` 使用 RoboDojo 每个任务的默认 episode 数；可改为 `--eval-num 2` 等小数值做短测。脚本将日志和汇总写入 `outputs/robodojo_wav_compare/wav_compare_001/`，主要文件为 `explore_false.json`、`explore_true.json` 及对应 `.md`、`.log`。

该仿真评测还要求相邻的 `../RoboDojo` 仓库已安装 XPolicyLab、RoboDojo 环境和评测资源，6 张 GPU 可用，端口 `18848` 至 `18850` 空闲。评测脚本中的 Conda 环境位置针对当前机器；若安装位置不同，需先修改 [`runner/eval_robodojo_wav_6gpu.sh`](runner/eval_robodojo_wav_6gpu.sh) 中对应的环境路径。可追加 `--dry-run` 查看资源计划；它仍会先检查 checkpoint、GPU 和 RoboDojo 环境。

## 配置和脚本索引

- [`configs/dataloader/robodojo.yaml`](configs/dataloader/robodojo.yaml)：数据集路径、采样、动作映射、sidecar 开关。
- [`configs/model/dual_system.yaml`](configs/model/dual_system.yaml)：`joint_self_attn` 与 WAV ValueBackbone 配置。
- [`configs/train.yaml`](configs/train.yaml)：训练步数、batch、学习率、精度、checkpoint 和日志默认值。
- [`configs/deploy.yaml`](configs/deploy.yaml)：推理与探索采样参数。
- [`scripts/prepare_robodojo_value_targets.py`](scripts/prepare_robodojo_value_targets.py)：生成 WAV value sidecar。
- [`scripts/inference_test/robodojo_openloop_eval.py`](scripts/inference_test/robodojo_openloop_eval.py)：单 episode 整集开环指标和可视化。
- [`runner/eval_robodojo_wav_6gpu.sh`](runner/eval_robodojo_wav_6gpu.sh)：多任务仿真探索开关对照。

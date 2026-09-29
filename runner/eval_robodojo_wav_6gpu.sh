#!/usr/bin/env bash
# Compare one WAV checkpoint with and without value-guided exploration on five RoboDojo tasks.
# Six GPUs: three policy/simulator pairs, baseline first, then exploration.
set -Eeuo pipefail

OPENWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-/lirunze/RoboDojo}"
POLICY_DIR="${ROBODOJO_ROOT}/XPolicyLab/policy/OpenWAM"
CHECKPOINT="${OPENWAM_CKPT_DIR:-${OPENWAM_ROOT}/pretrained_model/2026-09-22_23-15-46}"
GPU_CSV="0,1,2,3,4,5"
PORT_BASE=18848
SEED=0
EVAL_NUM=native
NUM_ENVS=25
CANDIDATE_BATCH_SIZE=8
RUN_ID="wav_robodojo_$(date -u +%Y%m%d_%H%M%S)"
DRY_RUN=0
TASKS=play_tic_tac_toe,fold_clothes,stack_bowls,build_tower,put_bottles_into_dustbin

usage() {
  cat <<'EOF'
Usage: bash runner/eval_robodojo_wav_6gpu.sh [options]

Runs exploration=true and exploration=false on the same five RoboDojo tasks.
Defaults to seed 0 and native episode counts (50, or 25 for generalization tasks).

  --checkpoint DIR           WAV checkpoint directory
  --gpus A,B,C,D,E,F        GPU ids: policy A,B,C; matching simulator D,E,F
  --seed N                   RoboDojo seed (default: 0)
  --eval-num native|N        Episodes per task (default: native)
  --num-envs N              Parallel Isaac scenes (default: 25)
  --candidate-batch-size N  Exploration candidate microbatch (default: 8)
  --port-base N              Three consecutive local ports (default: 18848)
  --run-id NAME              Output directory name
  --dry-run                  Check inputs and print the GPU/task plan
EOF
}

while (($#)); do
  case "$1" in
    --checkpoint|--gpus|--seed|--eval-num|--num-envs|--candidate-batch-size|--port-base|--run-id)
      if (($# < 2)); then echo "missing value for $1" >&2; exit 2; fi
      case "$1" in
        --checkpoint) CHECKPOINT="$2" ;;
        --gpus) GPU_CSV="$2" ;;
        --seed) SEED="$2" ;;
        --eval-num) EVAL_NUM="$2" ;;
        --num-envs) NUM_ENVS="$2" ;;
        --candidate-batch-size) CANDIDATE_BATCH_SIZE="$2" ;;
        --port-base) PORT_BASE="$2" ;;
        --run-id) RUN_ID="$2" ;;
      esac
      shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "$SEED" "$NUM_ENVS" "$CANDIDATE_BATCH_SIZE" "$PORT_BASE"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "expected a nonnegative integer: $value" >&2; exit 2; }
done
((NUM_ENVS > 0 && CANDIDATE_BATCH_SIZE > 0 && PORT_BASE >= 1024 && PORT_BASE <= 65533)) || {
  echo "invalid num-envs, candidate-batch-size, or port-base" >&2; exit 2;
}
[[ "$EVAL_NUM" == native || "$EVAL_NUM" =~ ^[1-9][0-9]*$ ]] || {
  echo "--eval-num must be native or a positive integer" >&2; exit 2;
}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid --run-id" >&2; exit 2; }
IFS=, read -r -a GPUS <<< "$GPU_CSV"
((${#GPUS[@]} == 6)) || { echo "--gpus requires six ids" >&2; exit 2; }
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU id: $gpu" >&2; exit 2; }
  [[ ! -v SEEN_GPUS[$gpu] ]] || { echo "duplicate GPU id: $gpu" >&2; exit 2; }
  SEEN_GPUS[$gpu]=1
  nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader >/dev/null || exit 2
done

[[ -f "$CHECKPOINT/config.yaml" && -f "$CHECKPOINT/checkpoint_step_60000.safetensors" ]] || {
  echo "WAV checkpoint is missing config.yaml or checkpoint_step_60000.safetensors: $CHECKPOINT" >&2
  exit 1
}
CHECKPOINT="$(realpath -e -- "$CHECKPOINT")"
[[ -f "$POLICY_DIR/setup_eval_policy_server.sh" && -f "$ROBODOJO_ROOT/scripts/internal/smoke_all_tasks.sh" ]] || {
  echo "RoboDojo/OpenWAM evaluation scripts are missing under $ROBODOJO_ROOT" >&2; exit 1
}
[[ -x /lirunze/miniconda3/envs/RoboDojo/bin/python ]] || {
  echo "RoboDojo conda environment is missing" >&2; exit 1
}
source /lirunze/miniconda3/etc/profile.d/conda.sh
conda activate RoboDojo
command -v python3 >/dev/null || { echo "python3 is unavailable after activating RoboDojo" >&2; exit 1; }
python3 -c 'import yaml' || { echo "RoboDojo Python cannot import PyYAML" >&2; exit 1; }

RUN_DIR="${OPENWAM_ROOT}/outputs/robodojo_wav_compare/${RUN_ID}"
printf 'both phases: model GPUs %s,%s,%s; simulator GPUs %s,%s,%s; ports %s,%s,%s\n' \
  "${GPUS[0]}" "${GPUS[1]}" "${GPUS[2]}" \
  "${GPUS[3]}" "${GPUS[4]}" "${GPUS[5]}" \
  "$PORT_BASE" "$((PORT_BASE+1))" "$((PORT_BASE+2))"
echo 'phase order: exploration=false, then exploration=true'
printf 'checkpoint=%s seed=%s eval_num=%s num_envs=%s tasks=%s\n' \
  "$CHECKPOINT" "$SEED" "$EVAL_NUM" "$NUM_ENVS" "$TASKS"
printf 'results=%s\n' "$RUN_DIR"
if ((DRY_RUN)); then exit 0; fi
[[ ! -e "$RUN_DIR" ]] || { echo "run directory already exists: $RUN_DIR" >&2; exit 1; }

for port in "$PORT_BASE" "$((PORT_BASE+1))" "$((PORT_BASE+2))"; do
  if timeout 1 bash -c '>/dev/tcp/127.0.0.1/$1' _ "$port" 2>/dev/null; then
    echo "port already in use: $port" >&2; exit 1
  fi
done
mkdir -p "$RUN_DIR"

# Separate process groups let Ctrl-C stop each simulator and model server cleanly.
declare -a ACTIVE_PIDS=()
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  for pid in "${ACTIVE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill -- "-$pid" 2>/dev/null || true; fi
  done
  for pid in "${ACTIVE_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit "$rc"
}
trap cleanup EXIT INT TERM

stop_phase() {
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill -- "-$pid" 2>/dev/null || true; fi
  done
  for pid in "${ACTIVE_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  ACTIVE_PIDS=()
}

start_server() {
  local name="$1" gpu="$2" port="$3" explore="$4"
  setsid env \
    OPENWAM_ROOT="$OPENWAM_ROOT" \
    OPENWAM_CKPT_DIR="$CHECKPOINT" \
    OPENWAM_EXPLORATION_ENABLED="$explore" \
    OPENWAM_CANDIDATE_BATCH_SIZE="$CANDIDATE_BATCH_SIZE" \
    bash "$POLICY_DIR/setup_eval_policy_server.sh" \
      RoboDojo play_tic_tac_toe "$name" arx_x5 ee "$SEED" "$gpu" RoboDojo "$port" 127.0.0.1 \
      > "$RUN_DIR/server_${name}_${gpu}.log" 2>&1 &
  local pid=$!
  ACTIVE_PIDS+=("$pid")
  SERVER_PID="$pid"
}

export ROBODOJO_PYTHON=/lirunze/miniconda3/envs/RoboDojo/bin/python
export ROBODOJO_NUM_ENVS="$NUM_ENVS" ROBODOJO_CPU_PHYSX=1 EVAL_ENV_TYPE=sim
# Keep the script's --eval-num value while preventing a parent EVAL_NUM from
# overriding RoboDojo's native per-task episode counts in child processes.
export -n EVAL_NUM

run_phase() {
  local phase="$1" enabled="$2" port pid client_pid rc=0
  local -a servers=()
  echo "starting exploration=$enabled ($phase)"
  for ((i=0; i<3; i++)); do
    port=$((PORT_BASE+i))
    start_server "wav_explore_${phase}" "${GPUS[i]}" "$port" "$enabled"
    servers+=("$SERVER_PID")
  done
  for ((i=0; i<3; i++)); do
    port=$((PORT_BASE+i))
    if ! bash "$ROBODOJO_ROOT/XPolicyLab/utils/wait_for_policy_server.sh" \
      127.0.0.1 "$port" "${servers[i]}" OpenWAM 1800; then
      echo "policy server failed on GPU ${GPUS[i]}; see $RUN_DIR/server_wav_explore_${phase}_${GPUS[i]}.log" >&2
      stop_phase
      return 1
    fi
  done

  setsid bash "$ROBODOJO_ROOT/scripts/internal/smoke_all_tasks.sh" \
    --mode client --only "$TASKS" --policy-name OpenWAM \
    --policy-host 127.0.0.1 --policy-port "$PORT_BASE,$((PORT_BASE+1)),$((PORT_BASE+2))" \
    --policy-gpu-ids "${GPUS[0]},${GPUS[1]},${GPUS[2]}" \
    --env-gpu-ids "${GPUS[3]},${GPUS[4]},${GPUS[5]}" \
    --ckpt "wav_explore_${phase}" --action-type ee --seed "$SEED" --eval-num "$EVAL_NUM" \
    --run-id "${RUN_ID}_${phase}" --summary "$RUN_DIR/explore_${phase}.json" \
    --markdown "$RUN_DIR/explore_${phase}.md" \
    > "$RUN_DIR/explore_${phase}.log" 2>&1 &
  client_pid=$!
  ACTIVE_PIDS+=("$client_pid")
  echo "running exploration=$enabled; progress: $RUN_DIR/explore_${phase}.log"
  wait "$client_pid" || rc=1
  if kill -0 -- "-$client_pid" 2>/dev/null; then
    kill -- "-$client_pid" 2>/dev/null || true
  fi
  ACTIVE_PIDS=("${servers[@]}")
  stop_phase
  return "$rc"
}

if ! run_phase false 0; then
  echo "exploration=false failed; see $RUN_DIR/explore_false.json and task logs" >&2
  exit 1
fi
if ! run_phase true 1; then
  echo "exploration=true failed; see $RUN_DIR/explore_true.json and task logs" >&2
  exit 1
fi
echo "finished successfully; summaries: $RUN_DIR/explore_{true,false}.json"

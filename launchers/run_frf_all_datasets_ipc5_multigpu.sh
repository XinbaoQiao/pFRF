#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${PROJECT_ROOT}/datasets"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${PROJECT_ROOT}/output}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_FIRST_TASK_ONLY="${RUN_FIRST_TASK_ONLY:-0}"
GPU_LAUNCH_STAGGER_SECONDS="${GPU_LAUNCH_STAGGER_SECONDS:-20}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${BASE_OUTPUT_ROOT}/frf_${RUN_TS}"
LOG_ROOT="${RUN_ROOT}/_runlogs"
mkdir -p "${LOG_ROOT}"
SCRIPT_LOG="${LOG_ROOT}/driver.log"
exec > >(tee -a "${SCRIPT_LOG}") 2>&1
echo "[$(date '+%F %T')] RUN_ROOT=${RUN_ROOT}"
echo "[$(date '+%F %T')] SCRIPT_LOG=${SCRIPT_LOG}"

DATASETS=(
  imagenet-1k
  imagenet-100
  cifar100
  cifar10
  spawrious
  waterbirds
  cub2011
)

MODEL_GPU_QUEUES=(
  # Format: "gpu_id|model1,model2,..."
  "5|mocov3_resnet50,eva02_vitb"
  "6|clip_vitb"
  "7|dinov2_vitb"
)

if (( RUN_FIRST_TASK_ONLY )); then
  DATASETS=("${DATASETS[0]}")
fi

run_model_queue() {
  local model="$1"
  local gpu_id="$2"
  local run_log="${LOG_ROOT}/${model}_gpu${gpu_id}.log"
  : > "${run_log}"

  local -a failed_tasks=()
  for dataset in "${DATASETS[@]}"; do
    local ds_root="${RUN_ROOT}/${dataset}"
    local exp_name="${model}"
    mkdir -p "${ds_root}"
    echo "[$(date '+%F %T')] START gpu=${gpu_id} dataset=${dataset} model=${model} ipc=5" | tee -a "${run_log}"
    if CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${PROJECT_ROOT}/main_fed.py" \
      --experiment_name "${exp_name}" \
      --dataset "${dataset}" \
      --model "${model}" \
      --data_root "${DATA_ROOT}" \
      --output_root "${RUN_ROOT}" \
      --ipc "5" \
      >> "${ds_root}/${exp_name}.log" 2>&1; then
      echo "[$(date '+%F %T')] END   gpu=${gpu_id} dataset=${dataset} model=${model}" | tee -a "${run_log}"
    else
      echo "[$(date '+%F %T')] FAIL  gpu=${gpu_id} dataset=${dataset} model=${model}" | tee -a "${run_log}"
      failed_tasks+=("${dataset}|${model}|gpu${gpu_id}")
    fi
  done

  if (( ${#failed_tasks[@]} > 0 )); then
    printf '%s\n' "${failed_tasks[@]}" > "${LOG_ROOT}/failed_${model}_gpu${gpu_id}.txt"
    return 1
  fi
  return 0
}

cd "${PROJECT_ROOT}"
echo "${RUN_ROOT}" > "${PROJECT_ROOT}/latest_frf_batch_root.txt"

declare -a pids=()
queue_idx=0
queue_total=${#MODEL_GPU_QUEUES[@]}
for queue in "${MODEL_GPU_QUEUES[@]}"; do
  IFS='|' read -r gpu_id models_csv <<< "${queue}"
  (
    q_rc=0
    IFS=',' read -r -a models <<< "${models_csv}"
    for model in "${models[@]}"; do
      if ! run_model_queue "${model}" "${gpu_id}"; then
        q_rc=1
      fi
    done
    exit "${q_rc}"
  ) &
  pids+=("$!")
  queue_idx=$((queue_idx + 1))
  if (( queue_idx < queue_total )); then
    echo "[$(date '+%F %T')] launch_stagger sleep ${GPU_LAUNCH_STAGGER_SECONDS}s before next GPU queue"
    sleep "${GPU_LAUNCH_STAGGER_SECONDS}"
  fi
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done

exit "${rc}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${PROJECT_ROOT}/datasets"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${PROJECT_ROOT}/output}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_FIRST_TASK_ONLY="${RUN_FIRST_TASK_ONLY:-0}"
GPU_ID="${GPU_ID:-7}"

IPC="${IPC:-1}"
AUGS_PER_BATCH="${AUGS_PER_BATCH:-1}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${BASE_OUTPUT_ROOT}/frf_aug1_${RUN_TS}"
LOG_ROOT="${RUN_ROOT}/_runlogs"
mkdir -p "${LOG_ROOT}"
SCRIPT_LOG="${LOG_ROOT}/driver.log"
exec > >(tee -a "${SCRIPT_LOG}") 2>&1

echo "[$(date '+%F %T')] RUN_ROOT=${RUN_ROOT}"
echo "[$(date '+%F %T')] SCRIPT_LOG=${SCRIPT_LOG}"
echo "[$(date '+%F %T')] AUGS_PER_BATCH=${AUGS_PER_BATCH} (forced aug=1 profile)"

DATASETS=(
  flowers102
  food101
  imagenette
  spawrious
  stanforddogs
  waterbirds
  artbench
  cifar10
  cifar100
  cub2011
  imagenet-1k
  imagenet-100
)

MODEL_ORDER=(
  "mocov3_resnet50"
  "clip_vitb"
  "dinov2_vitb"
  "eva02_vitb"
)

if (( RUN_FIRST_TASK_ONLY )); then
  DATASETS=("${DATASETS[0]}")
fi

run_model_queue() {
  local model="$1"
  local run_log="${LOG_ROOT}/${model}_gpu${GPU_ID}.log"
  : > "${run_log}"

  local -a failed_tasks=()
  for dataset in "${DATASETS[@]}"; do
    local ds_root="${RUN_ROOT}/${dataset}"
    local exp_name="${model}"
    mkdir -p "${ds_root}"

    echo "[$(date '+%F %T')] START gpu=${GPU_ID} dataset=${dataset} model=${model} ipc=${IPC} augs_per_batch=${AUGS_PER_BATCH}" | tee -a "${run_log}"

    if CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${PROJECT_ROOT}/main_fed.py" \
      --experiment_name "${exp_name}" \
      --dataset "${dataset}" \
      --model "${model}" \
      --data_root "${DATA_ROOT}" \
      --output_root "${RUN_ROOT}" \
      --ipc "${IPC}" \
      --augs_per_batch "${AUGS_PER_BATCH}" \
      >> "${ds_root}/${exp_name}.log" 2>&1; then
      echo "[$(date '+%F %T')] END   gpu=${GPU_ID} dataset=${dataset} model=${model}" | tee -a "${run_log}"
    else
      echo "[$(date '+%F %T')] FAIL  gpu=${GPU_ID} dataset=${dataset} model=${model}" | tee -a "${run_log}"
      failed_tasks+=("${dataset}|${model}|gpu${GPU_ID}")
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

rc=0
for model in "${MODEL_ORDER[@]}"; do
  if ! run_model_queue "${model}"; then
    rc=1
  fi
done

exit "${rc}"

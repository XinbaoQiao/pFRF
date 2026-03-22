#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${PROJECT_ROOT}/datasets"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${PROJECT_ROOT}/output}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT_OVERRIDE="${RUN_ROOT_OVERRIDE:-}"

RUN_FIRST_TASK_ONLY="${RUN_FIRST_TASK_ONLY:-0}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${RUN_ROOT_OVERRIDE}" ]]; then
  RUN_ROOT="${RUN_ROOT_OVERRIDE}"
else
  RUN_ROOT="${BASE_OUTPUT_ROOT}/frf_${RUN_TS}"
fi
LOG_ROOT="${RUN_ROOT}/_runlogs"
mkdir -p "${LOG_ROOT}"

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
  imagenet-100
  imagenet-1k
)

MODEL_GPU_PAIRS=(
  "mocov3_resnet50|5"
  "clip_vitb|5"
  "dinov2_vitb|6"
  "eva02_vitb|6"
)

if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
fi
if [[ -n "${MODEL_GPU_PAIRS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a MODEL_GPU_PAIRS <<< "${MODEL_GPU_PAIRS_OVERRIDE}"
fi
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
    echo "[$(date '+%F %T')] START gpu=${gpu_id} dataset=${dataset} model=${model} ipc=1" | tee -a "${run_log}"
    if CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${PROJECT_ROOT}/main_fed.py" \
      --experiment_name "${exp_name}" \
      --dataset "${dataset}" \
      --model "${model}" \
      --data_root "${DATA_ROOT}" \
      --output_root "${RUN_ROOT}" \
      --ipc "1" \
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

run_gpu_model_queue() {
  local gpu_id="$1"
  shift
  local -a models=("$@")
  local rc=0
  local model
  for model in "${models[@]}"; do
    if ! run_model_queue "${model}" "${gpu_id}"; then
      rc=1
    fi
  done
  return "${rc}"
}

declare -A GPU_MODEL_MAP=()
declare -a GPU_ORDER=()
for pair in "${MODEL_GPU_PAIRS[@]}"; do
  IFS='|' read -r model gpu_id <<< "${pair}"
  if [[ -z "${GPU_MODEL_MAP[${gpu_id}]+x}" ]]; then
    GPU_ORDER+=("${gpu_id}")
    GPU_MODEL_MAP["${gpu_id}"]="${model}"
  else
    GPU_MODEL_MAP["${gpu_id}"]+=" ${model}"
  fi
done

declare -a pids=()
for gpu_id in "${GPU_ORDER[@]}"; do
  IFS=' ' read -r -a gpu_models <<< "${GPU_MODEL_MAP[${gpu_id}]}"
  run_gpu_model_queue "${gpu_id}" "${gpu_models[@]}" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done

exit "${rc}"

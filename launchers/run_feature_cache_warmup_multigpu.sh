#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-fdd}"
DATA_ROOT="${PROJECT_ROOT}/datasets"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${PROJECT_ROOT}/output}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_FIRST_TASK_ONLY="${RUN_FIRST_TASK_ONLY:-0}"
SEED="${SEED:-3407}"
STATS_NUM_WORKERS="${STATS_NUM_WORKERS:-8}"
MODEL_START_STAGGER_SEC="${MODEL_START_STAGGER_SEC:-20}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}" || {
      echo "Failed to activate conda env: ${CONDA_ENV_NAME}" >&2
      exit 1
    }
  fi
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${BASE_OUTPUT_ROOT}/feature_cache_warmup_${RUN_TS}"
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
  "dinov2_vitb|1"
  "clip_vitb|2"
  "eva02_vitb|3"
  "mocov3_resnet50|4"
)

if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
fi
if [[ -n "${MODEL_GPU_PAIRS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a MODEL_GPU_PAIRS <<< "${MODEL_GPU_PAIRS_OVERRIDE}"
fi
if (( RUN_FIRST_TASK_ONLY )); then
  DATASETS=("${DATASETS[0]}")
  MODEL_GPU_PAIRS=("${MODEL_GPU_PAIRS[0]}")
fi

cd "${PROJECT_ROOT}"
echo "${RUN_ROOT}" > "${PROJECT_ROOT}/latest_feature_cache_warmup_root.txt"

run_model_queue() {
  local model="$1"
  local gpu_id="$2"
  local start_delay="$3"
  local run_log="${LOG_ROOT}/${model}_gpu${gpu_id}_warmup.log"
  : > "${run_log}"

  if (( start_delay > 0 )); then
    echo "[$(date '+%F %T')] DELAY gpu=${gpu_id} model=${model} seconds=${start_delay}" | tee -a "${run_log}"
    sleep "${start_delay}"
  fi

  local -a failed_tasks=()
  for dataset in "${DATASETS[@]}"; do
    local task_log="${RUN_ROOT}/${dataset}_${model}.log"
    echo "[$(date '+%F %T')] START gpu=${gpu_id} dataset=${dataset} model=${model}" | tee -a "${run_log}"
    if CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${PROJECT_ROOT}/main_fed.py" \
      --experiment_name "feature_cache_warmup" \
      --dataset "${dataset}" \
      --model "${model}" \
      --data_root "${DATA_ROOT}" \
      --output_root "${RUN_ROOT}" \
      --seed "${SEED}" \
      --stats_num_workers "${STATS_NUM_WORKERS}" \
      --only_build_feature_cache True \
      >> "${task_log}" 2>&1; then
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

declare -a pids=()
idx=0
for pair in "${MODEL_GPU_PAIRS[@]}"; do
  IFS='|' read -r model gpu_id <<< "${pair}"
  start_delay=$(( idx * MODEL_START_STAGGER_SEC ))
  run_model_queue "${model}" "${gpu_id}" "${start_delay}" &
  pids+=("$!")
  idx=$(( idx + 1 ))
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done

exit "${rc}"

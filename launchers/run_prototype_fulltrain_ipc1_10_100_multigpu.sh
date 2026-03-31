#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-fdd}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${PROJECT_ROOT}/output}"
MODEL_START_STAGGER_SEC="${MODEL_START_STAGGER_SEC:-15}"
RUN_FIRST_TASK_ONLY="${RUN_FIRST_TASK_ONLY:-0}"
AUTO_PROTOTYPE_TEMPERATURE="${AUTO_PROTOTYPE_TEMPERATURE:-1}"
PROTOTYPE_TEMPERATURE_MIN="${PROTOTYPE_TEMPERATURE_MIN:-0.01}"
PROTOTYPE_TEMPERATURE_MAX="${PROTOTYPE_TEMPERATURE_MAX:-1.0}"
PROTOTYPE_TEMPERATURE_STEPS="${PROTOTYPE_TEMPERATURE_STEPS:-41}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}" || {
      echo "Failed to activate conda env: ${CONDA_ENV_NAME}" >&2
      exit 1
    }
  fi
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${BASE_OUTPUT_ROOT}/prototype_eval_${RUN_TS}"
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

IPC_LIST=(1 10 100)

MODEL_GPU_PAIRS=(
  "mobilenetv3_large|1"
  "mobileone_s4|1"
  "repvit_m1_5|2"
  "efficientformer_l1|3"
  "dinov2_vitb|4"
  "clip_vitb|5"
  "eva02_vitb|6"
  "mocov3_resnet50|7"
)

if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
fi
if [[ -n "${IPC_LIST_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a IPC_LIST <<< "${IPC_LIST_OVERRIDE}"
fi
if [[ -n "${MODEL_GPU_PAIRS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a MODEL_GPU_PAIRS <<< "${MODEL_GPU_PAIRS_OVERRIDE}"
fi
if (( RUN_FIRST_TASK_ONLY )); then
  DATASETS=("${DATASETS[0]}")
  IPC_LIST=("${IPC_LIST[0]}")
  MODEL_GPU_PAIRS=("${MODEL_GPU_PAIRS[0]}")
fi

cd "${PROJECT_ROOT}"
echo "${RUN_ROOT}" > "${PROJECT_ROOT}/latest_prototype_eval_root.txt"

resolve_barycenter_path() {
  local dataset="$1"
  local model="$2"
  local ipc="$3"
  "${PYTHON_BIN}" - <<PY
import glob, os
base = r"${BASE_OUTPUT_ROOT}"
dataset = r"${dataset}"
model = r"${model}"
ipc = int("${ipc}")
pattern = os.path.join(
    base,
    "frf_*",
    dataset,
    f"{dataset}_ipc{ipc}_dp*_noniid_*_clients*",
    model,
    "artifacts",
    "barycenter_targets.pth",
)
cands = [p for p in glob.glob(pattern) if os.path.isfile(p)]
cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
print(cands[0] if cands else "")
PY
}

run_model_queue() {
  local model="$1"
  local gpu_id="$2"
  local ipc="$3"
  local start_delay="$4"
  local run_log="${LOG_ROOT}/ipc${ipc}_${model}_gpu${gpu_id}.log"
  : > "${run_log}"

  if (( start_delay > 0 )); then
    echo "[$(date '+%F %T')] DELAY ipc=${ipc} gpu=${gpu_id} model=${model} seconds=${start_delay}" | tee -a "${run_log}"
    sleep "${start_delay}"
  fi

  local -a failed_tasks=()
  for dataset in "${DATASETS[@]}"; do
    local run_dir="${RUN_ROOT}/ipc${ipc}/${dataset}/${model}"
    mkdir -p "${run_dir}"

    local task_log="${RUN_ROOT}/ipc${ipc}_${dataset}_${model}.log"
    echo "[$(date '+%F %T')] START ipc=${ipc} gpu=${gpu_id} dataset=${dataset} model=${model}" | tee -a "${run_log}"
    local -a cmd=(
      "${PYTHON_BIN}" "${PROJECT_ROOT}/src/distillation/eval.py"
      --dataset "${dataset}"
      --model "${model}"
      --eval_model "${model}"
      --data_root "${DATA_ROOT}"
      --real_res 224
      --crop_res 224
      --train_mode cosineprototype
      --prototype_source full_train
      --auto_prototype_temperature "${AUTO_PROTOTYPE_TEMPERATURE}"
      --prototype_temperature_min "${PROTOTYPE_TEMPERATURE_MIN}"
      --prototype_temperature_max "${PROTOTYPE_TEMPERATURE_MAX}"
      --prototype_temperature_steps "${PROTOTYPE_TEMPERATURE_STEPS}"
      --run_dir_override "${run_dir}"
      --skip_if_exists True
    )
    if CUDA_VISIBLE_DEVICES="${gpu_id}" \
      HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
      TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
      TORCH_HOME="${PROJECT_ROOT}/pretrained_models" \
      HF_HOME="${PROJECT_ROOT}/pretrained_models" \
      PYTHONPATH="${PROJECT_ROOT}/src" \
      "${cmd[@]}" >> "${task_log}" 2>&1; then
      echo "[$(date '+%F %T')] END   ipc=${ipc} gpu=${gpu_id} dataset=${dataset} model=${model}" | tee -a "${run_log}"
    else
      echo "[$(date '+%F %T')] FAIL  ipc=${ipc} gpu=${gpu_id} dataset=${dataset} model=${model}" | tee -a "${run_log}"
      failed_tasks+=("${ipc}|${dataset}|${model}|gpu${gpu_id}|runtime_fail")
    fi
  done

  if (( ${#failed_tasks[@]} > 0 )); then
    printf '%s\n' "${failed_tasks[@]}" > "${LOG_ROOT}/failed_ipc${ipc}_${model}_gpu${gpu_id}.txt"
    return 1
  fi
  return 0
}

overall_rc=0
for ipc in "${IPC_LIST[@]}"; do
  echo "[$(date '+%F %T')] IPC_STAGE_START ipc=${ipc}" | tee -a "${LOG_ROOT}/driver.log"
  declare -a pids=()
  idx=0
  for pair in "${MODEL_GPU_PAIRS[@]}"; do
    IFS='|' read -r model gpu_id <<< "${pair}"
    start_delay=$(( idx * MODEL_START_STAGGER_SEC ))
    run_model_queue "${model}" "${gpu_id}" "${ipc}" "${start_delay}" &
    pids+=("$!")
    idx=$(( idx + 1 ))
  done

  ipc_rc=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      ipc_rc=1
    fi
  done
  if (( ipc_rc != 0 )); then
    overall_rc=1
  fi
  echo "[$(date '+%F %T')] IPC_STAGE_END ipc=${ipc} rc=${ipc_rc}" | tee -a "${LOG_ROOT}/driver.log"
done

exit "${overall_rc}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-fdd}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-${PROJECT_ROOT}/pretrained_models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output}"
FULL_DATASET_ROOT="${FULL_DATASET_ROOT:-${OUTPUT_ROOT}/full_dataset}"
LOG_ROOT="${FULL_DATASET_ROOT}/_runlogs"

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

mkdir -p "${LOG_ROOT}"

GPU_IDS="${GPU_IDS:-1,2,3,4}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} != 4 )); then
  echo "run_full_dataset_multigpu.sh requires exactly 4 GPU ids, got: ${GPU_IDS}" >&2
  exit 1
fi

AVAILABLE_GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  AVAILABLE_GPU_COUNT="$(timeout 8s nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || true)"
fi
if (( AVAILABLE_GPU_COUNT > 0 )); then
  for gpu in "${GPU_ARRAY[@]}"; do
    if (( gpu < 0 || gpu >= AVAILABLE_GPU_COUNT )); then
      echo "GPU ${gpu} is not visible (visible count=${AVAILABLE_GPU_COUNT})" >&2
      exit 1
    fi
  done
fi

SSL_MODELS=("dinov2_vitb" "clip_vitb" "eva02_vitb" "mocov3_resnet50")
LIGHT_IMAGENET_MODELS=("mobilenetv3_large" "mobileone_s4" "repvit_m1_5" "efficientformer_l1")
CENTRAL_DATASET_ORDER=(
  "cifar100" "cifar10" "cub2011" "spawrious" "stanforddogs"
  "waterbirds" "flowers102" "food101" "artbench" "imagenet-100" "imagenet-1k"
)

if [[ -n "${SSL_MODELS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a SSL_MODELS <<< "${SSL_MODELS_OVERRIDE}"
fi
if [[ -n "${LIGHT_IMAGENET_MODELS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a LIGHT_IMAGENET_MODELS <<< "${LIGHT_IMAGENET_MODELS_OVERRIDE}"
fi
if [[ -n "${CENTRAL_DATASETS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a CENTRAL_DATASETS <<< "${CENTRAL_DATASETS_OVERRIDE}"
else
  declare -a CENTRAL_DATASETS=()
  for ds in "${CENTRAL_DATASET_ORDER[@]}"; do
    case "${ds}" in
      "stanforddogs")
        [[ -d "${DATA_ROOT}/StanfordDogs" || -d "${DATA_ROOT}/stanforddogs" ]] && CENTRAL_DATASETS+=("${ds}")
        ;;
      "cub2011")
        [[ -d "${DATA_ROOT}/cub" || -d "${DATA_ROOT}/cub2011" ]] && CENTRAL_DATASETS+=("${ds}")
        ;;
      "imagenet-1k"|"imagenet-100")
        [[ -d "${DATA_ROOT}/imagenet" || -d "${DATA_ROOT}/${ds}" ]] && CENTRAL_DATASETS+=("${ds}")
        ;;
      *)
        [[ -d "${DATA_ROOT}/${ds}" ]] && CENTRAL_DATASETS+=("${ds}")
        ;;
    esac
  done
fi
if (( ${#CENTRAL_DATASETS[@]} == 0 )); then
  echo "No centralized datasets detected under $(sanitize_path "${DATA_ROOT}")" >&2
  exit 1
fi

SEEDS="${SEEDS:-3407}"
MAX_ROUNDS="${MAX_ROUNDS:-1000}"
REAL_RES="${REAL_RES:-256}"
CROP_RES="${CROP_RES:-224}"
TRAIN_CROP_MODE="${TRAIN_CROP_MODE:-random}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-16}"
LOCAL_NUM_WORKERS="${LOCAL_NUM_WORKERS:-16}"
USE_AMP="${USE_AMP:-1}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"
PATIENCE_ROUNDS="${PATIENCE_ROUNDS:-10}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-20}"
GPU_START_STAGGER_SECONDS="${GPU_START_STAGGER_SECONDS:-30}"
INTRA_GPU_STAGE2_STAGGER_SECONDS="${INTRA_GPU_STAGE2_STAGGER_SECONDS:-10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_TASK1="${RUN_TASK1:-1}"
RUN_TASK2="${RUN_TASK2:-1}"

LP_BATCH_SIZE="${LP_BATCH_SIZE:-64}"
LP_FEATURE_BATCH_SIZE="${LP_FEATURE_BATCH_SIZE:-1024}"
LP_LOCAL_LR="${LP_LOCAL_LR:-0.001}"
LP_CACHE_FEATURES="${LP_CACHE_FEATURES:-1}"

FT_BATCH_SIZE="${FT_BATCH_SIZE:-64}"
FT_LOCAL_LR="${FT_LOCAL_LR:-0.001}"
FT_BACKBONE_LR="${FT_BACKBONE_LR:-0.0001}"
FT_WEIGHT_DECAY="${FT_WEIGHT_DECAY:-0.0}"
SSL_LP_CACHE_FEATURES="${SSL_LP_CACHE_FEATURES:-1}"
LIGHT_LP_CACHE_FEATURES="${LIGHT_LP_CACHE_FEATURES:-0}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

mode_dir_name() {
  case "$1" in
    linear_probe) echo "linearprobing" ;;
    finetune) echo "finetuning" ;;
    *)
      echo "Unsupported train_mode: $1" >&2
      return 1
      ;;
  esac
}

sanitize_path() {
  local raw="$1"
  if [[ -z "${raw}" ]]; then
    printf '%s\n' ""
    return 0
  fi
  if [[ "${raw}" == "${PROJECT_ROOT}" ]]; then
    printf '%s\n' '${PROJECT_ROOT}'
    return 0
  fi
  if [[ "${raw}" == "${PROJECT_ROOT}/"* ]]; then
    printf '%s\n' "\${PROJECT_ROOT}/${raw#"${PROJECT_ROOT}/"}"
    return 0
  fi
  printf '%s\n' "${raw}" | sed -E 's#^/data/[^/]+#/data/<user>#; s#^/home/[^/]+#/home/<user>#'
}

contains_item() {
  local needle="$1"
  shift
  local x
  for x in "$@"; do
    if [[ "${x}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

task_dir() {
  local train_mode="$1"
  local dataset="$2"
  local model="$3"
  local mode_dir
  mode_dir="$(mode_dir_name "${train_mode}")"
  printf '%s\n' "${FULL_DATASET_ROOT}/${mode_dir}/${dataset}/${model}"
}

task_log_file() {
  local train_mode="$1"
  local dataset="$2"
  local model="$3"
  local mode_dir
  mode_dir="$(mode_dir_name "${train_mode}")"
  printf '%s\n' "${FULL_DATASET_ROOT}/${mode_dir}_meta/logs/full_dataset_${mode_dir}_${dataset}_${model}.log"
}

task_is_complete() {
  local train_mode="$1"
  local dataset="$2"
  local model="$3"
  local out_dir
  out_dir="$(task_dir "${train_mode}" "${dataset}" "${model}")"
  [[ -f "${out_dir}/result_summary.json" ]] || return 1
  IFS=',' read -r -a seed_array <<< "${SEEDS}"
  local seed
  for seed in "${seed_array[@]}"; do
    [[ -f "${out_dir}/seed_${seed}/round_metrics.jsonl" ]] || return 1
  done
  return 0
}

prepare_task_outputs() {
  local train_mode="$1"
  local dataset="$2"
  local model="$3"
  local out_dir
  local log_file
  out_dir="$(task_dir "${train_mode}" "${dataset}" "${model}")"
  log_file="$(task_log_file "${train_mode}" "${dataset}" "${model}")"
  mkdir -p "${out_dir}" "$(dirname "${log_file}")"
  if [[ "${SKIP_EXISTING}" == "1" ]] && task_is_complete "${train_mode}" "${dataset}" "${model}"; then
    echo "[$(date '+%F %T')] SKIP  train_mode=${train_mode} dataset=${dataset} model=${model} reason=already_complete"
    return 1
  fi
  return 0
}

run_one_task() {
  local gpu_id="$1"
  local train_mode="$2"
  local dataset="$3"
  local model="$4"
  local log_file
  log_file="$(task_log_file "${train_mode}" "${dataset}" "${model}")"

  local local_batch_size feature_batch_size local_lr backbone_lr weight_decay cache_flag
  if [[ "${train_mode}" == "linear_probe" ]]; then
    local_batch_size="${LP_BATCH_SIZE}"
    feature_batch_size="${LP_FEATURE_BATCH_SIZE}"
    local_lr="${LP_LOCAL_LR}"
    backbone_lr="${FT_BACKBONE_LR}"
    weight_decay="0.0"
    if contains_item "${model}" "${LIGHT_IMAGENET_MODELS[@]}"; then
      cache_flag="${LIGHT_LP_CACHE_FEATURES}"
    else
      cache_flag="${SSL_LP_CACHE_FEATURES}"
    fi
  else
    local_batch_size="${FT_BATCH_SIZE}"
    feature_batch_size="${LP_FEATURE_BATCH_SIZE}"
    local_lr="${FT_LOCAL_LR}"
    backbone_lr="${FT_BACKBONE_LR}"
    weight_decay="${FT_WEIGHT_DECAY}"
    cache_flag="0"
  fi

  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/src/baselines/run_baselines.py" \
    --method full_dataset \
    --dataset "${dataset}" \
    --model "${model}" \
    --data_root "${DATA_ROOT}" \
    --pretrained_root "${PRETRAINED_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --full_dataset_output_layout by_mode \
    --train_mode "${train_mode}" \
    --seeds "${SEEDS}" \
    --max_rounds "${MAX_ROUNDS}" \
    --real_res "${REAL_RES}" \
    --crop_res "${CROP_RES}" \
    --train_crop_mode "${TRAIN_CROP_MODE}" \
    --local_batch_size "${local_batch_size}" \
    --feature_batch_size "${feature_batch_size}" \
    --local_lr "${local_lr}" \
    --backbone_lr "${backbone_lr}" \
    --weight_decay "${weight_decay}" \
    --grad_clip_norm "${GRAD_CLIP_NORM}" \
    --patience_rounds "${PATIENCE_ROUNDS}" \
    --min_delta "${MIN_DELTA}" \
    --warmup_rounds "${WARMUP_ROUNDS}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --eval_num_workers "${EVAL_NUM_WORKERS}" \
    --local_num_workers "${LOCAL_NUM_WORKERS}" \
    $([[ "${USE_AMP}" == "1" ]] && printf '%s ' --use_amp || printf '%s ' --no_amp) \
    $([[ "${cache_flag}" == "1" ]] && printf '%s ' --cache_features || printf '%s' '') \
    >> "${log_file}" 2>&1
}

run_one_task_bg() {
  local gpu_id="$1"
  local train_mode="$2"
  local dataset="$3"
  local model="$4"
  echo "[$(date '+%F %T')] START gpu=${gpu_id} train_mode=${train_mode} dataset=${dataset} model=${model}"
  if run_one_task "${gpu_id}" "${train_mode}" "${dataset}" "${model}"; then
    echo "[$(date '+%F %T')] END   gpu=${gpu_id} train_mode=${train_mode} dataset=${dataset} model=${model}"
    return 0
  fi
  echo "[$(date '+%F %T')] FAIL  gpu=${gpu_id} train_mode=${train_mode} dataset=${dataset} model=${model}" >&2
  return 1
}

wait_pids_or_fail() {
  local failures=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failures=$((failures + 1))
    fi
  done
  (( failures == 0 ))
}

run_task1_gpu_queue() {
  local gpu_id="$1"
  local model="$2"
  local dataset
  echo "[$(date '+%F %T')] TASK1 GPU-QUEUE START gpu=${gpu_id} model=${model}"
  for dataset in "${CENTRAL_DATASETS[@]}"; do
    if prepare_task_outputs "linear_probe" "${dataset}" "${model}"; then
      run_one_task_bg "${gpu_id}" "linear_probe" "${dataset}" "${model}" || return 1
    fi
  done
  echo "[$(date '+%F %T')] TASK1 GPU-QUEUE END   gpu=${gpu_id} model=${model}"
}

run_task2_gpu_queue() {
  local gpu_id="$1"
  local model="$2"
  local dataset
  echo "[$(date '+%F %T')] TASK2 GPU-QUEUE START gpu=${gpu_id} model=${model}"
  for dataset in "${CENTRAL_DATASETS[@]}"; do
    if prepare_task_outputs "linear_probe" "${dataset}" "${model}"; then
      run_one_task_bg "${gpu_id}" "linear_probe" "${dataset}" "${model}" || return 1
    fi
    sleep "${INTRA_GPU_STAGE2_STAGGER_SECONDS}"
    if prepare_task_outputs "finetune" "${dataset}" "${model}"; then
      run_one_task_bg "${gpu_id}" "finetune" "${dataset}" "${model}" || return 1
    fi
  done
  echo "[$(date '+%F %T')] TASK2 GPU-QUEUE END   gpu=${gpu_id} model=${model}"
}

echo "Output root: $(sanitize_path "${OUTPUT_ROOT}")"
echo "Full dataset root: $(sanitize_path "${FULL_DATASET_ROOT}")"
echo "GPUs: ${GPU_IDS}"
echo "Datasets: ${CENTRAL_DATASETS[*]}"
echo "TASK1 SSL models: ${SSL_MODELS[*]}"
echo "TASK2 lightweight ImageNet models: ${LIGHT_IMAGENET_MODELS[*]}"
echo "RUN_TASK1=${RUN_TASK1}"
echo "RUN_TASK2=${RUN_TASK2}"
echo "LP batch/lr/cache(ssl/light): ${LP_BATCH_SIZE}/${LP_LOCAL_LR}/${SSL_LP_CACHE_FEATURES}/${LIGHT_LP_CACHE_FEATURES}"
echo "FT batch/head_lr/backbone_lr/wd: ${FT_BATCH_SIZE}/${FT_LOCAL_LR}/${FT_BACKBONE_LR}/${FT_WEIGHT_DECAY}"

declare -a queue_pids=()
declare -a queue_names=()

if [[ "${RUN_TASK1}" == "1" ]]; then
  for idx in "${!SSL_MODELS[@]}"; do
    gpu="${GPU_ARRAY[$idx]}"
    model="${SSL_MODELS[$idx]}"
    run_task1_gpu_queue "${gpu}" "${model}" &
    queue_pids+=("$!")
    queue_names+=("TASK1 gpu=${gpu} model=${model}")
    if (( idx + 1 < ${#SSL_MODELS[@]} )); then
      sleep "${GPU_START_STAGGER_SECONDS}"
    fi
  done
fi

if [[ "${RUN_TASK2}" == "1" ]]; then
  for idx in "${!LIGHT_IMAGENET_MODELS[@]}"; do
    gpu="${GPU_ARRAY[$idx]}"
    model="${LIGHT_IMAGENET_MODELS[$idx]}"
    run_task2_gpu_queue "${gpu}" "${model}" &
    queue_pids+=("$!")
    queue_names+=("TASK2 gpu=${gpu} model=${model}")
    if (( idx + 1 < ${#LIGHT_IMAGENET_MODELS[@]} )); then
      sleep "${GPU_START_STAGGER_SECONDS}"
    fi
  done
fi

if (( ${#queue_pids[@]} > 0 )); then
  wait_pids_or_fail "${queue_pids[@]}"
fi

echo "[$(date '+%F %T')] ALL DONE"

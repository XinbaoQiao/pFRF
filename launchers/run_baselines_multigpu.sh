#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-fdd}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-${PROJECT_ROOT}/pretrained_models}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
RUN_ROOT_OVERRIDE="${RUN_ROOT_OVERRIDE:-}"
RERUN_INCOMPLETE="${RERUN_INCOMPLETE:-1}"
SKIP_IF_DONE_ANYWHERE="${SKIP_IF_DONE_ANYWHERE:-1}"

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

NUM_GPUS="${NUM_GPUS:-4}"
if (( NUM_GPUS < 1 )); then
  echo "NUM_GPUS must be >= 1" >&2
  exit 1
fi

AVAILABLE_GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  # GPU0 can be in a bad state and make nvidia-smi hang; guard with timeout.
  AVAILABLE_GPU_COUNT="$(timeout 8s nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || true)"
fi

GPU_IDS="${GPU_IDS:-1,2,3,4}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} == 0 )); then
  echo "GPU_IDS is empty" >&2
  exit 1
fi
if (( NUM_GPUS != ${#GPU_ARRAY[@]} )); then
  echo "NUM_GPUS=${NUM_GPUS} does not match GPU count in GPU_IDS (${#GPU_ARRAY[@]})" >&2
  exit 1
fi
if (( AVAILABLE_GPU_COUNT > 0 )); then
  for gpu in "${GPU_ARRAY[@]}"; do
    if (( gpu < 0 || gpu >= AVAILABLE_GPU_COUNT )); then
      echo "GPU ${gpu} is not visible (visible count=${AVAILABLE_GPU_COUNT})" >&2
      exit 1
    fi
  done
fi

# FL stage order: prioritize the main benchmark datasets first, then the rest.
FL_DATASETS=(
  "cifar10" "cifar100" "imagenet-100" "imagenet-1k" "cub2011"
  "flowers102" "food101" "imagenette" "spawrious" "stanforddogs"
  "waterbirds" "artbench"
)
# Centralized stage order: keep ImageNet last so other datasets finish earlier.
CENTRAL_DATASET_ORDER=(
  "cifar100" "cifar10" "cub2011" "spawrious" "stanforddogs"
  "waterbirds" "flowers102" "food101" "artbench" "imagenet-100" "imagenet-1k"
)
MODELS=("dinov2_vitb" "clip_vitb" "eva02_vitb" "mocov3_resnet50")
METHODS=("fedprox" "ccvr" "fedpcl" "fedproto" "fedncm" "afl" "full_dataset" "lgm")
FL_BASELINES=("fedprox" "ccvr" "fedpcl" "fedproto" "fedncm" "afl")
GPU1_FL_PARALLEL="${GPU1_FL_PARALLEL:-2}"
GPU2_FL_PARALLEL="${GPU2_FL_PARALLEL:-2}"
GPU3_FL_PARALLEL="${GPU3_FL_PARALLEL:-2}"
GPU4_FL_PARALLEL="${GPU4_FL_PARALLEL:-3}"
GPU_START_STAGGER_SECONDS="${GPU_START_STAGGER_SECONDS:-60}"
LGM_AUGS_IMAGENET1K="${LGM_AUGS_IMAGENET1K:-3}"
LGM_AUGS_OTHER="${LGM_AUGS_OTHER:-10}"

if [[ -n "${ALL_DATASETS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a FL_DATASETS <<< "${ALL_DATASETS_OVERRIDE}"
  CENTRAL_DATASET_ORDER=("${FL_DATASETS[@]}")
elif [[ -n "${FL_DATASETS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a FL_DATASETS <<< "${FL_DATASETS_OVERRIDE}"
elif [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a FL_DATASETS <<< "${DATASETS_OVERRIDE}"
fi
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "${MODELS_OVERRIDE}"
fi
if [[ -n "${METHODS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a METHODS <<< "${METHODS_OVERRIDE}"
fi

SEEDS="${SEEDS:-3407}"
CACHE_FEATURES="${CACHE_FEATURES:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-1000}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
PARTITION="${PARTITION:-dirichlet}"
DIRICHLET_BALANCE="${DIRICHLET_BALANCE:-1}"
DIRICHLET_MIN_SIZE="${DIRICHLET_MIN_SIZE:-1}"
SHARD_PER_CLIENT="${SHARD_PER_CLIENT:-2}"
CLASSES_PER_CLIENT="${CLASSES_PER_CLIENT:-2}"
# Keep FL and centralized hyper-parameters decoupled.
# Backward-compat: LOCAL_* still works as alias for FL_* if FL_* is unset.
FL_LOCAL_EPOCHS="${FL_LOCAL_EPOCHS:-${LOCAL_EPOCHS:-1}}"
FL_LOCAL_LR="${FL_LOCAL_LR:-${LOCAL_LR:-0.05}}"
FL_LOCAL_BATCH_SIZE="${FL_LOCAL_BATCH_SIZE:-${LOCAL_BATCH_SIZE:-256}}"
FL_LOCAL_MOMENTUM="${FL_LOCAL_MOMENTUM:-${LOCAL_MOMENTUM:-0.9}}"
FL_OPTIMIZER="${FL_OPTIMIZER:-adam}"
FL_SCHEDULER="${FL_SCHEDULER:-cosine}"

# Centralized defaults intentionally differ from FL defaults.
CENTRAL_LOCAL_LR="${CENTRAL_LOCAL_LR:-0.001}"
CENTRAL_LOCAL_BATCH_SIZE="${CENTRAL_LOCAL_BATCH_SIZE:-64}"
CENTRAL_LOCAL_MOMENTUM="${CENTRAL_LOCAL_MOMENTUM:-0.9}"

GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"
# Use a balanced default early-stop setting.
PATIENCE_ROUNDS="${PATIENCE_ROUNDS:-10}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-20}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
FULLDATASET_STAGE0_IMAGENET_FIRST="${FULLDATASET_STAGE0_IMAGENET_FIRST:-0}"
FL_STAGE_DELAY_SECONDS="${FL_STAGE_DELAY_SECONDS:-0}"

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
if (( ${#SEED_ARRAY[@]} == 0 )); then
  echo "SEEDS is empty" >&2
  exit 1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${RUN_ROOT_OVERRIDE}" ]]; then
  RUN_ROOT="${RUN_ROOT_OVERRIDE}"
else
  RUN_ROOT="${OUTPUT_BASE}/baseline_${RUN_TS}"
fi
LOG_ROOT="${RUN_ROOT}/logs"
mkdir -p "${LOG_ROOT}"
FAILED_TASKS_FILE="${LOG_ROOT}/failed_tasks.txt"
: > "${FAILED_TASKS_FILE}"

# Discover centralized datasets from data root, with canonical name mapping.
declare -a CENTRAL_DATASETS=()
if [[ -n "${CENTRAL_DATASETS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a CENTRAL_DATASETS <<< "${CENTRAL_DATASETS_OVERRIDE}"
else
  # Default centralized dataset order.
  declare -a CANDIDATES=("${CENTRAL_DATASET_ORDER[@]}")
  for ds in "${CANDIDATES[@]}"; do
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
  echo "No centralized datasets detected under ${DATA_ROOT}" >&2
  exit 1
fi

parse_csv_to_array() {
  local raw="$1"
  local -n arr_ref="$2"
  IFS=',' read -r -a arr_ref <<< "${raw}"
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

get_model_gpu() {
  local model="$1"
  case "${model}" in
    "dinov2_vitb") echo "1" ;;
    "clip_vitb") echo "2" ;;
    "eva02_vitb") echo "3" ;;
    "mocov3_resnet50") echo "4" ;;
    *) return 1 ;;
  esac
}

validate_model_gpu_bindings() {
  local model
  for model in "${MODELS[@]}"; do
    local g
    if ! g="$(get_model_gpu "${model}")"; then
      echo "Model ${model} has no fixed GPU binding" >&2
      return 1
    fi
    if ! contains_item "${g}" "${GPU_ARRAY[@]}"; then
      echo "Model ${model} requires GPU ${g}, but GPU_IDS=${GPU_IDS}" >&2
      return 1
    fi
  done
}

task_dir() {
  local method="$1"
  local dataset="$2"
  local model="$3"
  printf '%s\n' "${RUN_ROOT}/${method}/${dataset}/${model}"
}

task_log_file() {
  local method="$1"
  local dataset="$2"
  local model="$3"
  printf '%s\n' "${LOG_ROOT}/${method}_${dataset}_${model}.log"
}

is_centralized_method() {
  local method="$1"
  [[ "${method}" == "lgm" || "${method}" == "full_dataset" ]]
}

task_is_complete() {
  local method="$1"
  local dataset="$2"
  local model="$3"
  local seed
  local out_dir
  out_dir="$(task_dir "${method}" "${dataset}" "${model}")"
  if is_centralized_method "${method}"; then
    for seed in "${SEED_ARRAY[@]}"; do
      if [[ ! -f "${out_dir}/seed_${seed}/round_metrics.jsonl" ]]; then
        return 1
      fi
    done
    [[ -f "${out_dir}/result_summary.json" ]]
    return
  fi
  for seed in "${SEED_ARRAY[@]}"; do
    if [[ ! -f "${out_dir}/seed_${seed}/result_seed.json" ]]; then
      return 1
    fi
  done
  [[ -f "${out_dir}/result_summary.json" ]]
}

task_is_complete_anywhere() {
  local method="$1"
  local dataset="$2"
  local model="$3"
  local seed
  local summary_count
  summary_count="$(
    find "${OUTPUT_BASE}" \
      -path "${OUTPUT_BASE}/cost_estimates" -prune -o \
      -path "*/${method}/${dataset}/${model}/result_summary.json" -print | wc -l
  )"
  if [[ "${summary_count}" -eq 0 ]]; then
    return 1
  fi
  if is_centralized_method "${method}"; then
    for seed in "${SEED_ARRAY[@]}"; do
      local seed_count
      seed_count="$(
        find "${OUTPUT_BASE}" \
          -path "${OUTPUT_BASE}/cost_estimates" -prune -o \
          -path "*/${method}/${dataset}/${model}/seed_${seed}/round_metrics.jsonl" -print | wc -l
      )"
      if [[ "${seed_count}" -eq 0 ]]; then
        return 1
      fi
    done
    return 0
  fi
  for seed in "${SEED_ARRAY[@]}"; do
    local seed_count
    seed_count="$(
      find "${OUTPUT_BASE}" \
        -path "${OUTPUT_BASE}/cost_estimates" -prune -o \
        -path "*/${method}/${dataset}/${model}/seed_${seed}/result_seed.json" -print | wc -l
    )"
    if [[ "${seed_count}" -eq 0 ]]; then
      return 1
    fi
  done
  return 0
}

prepare_task_outputs() {
  local method="$1"
  local dataset="$2"
  local model="$3"
  local out_dir
  local log_file
  local backup_suffix
  out_dir="$(task_dir "${method}" "${dataset}" "${model}")"
  log_file="$(task_log_file "${method}" "${dataset}" "${model}")"

  if task_is_complete "${method}" "${dataset}" "${model}"; then
    echo "[$(date '+%F %T')] SKIP  method=${method} dataset=${dataset} model=${model} reason=already_complete"
    return 1
  fi
  if [[ "${SKIP_IF_DONE_ANYWHERE}" == "1" ]] && task_is_complete_anywhere "${method}" "${dataset}" "${model}"; then
    echo "[$(date '+%F %T')] SKIP  method=${method} dataset=${dataset} model=${model} reason=complete_under_output_root"
    return 1
  fi

  if [[ "${RERUN_INCOMPLETE}" == "1" && -d "${out_dir}" ]]; then
    backup_suffix="incomplete_backup_${RUN_TS}"
    mv "${out_dir}" "${out_dir}.${backup_suffix}"
    echo "[$(date '+%F %T')] RESET method=${method} dataset=${dataset} model=${model} moved_to=${out_dir}.${backup_suffix}"
  fi
  if [[ "${RERUN_INCOMPLETE}" == "1" && -f "${log_file}" ]]; then
    backup_suffix="incomplete_backup_${RUN_TS}"
    mv "${log_file}" "${log_file}.${backup_suffix}"
  fi
  mkdir -p "${out_dir}"
  return 0
}

run_one_task() {
  local gpu_id="$1"
  local method="$2"
  local dataset="$3"
  local model="$4"

  local exp_out
  local log_file
  exp_out="$(task_dir "${method}" "${dataset}" "${model}")"
  log_file="$(task_log_file "${method}" "${dataset}" "${model}")"
  local -a extra_args=()
  local local_lr local_batch_size local_momentum
  if is_centralized_method "${method}"; then
    local_lr="${CENTRAL_LOCAL_LR}"
    local_batch_size="${CENTRAL_LOCAL_BATCH_SIZE}"
    local_momentum="${CENTRAL_LOCAL_MOMENTUM}"
  else
    extra_args+=("--local_epochs" "${FL_LOCAL_EPOCHS}")
    local_lr="${FL_LOCAL_LR}"
    local_batch_size="${FL_LOCAL_BATCH_SIZE}"
    local_momentum="${FL_LOCAL_MOMENTUM}"
  fi
  if [[ "${CACHE_FEATURES}" == "1" ]]; then
    extra_args+=("--cache_features" "true")
  else
    extra_args+=("--cache_features" "false")
  fi
  if [[ "${method}" == "lgm" ]]; then
    if [[ "${dataset}" == "imagenet-1k" ]]; then
      extra_args+=("--augs_per_batch" "${LGM_AUGS_IMAGENET1K}")
    else
      extra_args+=("--augs_per_batch" "${LGM_AUGS_OTHER}")
    fi
  fi
  mkdir -p "${exp_out}"

  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/src/baselines/run_baselines.py" \
    --method "${method}" \
    --dataset "${dataset}" \
    --model "${model}" \
    --data_root "${DATA_ROOT}" \
    --pretrained_root "${PRETRAINED_ROOT}" \
    --output_root "${RUN_ROOT}" \
    --seeds "${SEEDS}" \
    --max_rounds "${MAX_ROUNDS}" \
    --num_clients "${NUM_CLIENTS}" \
    --partition "${PARTITION}" \
    $([[ "${DIRICHLET_BALANCE}" == "1" ]] && printf '%s ' --dirichlet_balance || printf '%s ' --no_dirichlet_balance) \
    --dirichlet_min_size "${DIRICHLET_MIN_SIZE}" \
    --shard_per_client "${SHARD_PER_CLIENT}" \
    --classes_per_client "${CLASSES_PER_CLIENT}" \
    --local_batch_size "${local_batch_size}" \
    --local_lr "${local_lr}" \
    --fl_optimizer "${FL_OPTIMIZER}" \
    --fl_scheduler "${FL_SCHEDULER}" \
    --local_momentum "${local_momentum}" \
    --grad_clip_norm "${GRAD_CLIP_NORM}" \
    --patience_rounds "${PATIENCE_ROUNDS}" \
    --min_delta "${MIN_DELTA}" \
    --warmup_rounds "${WARMUP_ROUNDS}" \
    --ccvr_calib_epochs "${CCVR_CALIB_EPOCHS:-5}" \
    --ccvr_calib_samples_per_class "${CCVR_CALIB_SPC:-100}" \
    --ccvr_calib_lr "${CCVR_CALIB_LR:-0.001}" \
    "${extra_args[@]}" \
    >> "${log_file}" 2>&1
}

run_one_task_bg() {
  local gpu_id="$1"
  local method="$2"
  local dataset="$3"
  local model="$4"
  echo "[$(date '+%F %T')] START gpu=${gpu_id} method=${method} dataset=${dataset} model=${model}"
  if run_one_task "${gpu_id}" "${method}" "${dataset}" "${model}"; then
    echo "[$(date '+%F %T')] END   gpu=${gpu_id} method=${method} dataset=${dataset} model=${model}"
    return 0
  fi
  echo "[$(date '+%F %T')] FAIL  gpu=${gpu_id} method=${method} dataset=${dataset} model=${model}" >&2
  # Do not abort the whole batch because one task fails (e.g., single-task OOM).
  printf '[%s] gpu=%s method=%s dataset=%s model=%s\n' \
    "$(date '+%F %T')" "${gpu_id}" "${method}" "${dataset}" "${model}" >> "${FAILED_TASKS_FILE}"
  return 0
}

method_enabled() {
  local method="$1"
  contains_item "${method}" "${METHODS[@]}"
}

wait_pids_or_fail() {
  local failures=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failures=$((failures + 1))
    fi
  done
  if (( failures > 0 )); then
    return 1
  fi
  return 0
}

run_gpu_method_queue_batched() {
  local gpu_id="$1"
  local dataset="$2"
  local model="$3"
  local max_parallel="$4"
  shift 4
  if (( max_parallel < 1 )); then
    echo "Invalid max_parallel=${max_parallel} for gpu=${gpu_id}" >&2
    return 1
  fi
  local method
  local ran_any=0
  local -a batch_pids=()
  local active=0
  for method in "$@"; do
    if ! method_enabled "${method}"; then
      continue
    fi
    if prepare_task_outputs "${method}" "${dataset}" "${model}"; then
      ran_any=1
      run_one_task_bg "${gpu_id}" "${method}" "${dataset}" "${model}" &
      batch_pids+=("$!")
      active=$((active + 1))
      if (( active >= max_parallel )); then
        if ! wait_pids_or_fail "${batch_pids[@]}"; then
          echo "[$(date '+%F %T')] FAIL  gpu=${gpu_id} dataset=${dataset} model=${model} reason=batched_methods_failed" >&2
          return 1
        fi
        batch_pids=()
        active=0
      fi
    fi
  done
  if (( active > 0 )); then
    if ! wait_pids_or_fail "${batch_pids[@]}"; then
      echo "[$(date '+%F %T')] FAIL  gpu=${gpu_id} dataset=${dataset} model=${model} reason=trailing_batched_methods_failed" >&2
      return 1
    fi
  fi
  if (( ran_any == 0 )); then
    echo "[$(date '+%F %T')] SKIP  gpu=${gpu_id} dataset=${dataset} model=${model} reason=no_pending_method"
  fi
  return 0
}

run_gpu_single_method_queue() {
  local gpu_id="$1"
  local dataset="$2"
  local model="$3"
  local method="$4"
  local ran_any=0
  if ! method_enabled "${method}"; then
    echo "[$(date '+%F %T')] SKIP  gpu=${gpu_id} method=${method} dataset=${dataset} model=${model} reason=method_disabled"
    return 0
  fi
  if prepare_task_outputs "${method}" "${dataset}" "${model}"; then
    ran_any=1
    if ! run_one_task_bg "${gpu_id}" "${method}" "${dataset}" "${model}"; then
      echo "[$(date '+%F %T')] FAIL  gpu=${gpu_id} method=${method} dataset=${dataset} model=${model}" >&2
      return 1
    fi
  fi
  if (( ran_any == 0 )); then
    echo "[$(date '+%F %T')] SKIP  gpu=${gpu_id} method=${method} dataset=${dataset} model=${model} reason=no_pending_method"
  fi
  return 0
}

run_gpu_single_method_queue_for_datasets() {
  local gpu_id="$1"
  local model="$2"
  local method="$3"
  shift 3
  local -a dataset_list=("$@")
  local dataset
  for dataset in "${dataset_list[@]}"; do
    if ! run_gpu_single_method_queue "${gpu_id}" "${dataset}" "${model}" "${method}"; then
      return 1
    fi
  done
  return 0
}

validate_parallel_rules() {
  if (( GPU1_FL_PARALLEL != 2 )); then
    echo "GPU1_FL_PARALLEL must be 2, got ${GPU1_FL_PARALLEL}" >&2
    return 1
  fi
  if (( GPU2_FL_PARALLEL != 2 )); then
    echo "GPU2_FL_PARALLEL must be 2, got ${GPU2_FL_PARALLEL}" >&2
    return 1
  fi
  if (( GPU3_FL_PARALLEL != 2 )); then
    echo "GPU3_FL_PARALLEL must be 2, got ${GPU3_FL_PARALLEL}" >&2
    return 1
  fi
  if (( GPU4_FL_PARALLEL != 3 )); then
    echo "GPU4_FL_PARALLEL must be 3, got ${GPU4_FL_PARALLEL}" >&2
    return 1
  fi
  return 0
}

validate_model_gpu_bindings
validate_parallel_rules

declare -a TASKS=()
for ds in "${FL_DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for method in "${FL_BASELINES[@]}"; do
      TASKS+=("${method}|${ds}|${model}")
    done
  done
done
for ds in "${CENTRAL_DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    TASKS+=("full_dataset|${ds}|${model}")
    TASKS+=("lgm|${ds}|${model}")
  done
done

echo "Run root: ${RUN_ROOT}"
echo "GPUs: ${GPU_IDS}"
if (( AVAILABLE_GPU_COUNT > 0 )); then
  echo "Visible GPU count: ${AVAILABLE_GPU_COUNT}"
fi
echo "Total tasks: ${#TASKS[@]}"
echo "FL dataset count: ${#FL_DATASETS[@]}"
echo "Centralized dataset count: ${#CENTRAL_DATASETS[@]}"
echo "FL datasets: ${FL_DATASETS[*]}"
echo "Centralized datasets: ${CENTRAL_DATASETS[*]}"
echo "Model count: ${#MODELS[@]}"
echo "Method count: ${#METHODS[@]}"
echo "Model->GPU binding: dinov2_vitb->1, clip_vitb->2, eva02_vitb->3, mocov3_resnet50->4"
echo "Stage1 centralized: full_dataset over all centralized datasets"
echo "Stage2 FL plan: all FL baselines per model queue (gpu1/2/3/4 parallel=2/2/2/3)"
echo "Stage2 FL in-GPU parallelism: gpu1=${GPU1_FL_PARALLEL}, gpu2=${GPU2_FL_PARALLEL}, gpu3=${GPU3_FL_PARALLEL}, gpu4=${GPU4_FL_PARALLEL}"
echo "Stage3 centralized: lgm over all centralized datasets (starts only after Stage1+Stage2 complete)"
echo "GPU launch stagger (seconds): ${GPU_START_STAGGER_SECONDS}"
echo "LGM augs_per_batch: imagenet-1k=${LGM_AUGS_IMAGENET1K}, others=${LGM_AUGS_OTHER}"
echo "SEEDS=${SEEDS}"
echo "MAX_ROUNDS=${MAX_ROUNDS}"
echo "NUM_CLIENTS=${NUM_CLIENTS}"
echo "PARTITION=${PARTITION}"
echo "DIRICHLET_BALANCE=${DIRICHLET_BALANCE}"
echo "DIRICHLET_MIN_SIZE=${DIRICHLET_MIN_SIZE}"
echo "SHARD_PER_CLIENT=${SHARD_PER_CLIENT}"
echo "CLASSES_PER_CLIENT=${CLASSES_PER_CLIENT}"
echo "FL_LOCAL_EPOCHS=${FL_LOCAL_EPOCHS}"
echo "FL_LOCAL_LR=${FL_LOCAL_LR}"
echo "FL_LOCAL_BATCH_SIZE=${FL_LOCAL_BATCH_SIZE}"
echo "FL_LOCAL_MOMENTUM=${FL_LOCAL_MOMENTUM}"
echo "CENTRAL_LOCAL_LR=${CENTRAL_LOCAL_LR}"
echo "CENTRAL_LOCAL_BATCH_SIZE=${CENTRAL_LOCAL_BATCH_SIZE}"
echo "CENTRAL_LOCAL_MOMENTUM=${CENTRAL_LOCAL_MOMENTUM}"
echo "GRAD_CLIP_NORM=${GRAD_CLIP_NORM}"
echo "PATIENCE_ROUNDS=${PATIENCE_ROUNDS}"
echo "MIN_DELTA=${MIN_DELTA}"
echo "WARMUP_ROUNDS=${WARMUP_ROUNDS}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"
echo "CCVR calib: epochs=${CCVR_CALIB_EPOCHS:-5}, samples/class=${CCVR_CALIB_SPC:-100}, base_lr=${CCVR_CALIB_LR:-0.05}"
echo "RUN_ROOT_OVERRIDE=${RUN_ROOT_OVERRIDE:-<new timestamped dir>}"
echo "RERUN_INCOMPLETE=${RERUN_INCOMPLETE}"
echo "SKIP_IF_DONE_ANYWHERE=${SKIP_IF_DONE_ANYWHERE}"
echo "FULLDATASET_STAGE0_IMAGENET_FIRST=${FULLDATASET_STAGE0_IMAGENET_FIRST}"
echo "FAILED_TASKS_FILE=${FAILED_TASKS_FILE}"

run_stage1_fl() {
  local dataset
  for dataset in "${FL_DATASETS[@]}"; do
    echo "[$(date '+%F %T')] STAGE1(FL) START dataset=${dataset}"
    declare -a STAGE1_PIDS=()

    run_gpu_method_queue_batched "1" "${dataset}" "dinov2_vitb" "${GPU1_FL_PARALLEL}" "${FL_BASELINES[@]}" &
    STAGE1_PIDS+=("$!")
    sleep "${GPU_START_STAGGER_SECONDS}"
    run_gpu_method_queue_batched "2" "${dataset}" "clip_vitb" "${GPU2_FL_PARALLEL}" "${FL_BASELINES[@]}" &
    STAGE1_PIDS+=("$!")
    sleep "${GPU_START_STAGGER_SECONDS}"
    run_gpu_method_queue_batched "3" "${dataset}" "eva02_vitb" "${GPU3_FL_PARALLEL}" "${FL_BASELINES[@]}" &
    STAGE1_PIDS+=("$!")
    sleep "${GPU_START_STAGGER_SECONDS}"
    run_gpu_method_queue_batched "4" "${dataset}" "mocov3_resnet50" "${GPU4_FL_PARALLEL}" "${FL_BASELINES[@]}" &
    STAGE1_PIDS+=("$!")

    if ! wait_pids_or_fail "${STAGE1_PIDS[@]}"; then
      echo "[$(date '+%F %T')] STAGE1(FL) FAIL dataset=${dataset}" >&2
      return 1
    fi
    echo "[$(date '+%F %T')] STAGE1(FL) END   dataset=${dataset}"
  done
  return 0
}

run_stage2_full_dataset_for_list() {
  local dataset
  local -a dataset_list=("$@")
  if ! method_enabled "full_dataset"; then
    return 0
  fi
  echo "[$(date '+%F %T')] STAGE2(CENTRALIZED) START method=full_dataset"
  declare -a STAGE2_PIDS=()
  run_gpu_single_method_queue_for_datasets "1" "dinov2_vitb" "full_dataset" "${dataset_list[@]}" &
  STAGE2_PIDS+=("$!")
  sleep "${GPU_START_STAGGER_SECONDS}"
  run_gpu_single_method_queue_for_datasets "2" "clip_vitb" "full_dataset" "${dataset_list[@]}" &
  STAGE2_PIDS+=("$!")
  sleep "${GPU_START_STAGGER_SECONDS}"
  run_gpu_single_method_queue_for_datasets "3" "eva02_vitb" "full_dataset" "${dataset_list[@]}" &
  STAGE2_PIDS+=("$!")
  sleep "${GPU_START_STAGGER_SECONDS}"
  run_gpu_single_method_queue_for_datasets "4" "mocov3_resnet50" "full_dataset" "${dataset_list[@]}" &
  STAGE2_PIDS+=("$!")
  if ! wait_pids_or_fail "${STAGE2_PIDS[@]}"; then
    echo "[$(date '+%F %T')] STAGE2(CENTRALIZED) FAIL method=full_dataset" >&2
    return 1
  fi
  echo "[$(date '+%F %T')] STAGE2(CENTRALIZED) END   method=full_dataset"
  return 0
}

# Optional pre-stage: prioritize full_dataset on ImageNet-1k/100 before the rest of full_dataset.
declare -a STAGE0_IMAGENET_DATASETS=()
declare -a STAGE2_REMAINING_DATASETS=()
if [[ "${FULLDATASET_STAGE0_IMAGENET_FIRST}" == "1" ]]; then
  for ds in "${CENTRAL_DATASETS[@]}"; do
    if [[ "${ds}" == "imagenet-1k" || "${ds}" == "imagenet-100" ]]; then
      STAGE0_IMAGENET_DATASETS+=("${ds}")
    else
      STAGE2_REMAINING_DATASETS+=("${ds}")
    fi
  done
else
  STAGE2_REMAINING_DATASETS=("${CENTRAL_DATASETS[@]}")
fi

if method_enabled "full_dataset" && (( ${#STAGE0_IMAGENET_DATASETS[@]} > 0 )); then
  echo "[$(date '+%F %T')] PRESTAGE(CENTRALIZED) START method=full_dataset datasets=${STAGE0_IMAGENET_DATASETS[*]}"
  if ! run_stage2_full_dataset_for_list "${STAGE0_IMAGENET_DATASETS[@]}"; then
    echo "[$(date '+%F %T')] PRESTAGE(CENTRALIZED) FAIL method=full_dataset" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] PRESTAGE(CENTRALIZED) END   method=full_dataset datasets=${STAGE0_IMAGENET_DATASETS[*]}"
fi

if method_enabled "full_dataset" && (( ${#STAGE2_REMAINING_DATASETS[@]} > 0 )); then
  if ! run_stage2_full_dataset_for_list "${STAGE2_REMAINING_DATASETS[@]}"; then
    echo "[$(date '+%F %T')] STAGE1(CENTRALIZED) FAIL method=full_dataset" >&2
    exit 1
  fi
fi

if (( FL_STAGE_DELAY_SECONDS > 0 )); then
  echo "[$(date '+%F %T')] STAGE2(FL) DELAY seconds=${FL_STAGE_DELAY_SECONDS}"
  sleep "${FL_STAGE_DELAY_SECONDS}"
fi
if ! run_stage1_fl; then
  echo "[$(date '+%F %T')] STAGE2(FL) FAIL" >&2
  exit 1
fi

# Stage3: lgm starts only after both Stage1(full_dataset) and Stage2(FL) are complete.
if method_enabled "lgm"; then
  echo "[$(date '+%F %T')] STAGE3(CENTRALIZED) START method=lgm"
  declare -a STAGE3_PIDS=()
  run_gpu_single_method_queue_for_datasets "1" "dinov2_vitb" "lgm" "${CENTRAL_DATASETS[@]}" &
  STAGE3_PIDS+=("$!")
  sleep "${GPU_START_STAGGER_SECONDS}"
  run_gpu_single_method_queue_for_datasets "2" "clip_vitb" "lgm" "${CENTRAL_DATASETS[@]}" &
  STAGE3_PIDS+=("$!")
  sleep "${GPU_START_STAGGER_SECONDS}"
  run_gpu_single_method_queue_for_datasets "3" "eva02_vitb" "lgm" "${CENTRAL_DATASETS[@]}" &
  STAGE3_PIDS+=("$!")
  sleep "${GPU_START_STAGGER_SECONDS}"
  run_gpu_single_method_queue_for_datasets "4" "mocov3_resnet50" "lgm" "${CENTRAL_DATASETS[@]}" &
  STAGE3_PIDS+=("$!")
  if ! wait_pids_or_fail "${STAGE3_PIDS[@]}"; then
    echo "[$(date '+%F %T')] STAGE3(CENTRALIZED) FAIL method=lgm" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] STAGE3(CENTRALIZED) END   method=lgm"
fi

echo "[$(date '+%F %T')] ALL DONE: ${RUN_ROOT}"
if [[ -s "${FAILED_TASKS_FILE}" ]]; then
  echo "[$(date '+%F %T')] DONE WITH FAILURES: $(wc -l < "${FAILED_TASKS_FILE}") task(s) failed; see ${FAILED_TASKS_FILE}" >&2
else
  echo "[$(date '+%F %T')] DONE WITH NO TASK FAILURES"
fi

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-${PROJECT_ROOT}/.conda}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LAUNCH_ROOT="${OUTPUT_BASE}/launch_mar22_${RUN_TS}"
mkdir -p "${LAUNCH_ROOT}"

COMMON_DATASETS=(
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

DP_DATASETS=(
  food101
  imagenette
  spawrious
  waterbirds
  artbench
  cifar10
  cifar100
  imagenet-100
  imagenet-1k
)

csv_join() {
  local IFS=','
  printf '%s' "$*"
}

write_session_script() {
  local script_path="$1"
  shift
  printf '%s\n' "#!/usr/bin/env bash" "set -euo pipefail" "$@" > "${script_path}"
  chmod +x "${script_path}"
}

start_tmux_session() {
  local session_name="$1"
  local script_path="$2"
  tmux has-session -t "${session_name}" 2>/dev/null && tmux kill-session -t "${session_name}" || true
  tmux new-session -d -s "${session_name}" "bash '${script_path}'"
}

COMMON_DATASETS_CSV="$(csv_join "${COMMON_DATASETS[@]}")"
DP_DATASETS_CSV="$(csv_join "${DP_DATASETS[@]}")"

BASE_ENV="source '${CONDA_SH}' && conda activate '${CONDA_ENV_NAME}' && cd '${PROJECT_ROOT}'"

TASK1_LOG="${LAUNCH_ROOT}/task1_warmup.tmux.log"
TASK2_LOG="${LAUNCH_ROOT}/task2_baselines.tmux.log"
TASK3_LOG="${LAUNCH_ROOT}/task3_frf_gpu56.tmux.log"
TASK4_LOG="${LAUNCH_ROOT}/task4_frf_dp_gpu7.tmux.log"

TASK1_SCRIPT="${LAUNCH_ROOT}/task1_warmup.sh"
TASK2_SCRIPT="${LAUNCH_ROOT}/task2_baselines.sh"
TASK3_SCRIPT="${LAUNCH_ROOT}/task3_frf_gpu56.sh"
TASK4_SCRIPT="${LAUNCH_ROOT}/task4_frf_dp_gpu7.sh"

write_session_script "${TASK1_SCRIPT}" \
  "source '${CONDA_SH}'" \
  "conda activate '${CONDA_ENV_NAME}'" \
  "cd '${PROJECT_ROOT}'" \
  "DATASETS_OVERRIDE='${COMMON_DATASETS_CSV}' MODEL_GPU_PAIRS_OVERRIDE='dinov2_vitb|1,clip_vitb|2,eva02_vitb|3,mocov3_resnet50|4' bash './launchers/run_feature_cache_warmup_multigpu.sh' >> '${TASK1_LOG}' 2>&1"

write_session_script "${TASK2_SCRIPT}" \
  "source '${CONDA_SH}'" \
  "conda activate '${CONDA_ENV_NAME}'" \
  "cd '${PROJECT_ROOT}'" \
  "CONDA_ENV_NAME='${CONDA_ENV_NAME}' ALL_DATASETS_OVERRIDE='${COMMON_DATASETS_CSV}' GPU_IDS='1,2,3,4' NUM_GPUS='4' SKIP_IF_DONE_ANYWHERE='0' bash './launchers/run_baselines_imagenet_multigpu.sh' >> '${TASK2_LOG}' 2>&1"

write_session_script "${TASK3_SCRIPT}" \
  "source '${CONDA_SH}'" \
  "conda activate '${CONDA_ENV_NAME}'" \
  "cd '${PROJECT_ROOT}'" \
  "sleep 300" \
  "DATASETS_OVERRIDE='${COMMON_DATASETS_CSV}' MODEL_GPU_PAIRS_OVERRIDE='mocov3_resnet50|5,dinov2_vitb|6,clip_vitb|5,eva02_vitb|6' bash './launchers/run_frf_all_datasets_ipc1_multigpu.sh' >> '${TASK3_LOG}' 2>&1"

write_session_script "${TASK4_SCRIPT}" \
  "source '${CONDA_SH}'" \
  "conda activate '${CONDA_ENV_NAME}'" \
  "cd '${PROJECT_ROOT}'" \
  "sleep 300" \
  "DATASETS_OVERRIDE='${DP_DATASETS_CSV}' MODEL_GPU_QUEUES_OVERRIDE='7|mocov3_resnet50,clip_vitb,dinov2_vitb,eva02_vitb' GPU_LAUNCH_STAGGER_SECONDS='20' bash './launchers/run_frf_all_datasets_ipc1_dp_multigpu.sh' >> '${TASK4_LOG}' 2>&1"

start_tmux_session "mar22_warmup" "${TASK1_SCRIPT}"
start_tmux_session "mar22_baselines" "${TASK2_SCRIPT}"
start_tmux_session "mar22_frf_gpu56" "${TASK3_SCRIPT}"
start_tmux_session "mar22_frf_dp_gpu7" "${TASK4_SCRIPT}"

printf '%s\n' "${LAUNCH_ROOT}" > "${PROJECT_ROOT}/latest_mar22_launch_root.txt"

echo "Launch root: ${LAUNCH_ROOT}"
echo "Task1 session: mar22_warmup"
echo "Task2 session: mar22_baselines"
echo "Task3 session: mar22_frf_gpu56"
echo "Task4 session: mar22_frf_dp_gpu7"

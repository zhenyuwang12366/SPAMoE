#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# shellcheck disable=SC1091
. "${REPO_ROOT}/scripts/bash_helpers.sh"

usage() {
  cat <<'EOF'
用法:
  bash run_from_scratch.sh --data_dir XXX --family FAMILY [选项]

说明:
  1. data_dir 必须是 OpenFWI 根目录，且其中必须存在 XXX/train_samples
  2. 脚本先调用 load_to_zarr.sh 转 zarr，再调用 school.sh 或 school_local.sh 启动训练

常用选项:
  --data_dir PATH              原始数据根目录，必须包含 train_samples/
  --family FAMILY              family 名称，例如 curve_vel_b / style_a
  --zarr_out PATH              输出 zarr 路径；默认 ./zarr_data/FAMILY.zarr
  --train_mode MODE            local / sbatch，默认 local
  --preset NAME                default / preset1 / preset2，默认 default
  --num_gpus N                 默认 2
  --num_workers N              默认 10
  --status_json PATH           默认 ./dataset_status/dataset_status.json
  --output_dir PATH            训练输出目录
  --seed N                     默认 0
  --include_test 0|1           仅单 family 有效，默认 0
  --remap_single_label 0|1     默认 0
  --chunks N                   zarr chunk 大小，默认 32
  --dtype TYPE                 float32 / float16，默认 float32
  --concat_channels 0|1        默认 1
  --conda_env NAME             可选，自动激活指定 conda 环境
  --                           后续参数原样透传给训练脚本
EOF
}

DATA_DIR=""
FAMILY=""
ZARR_OUT=""
TRAIN_MODE="local"
PRESET="default"
NUM_GPUS=2
NUM_WORKERS=10
STATUS_JSON="${REPO_ROOT}/dataset_status/dataset_status.json"
OUTPUT_DIR=""
SEED=0
INCLUDE_TEST=0
REMAP_SINGLE_LABEL=0
CHUNKS=32
DTYPE="float32"
CONCAT_CHANNELS=1
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
TRAIN_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --family) FAMILY="$2"; shift 2 ;;
    --zarr_out) ZARR_OUT="$2"; shift 2 ;;
    --train_mode) TRAIN_MODE="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --status_json) STATUS_JSON="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --include_test) INCLUDE_TEST="$2"; shift 2 ;;
    --remap_single_label) REMAP_SINGLE_LABEL="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --concat_channels) CONCAT_CHANNELS="$2"; shift 2 ;;
    --conda_env) CONDA_ENV_NAME="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --)
      shift
      TRAIN_EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${DATA_DIR}" || -z "${FAMILY}" ]]; then
  echo "必须提供 --data_dir 和 --family。" >&2
  usage
  exit 1
fi

if [[ ! -d "${DATA_DIR}/train_samples" ]]; then
  echo "数据目录必须包含 train_samples/: ${DATA_DIR}/train_samples" >&2
  exit 1
fi

FAMILY="$(normalize_family_name "${FAMILY}")"

if [[ -z "${ZARR_OUT}" ]]; then
  ZARR_OUT="${REPO_ROOT}/zarr_data/${FAMILY}.zarr"
fi

mkdir -p "$(dirname "${ZARR_OUT}")"

LOAD_CMD=(
  bash "${REPO_ROOT}/load_to_zarr.sh"
  --data_dir "${DATA_DIR}"
  --zarr_out "${ZARR_OUT}"
  --family "${FAMILY}"
  --include_test "${INCLUDE_TEST}"
  --remap_single_label "${REMAP_SINGLE_LABEL}"
  --chunks "${CHUNKS}"
  --dtype "${DTYPE}"
  --seed "${SEED}"
  --concat_channels "${CONCAT_CHANNELS}"
)

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  LOAD_CMD+=(--conda_env "${CONDA_ENV_NAME}")
fi

printf '步骤1/2，转换 zarr:'
printf ' %q' "${LOAD_CMD[@]}"
printf '\n'
"${LOAD_CMD[@]}"

TRAIN_CMD=(
  --family "${FAMILY}"
  --zarr_path "${ZARR_OUT}"
  --preset "${PRESET}"
  --num_gpus "${NUM_GPUS}"
  --num_workers "${NUM_WORKERS}"
  --status_json "${STATUS_JSON}"
  --seed "${SEED}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  TRAIN_CMD+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ -n "${CONDA_ENV_NAME}" ]]; then
  TRAIN_CMD+=(--conda_env "${CONDA_ENV_NAME}")
fi
if [[ ${#TRAIN_EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=(-- "${TRAIN_EXTRA_ARGS[@]}")
fi

case "${TRAIN_MODE}" in
  local)
    printf '步骤2/2，本地训练:'
    printf ' %q' bash "${REPO_ROOT}/school_local.sh" "${TRAIN_CMD[@]}"
    printf '\n'
    bash "${REPO_ROOT}/school_local.sh" "${TRAIN_CMD[@]}"
    ;;
  sbatch)
    if ! command -v sbatch >/dev/null 2>&1; then
      echo "当前环境没有 sbatch，不能使用 --train_mode sbatch。" >&2
      exit 1
    fi
    printf '步骤2/2，提交 sbatch:'
    printf ' %q' sbatch "${REPO_ROOT}/school.sh" "${TRAIN_CMD[@]}"
    printf '\n'
    sbatch "${REPO_ROOT}/school.sh" "${TRAIN_CMD[@]}"
    ;;
  *)
    echo "不支持的 --train_mode: ${TRAIN_MODE}，可选 local / sbatch" >&2
    exit 1
    ;;
esac

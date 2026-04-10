#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/scripts/bash_helpers.sh"

usage() {
  cat <<'EOF'
用法:
  bash load_to_zarr.sh --data_dir /path/to/XXX --zarr_out /path/to/out.zarr [选项]

要求:
  data_dir 必须是 OpenFWI 根目录，并且其中必须存在 train_samples/

选项:
  --data_dir PATH              原始数据根目录，必须包含 train_samples/
  --zarr_out PATH              输出 zarr 目录
  --family FAMILY              默认 all；也支持 curve_vel_a / style_a 等单 family
  --include_test 0|1           默认 0
  --remap_single_label 0|1     默认 0
  --chunks N                   默认 32
  --dtype TYPE                 float32 / float16，默认 float32
  --seed N                     默认 42
  --concat_channels 0|1        默认 1
  --conda_env NAME             可选，自动激活指定 conda 环境
EOF
}

DATA_DIR=""
ZARR_OUT=""
FAMILY="all"
INCLUDE_TEST=0
REMAP_SINGLE_LABEL=0
CHUNKS=32
DTYPE="float32"
SEED=42
CONCAT_CHANNELS=1
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --zarr_out) ZARR_OUT="$2"; shift 2 ;;
    --family) FAMILY="$2"; shift 2 ;;
    --include_test) INCLUDE_TEST="$2"; shift 2 ;;
    --remap_single_label) REMAP_SINGLE_LABEL="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --concat_channels) CONCAT_CHANNELS="$2"; shift 2 ;;
    --conda_env) CONDA_ENV_NAME="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${DATA_DIR}" || -z "${ZARR_OUT}" ]]; then
  echo "必须提供 --data_dir 和 --zarr_out。" >&2
  usage
  exit 1
fi

if [[ ! -d "${DATA_DIR}/train_samples" ]]; then
  echo "数据目录必须包含 train_samples/: ${DATA_DIR}/train_samples" >&2
  exit 1
fi

FAMILY="$(normalize_family_name "${FAMILY}")"
mkdir -p "$(dirname "${ZARR_OUT}")"

activate_conda_env "${CONDA_ENV_NAME}"

CMD=(
  python "${SCRIPT_DIR}/load_to_zarr.py"
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

printf '执行命令:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/scripts/bash_helpers.sh"

usage() {
  cat <<'EOF'
Usage:
  bash load_to_zarr.sh --data_dir /path/to/XXX --zarr_out /path/to/out.zarr [options]

Requirements:
  data_dir must be an OpenFWI-style root containing train_samples/

Options:
  --data_dir PATH              raw data root (must contain train_samples/)
  --zarr_out PATH              output Zarr directory
  --family FAMILY              default all; or a single family e.g. curve_vel_a / style_a
  --include_test 0|1           default 0
  --remap_single_label 0|1     default 0
  --chunks N                   default 32
  --dtype TYPE                 float32 / float16, default float32
  --seed N                     default 42
  --concat_channels 0|1        default 1
  --conda_env NAME             optional: activate this conda env
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
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${DATA_DIR}" || -z "${ZARR_OUT}" ]]; then
  echo "Both --data_dir and --zarr_out are required." >&2
  usage
  exit 1
fi

if [[ ! -d "${DATA_DIR}/train_samples" ]]; then
  echo "Data root must contain train_samples/: ${DATA_DIR}/train_samples" >&2
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

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

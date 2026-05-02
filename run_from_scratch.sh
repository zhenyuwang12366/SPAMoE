#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# shellcheck disable=SC1091
. "${REPO_ROOT}/scripts/bash_helpers.sh"

usage() {
  cat <<'EOF'
Usage:
  bash run_from_scratch.sh --data_dir XXX --family FAMILY [options]

Steps:
  1. data_dir must be an OpenFWI root containing <data_dir>/train_samples
  2. Runs load_to_zarr.sh, then school.sh or school_local.sh for training

Common options:
  --data_dir PATH              raw data root (must contain train_samples/)
  --family FAMILY              e.g. curve_vel_b / style_a
  --zarr_out PATH              output Zarr path; default ./zarr_data/FAMILY.zarr
  --train_mode MODE            local / sbatch, default local
  --preset NAME                default / preset1 / preset2, default default
  --num_gpus N                 default 2
  --num_workers N              default 10
  --status_json PATH           default ./dataset_status/dataset_status.json
  --output_dir PATH            training output directory
  --seed N                     default 0
  --include_test 0|1           single-family only, default 0
  --remap_single_label 0|1     default 0
  --chunks N                   Zarr chunk size, default 32
  --dtype TYPE                 float32 / float16, default float32
  --concat_channels 0|1        default 1
  --conda_env NAME             optional conda env
  --                           extra args forwarded to the training launcher
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
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${DATA_DIR}" || -z "${FAMILY}" ]]; then
  echo "Both --data_dir and --family are required." >&2
  usage
  exit 1
fi

if [[ ! -d "${DATA_DIR}/train_samples" ]]; then
  echo "Data root must contain train_samples/: ${DATA_DIR}/train_samples" >&2
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

printf 'Step 1/2 (Zarr):'
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
    printf 'Step 2/2 (local train):'
    printf ' %q' bash "${REPO_ROOT}/school_local.sh" "${TRAIN_CMD[@]}"
    printf '\n'
    bash "${REPO_ROOT}/school_local.sh" "${TRAIN_CMD[@]}"
    ;;
  sbatch)
    if ! command -v sbatch >/dev/null 2>&1; then
      echo "sbatch not found; cannot use --train_mode sbatch." >&2
      exit 1
    fi
    printf 'Step 2/2 (sbatch):'
    printf ' %q' sbatch "${REPO_ROOT}/school.sh" "${TRAIN_CMD[@]}"
    printf '\n'
    sbatch "${REPO_ROOT}/school.sh" "${TRAIN_CMD[@]}"
    ;;
  *)
    echo "Unsupported --train_mode: ${TRAIN_MODE} (use local or sbatch)" >&2
    exit 1
    ;;
esac

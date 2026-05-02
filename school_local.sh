#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# shellcheck disable=SC1091
. "${REPO_ROOT}/scripts/bash_helpers.sh"
# shellcheck disable=SC1091
. "${REPO_ROOT}/training_presets.sh"

usage() {
  cat <<'EOF'
Usage:
  bash school_local.sh --family FAMILY --zarr_path /path/to/data.zarr [options]
  bash school_local.sh --config train.json
  bash school_local.sh --args args.json

Common options:
  --family FAMILY              e.g. curve_vel_b / style_a
  --zarr_path PATH             Zarr dataset path
  --preset NAME                default / preset1 / preset2
  --num_gpus N                 default 2
  --num_workers N              default 10
  --status_json PATH           default ./dataset_status/dataset_status.json
  --output_dir PATH            training output directory
  --seed N                     default 0
  --conda_env NAME             optional: activate this conda env
  --                           remaining args forwarded to scripts/run_distributed_seismic_moe.sh

Examples:
  bash school_local.sh \
    --family curve_vel_b \
    --zarr_path /data/curve_vel_b.zarr \
    --preset preset1

  bash school_local.sh \
    --family style_a \
    --zarr_path /data/style_a.zarr \
    --preset preset2 \
    -- --eval_interval 2 --early_stop --early_stop_patience 20
EOF
}

FAMILY=""
ZARR_PATH=""
PRESET="default"
NUM_GPUS=2
NUM_WORKERS=10
STATUS_JSON="${REPO_ROOT}/dataset_status/dataset_status.json"
OUTPUT_DIR=""
SEED=0
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
CONFIG_PATH=""
ARGS_JSON=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --family) FAMILY="$2"; shift 2 ;;
    --zarr_path) ZARR_PATH="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --status_json) STATUS_JSON="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --conda_env) CONDA_ENV_NAME="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --args) ARGS_JSON="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

activate_conda_env "${CONDA_ENV_NAME}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${CONFIG_PATH}" && -n "${ARGS_JSON}" ]]; then
  echo "--config and --args cannot be used together." >&2
  exit 1
fi

if [[ -n "${CONFIG_PATH}" || -n "${ARGS_JSON}" ]]; then
  if [[ "${PRESET}" != "default" || -n "${FAMILY}" || -n "${ZARR_PATH}" || ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "With --config/--args, put all training settings in JSON; do not mix preset/family/zarr/extra args." >&2
    exit 1
  fi

  CMD=(bash "${REPO_ROOT}/scripts/run_distributed_seismic_moe.sh" --num_gpus "${NUM_GPUS}")
  if [[ -n "${CONFIG_PATH}" ]]; then
    CMD+=(--config "${CONFIG_PATH}")
  else
    CMD+=(--args "${ARGS_JSON}")
  fi
else
  if [[ -z "${FAMILY}" || -z "${ZARR_PATH}" ]]; then
    echo "Without --config/--args, both --family and --zarr_path are required." >&2
    usage
    exit 1
  fi

  FAMILY="$(normalize_family_name "${FAMILY}")"
  if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${REPO_ROOT}/exp/runs/${FAMILY}_${PRESET}_s${SEED}"
  fi

  set_training_preset_args "${FAMILY}" "${PRESET}"

  CMD=(
    bash "${REPO_ROOT}/scripts/run_distributed_seismic_moe.sh"
    --mode train
    --num_gpus "${NUM_GPUS}"
    --num_workers "${NUM_WORKERS}"
    --zarr_path "${ZARR_PATH}"
    --status_json "${STATUS_JSON}"
    --family "${FAMILY}"
    --seed "${SEED}"
    --output_dir "${OUTPUT_DIR}"
  )
  if [[ ${#PRESET_ARGS[@]} -gt 0 ]]; then
    CMD+=("${PRESET_ARGS[@]}")
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi
fi

echo "Python: $(command -v python)"
python -c "import torch; print('PyTorch:', torch.__version__)" || true
printf 'Launch:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

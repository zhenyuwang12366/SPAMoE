#!/usr/bin/env bash
#SBATCH --job-name=FWINO_optuna
#SBATCH --partition=gpu-4090-2
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/data1/home/teacher/teacher_s/t108790/results/logs/%x-%j.out
#SBATCH --error=/data1/home/teacher/teacher_s/t108790/results/logs/%x-%j.err
#SBATCH --no-requeue

set -Eeuo pipefail

cleanup() {
  if [[ -n "${MAIN_PGID:-}" ]]; then
    kill -TERM -"${MAIN_PGID}" 2>/dev/null || true
    sleep 2
    kill -KILL -"${MAIN_PGID}" 2>/dev/null || true
  fi
}
trap cleanup SIGINT SIGTERM EXIT

# ===================== Configuration =====================
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

FAMILY=${FAMILY:-flat_vel}
WAVETYPE=${WAVETYPE:-db8}

JOB_NAME=${SLURM_JOB_NAME:-${JOB_NAME:-FWINO_optuna}}
JOB_ID=${SLURM_JOB_ID:-${JOB_ID:-$(date +%s)}}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK:-10}}

RESULTS_ROOT=${RESULTS_ROOT:-/data1/home/teacher/teacher_s/t108790/results}
LOG_DIR=${LOG_DIR:-"${RESULTS_ROOT}/logs"}
OPTUNA_DIR=${OPTUNA_DIR:-"${RESULTS_ROOT}/optuna"}

DEFAULT_WORKDIR=/data1/home/teacher/teacher_s/t108790/FWINO_wzy_test
if [[ ! -d "${DEFAULT_WORKDIR}" ]]; then
  DEFAULT_WORKDIR="${SCRIPT_DIR}"
fi
WORKDIR=${WORKDIR:-"${DEFAULT_WORKDIR}"}

CONDA_BASE=${CONDA_BASE:-/data1/apps/anaconda3}
CONDA_ENV=${CONDA_ENV:-FWINO}

CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NUM_GPUS=${NUM_GPUS:-2}
NUM_WORKERS=${NUM_WORKERS:-10}
TOP_K=${TOP_K:-1}
CHOOSE_EXPERTS=${CHOOSE_EXPERTS:-1}
IS_SPECIFIC=${IS_SPECIFIC:-1}
N_TRIALS=${N_TRIALS:-30}
DATA_DIR=${DATA_DIR:-/data1/home/teacher/teacher_s/t108790/FWINO/FWINO_data}
BASH_LAUNCHER=${BASH_LAUNCHER:-scripts/run_distributed_seismic_moe.sh}
PYTHON_ENTRY=${PYTHON_ENTRY:-scripts/tuna_seismic_moe.py}

LOG_STDOUT=${LOG_STDOUT:-"${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out"}
LOG_STDERR=${LOG_STDERR:-"${LOG_DIR}/${JOB_NAME}-${JOB_ID}.err"}

mkdir -p "${LOG_DIR}" "${OPTUNA_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec > >(tee -a "${LOG_STDOUT}")
  exec 2> >(tee -a "${LOG_STDERR}" >&2)
fi

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  . "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "[WARN] Conda base not found at ${CONDA_BASE}; skipping activation." >&2
fi

cd "${WORKDIR}" || {
  echo "[ERROR] Failed to change directory to ${WORKDIR}" >&2
  exit 1
}

# ===================== Environment =====================

echo "[INFO] Host: $(hostname)"
echo "[INFO] PWD : $(pwd)"
echo "[INFO] JOB_NAME: ${JOB_NAME}"
echo "[INFO] JOB_ID : ${JOB_ID}"

echo "Python: $(which python)"
python - <<'PY'
import torch, os
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.version.cuda)
print("GPU cnt:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
PY

echo "Family: ${FAMILY}"
echo "WaveletType: ${WAVETYPE}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${CPUS_PER_TASK}"
export MKL_NUM_THREADS="${CPUS_PER_TASK}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS

# ===================== Optuna Launcher =====================
STORAGE_PATH="${OPTUNA_DIR}/wno_${FAMILY}_${WAVETYPE}.db"
STORAGE_URI="sqlite:////${STORAGE_PATH#/}"

CMD=(stdbuf -oL -eL python "${PYTHON_ENTRY}"
  --n_trials "${N_TRIALS}"
  --study_name "wno_${FAMILY}_${WAVETYPE}"
  --storage "${STORAGE_URI}"
  --bash_launcher "${BASH_LAUNCHER}"
  --num_gpus "${NUM_GPUS}"
  --cuda_visible_devices "${CUDA_DEVICES}"
  --data_dir "${DATA_DIR}"
  --family "${FAMILY}"
  --output_dir "${OPTUNA_DIR}/optuna_seismic_moe_${JOB_NAME}_${JOB_ID}"
  --num_workers "${NUM_WORKERS}"
  --top_k "${TOP_K}"
  --choose_experts "${CHOOSE_EXPERTS}"
  --wavelet_type "${WAVETYPE}"
)

if [[ "${IS_SPECIFIC}" -eq 1 ]]; then
  CMD+=(--is_specific)
fi

echo "[INFO] Launch: ${CMD[*]}"

"${CMD[@]}" &
MAIN_PID=$!
MAIN_PGID=$(ps -o pgid= "${MAIN_PID}" | tr -d ' ')
export MAIN_PGID

wait "${MAIN_PID}"
echo "Optuna tuning workflow finished."

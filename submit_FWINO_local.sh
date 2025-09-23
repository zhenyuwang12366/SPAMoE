#!/usr/bin/env bash
#SBATCH --job-name=FWINO_syx              # 作业名称
#SBATCH --partition=gpu-4090-2            # 分区名称（请根据实际情况调整）
#SBATCH --gres=gpu:2                      # 请求GPU资源
#SBATCH --ntasks=1                        # 启动1个任务（torchrun会管理GPU）
#SBATCH --cpus-per-task=10                # 分配CPU
#SBATCH --time=24:00:00                   # 最长运行时间
#SBATCH --output=../results/output%j.txt  # 输出日志
#SBATCH --no-requeue

# Portable launcher: works with or without SLURM.
set -Eeuo pipefail

cleanup() {
  if [[ -n "${MAIN_PGID:-}" ]]; then
    kill -TERM -"${MAIN_PGID}" 2>/dev/null || true
    sleep 2
    kill -KILL -"${MAIN_PGID}" 2>/dev/null || true
  fi
}
trap cleanup SIGINT SIGTERM EXIT

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

JOB_NAME=${SLURM_JOB_NAME:-${JOB_NAME:-FWINO_syx}}
JOB_ID=${SLURM_JOB_ID:-${JOB_ID:-$(date +%s)}}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK:-10}}

DEFAULT_WORKDIR=/data1/home/teacher/teacher_s/t108790/FWINO_wzy_test
if [[ ! -d "${DEFAULT_WORKDIR}" ]]; then
  DEFAULT_WORKDIR=${SCRIPT_DIR}
fi
WORKDIR=${WORKDIR:-"${DEFAULT_WORKDIR}"}

DEFAULT_RESULTS_ROOT=/data1/home/teacher/teacher_s/t108790/results
if [[ ! -d "${DEFAULT_RESULTS_ROOT}" ]]; then
  DEFAULT_RESULTS_ROOT="${SCRIPT_DIR}/../results"
fi
RESULTS_ROOT=${RESULTS_ROOT:-"${DEFAULT_RESULTS_ROOT}"}
LOG_DIR=${LOG_DIR:-"${RESULTS_ROOT}/logs"}

LOG_STDOUT=${LOG_STDOUT:-"${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out"}
LOG_STDERR=${LOG_STDERR:-"${LOG_DIR}/${JOB_NAME}-${JOB_ID}.err"}

CONDA_BASE=${CONDA_BASE:-/data1/apps/anaconda3}
CONDA_ENV=${CONDA_ENV:-FWINO}

CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NUM_GPUS=${NUM_GPUS:-2}
NUM_WORKERS=${NUM_WORKERS:-10}
FAMILY=${FAMILY:-vel}
BATCH_SIZE=${BATCH_SIZE:-4}
EPOCHS=${EPOCHS:-500}
DATA_DIR=${DATA_DIR:-/data1/home/teacher/teacher_s/t108790/FWINO/FWINO_data}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-seismic_moe}
OUTPUT_DIR=${OUTPUT_DIR:-"${RESULTS_ROOT}/${OUTPUT_SUBDIR}_${JOB_NAME}_${JOB_ID}"}
TOP_K=${TOP_K:-1}
CHOOSE_EXPERTS=${CHOOSE_EXPERTS:-1}
WNO_N_LEVELS_HEIGHT=${WNO_N_LEVELS_HEIGHT:-2}
WNO_N_LEVELS_WIDTH=${WNO_N_LEVELS_WIDTH:-2}
HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-64}
WNO_N_LAYERS=${WNO_N_LAYERS:-6}
WNO_BLOCK_N_LAYERS=${WNO_BLOCK_N_LAYERS:-3}
WNO_DROPOUT_RATE=${WNO_DROPOUT_RATE:-0.15}
WAVELET_TYPE=${WAVELET_TYPE:-db4}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
SCHEDULER_GAMMA=${SCHEDULER_GAMMA:-0.2}
ACCUM_STEPS=${ACCUM_STEPS:-1}
BASH_LAUNCHER=${BASH_LAUNCHER:-scripts/run_distributed_seismic_moe.sh}

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

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

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${CPUS_PER_TASK}"
export MKL_NUM_THREADS="${CPUS_PER_TASK}"

echo "[INFO] Host: $(hostname)"
echo "[INFO] PWD : $(pwd)"
echo "[INFO] JOB_NAME: ${JOB_NAME}"
echo "[INFO] JOB_ID : ${JOB_ID}"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES}}"

echo "Python: $(which python)"
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.version.cuda)
print("GPU cnt:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
PY

echo "Family: ${FAMILY}"
echo "WaveletType: ${WAVELET_TYPE}"
echo "OutputDir: ${OUTPUT_DIR}"

declare -a CMD=(
  bash "${BASH_LAUNCHER}"
  --num_gpus "${NUM_GPUS}"
  --num_workers "${NUM_WORKERS}"
  --data_dir "${DATA_DIR}"
  --family "${FAMILY}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --output_dir "${OUTPUT_DIR}"
  --top_k "${TOP_K}"
  --choose_experts "${CHOOSE_EXPERTS}"
  --WNO_n_levels_height "${WNO_N_LEVELS_HEIGHT}"
  --WNO_n_levels_width "${WNO_N_LEVELS_WIDTH}"
  --hidden_channels "${HIDDEN_CHANNELS}"
  --WNO_n_layers "${WNO_N_LAYERS}"
  --WNO_block_n_layers "${WNO_BLOCK_N_LAYERS}"
  --WNO_dropout_rate "${WNO_DROPOUT_RATE}"
  --wavelet_type "${WAVELET_TYPE}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --scheduler_gamma "${SCHEDULER_GAMMA}"
  --accum_steps "${ACCUM_STEPS}"
)

echo "[INFO] Launch: ${CMD[*]}"

"${CMD[@]}" &
MAIN_PID=$!
MAIN_PGID=$(ps -o pgid= "${MAIN_PID}" | tr -d ' ')
export MAIN_PGID

wait "${MAIN_PID}"
echo "FWINO training finished."

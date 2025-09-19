#!/usr/bin/env bash
#SBATCH --job-name=FWINO_optuna
#SBATCH --partition=gpu-4090-2
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/data1/home/teacher/teacher_s/t108790/results/logs/%x-%j.out
#SBATCH --error=/data1/home/teacher/teacher_s/t108790/results/logs/%x-%j.err
#SBATCH --no-requeue

# ========= 安全开关 & 清理 =========
set -Eeuo pipefail
cleanup() {
  # 尝试整组清理（避免残留 torchrun 子进程占GPU）
  if [[ -n "${MAIN_PGID:-}" ]]; then
    kill -TERM -"${MAIN_PGID}" 2>/dev/null || true
    sleep 2
    kill -KILL -"${MAIN_PGID}" 2>/dev/null || true
  fi
}
trap cleanup SIGINT SIGTERM EXIT

# ========= 路径与环境 =========
cd /data1/home/teacher/teacher_s/t108790/FWINO_wzy_test

# 确保日志/结果目录存在（用绝对路径，避免相对路径权限/基准目录不一致）
mkdir -p /data1/home/teacher/teacher_s/t108790/results/logs
mkdir -p /data1/home/teacher/teacher_s/t108790/results/optuna

# Conda
. "/data1/apps/anaconda3/etc/profile.d/conda.sh"
conda activate FWINO

echo "[INFO] Host: $(hostname)"
echo "[INFO] PWD : $(pwd)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "[INFO] SLURM_JOB_NAME: ${SLURM_JOB_NAME}"

echo "Python: $(which python)"
python - <<'PY'
import torch, os
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.version.cuda)
print("GPU cnt:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
PY

# ========= GPU/NCCL/线程 =========
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1            # Python无缓冲，便于父进程流式读取
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
export NCCL_IB_DISABLE=1             # 无IB时禁用，常能避免慢/挂
export NCCL_P2P_LEVEL=SYS
# 如需排障可打开：
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ========= Optuna 外层（单进程） =========
# 使用 stdbuf -oL -eL 强制行缓冲；父进程能实时捕捉 REPORT/VAL_LOSS
CMD=(stdbuf -oL -eL python scripts/tuna_seismic_moe.py
  --n_trials 30
  --study_name wno_flatvel
  --storage sqlite:////data1/home/teacher/teacher_s/t108790/results/optuna/wno_flatvel.db
  --bash_launcher scripts/run_distributed_seismic_moe.sh
  --num_gpus 2
  --cuda_visible_devices 0,1
  --data_dir /data1/home/teacher/teacher_s/t108790/FWINO/FWINO_data
  --family flat_vel
  --output_dir /data1/home/teacher/teacher_s/t108790/results/optuna/optuna_seismic_moe_${SLURM_JOB_NAME}_${SLURM_JOB_ID}
  --num_workers 10
  --top_k 1
  --choose_experts 1
  --is_specific
)

echo "[INFO] Launch: ${CMD[*]}"

# 可选：在某些集群用 srun 更稳（资源/环境传递更明确）
# srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --gres=gpu:2 --unbuffered "${CMD[@]}"

# 不用 srun 也可以，保持为单进程：
"${CMD[@]}" &
MAIN_PID=$!
MAIN_PGID=$(ps -o pgid= "$MAIN_PID" | tr -d ' ')
export MAIN_PGID

wait "$MAIN_PID"
echo "Optuna 调参流程结束。"
#!/bin/bash
#SBATCH --job-name=FWINO_wzy              # 作业名称
#SBATCH --partition=gpu-4090-2             # 分区名称（请根据实际情况调整）
#SBATCH --gres=gpu:2                       # 请求4个GPU
#SBATCH --ntasks=1                         # 启动1个任务（torchrun会管理GPU）
#SBATCH --cpus-per-task=10                  # 分配CPU
#SBATCH --output=../results/output%j.txt             # 输出日志
#SBATCH --no-requeue

cd /data1/home/teacher/teacher_s/t108790/FWINO_wzy || exit

# === 激活 Conda 虚拟环境 ===
. "/data1/apps/anaconda3/etc/profile.d/conda.sh"
conda activate FWINO

# === 打印验证信息（可选）===
echo "当前 Python: $(which python)"
python -c "import torch; print('PyTorch 版本:', torch.__version__)"

# === 启动训练 ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python exp/run_afreqmoe_pipeline.py \
  --num-gpus 2 \
  --families curve_vel_b \
  --seeds 0 \
  --seis-zarr /data1/home/teacher/teacher_s/t108790 \
  --seis-status-json ./dataset_status/dataset_status.json \
  --save-root ./exp/runs \
  --only e11_only_fno_curve_vel_b_s0
#!/bin/bash
#SBATCH --job-name=FWINO_wzy_inference     # 作业名称
#SBATCH --partition=gpu-4090-2             # 分区名称（请根据实际情况调整）
#SBATCH --gres=gpu:1                       # 请求4个GPU
#SBATCH --ntasks=1                         # 启动1个任务（torchrun会管理GPU）
#SBATCH --cpus-per-task=10                  # 分配CPU
#SBATCH --output=../results/output%j.txt             # 输出日志
#SBATCH --no-requeue

# === 进入工作目录（如果未用 sbatch --chdir） ===
cd /data1/home/teacher/teacher_s/t108790/FWINO_wzy || exit

# === 激活 Conda 虚拟环境 ===
. "/data1/apps/anaconda3/etc/profile.d/conda.sh"
conda activate FWINO

# === 打印验证信息（可选）===
echo "当前 Python: $(which python)"
python -c "import torch; print('PyTorch 版本:', torch.__version__)"

# === 启动训练 ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train_seismic_moe.py \
  --mode inference \
  --setting_path /data1/home/teacher/teacher_s/t108790/results/fourier_0/seismic_moe_curve_vel_b/FNO_router-basic_lr0.000267116_bs32_fourier_0_20251027-073251

echo "推理已完成"
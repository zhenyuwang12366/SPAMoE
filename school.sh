#!/bin/bash
#SBATCH --job-name=FWINO_wzy              # 作业名称
#SBATCH --partition=gpu-4090-2             # 分区名称（请根据实际情况调整）
#SBATCH --gres=gpu:2                       # 请求4个GPU
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
bash scripts/run_distributed_seismic_moe.sh \
  --mode train \
  --model_name afreqmoe \
  --num_gpus 2 \
  --num_workers 10 \
  --zarr_path /data1/home/teacher/teacher_s/t108790/curve_vel_b.zarr \
  --status_json ./dataset_status/dataset_status.json \
  --family curve_vel_b \
  --batch_size 32 \
  --epochs 150 \
  --output_dir ../results_seismic \
  --top_k 2 \
  --use_amp \
  --band_sharpness 10
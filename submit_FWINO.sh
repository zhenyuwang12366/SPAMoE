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
  --model_name WNO \
  --use_encoder \
  --concat_channels \
  --is_specific \
  --num_gpus 2 \
  --num_workers 10 \
  --data_dir /data1/home/teacher/teacher_s/t108790/DATAA \
  --family curve_vel_b \
  --is_specific \
  --batch_size 32 \
  --epochs 100 \
  --output_dir ../results \
  --top_k 1 \
  --choose_experts 1 \
  --hidden_channels 96 \
  --learning_rate 0.00026711555047527854 \
  --weight_decay 0.08952068376871994 \
  --scheduler_gamma 0.2966237496749535 \
  --accum_steps 1 \
  --FNO_n_layers 4 \
  --lambda_g1v 0.43947650935102966 \
  --lambda_g2v 0.35339805101397564 \
  --lambda_grad_l1 0.15 \
  --lambda_fourier_mag_l1 0.05 \
  --wavelet_type db6 \
  --WNO_n_levels_height 3 \
  --WNO_n_levels_width 2 \
  --WNO_n_layers 4 \
  --WNO_dropout_rate 0.10 \
  --use_amp
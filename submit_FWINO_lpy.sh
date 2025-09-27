#!/bin/bash
#SBATCH --job-name=FWINO_wzy              # 作业名称
#SBATCH --partition=gpu-4090-2             # 分区名称（请根据实际情况调整）
#SBATCH --gres=gpu:2                       # 请求4个GPU
#SBATCH --ntasks=1                         # 启动1个任务（torchrun会管理GPU）
#SBATCH --cpus-per-task=10                  # 分配CPU
#SBATCH --time=24:00:00                    # 最长运行时间
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
  --num_gpus 2 \
  --num_workers 10 \
  --data_dir /data1/home/teacher/teacher_s/t108790/FWINO/FWINO_data \
  --family flat_vel \
  --is_specific \
  --batch_size 4 \
  --epochs 500 \
  --output_dir ../results/seismic_moe_${SLURM_JOB_NAME}_${SLURM_JOB_ID} \
  --top_k 1 \
  --choose_experts 1 \
  --WNO_n_levels_height 2 \
  --WNO_n_levels_width 2 \
  --hidden_channels 64 \
  --WNO_n_layers 6 \
  --WNO_block_n_layers 3  \
  --WNO_dropout_rate 0.15 \
  --wavelet_type db8\
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --scheduler_gamma 0.2 \
  --accum_steps 1
#!/bin/bash
#SBATCH --job-name=FWINO_overfit1        # 作业名称
#SBATCH --partition=gpu-4090-2           # 分区
#SBATCH --gres=gpu:1                     # 只需单卡
#SBATCH --ntasks=1                       # 单任务
#SBATCH --cpus-per-task=6                # CPU 数可适当减小
#SBATCH --time=4:00:00                   # 过拟合测试时间不需要太长
#SBATCH --output=../results/overfit1_%j.txt
#SBATCH --no-requeue

# === 进入工作目录 ===
cd /data1/home/teacher/teacher_s/t108790/FWINO_wzy_test

# === 激活 Conda 环境 ===
. "/data1/apps/anaconda3/etc/profile.d/conda.sh"
conda activate FWINO

echo "当前 Python: $(which python)"
python -c "import torch; print('PyTorch 版本:', torch.__version__)"

# === 单样本过拟合测试 ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_seismic_moe.py \
  --mode overfit1 \
  --data_dir /data1/home/teacher/teacher_s/t108790/FWINO/FWINO_data \
  --family flat_vel \
  --is_specific\
  --batch_size 4 \
  --epochs 1000 \
  --accum_steps 2 \
  --output_dir /data1/home/teacher/teacher_s/t108790/results/overfit1_${SLURM_JOB_NAME}_${SLURM_JOB_ID} \
  --learning_rate 0.00010333997431453093 \
  --weight_decay 0.0862093097084247 \
  --scheduler_gamma 0.4676763741328047 \
  --hidden_channels 128 \
  --FNO_n_layers 6 \
  --WNO_n_layers 6 \
  --MNO_n_layers 3 \
  --LNO_n_layers 3 \
  --WNO_block_n_layers 2 \
  --WNO_dropout_rate 0.12328124398155824 \
  --WNO_n_levels_height 2 \
  --WNO_n_levels_width 3 \
  --lambda_g1v 0.30848714348186856 \
  --lambda_g2v 0.36656343480393866 \
  --top_k 1 \
  --choose_experts 1 \
  --wavelet_type db4 \
  --vis_freq 5

python -m pdb \
  scripts/train_seismic_moe.py \
  --mode overfit1 \
  --data_dir /data1/home/teacher/teacher_s/t108790/FWINO/FWINO_data \
  --family flat_vel \
  --is_specific \
  --batch_size 4 \
  --epochs 1000 \
  --accum_steps 2 \
  --output_dir /data1/home/teacher/teacher_s/t108790/results/overfit1_${SLURM_JOB_NAME}_${SLURM_JOB_ID} \
  --learning_rate 0.00010333997431453093 \
  --weight_decay 0.0862093097084247 \
  --scheduler_gamma 0.4676763741328047 \
  --hidden_channels 128 \
  --FNO_n_layers 6 \
  --WNO_n_layers 6 \
  --MNO_n_layers 3 \
  --LNO_n_layers 3 \
  --WNO_block_n_layers 2 \
  --WNO_dropout_rate 0.12328124398155824 \
  --WNO_n_levels_height 2 \
  --WNO_n_levels_width 3 \
  --lambda_g1v 0.30848714348186856 \
  --lambda_g2v 0.36656343480393866 \
  --top_k 1 \
  --choose_experts 1 \
  --wavelet_type db4 \
  --vis_freq 5

  
#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/seismic_moe_${TIMESTAMP}.log

# === 进入工作目录（如果未用 sbatch --chdir） ===
#cd /data1/home/teacher/teacher_s/t108790/FWINO_wzy || exit

# === 激活 Conda 虚拟环境 ===
. "/root/miniconda3/etc/profile.d/conda.sh"
conda activate seismic_moe

# === 打印验证信息（可选）===
echo "当前 Python: $(which python)"
python -c "import torch; print('PyTorch 版本:', torch.__version__)"

# === 启动训练 ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup bash scripts/run_distributed_seismic_moe.sh \
  --mode train \
  --model_name WNO \
  --use_encoder \
  --concat_channels \
  --is_specific \
  --num_gpus 2 \
  --num_workers 32 \
  --data_dir /root/autodl-tmp/FWINO/FWINO_data \
  --family curve_vel_a \
  --is_specific \
  --batch_size 32 \
  --epochs 180 \
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
  --WNO_n_levels_height 1 \
  --WNO_n_levels_width 1 \
  --WNO_n_layers 3 \
  --WNO_dropout_rate 0.10 \
  --wavelet_type db4 \
  > "$LOGFILE" 2>&1 &
echo "ddp训练已启动，日志记录在：$LOGFILE"
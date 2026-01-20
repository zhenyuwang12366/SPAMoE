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
nohup python scripts/train_seismic_moe.py \
  --mode inference \
  --setting_path /data1/home/teacher/teacher_s/t108790/results/fourier_0/seismic_moe_curve_vel_b/FNO_router-basic_lr0.000267116_bs32_fourier_0_20251027-073251 \
  > "$LOGFILE" 2>&1 &
echo "推理已完成"
# moe_method basic/afreqmoe
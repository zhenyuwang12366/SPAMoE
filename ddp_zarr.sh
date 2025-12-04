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
  --model_name afreqmoe \
  --num_gpus 2 \
  --num_workers 16 \
  --zarr_path /root/autodl-tmp/curve_vel_a.zarr \
  --status_json ./dataset_status/dataset_status.json \
  --family curve_vel_a \
  --batch_size 32 \
  --epochs 200 \
  --output_dir ../results_seismic \
  --top_k 2 \
  --use_amp \
  > "$LOGFILE" 2>&1 &
echo "ddp训练已启动，日志记录在：$LOGFILE"
# moe_method basic/afreqmoe
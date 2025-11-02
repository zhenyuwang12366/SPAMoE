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
  --model_name moe_type \
  --num_gpus 2 \
  --num_workers 10 \
  --zarr_path /root/autodl-tmp/all.zarr \
  --status_json ./dataset_status/dataset_status.json \
  --family all \
  --batch_size 32 \
  --epochs 100 \
  --output_dir ../results \
  --use_moe \
  --use_experts_path /root/autodl-tmp/model_path_type_sum \
  --top_k 10 \
  --moe_mode group \
  --router_type swa \
  --fusion_type basic \
  --s_processor_type sum \
  --w_processor_type sum \
  --beta 0.5 \
  --is_specific \
  --is_classifier \
  --v_type_num 10 \
  --learning_rate 1e-4 \
  --lambda_g1v 0.43947650935102966 \
  --lambda_g2v 0.35339805101397564 \
  --lambda_grad_l1 0.15 \
  --lambda_fourier_mag_l1 0.05 \
  --use_amp \
  > "$LOGFILE" 2>&1 &
echo "ddp训练已启动，日志记录在：$LOGFILE"
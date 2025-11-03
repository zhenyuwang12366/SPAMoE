#!/bin/bash
. "/root/miniconda3/etc/profile.d/conda.sh"
conda activate seismic_moe

export PATH="$CONDA_PREFIX/bin:$PATH"
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGDIR=logs
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGDIR/deepspeed_${TIMESTAMP}.log"

echo "Using DeepSpeed from: $(which deepspeed)"
nohup deepspeed --num_gpus=4 scripts/train_seismic_moe.py \
  --mode train \
  --use_deepspeed \
  --ds_config ./scripts/deepspeed_zero3.json \
  --zarr_path /root/autodl-tmp/all.zarr \
  --status_json ./dataset_status/dataset_status.json \
  --family all \
  --batch_size 32 \
  --epochs 100 \
  --output_dir ../results \
  --use_moe \
  --use_experts_path /root/autodl-tmp/other_experts \
  --top_k 10 \
  --moe_mode group \
  --router_type basic \
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
  --use_amp > "$LOGFILE" 2>&1 &
echo "DeepSpeed 启动完成，日志：$LOGFILE"
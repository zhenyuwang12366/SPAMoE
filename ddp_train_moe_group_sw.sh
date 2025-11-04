#!/bin/bash
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/seismic_moe_${TIMESTAMP}.log"

# === Conda 环境 ===
. "/root/miniconda3/etc/profile.d/conda.sh"
conda activate seismic_moe

# === CUDA 内存池（4090 + torch 2.5 推荐 async allocator）===
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"

# === 线程与数学库 ===
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# === NCCL 与通信稳定性 ===
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=240

# === TF32（4090 推荐）===
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export NVIDIA_TF32_OVERRIDE=1

# === 打印运行环境到日志 ===
{
  echo "当前 Python: $(which python)"
  python - <<'PY'
import torch, os, platform
print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
backend = None
try:
    from torch.cuda import memory as tcm
    if hasattr(tcm, "get_allocator_backend"):
        backend = tcm.get_allocator_backend()
except Exception:
    pass
print("Allocator Backend:", backend or "unknown")
print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
print("Platform:", platform.platform())
PY
} | tee "$LOGFILE"

# === 启动 DDP 训练 ===
nohup bash scripts/run_distributed_seismic_moe.sh \
  --mode train \
  --model_name moe_sw \
  --num_gpus 4 \
  --num_workers 6 \
  --zarr_path /root/autodl-tmp/all.zarr \
  --status_json ./dataset_status/dataset_status.json \
  --family all \
  --batch_size 8 \
  --epochs 100 \
  --output_dir ../results \
  --use_moe \
  --use_experts_path /root/autodl-tmp/other_experts \
  --top_k 2 \
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
  >> "$LOGFILE" 2>&1 &
echo "DDP 训练已启动，日志路径：$LOGFILE"
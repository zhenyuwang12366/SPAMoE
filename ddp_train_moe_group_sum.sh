#!/bin/bash
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/seismic_moe_${TIMESTAMP}.log"

# === 进入工作目录（如需）===
# cd /data1/home/teacher/teacher_s/t108790/FWINO_wzy || exit 1

# === Conda ===
. "/root/miniconda3/etc/profile.d/conda.sh"
conda activate seismic_moe

# === 环境优化（与训练逻辑无关，但更稳）===
# 1) 优先使用 CUDA async 内存池；失败则回退到默认池 + 碎片缓解
choose_allocator_conf() {
  # 在子进程里设置 env，不污染当前；只做一次最小可运行检查
  if PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,max_split_size_mb:256,garbage_collection_threshold:0.8" \
     python - <<'PY' >/dev/null 2>&1
import torch  # 触发 CUDA/allocator 初始化路径的最小导入
PY
  then
    echo "backend:cudaMallocAsync,max_split_size_mb:256,garbage_collection_threshold:0.8"
  else
    echo "max_split_size_mb:256,garbage_collection_threshold:0.8"
  fi
}

export PYTORCH_CUDA_ALLOC_CONF="$(choose_allocator_conf)"

# 2) 线程/数学库
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# 3) NCCL 常用稳定性参数（按需微调）
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1               # 没有 IB 网络的话关掉以减少握手问题
export NCCL_SOCKET_IFNAME=eth0         # 依据你的容器网卡名调整
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=240

# 4) TF32（4090 推荐打开，省显存/提吞吐）
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export NVIDIA_TF32_OVERRIDE=1

# === 打印关键信息到日志 ===
{
  echo "当前 Python: $(which python)"
  python - <<'PY'
import torch, os, platform
print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
# 读取 allocator 后端（2.5+ 提供该 API，否则显示 unknown）
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

# === 启动训练 ===
nohup bash scripts/run_distributed_seismic_moe.sh \
  --mode train \
  --model_name moe_group_sum \
  --num_gpus 6 \
  --num_workers 72 \
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
  --use_amp \
  >> "$LOGFILE" 2>&1 &

echo "DDP 训练已启动，日志：$LOGFILE"
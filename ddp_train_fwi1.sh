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

# ==== 打印环境信息 ====
echo "当前 Python: $(which python)"
python - <<'PY'
import torch, torchvision
print("PyTorch 版本:", torch.__version__)
print("torchvision 版本:", torchvision.__version__ if hasattr(torchvision,'__version__') else 'N/A')
print("CUDA 可见设备:", torch.cuda.device_count())
PY

# ==== 通信/性能环境变量（单机稳定）====
export OMP_NUM_THREADS=5
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1              # 无 IB 时禁用 RDMA
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=lo          # 单机回环最稳；跨机改为实际网卡如 eth0/ens*
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==== 选择一个不冲突的端口 ====
MASTER_ADDR=127.0.0.1
MASTER_PORT=$((29000 + 1234 % 1000))

# === 启动训练 ===
CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup torchrun --standalone \
  --nproc-per-node=2 \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  /root/autodl-tmp/FWINO_wzy/openfwi/train.py \
  --mode train \
  --distributed \
  --zarr_path /root/autodl-tmp/all.zarr \
  --status_json ./dataset_status/dataset_status.json \
  --family all \
  --generator UPFWI \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --weight_decay 1e-2 \
  --use_amp \
  --output_dir ../results/openfwi_up \
  --log_root ./runs/plain_up \
  > "$LOGFILE" 2>&1 &
echo "openfwi训练已启动，日志记录在：$LOGFILE"
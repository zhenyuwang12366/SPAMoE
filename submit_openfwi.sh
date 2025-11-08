#!/bin/bash
#SBATCH --job-name=Openfwi_wzy
#SBATCH --partition=gpu-4090-2
#SBATCH --gres=gpu:2                 # 2 张 GPU
#SBATCH --ntasks=1                   # 由 torchrun 管理多进程
#SBATCH --cpus-per-task=10
#SBATCH --output=../results/output%j.txt
#SBATCH --no-requeue

# ==== 进入工作目录 ====
cd /data1/home/teacher/teacher_s/t108790/FWINO_wzy || exit 1

# ==== 激活 Conda ====
. "/data1/apps/anaconda3/etc/profile.d/conda.sh"
conda activate FWINO

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
MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))

# ==== 启动训练（单机 2 进程）====
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone \
  --nproc-per-node=2 \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  /data1/home/teacher/teacher_s/t108790/FWINO_wzy/openfwi/train.py \
  --mode train \
  --distributed \
  --zarr_path /data1/home/teacher/teacher_s/t108790/curve_vel_b.zarr \
  --status_json ./dataset_status/dataset_status.json \
  --family curve_vel_b \
  --generator InversionNet \
  --epochs 120 \
  --batch_size 256 \
  --learning_rate 1e-4 \
  --weight_decay 1e-2 \
  --use_amp \
  --output_dir ../results/openfwi \
  --log_root ./runs/plain
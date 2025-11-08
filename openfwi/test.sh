#!/bin/bash
#SBATCH --job-name=Openfwi_wzy
#SBATCH --partition=gpu-4090-2
#SBATCH --gres=gpu:2                 # 2 张 GPU
#SBATCH --ntasks=1                   # 由 torchrun 管理多进程
#SBATCH --cpus-per-task=10
#SBATCH --output=../results/output%j.txt
#SBATCH --no-requeue

EXPDIR=../results/seismic_moe_Openfwi_wzy_639

# ==== 进入工作目录 ====
cd /data1/home/teacher/teacher_s/t108790/OpenFWI || exit 1

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

python test.py \
  -ds curvevel-b \
  -n "$EXPDIR" \
  -m InversionNet \
  -v /data1/home/teacher/teacher_s/t108790/OpenFWI/split_files_local_full/curvevel_b_val.txt \
  -r /data1/home/teacher/teacher_s/t108790/OpenFWI/results/seismic_moe_Openfwi_wzy_639/checkpoint.pth \
  --vis -vb 2 -vsa 3
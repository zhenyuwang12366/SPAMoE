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

# === 频段对照实验列表（FNO/MNO/LNO + 融合） ===
ONLY_EXPS=(
  freq_fno_curve_fault_a_s0
  freq_mno_curve_fault_a_s0
  freq_lno_curve_fault_a_s0
  freq_fusion_curve_fault_a_s0
  freq_fno_curve_fault_a_s1
  freq_mno_curve_fault_a_s1
  freq_lno_curve_fault_a_s1
  freq_fusion_curve_fault_a_s1
  freq_fno_curve_fault_a_s2
  freq_mno_curve_fault_a_s2
  freq_lno_curve_fault_a_s2
  freq_fusion_curve_fault_a_s2
)

# === 启动训练 ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python exp/run_afreqmoe_pipeline.py \
  --num-gpus 2 \
  --families curve_fault_a \
  --seeds 0 1 2 \
  --seis-zarr /root/autodl-tmp \
  --seis-status-json ./dataset_status/dataset_status.json \
  --save-root ./exp/runs \
  --only "${ONLY_EXPS[@]}" \
  > "$LOGFILE" 2>&1 &
echo "exp已启动，日志记录在：$LOGFILE"
# moe_method basic/afreqmoe

#!/bin/bash
#SBATCH --job-name=FWINO_merge              # 作业名称
#SBATCH --partition=gpu-4090-2             # 分区名称（请根据实际情况调整）
#SBATCH --gres=gpu:2                       # 请求4个GPU
#SBATCH --ntasks=1                         # 启动1个任务（torchrun会管理GPU）
#SBATCH --cpus-per-task=10                  # 分配CPU
#SBATCH --time=24:00:00                    # 最长运行时间
#SBATCH --output=../results/output%j.txt             # 输出日志
#SBATCH --no-requeue

# === 进入工作目录（如果未用 sbatch --chdir） ===
cd /data1/home/teacher/teacher_s/t108790/FWINO_wzy || exit

# === 激活 Conda 虚拟环境 ===
. "/data1/apps/anaconda3/etc/profile.d/conda.sh"
conda activate FWINO

# === 打印验证信息（可选）===
echo "当前 Python: $(which python)"
python -c "import torch; print('PyTorch 版本:', torch.__version__)"

# === 启动训练 ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/run_distributed_seismic_moe.sh \
  --mode train \
  --num_gpus 2 \
  --num_workers 10 \
  --data_dir /data1/home/teacher/teacher_s/t108790/FWINO/FWINO_data \
  --family curve_vel \
  --batch_size 4 \
  --epochs 100 \
  --output_dir ../results/seismic_moe_${SLURM_JOB_NAME}_${SLURM_JOB_ID} \
  --use_moe \
  --use_experts_path /data1/home/teacher/teacher_s/t108790/model_path_specific \
  --choose_experts 0 2\
  --top_k 2 \
  --router_type basic \
  --fusion_type linear \
  --s_processor_type linear \
  --w_processor_type linear \
  --beta 0.5 \
  --is_specific \
  --is_classier \
  --v_type_num 1 \
  --learning_rate 1e-4



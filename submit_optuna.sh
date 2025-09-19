#!/bin/bash
#SBATCH --job-name=FWINO_optuna              # 作业名称
#SBATCH --partition=gpu-4090-2             # 分区名称（请根据实际情况调整）
#SBATCH --gres=gpu:1                       # 请求4个GPU
#SBATCH --ntasks=1                         # 启动1个任务（torchrun会管理GPU）
#SBATCH --cpus-per-task=10                  # 分配CPU
#SBATCH --time=24:00:00                    # 最长运行时间
#SBATCH --output=../results/output%j.txt             # 输出日志
#SBATCH --no-requeue

# === 进入工作目录（如果未用 sbatch --chdir） ===
cd /data1/home/teacher/teacher_s/t108790/FWINO_wzy_test

# === 激活 Conda 虚拟环境 ===
. "/data1/apps/anaconda3/etc/profile.d/conda.sh"
conda activate FWINO

# === 打印验证信息（可选）===
echo "当前 Python: $(which python)"
python -c "import torch; print('PyTorch 版本:', torch.__version__)"

# === 启动训练 ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/tune_seismic_moe.py \
  --data_dir /data1/home/teacher/teacher_s/t108790/FWINO/FWINO_data \
  --family flat_vel \
  --output_dir ../results/optuna/optuna_seismic_moe_${SLURM_JOB_NAME}_${SLURM_JOB_ID} \
  --num_workers 10 \
  --n_trials 30 \
  --study_name moe_flatvel_tpe\
  --num_gpus 1\
  --top_k 1\
  --choose_experts 1\
  --is_specific

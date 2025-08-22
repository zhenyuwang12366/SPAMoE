#!/bin/bash

# === 进入工作目录（如果未用 sbatch --chdir） ===
# cd autodl-tmp/FWINO
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/seismic_moe_${TIMESTAMP}.log
# === 激活 Conda 虚拟环境 ===
. "/root/miniconda3/etc/profile.d/conda.sh"
conda activate seismic_moe

# === 打印验证信息（可选）===
echo "当前 Python: $(which python)"
python -c "import torch; print('PyTorch 版本:', torch.__version__)"

# === 启动训练 ===
export WANDB_API_KEY=a8a4a60dbf66755b4d2af1a67ef020f69278f6a6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup bash scripts/run_distributed_seismic_moe.sh --num_gpus 2 --num_workers 32 --data_dir /root/autodl-tmp/FWINO/FWINO_data --family vel \
                                                --batch_size 4 --epochs 500 --output_dir ../results/seismic_moe_${TIMESTAMP} \
                                                --top_k 1 --choose_experts 0 --FNO_n_modes_height 64 --FNO_n_modes_width 64 --FNO_n_layers 8 --hidden_channels 128 \
                                                --learning_rate 2e-5 --weight_decay 0.05 --scheduler_gamma 0.2 --accum_steps 8\
    > "$LOGFILE" 2>&1 &
echo "训练已启动，日志记录在：$LOGFILE"

# bash scripts/run_distributed_seismic_moe.sh --num_gpus 2 --num_workers 32 --data_dir ./FWINO_data --family vel --batch_size 8 --epochs 500 --output_dir ../results/seismic_moe  --top_k 1 --choose_experts 1 --FNO_n_modes_height 32 --FNO_n_modes_width 32




#conda activate seismic_moe 
#cd ~/autodl-tmp/FWINO_wzy
#python scripts/train_seismic_moe.py --num_workers 8 --data_dir ./FWINO_data --family vel --batch_size 4 --epochs 500 --output_dir ../results/seismic_moe_${TIMESTAMP}  --top_k 0 --choose_experts 1 --FNO_n_modes_height 32 --FNO_n_modes_width 32 
#--resume_path /root/autodl-tmp/results/seismic_moe_20250820_195645/fourier_0/seismic_moe_vel/best_model_fourier_0.pt
#--resume_path /root/autodl-tmp/results/seismic_moe/fourier_0/seismic_moe_vel/best_model_fourier_0.pt
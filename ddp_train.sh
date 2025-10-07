#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/seismic_moe_${TIMESTAMP}.log

. "/root/miniconda3/etc/profile.d/conda.sh"
conda activate seismic_moe

echo "当前 Python: $(which python)"
python -c "import torch; print('PyTorch 版本:', torch.__version__)"

export WANDB_API_KEY=a8a4a60dbf66755b4d2af1a67ef020f69278f6a6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup bash scripts/run_distributed_seismic_moe.sh --num_gpus 2 --num_workers 32 --data_dir /root/autodl-tmp/FWINO/FWINO_data --family vel \
                                        --batch_size 4 --epochs 200 --output_dir ../results/seismic_moe_${TIMESTAMP} \
                                        --top_k 1 --choose_experts 0 --FNO_n_modes_height 64 --FNO_n_modes_width 64 --FNO_n_layers 8 --hidden_channels 128 \
                                        --learning_rate 1e-4 --weight_decay 0.05 --scheduler_gamma 0.2 \
    > "$LOGFILE" 2>&1 &
echo "ddp训练已启动，日志记录在：$LOGFILE"
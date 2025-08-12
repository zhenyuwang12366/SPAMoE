#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/seismic_moe_${TIMESTAMP}_single.log

. "/root/miniconda3/etc/profile.d/conda.sh"
conda activate seismic_moe

echo "当前 Python: $(which python)"
python -c "import torch; print('PyTorch 版本:', torch.__version__)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python scripts/train_seismic_moe.py --data_dir ../FWINO/FWINO_data --num_workers 4 --family vel --batch_size 8 --epochs 100 --output_dir ../results/seismic_moe_${TIMESTAMP}  --top_k 1 --choose_experts 0 --FNO_n_modes_height 48 --FNO_n_modes_width 48 --FNO_n_layers 10 --learning_rate 1e-4 --hidden_channel 128  \
    > "$LOGFILE" 2>&1 &
echo "单gpu训练已启动，日志记录在：$LOGFILE"
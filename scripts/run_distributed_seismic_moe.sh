#!/bin/bash

# 设置默认参数
NUM_GPUS=2
DATA_DIR="/data1/wuruoyu/waveform-inversion"
FAMILY="all"
BATCH_SIZE=8
EPOCHS=100
OUTPUT_DIR="./results/distributed_seismic_moe"
VIS_FREQ=5
USE_WANDB=0
VAL_RATIO=0.2

TOP_K=1
CHOOSE_EXPERTS=(0)
FNO_H=16
FNO_W=16
FNO_N_LAYERS=8
WNO_H=2
WNO_W=2
MNO_SCALES=3
MNO_FACTORS=(1.0 0.5 0.25)
MNO_LAYERS=3
LNO_MODES=(16 16)
LNO_LAYERS=3
NUM_WORKERS=4
SEED=42
K=1
HIDDEN_CHANNELS=64
LEARNING_RATE=0.001
RESUME_PATH="./results/distributed_seismic_moe"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --family) FAMILY="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --vis_freq) VIS_FREQ="$2"; shift 2 ;;
    --use_wandb) USE_WANDB=1; shift ;;
    --val_ratio) VAL_RATIO="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --choose_experts) shift; CHOOSE_EXPERTS=(); while [[ $# -gt 0 && $1 != --* ]]; do CHOOSE_EXPERTS+=("$1"); shift; done ;;
    --FNO_n_modes_height) FNO_H="$2"; shift 2 ;;
    --FNO_n_modes_width) FNO_W="$2"; shift 2 ;;
    --FNO_n_layers) FNO_N_LAYERS="$2"; shift 2 ;;
    --WNO_n_levels_height) WNO_H="$2"; shift 2 ;;
    --WNO_n_levels_width) WNO_W="$2"; shift 2 ;;
    --MNO_n_scales) MNO_SCALES="$2"; shift 2 ;;
    --MNO_scale_factors) shift; MNO_FACTORS=(); while [[ $# -gt 0 && $1 != --* ]]; do MNO_FACTORS+=("$1"); shift; done ;;
    --MNO_n_layers) MNO_LAYERS="$2"; shift 2 ;;
    --LNO_n_modes) shift; LNO_MODES=(); while [[ $# -gt 0 && $1 != --* ]]; do LNO_MODES+=("$1"); shift; done ;;
    --LNO_n_layers) LNO_LAYERS="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --hidden_channels) HIDDEN_CHANNELS="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --resume_path) RESUME_PATH="$2"; shift 2 ;;
    *)
      echo "未知参数: $1"
      exit 1 ;;
  esac
done

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 准备WandB参数
WANDB_ARG=""
if [ "$USE_WANDB" -eq 1 ]; then
  WANDB_ARG="--use_wandb"
fi

# 打印配置
echo "启动分布式训练，配置如下："
echo "GPU数量: $NUM_GPUS"
echo "数据目录: $DATA_DIR"
echo "数据系列: $FAMILY"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "输出目录: $OUTPUT_DIR"
echo "可视化频率: $VIS_FREQ"
echo "使用WandB: $USE_WANDB"
echo "验证集比例: $VAL_RATIO"
echo "专家选择: ${CHOOSE_EXPERTS[*]}"
echo "FNO 傅里叶变换后保留的模态数量(H, W): ($FNO_H, $FNO_W)"
echo "FNO 中傅里叶层的层数: $FNO_N_LAYERS"
echo "WNO 小波变换减少级别(H, W): WNO=($WNO_H, $WNO_W)"
echo "MNO 多尺度数量: $MNO_SCALES"
echo "MNO 每个尺度的缩放因子: ${MNO_FACTORS[*]}"
echo "MNO 每个尺度使用的神经网络层数: $MNO_LAYERS"
echo "LNO 局部变换后保留的模态数量(H, W): ${LNO_MODES[*]}"
echo "LNO 每个尺度使用的神经网络层数: $LNO_LAYERS"
echo "Num Workers: $NUM_WORKERS"
echo "Seed: $SEED"
echo "数据预处理缩放因子: $K"
echo "ResumePath: $RESUME_PATH"

# 启动分布式训练
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NUM_GPUS" \
  scripts/train_seismic_moe.py \
  --mode train \
  --data_dir "$DATA_DIR" \
  --family "$FAMILY" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --output_dir "$OUTPUT_DIR" \
  --vis_freq "$VIS_FREQ" \
  --val_ratio "$VAL_RATIO" \
  --distributed \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --top_k "$TOP_K" \
  --choose_experts "${CHOOSE_EXPERTS[@]}" \
  --FNO_n_modes_height "$FNO_H" \
  --FNO_n_modes_width "$FNO_W" \
  --FNO_n_layers "$FNO_N_LAYERS" \
  --WNO_n_levels_height "$WNO_H" \
  --WNO_n_levels_width "$WNO_W" \
  --MNO_n_scales "$MNO_SCALES" \
  --MNO_scale_factors "${MNO_FACTORS[@]}" \
  --MNO_n_layers "$MNO_LAYERS" \
  --LNO_n_modes "${LNO_MODES[@]}" \
  --LNO_n_layers "$LNO_LAYERS" \
  --k "$K" \
  --hidden_channels "$HIDDEN_CHANNELS" \
  --learning_rate "$LEARNING_RATE" \
  --resume_path "$RESUME_PATH" \

  $WANDB_ARG

echo "分布式训练完成！"

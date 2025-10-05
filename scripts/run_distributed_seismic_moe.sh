#!/bin/bash

# ================================
# 默认参数
# ================================
MODE="train"                     # train / inference / overfit1
NUM_GPUS=2
DATA_DIR="/data1/wuruoyu/waveform-inversion"
FAMILY="all"
BATCH_SIZE=8
EPOCHS=100
OUTPUT_DIR="./results/distributed_seismic_moe"
VIS_FREQ=5
USE_WANDB=0
VAL_RATIO=0.2
ACCUM_STEPS=1
NUM_WORKERS=4
SEED=42

# 训练超参
LEARNING_RATE=0.001
WEIGHT_DECAY=0.05
LR_WARMUP=5
LR_WARMUP_FACTOR=0.3333333333
LR_WARMUP_METHOD="linear"
LR_SCHEDULER_TYPE="cos_restart"
MILESTONES=(30 60 90)            # list 参数
SCHEDULER_GAMMA=0.2
LR_COSINE_TMAX_EPOCHS=50
LR_COSINE_RESTART_T0_EPOCHS=10
LR_COSINE_RESTART_T_MULT=2
LR_COSINE_ETA_MIN=1e-6
LAMBDA_G1V=1.0
LAMBDA_G2V=1.0
LAMBDA_GRAD_L1=0.0
LAMBDA_FOURIER_MAG_L1=0.0

# 预处理与路由/MoE
K=1
TOP_K=1
CHOOSE_EXPERTS=(0)

# FNO/WNO/MNO/LNO
FNO_H=16
FNO_W=16
FNO_N_LAYERS=8

WNO_H=2
WNO_W=2
# —— WNO 结构新增 ——
WNO_N_LAYERS=4
WNO_BLOCK_N_LAYERS=2
WNO_DROPOUT_RATE=0.1
WAVELET_TYPE="db6"   # haar | db4
WNO_PAD_MODE=""
WNO_ENSURE_SHAPES=""
WNO_ADAPTIVE_PADDING=""
WNO_USE_CHANNEL_MLP=""
WNO_CHANNEL_MLP_DROPOUT=""
WNO_CHANNEL_MLP_EXPANSION=""
DTCWT_TYPE=()

MNO_SCALES=3
MNO_FACTORS=(1.0 0.5 0.25)
MNO_LAYERS=3
LNO_MODES=(16 16)
LNO_LAYERS=3

# 模型结构 / MoE 融合相关
HIDDEN_CHANNELS=64
USE_EXPERTS_PATH=""
USE_MOE=0                       # flag -> --use_moe
ROUTER_TYPE="basic"             # 'basic' / 'adamv'
FUSION_TYPE="linear"            # 'linear' / 'attention' / 'swa'
S_PROCESSOR_TYPE="linear"       # 'linear' / 'atten' / 'mean' / 'sum'
W_PROCESSOR_TYPE="linear"       # 'linear' / 'atten' / 'mean' / 'sum'
BETA=0.5
IS_SPECIFIC=0                   # flag -> --is_specific
IS_CLASSIER=0                   # flag -> --is_classier
V_TYPE_NUM=""                  # 可选 -> --v_type_num

# 推理/恢复
MODEL_PATH=""                   # inference 时常用
RESUME_PATH=""

# 性能统计
PROFILE_TIMING=0                # flag -> --profile_timing

# ================================
# 解析命令行参数
# ================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode) MODE="$2"; shift 2 ;;
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --family) FAMILY="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --vis_freq) VIS_FREQ="$2"; shift 2 ;;
    --use_wandb) USE_WANDB=1; shift ;;

    --val_ratio) VAL_RATIO="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --accum_steps) ACCUM_STEPS="$2"; shift 2 ;;

    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --lr_warmup_epochs) LR_WARMUP="$2"; shift 2 ;;
    --lr_warmup_factor) LR_WARMUP_FACTOR="$2"; shift 2 ;;
    --lr_warmup_method) LR_WARMUP_METHOD="$2"; shift 2 ;;
    --lr_scheduler_type) LR_SCHEDULER_TYPE="$2"; shift 2 ;;
    --milestones) shift; MILESTONES=(); while [[ $# -gt 0 && $1 != --* ]]; do MILESTONES+=("$1"); shift; done ;;
    --scheduler_gamma) SCHEDULER_GAMMA="$2"; shift 2 ;;
    --lr_cosine_tmax_epochs) LR_COSINE_TMAX_EPOCHS="$2"; shift 2 ;;
    --lr_cosine_restart_t0_epochs) LR_COSINE_RESTART_T0_EPOCHS="$2"; shift 2 ;;
    --lr_cosine_restart_t_mult) LR_COSINE_RESTART_T_MULT="$2"; shift 2 ;;
    --lr_cosine_eta_min) LR_COSINE_ETA_MIN="$2"; shift 2 ;;
    --lambda_g1v|--g1v) LAMBDA_G1V="$2"; shift 2 ;;
    --lambda_g2v|--g2v) LAMBDA_G2V="$2"; shift 2 ;;
    --lambda_grad_l1) LAMBDA_GRAD_L1="$2"; shift 2 ;;
    --lambda_fourier_mag_l1) LAMBDA_FOURIER_MAG_L1="$2"; shift 2 ;;

    --k) K="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --choose_experts) shift; CHOOSE_EXPERTS=(); while [[ $# -gt 0 && $1 != --* ]]; do CHOOSE_EXPERTS+=("$1"); shift; done ;;

    --FNO_n_modes_height) FNO_H="$2"; shift 2 ;;
    --FNO_n_modes_width) FNO_W="$2"; shift 2 ;;
    --FNO_n_layers) FNO_N_LAYERS="$2"; shift 2 ;;

    --WNO_n_levels_height) WNO_H="$2"; shift 2 ;;
    --WNO_n_levels_width) WNO_W="$2"; shift 2 ;;
    --WNO_n_layers) WNO_N_LAYERS="$2"; shift 2 ;;
    --WNO_block_n_layers) WNO_BLOCK_N_LAYERS="$2"; shift 2 ;;
    --WNO_dropout_rate) WNO_DROPOUT_RATE="$2"; shift 2 ;;
    --wavelet_type) WAVELET_TYPE="$2"; shift 2 ;;
    --WNO_pad_mode) WNO_PAD_MODE="$2"; shift 2 ;;
    --WNO_ensure_even_shapes) WNO_ENSURE_SHAPES="1"; shift ;;
    --WNO_disable_ensure_even_shapes) WNO_ENSURE_SHAPES="0"; shift ;;
    --WNO_adaptive_padding) WNO_ADAPTIVE_PADDING="1"; shift ;;
    --WNO_disable_adaptive_padding) WNO_ADAPTIVE_PADDING="0"; shift ;;
    --WNO_use_channel_mlp) WNO_USE_CHANNEL_MLP="1"; shift ;;
    --WNO_disable_channel_mlp) WNO_USE_CHANNEL_MLP="0"; shift ;;
    --WNO_channel_mlp_dropout) WNO_CHANNEL_MLP_DROPOUT="$2"; shift 2 ;;
    --WNO_channel_mlp_expansion) WNO_CHANNEL_MLP_EXPANSION="$2"; shift 2 ;;
    --dtcwt_type) shift; DTCWT_TYPE=(); while [[ $# -gt 0 && $1 != --* ]]; do DTCWT_TYPE+=("$1"); shift; done ;;

    --MNO_n_scales) MNO_SCALES="$2"; shift 2 ;;
    --MNO_scale_factors) shift; MNO_FACTORS=(); while [[ $# -gt 0 && $1 != --* ]]; do MNO_FACTORS+=("$1"); shift; done ;;
    --MNO_n_layers) MNO_LAYERS="$2"; shift 2 ;;
    --LNO_n_modes) shift; LNO_MODES=(); while [[ $# -gt 0 && $1 != --* ]]; do LNO_MODES+=("$1"); shift; done ;;
    --LNO_n_layers) LNO_LAYERS="$2"; shift 2 ;;

    --hidden_channels) HIDDEN_CHANNELS="$2"; shift 2 ;;
    --use_experts_path) USE_EXPERTS_PATH="$2"; shift 2 ;;
    --use_moe) USE_MOE=1; shift ;;
    --router_type) ROUTER_TYPE="$2"; shift 2 ;;
    --fusion_type) FUSION_TYPE="$2"; shift 2 ;;
    --s_processor_type) S_PROCESSOR_TYPE="$2"; shift 2 ;;
    --w_processor_type) W_PROCESSOR_TYPE="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    --is_specific) IS_SPECIFIC=1; shift ;;
    --is_classier) IS_CLASSIER=1; shift ;;
    --v_type_num) V_TYPE_NUM="$2"; shift 2 ;;

    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --resume_path) RESUME_PATH="$2"; shift 2 ;;

    --profile_timing) PROFILE_TIMING=1; shift ;;

    *)
      echo "未知参数: $1"
      exit 1 ;;
  esac
done

# ================================
# 目录与可选参数拼装
# ================================
mkdir -p "$OUTPUT_DIR"

WANDB_ARG=""
[[ "$USE_WANDB" -eq 1 ]] && WANDB_ARG="--use_wandb"

USE_MOE_ARG=""
[[ "$USE_MOE" -eq 1 ]] && USE_MOE_ARG="--use_moe"

IS_SPECIFIC_ARG=""
[[ "$IS_SPECIFIC" -eq 1 ]] && IS_SPECIFIC_ARG="--is_specific"

IS_CLASSIER_ARG=""
[[ "$IS_CLASSIER" -eq 1 ]] && IS_CLASSIER_ARG="--is_classier"

USE_EXPERTS_PATH_ARG=""
[[ -n "$USE_EXPERTS_PATH" ]] && USE_EXPERTS_PATH_ARG="--use_experts_path \"$USE_EXPERTS_PATH\""

MODEL_PATH_ARG=""
[[ -n "$MODEL_PATH" ]] && MODEL_PATH_ARG="--model_path \"$MODEL_PATH\""

# ================================
# 打印配置
# ================================
echo "启动分布式训练/推理，配置如下："
echo "Mode: $MODE"
echo "GPU数量: $NUM_GPUS"
echo "数据目录: $DATA_DIR"
echo "数据系列: $FAMILY"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "输出目录: $OUTPUT_DIR"
echo "可视化频率: $VIS_FREQ"
echo "使用WandB: $USE_WANDB"
echo "验证集比例: $VAL_RATIO"
echo "Num Workers: $NUM_WORKERS"
echo "Seed: $SEED"
echo "梯度累计步数: $ACCUM_STEPS"
echo "学习率: $LEARNING_RATE"
echo "WeightDecay: $WEIGHT_DECAY"
echo "LR Warmup(Epochs): $LR_WARMUP"
echo "LR Warmup Factor: $LR_WARMUP_FACTOR"
echo "LR Warmup Method: $LR_WARMUP_METHOD"
echo "Milestones: ${MILESTONES[*]}"
echo "Scheduler Gamma: $SCHEDULER_GAMMA"
echo "LR Scheduler Type: $LR_SCHEDULER_TYPE"
echo "Cosine T_max (epochs): $LR_COSINE_TMAX_EPOCHS"
echo "Cosine Restart T0 (epochs): $LR_COSINE_RESTART_T0_EPOCHS"
echo "Cosine Restart T_mult: $LR_COSINE_RESTART_T_MULT"
echo "Cosine Eta Min: $LR_COSINE_ETA_MIN"
echo "Loss λ_g1v: $LAMBDA_G1V, λ_g2v: $LAMBDA_G2V, λ_grad_l1: $LAMBDA_GRAD_L1, λ_fourier_mag_l1: $LAMBDA_FOURIER_MAG_L1"
echo "数据预处理缩放因子 k: $K"
echo "Top-K 专家数: $TOP_K"
echo "专家选择: ${CHOOSE_EXPERTS[*]}"
echo "FNO(H,W,layers): ($FNO_H, $FNO_W, $FNO_N_LAYERS)"
echo "WNO(levels H,W): ($WNO_H, $WNO_W)"
echo "WNO(n_layers, block_layers, dropout, wavelet): ($WNO_N_LAYERS, $WNO_BLOCK_N_LAYERS, $WNO_DROPOUT_RATE, $WAVELET_TYPE)"
echo "WNO pad_mode override: ${WNO_PAD_MODE:-<default>}"
echo "WNO ensure_even_shapes flag: ${WNO_ENSURE_SHAPES:-<default>}"
echo "WNO adaptive_padding flag: ${WNO_ADAPTIVE_PADDING:-<default>}"
echo "WNO channel MLP usage flag: ${WNO_USE_CHANNEL_MLP:-<default>}"
echo "WNO channel MLP dropout: ${WNO_CHANNEL_MLP_DROPOUT:-<default>}"
echo "WNO channel MLP expansion: ${WNO_CHANNEL_MLP_EXPANSION:-<default>}"
echo "DTCWT type: ${DTCWT_TYPE[*]:-<None>}"
echo "MNO(scales,layers,factors): ($MNO_SCALES, $MNO_LAYERS, ${MNO_FACTORS[*]})"
echo "LNO(modes H W, layers): (${LNO_MODES[*]}, $LNO_LAYERS)"
echo "Hidden Channels: $HIDDEN_CHANNELS"
echo "Use Experts Path: ${USE_EXPERTS_PATH:-<None>}"
echo "Use MoE: $USE_MOE"
echo "Router Type: $ROUTER_TYPE"
echo "Fusion Type: $FUSION_TYPE"
echo "Strong-Group Processor: $S_PROCESSOR_TYPE"
echo "Weak-Group Processor: $W_PROCESSOR_TYPE"
echo "Beta(强弱激活): $BETA"
echo "is_specific: $IS_SPECIFIC, is_classier: $IS_CLASSIER"
echo "v_type_num: ${V_TYPE_NUM:-<auto>}"
echo "Model Path(仅 inference 用): ${MODEL_PATH:-<None>}"
echo "ResumePath: $RESUME_PATH"
echo "Profile Timing: $PROFILE_TIMING"

# ================================
# 启动 torchrun
# ================================
# 注意：为避免引号问题，拆分为数组传参更安全
ARGS=(
  --standalone
  --nnodes=1
  --nproc_per_node="$NUM_GPUS"
  scripts/train_seismic_moe.py
  --mode "$MODE"
  --data_dir "$DATA_DIR"
  --family "$FAMILY"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --output_dir "$OUTPUT_DIR"
  --vis_freq "$VIS_FREQ"
  --val_ratio "$VAL_RATIO"
  --distributed
  --num_workers "$NUM_WORKERS"
  --seed "$SEED"
  --top_k "$TOP_K"
  --choose_experts "${CHOOSE_EXPERTS[@]}"
  --FNO_n_modes_height "$FNO_H"
  --FNO_n_modes_width "$FNO_W"
  --FNO_n_layers "$FNO_N_LAYERS"
  --WNO_n_levels_height "$WNO_H"
  --WNO_n_levels_width "$WNO_W"
  --WNO_n_layers "$WNO_N_LAYERS"
  --WNO_block_n_layers "$WNO_BLOCK_N_LAYERS"
  --WNO_dropout_rate "$WNO_DROPOUT_RATE"
  --wavelet_type "$WAVELET_TYPE"
  --MNO_n_scales "$MNO_SCALES"
  --MNO_scale_factors "${MNO_FACTORS[@]}"
  --MNO_n_layers "$MNO_LAYERS"
  --LNO_n_modes "${LNO_MODES[@]}"
  --LNO_n_layers "$LNO_LAYERS"
  --k "$K"
  --hidden_channels "$HIDDEN_CHANNELS"
  --learning_rate "$LEARNING_RATE"
  --resume_path "$RESUME_PATH"
  --weight_decay "$WEIGHT_DECAY"
  --lr_warmup_epochs "$LR_WARMUP"
  --lr_warmup_factor "$LR_WARMUP_FACTOR"
  --lr_warmup_method "$LR_WARMUP_METHOD"
  --lr_scheduler_type "$LR_SCHEDULER_TYPE"
  --milestones "${MILESTONES[@]}"
  --scheduler_gamma "$SCHEDULER_GAMMA"
  --lr_cosine_tmax_epochs "$LR_COSINE_TMAX_EPOCHS"
  --lr_cosine_restart_t0_epochs "$LR_COSINE_RESTART_T0_EPOCHS"
  --lr_cosine_restart_t_mult "$LR_COSINE_RESTART_T_MULT"
  --lr_cosine_eta_min "$LR_COSINE_ETA_MIN"
  --accum_steps "$ACCUM_STEPS"
  --router_type "$ROUTER_TYPE"
  --fusion_type "$FUSION_TYPE"
  --s_processor_type "$S_PROCESSOR_TYPE"
  --w_processor_type "$W_PROCESSOR_TYPE"
  --beta "$BETA"
  --lambda_g1v "$LAMBDA_G1V"
  --lambda_g2v "$LAMBDA_G2V"
  --lambda_grad_l1 "$LAMBDA_GRAD_L1"
  --lambda_fourier_mag_l1 "$LAMBDA_FOURIER_MAG_L1"
)

[[ ${#DTCWT_TYPE[@]} -eq 2 ]] && ARGS+=( --dtcwt_type "${DTCWT_TYPE[@]}" )
[[ -n "$WNO_PAD_MODE" ]] && ARGS+=( --WNO_pad_mode "$WNO_PAD_MODE" )
if [[ "$WNO_ENSURE_SHAPES" == "1" ]]; then
  ARGS+=( --WNO_ensure_even_shapes )
elif [[ "$WNO_ENSURE_SHAPES" == "0" ]]; then
  ARGS+=( --WNO_disable_ensure_even_shapes )
fi
if [[ "$WNO_ADAPTIVE_PADDING" == "1" ]]; then
  ARGS+=( --WNO_adaptive_padding )
elif [[ "$WNO_ADAPTIVE_PADDING" == "0" ]]; then
  ARGS+=( --WNO_disable_adaptive_padding )
fi
if [[ "$WNO_USE_CHANNEL_MLP" == "1" ]]; then
  ARGS+=( --WNO_use_channel_mlp )
elif [[ "$WNO_USE_CHANNEL_MLP" == "0" ]]; then
  ARGS+=( --WNO_disable_channel_mlp )
fi
[[ -n "$WNO_CHANNEL_MLP_DROPOUT" ]] && ARGS+=( --WNO_channel_mlp_dropout "$WNO_CHANNEL_MLP_DROPOUT" )
[[ -n "$WNO_CHANNEL_MLP_EXPANSION" ]] && ARGS+=( --WNO_channel_mlp_expansion "$WNO_CHANNEL_MLP_EXPANSION" )

# 可选开关类参数
[[ "$USE_WANDB" -eq 1 ]]   && ARGS+=( --use_wandb )
[[ "$USE_MOE" -eq 1 ]]     && ARGS+=( --use_moe )
[[ "$IS_SPECIFIC" -eq 1 ]] && ARGS+=( --is_specific )
[[ "$IS_CLASSIER" -eq 1 ]] && ARGS+=( --is_classier )
[[ "$PROFILE_TIMING" -eq 1 ]] && ARGS+=( --profile_timing )

# 可选路径参数（非空才追加）
[[ -n "$USE_EXPERTS_PATH" ]] && ARGS+=( --use_experts_path "$USE_EXPERTS_PATH" )
[[ -n "$MODEL_PATH" ]]       && ARGS+=( --model_path "$MODEL_PATH" )
[[ -n "$V_TYPE_NUM" ]]       && ARGS+=( --v_type_num "$V_TYPE_NUM" )

torchrun "${ARGS[@]}"

echo "分布式流程结束！"

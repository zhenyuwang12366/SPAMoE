#!/bin/bash

# ================================
# 默认参数
# ================================
MODE="train"                     # train / inference / overfit1
NUM_GPUS=2
DATA_DIR="/data1/wuruoyu/waveform-inversion"
FAMILY="all"
MODEL_NAME="MOE"
BATCH_SIZE=8
TEST_BATCH_SIZE=""
EPOCHS=100
OUTPUT_DIR="./results/distributed_seismic_moe"
VIS_FREQ=5
USE_WANDB=0
USE_AMP=0
MIXED_PRECISION=""
VAL_RATIO=0.2
ACCUM_STEPS=1
NUM_WORKERS=4
SEED=42
N_TRAIN_SAMPLES=""
N_TEST_SAMPLES=""
CHANNEL_DIM=""
CONCAT_CHANNELS=""

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
LAMBDA_G1V=0.6
LAMBDA_G2V=0.4
LAMBDA_GRAD_L1=0.15
LAMBDA_FOURIER_MAG_L1=0.1
LAMBDA_CE=0.2

# 预处理与路由/MoE
K=1
TOP_K=1
CHOOSE_EXPERTS=(0)
MOE_MODE=""

# FNO/WNO/MNO/LNO/GeoFNO
FNO_H=16
FNO_W=16
FNO_N_LAYERS=8

WNO_H=2
WNO_W=2
# —— WNO 结构新增 ——
WNO_N_LAYERS=4
WNO_DROPOUT_RATE=0.1
WAVELET_TYPE="db6"   # haar | db4
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
ROUTER_HIDDEN_DIM=""
NOISY_GATING=""
IS_SPECIFIC=0                   # flag -> --is_specific
IS_CLASSIFIER=0                   # flag -> --is_classifier
V_TYPE_NUM=""                  # 可选 -> --v_type_num
USE_GPU_PROXY=0
USE_ENCODER=""
BACKBONE="vit"
ENCODER_PATH=""

# 推理/恢复
MODEL_PATH=""                   # inference 时常用
RESUME_PATH=""
LOG_ROOT=""

# 性能统计
PROFILE_TIMING=0                # flag -> --profile_timing
IS_RESIZE=0
H_SIZE=256
W_SIZE=256
USE_ONECYCLE=""
EARLY_STOP=0
EARLY_STOP_PATIENCE=""
EARLY_STOP_MIN_DELTA=""
EARLY_STOP_WARMUP=""
EVAL_INTERVAL=""
VERBOSE=""

# ================================
# 解析命令行参数
# ================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode) MODE="$2"; shift 2 ;;
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --family) FAMILY="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --test_batch_size) TEST_BATCH_SIZE="$2"; shift 2 ;;
    --n_train_samples) N_TRAIN_SAMPLES="$2"; shift 2 ;;
    --n_test_samples) N_TEST_SAMPLES="$2"; shift 2 ;;
    --channel_dim) CHANNEL_DIM="$2"; shift 2 ;;
    --concat_channels) CONCAT_CHANNELS=1; shift ;;
    --no_concat_channels) CONCAT_CHANNELS=0; shift ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --log_root) LOG_ROOT="$2"; shift 2 ;;
    --vis_freq) VIS_FREQ="$2"; shift 2 ;;
    --use_wandb) USE_WANDB=1; shift ;;
    --use_amp) USE_AMP=1; shift ;;
    --mixed_precision) MIXED_PRECISION=1; shift ;;
    --disable_mixed_precision) MIXED_PRECISION=0; shift ;;

    --val_ratio) VAL_RATIO="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --accum_steps) ACCUM_STEPS="$2"; shift 2 ;;
    --use_onecycle) USE_ONECYCLE=1; shift ;;
    --disable_onecycle) USE_ONECYCLE=0; shift ;;
    --early_stop) EARLY_STOP=1; shift ;;
    --early_stop_patience) EARLY_STOP_PATIENCE="$2"; shift 2 ;;
    --early_stop_min_delta) EARLY_STOP_MIN_DELTA="$2"; shift 2 ;;
    --early_stop_warmup_epochs) EARLY_STOP_WARMUP="$2"; shift 2 ;;
    --eval_interval) EVAL_INTERVAL="$2"; shift 2 ;;
    --distributed) shift ;;

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
    --lambda_ce) LAMBDA_CE="$2"; shift 2 ;;

    --k) K="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --choose_experts) shift; CHOOSE_EXPERTS=(); while [[ $# -gt 0 && $1 != --* ]]; do CHOOSE_EXPERTS+=("$1"); shift; done ;;

    --FNO_n_modes_height) FNO_H="$2"; shift 2 ;;
    --FNO_n_modes_width) FNO_W="$2"; shift 2 ;;
    --FNO_n_layers) FNO_N_LAYERS="$2"; shift 2 ;;

    --WNO_n_levels_height) WNO_H="$2"; shift 2 ;;
    --WNO_n_levels_width) WNO_W="$2"; shift 2 ;;
    --WNO_n_layers) WNO_N_LAYERS="$2"; shift 2 ;;
    --WNO_dropout_rate) WNO_DROPOUT_RATE="$2"; shift 2 ;;
    --wavelet_type) WAVELET_TYPE="$2"; shift 2 ;;
    --dtcwt_type) shift; DTCWT_TYPE=(); while [[ $# -gt 0 && $1 != --* ]]; do DTCWT_TYPE+=("$1"); shift; done ;;

    --MNO_n_scales) MNO_SCALES="$2"; shift 2 ;;
    --MNO_scale_factors) shift; MNO_FACTORS=(); while [[ $# -gt 0 && $1 != --* ]]; do MNO_FACTORS+=("$1"); shift; done ;;
    --MNO_n_layers) MNO_LAYERS="$2"; shift 2 ;;
    --LNO_n_modes) shift; LNO_MODES=(); while [[ $# -gt 0 && $1 != --* ]]; do LNO_MODES+=("$1"); shift; done ;;
    --LNO_n_layers) LNO_LAYERS="$2"; shift 2 ;;

    --hidden_channels) HIDDEN_CHANNELS="$2"; shift 2 ;;
    --use_experts_path) USE_EXPERTS_PATH="$2"; shift 2 ;;
    --use_moe) USE_MOE=1; shift ;;
    --moe_mode) MOE_MODE="$2"; shift 2 ;;
    --router_type) ROUTER_TYPE="$2"; shift 2 ;;
    --router_hidden_dim) ROUTER_HIDDEN_DIM="$2"; shift 2 ;;
    --fusion_type) FUSION_TYPE="$2"; shift 2 ;;
    --s_processor_type) S_PROCESSOR_TYPE="$2"; shift 2 ;;
    --w_processor_type) W_PROCESSOR_TYPE="$2"; shift 2 ;;
    --beta) BETA="$2"; shift 2 ;;
    --enable_noisy_gating) NOISY_GATING=1; shift ;;
    --disable_noisy_gating) NOISY_GATING=0; shift ;;
    --is_specific) IS_SPECIFIC=1; shift ;;
    --is_classifier) IS_CLASSIFIER=1; shift ;;
    --v_type_num) V_TYPE_NUM="$2"; shift 2 ;;
    --use_gpu_proxy) USE_GPU_PROXY=1; shift ;;
    --use_encoder) USE_ENCODER=1; shift ;;
    --disable_encoder) USE_ENCODER=0; shift ;;
    --backbone) BACKBONE="$2"; shift 2 ;;

    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --resume_path) RESUME_PATH="$2"; shift 2 ;;
    --encoder_path) ENCODER_PATH="$2"; shift 2 ;;

    --profile_timing) PROFILE_TIMING=1; shift ;;
    --is_resize) IS_RESIZE=1; shift ;;
    --H_size) H_SIZE="$2"; shift 2 ;;
    --W_size) W_SIZE="$2"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    --quiet) VERBOSE=0; shift ;;

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
[[ "$IS_CLASSIFIER" -eq 1 ]] && IS_CLASSIFIER_ARG="--is_classifier"

USE_EXPERTS_PATH_ARG=""
[[ -n "$USE_EXPERTS_PATH" ]] && USE_EXPERTS_PATH_ARG="--use_experts_path \"$USE_EXPERTS_PATH\""

MODEL_PATH_ARG=""
[[ -n "$MODEL_PATH" ]] && MODEL_PATH_ARG="--model_path \"$MODEL_PATH\""

if [[ $EARLY_STOP -eq 0 ]]; then
  if [[ -n "$EARLY_STOP_PATIENCE" || -n "$EARLY_STOP_MIN_DELTA" || -n "$EARLY_STOP_WARMUP" ]]; then
    EARLY_STOP=1
  fi
fi

if [[ "$CONCAT_CHANNELS" == "1" ]]; then
  CONCAT_DISPLAY="enabled"
elif [[ "$CONCAT_CHANNELS" == "0" ]]; then
  CONCAT_DISPLAY="disabled"
else
  CONCAT_DISPLAY="<config>"
fi

if [[ "$MIXED_PRECISION" == "1" ]]; then
  MIXED_PRECISION_DISPLAY="enabled"
elif [[ "$MIXED_PRECISION" == "0" ]]; then
  MIXED_PRECISION_DISPLAY="disabled"
else
  MIXED_PRECISION_DISPLAY="<config>"
fi

if [[ "$USE_ONECYCLE" == "1" ]]; then
  ONECYCLE_DISPLAY="enabled"
elif [[ "$USE_ONECYCLE" == "0" ]]; then
  ONECYCLE_DISPLAY="disabled"
else
  ONECYCLE_DISPLAY="<config>"
fi

if [[ "$NOISY_GATING" == "1" ]]; then
  NOISY_GATING_DISPLAY="enabled"
elif [[ "$NOISY_GATING" == "0" ]]; then
  NOISY_GATING_DISPLAY="disabled"
else
  NOISY_GATING_DISPLAY="<config>"
fi

if [[ "$VERBOSE" == "1" ]]; then
  VERBOSE_DISPLAY="verbose"
elif [[ "$VERBOSE" == "0" ]]; then
  VERBOSE_DISPLAY="quiet"
else
  VERBOSE_DISPLAY="<config>"
fi

if [[ "$USE_GPU_PROXY" -eq 1 ]]; then
  GPU_PROXY_DISPLAY="enabled"
else
  GPU_PROXY_DISPLAY="disabled"
fi

if [[ "$USE_ENCODER" == "1" ]]; then
  USE_ENCODER_DISPLAY="enabled"
elif [[ "$USE_ENCODER" == "0" ]]; then
  USE_ENCODER_DISPLAY="disabled"
else
  USE_ENCODER_DISPLAY="<config>"
fi

if [[ "$EARLY_STOP" -eq 1 ]]; then
  EARLY_STOP_DISPLAY="enabled"
else
  EARLY_STOP_DISPLAY="disabled"
fi

MOE_MODE_DISPLAY="${MOE_MODE:-<config>}"

# ================================
# 打印配置
# ================================
echo "启动分布式训练/推理，配置如下："
echo "Mode: $MODE"
echo "GPU数量: $NUM_GPUS"
echo "数据目录: $DATA_DIR"
echo "数据系列: $FAMILY"
echo "模型名称: $MODEL_NAME"
echo "批次大小: $BATCH_SIZE"
echo "测试批次大小: ${TEST_BATCH_SIZE:-<config>}"
echo "训练轮数: $EPOCHS"
echo "输出目录: $OUTPUT_DIR"
echo "TensorBoard 日志根目录: ${LOG_ROOT:-<config>}"
echo "可视化频率: $VIS_FREQ"
echo "使用WandB: $USE_WANDB"
echo "使用AMP: $USE_AMP"
echo "Mixed Precision override: $MIXED_PRECISION_DISPLAY"
echo "验证集比例: $VAL_RATIO"
echo "Num Workers: $NUM_WORKERS"
echo "Seed: $SEED"
echo "训练/测试子集: train=${N_TRAIN_SAMPLES:-<all>}, test=${N_TEST_SAMPLES:-<all>}"
echo "Channel dim override: ${CHANNEL_DIM:-<config>}, concat_channels: $CONCAT_DISPLAY"
echo "是否Resize输入: $IS_RESIZE, H_size: $H_SIZE, W_size: $W_SIZE"
echo "梯度累计步数: $ACCUM_STEPS"
echo "OneCycle 调度: $ONECYCLE_DISPLAY"
echo "早停: $EARLY_STOP_DISPLAY (patience=${EARLY_STOP_PATIENCE:-<config>}, min_delta=${EARLY_STOP_MIN_DELTA:-<config>}, warmup=${EARLY_STOP_WARMUP:-<config>})"
echo "验证间隔: ${EVAL_INTERVAL:-<config>}"
echo "日志输出模式: $VERBOSE_DISPLAY"
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
echo "Loss λ_g1v: $LAMBDA_G1V, λ_g2v: $LAMBDA_G2V, λ_grad_l1: $LAMBDA_GRAD_L1, λ_fourier_mag_l1: $LAMBDA_FOURIER_MAG_L1, λ_ce: $LAMBDA_CE"
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
echo "MoE 模式: $MOE_MODE_DISPLAY"
echo "Router Type: $ROUTER_TYPE"
echo "Router Hidden Dim: ${ROUTER_HIDDEN_DIM:-<config>}"
echo "Fusion Type: $FUSION_TYPE"
echo "Strong-Group Processor: $S_PROCESSOR_TYPE"
echo "Weak-Group Processor: $W_PROCESSOR_TYPE"
echo "Beta(强弱激活): $BETA"
echo "Noisy gating: $NOISY_GATING_DISPLAY"
echo "GPU proxy: $GPU_PROXY_DISPLAY"
echo "Encoder: $USE_ENCODER_DISPLAY"
echo "Encoder Backbone: $BACKBONE"
echo "Encoder Checkpoint: ${ENCODER_PATH:-<None>}"
echo "is_specific: $IS_SPECIFIC, is_classifier: $IS_CLASSIFIER"
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
  --model_name "$MODEL_NAME"
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
  --WNO_dropout_rate "$WNO_DROPOUT_RATE"
  --wavelet_type "$WAVELET_TYPE"
  --MNO_n_scales "$MNO_SCALES"
  --MNO_scale_factors "${MNO_FACTORS[@]}"
  --MNO_n_layers "$MNO_LAYERS"
  --LNO_n_modes "${LNO_MODES[@]}"
  --LNO_n_layers "$LNO_LAYERS"
  --k "$K"
  --H_size "$H_SIZE"
  --W_size "$W_SIZE"
  --hidden_channels "$HIDDEN_CHANNELS"
  --backbone "$BACKBONE"
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
  --lambda_ce "$LAMBDA_CE"
)

[[ -n "$TEST_BATCH_SIZE" ]] && ARGS+=( --test_batch_size "$TEST_BATCH_SIZE" )
[[ -n "$N_TRAIN_SAMPLES" ]] && ARGS+=( --n_train_samples "$N_TRAIN_SAMPLES" )
[[ -n "$N_TEST_SAMPLES" ]] && ARGS+=( --n_test_samples "$N_TEST_SAMPLES" )
[[ -n "$CHANNEL_DIM" ]] && ARGS+=( --channel_dim "$CHANNEL_DIM" )
if [[ "$CONCAT_CHANNELS" == "1" ]]; then
  ARGS+=( --concat_channels )
elif [[ "$CONCAT_CHANNELS" == "0" ]]; then
  ARGS+=( --no_concat_channels )
fi
if [[ "$MIXED_PRECISION" == "1" ]]; then
  ARGS+=( --mixed_precision )
elif [[ "$MIXED_PRECISION" == "0" ]]; then
  ARGS+=( --disable_mixed_precision )
fi
if [[ "$USE_ONECYCLE" == "1" ]]; then
  ARGS+=( --use_onecycle )
elif [[ "$USE_ONECYCLE" == "0" ]]; then
  ARGS+=( --disable_onecycle )
fi
if [[ "$EARLY_STOP" -eq 1 ]]; then
  ARGS+=( --early_stop )
fi
[[ -n "$EARLY_STOP_PATIENCE" ]] && ARGS+=( --early_stop_patience "$EARLY_STOP_PATIENCE" )
[[ -n "$EARLY_STOP_MIN_DELTA" ]] && ARGS+=( --early_stop_min_delta "$EARLY_STOP_MIN_DELTA" )
[[ -n "$EARLY_STOP_WARMUP" ]] && ARGS+=( --early_stop_warmup_epochs "$EARLY_STOP_WARMUP" )
[[ -n "$EVAL_INTERVAL" ]] && ARGS+=( --eval_interval "$EVAL_INTERVAL" )
if [[ "$VERBOSE" == "1" ]]; then
  ARGS+=( --verbose )
elif [[ "$VERBOSE" == "0" ]]; then
  ARGS+=( --quiet )
fi
[[ -n "$MOE_MODE" ]] && ARGS+=( --moe_mode "$MOE_MODE" )
[[ -n "$ROUTER_HIDDEN_DIM" ]] && ARGS+=( --router_hidden_dim "$ROUTER_HIDDEN_DIM" )
if [[ "$NOISY_GATING" == "1" ]]; then
  ARGS+=( --enable_noisy_gating )
elif [[ "$NOISY_GATING" == "0" ]]; then
  ARGS+=( --disable_noisy_gating )
fi
if [[ "$USE_GPU_PROXY" -eq 1 ]]; then
  ARGS+=( --use_gpu_proxy )
fi
[[ -n "$LAMBDA_GRAD" ]] && ARGS+=( --lambda_grad "$LAMBDA_GRAD" )
[[ -n "$LAMBDA_SSIM" ]] && ARGS+=( --lambda_ssim "$LAMBDA_SSIM" )

[[ ${#DTCWT_TYPE[@]} -eq 2 ]] && ARGS+=( --dtcwt_type "${DTCWT_TYPE[@]}" )

# 可选开关类参数
[[ "$USE_AMP" -eq 1 ]]   && ARGS+=( --use_amp )
[[ "$USE_WANDB" -eq 1 ]]   && ARGS+=( --use_wandb )
[[ "$USE_MOE" -eq 1 ]]     && ARGS+=( --use_moe )
[[ "$IS_SPECIFIC" -eq 1 ]] && ARGS+=( --is_specific )
[[ "$IS_CLASSIFIER" -eq 1 ]] && ARGS+=( --is_classifier )
[[ "$PROFILE_TIMING" -eq 1 ]] && ARGS+=( --profile_timing )
[[ "$IS_RESIZE" -eq 1 ]]  && ARGS+=( --is_resize )
if [[ "$USE_ENCODER" == "1" ]]; then
  ARGS+=( --use_encoder )
elif [[ "$USE_ENCODER" == "0" ]]; then
  ARGS+=( --disable_encoder )
fi

# 可选路径参数（非空才追加）
[[ -n "$USE_EXPERTS_PATH" ]] && ARGS+=( --use_experts_path "$USE_EXPERTS_PATH" )
[[ -n "$MODEL_PATH" ]]       && ARGS+=( --model_path "$MODEL_PATH" )
[[ -n "$ENCODER_PATH" ]]     && ARGS+=( --encoder_path "$ENCODER_PATH" )
[[ -n "$LOG_ROOT" ]]         && ARGS+=( --log_root "$LOG_ROOT" )
[[ -n "$V_TYPE_NUM" ]]       && ARGS+=( --v_type_num "$V_TYPE_NUM" )

torchrun "${ARGS[@]}"

echo "分布式流程结束！"

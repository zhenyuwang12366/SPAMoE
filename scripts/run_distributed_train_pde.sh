#!/bin/bash

# 分布式启动 pde/train_pde.py，参考 scripts/run_distributed_seismic_moe.sh
set -e

# 默认参数，可通过命令行覆盖
NUM_GPUS=2
TASK="navier2d"
DATA_ROOT_BASE="./pdebench_data"
DATA_ROOT=""
STATUS_JSON="./pde_status.json"
SAVE_DIR_BASE="./results_pde"
SAVE_DIR=""
EPOCHS=200
BATCH_SIZE=16
TEST_BATCH_SIZE=16
LR=1e-4
WEIGHT_DECAY=0.0
USE_AMP=1
AMP_DTYPE="bfloat16"
NUM_WORKERS=4
SEED=42
ROUTER_TYPE="sar"
TOP_K=2
HIDDEN_CHANNELS=128
BACKBONE="vit"
AUX_LOSS_WEIGHT=0.1
RESUME_PATH=""
LOG_EVERY=50
VIS_EVERY=200
VIS_ROUTER_EVERY=400
BAND_SHARPNESS=20.0
FREQ_AFFINITY_SHARPNESS=10.0
DISABLE_SOFT_BANDS=0
DISABLE_FREQ_ATTN=0
DISABLE_BAND_MIXING=0

USE_DEFAULT_DATA_ROOT=1
USE_DEFAULT_SAVE_DIR=1

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; USE_DEFAULT_DATA_ROOT=0; shift 2 ;;
    --status_json) STATUS_JSON="$2"; shift 2 ;;
    --save_dir) SAVE_DIR="$2"; USE_DEFAULT_SAVE_DIR=0; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --test_batch_size) TEST_BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --use_amp) USE_AMP=1; shift ;;
    --no_amp) USE_AMP=0; shift ;;
    --amp_dtype) AMP_DTYPE="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --router_type) ROUTER_TYPE="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --hidden_channels) HIDDEN_CHANNELS="$2"; shift 2 ;;
    --backbone) BACKBONE="$2"; shift 2 ;;
    --aux_loss_weight) AUX_LOSS_WEIGHT="$2"; shift 2 ;;
    --band_sharpness) BAND_SHARPNESS="$2"; shift 2 ;;
    --freq_affinity_sharpness) FREQ_AFFINITY_SHARPNESS="$2"; shift 2 ;;
    --disable_soft_bands) DISABLE_SOFT_BANDS=1; shift ;;
    --disable_freq_attn) DISABLE_FREQ_ATTN=1; shift ;;
    --disable_band_mixing) DISABLE_BAND_MIXING=1; shift ;;
    --resume_path) RESUME_PATH="$2"; shift 2 ;;
    --log_every) LOG_EVERY="$2"; shift 2 ;;
    --vis_every) VIS_EVERY="$2"; shift 2 ;;
    --vis_router_every) VIS_ROUTER_EVERY="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# 根据 task 生成默认路径
if [[ $USE_DEFAULT_DATA_ROOT -eq 1 ]]; then
  DATA_ROOT="${DATA_ROOT_BASE}/${TASK}"
fi
if [[ $USE_DEFAULT_SAVE_DIR -eq 1 ]]; then
  SAVE_DIR="${SAVE_DIR_BASE}/${TASK}"
fi

mkdir -p "$SAVE_DIR"

echo "启动分布式 PDE 训练:"
echo "  GPUs: $NUM_GPUS"
echo "  Task: $TASK"
echo "  Data root: $DATA_ROOT"
echo "  Status JSON: $STATUS_JSON"
echo "  Save dir: $SAVE_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE (test=$TEST_BATCH_SIZE)"
echo "  LR / WD: $LR / $WEIGHT_DECAY"
echo "  AMP: $USE_AMP ($AMP_DTYPE)"
echo "  Num workers: $NUM_WORKERS"
echo "  Seed: $SEED"
echo "  Router: $ROUTER_TYPE, top_k=$TOP_K, hidden=$HIDDEN_CHANNELS, backbone=$BACKBONE"
echo "  AFreqMoE: band_sharpness=$BAND_SHARPNESS, freq_affinity_sharpness=$FREQ_AFFINITY_SHARPNESS, soft_bands=$((1-DISABLE_SOFT_BANDS)), freq_attn=$((1-DISABLE_FREQ_ATTN)), band_mixing=$((1-DISABLE_BAND_MIXING))"
echo "  Aux loss weight: $AUX_LOSS_WEIGHT"
echo "  Resume: ${RESUME_PATH:-<none>}"
echo "  Log every: $LOG_EVERY; Vis every: $VIS_EVERY; Router vis every: $VIS_ROUTER_EVERY"

ARGS=(
  --standalone
  --nnodes=1
  --nproc_per_node="$NUM_GPUS"
  pde/train_pde.py
  --distributed
  --task "$TASK"
  --data_root "$DATA_ROOT"
  --status_json "$STATUS_JSON"
  --save_dir "$SAVE_DIR"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --test_batch_size "$TEST_BATCH_SIZE"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --seed "$SEED"
  --amp_dtype "$AMP_DTYPE"
  --num_workers "$NUM_WORKERS"
  --router_type "$ROUTER_TYPE"
  --top_k "$TOP_K"
  --hidden_channels "$HIDDEN_CHANNELS"
  --backbone "$BACKBONE"
  --aux_loss_weight "$AUX_LOSS_WEIGHT"
  --band_sharpness "$BAND_SHARPNESS"
  --freq_affinity_sharpness "$FREQ_AFFINITY_SHARPNESS"
  --log_every "$LOG_EVERY"
  --vis_every "$VIS_EVERY"
  --vis_router_every "$VIS_ROUTER_EVERY"
)

if [[ $USE_AMP -eq 1 ]]; then
  ARGS+=( --use_amp )
else
  ARGS+=( --no_amp )
fi

[[ -n "$RESUME_PATH" ]] && ARGS+=( --resume_path "$RESUME_PATH" )
[[ $DISABLE_SOFT_BANDS -eq 1 ]] && ARGS+=( --disable_soft_bands )
[[ $DISABLE_FREQ_ATTN -eq 1 ]] && ARGS+=( --disable_freq_attn )
[[ $DISABLE_BAND_MIXING -eq 1 ]] && ARGS+=( --disable_band_mixing )

torchrun "${ARGS[@]}"

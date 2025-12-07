#!/bin/bash

# 分布式启动 pde/test_pde.py，参考 scripts/run_distributed_seismic_moe.sh
set -e

NUM_GPUS=2
TASK="navier2d"
DATA_ROOT_BASE="../pdebench_data"
DATA_ROOT=""
STATUS_JSON="../pdebench_data/pde_status.json"
SAVE_DIR_BASE="../results_pde"
SAVE_DIR=""
CHECKPOINT=""
SPLIT="test"
BATCH_SIZE=16
USE_AMP=1
AMP_DTYPE="bfloat16"
NUM_WORKERS=4
SEED=42
VIS_EVERY=1
BAND_SHARPNESS=""
FREQ_AFFINITY_SHARPNESS=""
DISABLE_SOFT_BANDS=0
DISABLE_FREQ_ATTN=0
DISABLE_BAND_MIXING=0

USE_DEFAULT_DATA_ROOT=1
USE_DEFAULT_SAVE_DIR=1
USE_DEFAULT_CHECKPOINT=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; USE_DEFAULT_DATA_ROOT=0; shift 2 ;;
    --status_json) STATUS_JSON="$2"; shift 2 ;;
    --save_dir) SAVE_DIR="$2"; USE_DEFAULT_SAVE_DIR=0; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; USE_DEFAULT_CHECKPOINT=0; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --use_amp) USE_AMP=1; shift ;;
    --no_amp) USE_AMP=0; shift ;;
    --amp_dtype) AMP_DTYPE="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --vis_every) VIS_EVERY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --band_sharpness) BAND_SHARPNESS="$2"; shift 2 ;;
    --freq_affinity_sharpness) FREQ_AFFINITY_SHARPNESS="$2"; shift 2 ;;
    --disable_soft_bands) DISABLE_SOFT_BANDS=1; shift ;;
    --disable_freq_attn) DISABLE_FREQ_ATTN=1; shift ;;
    --disable_band_mixing) DISABLE_BAND_MIXING=1; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ $USE_DEFAULT_DATA_ROOT -eq 1 ]]; then
  DATA_ROOT="${DATA_ROOT_BASE}/${TASK}"
fi
if [[ $USE_DEFAULT_SAVE_DIR -eq 1 ]]; then
  SAVE_DIR="${SAVE_DIR_BASE}/${TASK}_test"
fi
if [[ $USE_DEFAULT_CHECKPOINT -eq 1 ]]; then
  CHECKPOINT="${SAVE_DIR_BASE}/${TASK}/checkpoint_best.pt"
fi

mkdir -p "$SAVE_DIR"

echo "启动分布式 PDE 测试:"
echo "  GPUs: $NUM_GPUS"
echo "  Task: $TASK"
echo "  Data root: $DATA_ROOT"
echo "  Status JSON: $STATUS_JSON"
echo "  Checkpoint: $CHECKPOINT"
echo "  Split: $SPLIT"
echo "  Save dir: $SAVE_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  AMP: $USE_AMP ($AMP_DTYPE)"
echo "  Num workers: $NUM_WORKERS"
echo "  Seed: $SEED"
echo "  Vis every: $VIS_EVERY"
echo "  AFreqMoE overrides: band_sharpness=${BAND_SHARPNESS:-<ckpt>}, freq_affinity_sharpness=${FREQ_AFFINITY_SHARPNESS:-<ckpt>}, soft_bands=$((1-DISABLE_SOFT_BANDS)), freq_attn=$((1-DISABLE_FREQ_ATTN)), band_mixing=$((1-DISABLE_BAND_MIXING))"

ARGS=(
  --standalone
  --nnodes=1
  --nproc_per_node="$NUM_GPUS"
  pde/test_pde.py
  --distributed
  --task "$TASK"
  --data_root "$DATA_ROOT"
  --status_json "$STATUS_JSON"
  --checkpoint "$CHECKPOINT"
  --split "$SPLIT"
  --batch_size "$BATCH_SIZE"
  --amp_dtype "$AMP_DTYPE"
  --num_workers "$NUM_WORKERS"
  --save_dir "$SAVE_DIR"
  --vis_every "$VIS_EVERY"
  --seed "$SEED"
)

if [[ $USE_AMP -eq 1 ]]; then
  ARGS+=( --use_amp )
else
  ARGS+=( --no_amp )
fi

[[ -n "$BAND_SHARPNESS" ]] && ARGS+=( --band_sharpness "$BAND_SHARPNESS" )
[[ -n "$FREQ_AFFINITY_SHARPNESS" ]] && ARGS+=( --freq_affinity_sharpness "$FREQ_AFFINITY_SHARPNESS" )
[[ $DISABLE_SOFT_BANDS -eq 1 ]] && ARGS+=( --disable_soft_bands )
[[ $DISABLE_FREQ_ATTN -eq 1 ]] && ARGS+=( --disable_freq_attn )
[[ $DISABLE_BAND_MIXING -eq 1 ]] && ARGS+=( --disable_band_mixing )

torchrun "${ARGS[@]}"

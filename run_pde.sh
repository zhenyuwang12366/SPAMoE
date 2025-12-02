#!/bin/bash

# === Conda 环境 ===
. "/data1/apps/anaconda3/etc/profile.d/conda.sh"
conda activate FWINO

MODE=TRAIN   # train/test
TASK=pipe    # pipe/darcy/navier/plasticity/airfoil

# === 日志 ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs
mkdir -p "$LOGDIR"
LOGFILE=$LOGDIR/pde_${TASK}_${MODE}_${TIMESTAMP}.log
echo "日志文件: $LOGFILE"
exec > >(tee -a "$LOGFILE") 2>&1

# === 数据路径 ===
LAMO_DATA=../LaMO/data/$TASK
PDE_DATA=../pdebench_data/$TASK
STATUS_JSON=$PDE_DATA/${TASK}_train_stats.json
SAVE_DIR=../results_pde/$TASK
CKPT=$SAVE_DIR/checkpoint_best.pt

# === 让 PyTorch 使用分段显存，减少 OOM ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === SLURM 分配到的可见 GPU ===
# torchrun 会自动创建 LOCAL_RANK / RANK / WORLD_SIZE
export CUDA_VISIBLE_DEVICES=0,1

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

if [[ "$MODE" == "TRAIN" ]]; then
  echo "[train] pde"
  bash scripts/run_distributed_train_pde.sh \
      --task "$TASK" \
      --data_root "$PDE_DATA" \
      --status_json "$STATUS_JSON" \
      --save_dir "$SAVE_DIR" \
else
  echo "[test] pde"
  bash scripts/run_distributed_test_pde.sh \
      --task "$TASK" \
      --data_root "$PDE_DATA" \
      --status_json "$STATUS_JSON" \
      --save_dir "$SAVE_DIR"
fi

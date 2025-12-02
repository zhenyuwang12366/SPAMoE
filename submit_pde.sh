#!/bin/bash
#SBATCH --job-name=pde
#SBATCH --partition=gpu-4090-2
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=../results/output%j.txt
#SBATCH --no-requeue

cd /data1/home/teacher/teacher_s/t108790/FWINO_wzy || exit

# === Conda 环境 ===
. "/data1/apps/anaconda3/etc/profile.d/conda.sh"
conda activate FWINO

# === 与 run_pde.sh 保持一致的默认配置 ===
MODE=TRAIN   # TRAIN / TEST
TASK=pipe    # pipe/darcy/navier/plasticity/airfoil

LAMO_DATA=${LAMO_DATA:-../LaMO/data/$TASK}
PDE_DATA=${PDE_DATA:-../pdebench_data/$TASK}
STATUS_JSON=${STATUS_JSON:-$PDE_DATA/${TASK}_train_stats.json}
SAVE_DIR=${SAVE_DIR:-../results_pde/$TASK}
CKPT=${CKPT:-$SAVE_DIR/checkpoint_best.pt}
NUM_GPUS=${NUM_GPUS:-2}

# === GPU & 运行目录 ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0,1}}

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

if [[ "$MODE" == "TRAIN" ]]; then
  echo "[train] pde"
  bash scripts/run_distributed_train_pde.sh \
    --num_gpus "$NUM_GPUS" \
    --task "$TASK" \
    --data_root "$PDE_DATA" \
    --status_json "$STATUS_JSON" \
    --save_dir "$SAVE_DIR"
else
  echo "[test] pde"
  bash scripts/run_distributed_test_pde.sh \
    --num_gpus "$NUM_GPUS" \
    --task "$TASK" \
    --data_root "$PDE_DATA" \
    --status_json "$STATUS_JSON" \
    --save_dir "$SAVE_DIR" \
    --checkpoint "$CKPT"
fi

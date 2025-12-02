#!/bin/bash

# 一键将 LaMO 原始数据转为 PDEBench 风格 HDF5（含 train/val/test 划分与全局 stats）。
# 可根据 TASK 自动选择数据根目录和输出路径。

set -e

# ---- 工作目录与环境（按需修改）----
cd /data1/home/teacher/teacher_s/t108790/FWINO_wzy || exit 1

. "/data1/apps/anaconda3/etc/profile.d/conda.sh"
conda activate FWINO

# ---- 配置任务 ----
# 可选任务：pipe | airfoil | darcy | navier | plasticity
TASK=${TASK:-darcy}
DATA_ROOT=${DATA_ROOT:-../LaMO/data/$TASK}
# 输出根目录（存放划分后的 h5 和 *_stats.json）
PDE_DATA=${PDE_DATA:-../pdebench_data}
mkdir -p "$PDE_DATA"
OUTPUT="${PDE_DATA}/${TASK}/${TASK}.h5"

echo "[convert] TASK=$TASK data_root=$DATA_ROOT -> $OUTPUT"
python pde/convert_lamo_to_h5.py \
  --task "$TASK" \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT" \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

echo "Done. 生成文件位于 $PDE_DATA 下（*_train/val/test.h5 及 *_stats.json）。"

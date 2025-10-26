#!/bin/bash

. "/data1/apps/anaconda3/etc/profile.d/conda.sh"
conda activate FWINO

python load_to_zarr.py \
  --data_dir /data1/home/teacher/teacher_s/t108790/DATAA \
  --zarr_out /data1/home/teacher/teacher_s/t108790/curve_vel_b.zarr \
  --family curve_vel_b \
  --remap_single_label 0 \
  --include_test 0


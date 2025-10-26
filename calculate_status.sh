#!/bin/bash

. "/data1/apps/anaconda3/etc/profile.d/conda.sh"
conda activate FWINO

python calculate_dataset_status.py \
  --data-dir /data1/home/teacher/teacher_s/t108790/DATAA \
  --output ./dataset_status/dataset_status.json

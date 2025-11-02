#!/bin/bash

. "/root/miniconda3/etc/profile.d/conda.sh"
conda activate seismic_moe

python calculate_dataset_status.py \
  --data-dir /root/autodl-tmp/All \
  --output ./dataset_status/dataset_status.json

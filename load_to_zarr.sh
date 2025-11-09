#!/bin/bash

. "/root/miniconda3/etc/profile.d/conda.sh"
conda activate seismic_moe

python load_to_zarr.py \
  --data_dir /root/autodl-tmp/All \
  --zarr_out /root/autodl-tmp/all.zarr \
  --family all \
  --remap_single_label 0 \
  --include_test 0 \
  --concat_channels 0


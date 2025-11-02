#!/bin/bash

srun --partition=gpu-4090-2 --gres=gpu:1 --cpus-per-task=10 --pty bash

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m pdb scripts/train_seismic_moe.py \
  --mode train \
  --model_name FNO_no_enocder_70 \
  --concat_channels \
  --is_resize \
  --H_size 70 \
  --W_size 70 \
  --is_specific \
  --num_workers 10 \
  --zarr_path /data1/home/teacher/teacher_s/t108790/curve_vel_b.zarr \
  --status_json ./dataset_status/dataset_status.json \
  --family curve_vel_b \
  --is_specific \
  --batch_size 4 \
  --accum_steps 8\
  --epochs 100 \
  --output_dir ../results \
  --top_k 1 \
  --choose_experts 0 \
  --hidden_channels 128 \
  --learning_rate 0.00026711555047527854 \
  --weight_decay 0.08952068376871994 \
  --scheduler_gamma 0.2966237496749535 \
  --FNO_n_layers 6 \
  --FNO_n_modes_height 64 \
  --FNO_n_modes_width 64 \
  --lambda_g1v 0.43947650935102966 \
  --lambda_g2v 0.35339805101397564 \
  --lambda_grad_l1 0.15 \
  --lambda_fourier_mag_l1 0.05 \
  --wavelet_type db6 \
  --WNO_n_levels_height 3 \
  --WNO_n_levels_width 2 \
  --WNO_n_layers 4 \
  --WNO_dropout_rate 0.10 \
  --use_amp
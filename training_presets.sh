#!/usr/bin/env bash

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/scripts/bash_helpers.sh"

PRESET_ARGS=()

set_training_preset_args() {
  local family preset
  family="$(normalize_family_name "${1:-}")"
  preset="${2:-default}"
  PRESET_ARGS=()

  case "${preset}" in
    default)
      return 0
      ;;
    preset1)
      case "${family}" in
        style_style_a)
          PRESET_ARGS=(
            --use_amp
            --batch_size 32
            --epochs 180
            --top_k 2
            --hidden_channels 128
            --learning_rate 8e-5
            --weight_decay 8e-5
            --band_sharpness 24
            --freq_affinity_sharpness 12
            --lambda_grad_l1 0.12
            --lambda_fourier_mag_l1 0.14
          )
          ;;
        style_style_b)
          PRESET_ARGS=(
            --use_amp
            --batch_size 32
            --epochs 180
            --top_k 2
            --hidden_channels 128
            --learning_rate 9e-5
            --weight_decay 8e-5
            --band_sharpness 22
            --freq_affinity_sharpness 12
            --lambda_grad_l1 0.10
            --lambda_fourier_mag_l1 0.16
          )
          ;;
        flat_fault_b)
          PRESET_ARGS=(
            --use_amp
            --batch_size 32
            --epochs 180
            --top_k 2
            --hidden_channels 128
            --learning_rate 8e-5
            --weight_decay 1e-4
            --band_sharpness 26
            --freq_affinity_sharpness 13
            --FNO_n_modes_height 20
            --FNO_n_modes_width 20
            --lambda_grad_l1 0.20
            --lambda_fourier_mag_l1 0.14
          )
          ;;
        curve_vel_a)
          PRESET_ARGS=(
            --use_amp
            --batch_size 32
            --epochs 160
            --top_k 2
            --hidden_channels 96
            --learning_rate 1e-4
            --weight_decay 8e-5
            --band_sharpness 20
            --freq_affinity_sharpness 10
            --lambda_grad_l1 0.12
            --lambda_fourier_mag_l1 0.12
          )
          ;;
        curve_fault_b)
          PRESET_ARGS=(
            --use_amp
            --batch_size 32
            --epochs 180
            --top_k 2
            --hidden_channels 128
            --learning_rate 8e-5
            --weight_decay 8e-5
            --band_sharpness 24
            --freq_affinity_sharpness 12
            --FNO_n_modes_height 20
            --FNO_n_modes_width 20
            --lambda_grad_l1 0.20
            --lambda_fourier_mag_l1 0.14
          )
          ;;
        curve_vel_b)
          PRESET_ARGS=(
            --use_amp
            --batch_size 32
            --epochs 160
            --top_k 2
            --hidden_channels 96
            --learning_rate 1e-4
            --weight_decay 1e-4
            --band_sharpness 20
            --freq_affinity_sharpness 10
          )
          ;;
        *)
          echo "Unsupported family: ${family}" >&2
          return 1
          ;;
      esac
      ;;
    preset2)
      case "${family}" in
        style_style_a)
          PRESET_ARGS=(
            --use_amp
            --batch_size 32
            --epochs 220
            --top_k 2
            --hidden_channels 160
            --learning_rate 6e-5
            --weight_decay 5e-5
            --band_sharpness 28
            --freq_affinity_sharpness 14
            --FNO_n_modes_height 20
            --FNO_n_modes_width 20
            --MNO_n_scales 4
            --MNO_scale_factors 1.0 0.75 0.5 0.25
            --LNO_n_layers 4
            --lambda_fourier_mag_l1 0.18
          )
          ;;
        style_style_b)
          PRESET_ARGS=(
            --use_amp
            --batch_size 32
            --epochs 220
            --top_k 2
            --hidden_channels 160
            --learning_rate 6e-5
            --weight_decay 5e-5
            --band_sharpness 26
            --freq_affinity_sharpness 14
            --FNO_n_modes_height 24
            --FNO_n_modes_width 24
            --MNO_n_scales 4
            --MNO_scale_factors 1.0 0.8 0.55 0.3
            --LNO_n_modes 20 20
            --LNO_n_layers 4
            --lambda_grad_l1 0.12
            --lambda_fourier_mag_l1 0.18
          )
          ;;
        flat_fault_b)
          PRESET_ARGS=(
            --use_amp
            --batch_size 32
            --epochs 220
            --top_k 2
            --hidden_channels 160
            --learning_rate 5e-5
            --weight_decay 8e-5
            --band_sharpness 30
            --freq_affinity_sharpness 15
            --FNO_n_modes_height 24
            --FNO_n_modes_width 24
            --MNO_n_scales 4
            --MNO_scale_factors 1.0 0.7 0.45 0.25
            --LNO_n_modes 20 20
            --LNO_n_layers 4
            --beta 0.6
            --lambda_grad_l1 0.25
            --lambda_fourier_mag_l1 0.18
          )
          ;;
        curve_vel_a)
          PRESET_ARGS=(
            --use_amp
            --batch_size 32
            --epochs 200
            --top_k 2
            --hidden_channels 128
            --learning_rate 7e-5
            --weight_decay 5e-5
            --band_sharpness 24
            --freq_affinity_sharpness 12
            --FNO_n_modes_height 20
            --FNO_n_modes_width 20
            --MNO_n_scales 4
            --MNO_scale_factors 1.0 0.75 0.5 0.25
            --LNO_n_layers 4
            --lambda_grad_l1 0.14
            --lambda_fourier_mag_l1 0.14
          )
          ;;
        curve_fault_b)
          PRESET_ARGS=(
            --use_amp
            --batch_size 32
            --epochs 220
            --top_k 2
            --hidden_channels 160
            --learning_rate 5e-5
            --weight_decay 8e-5
            --band_sharpness 28
            --freq_affinity_sharpness 14
            --FNO_n_modes_height 24
            --FNO_n_modes_width 24
            --MNO_n_scales 4
            --MNO_scale_factors 1.0 0.7 0.45 0.25
            --LNO_n_modes 20 20
            --LNO_n_layers 4
            --beta 0.6
            --lambda_grad_l1 0.24
            --lambda_fourier_mag_l1 0.18
          )
          ;;
        curve_vel_b)
          PRESET_ARGS=(
            --use_amp
            --batch_size 32
            --epochs 200
            --top_k 2
            --hidden_channels 128
            --learning_rate 7e-5
            --weight_decay 5e-5
            --band_sharpness 24
            --freq_affinity_sharpness 12
            --FNO_n_modes_height 20
            --FNO_n_modes_width 20
            --MNO_n_scales 4
            --MNO_scale_factors 1.0 0.75 0.5 0.25
            --LNO_n_layers 4
            --lambda_grad_l1 0.14
            --lambda_fourier_mag_l1 0.14
          )
          ;;
        *)
          echo "Unsupported family: ${family}" >&2
          return 1
          ;;
      esac
      ;;
    *)
      echo "Unsupported preset: ${preset} (allowed: default / preset1 / preset2)" >&2
      return 1
      ;;
  esac
}

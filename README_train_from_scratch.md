# Train from scratch workflow

Download and arrange raw data from the **OpenFWI** site as documented: **[OpenFWI dataset documentation](https://openfwi-lanl.github.io/docs/data.html)**. The project root [README.md](README.md) links Hugging Face pretrained backbones (DINOv3 ViT / ConvNeXt) and how to place files under `pretrain_weight/`.

This README describes the current seismic training pipeline in this repository: **four entry scripts** (miscellaneous `submit_*.sh` / `ddp_*.sh` wrappers were removed—use the scripts below and edit the Slurm header in `school.sh` when needed):

- `load_to_zarr.sh`: convert an OpenFWI-style directory to Zarr
- `school.sh`: submit training with `sbatch` (`#SBATCH` configured in-file)
- `school_local.sh`: run training locally without `sbatch`
- `run_from_scratch.sh`: one-shot “convert to Zarr + launch training”

## 1. Data directory layout

The raw data root must at minimum look like:

```text
XXX/
├── train_samples/
│   ├── CurveVel_A/
│   ├── CurveVel_B/
│   ├── CurveFault_A/
│   ├── CurveFault_B/
│   ├── FlatVel_A/
│   ├── FlatVel_B/
│   ├── FlatFault_A/
│   ├── FlatFault_B/
│   ├── Style_A/
│   └── Style_B/
└── test/                  # optional
```

The key requirement is that `XXX/train_samples` exists; otherwise the conversion script errors out immediately.

## 2. `family` naming

Training ultimately uses these family names:

- `curve_vel_a`
- `curve_vel_b`
- `curve_fault_a`
- `curve_fault_b`
- `flat_vel_a`
- `flat_vel_b`
- `flat_fault_a`
- `flat_fault_b`
- `style_style_a`
- `style_style_b`

For convenience, these aliases are accepted:

- `style_a` maps to `style_style_a`
- `style_b` maps to `style_style_b`

## 3. Step 1: convert to Zarr

Minimal usage:

```bash
bash load_to_zarr.sh \
  --data_dir /path/to/XXX \
  --zarr_out /path/to/curve_vel_b.zarr \
  --family curve_vel_b
```

Common extra flags:

- `--include_test 0|1`: when converting a single family, whether to include unlabeled samples from `test/` in the Zarr (default `0`)
- `--remap_single_label 0|1`: for a single family, whether to remap labels to `0` (default `0`)
- `--chunks 32`: Zarr chunk size
- `--dtype float32|float16`
- `--concat_channels 1`: by default concat `[5,1000,70]` into `[1,1000,350]`
- `--conda_env FWINO`: optional, auto-activate the Conda environment

## 4. Step 2: launch training

### Submit with `sbatch` via `school.sh`

```bash
sbatch school.sh \
  --family curve_vel_b \
  --zarr_path /path/to/curve_vel_b.zarr \
  --preset preset1
```

If your cluster requires an explicit partition or GPU reservation, override at submit time, e.g.:

```bash
sbatch --partition=gpu --gres=gpu:2 school.sh \
  --family curve_vel_b \
  --zarr_path /path/to/curve_vel_b.zarr \
  --preset preset1
```

### Run locally without `sbatch`

```bash
bash school_local.sh \
  --family curve_vel_b \
  --zarr_path /path/to/curve_vel_b.zarr \
  --preset preset1
```

To pass extra training arguments after the script’s own flags, use `--` and forward them:

```bash
bash school_local.sh \
  --family curve_vel_b \
  --zarr_path /path/to/curve_vel_b.zarr \
  --preset preset2 \
  -- --eval_interval 2 --early_stop --early_stop_patience 20
```

## 5. End-to-end one-shot

### Local from scratch

```bash
bash run_from_scratch.sh \
  --data_dir /path/to/XXX \
  --family curve_vel_b \
  --zarr_out /path/to/curve_vel_b.zarr \
  --train_mode local \
  --preset preset1
```

### From scratch with `sbatch`

```bash
bash run_from_scratch.sh \
  --data_dir /path/to/XXX \
  --family curve_vel_b \
  --zarr_out /path/to/curve_vel_b.zarr \
  --train_mode sbatch \
  --preset preset1
```

## 6. Recommended startup sequence

### Plan A: two steps

1. Convert to Zarr

```bash
bash load_to_zarr.sh \
  --data_dir /path/to/XXX \
  --zarr_out /path/to/curve_vel_b.zarr \
  --family curve_vel_b
```

2. Launch training

```bash
bash school_local.sh \
  --family curve_vel_b \
  --zarr_path /path/to/curve_vel_b.zarr \
  --preset preset1
```

### Plan B: single command

```bash
bash run_from_scratch.sh \
  --data_dir /path/to/XXX \
  --family curve_vel_b \
  --train_mode local \
  --preset preset1
```

## 7. Default training baseline

If you omit a preset, `school.sh` / `school_local.sh` only inject these essentials:

- `--mode train`
- `--num_gpus 2`
- `--num_workers 10`
- `--family`
- `--zarr_path`
- `--status_json ./dataset_status/dataset_status.json`
- `--seed 0`
- `--output_dir ./exp/runs/<family>_default_s0`

Everything else falls back to repository defaults, for example:

- `batch_size=8`
- `epochs=100`
- `top_k=1`
- `hidden_channels=128`
- `learning_rate=1e-4`
- `weight_decay=1e-4`
- `band_sharpness=20`
- `freq_affinity_sharpness=10`

## 8. Two preset candidates for six families

The lists below are **overrides** relative to defaults. `preset1` is conservative; `preset2` uses larger models and stronger regularization—often worth trying first if you have headroom—but uses more VRAM.

### `style_a` / `style_style_a`

`preset1`

- `--use_amp`
- `--batch_size 32`
- `--epochs 180`
- `--top_k 2`
- `--hidden_channels 128`
- `--learning_rate 8e-5`
- `--weight_decay 8e-5`
- `--band_sharpness 24`
- `--freq_affinity_sharpness 12`
- `--lambda_grad_l1 0.12`
- `--lambda_fourier_mag_l1 0.14`

`preset2`

- `--use_amp`
- `--batch_size 32`
- `--epochs 220`
- `--top_k 2`
- `--hidden_channels 160`
- `--learning_rate 6e-5`
- `--weight_decay 5e-5`
- `--band_sharpness 28`
- `--freq_affinity_sharpness 14`
- `--FNO_n_modes_height 20`
- `--FNO_n_modes_width 20`
- `--MNO_n_scales 4`
- `--MNO_scale_factors 1.0 0.75 0.5 0.25`
- `--LNO_n_layers 4`
- `--lambda_fourier_mag_l1 0.18`

### `style_b` / `style_style_b`

`preset1`

- `--use_amp`
- `--batch_size 32`
- `--epochs 180`
- `--top_k 2`
- `--hidden_channels 128`
- `--learning_rate 9e-5`
- `--weight_decay 8e-5`
- `--band_sharpness 22`
- `--freq_affinity_sharpness 12`
- `--lambda_grad_l1 0.10`
- `--lambda_fourier_mag_l1 0.16`

`preset2`

- `--use_amp`
- `--batch_size 32`
- `--epochs 220`
- `--top_k 2`
- `--hidden_channels 160`
- `--learning_rate 6e-5`
- `--weight_decay 5e-5`
- `--band_sharpness 26`
- `--freq_affinity_sharpness 14`
- `--FNO_n_modes_height 24`
- `--FNO_n_modes_width 24`
- `--MNO_n_scales 4`
- `--MNO_scale_factors 1.0 0.8 0.55 0.3`
- `--LNO_n_modes 20 20`
- `--LNO_n_layers 4`
- `--lambda_grad_l1 0.12`
- `--lambda_fourier_mag_l1 0.18`

### `flat_fault_b`

`preset1`

- `--use_amp`
- `--batch_size 32`
- `--epochs 180`
- `--top_k 2`
- `--hidden_channels 128`
- `--learning_rate 8e-5`
- `--weight_decay 1e-4`
- `--band_sharpness 26`
- `--freq_affinity_sharpness 13`
- `--FNO_n_modes_height 20`
- `--FNO_n_modes_width 20`
- `--lambda_grad_l1 0.20`
- `--lambda_fourier_mag_l1 0.14`

`preset2`

- `--use_amp`
- `--batch_size 32`
- `--epochs 220`
- `--top_k 2`
- `--hidden_channels 160`
- `--learning_rate 5e-5`
- `--weight_decay 8e-5`
- `--band_sharpness 30`
- `--freq_affinity_sharpness 15`
- `--FNO_n_modes_height 24`
- `--FNO_n_modes_width 24`
- `--MNO_n_scales 4`
- `--MNO_scale_factors 1.0 0.7 0.45 0.25`
- `--LNO_n_modes 20 20`
- `--LNO_n_layers 4`
- `--beta 0.6`
- `--lambda_grad_l1 0.25`
- `--lambda_fourier_mag_l1 0.18`

### `curve_vel_a`

`preset1`

- `--use_amp`
- `--batch_size 32`
- `--epochs 160`
- `--top_k 2`
- `--hidden_channels 96`
- `--learning_rate 1e-4`
- `--weight_decay 8e-5`
- `--band_sharpness 20`
- `--freq_affinity_sharpness 10`
- `--lambda_grad_l1 0.12`
- `--lambda_fourier_mag_l1 0.12`

`preset2`

- `--use_amp`
- `--batch_size 32`
- `--epochs 200`
- `--top_k 2`
- `--hidden_channels 128`
- `--learning_rate 7e-5`
- `--weight_decay 5e-5`
- `--band_sharpness 24`
- `--freq_affinity_sharpness 12`
- `--FNO_n_modes_height 20`
- `--FNO_n_modes_width 20`
- `--MNO_n_scales 4`
- `--MNO_scale_factors 1.0 0.75 0.5 0.25`
- `--LNO_n_layers 4`
- `--lambda_grad_l1 0.14`
- `--lambda_fourier_mag_l1 0.14`

### `curve_fault_b`

`preset1`

- `--use_amp`
- `--batch_size 32`
- `--epochs 180`
- `--top_k 2`
- `--hidden_channels 128`
- `--learning_rate 8e-5`
- `--weight_decay 8e-5`
- `--band_sharpness 24`
- `--freq_affinity_sharpness 12`
- `--FNO_n_modes_height 20`
- `--FNO_n_modes_width 20`
- `--lambda_grad_l1 0.20`
- `--lambda_fourier_mag_l1 0.14`

`preset2`

- `--use_amp`
- `--batch_size 32`
- `--epochs 220`
- `--top_k 2`
- `--hidden_channels 160`
- `--learning_rate 5e-5`
- `--weight_decay 8e-5`
- `--band_sharpness 28`
- `--freq_affinity_sharpness 14`
- `--FNO_n_modes_height 24`
- `--FNO_n_modes_width 24`
- `--MNO_n_scales 4`
- `--MNO_scale_factors 1.0 0.7 0.45 0.25`
- `--LNO_n_modes 20 20`
- `--LNO_n_layers 4`
- `--beta 0.6`
- `--lambda_grad_l1 0.24`
- `--lambda_fourier_mag_l1 0.18`

### `curve_vel_b`

`preset1`

- `--use_amp`
- `--batch_size 32`
- `--epochs 160`
- `--top_k 2`
- `--hidden_channels 96`
- `--learning_rate 1e-4`
- `--weight_decay 1e-4`
- `--band_sharpness 20`
- `--freq_affinity_sharpness 10`

`preset2`

- `--use_amp`
- `--batch_size 32`
- `--epochs 200`
- `--top_k 2`
- `--hidden_channels 128`
- `--learning_rate 7e-5`
- `--weight_decay 5e-5`
- `--band_sharpness 24`
- `--freq_affinity_sharpness 12`
- `--FNO_n_modes_height 20`
- `--FNO_n_modes_width 20`
- `--MNO_n_scales 4`
- `--MNO_scale_factors 1.0 0.75 0.5 0.25`
- `--LNO_n_layers 4`
- `--lambda_grad_l1 0.14`
- `--lambda_fourier_mag_l1 0.14`

## 9. Commands applying presets directly

Example for `curve_vel_b`:

```bash
bash school_local.sh \
  --family curve_vel_b \
  --zarr_path /path/to/curve_vel_b.zarr \
  --preset preset1
```

Example for `style_a`:

```bash
bash school_local.sh \
  --family style_a \
  --zarr_path /path/to/style_a.zarr \
  --preset preset2
```

## 10. Tips

- Start with `preset1` for a smoke run, then try `preset2`
- `preset2` uses more VRAM; on OOM, lower `batch_size` first
- For single-machine debugging, prefer `school_local.sh`
- For cluster production jobs, prefer `school.sh` + `sbatch`

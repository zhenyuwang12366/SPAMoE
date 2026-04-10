# 从头开始训练流程

这份 README 对应当前仓库的地震数据训练流程，包含 4 个入口：

- `load_to_zarr.sh`：把 OpenFWI 风格目录转换成 zarr
- `school.sh`：用 `sbatch` 提交训练
- `school_local.sh`：不使用 `sbatch`，直接本地启动训练
- `run_from_scratch.sh`：一键完成“转 zarr + 启动训练”

## 1. 数据目录要求

原始数据根目录必须至少满足：

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
└── test/                  # 可选
```

重点是 `XXX/train_samples` 必须存在，否则转换脚本会直接报错。

## 2. family 命名说明

训练脚本内部最终使用这些 family 名称：

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

为了兼容你的使用习惯，下面这些别名也可以直接传：

- `style_a` 会自动映射成 `style_style_a`
- `style_b` 会自动映射成 `style_style_b`

## 3. 第一步：转换成 zarr

最直接的用法：

```bash
bash load_to_zarr.sh \
  --data_dir /path/to/XXX \
  --zarr_out /path/to/curve_vel_b.zarr \
  --family curve_vel_b
```

常用附加参数：

- `--include_test 0|1`：单 family 转换时是否把 `test/` 里的无标签样本也写进去，默认 `0`
- `--remap_single_label 0|1`：单 family 时是否把标签重映射到 `0`，默认 `0`
- `--chunks 32`：zarr chunk 大小
- `--dtype float32|float16`
- `--concat_channels 1`：默认把 `[5,1000,70]` 拼成 `[1,1000,350]`
- `--conda_env FWINO`：可选，自动激活 conda 环境

## 4. 第二步：启动训练

### 用 `school.sh` 通过 `sbatch` 启动

```bash
sbatch school.sh \
  --family curve_vel_b \
  --zarr_path /path/to/curve_vel_b.zarr \
  --preset preset1
```

如果集群要求显式指定分区或 GPU 资源，直接在提交时覆盖即可，例如：

```bash
sbatch --partition=gpu --gres=gpu:2 school.sh \
  --family curve_vel_b \
  --zarr_path /path/to/curve_vel_b.zarr \
  --preset preset1
```

### 不使用 `sbatch`，直接本地启动

```bash
bash school_local.sh \
  --family curve_vel_b \
  --zarr_path /path/to/curve_vel_b.zarr \
  --preset preset1
```

如果要继续加训练参数，用 `--` 之后透传：

```bash
bash school_local.sh \
  --family curve_vel_b \
  --zarr_path /path/to/curve_vel_b.zarr \
  --preset preset2 \
  -- --eval_interval 2 --early_stop --early_stop_patience 20
```

## 5. 一键完整流程

### 本地从头跑

```bash
bash run_from_scratch.sh \
  --data_dir /path/to/XXX \
  --family curve_vel_b \
  --zarr_out /path/to/curve_vel_b.zarr \
  --train_mode local \
  --preset preset1
```

### 通过 `sbatch` 从头跑

```bash
bash run_from_scratch.sh \
  --data_dir /path/to/XXX \
  --family curve_vel_b \
  --zarr_out /path/to/curve_vel_b.zarr \
  --train_mode sbatch \
  --preset preset1
```

## 6. 开始训练的推荐步骤

### 方案 A：分两步执行

1. 转 zarr

```bash
bash load_to_zarr.sh \
  --data_dir /path/to/XXX \
  --zarr_out /path/to/curve_vel_b.zarr \
  --family curve_vel_b
```

2. 启动训练

```bash
bash school_local.sh \
  --family curve_vel_b \
  --zarr_path /path/to/curve_vel_b.zarr \
  --preset preset1
```

### 方案 B：一条命令直接跑完

```bash
bash run_from_scratch.sh \
  --data_dir /path/to/XXX \
  --family curve_vel_b \
  --train_mode local \
  --preset preset1
```

## 7. 默认训练基线

如果你不选 preset，那么 `school.sh` / `school_local.sh` 只会补这几个必要参数：

- `--mode train`
- `--num_gpus 2`
- `--num_workers 10`
- `--family`
- `--zarr_path`
- `--status_json ./dataset_status/dataset_status.json`
- `--seed 0`
- `--output_dir ./exp/runs/<family>_default_s0`

其余训练参数会回退到仓库默认值，例如：

- `batch_size=8`
- `epochs=100`
- `top_k=1`
- `hidden_channels=128`
- `learning_rate=1e-4`
- `weight_decay=1e-4`
- `band_sharpness=20`
- `freq_affinity_sharpness=10`

## 8. 六个 family 的两组候选增强配置

下面列出的都是相对于默认参数的“覆盖项”。`preset1` 偏稳妥，`preset2` 偏大模型和更强正则，通常更值得优先尝试，但显存占用也更高。

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

## 9. 直接套用 preset 的命令

例如 `curve_vel_b`：

```bash
bash school_local.sh \
  --family curve_vel_b \
  --zarr_path /path/to/curve_vel_b.zarr \
  --preset preset1
```

例如 `style_a`：

```bash
bash school_local.sh \
  --family style_a \
  --zarr_path /path/to/style_a.zarr \
  --preset preset2
```

## 10. 建议

- 先用 `preset1` 跑通，再尝试 `preset2`
- `preset2` 显存压力更大，如果 OOM，先降 `batch_size`
- 如果是本地单机调试，优先 `school_local.sh`
- 如果是集群正式训练，优先 `school.sh` + `sbatch`

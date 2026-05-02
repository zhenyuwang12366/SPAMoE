# FWINO: Full-Waveform Inversion (FWI) and Mixture-of-Experts Neural Operators

This project builds on [NeuralOperator](https://github.com/neuraloperator/neuraloperator) for **velocity model reconstruction** with neural operators on seismic datasets such as **OpenFWI**, and supports training and inference with **AFreqMoE (adaptive frequency-domain mixture of experts)**, DINOv3 **ViT / ConvNeXt** encoders, and related components.

---

## Table of Contents

1. [Project layout](#project-layout)
2. [Data and pretrained weights](#data-and-pretrained-weights)
3. [Environment setup](#environment-setup)
4. [Training workflow overview](#training-workflow-overview)
5. [Encoder backbones (timm / DINOv3)](#encoder-backbones-timm--dinov3)
6. [MoE training and scripts](#moe-training-and-scripts)
7. [Parameters and configuration](#parameters-and-configuration)
8. [Model architecture highlights](#model-architecture-highlights)
9. [Customization and extensions](#customization-and-extensions)
10. [Distributed training and evaluation](#distributed-training-and-evaluation)
11. [Further documentation](#further-documentation)

---

## Project layout

| Path | Description |
|------|-------------|
| `neuralop/` | Neural operator core: layers, models (including `afreqmoe.py`), dataloaders, trainers |
| `neuralop/models/encoder.py` | DINOv3 ViT / ConvNeXt encoders and `get_encoder()` factory |
| `scripts/train_seismic_moe.py` | Main entry point for seismic MoE training |
| `exp/run_afreqmoe_pipeline.py` | Multi-task / multi-backbone experiment orchestration |
| `config/` | Task-specific configurations |
| `openfwi/` | OpenFWI-style training entry points (e.g. `train.py`) |
| Root `*.sh` | Startup scripts kept for the seismic Zarr pipeline only (see table below) |
| `scripts/*.sh` | Distributed training wrappers for seismic / PDE workloads |

### Launcher scripts (repository slimmed)

The root directory only keeps scripts directly tied to **OpenFWI → Zarr → training**. Historical cluster-specific wrappers such as many `submit_*.sh` / `ddp_*.sh` files have been removed; adjust `#SBATCH` lines in `school.sh`, or call the same commands as `school_local.sh` directly with `sbatch`.

| Script | Role |
|------|------|
| `load_to_zarr.sh` | Invokes `load_to_zarr.py` to convert a raw directory with `train_samples/` into Zarr |
| `school_local.sh` | Local or interactive multi-GPU: parses args and presets, calls `scripts/run_distributed_seismic_moe.sh` |
| `school.sh` | Thin Slurm wrapper with `#SBATCH`; forwards to `school_local.sh` (usage: `sbatch school.sh --family ... --zarr_path ...`) |
| `run_from_scratch.sh` | Runs `load_to_zarr.sh`, then `school_local.sh` or `sbatch school.sh` |
| `training_presets.sh` | **Sourced** by `school_local.sh`; supplies `preset1` / `preset2` hyperparameter bundles (not run standalone) |
| `scripts/bash_helpers.sh` | **Sourced** by the above (conda activation, `normalize_family_name`, etc.) |
| `scripts/run_distributed_seismic_moe.sh` | Main seismic MoE / Zarr entry; launches `torchrun` + `train_seismic_moe.py` |
| `scripts/run_distributed_train_pde.sh` | Distributed PDE training (`pde/train_pde.py`), independent of the seismic pipeline |

**Marmousi-style inference**: With no dedicated `.sh`, you can run `python scripts/infer_marmousi2.py` directly (see `--help` in the script).

---

## Data and pretrained weights

### OpenFWI dataset

Official descriptions and download links (Vel / Fault / Style / Kimberlina families—input/output shapes and sample counts are documented):

**[OpenFWI dataset documentation](https://openfwi-lanl.github.io/docs/data.html)**

Typical 2D subsets: **inputs** roughly `(5, 1000, 70)` (sources × time × traces), **outputs** roughly `(1, 70, 70)` velocity models, aligned with the Zarr / waveform pipeline in this repository.

### Pretrained backbones (Hugging Face → timm)

Encoders default to **DINOv3 distilled weights** (LVD-1689M). Model cards and weight files:

| Backbone | Hugging Face model page |
|------|----------------------|
| ViT-S/16 DINOv3 | [timm/vit_small_patch16_dinov3.lvd1689m](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m) |
| ConvNeXt-Tiny DINOv3 | [timm/convnext_tiny.dinov3_lvd1689m](https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m) |

**Local weights (recommended for offline or pinned versions)**: Place the corresponding `.safetensors` under `pretrain_weight/` at the project root; filenames must match what the code expects (`get_encoder()` prefers local files when present):

- `pretrain_weight/vit_small_patch16_dinov3.lvd1689m.safetensors`
- `pretrain_weight/convnext_tiny.dinov3_lvd1689m.safetensors`

If these files are missing, loading falls back to `timm.create_model(..., pretrained=True)` pulling from Hugging Face (network and `huggingface_hub` required).

---

## Environment setup

### Requirements

- **Python 3.10+** (matches `environment.yml` and `pyproject.toml`)
- **CUDA** and **PyTorch** versions matched to your GPU (`requirements.txt` shows CUDA 12.x examples as a guide)
- Sufficient disk space for datasets and Zarr caches

### Installation

```bash
git clone <your-repo-url> FWINO_wzy
cd FWINO_wzy
```

Conda is recommended (example env name `FWINO`):

```bash
conda env create -f environment.yml
conda activate FWINO
pip install -r requirements.txt
```

Or directly:

```bash
pip install -r requirements.txt
```

Typical dependencies include `torch`, `timm`, `zarr`, `h5py`, `wandb`, `tensorly`, `tensorly-torch`, etc. (see `requirements.txt`).

---

## Training workflow overview

### Option A: OpenFWI directory → Zarr → training (recommended for current repo scripts)

Full steps, `family` naming, presets, and usage of `load_to_zarr.sh` / `school_local.sh` / `run_from_scratch.sh`:

**[README_train_from_scratch.md](README_train_from_scratch.md)**

The data root must contain `train_samples/` with subfolders such as `CurveVel_A`, `FlatFault_B`, `Style_A`, etc., consistent with OpenFWI releases.

### Option B: PDE distributed (not seismic-specific)

```bash
bash scripts/run_distributed_train_pde.sh --help
```

### Option C: Call the seismic MoE training script directly

Single-GPU example (replace data paths):

```bash
python scripts/train_seismic_moe.py \
    --data_dir /path/to/your/data \
    --family all \
    --batch_size 8 \
    --epochs 100 \
    --output_dir ./results/seismic_moe
```

For distributed runs:

```bash
bash scripts/run_distributed_seismic_moe.sh \
    --num_gpus 4 \
    --data_dir /path/to/your/data \
    --family all \
    --batch_size 16 \
    --epochs 100 \
    --output_dir ./results/distributed_seismic_moe
```

On clusters with Slurm, **`sbatch school.sh`** is recommended (edit partition, GPU count, and other `#SBATCH` directives at the top of `school.sh`), consistent with the Zarr workflow docs.

---

## Encoder backbones (timm / DINOv3)

Training scripts choose the backbone via CLI flags (exact names follow `choices` in `utils/parser_utils.py`), for example:

- `vit` → `vit_small_patch16_dinov3.lvd1689m`
- `convnext_tiny` → `convnext_tiny.dinov3_lvd1689m`

Implementation: `Encoder_Dino`, `Encoder_ConvNeXt`, and **`get_encoder()`** in `neuralop/models/encoder.py`.

---

## MoE training and scripts

- **Frozen experts**: When `--use_moe` and `--use_experts_path` are set, `--top_k > 1`, and multiple `--choose_experts` are selected, loaded expert weights are frozen by design; otherwise experts are typically trained from scratch.
- **Expert checkpoint directory**: Place each expert’s `.pt` in one directory and pass that path via `--use_experts_path`.
- **Experiment orchestration**: `python exp/run_afreqmoe_pipeline.py` batches seismic / PDE tasks, router types, `top_k`, `backbone`, etc. (see in-script docs and `--help`).

### Inference example

```bash
python scripts/train_seismic_moe.py \
    --mode inference \
    --model_path ./results/seismic_moe/best_model.pt \
    --data_dir /path/to/test/data \
    --output_dir ./results/predictions
```

---

## Parameters and configuration

### Common CLI flags (excerpt)

| Flag | Description |
|------|------|
| `--mode` | `train` / `inference` |
| `--data_dir` | Dataset root directory |
| `--family` | Data subgroup, e.g. `vel`, `fault`, `style`, `all`, or finer names like `curve_vel_b` (must match layout) |
| `--batch_size`, `--epochs` | Batch size and number of epochs |
| `--top_k`, `--choose_experts` | MoE routing: in `choose_experts`, `0=FNO, 1=WNO, 2=MNO, 3=LNO` (confirm in current scripts) |
| `--use_moe`, `--use_experts_path` | Enable MoE and expert checkpoint directory |
| `--FNO_n_modes_height/width`, `--MNO_n_scales`, ... | Per-expert architecture hyperparameters |

A fuller table and defaults documented in **`config/seismic_moe_config.py`** appear in the “Work split and naming” section and historical docs.

### MoE-related extended parameters (`MOEOperator`, etc.)

- `router_type`: e.g. `basic`, `adamv`, and other adaptive routing policies  
- `fusion_type`: `linear`, `attention`, `swa`, etc., for merging expert outputs  
- `s_processor_type` / `w_processor_type`: fusion blocks for strong / weak expert outputs  
- `beta`: strength of weak-activation influence  
- `is_specific`: whether checkpoint names use fine-grained subclasses (curve/flat/style and A/B)  
- `is_classifier`: whether to use grouped expert networks (GMoE)

---

## Model architecture highlights

The seismic MoE model follows a **mixture-of-experts neural operator** design; main pieces:

1. **Router**: Assigns inputs to experts; may include task-aware routing (e.g. `TaskAwareRouter`).  
2. **Experts**: Defaults include FNO, WNO, MNO, LNO as frequency / multiscale / local experts.  
3. **Fusion**: Merges expert outputs via linear layers, attention, etc.  
4. **Encoder**: Optional DINOv3 ViT or ConvNeXt mapping observed waveforms to representations for downstream operators (see `encoder.py`).

---

## Customization and extensions

- **Expert hyperparameters**: Prefer CLI overrides (see table above); for complex combos edit `expert_configs` in `config/seismic_moe_config.py`.  
- **Loss or optimizer**: Adjust `criterion`, `optimizer`, `scheduler` in `scripts/train_seismic_moe.py`.  
- **Data pipeline**: Custom data should match dimensions and keys expected by `SeismicDataset` / `SeismicDataProcessor`; Zarr details in `neuralop/data/dataloader/zarr_seismic_dataloader.py` and `README_train_from_scratch.md`.

---

## Distributed training and evaluation

### Distributed tips

- Scale `batch_size` and `num_workers` appropriately  
- Enable mixed precision when supported (e.g. `--use_amp` if present in the current `train_seismic_moe.py`)

### Validation and metrics

During training, `train_seismic_moe.py` and `utils/train_process.py` compute validation loss and related metrics; for standalone inference use `--mode inference` with `--model_path` (see `run_inference` and argparse in the script). The repo also has `openfwi/train.py` and other standalone entry points for experiments.

---

## Further documentation

- **[README_train_from_scratch.md](README_train_from_scratch.md)**: Zarr conversion, `family` aliases, `preset1`/`preset2`, Slurm vs local one-shot workflows.  
- **OpenFWI**: [dataset page](https://openfwi-lanl.github.io/docs/data.html).  
- **Pretrained backbones**: [ViT-S/16 DINOv3](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m), [ConvNeXt-Tiny DINOv3](https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m).

---

## Work split and checkpoint naming (team notes)

**Phase one**: Single-expert behavior and characterization (debugging division of labor among WNO / MNO / FNO / LNO), aiming for stable runs near public baselines.

**Checkpoint naming**

- **normal**: Three coarse families `vel` / `fault` / `style`: `best_expert_{experts_name}_{i}_{vel|fault|style}.pt`  
- **specific**: Finer splits `curve_vel`, `curve_fault`, `flat_vel`, `flat_fault`, `style` with `_a`/`_b` suffixes where applicable: `best_expert_{experts_name}_{i}_{curve|flat|style}_{vel|fault|style}.pt`

---

## Dataset directory example (legacy / NumPy workflow)

Some scripts expect a layout like this when not using Zarr (otherwise follow the actual `SeismicDataset` contract):

```text
data_dir/
├── train_samples/
│   └── <subset_name>/
│       ├── model/model{i}.npy    # velocity model
│       └── data/data{i}.npy      # seismic waveforms
└── test/
```

Typical tensor shapes: waveforms `[B, 5, 1000, 70]`, velocity `[B, 70, 70]` or `[B, 1, 70, 70]`.

---

## License and acknowledgments

- Follow **OpenFWI** and LANL licensing/terms for datasets.  
- Follow each model card’s **License** on Hugging Face for DINOv3 / timm weights (e.g. DINOv3 license).  
- Neural operators and scientific stack: respect each dependency’s open-source license.

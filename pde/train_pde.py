#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_pdebench_emo_afmoe.py

PDEBench training script skeleton adapted from the existing FWI EMO + MoE stack:
- Multi-task: Burgers (1D), Navier-Stokes (2D), Darcy (2D)
- When router_type = 'sar', use AF-MoE (AdaptiveFreqMoE)
- Other router_type values use the original MOEOperator
"""

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ======= Project modules (names assumed here) =======
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.distributed import DistributedConfig
import neuralop.mpu.comm as comm
from config.default_config import Default
from neuralop.training import setup
from neuralop.models.EMO import EMO
from neuralop.models.moe import MOEOperator
from neuralop.models.afreqmoe import AdaptiveFreqMoE  # AF-MoE implementation
from neuralop.models.expert_factory import ExpertFactory
from neuralop.data.datasets.pde_dataset import PDEBenchDataset   # implement or adapt as needed
from neuralop.layers.spectral_convolution import SpectralConv
import scripts.transforms as T
from torchvision.transforms import Compose
from neuralop.losses.seismic_loss import PDECombinedLoss  # includes RelativeL2 + optional Grad/Fourier
from scripts.scheduler import (
    WarmupMultiStepLR,
    WarmupCosineLR,
    WarmupCosineAnnealingWarmRestarts,
)
from utils import (
    visualize_router_selection_from_stats,
    visualize_routed_bands,
    SeismicMetrics,
    patch_spectral_conv_forward,
)


patch_spectral_conv_forward(SpectralConv)


PhysicsMetrics = SeismicMetrics  # reuse seismic metrics helpers for PDE tasks
# ===========================================================
# 1. Config: same style as SeismicMOEConfig, simplified
# ===========================================================

@dataclass
class PDEBenchConfig(Default):
    # ---- General network hyperparameters ----
    in_channels: int = 1
    out_channels: int = 1
    hidden_channels: int = 128
    backbone: str = "vit"
    use_encoder: bool = True
    v_type_num: int = 1               # PDE tasks usually need a single type; keep at 1

    # ---- MoE & Router ----
    top_k: int = 2
    noisy_gating: bool = False
    fusion_type: str = "linear"
    router_hidden_dim: int = 128
    router_type: str = "sar"          # 'basic' or 'sar'
    s_processor_type: str = "linear"
    w_processor_type: str = "linear"
    beta: float = 0.5
    band_sharpness: float = 20.0
    freq_affinity_sharpness: float = 10.0
    use_soft_bands: bool = True
    enable_freq_attn: bool = True
    enable_band_mixing: bool = True
    is_specific: bool = False
    is_classifier: bool = False
    use_gpu_proxy: bool = False
    moe_mode: str = "standard"
    aux_loss_weight: float = 0.1

    # ---- Training ----
    batch_size: int = 16
    test_batch_size: int = 16
    epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cos_restart"  # multistep / cos / cos_restart / none
    milestones: tuple = (60, 90, 110)
    scheduler_gamma: float = 0.3
    lr_warmup_epochs: float = 5.0
    lr_warmup_factor: float = 1.0 / 3
    lr_warmup_method: str = "linear"
    lr_cosine_eta_min: float = 1e-6
    lr_cosine_tmax_epochs: int = 50
    lr_cosine_restart_t0_epochs: int = 10
    lr_cosine_restart_t_mult: int = 2
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    seed: int = 42
    num_workers: int = 4
    
    # Distributed training
    distributed = DistributedConfig(
        use_distributed=False,
        model_parallel_size=1,
        seed=42
    )
    is_logger = False

    # ---- Tasks & data ----
    # PDEBench-style task names: navier/darcy/pipe/airfoil/plasticity
    task: str = "darcy"
    data_root: str = "./pdebench_data"
    save_dir: str = "./results_pdebench"
    log_every: int = 50
    vis_every: int = 100
    k: float = 1.0
    
    # Expert configurations
    expert_configs = [
        # Fourier-domain expert — frequency structure (FNO)
        {
            'type': 'domain',
            'domain_type': 'fourier',
            'n_dim': 2,
            'n_modes_height': 16,
            'n_modes_width': 16,
            'lifting_channel_ratio': 2,
            'projection_channel_ratio': 2,
            'n_layers': 4,
        },
        # Native multiscale neural operator expert — multiscale structure (MNO)
        {
            'type': 'scale',
            'scale_expert_type': 'native',  # use scale_expert_type
            'n_dim': 2,
            'n_scales': 3,
            'scale_factors': [1.0, 0.6, 0.3],
            'fusion_mode': 'hierarchical',
            'n_layers': 4,
        },
        # Local expert — spatial detail reconstruction (LNO)
        {
            'type': 'local',
            'local_type': 'basic',  # basic local type
            'n_dim': 2,
            'n_modes': (16, 16),
            'disco_layers': True,  # enable DISCO layers
            'diff_layers': True,   # enable finite-difference layers
            'n_layers': 3,         # layer count
            'default_in_shape': (70, 70),  # placeholder; overwritten from data later
            'domain_length': [2, 2],
        },
        # # Wavelet-domain expert — local features and multiscale (WNO)
        # {
        #     'type': 'domain',
        #     'domain_type': 'wavelet',
        #     'n_dim': 2,
        #     'n_levels_height': 2,  # fewer levels to avoid shape mismatch
        #     'n_levels_width': 2,   # fewer levels to avoid shape mismatch
        #     'conv_kind': 'dwt',
        #     'wavelet': 'db6',
        #     'biort': 'near_sym_b',
        #     'qshift': 'qshift_b',
        #     'n_layers': 4,
        #     'dropout_rate': 0.10,
        #     'base_size': (70, 70),
        # },
    ]


# ===========================================================
# 2. Argparse
# ===========================================================

import argparse

def build_argparser():
    parser = argparse.ArgumentParser("PDEBench EMO + (AF-)MoE training script")

    # Task & data
    parser.add_argument("--task", type=str, default="darcy",
                        choices=["navier", "darcy", "pipe", "airfoil", "plasticity"])
    parser.add_argument("--data_root", type=str, default="./pdebench_data")
    parser.add_argument("--save_dir", type=str, default="./results_pdebench")
    parser.add_argument("--status_json", type=str, default="./pde_status.json")
    
    # Training controls
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--amp_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--num_workers", type=int, default=4)

    # MoE / Router
    parser.add_argument("--router_type", type=str, default="sar",
                        choices=["basic", "sar"])
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--backbone", type=str, default="vit",
                        choices=["vit", "convnext_tiny"])
    parser.add_argument("--aux_loss_weight", type=float, default=0.1,
                        help="Coefficient for aux loss balancing in routers.")
    parser.add_argument("--band_sharpness", type=float, default=20.0,
                        help="AFreqMoE soft band sharpness (higher ~= harder band splits).")
    parser.add_argument("--freq_affinity_sharpness", type=float, default=10.0,
                        help="Sharpness matching expert freq preference to band centers.")
    parser.add_argument("--disable_soft_bands", action="store_true",
                        help="Ablation: disable soft bands (use hard splits).")
    parser.add_argument("--disable_freq_attn", action="store_true",
                        help="Ablation: disable frequency-domain self-attention.")
    parser.add_argument("--disable_band_mixing", action="store_true",
                        help="Ablation: disable band mixing (experts receive only their band).")
    parser.add_argument("--resume_path", type=str, default=None,
                        help="Checkpoint path to resume training from.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cos_restart",
                        choices=["multistep", "cos", "cos_restart", "none"])
    parser.add_argument("--milestones", type=int, nargs="+", default=[60, 90, 110])
    parser.add_argument("--scheduler_gamma", type=float, default=0.3)
    parser.add_argument("--lr_warmup_epochs", type=float, default=5.0)
    parser.add_argument("--lr_warmup_factor", type=float, default=1.0 / 3)
    parser.add_argument("--lr_warmup_method", type=str, default="linear",
                        choices=["linear", "constant"])
    parser.add_argument("--lr_cosine_eta_min", type=float, default=1e-6)
    parser.add_argument("--lr_cosine_tmax_epochs", type=int, default=50)
    parser.add_argument("--lr_cosine_restart_t0_epochs", type=int, default=10)
    parser.add_argument("--lr_cosine_restart_t_mult", type=int, default=2)

    # DDP
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)

    # Visualization / logging cadence
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--vis_every", type=int, default=200)
    parser.add_argument("--vis_router_every", type=int, default=400)

    return parser


# ===========================================================
# 3. DDP init & utilities
# ===========================================================

def is_main_process(cfg):
    return cfg.is_logger

def get_amp_dtype(name: Optional[str]) -> torch.dtype:
    if not name:
        return torch.bfloat16
    norm = name.lower()
    if norm in ("float16", "fp16", "half"):
        return torch.float16
    return torch.bfloat16


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
        return model.module
    return model


# ===========================================================
# 3.5 Auto-compute radius_cutoff so DISCO output matches input spatial size
# ===========================================================

def auto_radius_cutoff_same_size(
    in_shape,
    domain_length=(2.0, 2.0),
    target_support: int = 3,
) -> float:
    """
    From EquidistantDiscreteContinuousConv2d:
        psi_local_h = floor(2 * radius_cutoff * H / Lx) + 1
        psi_local_w = floor(2 * radius_cutoff * W / Ly) + 1

    Choose radius_cutoff so that:
        psi_local_h = psi_local_w = target_support (odd values 3, 5, 7, ... recommended)
    so that with stride=1 and same padding the spatial size is unchanged.
    """
    H, W = in_shape
    Lx, Ly = float(domain_length[0]), float(domain_length[1])
    assert target_support % 2 == 1, "target_support must be odd (3, 5, 7, ...)"

    # A, B are coefficients in H, W, domain_length
    A = 2.0 * H / max(Lx, 1e-6)
    B = 2.0 * W / max(Ly, 1e-6)
    k = float(target_support)

    # Require:
    #   k-1 ≤ radius*A < k
    #   k-1 ≤ radius*B < k
    # → intersection of the two radius intervals:
    low = max((k - 1.0) / A, (k - 1.0) / B)
    high = min(k / A, k / B)

    if low >= high:
        # Fallback in edge cases
        radius = low
    else:
        # Midpoint of the intersection to avoid boundary issues
        radius = 0.5 * (low + high)

    return float(radius)


# ===========================================================
# 4. Datasets (PDEBench)
# ===========================================================

def build_pdebench_dataloaders(cfg: PDEBenchConfig, args):
    """
    Assumes PDEBenchDataset is implemented as:
      PDEBenchDataset(task, root, split, transform=None)
    Each sample returns:
      {"input": tensor [C_in, H, W] or [C_in, T, X],
       "output": tensor [C_out, H, W] or same layout}
    Adapt channels/dims inside the dataset by task as needed.
    """
    status_path = Path(args.status_json).expanduser()
    if not status_path.exists():
        raise FileNotFoundError(
            f"PDEBench normalization stats file not found: {status_path}. "
            "Generate *_stats.json first (e.g. via convert_pdebench_to_emo_format.py)."
        )

    k_value = float(getattr(cfg, "k", 1.0))
    with open(status_path, "r", encoding="utf-8") as f:
        stats_dict = json.load(f)

    input_min = T.log_transform(stats_dict["input_min"], k=k_value)
    input_max = T.log_transform(stats_dict["input_max"], k=k_value)
    output_min = stats_dict["output_min"]
    output_max = stats_dict["output_max"]

    input_transform = Compose(
        [
            T.LogTransform(k=k_value),
            T.MinMaxNormalize(input_min, input_max),
        ]
    )
    output_transform = Compose(
        [
            T.MinMaxNormalize(output_min, output_max),
        ]
    )
    input_inverse_transform = Compose(
        [
            T.InverseMinMaxNormalize(input_min, input_max),
            T.InverseLogTransform(k=k_value),
        ]
    )
    output_inverse_transform = Compose(
        [
            T.InverseMinMaxNormalize(output_min, output_max),
        ]
    )

    cfg.input_inverse_transform = input_inverse_transform
    cfg.output_inverse_transform = output_inverse_transform

    train_set = PDEBenchDataset(
        task=cfg.task,
        root=cfg.data_root,
        split="train",
        input_transform=input_transform,
        output_transform=output_transform
    )
    val_set = PDEBenchDataset(
        task=cfg.task,
        root=cfg.data_root,
        split="val",
        input_transform=input_transform,
        output_transform=output_transform,
    )

    if args.distributed and args.world_size > 1:
        train_sampler = DistributedSampler(
            train_set, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_set, num_replicas=args.world_size, rank=args.rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    num_workers = cfg.num_workers
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, train_sampler, val_sampler


# ===========================================================
# 5. Build EMO + MoE / AF-MoE
# ===========================================================

def build_emo_model(cfg: PDEBenchConfig, device):
    """
    Keep the same high-level interface as the FWI stack:
      - Encoder: maps PDE field -> latent feature map
      - MoE / AF-MoE: operator on the latent
      - EMO: wraps encoder + MoE
    """

    # 1) Encoder (same pattern as get_encoder elsewhere)
    from neuralop.models.encoder import get_encoder

    encoder_model = None
    if cfg.use_encoder:
        num_types = 1  # PDE tasks typically have no multi-type head
        encoder_model = get_encoder(
            in_channels=cfg.in_channels,
            out_channels=cfg.hidden_channels,
            num_types=num_types,
            type_act="identity",
            backbone=cfg.backbone,
            img_size=cfg.img_size,
        ).to(device)

    experts = ExpertFactory.create_expert_ensemble(
        expert_configs=cfg.expert_configs,
        in_channels=cfg.hidden_channels if cfg.use_encoder else cfg.in_channels,
        out_channels=cfg.out_channels,
        hidden_channels=cfg.hidden_channels,
    )

    # 3) MoE or AF-MoE
    moe_in_channels = cfg.hidden_channels if cfg.use_encoder else cfg.in_channels

    if cfg.router_type == "sar":
        expert_count = len(experts)
        aux_alpha = 0.0
        if cfg.top_k > 0 and expert_count > 1:
            base_alpha = max(0.0, getattr(cfg, "aux_loss_weight", 0.0))
            load_balance_scale = min(1.0, cfg.top_k / float(expert_count))
            aux_alpha = base_alpha * load_balance_scale

        # AdaptiveFreqMoE with SpectralAttentionRouter
        moe = AdaptiveFreqMoE(
            experts=experts,
            in_channels=moe_in_channels,
            topk=cfg.top_k,
            alpha=aux_alpha,
            band_sharpness=getattr(cfg, "band_sharpness", 20.0),
            freq_affinity_sharpness=getattr(cfg, "freq_affinity_sharpness", 10.0),
            use_soft_bands=getattr(cfg, "use_soft_bands", True),
            enable_freq_attn=getattr(cfg, "enable_freq_attn", True),
            enable_band_mixing=getattr(cfg, "enable_band_mixing", True),
        )
    else:
        # Original MOEOperator (basic / velocity_type / group, etc.)
        moe = MOEOperator(
            experts=experts,
            in_channels=moe_in_channels,
            out_channels=cfg.out_channels,
            hidden_channels=cfg.hidden_channels,
            top_k=cfg.top_k,
            noisy_gating=cfg.noisy_gating,
            fusion_type=cfg.fusion_type,
            router_hidden_dim=cfg.router_hidden_dim,
            moe_mode=cfg.moe_mode,
            is_logger=True,
            router_type=cfg.router_type,
            s_processor_type=cfg.s_processor_type,
            w_processor_type=cfg.w_processor_type,
            beta=cfg.beta,
            is_specific=cfg.is_specific,
            is_classifier=cfg.is_classifier,
            batch_size=cfg.batch_size,
            v_type_num=cfg.v_type_num,
            use_expert_memory_proxy=cfg.use_gpu_proxy,
            use_encoder=cfg.use_encoder,
            device=device,
        )

    emo = EMO(encoder_model, moe, pass_encoder_logits_as_weights=True).to(device)
    return emo


# ===========================================================
# 6. Train & validation
# ===========================================================

def train_one_epoch(
    emo,
    optimizer,
    train_loader,
    device,
    metrics,
    scaler,
    cfg,
    args,
    epoch,
    criterion,
    lr_scheduler=None,
    scheduler_step_mode: str = "per_step",
):
    emo.train()
    total_loss = 0.0
    mse_sum = mae_sum = 0.0
    count = 0

    amp_enabled = scaler is not None and scaler.is_enabled()
    amp_dtype = get_amp_dtype(cfg.amp_dtype)

    for step, batch in enumerate(train_loader):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["output"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        preds, aux_loss, _ = emo(inputs, use_amp=amp_enabled, amp_dtype=amp_dtype)

        # ---- Main PDE loss: PDECombinedLoss (RelativeL2 + optional Grad/Fourier) ----
        loss_dict = criterion(preds, targets)
        loss_main = loss_dict["loss"]
        aux = aux_loss if aux_loss is not None else loss_main.new_zeros(())
        loss = loss_main + aux

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if lr_scheduler is not None and scheduler_step_mode == "per_step":
            lr_scheduler.step()

        total_loss += float(loss_main.detach().item())
        # MSE/MAE for comparable baselines
        mse_sum += metrics.calculate_mse(preds, targets)
        mae_sum += metrics.calculate_mae(preds, targets)
        count += 1

        if is_main_process(cfg) and (step % cfg.log_every == 0):
            print(
                f"[Train] Epoch {epoch} Step {step}/{len(train_loader)} "
                f"Loss={total_loss/max(count,1):.6e} "
                f"MSE={mse_sum/max(count,1):.6e} "
                f"MAE={mae_sum/max(count,1):.6e}"
            )

    if lr_scheduler is not None and scheduler_step_mode == "per_epoch":
        lr_scheduler.step()

    return total_loss / max(count, 1), mse_sum / max(count, 1), mae_sum / max(count, 1)


@torch.no_grad()
def evaluate(emo, val_loader, device, metrics, cfg, args, epoch):
    emo.eval()
    mse_sum = mae_sum = psnr_sum = rmse_sum = ssim_sum = l2r_sum = 0.0
    count = 0

    sample_inputs = sample_targets = sample_preds = sample_encoded = None
    amp_enabled = getattr(cfg, "amp_enabled", cfg.use_amp and device.type == "cuda")
    amp_dtype = get_amp_dtype(cfg.amp_dtype)
    emo_module = unwrap_model(emo)

    for step, batch in enumerate(val_loader):
        inputs  = batch["input"].to(device, non_blocking=True)
        targets = batch["output"].to(device, non_blocking=True).float()

        preds, aux_loss, enc_weights = emo(inputs, use_amp=amp_enabled, amp_dtype=amp_dtype)
        preds = preds.float()

        mse_sum  += metrics.calculate_mse(preds, targets)
        mae_sum  += metrics.calculate_mae(preds, targets)
        psnr_sum += metrics.calculate_psnr(preds, targets)
        rmse_sum += metrics.calculate_rmse(preds, targets)
        ssim_sum += metrics.calculate_ssim(preds, targets)
        l2r_sum  += metrics.calculate_relative_l2(preds, targets)
        count    += 1

        # Keep first batch for visualization in downstream scripts
        if sample_inputs is None:
            sample_inputs  = inputs.detach().cpu()
            sample_targets = targets.detach().cpu()
            sample_preds   = preds.detach().cpu()
            if emo_module.encoder is not None:
                encoded, _, _ = emo_module.encoder(inputs)
                sample_encoded = encoded.detach().cpu()

    if count == 0:
        return 0, 0, 0, 0, 0, 0, None

    # all_reduce stats across ranks
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        stats_tensor = torch.tensor(
            [mse_sum, mae_sum, psnr_sum, rmse_sum, ssim_sum, l2r_sum, float(count)],
            device=device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(stats_tensor, op=torch.distributed.ReduceOp.SUM)
        mse_sum, mae_sum, psnr_sum, rmse_sum, ssim_sum, l2r_sum, count = stats_tensor.tolist()

    # count is float after all_reduce; guard division
    denom = max(count, 1.0)

    mse  = mse_sum / denom
    mae  = mae_sum / denom
    psnr = psnr_sum / denom
    rmse = rmse_sum / denom
    ssim = ssim_sum / denom
    l2r  = l2r_sum / denom

    if is_main_process(cfg):
        print(
            f"[Eval] Epoch {epoch} | "
            f"MSE={mse:.6e} MAE={mae:.6e} "
            f"PSNR={psnr:.4f} RMSE={rmse:.6e} "
            f"SSIM={ssim:.4f} L2RE={l2r}"
        )

    return mse, mae, psnr, rmse, ssim, l2r, {
        "inputs": sample_inputs,
        "targets": sample_targets,
        "preds": sample_preds,
        "encoded": sample_encoded,
    }


# ===========================================================
# 7. main
# ===========================================================

def main():
    parser = build_argparser()
    args = parser.parse_args()
    # Training: metrics only; skip visualization hooks
    args.vis_every = 0
    args.vis_router_every = 0
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Build config
    cfg = PDEBenchConfig(
        task=args.task,
        data_root=args.data_root,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        seed=args.seed,
        num_workers=args.num_workers,
        router_type=args.router_type,
        top_k=args.top_k,
        hidden_channels=args.hidden_channels,
        backbone=args.backbone,
        band_sharpness=args.band_sharpness,
        freq_affinity_sharpness=args.freq_affinity_sharpness,
        use_soft_bands=not args.disable_soft_bands,
        enable_freq_attn=not args.disable_freq_attn,
        enable_band_mixing=not args.disable_band_mixing,
        log_every=args.log_every,
        vis_every=args.vis_every,
        aux_loss_weight=args.aux_loss_weight,
        lr_scheduler_type=args.lr_scheduler_type,
        milestones=tuple(args.milestones),
        scheduler_gamma=args.scheduler_gamma,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_warmup_factor=args.lr_warmup_factor,
        lr_warmup_method=args.lr_warmup_method,
        lr_cosine_eta_min=args.lr_cosine_eta_min,
        lr_cosine_tmax_epochs=args.lr_cosine_tmax_epochs,
        lr_cosine_restart_t0_epochs=args.lr_cosine_restart_t0_epochs,
        lr_cosine_restart_t_mult=args.lr_cosine_restart_t_mult,
    )

    if args.distributed:
        cfg.distributed.use_distributed = True 
    device, is_logger = setup(cfg)
    cfg.is_logger = is_logger
    
    # Sync rank / world_size
    args.rank = comm.get_global_rank()
    args.world_size = comm.get_world_size()
    args.local_rank = comm.get_local_rank()

    if is_main_process(cfg):
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.save_dir) / "pdebench_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)

    # === Data ===
    train_loader, val_loader, train_sampler, val_sampler = build_pdebench_dataloaders(cfg, args)

    # === Model ===
    # Infer in/out channels
    try:
        sample_batch = train_loader.dataset[0]
    except Exception:
        sample_batch = next(iter(train_loader))
    cfg.in_channels = int(sample_batch["input"].shape[0])
    cfg.out_channels = int(sample_batch["output"].shape[0])
    
    # fno 0, mno 1, lno 2
    cfg.img_size = tuple(sample_batch["output"].shape[-2:])
    # cfg.expert_configs[1]["base_size"] = cfg.img_size
    cfg.expert_configs[2]["default_in_shape"] = cfg.img_size

    # Auto radius_cutoff for LNO/DISCO from img_size + domain_length
    lno_cfg = cfg.expert_configs[2]
    domain_length = lno_cfg.get("domain_length", [2, 2])
    radius_cutoff = auto_radius_cutoff_same_size(
        in_shape=cfg.img_size,
        domain_length=(float(domain_length[0]), float(domain_length[1])),
        target_support=3,   # try 5/7 for a larger DISCO receptive field
    )
    lno_cfg["radius_cutoff"] = radius_cutoff

    if is_main_process(cfg):
        print(
            f"[LNO/DISCO] img_size={cfg.img_size}, domain_length={domain_length}, "
            f"auto radius_cutoff={radius_cutoff:.6f}"
        )

    emo = build_emo_model(cfg, device)

    cfg.has_complex_params = any(torch.is_complex(p) for p in emo.parameters())
    cfg.amp_enabled = cfg.use_amp and device.type == "cuda" and not cfg.has_complex_params
    if cfg.use_amp and cfg.has_complex_params and is_main_process(cfg):
        print("[AMP] Detected complex parameters (e.g., Fourier-domain weights); disabling AMP/GradScaler.")
    
    if args.distributed and args.world_size > 1:
        emo = nn.parallel.DistributedDataParallel(
            emo, 
            device_ids=[args.local_rank], 
            output_device=args.local_rank, 
            find_unused_parameters=True,
            static_graph=False,
            gradient_as_bucket_view=True,
        )

    # === Optimizer ===
    optimizer = torch.optim.AdamW(
        emo.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader))
    warmup_iters = int(max(0.0, getattr(cfg, "lr_warmup_epochs", 0.0)) * steps_per_epoch)
    warmup_kwargs = dict(
        warmup_factor=getattr(cfg, "lr_warmup_factor", 1.0 / 3),
        warmup_iters=warmup_iters,
        warmup_method=getattr(cfg, "lr_warmup_method", "linear"),
    )
    scheduler_type = str(getattr(cfg, "lr_scheduler_type", "cos_restart") or "none").lower()
    lr_scheduler = None
    scheduler_step_mode = "per_step"
    if scheduler_type == "multistep":
        lr_milestones = [int(steps_per_epoch * m) for m in getattr(cfg, "milestones", ())]
        lr_scheduler = WarmupMultiStepLR(
            optimizer=optimizer,
            milestones=lr_milestones,
            gamma=getattr(cfg, "scheduler_gamma", 0.1),
            **warmup_kwargs,
        )
    elif scheduler_type == "cos":
        t_max_epochs = getattr(
            cfg,
            "lr_cosine_tmax_epochs",
            max(1, getattr(cfg, "epochs", 100) - getattr(cfg, "lr_warmup_epochs", 0)),
        )
        t_max = max(1, int(t_max_epochs * steps_per_epoch))
        lr_scheduler = WarmupCosineLR(
            optimizer=optimizer,
            T_max=t_max,
            eta_min=getattr(cfg, "lr_cosine_eta_min", 1e-6),
            **warmup_kwargs,
        )
    elif scheduler_type == "cos_restart":
        t0_epochs = getattr(cfg, "lr_cosine_restart_t0_epochs", 10)
        t_mult = getattr(cfg, "lr_cosine_restart_t_mult", 2)
        T_0 = max(1, int(t0_epochs * steps_per_epoch))
        lr_scheduler = WarmupCosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=T_0,
            T_mult=t_mult,
            eta_min=getattr(cfg, "lr_cosine_eta_min", 1e-6),
            **warmup_kwargs,
        )
    elif scheduler_type in ("none", ""):
        lr_scheduler = None
    else:
        raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}")

    scaler = torch.amp.GradScaler(enabled=cfg.amp_enabled)
    metrics = PhysicsMetrics()

    best_mse = float("inf")
    best_l2r = float("inf")
    start_epoch = 0
    resume_path = getattr(args, "resume_path", None)
    if resume_path:
        resume_file = Path(resume_path)
        if resume_file.is_file():
            target_emo = unwrap_model(emo)
            checkpoint = torch.load(resume_file, map_location="cpu", weights_only=False)
            model_state = checkpoint.get("model_state_dict")
            if model_state is not None:
                target_emo.load_state_dict(model_state)
            optim_state = checkpoint.get("optimizer_state_dict")
            if optim_state is not None:
                optimizer.load_state_dict(optim_state)
            scaler_state = checkpoint.get("scaler_state_dict")
            if scaler_state is not None and scaler is not None:
                scaler.load_state_dict(scaler_state)
            sched_state = checkpoint.get("lr_scheduler_state_dict")
            if sched_state is not None and lr_scheduler is not None:
                lr_scheduler.load_state_dict(sched_state)
            best_mse = checkpoint.get(
                "best_mse",
                checkpoint.get("metrics", {}).get("mse", best_mse),
            )
            best_l2r = checkpoint.get(
                "best_l2r",
                checkpoint.get("metrics", {}).get("l2_relative_error", best_l2r),
            )
            start_epoch = max(0, int(checkpoint.get("epoch", -1)) + 1)
            if is_main_process(cfg):
                print(f"[Resume] Loaded checkpoint from {resume_file} (start_epoch={start_epoch}).")
        elif is_main_process(cfg):
            print(f"[Resume] Provided resume_path '{resume_path}' not found; starting fresh.")

    # ==== PDECombinedLoss weights per task ====
    if cfg.task == "darcy":
        lambda_rel = 1.0
        lambda_grad = 0.05
        lambda_fourier = 0.02
    elif cfg.task == "navier":
        lambda_rel = 1.0
        lambda_grad = 0.0
        lambda_fourier = 0.02
    elif cfg.task == "plasticity":
        lambda_rel = 1.0
        lambda_grad = 0.05
        lambda_fourier = 0.01
    elif cfg.task == "pipe":
        lambda_rel = 1.0
        lambda_grad = 0.0
        lambda_fourier = 0.01
    elif cfg.task == "airfoil":
        lambda_rel = 1.0
        lambda_grad = 0.0
        lambda_fourier = 0.01
    else:
        raise ValueError(f"Unsupported task '{cfg.task}' for PDECombinedLoss.")

    criterion = PDECombinedLoss(
        lambda_rel=lambda_rel,
        lambda_grad=lambda_grad,
        lambda_fourier=lambda_fourier,
    )
    
    criterion.to(device)
    
    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # ---- Train ----
        train_loss, train_mse, train_mae = train_one_epoch(
            emo,
            optimizer,
            train_loader,
            device,
            metrics,
            scaler,
            cfg,
            args,
            epoch,
            criterion,
            lr_scheduler=lr_scheduler,
            scheduler_step_mode=scheduler_step_mode,
        )

        # ---- Eval ----
        mse, mae, psnr, rmse, ssim, l2r, vis_payload = evaluate(
            emo, val_loader, device, metrics, cfg, args, epoch
        )

        if is_main_process(cfg):
            is_best = l2r < best_l2r
            current_best_mse = mse if is_best else best_mse
            current_best_l2r = l2r if is_best else best_l2r
            model_for_save = unwrap_model(emo)
            ckpt = {
                "epoch": epoch,
                "config": asdict(cfg),
                "model_state_dict": model_for_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                "metrics": {
                    "mse": mse, "mae": mae, "psnr": psnr, "rmse": rmse, "ssim": ssim, "l2_relative_error": l2r
                },
                "best_mse": current_best_mse,
                "best_l2r": current_best_l2r,
            }
            ckpt_path = Path(cfg.save_dir) / f"checkpoint_last.pt"
            torch.save(ckpt, ckpt_path)
            print(f"[Checkpoint] Saved to {ckpt_path}")

            if is_best:
                best_mse = current_best_mse
                best_l2r = current_best_l2r
                best_path = Path(cfg.save_dir) / "checkpoint_best.pt"
                torch.save(ckpt, best_path)
                print(f"[Checkpoint] New best l2r={best_l2r:.6e}, best mse={best_mse:.6e}, saved to {best_path}")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
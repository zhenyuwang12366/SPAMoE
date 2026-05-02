# scripts/train_seismic_moe.py
"""
Train neural operator models on seismic data with a Mixture of Experts (MoE).
Supports distributed training and DeepSpeed.
"""
import optuna
import os
import sys
import re
import math
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Union
from argparse import Namespace
import tqdm
from pathlib import Path
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, random_split, Subset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

# >>> DeepSpeed (import only; harmless when not using DS)
try:
    import deepspeed
except Exception:
    deepspeed = None

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.transforms as T
from neuralop.models import MOEOperator, ExpertFactory
from neuralop.models.afreqmoe import AdaptiveFreqMoE
from neuralop.models.encoder import get_encoder
from neuralop.models.EMO import EMO
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.data.datasets import SeismicDataset, ZarrSeismicDataset
from neuralop.data.dataloader.zarr_seismic_dataloader import build_loaders
from neuralop.utils import count_model_params
from scripts.scheduler import (
    WarmupMultiStepLR,
    WarmupCosineLR,
    WarmupCosineAnnealingWarmRestarts,
)
from neuralop.losses import L1L2Loss, SobelLoss, FourierMag_L1
from utils import *

print("-----------------------------------------------------------")
patch_spectral_conv_forward(SpectralConv)

# =========================
# ONLY-ROUTER LOADER UTILS
# =========================
def _filter_state_by_prefix(sd: Dict[str, torch.Tensor], prefixes) -> Dict[str, torch.Tensor]:
    if isinstance(prefixes, str):
        prefixes = (prefixes, )
    keep = {}
    for k, v in sd.items():
        if any(k.startswith(p) for p in prefixes):
            keep[k] = v
    return keep


def _load_router_weights(model: nn.Module, router_path: Union[str, Path], map_location="cpu", is_logger=False):
    router_path = str(router_path)
    ckpt = torch.load(router_path, map_location=map_location, weights_only=False)
    sd = (ckpt.get("router_state_dict")
          or ckpt.get("model_state_dict")
          or ckpt.get("state_dict")
          or ckpt)
    if not isinstance(sd, dict):
        raise ValueError(f"Could not extract a usable state_dict from {router_path}.")

    router_prefixes = ("router.", "gate.", "routing.", "router_net.", "router_module.")
    router_sd = _filter_state_by_prefix(sd, router_prefixes)

    missing, unexpected = model.load_state_dict(router_sd, strict=False)
    if is_logger:
        print(f"[Router] Loaded router weights from {router_path} (prefix filter: {router_prefixes}).")
        if missing:
            print(f"[Router] Missing keys (non-router keys or name mismatch): {missing}")
        if unexpected:
            print(f"[Router] Unexpected keys (checkpoint may contain non-router keys): {unexpected}")

from collections import defaultdict
import torch.distributed as dist
def summarize_module_devices(module):
    dev_map = defaultdict(int)
    for n, p in module.named_parameters(recurse=True):
        dev_map[str(p.device)] += p.numel()
    for n, b in module.named_buffers(recurse=True):
        dev_map[str(b.device)] += b.numel()
    print(f"[Rank {dist.get_rank()}] Param/Buffer device distribution:")
    for k, v in dev_map.items():
        print(f"  {k}: {v} elems", flush=True)
        
class TransformedSubset(Subset):
    def __init__(self, dataset, transform=None):
        if hasattr(dataset, 'indices'):
            super().__init__(dataset.dataset, dataset.indices)
        else:
            super().__init__(dataset, list(range(len(dataset))))
        self.transform = transform

    def _get_single(self, index: int):
        sample = self.dataset[self.indices[index]]
        sample = {**sample, 'idx': index}
        if self.transform:
            return self.transform(sample)
        return sample

    def __getitem__(self, index):
        if isinstance(index, list):
            return [self._get_single(i) for i in index]
        return self._get_single(index)

    def __getitems__(self, idx: list[int]):
        return [self._get_single(index) for index in idx]


def run_training(args, trial: Optional["optuna.trial.Trial"] = None):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    config, runtime_ctx = get_seismic_config(args)

    use_deepspeed: bool = bool(getattr(args, "use_deepspeed", False))
    device = runtime_ctx["device"]
    is_logger = runtime_ctx["is_logger"]
    world_size = runtime_ctx["world_size"]
    local_rank = args.local_rank if use_deepspeed else runtime_ctx["local_rank"]
    experts_name = runtime_ctx["experts_name"]
    experts_name_str = runtime_ctx["experts_name_str"]
    use_amp = config.use_amp

    # DeepSpeed: bind this process to its local GPU
    if use_deepspeed:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    
    # ======================
    # Data loading (Zarr or on-disk files)
    # ======================
    if args.zarr_path is not None:
        json_path = getattr(args, 'status_json', None)
        assert json_path is not None, "Zarr format requires status_json for normalization stats"

        zarr_path = getattr(args, 'zarr_path', None)
        assert zarr_path is not None, "Set args.zarr_path when using Zarr data"

        with open(json_path, 'r') as f:
            data_dict_raw = json.load(f)

        if config.family == 'all':
            data_dict = data_dict_raw['overall']
        else:
            data_dict = data_dict_raw['per_type'][config.family]

        from neuralop.data.datasets.seismic_dataset import SeismicDataProcessor
        input_transform = Compose([
            T.LogTransform(k=args.k),
            T.MinMaxNormalize(T.log_transform(data_dict['input_min'], k=args.k),
                              T.log_transform(data_dict['input_max'], k=args.k))
        ])
        output_transform = Compose([
            T.MinMaxNormalize(data_dict['output_min'], data_dict['output_max'])
        ])
        input_inverse_transform = Compose([
            T.InverseMinMaxNormalize(T.log_transform(data_dict['input_min'], k=args.k),
                                     T.log_transform(data_dict['input_max'], k=args.k)),
            T.InverseLogTransform(k=args.k)
        ])
        output_inverse_transform = Compose([
            T.InverseMinMaxNormalize(data_dict['output_min'], data_dict['output_max'])
        ])

        data_processor = SeismicDataProcessor(
            input_transform=input_transform,
            output_transform=output_transform,
            channel_dim=config.channel_dim,
            config=config,
        )

        train_dataset = ZarrSeismicDataset(
            zarr_path=zarr_path,
            split='train',
            input_transform=None,
            output_transform=None,
            expect_input_shape=(1, 1000, 350),
            to_float32=True,
        )
        val_dataset = ZarrSeismicDataset(
            zarr_path=zarr_path,
            split='val',
            input_transform=None,
            output_transform=None,
            expect_input_shape=(1, 1000, 350),
            to_float32=True,
        )
        train_dataset_with_transform = TransformedSubset(train_dataset, data_processor)
        val_dataset_with_transform = TransformedSubset(val_dataset, data_processor)

        train_loader, val_loader, train_sampler, val_sampler = build_loaders(
            args=args,
            config=config,
            train_dataset_with_transform=train_dataset_with_transform,
            val_dataset_with_transform=val_dataset_with_transform,
            chunks=32,
            world_size=world_size,
            local_rank=local_rank,
        )
    else:
        if config.family in ['curve_vel_a', 'curve_vel_b', 'flat_vel_a', 'flat_vel_b']:
            val_ratio = 6 / 30
        elif config.family in ['curve_fault_a', 'curve_fault_b', 'flat_fault_a', 'flat_fault_b']:
            val_ratio = 6 / 54
        elif config.family in ['style_a', 'style_b', 'style_style_a', 'style_style_b']:
            val_ratio = 7 / 67
        else:
            raise ValueError("Unsupported family")

        full_dataset = SeismicDataset(
            data_dir=config.data_dir,
            family=config.family,
            is_specific=config.is_specific,
            split='train',
            concat_channels=config.concat_channels,
            config=config
        )

        dataset_size = len(full_dataset)
        train_size, val_size = safe_random_split(dataset_size, [1 - val_ratio, val_ratio])

        if is_logger:
            print(f"Dataset size: {dataset_size}")
            print(f"Train size: {train_size}")
            print(f"Val size: {val_size}")

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

        data_dict = full_dataset.getStats()

        if is_logger:
            train_indices_set = set(train_dataset.indices)
            val_indices_set = set(val_dataset.indices)
            overlap = train_indices_set.intersection(val_indices_set)
            if overlap:
                print(f"Warning: train and val share {len(overlap)} overlapping samples!")
            else:
                print("OK: train and val have no overlap")
            assert len(train_dataset) == train_size
            assert len(val_dataset) == val_size

        from neuralop.data.datasets.seismic_dataset import SeismicDataProcessor
        input_transform = Compose([
            T.LogTransform(k=args.k),
            T.MinMaxNormalize(T.log_transform(data_dict['input_min'], k=args.k),
                              T.log_transform(data_dict['input_max'], k=args.k))
        ])
        output_transform = Compose([
            T.MinMaxNormalize(data_dict['output_min'], data_dict['output_max'])
        ])
        input_inverse_transform = Compose([
            T.InverseMinMaxNormalize(T.log_transform(data_dict['input_min'], k=args.k),
                                     T.log_transform(data_dict['input_max'], k=args.k)),
            T.InverseLogTransform(k=args.k)
        ])
        output_inverse_transform = Compose([
            T.InverseMinMaxNormalize(data_dict['output_min'], data_dict['output_max'])
        ])

        data_processor = SeismicDataProcessor(
            input_transform=input_transform,
            output_transform=output_transform,
            channel_dim=config.channel_dim,
            config=config,
        )

        train_dataset_with_transform = TransformedSubset(train_dataset, data_processor)
        val_dataset_with_transform = TransformedSubset(val_dataset, data_processor)

        if args.distributed:
            train_num_workers = max(0, args.num_workers // 2)
            train_sampler = DistributedSampler(
                train_dataset_with_transform,
                num_replicas=world_size,
                rank=local_rank,
                drop_last=True,
                shuffle=True
            )
            train_loader = DataLoader(
                train_dataset_with_transform,
                sampler=train_sampler,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=train_num_workers,
                pin_memory=True,
                persistent_workers=train_num_workers > 0
            )
            val_num_workers = train_num_workers
            val_sampler = DistributedSampler(
                val_dataset_with_transform,
                num_replicas=world_size,
                rank=local_rank,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_dataset_with_transform,
                sampler=val_sampler,
                batch_size=config.test_batch_size,
                shuffle=False,
                num_workers=val_num_workers,
                pin_memory=True,
                persistent_workers=val_num_workers > 0
            )
        else:
            train_num_workers = max(0, args.num_workers)
            train_loader = DataLoader(
                train_dataset_with_transform,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=train_num_workers,
                pin_memory=True,
                persistent_workers=train_num_workers > 0
            )
            val_loader = DataLoader(
                val_dataset_with_transform,
                batch_size=config.test_batch_size,
                shuffle=False,
                num_workers=train_num_workers,
                pin_memory=True,
                persistent_workers=train_num_workers > 0
            )

    if is_logger:
        prefetch = getattr(train_loader, "prefetch_factor", None)
        if prefetch is not None:
            print(f'prefetch_factor={prefetch}')

    # Shape sanity check
    sample_batch = next(iter(train_loader))
    if is_logger:
        input_shape = sample_batch['input'].shape
        output_shape = sample_batch['output'].shape
        print(f"Input tensor shape: {input_shape}")
        print(f"Output tensor shape: {output_shape}")
        if len(input_shape) < 3 or len(input_shape) > 4:
            print(f"Warning: unexpected input rank (expected 3D/4D), got {len(input_shape)}D")
        if len(output_shape) < 3:
            print(f"Warning: unexpected output rank (expected 3D+), got {len(output_shape)}D")

    in_channels = sample_batch['input'].shape[1]
    config.in_channels = in_channels

    if is_logger:
        print(f"Updated in_channels: {config.in_channels}")
        print(f"out_channels: {config.out_channels}")
        print(f"hidden_channels: {config.hidden_channels}")
        print(f"Expert configs count (use_moe=False path): {len(config.expert_configs)}")

    # ======================
    # Build encoder
    # ======================
    encoder_model = None
    encoder_freeze = False
    if config.use_encoder:
        if config.train_encoder:
            encoder_model = get_encoder(
                in_channels=config.in_channels,
                out_channels=128,
                num_types=10,
                type_act='identity',
                backbone=config.backbone,
            )
        else:
            encoder_model = get_encoder(
                in_channels=config.in_channels,
                out_channels=128,
                num_types=10,
                type_act='softmax',
                backbone=config.backbone,
            )

        if getattr(args, "encoder_path", None):
            missing, unexpected = load_encoder_weights(
                encoder_model,
                args.encoder_path,
                map_location="cpu",
                strict=False,
            )
            if is_logger:
                print(f"[Encoder] Loaded pretrained weights from {args.encoder_path}.")
                if missing:
                    print(f"[Encoder] Missing keys: {missing}")
                if unexpected:
                    print(f"[Encoder] Unexpected keys: {unexpected}")
            for p in encoder_model.parameters():
                p.requires_grad_(False)
            encoder_freeze = True

        if not config.is_classifier:
            if config.backbone == 'vit':
                for p in encoder_model.type_head.parameters():
                    p.requires_grad_(False)
            else:
                for n, p in encoder_model.backbone.named_parameters():
                    if n.startswith("head."):
                        p.requires_grad_(False)

        # Resolve moe_in_channels
        if use_deepspeed:
            moe_in_channels = 128
        else:
            encoder_model.eval()
            with torch.no_grad():
                encoder_probe, _, _ = encoder_model(sample_batch['input'])
            moe_in_channels = encoder_probe.shape[1]
            del encoder_probe
            if encoder_freeze:
                encoder_model.eval()
            else:
                encoder_model.train()

        if is_logger:
            print(f"Encoder output channels (for MoE): {moe_in_channels}")
    else:
        moe_in_channels = config.in_channels
        if is_logger:
            print("[Encoder] use_encoder=False: feed raw input to MoE.")

    config.moe_in_channels = moe_in_channels

    # ======================
    # Build experts + MoE
    # ======================
    experts_map_location = "cpu" if use_deepspeed else device

    if config.use_moe and config.use_experts_path:
        experts = load_moe_experts(
            experts_config=config.load_expert_configs,
            in_channels=moe_in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels,
            model_path=config.use_experts_path,
            is_specific=config.is_specific,
            map_location=experts_map_location,
            type_dict=config.type_id,
            moe_mode=config.moe_mode,
        )
    else:
        experts = ExpertFactory.create_expert_ensemble(
            expert_configs=config.expert_configs,
            in_channels=moe_in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels
        )

    if config.moe_method == "basic":
        moe_model = MOEOperator(
            experts=experts,
            in_channels=moe_in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels,
            top_k=config.top_k,
            noisy_gating=config.noisy_gating,
            fusion_type=config.fusion_type,
            router_hidden_dim=config.router_hidden_dim,
            moe_mode=getattr(config, "moe_mode", "standard"),
            is_logger=is_logger,
            router_type=config.router_type,
            s_processor_type=config.s_processor_type,
            w_processor_type=config.w_processor_type,
            beta=config.beta,
            is_specific=config.is_specific,
            is_classifier=config.is_classifier,
            batch_size=config.batch_size,
            v_type_num=config.v_type_num,
            use_expert_memory_proxy=config.use_gpu_proxy,
            use_encoder=config.use_encoder,
            device=device,
        )
    elif config.moe_method == "afmoe":
        moe_model = AdaptiveFreqMoE(
            experts=experts,
            in_channels=moe_in_channels,
            topk=config.top_k,
            alpha=float(getattr(config, "router_alpha", 0.1)),
            band_sharpness=float(getattr(config, "band_sharpness", 20.0)),
            freq_affinity_sharpness=float(getattr(config, "freq_affinity_sharpness", 10.0)),
            use_soft_bands=bool(getattr(config, "use_soft_bands", True)),
            enable_freq_attn=bool(getattr(config, "enable_freq_attn", True)),
            enable_band_mixing=bool(getattr(config, "enable_band_mixing", True)),
            enable_band_decomposition=bool(getattr(config, "enable_band_decomposition", True)),
            routing_mode=str(getattr(config, "routing_mode", "learned")),
        )

    # ======================
    # EMO + DDP / DeepSpeed
    # ======================
    if use_deepspeed:
        if deepspeed is None:
            raise RuntimeError("deepspeed is not installed but --use_deepspeed was set")
        emo_model = EMO(encoder_model, moe_model, pass_encoder_logits_as_weights=True)
        model = emo_model  # placeholder; replaced by DeepSpeed engine after init
    else:
        if experts_name_str == "all":
            static_graph = True
            find_unused_parameters = False
        else:
            static_graph = False
            find_unused_parameters = True
        emo_model = EMO(encoder_model, moe_model, pass_encoder_logits_as_weights=True)
        if config.distributed.use_distributed:
            emo_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(emo_model).to(device)
            model = DDP(
                emo_model, device_ids=[device.index],
                output_device=device.index,
                static_graph=static_graph,
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=True,
            )
        else:
            model = emo_model.to(device)

    if encoder_model is not None and encoder_freeze:
        encoder_model.eval()
        if is_logger:
            print("[Encoder] Encoder parameters frozen.")

    # ======================
    # Optimizer & LR scheduler
    # ======================
    if config.distributed.use_distributed and world_size > 2:
        lr = float(config.learning_rate) * math.sqrt(world_size)
    else:
        lr = float(config.learning_rate)

    steps_per_epoch = max(1, len(train_loader))
    total_num_steps = int(getattr(config, "epochs", 100) * steps_per_epoch)

    eta_min = float(getattr(config, "lr_cosine_eta_min", 1e-6))
    cos_min_ratio = 0.0 if lr <= 0 else max(0.0, min(1.0, eta_min / lr))

    warmup_min_ratio = float(getattr(config, "lr_warmup_min_ratio", 0.0))
    warmup_min_ratio = max(0.0, min(1.0, warmup_min_ratio))
    warmup_method = str(getattr(config, "lr_warmup_method", "linear")).lower()
    warmup_type = "linear" if "lin" in warmup_method else ("log" if "log" in warmup_method else "linear")

    warmup_epochs = float(getattr(config, "lr_warmup_epochs", 0.0))
    warmup_iters = int(max(0.0, warmup_epochs) * steps_per_epoch)

    weight_decay = float(getattr(config, "weight_decay", 1e-2))

    warmup_kwargs = dict(
        warmup_factor=getattr(config, "lr_warmup_factor", 1.0 / 3),
        warmup_iters=warmup_iters,
        warmup_method=getattr(config, "lr_warmup_method", "linear"),
    )

    _param_src = emo_model if use_deepspeed else model
    optim_params = [p for p in _param_src.parameters() if p.requires_grad]
    assert len(optim_params) > 0, "No trainable parameters collected for optimizer!"

    optimizer = None
    if not use_deepspeed:
        optimizer = torch.optim.AdamW(
            optim_params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay
        )

    encoder_requires_grad = False
    if getattr(emo_model, "encoder", None) is not None:
        encoder_requires_grad = any(p.requires_grad for p in emo_model.encoder.parameters())

    lr_scheduler = None
    if use_deepspeed:
        if not args.ds_config:
            raise ValueError("Provide --deepspeed_config")
        ds_cfg_path = Path(args.ds_config)
        if not ds_cfg_path.exists():
            raise ValueError(f"DeepSpeed config not found: {ds_cfg_path}")

        with open(ds_cfg_path, "r", encoding="utf-8") as f:
            ds_cfg = json.load(f)

        ds_cfg["optimizer"] = {
            "type": "Adam",
            "params": {
                "lr": float(lr),
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": float(weight_decay),
                "adam_w_mode": True
            }
        }

        sched = ds_cfg.get("scheduler")
        if sched is None:
            ds_cfg["scheduler"] = {
                "type": "WarmupCosineLR",
                "params": {
                    "total_num_steps": int(total_num_steps),
                    "warmup_num_steps": int(warmup_iters),
                    "warmup_min_ratio": float(warmup_min_ratio),
                    "warmup_type": str(warmup_type),
                    "cos_min_ratio": float(cos_min_ratio),
                    "last_batch_iteration": -1
                }
            }
        else:
            sched.setdefault("type", "WarmupCosineLR")
            params = sched.setdefault("params", {})
            params.setdefault("total_num_steps", int(total_num_steps))
            params.setdefault("warmup_num_steps", int(warmup_iters))
            params.setdefault("warmup_min_ratio", float(warmup_min_ratio))
            params.setdefault("warmup_type", str(warmup_type))
            params.setdefault("cos_min_ratio", float(cos_min_ratio))
            params.setdefault("last_batch_iteration", -1)

        zero_cfg = ds_cfg.get("zero_optimization", {})
        off_cfg = zero_cfg.get("offload_optimizer", {})
        if isinstance(off_cfg, dict):
            off_cfg.setdefault("device", "cpu")
            off_cfg.setdefault("pin_memory", True)
        zero_cfg["offload_optimizer"] = off_cfg
        ds_cfg["zero_optimization"] = zero_cfg

        ds_init_kwargs = {
            "model": emo_model,
            "model_parameters": (p for p in emo_model.parameters() if p.requires_grad),
            "config": ds_cfg,
        }
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(**ds_init_kwargs)
        model = model_engine
        if is_logger:
            print("[DeepSpeed] DeepSpeed training enabled.")
            try:
                print(f"[DeepSpeed] Zero Stage: {model.zero_optimization_stage()}")
            except Exception:
                pass

    if not use_deepspeed:
        scheduler_type = getattr(config, "lr_scheduler_type", "cos_restart")
        if scheduler_type == "multistep":
            lr_milestones = [int(steps_per_epoch * m) for m in getattr(config, "milestones", [])]
            lr_scheduler = WarmupMultiStepLR(
                optimizer=optimizer,
                milestones=lr_milestones,
                gamma=getattr(config, "scheduler_gamma", 0.1),
                **warmup_kwargs,
            )
        elif scheduler_type == "cos":
            t_max_epochs = getattr(config, "lr_cosine_tmax_epochs",
                                   max(1, getattr(config, "epochs", 100) - getattr(config, "lr_warmup_epochs", 0)))
            t_max = max(1, int(t_max_epochs * steps_per_epoch))
            lr_scheduler = WarmupCosineLR(
                optimizer=optimizer,
                T_max=t_max,
                eta_min=getattr(config, "lr_cosine_eta_min", 1e-6),
                **warmup_kwargs,
            )
        elif scheduler_type == "cos_restart":
            t0_epochs = getattr(config, "lr_cosine_restart_t0_epochs", 10)
            T_0 = max(1, int(t0_epochs * steps_per_epoch))
            lr_scheduler = WarmupCosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=T_0,
                T_mult=getattr(config, "lr_cosine_restart_t_mult", 2),
                eta_min=getattr(config, "lr_cosine_eta_min", 1e-6),
                **warmup_kwargs,
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}")

    summarize_module_devices(model)
    
    # ======================
    # Loss functions
    # ======================
    lambda_grad_l1 = float(getattr(config, "lambda_grad_l1", 0.0))
    lambda_fourier_mag_l1 = float(getattr(config, "lambda_fourier_mag_l1", 0.0))
    lambda_ce = float(getattr(config, "lambda_ce", 0.0))

    base_loss: Callable = L1L2Loss(config.lambda_g1v, config.lambda_g2v).to(device)
    grad_loss_module = SobelLoss().to(device) if lambda_grad_l1 > 0 else None
    fourier_loss_module = FourierMag_L1().to(device) if lambda_fourier_mag_l1 > 0 else None

    # Train classification + regression (CE on encoder logits)
    if config.train_encoder:
        def criterion(pred: torch.Tensor, gt: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
            # Disable AMP in loss; compute in fp32 to avoid dtype issues under mixed precision
            with torch.amp.autocast(device_type=pred.device.type, enabled=False):
                pred32 = pred.float()
                gt32 = gt.float()

                # base_loss must return tensor fields {"loss", "l1", "l2"}
                loss_dict = base_loss(pred32, gt32)
                total_loss_t = loss_dict["loss"]  # scalar tensor

                # grad L1
                grad_val_t = pred32.new_zeros(())
                if grad_loss_module is not None:
                    grad_res = grad_loss_module(pred32, gt32)   # {"loss": tensor}
                    grad_val_t = grad_res["loss"]
                    total_loss_t = total_loss_t + lambda_grad_l1 * grad_val_t

                # fourier L1
                fourier_val_t = pred32.new_zeros(())
                if fourier_loss_module is not None:
                    fourier_res = fourier_loss_module(pred32, gt32)  # {"loss": tensor}
                    fourier_val_t = fourier_res["loss"]
                    total_loss_t = total_loss_t + lambda_fourier_mag_l1 * fourier_val_t

                # Cross-entropy in fp32
                ce_val_t = pred32.new_zeros(())
                if logits is not None and labels is not None:
                    ce_val_t = F.cross_entropy(logits.float(), labels.long(), reduction="mean")
                    total_loss_t = total_loss_t + lambda_ce * ce_val_t

            # Return tensors (*_t) for backward and floats for logging
            combined = {
                # tensors for backward
                "loss_t":       total_loss_t,
                "l1_t":         loss_dict["l1"],
                "l2_t":         loss_dict["l2"],
                "grad_l1_t":    grad_val_t,
                "fourier_l1_t": fourier_val_t,
                "ce_t":         ce_val_t,

                # floats for logging
                "loss":         float(total_loss_t.detach().item()),
                "l1":           float(loss_dict["l1"].detach().item()),
                "l2":           float(loss_dict["l2"].detach().item()),
                "grad_l1":      float(grad_val_t.detach().item()),
                "fourier_l1":   float(fourier_val_t.detach().item()),
                "ce":           float(ce_val_t.detach().item()),
            }
            return combined
    else:
        def criterion(pred: torch.Tensor, gt: torch.Tensor):
            # Disable AMP in loss; use fp32
            with torch.amp.autocast(device_type=pred.device.type, enabled=False):
                pred32 = pred.float()
                gt32 = gt.float()

                loss_dict = base_loss(pred32, gt32)
                total_loss_t = loss_dict["loss"]

                grad_val_t = pred32.new_zeros(())
                if grad_loss_module is not None:
                    grad_res = grad_loss_module(pred32, gt32)
                    grad_val_t = grad_res["loss"]
                    total_loss_t = total_loss_t + lambda_grad_l1 * grad_val_t

                fourier_val_t = pred32.new_zeros(())
                if fourier_loss_module is not None:
                    fourier_res = fourier_loss_module(pred32, gt32)
                    fourier_val_t = fourier_res["loss"]
                    total_loss_t = total_loss_t + lambda_fourier_mag_l1 * fourier_val_t

            combined = {
                # tensors for backward
                "loss_t":       total_loss_t,
                "l1_t":         loss_dict["l1"],
                "l2_t":         loss_dict["l2"],
                "grad_l1_t":    grad_val_t,
                "fourier_l1_t": fourier_val_t,

                # floats for logging
                "loss":         float(total_loss_t.detach().item()),
                "l1":           float(loss_dict["l1"].detach().item()),
                "l2":           float(loss_dict["l2"].detach().item()),
                "grad_l1":      float(grad_val_t.detach().item()),
                "fourier_l1":   float(fourier_val_t.detach().item()),
            }
            return combined

    # ======================
    # Run dirs & logging
    # ======================
    def _slugify(text: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in str(text))

    run_group = _slugify(config.family or "all")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name_parts = [
        _slugify(config.model_name or "model"),
        f"router-{_slugify(config.router_type)}",
        f"lr{config.learning_rate:g}",
        f"bs{config.batch_size}",
        _slugify(experts_name_str or "experts"),
        timestamp,
    ]
    run_name = "_".join(part for part in run_name_parts if part)

    output_root = Path(config.output_dir) / f"seismic_moe_{run_group}"
    results_dir = output_root / run_name
    log_file = results_dir / "training_log.txt"

    tb_root = Path(getattr(config, "log_root", "./runs")).expanduser()
    tb_dir = tb_root / run_group / run_name
    tb_writer: Optional[SummaryWriter] = None

    if is_logger:
        results_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))

        config.experiment_dir = str(results_dir)
        config.tensorboard_dir = str(tb_dir)

        config_payload = config.to_dict() if hasattr(config, "to_dict") else vars(config)
        with open(results_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2, default=str)
        with open(results_dir / "args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, default=str)

        if config.train_encoder:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("    Epoch    |    Train Loss    |    Val Loss    |    MAE    |    MSE    |    PSNR    |    RMSE    |    SSIM    |    CE    |\n")
        else:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("    Epoch    |    Train Loss    |    Val Loss    |    MAE    |    MSE    |    PSNR    |    RMSE    |    SSIM    |\n")
    else:
        config.experiment_dir = str(results_dir)
        config.tensorboard_dir = str(tb_dir)

    # ======================
    # Best-checkpoint paths
    # ======================
    best_val_loss = float("inf")
    best_model_path = results_dir / f"best_model_{experts_name_str}.pt"
    best_expert_path = (results_dir / f"best_expert_{experts_name_str}.pt") if len(experts_name) == 1 and experts_name_str != "all" else None
    last_model_path = results_dir / f"last_model_{experts_name_str}.pt"
    last_expert_path = (results_dir / f"last_expert_{experts_name_str}.pt") if len(experts_name) == 1 and experts_name_str != "all" else None
    if config.train_encoder or (len(experts_name) == 1 and experts_name[0] == "all"):
        best_encoder_path = results_dir / f"best_encoder.pt"
        last_encoder_path = results_dir / f"last_encoder.pt"
        best_router_path = results_dir / f"best_router.pt"
        last_router_path = results_dir / f"last_router.pt"
    else:
        best_encoder_path = None
        last_encoder_path = None
        best_router_path = None
        last_router_path = None

    metrics = SeismicMetrics()

    if config.early_stop:
        early_patience = getattr(config, "early_stop_patience", 20)
        early_min_delta = getattr(config, "early_stop_min_delta", 0.0)
        early_warmup   = getattr(config, "early_stop_warmup_epochs", 10)
        early_stopper = EarlyStopping(
            patience=early_patience,
            min_delta=early_min_delta,
            warmup_epochs=early_warmup,
            mode="min"
        )

    # ======================
    # Resume (mirrors save logic; no separate moe load)
    # ======================
    start_epoch = 0
    if hasattr(args, "resume_path") and args.resume_path is not None and os.path.exists(args.resume_path):
        ckpt_path = args.resume_path
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Works with DDP(EMO), EMO, or DeepSpeedEngine
        target_model = model.module if hasattr(model, "module") else model
        target_moe = getattr(target_model, "moe", None)

        # ---------- 1) DeepSpeed engine state (if saved via engine.save_checkpoint) ----------
        if use_deepspeed and ("engine" in globals() or "engine" in locals()) and (model is not None):
            resume_dir = os.path.dirname(ckpt_path)
            resume_file = os.path.basename(ckpt_path)
            ds_tag = getattr(args, "resume_tag", None)
            if ds_tag is None:
                m = re.search(r"(best|last-ep\d+)", resume_file)
                ds_tag = m.group(1) if m else None

            if ds_tag is not None:
                load_path, client_state = model.load_checkpoint(resume_dir, tag=ds_tag)
                if is_logger:
                    print(f"[DeepSpeed] Loaded engine state: dir={resume_dir}, tag={ds_tag}, load_path={load_path}")
            else:
                if is_logger:
                    print("[DeepSpeed][WARN] Could not infer tag; if you used engine.save_checkpoint, pass --resume_tag=best or last-epN")

        # ---------- 2) Parse checkpoint shards (mirror your save layout) ----------
        model_check   = checkpoint.get("model_state_dict", None)       # present in some setups
        router_check  = checkpoint.get("router_state_dict", None)      # non-DS + "all" saves
        encoder_check = checkpoint.get("encoder_state_dict", None)     # may exist in both paths
        expert_check  = checkpoint.get("expert_state_dict", None)      # optional in single-expert mode

        # Case A: not DeepSpeed and experts_name_str == "all"
        # Saver dropped model_state_dict; keeps router_state_dict (+ encoder)
        if (not use_deepspeed) and (experts_name_str == "all"):
            if router_check is not None and (target_moe is not None) and hasattr(target_moe, "router") and (target_moe.router is not None):
                missing, unexpected = target_moe.router.load_state_dict(router_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Resume][all][router] missing: {missing}, unexpected: {unexpected}")
            else:
                if is_logger:
                    print("[Resume][all][router] no router_state_dict in ckpt (or no moe.router); skip router")

            # Encoder: restore from ckpt if no external encoder_path
            if hasattr(target_model, "encoder") and (encoder_check is not None) and not getattr(args, "encoder_path", None):
                missing, unexpected = load_encoder_weights(target_model.encoder, encoder_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Resume][all][encoder] missing: {missing}, unexpected: {unexpected}")
            elif encoder_check is not None and getattr(args, "encoder_path", None) and is_logger:
                print(f"[Resume][all][encoder] --encoder_path set; skip encoder from ckpt")

        else:
            # Case B: all other setups (DeepSpeed or non-DS, single/all experts)
            # Saver usually includes full model_state_dict (DS sidecar or non-DS default)
            if model_check is not None:
                missing, unexpected = target_model.load_state_dict(model_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Resume][generic][model] missing: {missing}, unexpected: {unexpected}")
            else:
                if is_logger:
                    print("[Resume][generic] no model_state_dict in ckpt; skip full-model restore")

            # Single-expert: optional --resume_expert_path overrides experts[0]
            if (experts_name is not None and len(experts_name) == 1 and experts_name[0] != "all"):
                # 1) Full model loaded from main ckpt above
                # 2) Optionally overwrite experts[0]
                resume_expert_path = getattr(args, "resume_expert_path", None)
                if resume_expert_path and os.path.exists(resume_expert_path):
                    try:
                        expert_blob = torch.load(resume_expert_path, map_location="cpu", weights_only=False)
                        expert_sd = expert_blob.get("expert_state_dict", None)
                        if expert_sd is not None and (target_moe is not None) and hasattr(target_moe, "experts") and len(target_moe.experts) > 0:
                            missing, unexpected = target_moe.experts[0].load_state_dict(expert_sd, strict=False)
                            if is_logger and (missing or unexpected):
                                print(f"[Resume][single-expert][experts[0]] missing: {missing}, unexpected: {unexpected}")
                            if is_logger:
                                print(f"[Resume][single-expert] Overwrote experts[0] from {resume_expert_path}")
                        elif is_logger:
                            print(f"[Resume][single-expert] {resume_expert_path} has no expert_state_dict or no experts[0]")
                    except Exception as e:
                        if is_logger:
                            print(f"[Resume][single-expert] Failed to load resume_expert_path: {e}")
                elif expert_check is not None and (target_moe is not None) and hasattr(target_moe, "experts") and len(target_moe.experts) > 0:
                    # expert_state_dict bundled in main ckpt (DeepSpeed attachment)
                    missing, unexpected = target_moe.experts[0].load_state_dict(expert_check, strict=False)
                    if is_logger and (missing or unexpected):
                        print(f"[Resume][single-expert][experts[0]] missing: {missing}, unexpected: {unexpected}")

            # Encoder: present in both DS and non-DS checkpoints
            if hasattr(target_model, "encoder") and (encoder_check is not None) and not getattr(args, "encoder_path", None):
                missing, unexpected = load_encoder_weights(target_model.encoder, encoder_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Resume][generic][encoder] missing: {missing}, unexpected: {unexpected}")
            elif encoder_check is not None and getattr(args, "encoder_path", None) and is_logger:
                print(f"[Resume][generic][encoder] --encoder_path set; skip encoder from ckpt")

            # Router: even when not "all", try router_state_dict if present
            if router_check is not None and (target_moe is not None) and hasattr(target_moe, "router") and (target_moe.router is not None):
                try:
                    missing, unexpected = target_moe.router.load_state_dict(router_check, strict=False)
                    if is_logger and (missing or unexpected):
                        print(f"[Resume][generic][router] missing: {missing}, unexpected: {unexpected}")
                except Exception as e:
                    if is_logger:
                        print(f"[Resume][generic][router] Skip router_state_dict load: {e}")

        # ---------- 3) Optimizer (only non-DS checkpoints store it) ----------
        if (not use_deepspeed) and (optimizer is not None):
            try:
                opt_state = checkpoint.get("optimizer_state_dict", None)
                if opt_state:
                    optimizer.load_state_dict(opt_state)
                else:
                    if is_logger:
                        print("[Resume][optimizer] ckpt has no optimizer_state_dict; skip")
            except Exception as e:
                if is_logger:
                    print(f"[Resume][optimizer] Skip optimizer_state_dict load: {e}")

        # ---------- 4) Training metadata ----------
        data_dict   = checkpoint.get("data_dict", None)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if start_epoch < 0:
            start_epoch = 0

        if is_logger:
            print(f"==> Resumed from {ckpt_path}, start_epoch = {start_epoch}")
    else:
        if is_logger:
            print("No valid resume path; training from scratch.")

    amp_enabled = bool(use_amp) and (device.type == 'cuda')
    if use_deepspeed:
        model.zero_grad()
    else:
        optimizer.zero_grad(set_to_none=True)

    # Init best-metrics cache
    best_epoch_metrics = None
    best_epoch_index = 0

    success_flag = results_dir / "TRAIN_DONE"
    failed_flag = results_dir / "TRAIN_FAILED"

    # Remove stale success/fail flags before run
    if is_logger:
        if success_flag.exists():
            success_flag.unlink()
        if failed_flag.exists():
            failed_flag.unlink()

    try:
        # ======================
        # Main training loop
        # ======================
        try:
            for epoch in range(start_epoch, config.epochs):
                vis_now = (is_logger and ((epoch + 1) % args.vis_freq == 0))

                stats, best_val_loss, stop_flag = train_one_epoch(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    epoch=epoch,
                    config=config,
                    is_logger=is_logger,
                    log_file=log_file,
                    results_dir=results_dir,
                    lr_scheduler=lr_scheduler,
                    scheduler_step_mode=("per_step" if config.use_onecycle else "per_epoch"),
                    accum_steps=config.accum_steps,
                    vis_now=vis_now,
                    input_inverse_transform=input_inverse_transform,
                    output_inverse_transform=output_inverse_transform,
                    use_wandb=args.use_wandb,
                    wandb_module=wandb if args.use_wandb else None,
                    early_stopper=early_stopper if config.early_stop else None,
                    best_val_loss=best_val_loss,
                    best_model_path=best_model_path,
                    best_expert_path=best_expert_path,
                    best_encoder_path=best_encoder_path,
                    best_router_path=best_router_path,
                    last_model_path=last_model_path,
                    last_expert_path=last_expert_path,
                    last_encoder_path=last_encoder_path,
                    last_router_path=last_router_path,
                    experts_name=experts_name,
                    experts_name_str=experts_name_str,
                    data_dict=data_dict,
                    metrics_module=metrics,
                    tqdm_module=tqdm,
                    profile_timing=args.profile_timing,
                    amp_enabled=amp_enabled,
                    encoder_frozen=not any(p.requires_grad for p in encoder_model.parameters()) if encoder_model else True,
                    train_encoder=config.train_encoder,
                    tb_writer=tb_writer,
                    engine=(model if use_deepspeed else None),
                    use_deepspeed=use_deepspeed,
                )

                if (
                    tb_writer is not None
                    and is_logger
                    and math.isfinite(stats.get("val_loss", float("inf")))
                    and math.isclose(stats.get("val_loss", float("inf")), best_val_loss, rel_tol=0.0, abs_tol=1e-8)
                ):
                    best_epoch_metrics = dict(stats)
                    best_epoch_metrics["epoch"] = epoch + 1
                    best_epoch_index = epoch + 1

                if stop_flag == 1:
                    break

            if is_logger:
                print(f"VAL_LOSS:{best_val_loss}", flush=True)
                plot_loss_curve(log_file, save_path=results_dir / "loss_curve.png")

            if tb_writer is not None and is_logger:
                hparam_summary = {
                    "family": str(config.family),
                    "router_type": str(config.router_type),
                    "learning_rate": float(config.learning_rate),
                    "batch_size": int(config.batch_size),
                    "top_k": int(getattr(config, "top_k", 1)),
                }
                metric_summary = {
                    "best/val_loss": float(best_val_loss),
                }
                if best_epoch_metrics is not None:
                    metric_summary.update({
                        "best/train_loss": float(best_epoch_metrics.get("train_loss", float("nan"))),
                        "best/psnr": float(best_epoch_metrics.get("psnr", float("nan"))),
                        "best/mse": float(best_epoch_metrics.get("mse", float("nan"))),
                        "best/mae": float(best_epoch_metrics.get("mae", float("nan"))),
                        "best/rmse": float(best_epoch_metrics.get("rmse", float("nan"))),
                        "best/ssim": float(best_epoch_metrics.get("ssim", float("nan"))),
                        "best/epoch": float(best_epoch_index or 0),
                    })
                tb_writer.add_hparams(hparam_summary, metric_summary)
                print("Logged hparams to TensorBoard")

        finally:
            if tb_writer is not None:
                tb_writer.flush()
                tb_writer.close()

        # Reached here: training body and finally cleanup finished
        if is_logger:
            success_flag.write_text("ok\n", encoding="utf-8")
            print(f"[done] success flag written to: {success_flag}", flush=True)

    except Exception as e:
        # Logger rank writes failure flag only
        if is_logger:
            failed_flag.write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
            print(f"[failed] failure flag written to: {failed_flag}", flush=True)
        raise

    return model, best_val_loss


def run_inference(n_args):
    """
    Inference pipeline (mirrors training):
      1) Load training args/config and apply CLI overrides
      2) Build data pipeline (Zarr + normalization)
      3) Build Encoder + MOE -> EMO (same as training)
      4) Load weights
      5) AMP (bf16) inference + full-batch metrics
      6) Spectral assumption check proxies (A1–A3): encoder vs bilinear interpolation
      7) Visualization + router visualization (original behavior)
    """
    from utils.calculate import SpectralAccumulator

    # =======================
    # 1) Load training args/config; CLI overrides
    # =======================
    setting_dir = Path(getattr(n_args, "setting_path", ""))
    if not setting_dir:
        raise ValueError("Inference requires --setting_path (args.json and config.json from training)")
    if not setting_dir.exists():
        raise ValueError(f"--setting_path not found: {setting_dir}")

    args_path = setting_dir / "args.json"
    config_path = setting_dir / "config.json"
    if not args_path.exists():
        raise ValueError(f"Missing saved args file: {args_path}")
    if not config_path.exists():
        raise ValueError(f"Missing saved config file: {config_path}")

    with open(args_path, "r", encoding="utf-8") as f:
        stored_args_dict = json.load(f)
    with open(config_path, "r", encoding="utf-8") as f:
        stored_config_dict = json.load(f)

    runtime_args_dict = dict(stored_args_dict)
    for key, value in vars(n_args).items():
        if value is not None:
            runtime_args_dict[key] = value
    runtime_args_dict["mode"] = "inference"
    runtime_args = Namespace(**runtime_args_dict)

    # =======================
    # 2) Init config / runtime (same as training)
    # =======================
    config, runtime_ctx = get_seismic_config(runtime_args)
    device = runtime_ctx["device"]
    is_logger = runtime_ctx["is_logger"]
    world_size = runtime_ctx["world_size"]
    local_rank = runtime_ctx["local_rank"]
    experts_name = runtime_ctx["experts_name"]
    experts_name_str = runtime_ctx["experts_name_str"]

    def _recursive_update(obj, payload):
        for k, v in payload.items():
            if isinstance(v, dict) and hasattr(obj, k):
                child = getattr(obj, k)
                if hasattr(child, "__dict__"):
                    _recursive_update(child, v)
                else:
                    setattr(obj, k, v)
            else:
                setattr(obj, k, v)

    _recursive_update(config, stored_config_dict)

    # =======================
    # 3) Norm stats (--status_json first, else data_dict in checkpoint)
    # =======================
    checkpoint = None
    checkpoint_path = None
    model_path_arg = getattr(runtime_args, "model_path", None)
    if model_path_arg:
        checkpoint_path = Path(model_path_arg)
        if not checkpoint_path.exists():
            raise ValueError(f"[MoE] Model file not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    elif experts_name_str != "all":
        raise ValueError("[MoE] Inference requires --model_path")

    stats_dict = None
    status_json = getattr(runtime_args, "status_json", None)
    if status_json and Path(status_json).exists():
        with open(status_json, "r", encoding="utf-8") as f:
            status_payload = json.load(f)
        if config.family == "all":
            stats_dict = status_payload.get("overall")
        else:
            stats_dict = status_payload.get("per_type", {}).get(config.family)

    if stats_dict is None and isinstance(checkpoint, dict):
        stats_dict = checkpoint.get("data_dict")

    if stats_dict is None:
        hint = " (ALL mode: provide --status_json for norm stats if no global checkpoint)" if experts_name_str == "all" else ""
        raise ValueError(f"Cannot get normalization stats; pass --status_json or include data_dict in checkpoint.{hint}")

    # =======================
    # 4) Transforms & Zarr DataLoader
    # =======================
    k_value = float(getattr(runtime_args, "k", 1.0))
    input_transform = Compose([
        T.LogTransform(k=k_value),
        T.MinMaxNormalize(
            T.log_transform(stats_dict["input_min"], k=k_value),
            T.log_transform(stats_dict["input_max"], k=k_value),
        ),
    ])
    output_transform = Compose([
        T.MinMaxNormalize(stats_dict["output_min"], stats_dict["output_max"])
    ])
    input_inverse_transform = Compose([
        T.InverseMinMaxNormalize(
            T.log_transform(stats_dict["input_min"], k=k_value),
            T.log_transform(stats_dict["input_max"], k=k_value),
        ),
        T.InverseLogTransform(k=k_value),
    ])
    output_inverse_transform = Compose([
        T.InverseMinMaxNormalize(stats_dict["output_min"], stats_dict["output_max"])
    ])

    from neuralop.data.datasets.seismic_dataset import SeismicDataProcessor
    data_processor = SeismicDataProcessor(
        channel_dim=config.channel_dim,
        input_transform=input_transform,
        output_transform=output_transform,
        config=config,
    )

    zarr_path = getattr(runtime_args, "zarr_path", None)
    if not zarr_path:
        raise ValueError("Inference requires zarr path (--zarr_path)")
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise ValueError(f"Zarr path not found: {zarr_path}")

    eval_split = getattr(runtime_args, "eval_split", "val")
    test_dataset = ZarrSeismicDataset(
        zarr_path=str(zarr_path),
        split=eval_split,
        input_transform=None,
        output_transform=None,
        expect_input_shape=(1, 1000, 350),
        to_float32=True,
    )
    test_dataset_with_transform = TransformedSubset(test_dataset, data_processor)

    infer_one = getattr(runtime_args, "infer_one", None)
    if infer_one is not None:
        infer_one = int(infer_one)
        dataset_len = len(test_dataset_with_transform)
        if infer_one < 0 or infer_one >= dataset_len:
            raise ValueError(f"--infer_one out of range: {infer_one} (valid: 0..{dataset_len - 1})")
        test_dataset_with_transform = Subset(test_dataset_with_transform, [infer_one])
        if is_logger:
            print(f"[Inference] infer_one={infer_one}; running single-sample inference.")

    effective_test_batch_size = 1 if infer_one is not None else int(config.test_batch_size)
    if getattr(runtime_args, "distributed", False) and world_size > 1 and infer_one is None:
        test_sampler = DistributedSampler(
            test_dataset_with_transform,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False,
            drop_last=False,
        )
        num_workers = max(0, getattr(runtime_args, "num_workers", 0) // 2)
        test_loader = DataLoader(
            test_dataset_with_transform,
            sampler=test_sampler,
            batch_size=effective_test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        test_sampler = None
        num_workers = max(0, getattr(runtime_args, "num_workers", 0))
        test_loader = DataLoader(
            test_dataset_with_transform,
            batch_size=effective_test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    if len(test_loader) == 0:
        raise RuntimeError("Test DataLoader is empty; check dataset config.")

    model_state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    router_state_dict = checkpoint.get("router_state_dict") if isinstance(checkpoint, dict) else None
    expert_state_dict = checkpoint.get("expert_state_dict") if isinstance(checkpoint, dict) else None
    encoder_state_dict = checkpoint.get("encoder_state_dict") if isinstance(checkpoint, dict) else None

    # =======================
    # 5) Build EMO (encoder + MoE)
    # =======================
    sample_batch = next(iter(test_loader))
    config.in_channels = int(sample_batch["input"].shape[1])

    encoder_loaded_from_ckpt = False
    encoder_model = None
    if getattr(config, "use_encoder", False):
        num_types = int(getattr(config, "v_type_num", 10) or 10)
        type_act = "identity" if getattr(config, "train_encoder", False) else "softmax"
        encoder_model = get_encoder(
            in_channels=config.in_channels,
            out_channels=128,
            num_types=num_types,
            type_act=type_act,
            backbone=config.backbone,
        ).to(device)
        encoder_model.eval()
        for p in encoder_model.parameters():
            p.requires_grad_(False)

        enc_ckpt = getattr(runtime_args, "encoder_path", None)
        if enc_ckpt:
            missing, unexpected = load_encoder_weights(encoder_model, enc_ckpt, map_location=device, strict=False)
            if is_logger:
                print(f"[Encoder] Loaded inference weights from {enc_ckpt}.")
                if missing:
                    print(f"[Encoder] Missing keys: {missing}")
                if unexpected:
                    print(f"[Encoder] Unexpected keys: {unexpected}")
        elif encoder_state_dict is not None:
            missing, unexpected = load_encoder_weights(encoder_model, encoder_state_dict, map_location=device, strict=False)
            encoder_loaded_from_ckpt = True
            if is_logger and (missing or unexpected):
                print(f"[Encoder] Loaded encoder from checkpoint: missing {missing}, unexpected {unexpected}")
        elif is_logger:
            print("[Encoder] No encoder_path and no encoder_state_dict; using random encoder init.")

        with torch.no_grad():
            probe_inputs = sample_batch["input"].to(device, non_blocking=True)
            encoder_probe, _, _ = encoder_model(probe_inputs)
        config.moe_in_channels = int(encoder_probe.shape[1])
        del encoder_probe
    else:
        config.moe_in_channels = config.in_channels
        if is_logger:
            print("[Encoder] use_encoder=False: use raw input in inference.")

    # --- Experts ---
    if experts_name_str == "all":
        experts_dir = getattr(config, "use_experts_path", None) or getattr(runtime_args, "experts_path", None)
        if not experts_dir:
            raise ValueError("[Experts] experts_name_str='all' requires expert weight dir (config.use_experts_path or --experts_path)")
        if not Path(experts_dir).exists():
            raise ValueError(f"[Experts] Expert weight dir not found: {experts_dir}")

        experts = load_moe_experts(
            experts_config=getattr(config, "load_expert_configs", config.expert_configs),
            in_channels=config.moe_in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels,
            model_path=experts_dir,
            is_specific=config.is_specific,
            map_location=device,
            type_dict=config.type_id,
            moe_mode=getattr(config, "moe_mode", "standard"),
        )
        if is_logger:
            print(f"[Experts] (ALL) Loaded experts from {experts_dir}")
    else:
        if getattr(config, "use_moe", False) and getattr(config, "use_experts_path", None):
            experts = load_moe_experts(
                experts_config=getattr(config, "load_expert_configs", config.expert_configs),
                in_channels=config.moe_in_channels,
                out_channels=config.out_channels,
                hidden_channels=config.hidden_channels,
                model_path=config.use_experts_path,
                is_specific=config.is_specific,
                map_location=device,
                type_dict=config.type_id,
                moe_mode=getattr(config, "moe_mode", "standard"),
            )
        else:
            experts = ExpertFactory.create_expert_ensemble(
                expert_configs=config.expert_configs,
                in_channels=config.moe_in_channels,
                out_channels=config.out_channels,
                hidden_channels=config.hidden_channels,
            )

    # --- MoE ---
    moe_method = getattr(config, "moe_method", "basic")
    if moe_method == "afmoe":
        router_alpha = float(getattr(config, "router_alpha", 0.0))
        band_sharpness = float(getattr(config, "band_sharpness", 20.0))
        freq_affinity_sharpness = float(getattr(config, "freq_affinity_sharpness", 10.0))
        use_soft_bands = bool(getattr(config, "use_soft_bands", True))
        enable_freq_attn = bool(getattr(config, "enable_freq_attn", True))
        enable_band_mixing = bool(getattr(config, "enable_band_mixing", True))
        enable_band_decomposition = bool(getattr(config, "enable_band_decomposition", False))

        if is_logger:
            print(
                "[MoE] Using AdaptiveFreqMoE (afmoe), "
                f"alpha={router_alpha}, band_sharpness={band_sharpness}, "
                f"freq_affinity_sharpness={freq_affinity_sharpness}, "
                f"use_soft_bands={use_soft_bands}, enable_freq_attn={enable_freq_attn}, "
                f"enable_band_mixing={enable_band_mixing}, enable_band_decomposition={enable_band_decomposition}"
            )

        moe = AdaptiveFreqMoE(
            experts=experts,
            in_channels=config.moe_in_channels,
            topk=config.top_k,
            alpha=router_alpha,
            band_sharpness=band_sharpness,
            freq_affinity_sharpness=freq_affinity_sharpness,
            use_soft_bands=use_soft_bands,
            enable_freq_attn=enable_freq_attn,
            enable_band_mixing=enable_band_mixing,
            enable_band_decomposition=enable_band_decomposition,
        ).to(device)
    else:
        if is_logger:
            print(f"[MoE] Using standard MOEOperator (moe_method={moe_method})")
        moe = MOEOperator(
            experts=experts,
            in_channels=config.moe_in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels,
            top_k=config.top_k,
            noisy_gating=config.noisy_gating,
            fusion_type=config.fusion_type,
            router_hidden_dim=config.router_hidden_dim,
            moe_mode=getattr(config, "moe_mode", "standard"),
            is_logger=is_logger,
            router_type=config.router_type,
            s_processor_type=config.s_processor_type,
            w_processor_type=config.w_processor_type,
            beta=config.beta,
            is_specific=config.is_specific,
            is_classifier=config.is_classifier,
            batch_size=config.batch_size,
            v_type_num=config.v_type_num,
            use_expert_memory_proxy=config.use_gpu_proxy,
            use_encoder=getattr(config, "use_encoder", False),
            device=device,
        )

    emo = EMO(encoder_model, moe, pass_encoder_logits_as_weights=True).to(device)
    emo.eval()

    # =======================
    # 6) Load weights (mirror training)
    # =======================
    use_deepspeed = bool(getattr(runtime_args, "use_deepspeed", False))
    target_model = emo
    target_moe = getattr(target_model, "moe", None)

    if (not use_deepspeed) and (experts_name_str == "all"):
        router_path = getattr(runtime_args, "router_path", None)
        if router_path:
            _load_router_weights(target_moe, router_path, map_location=device, is_logger=is_logger)
        elif router_state_dict is not None and (target_moe is not None) and hasattr(target_moe, "router") and (target_moe.router is not None):
            missing, unexpected = target_moe.router.load_state_dict(router_state_dict, strict=False)
            if is_logger and (missing or unexpected):
                print(f"[Inference][all][router] missing: {missing}, unexpected: {unexpected}")
        else:
            raise ValueError("[Router] experts_name_str='all' needs --router_path or checkpoint.router_state_dict")

        if hasattr(target_model, "encoder") and (encoder_state_dict is not None) and not getattr(runtime_args, "encoder_path", None) and not encoder_loaded_from_ckpt:
            missing, unexpected = load_encoder_weights(target_model.encoder, encoder_state_dict, map_location=device, strict=False)
            if is_logger and (missing or unexpected):
                print(f"[Inference][all][encoder] missing: {missing}, unexpected: {unexpected}")
            encoder_loaded_from_ckpt = True
        elif encoder_state_dict is not None and getattr(runtime_args, "encoder_path", None) and is_logger:
            print("[Inference][all][encoder] --encoder_path set; skip encoder from checkpoint")
    else:
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint is not a dict; cannot parse state_dict.")

        if model_state_dict is not None:
            missing, unexpected = target_model.load_state_dict(model_state_dict, strict=False)
            if is_logger and (missing or unexpected):
                print(f"[Inference][generic][model] missing: {missing}, unexpected: {unexpected}")
            elif is_logger:
                print("[Inference][generic][model] Loaded model_state_dict from checkpoint")
        elif router_state_dict is not None:
            if (target_moe is not None) and hasattr(target_moe, "router") and (target_moe.router is not None):
                missing, unexpected = target_moe.router.load_state_dict(router_state_dict, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Inference][generic][router-only] missing: {missing}, unexpected: {unexpected}")
                elif is_logger:
                    print("[Inference][generic][router-only] Loaded router_state_dict from checkpoint")
            else:
                raise ValueError("[MoE][generic] checkpoint has only router_state_dict but model has no moe.router")
        else:
            raise ValueError("checkpoint has neither model_state_dict nor router_state_dict; cannot restore MoE.")

        if experts_name is not None and isinstance(experts_name, (list, tuple)) and len(experts_name) == 1 and experts_name[0] != "all":
            resume_expert_path = getattr(runtime_args, "resume_expert_path", None)
            if resume_expert_path and os.path.exists(resume_expert_path):
                try:
                    expert_blob = torch.load(resume_expert_path, map_location="cpu", weights_only=False)
                    expert_sd = expert_blob.get("expert_state_dict", None)
                    if expert_sd is not None and (target_moe is not None) and hasattr(target_moe, "experts") and len(target_moe.experts) > 0:
                        missing, unexpected = target_moe.experts[0].load_state_dict(expert_sd, strict=False)
                        if is_logger and (missing or unexpected):
                            print(f"[Inference][single-expert][experts[0]] missing: {missing}, unexpected: {unexpected}")
                        if is_logger:
                            print(f"[Inference][single-expert] Overwrote experts[0] from {resume_expert_path}")
                except Exception as e:
                    if is_logger:
                        print(f"[Inference][single-expert] Failed to load resume_expert_path: {e}")
            elif expert_state_dict is not None and (target_moe is not None) and hasattr(target_moe, "experts") and len(target_moe.experts) > 0:
                missing, unexpected = target_moe.experts[0].load_state_dict(expert_state_dict, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Inference][single-expert][experts[0]] missing: {missing}, unexpected: {unexpected}")

        if hasattr(target_model, "encoder") and (encoder_state_dict is not None) and not getattr(runtime_args, "encoder_path", None) and not encoder_loaded_from_ckpt:
            missing, unexpected = load_encoder_weights(target_model.encoder, encoder_state_dict, map_location=device, strict=False)
            if is_logger and (missing or unexpected):
                print(f"[Inference][generic][encoder] missing: {missing}, unexpected: {unexpected}")
            encoder_loaded_from_ckpt = True
        elif encoder_state_dict is not None and getattr(runtime_args, "encoder_path", None) and is_logger:
            print("[Inference][generic][encoder] --encoder_path set; skip encoder from checkpoint")

        if router_state_dict is not None and (target_moe is not None) and hasattr(target_moe, "router") and (target_moe.router is not None):
            try:
                missing, unexpected = target_moe.router.load_state_dict(router_state_dict, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Inference][generic][router] Router overlay: missing: {missing}, unexpected: {unexpected}")
            except Exception as e:
                if is_logger:
                    print(f"[Inference][generic][router] Failed to overlay router_state_dict: {e}")

    # =======================
    # 7) Inference loop + metrics + spectral check
    # =======================
    amp_enabled = bool(getattr(config, "use_amp", False)) and device.type == "cuda"

    default_root = getattr(config, "experiment_dir", None) or getattr(config, "output_dir", None) or setting_dir
    output_root = Path(getattr(n_args, "output_dir", None) or default_root)
    results_dir = output_root / "inference"
    log_path = results_dir / "metrics.txt"
    img_path = results_dir / "vis"
    results_dir.mkdir(parents=True, exist_ok=True)
    img_path.mkdir(parents=True, exist_ok=True)

    metrics_module = SeismicMetrics()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("    Epoch    |    Train Loss    |    Val Loss    |    MAE    |    MSE    |    PSNR    |    RMSE    |    SSIM    |\n")

    mse_sum = mae_sum = psnr_sum = rmse_sum = ssim_sum = 0.0
    batch_count = 0
    visual_payload = None
    band_accum = {}

    # Two accumulators: encoder vs interpolation
    spec_enc = SpectralAccumulator(name="encoder")
    spec_int = SpectralAccumulator(name="bilinear_interp")

    # Optional band limits from config/args (defaults otherwise)
    # e.g. config.low_band=(0.05,0.3), config.high_band=(0.4,0.85)
    if hasattr(config, "low_band") and hasattr(config, "high_band"):
        try:
            spec_enc.low_band = tuple(config.low_band)
            spec_enc.high_band = tuple(config.high_band)
            spec_int.low_band = tuple(config.low_band)
            spec_int.high_band = tuple(config.high_band)
        except Exception:
            pass

    with torch.no_grad():
        global_sample_id = 0

        for batch in tqdm.tqdm(test_loader, desc=f"Inference ({eval_split})", disable=not is_logger):
            inputs = batch["input"].to(device, non_blocking=True)  # [B,1,1000,350]
            targets = batch.get("output")
            if targets is None:
                continue
            targets = targets.to(device, non_blocking=True).to(dtype=torch.float32)  # [B,1,70,70]

            # --- Forward ---
            if amp_enabled:
                with torch.amp.autocast(device_type=device.type, enabled=True, dtype=torch.bfloat16):
                    preds, aux, enc_weights = emo(inputs)
            else:
                preds, aux, enc_weights = emo(inputs)

            preds = preds.to(dtype=torch.float32)

            # --- Spatial metrics ---
            mse_sum += metrics_module.calculate_mse(preds, targets)
            mae_sum += metrics_module.calculate_mae(preds, targets)
            psnr_sum += metrics_module.calculate_psnr(preds, targets)
            rmse_sum += metrics_module.calculate_rmse(preds, targets)
            ssim_sum += metrics_module.calculate_ssim(preds, targets)
            batch_count += 1

            if getattr(config, "enable_freq_metrics", False):
                band_metrics = metrics_module.calculate_freq_band_metrics(preds, targets)
                for name, vals in band_metrics.items():
                    acc = band_accum.setdefault(
                        name,
                        {"rel_l2": 0.0, "mae": 0.0, "pred_energy_ratio": 0.0, "tgt_energy_ratio": 0.0, "count": 0},
                    )
                    acc["rel_l2"] += vals["rel_l2"]
                    acc["mae"] += vals["mae"]
                    acc["pred_energy_ratio"] += vals["pred_energy_ratio"]
                    acc["tgt_energy_ratio"] += vals["tgt_energy_ratio"]
                    acc["count"] += 1

            # --- Denorm to physical domain (for spectra) ---
            inputs_cpu = inputs.detach().cpu()
            preds_cpu = preds.detach().cpu()
            targets_cpu = targets.detach().cpu()

            if input_inverse_transform is not None:
                inputs_cpu = input_inverse_transform(inputs_cpu)
            if output_inverse_transform is not None:
                preds_cpu = output_inverse_transform(preds_cpu)
                targets_cpu = output_inverse_transform(targets_cpu)

            # --- Encoder u_c: channel-mean readout ---
            encoded_cpu = None
            if encoder_model is not None:
                if amp_enabled:
                    with torch.amp.autocast(device_type=device.type, enabled=True, dtype=torch.bfloat16):
                        z, _, _ = encoder_model(inputs)
                else:
                    z, _, _ = encoder_model(inputs)
                encoded_cpu = z.detach().cpu()  # [B,C,H,W]

            if encoded_cpu is not None and input_inverse_transform is not None:
                encoded_cpu = input_inverse_transform(encoded_cpu)
            
            # --- Interp u_int: bilinear resize to (H,W) ---
            H, W = preds_cpu.shape[-2], preds_cpu.shape[-1]
            u_int = F.interpolate(
                inputs_cpu.to(dtype=torch.float32),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )  # [B,1,H,W]

            # --- Per-sample accumulation (full batch) ---
            B = preds_cpu.shape[0]
            for b in range(B):
                y_hat = preds_cpu[b, 0].numpy()
                y_gt = targets_cpu[b, 0].numpy()

                # interpolation always exists
                u_int_b = u_int[b, 0].numpy()
                spec_int.add_sample(u_front=u_int_b, y_pred=y_hat, y_gt=y_gt, sample_id=global_sample_id)

                # encoder only if available
                if encoded_cpu is not None:
                    u_enc_b = encoded_cpu[b].mean(0).numpy()  # [H,W]
                    spec_enc.add_sample(u_front=u_enc_b, y_pred=y_hat, y_gt=y_gt, sample_id=global_sample_id)
                else:
                    # Skip encoder group if no encoder
                    pass

                global_sample_id += 1

            # --- Single visualization batch (original behavior) ---
            if visual_payload is None and is_logger:
                logits_cpu = enc_weights.detach().cpu() if enc_weights is not None else None
                visual_payload = {
                    "inputs": inputs_cpu,
                    "targets": targets_cpu,
                    "preds": preds_cpu,
                    "encoded": encoded_cpu,
                    "logits": logits_cpu,
                    "batch": batch,
                }

    if batch_count == 0:
        raise RuntimeError("No evaluable samples in test set.")

    mse = mse_sum / batch_count
    mae = mae_sum / batch_count
    psnr = psnr_sum / batch_count
    rmse = rmse_sum / batch_count
    ssim = ssim_sum / batch_count

    if is_logger:
        print(f"Inference metrics: MSE={mse:.6f} | MAE={mae:.6f} | PSNR={psnr:.4f} | RMSE={rmse:.6f} | SSIM={ssim:.6f}")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"  Inference |        -        |        -        | {mae:.6f} | {mse:.6f} | {psnr:.4f} | {rmse:.6f} | {ssim:.6f} |\n")

    # =======================
    # 8) DDP: all_gather_object to merge per-rank records
    # =======================
    def _dist_ready():
        return (dist is not None) and dist.is_available() and dist.is_initialized()

    def _gather_records(records):
        if not _dist_ready():
            return records
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, records)
        merged = []
        for g in gathered:
            if g:
                merged.extend(g)
        return merged

    spec_int.records = _gather_records(spec_int.records)
    if encoder_model is not None:
        spec_enc.records = _gather_records(spec_enc.records)

    # =======================
    # 9) Export spectral check (rank0 / logger saves)
    # =======================
    if is_logger:
        out_payload = {
            "interpolation": spec_int.summary(),
        }
        if encoder_model is not None:
            out_payload["encoder"] = spec_enc.summary()

        out_json = results_dir / "spectral_assumption_check.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2, ensure_ascii=False)

        print(f"[AssumptionCheck] Saved: {out_json}")

        # Optional brief summary in log (e.g. for appendix)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n[AssumptionCheck]\n")
            for key in out_payload:
                f.write(f"  [{key}]\n")
                summ = out_payload[key]
                f.write(f"    count={summ.get('count')}\n")
                for k in ["a1_ratio_h", "a2_ratio_l", "gain_H_pred_over_u", "gain_L_pred_over_u", "spec_corr_u_gt", "spec_corr_pred_gt"]:
                    if k in summ:
                        vv = summ[k]
                        f.write(f"    {k}: mean={vv.get('mean')}, p05={vv.get('p05')}, p50={vv.get('p50')}, p95={vv.get('p95')}\n")
                if "A3_empirical_bounds" in summ:
                    bnd = summ["A3_empirical_bounds"]
                    f.write(f"    A3_empirical_bounds: {bnd}\n")

    # =======================
    # 10) Frequency-band metrics summary (unchanged)
    # =======================
    band_summary = {}
    if band_accum:
        for name, agg in band_accum.items():
            cnt = max(1, int(agg.get("count", 1)))
            band_summary[name] = {
                "rel_l2": float(agg["rel_l2"] / cnt),
                "mae": float(agg["mae"] / cnt),
                "pred_energy_ratio": float(agg["pred_energy_ratio"] / cnt),
                "tgt_energy_ratio": float(agg["tgt_energy_ratio"] / cnt),
                "count": cnt,
            }
        if is_logger:
            pretty = " | ".join(
                f"{k}: rel_l2={v['rel_l2']:.4f}, mae={v['mae']:.4f}, predE={v['pred_energy_ratio']:.3f}, tgtE={v['tgt_energy_ratio']:.3f}"
                for k, v in band_summary.items()
            )
            print(f"[FreqMetrics] {pretty}")

        freq_json = results_dir / "freq_band_metrics.json"
        with open(freq_json, "w", encoding="utf-8") as f:
            json.dump(band_summary, f, indent=2, ensure_ascii=False)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("[FreqMetrics]\n")
            for k, v in band_summary.items():
                f.write(
                    f"  {k}: rel_l2={v['rel_l2']:.6f}, mae={v['mae']:.6f}, predE={v['pred_energy_ratio']:.6f}, tgtE={v['tgt_energy_ratio']:.6f}\n"
                )

    # =======================
    # 11) Viz + router viz (original behavior)
    # =======================
    if visual_payload is not None:
        inputs_vis = visual_payload["inputs"]
        targets_vis = visual_payload["targets"]
        preds_vis = visual_payload["preds"]
        logits_vis = visual_payload["logits"]
        encoded_vis = visual_payload["encoded"]
        vis_batch = visual_payload["batch"]
        num_samples = min(4, inputs_vis.shape[0])

        visualize_results(inputs_vis, targets_vis, preds_vis, save_dir=img_path, max_samples=num_samples)

        save_type_predictions_txt(
            logits=logits_vis,
            batch=vis_batch,
            save_dir=img_path,
            epoch=0,
            config=config,
            filename="type_predictions.txt",
            append=False,
            is_logger=is_logger,
        )

        analyze_fourier_domain(inputs_vis, targets_vis, preds_vis, save_dir=img_path, max_samples=num_samples)

        visualize_encoded(encoded_vis, save_dir=img_path, max_samples=num_samples, selection="l2")

        if isinstance(emo.moe, AdaptiveFreqMoE):
            router_vis_dir = img_path / "router"
            router_vis_dir.mkdir(parents=True, exist_ok=True)

            try:
                router_stats = emo.moe.get_router_stats()
                visualize_router_selection_from_stats(
                    router_stats,
                    save_dir=router_vis_dir,
                    epoch=0,
                    router_name=getattr(config, "router_type", "sar"),
                    tb_writer=None,
                    wandb_run=None,
                    global_step=None,
                )
                if is_logger:
                    print(f"[RouterVis] Band/expert selection stats saved to {router_vis_dir}")
            except Exception as e:
                if is_logger:
                    print(f"[RouterVis] Band/expert selection viz failed: {e}")

            try:
                routed_bands = emo.moe.get_last_routed_bands()
                if routed_bands is not None:
                    band_centers = router_stats.get("band_centers", None) if "router_stats" in locals() else None
                    visualize_routed_bands(
                        routed_bands,
                        save_dir=router_vis_dir,
                        sample_idx=0,
                        max_channels=4,
                        band_centers=band_centers,
                        tb_writer=None,
                        wandb_run=None,
                        global_step=None,
                    )
                    if is_logger:
                        print(f"[RouterVis] routed_bands viz saved to {router_vis_dir}")
            except Exception as e:
                if is_logger:
                    print(f"[RouterVis] routed_bands viz failed: {e}")

    if test_sampler is not None and hasattr(test_sampler, "set_epoch"):
        test_sampler.set_epoch(0)

    print(f"Inference done. Outputs: {results_dir}")

if __name__ == '__main__':
    args = build_argparser_and_parse()
    if args.mode in ('train', 'train_encoder'):
        run_training(args)
    elif args.mode == 'inference':
        run_inference(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

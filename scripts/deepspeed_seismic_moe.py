#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSpeed training script (minimal CLI: override config class via JSON, or restore via setting_path).
- Does not use DeepSpeed built-in MoE; keeps custom MoE/router/expert implementation.
- ds_config controls ZeRO/offload/activation checkpointing (bf16/fp16 globally disabled).
- Encoder runs under autocast(bf16) only; MoE/backbone stay FP32 (AMP disabled).
- Uses DeepSpeed Adam (adam_w_mode=True) + built-in WarmupCosineLR (ratios derived from local config).
- engine.backward()/engine.step() & engine.save/load_checkpoint().
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.utils.tensorboard import SummaryWriter
from argparse import Namespace
from torchvision.transforms import Compose

import deepspeed
from deepspeed import zero

# tqdm progress bar
from tqdm.auto import tqdm
# ===== In-repo modules (unchanged layout) =====
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuralop.data.datasets.seismic_dataset import SeismicDataProcessor
from scripts import transforms as T
from neuralop.data.datasets.zarr_seismic_dataset import ZarrSeismicDataset
from neuralop.data.datasets.seismic_dataset import SeismicDataset
from scripts.train_seismic_moe import TransformedSubset
from neuralop.data.dataloader.zarr_seismic_dataloader import build_loaders
from neuralop.models.encoder import get_encoder
from neuralop.models.moe import MOEOperator
from neuralop.models.expert_factory import ExpertFactory
from neuralop.models.EMO import EMO
from neuralop.losses import L1L2Loss, SobelLoss, FourierMag_L1
from neuralop.layers.spectral_convolution import SpectralConv
from tltorch.factorized_tensors.core import FactorizedTensor
from neuralop.utils import count_model_params
from utils import *  # get_seismic_config, EarlyStopping, SeismicMetrics, load_moe_experts, load_encoder_weights, plot_loss_curve, safe_random_split

try:
    import wandb  # noqa: F401
except Exception:
    wandb = None


# ---------------- Utils: recursively apply JSON onto config ----------------
def _recursive_update(obj, payload):
    """Recursively write dict fields onto config object (including nested sub-objects)."""
    for k, v in payload.items():
        if isinstance(v, dict) and hasattr(obj, k):
            child = getattr(obj, k)
            if hasattr(child, "__dict__"):
                _recursive_update(child, v)
            else:
                setattr(obj, k, v)
        else:
            setattr(obj, k, v)


# ---------------- Utils: move batch recursively to device (no dtype coercion; forward controls dtypes) ----------------
def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_to_device(v, device) for v in x]
        return type(x)(t) if not isinstance(x, tuple) else tuple(t)
    return x


def _load_args_and_config(args):
    """
    Returns: merged_args (Namespace), config (dict or obj-like), runtime_ctx.
    - If --setting_path: load setting_path/args.json and config.json, then overlay current CLI.
    - Else use --config_path JSON and overlay CLI onto the config class.
    """
    base_args = args

    # setting_path mode
    if args.setting_path is not None:
        setting_dir = Path(args.setting_path)
        if not setting_dir.exists():
            raise ValueError(f"Setting directory does not exist: {setting_dir}")
        args_path = setting_dir / "args.json"
        config_path = setting_dir / "config.json"
        if not args_path.exists():
            raise ValueError(f"Missing saved training args file: {args_path}")
        if not config_path.exists():
            raise ValueError(f"Missing saved training config file: {config_path}")

        with open(args_path, "r", encoding="utf-8") as f:
            stored_args_dict = json.load(f)
        with open(config_path, "r", encoding="utf-8") as f:
            stored_config_dict = json.load(f)

        runtime_args_dict = dict(stored_args_dict)
        for key, value in vars(base_args).items():
            if value is not None:
                runtime_args_dict[key] = value
        runtime_args = Namespace(**runtime_args_dict)

        config, runtime_ctx = get_seismic_config(runtime_args)
        _recursive_update(config, stored_config_dict)
        return runtime_args, config, runtime_ctx

    # config_path mode
    if args.config_path is None:
        raise ValueError("Provide either --config_path or --setting_path")
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise ValueError(f"Missing config file: {config_path}")

    config, runtime_ctx = get_seismic_config(base_args)
    with open(config_path, "r", encoding="utf-8") as f:
        stored_config_dict = json.load(f)
    _recursive_update(config, stored_config_dict)
    return base_args, config, runtime_ctx


# ---------------- SpectralConv dtype/device patch ----------------
def patch_spectral_conv_dtype_fix(module, verbose=True):
    """
    Patch module (may contain several SpectralConv):
    - If weight is FactorizedTensor, call .to_tensor() first.
    - If x is complex and weight is not, promote weight to complex.
    - Align weight dtype/device to x.
    - Then call original _contract for actual frequency-domain multiply.
    """
    total = 0
    for m in module.modules():
        if isinstance(m, SpectralConv):
            total += 1
            if not hasattr(m, "_contract_impl"):
                m._contract_impl = m._contract  # save original

                def _wrapped_contract(x, weight, separable=False, _m=m):
                    # 1) factorized -> dense
                    if isinstance(weight, FactorizedTensor):
                        weight = weight.to_tensor()
                    # 2) complex/real alignment
                    if torch.is_complex(x) and not torch.is_complex(weight):
                        weight = torch.complex(weight, torch.zeros_like(weight))
                    # 3) dtype/device alignment (follow x)
                    weight = weight.to(dtype=x.dtype, device=x.device)
                    # 4) delegate to original impl
                    return _m._contract_impl(x, weight, separable=separable)

                m._contract = _wrapped_contract
    if verbose:
        print(f"[Patch] SpectralConv dtype/device wrapper applied to {total} modules.")


def _autocast_device_str(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else device.type


def run_training_deepspeed(args):
    """DeepSpeed training: encoder bf16 (autocast), MoE/backbone FP32; DS optimizer/scheduler from ds-config."""
    # -------- Basics --------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1) Merge/load args and config
    args, config, runtime_ctx = _load_args_and_config(args)

    # runtime
    device = runtime_ctx["device"]
    is_logger = runtime_ctx["is_logger"]
    world_size = runtime_ctx["world_size"]
    local_rank = args.local_rank
    experts_name = runtime_ctx["experts_name"]
    experts_name_str = runtime_ctx["experts_name_str"]

    # -------- Data (same as original) --------
    data_dict = None
    if getattr(args, "zarr_path", None) or getattr(config, "zarr_path", None):
        zarr_path = getattr(args, "zarr_path", None) or getattr(config, "zarr_path", None)
        json_path = getattr(args, "status_json", None) or getattr(config, "status_json", None)
        assert zarr_path is not None, "zarr_path is required when using Zarr data"
        assert json_path is not None, "For Zarr dataset, provide normalization stats JSON (status_json)"

        with open(json_path, "r") as f:
            data_dict_raw = json.load(f)

        if config.family == 'all':
            data_dict = data_dict_raw['overall']
        else:
            data_dict = data_dict_raw['per_type'][config.family]

        k_val = getattr(config, "k", getattr(args, "k", 1.0))
        input_transform = Compose([
            T.LogTransform(k=k_val),
            T.MinMaxNormalize(T.log_transform(data_dict['input_min'], k=k_val),
                              T.log_transform(data_dict['input_max'], k=k_val))
        ])
        output_transform = Compose([
            T.MinMaxNormalize(data_dict['output_min'], data_dict['output_max'])
        ])
        # Inverse transforms (for visualization)
        input_inverse_transform = Compose([
            T.InverseMinMaxNormalize(T.log_transform(data_dict['input_min'], k=args.k), T.log_transform(data_dict['input_max'], k=args.k)),
            T.InverseLogTransform(k=args.k)
        ])
        output_inverse_transform = Compose([
            T.InverseMinMaxNormalize(data_dict['output_min'], data_dict['output_max'])
        ])
        data_processor = SeismicDataProcessor(
            input_transform=input_transform, output_transform=output_transform,
            channel_dim=config.channel_dim, config=config
        )

        train_dataset = ZarrSeismicDataset(
            zarr_path=zarr_path, split='train',
            input_transform=None, output_transform=None,
            expect_input_shape=(1, 1000, 350), to_float32=True
        )
        val_dataset = ZarrSeismicDataset(
            zarr_path=zarr_path, split='val',
            input_transform=None, output_transform=None,
            expect_input_shape=(1, 1000, 350), to_float32=True
        )

        train_dataset_with_transform = TransformedSubset(train_dataset, data_processor)
        val_dataset_with_transform = TransformedSubset(val_dataset, data_processor)

        train_loader, val_loader, train_sampler, val_sampler = build_loaders(
            args=args, config=config,
            train_dataset_with_transform=train_dataset_with_transform,
            val_dataset_with_transform=val_dataset_with_transform,
            chunks=32,
            world_size=world_size, local_rank=local_rank
        )
    else:
        # family -> val split ratio
        if config.family in ['curve_vel_a', 'curve_vel_b', 'flat_vel_a', 'flat_vel_b']:
            val_ratio = 6 / 30
        elif config.family in ['curve_fault_a', 'curve_fault_b', 'flat_fault_a', 'flat_fault_b']:
            val_ratio = 6 / 54
        elif config.family in ['style_a', 'style_b', 'style_style_a', 'style_style_b']:
            val_ratio = 7 / 67
        else:
            raise ValueError("Unsupported family")

        full_dataset = SeismicDataset(
            data_dir=config.data_dir, family=config.family,
            is_specific=config.is_specific, split='train',
            concat_channels=config.concat_channels, config=config
        )

        dataset_size = len(full_dataset)
        train_size, val_size = safe_random_split(dataset_size, [1 - val_ratio, val_ratio])
        if is_logger:
            print(f"Dataset size: {dataset_size} | train: {train_size} | val: {val_size}")

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(getattr(config, "seed", getattr(args, "seed", 42)))
        )
        data_dict = full_dataset.getStats()

        k_val = getattr(config, "k", getattr(args, "k", 1.0))
        input_transform = Compose([
            T.LogTransform(k=k_val),
            T.MinMaxNormalize(T.log_transform(data_dict['input_min'], k=k_val),
                              T.log_transform(data_dict['input_max'], k=k_val))
        ])
        output_transform = Compose([
            T.MinMaxNormalize(data_dict['output_min'], data_dict['output_max'])
        ])
        # Inverse transforms (for visualization)
        input_inverse_transform = Compose([
            T.InverseMinMaxNormalize(T.log_transform(data_dict['input_min'], k=args.k), T.log_transform(data_dict['input_max'], k=args.k)),
            T.InverseLogTransform(k=args.k)
        ])
        output_inverse_transform = Compose([
            T.InverseMinMaxNormalize(data_dict['output_min'], data_dict['output_max'])
        ])
        data_processor = SeismicDataProcessor(
            input_transform=input_transform, output_transform=output_transform,
            channel_dim=config.channel_dim, config=config
        )

        train_dataset_with_transform = TransformedSubset(train_dataset, data_processor)
        val_dataset_with_transform = TransformedSubset(val_dataset, data_processor)

        if getattr(config, "distributed", getattr(args, "distributed", True)) and world_size > 1:
            train_num_workers = max(0, getattr(config, "num_workers", getattr(args, "num_workers", 8)) // 2)
            train_sampler = DistributedSampler(
                train_dataset_with_transform, num_replicas=world_size, rank=local_rank,
                drop_last=True, shuffle=True
            )
            train_loader = DataLoader(
                train_dataset_with_transform, sampler=train_sampler,
                batch_size=config.batch_size, shuffle=False,
                num_workers=train_num_workers, pin_memory=True,
                persistent_workers=train_num_workers > 0
            )
            val_num_workers = train_num_workers
            val_sampler = DistributedSampler(
                val_dataset_with_transform, num_replicas=world_size, rank=local_rank, drop_last=False
            )
            val_loader = DataLoader(
                val_dataset_with_transform, sampler=val_sampler,
                batch_size=config.test_batch_size, shuffle=False,
                num_workers=val_num_workers, pin_memory=True,
                persistent_workers=val_num_workers > 0
            )
        else:
            train_num_workers = max(0, getattr(config, "num_workers", getattr(args, "num_workers", 8)))
            train_loader = DataLoader(
                train_dataset_with_transform, batch_size=config.batch_size, shuffle=True,
                num_workers=train_num_workers, pin_memory=True, persistent_workers=train_num_workers > 0
            )
            val_loader = DataLoader(
                val_dataset_with_transform, batch_size=config.test_batch_size, shuffle=False,
                num_workers=train_num_workers, pin_memory=True, persistent_workers=train_num_workers > 0
            )

    if is_logger:
        prefetch = getattr(train_loader, "prefetch_factor", None)
        if prefetch is not None:
            print(f"prefetch_factor={prefetch}")

    # -------- Shape probe / in_channels --------
    sample_batch = next(iter(train_loader))
    config.in_channels = sample_batch['input'].shape[1]
    if is_logger:
        print(f"[Shape] input={sample_batch['input'].shape}, output={sample_batch['output'].shape}")
        print(f"[Config] in={config.in_channels}, out={config.out_channels}, hidden={config.hidden_channels}")
        print(f"[Experts] num={len(config.expert_configs)}")

    # -------- Optional encoder --------
    encoder_model = None
    encoder_freeze = False
    if config.use_encoder:
        if config.train_encoder:
            encoder_model = get_encoder(
                in_channels=config.in_channels, out_channels=128, num_types=10, type_act='identity',
                backbone=config.backbone
            )
        else:
            encoder_model = get_encoder(
                in_channels=config.in_channels, out_channels=128, num_types=10, type_act='softmax',
                backbone=config.backbone
            )
        if getattr(args, "encoder_path", None):
            missing, unexpected = load_encoder_weights(
                encoder_model, args.encoder_path, map_location="cpu", strict=False
            )
            if is_logger:
                print(f"[Encoder] load {args.encoder_path}")
                if missing: print(f"  missing: {missing}")
                if unexpected: print(f"  unexpected: {unexpected}")
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

        encoder_model.eval()
        with torch.no_grad():
            _probe, _, _ = encoder_model(sample_batch['input'])
        moe_in_channels = _probe.shape[1]
        del _probe
        if encoder_freeze:
            encoder_model.eval()
        else:
            encoder_model.train()
        if is_logger:
            print(f"[Encoder] moe_in_channels={moe_in_channels}")
    else:
        moe_in_channels = config.in_channels
        if is_logger:
            print("[Encoder] use_encoder=False")

    config.moe_in_channels = moe_in_channels

    # -------- Experts and MoE backbone (unchanged implementation) --------
    if config.use_moe and config.use_experts_path:
        experts = load_moe_experts(
            experts_config=config.load_expert_configs,
            in_channels=moe_in_channels, out_channels=config.out_channels, hidden_channels=config.hidden_channels,
            model_path=config.use_experts_path, is_specific=config.is_specific,
            map_location="cpu", type_dict=config.type_id, moe_mode=config.moe_mode,
        )
    else:
        experts = ExpertFactory.create_expert_ensemble(
            expert_configs=config.expert_configs,
            in_channels=moe_in_channels, out_channels=config.out_channels, hidden_channels=config.hidden_channels
        )

    moe_model = MOEOperator(
        experts=experts,
        in_channels=moe_in_channels, out_channels=config.out_channels, hidden_channels=config.hidden_channels,
        top_k=config.top_k, noisy_gating=config.noisy_gating, fusion_type=config.fusion_type,
        router_hidden_dim=config.router_hidden_dim, moe_mode=getattr(config, "moe_mode", "standard"),
        is_logger=is_logger, router_type=config.router_type,
        s_processor_type=config.s_processor_type, w_processor_type=config.w_processor_type, beta=config.beta,
        is_specific=config.is_specific, is_classifier=config.is_classifier,
        batch_size=config.batch_size, v_type_num=config.v_type_num, use_expert_memory_proxy=config.use_gpu_proxy
    )

    # -------- Apply SpectralConv patch before DeepSpeed init --------
    patch_spectral_conv_dtype_fix(moe_model, verbose=is_logger)

    # -------- Combined model (matches train_seismic_moe.py) --------
    emo_model = EMO(
        encoder_model if config.use_encoder else None,
        moe_model,
        pass_encoder_logits_as_weights=True,
    )

    if encoder_freeze and emo_model.encoder is not None:
        emo_model.freeze_encoder()
    elif emo_model.encoder is not None:
        if config.train_encoder:
            emo_model.encoder.train()
        else:
            emo_model.encoder.eval()

    class DeepSpeedEMOWrapper(torch.nn.Module):
        def __init__(self, emo: EMO, device: torch.device, train_encoder: bool):
            super().__init__()
            self.emo = emo
            self.device = device
            self.train_encoder = train_encoder

        @property
        def moe(self):
            return self.emo.moe

        def forward(self, batch):
            x = batch['input']
            y = batch.get('output', None)
            labels = None

            encoder = self.emo.encoder
            logits = None
            if encoder is not None:
                with torch.amp.autocast(device_type=_autocast_device_str(self.device), dtype=torch.bfloat16, enabled=True):
                    feat, logits, _ = encoder(x)
                x = feat.float()
            else:
                x = x.float()

            with torch.amp.autocast(device_type=_autocast_device_str(self.device), enabled=False):
                pred, aux_loss = self.emo.moe(x, logits)

            if self.train_encoder:
                labels = batch.get('v_type', batch.get('labels', None))
                return pred, y, logits, labels, aux_loss
            return pred, y, aux_loss

    train_encoder_flag = bool(
        emo_model.encoder is not None and
        (config.train_encoder or any(p.requires_grad for p in emo_model.encoder.parameters()))
    )
    model_for_engine = DeepSpeedEMOWrapper(emo_model, device=device, train_encoder=train_encoder_flag)

    # ======================= DeepSpeed: AdamW + WarmupCosineLR (ratios); global AMP forced off =======================

    # Step counts for WarmupCosineLR
    steps_per_epoch = max(1, len(train_loader))
    total_num_steps = int(getattr(config, "epochs", 100) * steps_per_epoch)

    warmup_epochs = float(getattr(config, "lr_warmup_epochs", 0.0))
    warmup_steps = int(max(0.0, warmup_epochs) * steps_per_epoch)

    base_lr = float(getattr(config, "learning_rate", 3e-4))
    weight_decay = float(getattr(config, "weight_decay", 1e-2))
    eta_min = float(getattr(config, "lr_cosine_eta_min", 1e-6))
    cos_min_ratio = 0.0 if base_lr <= 0 else max(0.0, min(1.0, eta_min / base_lr))

    warmup_min_ratio = float(getattr(config, "lr_warmup_min_ratio", 0.0))
    warmup_min_ratio = max(0.0, min(1.0, warmup_min_ratio))
    warmup_method = str(getattr(config, "lr_warmup_method", "linear")).lower()
    warmup_type = "linear" if "lin" in warmup_method else ("log" if "log" in warmup_method else "linear")

    if not args.deepspeed_config:
        raise ValueError("Provide --deepspeed_config")
    ds_cfg_path = Path(args.deepspeed_config)
    if not ds_cfg_path.exists():
        raise ValueError(f"DeepSpeed config file not found: {ds_cfg_path}")

    with open(ds_cfg_path, "r", encoding="utf-8") as f:
        ds_cfg = json.load(f)

    # Force global AMP off so DS does not cast whole model to bf16/fp16
    ds_cfg["bf16"] = {"enabled": False}
    ds_cfg["fp16"] = {"enabled": False}

    # Optimizer: DeepSpeed Adam + AdamW mode
    ds_cfg["optimizer"] = {
        "type": "Adam",
        "params": {
            "lr": base_lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": weight_decay,
            "adam_w_mode": True
        }
    }

    # Scheduler: WarmupCosineLR (ratio params)
    ds_cfg["scheduler"] = {
        "type": "WarmupCosineLR",
        "params": {
            "total_num_steps": int(total_num_steps),
            "warmup_min_ratio": float(warmup_min_ratio),
            "warmup_num_steps": int(warmup_steps),
            "warmup_type": warmup_type,
            "cos_min_ratio": float(cos_min_ratio),
            "last_batch_iteration": -1
        }
    }

    # ZeRO offload defaults if missing in file
    zero_cfg = ds_cfg.get("zero_optimization", {})
    off_cfg = zero_cfg.get("offload_optimizer", {})
    if isinstance(off_cfg, dict):
        off_cfg.setdefault("device", "cpu")
        off_cfg.setdefault("pin_memory", True)
    zero_cfg["offload_optimizer"] = off_cfg
    ds_cfg["zero_optimization"] = zero_cfg

    # DeepSpeed init: optimizer/scheduler from ds_cfg
    engine, _, _, _ = deepspeed.initialize(
        model=model_for_engine,
        model_parameters=(p for p in model_for_engine.parameters() if p.requires_grad),
        config=ds_cfg
    )

    # -------- Loss / metrics / early stopping --------
    lambda_grad_l1 = float(getattr(config, "lambda_grad_l1", 0.0))
    lambda_fourier_mag_l1 = float(getattr(config, "lambda_fourier_mag_l1", 0.0))
    lambda_ce = float(getattr(config, "lambda_ce", 0.0))

    base_loss: Callable = L1L2Loss(config.lambda_g1v, config.lambda_g2v)
    grad_loss_module = SobelLoss() if lambda_grad_l1 > 0 else None
    fourier_loss_module = FourierMag_L1() if lambda_fourier_mag_l1 > 0 else None

    # Move loss modules to this rank's device to avoid conv2d weight/input device mismatch
    if base_loss is not None:
        base_loss = base_loss.to(engine.device)
    if grad_loss_module is not None:
        grad_loss_module = grad_loss_module.to(engine.device)
    if fourier_loss_module is not None:
        fourier_loss_module = fourier_loss_module.to(engine.device)

    # Cast to FP32 in loss for dtype consistency
    if train_encoder_flag:
        def criterion(pred: torch.Tensor, gt: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
            pred = pred.float(); gt = gt.float()
            loss_dict = base_loss(pred, gt)
            total = loss_dict["loss"]
            if grad_loss_module is not None:
                total = total + lambda_grad_l1 * grad_loss_module(pred, gt)["loss"]
            if fourier_loss_module is not None:
                total = total + lambda_fourier_mag_l1 * fourier_loss_module(pred, gt)["loss"]
            ce_val = pred.new_zeros(())
            if logits is not None and labels is not None:
                ce_val = F.cross_entropy(logits.float(), labels.long(), reduction="mean")
                total = total + lambda_ce * ce_val
            return {"loss": total, "l1": loss_dict["l1"], "l2": loss_dict["l2"], "ce": ce_val.detach()}
    else:
        def criterion(pred: torch.Tensor, gt: torch.Tensor):
            pred = pred.float(); gt = gt.float()
            loss_dict = base_loss(pred, gt)
            total = loss_dict["loss"]
            if grad_loss_module is not None:
                total = total + lambda_grad_l1 * grad_loss_module(pred, gt)["loss"]
            if fourier_loss_module is not None:
                total = total + lambda_fourier_mag_l1 * fourier_loss_module(pred, gt)["loss"]
            return {"loss": total, "l1": loss_dict["l1"], "l2": loss_dict["l2"]}

    metrics = SeismicMetrics()
    use_wandb = bool(getattr(args, "use_wandb", False) and wandb is not None)

    # -------- Output dirs / logging (rank0 creates; all ranks barrier) --------
    def _slugify(text: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in str(text))

    run_group = _slugify(getattr(config, "family", "all") or "all")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = "_".join([
        _slugify(getattr(config, "model_name", "model") or "model"),
        f"router-{_slugify(getattr(config, 'router_type', 'router'))}",
        f"lr{getattr(config, 'learning_rate', 1e-3):g}",
        f"bs{getattr(config, 'batch_size', 8)}",
        _slugify(experts_name_str or "experts"),
        timestamp,
    ])

    output_root = Path(config.output_dir) / f"seismic_moe_{run_group}"
    results_dir = output_root / run_name
    log_file = results_dir / "training_log.txt"
    tb_root = Path(getattr(config, "log_root", "./runs")).expanduser()
    tb_dir = tb_root / run_group / run_name
    tb_writer: Optional[SummaryWriter] = None

    # rank0 only creates dirs
    if deepspeed.comm.get_rank() == 0:
        results_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)

    # All ranks wait for dirs
    if deepspeed.comm.get_world_size() > 1:
        deepspeed.comm.barrier()

    # rank0 writes logs/config; non-loggers still hold path strings
    if is_logger:
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        config.experiment_dir = str(results_dir)
        config.tensorboard_dir = str(tb_dir)
        with open(results_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump((config.to_dict() if hasattr(config, "to_dict") else vars(config)), f, indent=2, default=str)
        with open(results_dir / "args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, default=str)
        if data_dict is not None:
            with open(results_dir / "data_stats.json", "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=2, default=float)
        header = ("    Epoch    |    Train Loss    |    Val Loss    |    MAE    |    MSE    |    PSNR    |    RMSE    |    SSIM    |    CE    |\n"
                  if train_encoder_flag else
                  "    Epoch    |    Train Loss    |    Val Loss    |    MAE    |    MSE    |    PSNR    |    RMSE    |    SSIM    |\n")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(header)
    else:
        config.experiment_dir = str(results_dir)
        config.tensorboard_dir = str(tb_dir)

    # -------- Early stopping & resume --------
    best_val_loss = float("inf")
    best_epoch_metrics = None
    best_epoch_index = None
    start_epoch = 0

    if getattr(config, "early_stop", False):
        early_stopper = EarlyStopping(
            patience=getattr(config, "early_stop_patience", 20),
            min_delta=getattr(config, "early_stop_min_delta", 0.0),
            warmup_epochs=getattr(config, "early_stop_warmup_epochs", 10),
            mode="min"
        )
    else:
        early_stopper = None

    if args.resume_path and Path(args.resume_path).is_dir():
        load_path, _ = engine.load_checkpoint(args.resume_path, tag=None)
        if is_logger:
            print(f"[DeepSpeed] resume from {args.resume_path} | load_path={load_path}")
        start_epoch = 0

    # -------- Train/val loop (DeepSpeed-managed) --------
    try:
        for epoch in range(start_epoch, config.epochs):
            engine.train()
            if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            # Train
            aux_coef = 0.01
            train_total = 0.0
            train_base = 0.0
            train_aux = 0.0
            train_l1 = 0.0
            train_l2 = 0.0
            train_ce = 0.0
            train_steps = 0
            type_weight_hist_sample = None
            max_hist_samples = 65536
            lr_this_epoch = None

            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False, disable=not is_logger)
            for step, batch in enumerate(train_iter):
                batch = _to_device(batch, engine.device)
                out = engine(batch)
                if train_encoder_flag:
                    pred, y, logits, labels, aux_loss = out
                    loss_dict = criterion(pred, y, logits, labels)
                    if is_logger and type_weight_hist_sample is None and logits is not None:
                        flat = logits.detach().float().cpu().reshape(-1)
                        if flat.numel() > max_hist_samples:
                            flat = flat[:max_hist_samples]
                        type_weight_hist_sample = flat
                else:
                    pred, y, aux_loss = out
                    loss_dict = criterion(pred, y)
                    logits = None
                    labels = None
                if aux_loss is None:
                    aux_loss = pred.new_zeros(())
                total_loss = loss_dict["loss"] + aux_coef * aux_loss
                if not torch.isfinite(total_loss):
                    raise RuntimeError(f"Encountered non-finite loss at step {step}: {total_loss.item()}")

                engine.backward(total_loss)
                engine.step()      # scheduler stepped inside DeepSpeed

                step_total = float(total_loss.detach().item())
                train_total += step_total
                train_base += float(loss_dict["loss"].detach().item())
                train_aux += float(aux_loss.detach().item())
                if "l1" in loss_dict:
                    train_l1 += float(loss_dict["l1"])
                if "l2" in loss_dict:
                    train_l2 += float(loss_dict["l2"])
                if train_encoder_flag and "ce" in loss_dict:
                    train_ce += float(loss_dict["ce"])
                train_steps += 1

                lr_list = engine.get_lr() if hasattr(engine, "get_lr") else None
                if lr_list:
                    lr_this_epoch = float(lr_list[0])
                elif engine.optimizer is not None and engine.optimizer.param_groups:
                    lr_this_epoch = float(engine.optimizer.param_groups[0]["lr"])

                if is_logger:
                    postfix = {"train_loss": f"{step_total:.6f}"}
                    if train_encoder_flag and "ce" in loss_dict:
                        postfix["ce"] = f"{float(loss_dict['ce']):.4f}"
                    train_iter.set_postfix(postfix)

            num_steps = max(1, train_steps)
            avg_train_loss = train_total / num_steps
            avg_train_base = train_base / num_steps
            avg_train_aux = train_aux / num_steps
            avg_train_l1 = train_l1 / num_steps if num_steps > 0 else 0.0
            avg_train_l2 = train_l2 / num_steps if num_steps > 0 else 0.0
            avg_train_ce = train_ce / num_steps if (train_encoder_flag and num_steps > 0) else 0.0

            # ---------- Val ----------
            engine.eval()
            with torch.no_grad():
                # Accumulate sums for cross-rank reduction
                sum_loss = 0.0
                sum_l1 = 0.0
                sum_l2 = 0.0
                sum_psnr = 0.0
                sum_rmse = 0.0
                sum_ssim = 0.0
                sum_ce = 0.0
                cnt_batches = 0

                val_iter = tqdm(val_loader, desc=f"Eval {epoch+1}/{config.epochs}", leave=False, disable=not is_logger)
                for vstep, vbatch in enumerate(val_iter):
                    vbatch = _to_device(vbatch, engine.device)
                    out = engine(vbatch)
                    if train_encoder_flag:
                        pred, y, logits, labels, _ = out
                        loss_dict = criterion(pred, y, logits, labels)
                    else:
                        pred, y, _ = out
                        loss_dict = criterion(pred, y)
                    res = metrics(pred, y)
                    sum_loss += float(loss_dict["loss"])
                    sum_l1 += float(loss_dict["l1"])
                    sum_l2 += float(loss_dict["l2"])
                    sum_psnr += float(res.get("psnr", 0.0))
                    sum_rmse += float(res.get("rmse", 0.0))
                    sum_ssim += float(res.get("ssim", 0.0))
                    if train_encoder_flag:
                        sum_ce += float(loss_dict.get("ce", 0.0))
                    cnt_batches += 1
                    if is_logger:
                        val_iter.set_postfix(
                            val_loss=f"{float(loss_dict['loss']):.6f}",
                            mae=f"{float(loss_dict['l1']):.4f}",
                            psnr=f"{float(res.get('psnr', 0.0)):.2f}"
                        )

                # This rank's sums and count
                if train_encoder_flag:
                    local_sums = torch.tensor(
                        [sum_loss, sum_l1, sum_l2, sum_psnr, sum_rmse, sum_ssim, sum_ce],
                        dtype=torch.float64, device=engine.device
                    )
                else:
                    local_sums = torch.tensor(
                        [sum_loss, sum_l1, sum_l2, sum_psnr, sum_rmse, sum_ssim],
                        dtype=torch.float64, device=engine.device
                    )
                local_cnt = torch.tensor([cnt_batches], dtype=torch.float64, device=engine.device)

                # Cross-rank reduce: SUM
                if deepspeed.comm.get_world_size() > 1:
                    deepspeed.comm.all_reduce(local_sums, op=deepspeed.comm.ReduceOp.SUM)
                    deepspeed.comm.all_reduce(local_cnt,  op=deepspeed.comm.ReduceOp.SUM)

                # Globally consistent means
                denom = max(1.0, float(local_cnt.item()))
                val_loss = float(local_sums[0].item() / denom)
                val_mae  = float(local_sums[1].item() / denom)
                val_mse  = float(local_sums[2].item() / denom)
                val_psnr = float(local_sums[3].item() / denom)
                val_rmse = float(local_sums[4].item() / denom)
                val_ssim = float(local_sums[5].item() / denom)
                val_ce = float(local_sums[6].item() / denom) if train_encoder_flag else 0.0

            epoch_step = epoch + 1

            # TensorBoard logging (rank0)
            if tb_writer is not None and is_logger:
                tb_writer.add_scalar("train/epoch_loss", avg_train_loss, epoch_step)
                tb_writer.add_scalar("train/base_loss", avg_train_base, epoch_step)
                tb_writer.add_scalar("train/epoch_aux_loss", avg_train_aux, epoch_step)
                tb_writer.add_scalar("train/l1", avg_train_l1, epoch_step)
                tb_writer.add_scalar("train/l2", avg_train_l2, epoch_step)
                if train_encoder_flag:
                    tb_writer.add_scalar("train/ce", avg_train_ce, epoch_step)
                tb_writer.add_scalar("val/loss", val_loss, epoch_step)
                tb_writer.add_scalar("val/mae", val_mae, epoch_step)
                tb_writer.add_scalar("val/mse", val_mse, epoch_step)
                tb_writer.add_scalar("val/psnr", val_psnr, epoch_step)
                tb_writer.add_scalar("val/rmse", val_rmse, epoch_step)
                tb_writer.add_scalar("val/ssim", val_ssim, epoch_step)
                if train_encoder_flag:
                    tb_writer.add_scalar("val/ce", val_ce, epoch_step)
                tb_writer.add_scalars(
                    "loss/epoch",
                    {"train": avg_train_loss, "val": val_loss},
                    epoch_step,
                )
                if lr_this_epoch is not None:
                    tb_writer.add_scalar("train/learning_rate", lr_this_epoch, epoch_step)
                if type_weight_hist_sample is not None:
                    tb_writer.add_histogram("encoder/type_logits", type_weight_hist_sample, epoch_step)

            # ---------- Log file (logger only prints/writes) ----------
            if is_logger:
                if train_encoder_flag:
                    row = f"{epoch_step:>8d} | {avg_train_loss:>14.6f} | {val_loss:>12.6f} | {val_mae:>8.6f} | {val_mse:>8.6f} | {val_psnr:>8.4f} | {val_rmse:>8.6f} | {val_ssim:>8.6f} | {val_ce:>8.6f} |\n"
                else:
                    row = f"{epoch_step:>8d} | {avg_train_loss:>14.6f} | {val_loss:>12.6f} | {val_mae:>8.6f} | {val_mse:>8.6f} | {val_psnr:>8.4f} | {val_rmse:>8.6f} | {val_ssim:>8.6f} |\n"
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(row)

            if use_wandb and is_logger:
                wandb_log = {
                    "epoch": epoch_step,
                    "train/loss": avg_train_loss,
                    "train/base_loss": avg_train_base,
                    "train/aux_loss": avg_train_aux,
                    "train/l1": avg_train_l1,
                    "train/l2": avg_train_l2,
                    "val/loss": val_loss,
                    "val/mae": val_mae,
                    "val/mse": val_mse,
                    "val/psnr": val_psnr,
                    "val/rmse": val_rmse,
                    "val/ssim": val_ssim,
                }
                if train_encoder_flag:
                    wandb_log["train/ce"] = avg_train_ce
                    wandb_log["val/ce"] = val_ce
                if lr_this_epoch is not None:
                    wandb_log["train/learning_rate"] = lr_this_epoch
                wandb.log(wandb_log, step=epoch_step)

            # Router (adamv) validation hook
            if hasattr(engine, "module") and hasattr(engine.module, "moe"):
                moe_module = engine.module.moe
                router = getattr(moe_module, "router", None)
                router_type = getattr(moe_module, "router_type", "")
                if router_type == "adamv" and router is not None and hasattr(router, "step_validation"):
                    signal = router.step_validation(val_loss)
                    should_break = (signal == "should_break")
                    if deepspeed.comm.get_world_size() > 1:
                        flag_tensor = torch.tensor([1 if should_break else 0], device=engine.device, dtype=torch.int64)
                        deepspeed.comm.all_reduce(flag_tensor, op=deepspeed.comm.ReduceOp.MAX)
                        should_break = bool(flag_tensor.item())
                    if should_break:
                        current_k = int(getattr(router, "k", getattr(router, "top_k", moe_module.top_k)))
                        new_k = max(1, current_k - 1)
                        if hasattr(router, "k"):
                            router.k = new_k
                        if hasattr(router, "top_k"):
                            router.top_k = new_k
                        if hasattr(router, "fixed"):
                            router.fixed = True
                        moe_module.top_k = new_k
                        moe_module.w_k = max(0, moe_module.num_experts - new_k)
                        if deepspeed.comm.get_world_size() > 1:
                            shared_k = torch.tensor([new_k], device=engine.device, dtype=torch.int64)
                            deepspeed.comm.broadcast(shared_k, src=0)
                            new_k = int(shared_k.item())
                            if hasattr(router, "k"):
                                router.k = new_k
                            if hasattr(router, "top_k"):
                                router.top_k = new_k
                            moe_module.top_k = new_k
                            moe_module.w_k = max(0, moe_module.num_experts - new_k)
                        if is_logger:
                            print(f"[Router] AdamV adjust top_k -> {new_k}")

            epoch_metrics = {
                "epoch": epoch_step,
                "train_loss": float(avg_train_loss),
                "train_base_loss": float(avg_train_base),
                "train_aux_loss": float(avg_train_aux),
                "train_l1": float(avg_train_l1),
                "train_l2": float(avg_train_l2),
                "val_loss": float(val_loss),
                "mae": float(val_mae),
                "mse": float(val_mse),
                "psnr": float(val_psnr),
                "rmse": float(val_rmse),
                "ssim": float(val_ssim),
                "val_mae": float(val_mae),
                "val_mse": float(val_mse),
                "val_psnr": float(val_psnr),
                "val_rmse": float(val_rmse),
                "val_ssim": float(val_ssim),
            }
            if train_encoder_flag:
                epoch_metrics["train_ce"] = float(avg_train_ce)
                epoch_metrics["val_ce"] = float(val_ce)
            if lr_this_epoch is not None:
                epoch_metrics["learning_rate"] = float(lr_this_epoch)
            epoch_metrics["router_top_k"] = int(getattr(engine.module.moe, "top_k", 0))

            # ---------- Global best sync; improved on same scale ----------
            if deepspeed.comm.get_world_size() > 1:
                best_t = torch.tensor([best_val_loss], dtype=torch.float64, device=engine.device)
                deepspeed.comm.all_reduce(best_t, op=deepspeed.comm.ReduceOp.MIN)
                best_val_loss_global = float(best_t.item())
            else:
                best_val_loss_global = best_val_loss

            improved = (val_loss < (best_val_loss_global - 1e-12))

            # ====== Save checkpoint: all ranks must enter together ======
            if improved:
                # Update best to globally consistent new value first
                best_val_loss = val_loss
                save_tag = f"best_e{epoch+1}"

                # Older DS versions lack save_zero_checkpoint_only
                try:
                    engine.save_checkpoint(results_dir.as_posix(), tag=save_tag, save_zero_checkpoint_only=True)
                except TypeError:
                    engine.save_checkpoint(results_dir.as_posix(), tag=save_tag)

                # Barrier after save to avoid races
                if deepspeed.comm.get_world_size() > 1:
                    deepspeed.comm.barrier()

                if is_logger:
                    print(f"[DeepSpeed] saved checkpoint: {save_tag}")

                best_epoch_metrics = dict(epoch_metrics)
                best_epoch_index = epoch + 1
                if is_logger:
                    with open(results_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                        json.dump(best_epoch_metrics, f, indent=2)
            # ====== End save-best block ======

            # Save latest checkpoint
            try:
                engine.save_checkpoint(results_dir.as_posix(), tag="last", save_zero_checkpoint_only=True)
            except TypeError:
                engine.save_checkpoint(results_dir.as_posix(), tag="last")
            if deepspeed.comm.get_world_size() > 1:
                deepspeed.comm.barrier()
            if is_logger:
                with open(results_dir / "last_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(epoch_metrics, f, indent=2)

            if is_logger:
                print(
                    f"[Epoch {epoch_step:03d}] train_loss={avg_train_loss:.6f} "
                    f"val_loss={val_loss:.6f} psnr={val_psnr:.4f} mae={val_mae:.6f}"
                )

            # Visualization (logger only; on improvement or every vis_every epochs pattern)
            if is_logger and (improved or ((epoch + 1) % int(getattr(config, "vis_every", 5)) == 0)):
                try:
                    vis_batch = next(iter(val_loader))
                    vis_batch = _to_device(vis_batch, engine.device)
                    engine.eval()
                    with torch.no_grad():
                        vis_out = engine(vis_batch)
                        if train_encoder_flag:
                            vis_pred, vis_y, vis_logits = vis_out[0], vis_out[1], vis_out[2]
                        else:
                            vis_pred, vis_y, vis_logits = vis_out[0], vis_out[1], None

                    # Denormalize
                    vis_in = vis_batch['input']
                    vis_in_v = input_inverse_transform(vis_in)
                    vis_pred_v = output_inverse_transform(vis_pred)
                    vis_y_v = output_inverse_transform(vis_y)

                    num_samples = min(4, vis_pred.shape[0])
                    vis_dir = results_dir / f"vis_epoch_{epoch+1}"
                    visualize_results(
                        vis_in_v, vis_y_v, vis_pred_v,
                        save_dir=vis_dir, max_samples=num_samples,
                        tb_writer=tb_writer,
                        wandb_run=(wandb if use_wandb else None),
                        global_step=epoch
                    )
                    if vis_logits is not None:
                        save_type_predictions_txt(
                            logits=vis_logits,
                            batch=vis_batch,
                            save_dir=vis_dir,
                            epoch=epoch,
                            config=config,
                            filename="type_predictions.txt",
                            append=True,
                            is_logger=is_logger,
                        )
                    analyze_fourier_domain(
                        vis_in_v, vis_y_v, vis_pred_v,
                        save_dir=results_dir / f"fourier_analysis_epoch_{epoch+1}",
                        max_samples=num_samples,
                        tb_writer=tb_writer,
                        wandb_run=(wandb if use_wandb else None),
                        global_step=epoch,
                    )
                except Exception as _viz_err:
                    print(f"[Viz] Skipped visualization due to error: {_viz_err}")

            if early_stopper is not None and early_stopper.step(val_loss, epoch):
                if is_logger:
                    print("[EarlyStop] Early stopping triggered.")
                break

        if is_logger:
            print(f"VAL_LOSS:{best_val_loss}", flush=True)
            plot_loss_curve(log_file, save_path=results_dir / "loss_curve.png")
            if tb_writer is not None:
                hparam_summary = {
                    "family": str(getattr(config, "family", "")),
                    "router_type": str(getattr(config, "router_type", "")),
                    "learning_rate": float(getattr(config, "learning_rate", 0.0)),
                    "batch_size": int(getattr(config, "batch_size", 0)),
                    "top_k": int(getattr(config, "top_k", 1)),
                }
                metric_summary = {"best/val_loss": float(best_val_loss)}
                if best_epoch_metrics is not None:
                    metric_summary.update({
                        "best/train_loss": best_epoch_metrics.get("train_loss", float("nan")),
                        "best/psnr": best_epoch_metrics.get("psnr", float("nan")),
                        "best/mse": best_epoch_metrics.get("mse", float("nan")),
                        "best/mae": best_epoch_metrics.get("mae", float("nan")),
                        "best/rmse": best_epoch_metrics.get("rmse", float("nan")),
                        "best/ssim": best_epoch_metrics.get("ssim", float("nan")),
                        "best/epoch": float(best_epoch_index or 0),
                    })
                    if "val_ce" in best_epoch_metrics:
                        metric_summary["best/val_ce"] = best_epoch_metrics.get("val_ce", float("nan"))
                    if "train_ce" in best_epoch_metrics:
                        metric_summary["best/train_ce"] = best_epoch_metrics.get("train_ce", float("nan"))
                tb_writer.add_hparams(hparam_summary, metric_summary)
                print("Wrote hparams to TensorBoard")
    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        # Clean shutdown
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass

    return engine, best_val_loss


# ---------------- CLI ----------------
def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser("DeepSpeed training for seismic MoE (config JSON driven or setting_path restored)")

    # Mutually exclusive: setting_path OR config_path
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--setting_path", type=str, default=None,
                   help="Directory with args.json & config.json (restore experiment config)")
    g.add_argument("--config_path", type=str, default=None,
                   help="Path to config JSON (overrides config class)")

    p.add_argument("--deepspeed_config", type=str, default="scripts/ds_zero3_bf16_offload.json", help="Path to ds_config.json")

    # Optional runtime data paths
    p.add_argument("--zarr_path", type=str, default=None)
    p.add_argument("--status_json", type=str, default=None)

    # resume / encoder
    p.add_argument("--resume_path", type=str, default=None, help="DeepSpeed checkpoint dir to resume")
    p.add_argument("--encoder_path", type=str, default=None, help="Pretrained encoder weights (optional)")

    # Minimal args for get_seismic_config
    p.add_argument("--distributed", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--local_rank", type=int, default=-1, help="provided by deepspeed/torchrun")

    return p


def main():
    args = build_arg_parser().parse_args()
    # Common accel/stability env vars (keep as-is)
    os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Optional: avoid Triton autotune cache on NFS slowing shutdown (adjust path per machine)
    cache_dir = os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.triton_cache"))
    os.makedirs(cache_dir, exist_ok=True)

    # Normalize paths
    if args.config_path:
        args.config_path = Path(args.config_path)
    if args.setting_path:
        args.setting_path = Path(args.setting_path)
    args.deepspeed_config = Path(args.deepspeed_config)
    if args.resume_path:
        args.resume_path = Path(args.resume_path)

    engine, best_val = run_training_deepspeed(args)
    if deepspeed.comm.get_rank() == 0:
        print(f"[DONE] Best Val Loss: {best_val:.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSpeed 版训练脚本（精简命令行：用 config JSON 覆盖 config 类，或用 setting_path 恢复）
- 不使用 DeepSpeed 自带 MoE；保留你的自定义 MoE/路由/专家实现
- ds_config 管理 ZeRO/offload/activation checkpointing（bf16/fp16 全局关闭）
- Encoder 仅在 autocast(bf16) 下运行；MoE/主干全程 FP32（禁用 AMP）
- 使用 DeepSpeed Adam(adam_w_mode=True) + 内置 WarmupCosineLR（比例参数由本地 config 推导）
- engine.backward()/engine.step() & engine.save/load_checkpoint()
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

# >>> NEW: 进度条与绘图工具导入
from tqdm.auto import tqdm
from utils.plot_fig import (
    analyze_fourier_domain, visualize_results, save_type_predictions_txt
)

# ===== 你的工程内模块（保持原结构）=====
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
from neuralop.losses import L1L2Loss, SobelLoss, FourierMag_L1
from neuralop.layers.spectral_convolution import SpectralConv
from tltorch.factorized_tensors.core import FactorizedTensor
from neuralop.utils import count_model_params
from utils import *  # 包含：get_seismic_config / EarlyStopping / SeismicMetrics / load_moe_experts / load_encoder_weights / plot_loss_curve / safe_random_split

try:
    import wandb  # noqa: F401
except Exception:
    wandb = None


# ---------------- 工具：用 JSON 递归覆盖 config ----------------
def _recursive_update(obj, payload):
    """将 dict 递归写回到 config 对象（含嵌套子对象）"""
    for k, v in payload.items():
        if isinstance(v, dict) and hasattr(obj, k):
            child = getattr(obj, k)
            if hasattr(child, "__dict__"):
                _recursive_update(child, v)
            else:
                setattr(obj, k, v)
        else:
            setattr(obj, k, v)


# ---------------- 工具：把 batch 递归搬到指定设备（不做 dtype 强制转换！交给 forward 内部控制） ----------------
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
    返回：merged_args(Namespace), config(dict or obj-like), runtime_ctx
    - 若提供 --setting_path：读取 setting_path/args.json 与 config.json，然后用当前 CLI 覆盖
    - 否则使用 --config_path 读取配置，并按 CLI 传参覆盖 config 类
    """
    base_args = args

    # setting_path 模式
    if args.setting_path is not None:
        setting_dir = Path(args.setting_path)
        if not setting_dir.exists():
            raise ValueError(f"配置目录不存在: {setting_dir}")
        args_path = setting_dir / "args.json"
        config_path = setting_dir / "config.json"
        if not args_path.exists():
            raise ValueError(f"缺少训练时保存的参数文件: {args_path}")
        if not config_path.exists():
            raise ValueError(f"缺少训练时保存的配置文件: {config_path}")

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

    # config_path 模式
    if args.config_path is None:
        raise ValueError("需要提供 --config_path 或 --setting_path 其中之一")
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise ValueError(f"缺少训练时保存的配置文件: {config_path}")

    config, runtime_ctx = get_seismic_config(base_args)
    with open(config_path, "r", encoding="utf-8") as f:
        stored_config_dict = json.load(f)
    _recursive_update(config, stored_config_dict)
    return base_args, config, runtime_ctx


# ---------------- SpectralConv dtype/device 修复补丁 ----------------
def patch_spectral_conv_dtype_fix(module, verbose=True):
    """
    给 module（包含若干 SpectralConv）打补丁：
    - 若权重为 FactorizedTensor，先 .to_tensor()
    - 若 x 为 complex 而权重非 complex，则将权重提升为 complex
    - 将权重对齐到 x 的 dtype 与 device
    - 之后调用原先的 _contract 做真正的频域乘法
    """
    total = 0
    for m in module.modules():
        if isinstance(m, SpectralConv):
            total += 1
            if not hasattr(m, "_contract_impl"):
                m._contract_impl = m._contract  # 保存原函数

                def _wrapped_contract(x, weight, separable=False, _m=m):
                    # 1) factorized -> dense
                    if isinstance(weight, FactorizedTensor):
                        weight = weight.to_tensor()
                    # 2) 复/实对齐
                    if torch.is_complex(x) and not torch.is_complex(weight):
                        weight = torch.complex(weight, torch.zeros_like(weight))
                    # 3) dtype/device 对齐（以 x 为准）
                    weight = weight.to(dtype=x.dtype, device=x.device)
                    # 4) 交回原实现
                    return _m._contract_impl(x, weight, separable=separable)

                m._contract = _wrapped_contract
    if verbose:
        print(f"[Patch] SpectralConv dtype/device wrapper applied to {total} modules.")


def _autocast_device_str(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else device.type


def run_training_deepspeed(args):
    """DeepSpeed 版本训练：Encoder 用 bf16(autocast)，MoE/主干 FP32；DS 优化器/调度器由 ds-config 决定"""
    # -------- 基础设置 --------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1) 合并/加载 args 与 config
    args, config, runtime_ctx = _load_args_and_config(args)

    # runtime 信息
    device = runtime_ctx["device"]
    is_logger = runtime_ctx["is_logger"]
    world_size = runtime_ctx["world_size"]
    local_rank = args.local_rank
    experts_name = runtime_ctx["experts_name"]
    experts_name_str = runtime_ctx["experts_name_str"]

    # -------- 数据准备（与你原逻辑一致）--------
    if getattr(args, "zarr_path", None) or getattr(config, "zarr_path", None):
        zarr_path = getattr(args, "zarr_path", None) or getattr(config, "zarr_path", None)
        json_path = getattr(args, "status_json", None) or getattr(config, "status_json", None)
        assert zarr_path is not None, "使用zarr数据格式时需要 zarr_path"
        assert json_path is not None, "使用zarr数据集格式，需要指定归一化统计量 json（status_json）"

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
        # >>> NEW: 反归一化（根据你参考代码保留，未改其他逻辑）
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
            world_size=world_size, local_rank=local_rank
        )
    else:
        # family → 验证比例
        if config.family in ['curve_vel_a', 'curve_vel_b', 'flat_vel_a', 'flat_vel_b']:
            val_ratio = 6 / 30
        elif config.family in ['curve_fault_a', 'curve_fault_b', 'flat_fault_a', 'flat_fault_b']:
            val_ratio = 6 / 54
        elif config.family in ['style_a', 'style_b', 'style_style_a', 'style_style_b']:
            val_ratio = 7 / 67
        else:
            raise ValueError("不支持的 family")

        full_dataset = SeismicDataset(
            data_dir=config.data_dir, family=config.family,
            is_specific=config.is_specific, split='train',
            concat_channels=config.concat_channels, config=config
        )

        dataset_size = len(full_dataset)
        train_size, val_size = safe_random_split(dataset_size, [1 - val_ratio, val_ratio])
        if is_logger:
            print(f"数据集总大小: {dataset_size} | 训练: {train_size} | 验证: {val_size}")

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
        # >>> NEW: 反归一化（根据你参考代码保留，未改其他逻辑）
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

    # -------- 形状探测 / in_channels --------
    sample_batch = next(iter(train_loader))
    config.in_channels = sample_batch['input'].shape[1]
    if is_logger:
        print(f"[Shape] input={sample_batch['input'].shape}, output={sample_batch['output'].shape}")
        print(f"[Config] in={config.in_channels}, out={config.out_channels}, hidden={config.hidden_channels}")
        print(f"[Experts] num={len(config.expert_configs)}")

    # -------- 可选 encoder --------
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

    # -------- 专家与 MoE 主干（保持你的实现）--------
    if config.use_moe and config.use_experts_path:
        experts = load_moe_experts(
            experts_config=config.load_expert_configs,
            in_channels=moe_in_channels, out_channels=config.out_channels, hidden_channels=config.hidden_channels,
            model_path=config.use_experts_path, is_specific=config.is_specific,
            map_location="cpu", type_dict=config.type_id
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

    # -------- 在 DeepSpeed 初始化之前，打 SpectralConv 修复补丁 --------
    patch_spectral_conv_dtype_fix(moe_model, verbose=is_logger)

    # -------- 组合模型（Encoder bf16，MoE FP32）--------
    class CombinedModel(torch.nn.Module):
        def __init__(self, enc, moe, train_encoder: bool, device: torch.device):
            super().__init__()
            self.enc = enc
            self.moe = moe
            self.train_encoder = train_encoder
            self.device = device

        def forward(self, batch):
            x = batch['input']
            y = batch.get('output', None)
            labels = batch.get('labels', None)

            logits = None
            # 1) 仅 Encoder 在 bf16 autocast 下运行
            if self.enc is not None:
                with torch.amp.autocast(device_type=_autocast_device_str(self.device), dtype=torch.bfloat16, enabled=True):
                    feat, logits, _ = self.enc(x)  # 计算发生在 bf16
                x = feat.float()  # 回到 FP32，避免后续 bf16 传染到 MoE
            else:
                x = x.float()

            # 2) MoE/主干在 FP32 小岛中运行（显式禁用 AMP）
            with torch.amp.autocast(device_type=_autocast_device_str(self.device), enabled=False):
                pred, aux_loss = self.moe(x, logits)  # 全程 FP32
            if self.train_encoder:
                return pred, y, logits, labels, aux_loss
            return pred, y, aux_loss

    train_encoder_flag = bool(
        config.use_encoder and (encoder_model is not None) and
        (config.train_encoder or any(p.requires_grad for p in encoder_model.parameters()))
    )
    model_for_engine = CombinedModel(encoder_model if config.use_encoder else None, moe_model, train_encoder_flag, device=device)

    # ======================= DeepSpeed：AdamW+WarmupCosineLR（比例），并强制关闭全局 AMP =======================

    # 步数（供 WarmupCosineLR 使用）
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
        raise ValueError("需提供 --deepspeed_config")
    ds_cfg_path = Path(args.deepspeed_config)
    if not ds_cfg_path.exists():
        raise ValueError(f"DeepSpeed 配置文件不存在: {ds_cfg_path}")

    with open(ds_cfg_path, "r", encoding="utf-8") as f:
        ds_cfg = json.load(f)

    # 强制关闭全局 AMP，避免 DS 把全模型参数转 bf16/fp16
    ds_cfg["bf16"] = {"enabled": False}
    ds_cfg["fp16"] = {"enabled": False}

    # 优化器：DeepSpeed 通用 Adam + AdamW 模式
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

    # 调度器：WarmupCosineLR（比例入参）
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

    # ZeRO Offload（若原文件未给出，则补默认）
    zero_cfg = ds_cfg.get("zero_optimization", {})
    off_cfg = zero_cfg.get("offload_optimizer", {})
    if isinstance(off_cfg, dict):
        off_cfg.setdefault("device", "cpu")
        off_cfg.setdefault("pin_memory", True)
    zero_cfg["offload_optimizer"] = off_cfg
    ds_cfg["zero_optimization"] = zero_cfg

    # DeepSpeed 初始化：交由 ds_cfg 创建优化器/调度器
    engine, _, _, _ = deepspeed.initialize(
        model=model_for_engine,
        model_parameters=(p for p in model_for_engine.parameters() if p.requires_grad),
        config=ds_cfg
    )

    # -------- Loss/Metric/早停 --------
    lambda_grad_l1 = float(getattr(config, "lambda_grad_l1", 0.0))
    lambda_fourier_mag_l1 = float(getattr(config, "lambda_fourier_mag_l1", 0.0))
    lambda_ce = float(getattr(config, "lambda_ce", 0.0))

    base_loss: Callable = L1L2Loss(config.lambda_g1v, config.lambda_g2v)
    grad_loss_module = SobelLoss() if lambda_grad_l1 > 0 else None
    fourier_loss_module = FourierMag_L1() if lambda_fourier_mag_l1 > 0 else None

    # 把 loss 模块搬到当前 rank 的设备，避免 conv2d 权重/输入 device 不一致
    if base_loss is not None:
        base_loss = base_loss.to(engine.device)
    if grad_loss_module is not None:
        grad_loss_module = grad_loss_module.to(engine.device)
    if fourier_loss_module is not None:
        fourier_loss_module = fourier_loss_module.to(engine.device)

    # 统一在损失处转 FP32，避免 dtype 不一致
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

    # -------- 输出目录 / 日志（修复：rank0 创建，所有 rank barrier）--------
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

    # 仅 rank0 创建目录
    if deepspeed.comm.get_rank() == 0:
        results_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)

    # 全员等待目录就绪
    if deepspeed.comm.get_world_size() > 1:
        deepspeed.comm.barrier()

    # rank0 写入日志/配置；非 logger 也持有路径字符串
    if is_logger:
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        config.experiment_dir = str(results_dir)
        config.tensorboard_dir = str(tb_dir)
        with open(results_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump((config.to_dict() if hasattr(config, "to_dict") else vars(config)), f, indent=2, default=str)
        with open(results_dir / "args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, default=str)
        header = ("    Epoch    |    Train Loss    |    Val Loss    |    MAE    |    MSE    |    PSNR    |    RMSE    |    SSIM    |    CE    |\n"
                  if train_encoder_flag else
                  "    Epoch    |    Train Loss    |    Val Loss    |    MAE    |    MSE    |    PSNR    |    RMSE    |    SSIM    |\n")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(header)
    else:
        config.experiment_dir = str(results_dir)
        config.tensorboard_dir = str(tb_dir)

    # -------- 早停 & 恢复 --------
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

    # -------- 训练/验证循环（DeepSpeed 托管）--------
    try:
        for epoch in range(start_epoch, config.epochs):
            engine.train()
            if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            # Train
            run_train = {"loss": 0.0}
            # >>> NEW: 训练进度条（仅 logger 进程展示）
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False, disable=not is_logger)
            for step, batch in enumerate(train_iter):
                batch = _to_device(batch, engine.device)
                out = engine(batch)
                if train_encoder_flag:
                    pred, y, logits, labels, aux_loss = out
                    loss_dict = criterion(pred, y, logits, labels)
                else:
                    pred, y, aux_loss = out
                    loss_dict = criterion(pred, y)
                if aux_loss is None:
                    aux_loss = pred.new_zeros(())
                loss = loss_dict["loss"] + 0.01 * aux_loss
                engine.backward(loss)
                engine.step()      # 调度器由 DeepSpeed 内部推进
                step_loss = float(loss.detach().item())
                run_train["loss"] += step_loss
                if is_logger:
                    train_iter.set_postfix(train_loss=f"{step_loss:.6f}")

            num_steps = max(1, step + 1)
            train_loss = run_train["loss"] / num_steps

            # Val
            engine.eval()
            with torch.no_grad():
                v_meter = {"loss": 0.0, "mae": 0.0, "mse": 0.0, "psnr": 0.0, "rmse": 0.0, "ssim": 0.0}
                # >>> NEW: 验证进度条（仅 logger 进程展示）
                val_iter = tqdm(val_loader, desc=f"Eval {epoch+1}/{config.epochs}", leave=False, disable=not is_logger)
                for vstep, vbatch in enumerate(val_iter):
                    vbatch = _to_device(vbatch, engine.device)
                    out = engine(vbatch)
                    pred, y = out[0], out[1]
                    res = metrics(pred, y)
                    for k in v_meter:
                        v_meter[k] += float(res.get(k, 0.0))
                    if is_logger:
                        val_iter.set_postfix(
                            val_loss=f"{float(res.get('loss', 0.0)):.6f}",
                            mae=f"{float(res.get('mae', 0.0)):.4f}",
                            psnr=f"{float(res.get('psnr', 0.0)):.2f}"
                        )
                vsteps = max(1, vstep + 1)
                val_loss = v_meter["loss"] / vsteps
                val_mae  = v_meter["mae"]  / vsteps
                val_mse  = v_meter["mse"]  / vsteps
                val_psnr = v_meter["psnr"] / vsteps
                val_rmse = v_meter["rmse"] / vsteps
                val_ssim = v_meter["ssim"] / vsteps

            # 记录/保存
            if is_logger:
                if train_encoder_flag:
                    row = f"{epoch+1:>8d} | {train_loss:>14.6f} | {val_loss:>12.6f} | {val_mae:>8.6f} | {val_mse:>8.6f} | {val_psnr:>8.4f} | {val_rmse:>8.6f} | {val_ssim:>8.6f} | {'-':>6} |\n"
                else:
                    row = f"{epoch+1:>8d} | {train_loss:>14.6f} | {val_loss:>12.6f} | {val_mae:>8.6f} | {val_mse:>8.6f} | {val_psnr:>8.4f} | {val_rmse:>8.6f} | {val_ssim:>8.6f} |\n"
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(row)

            # ====== 改动点：保存 checkpoint 必须所有 rank 都执行 ======
            improved = val_loss < best_val_loss - 1e-12
            if improved:
                best_val_loss = val_loss
                save_tag = f"best_e{epoch+1}"

                # 所有 rank 都保存（避免死锁）；建议仅保存 ZeRO 分片，空间更省
                engine.save_checkpoint(
                    results_dir.as_posix(),
                    tag=save_tag,
                )

                # 保存后 barrier，避免竞态
                if deepspeed.comm.get_world_size() > 1:
                    deepspeed.comm.barrier()

                if is_logger:
                    print(f"[DeepSpeed] saved checkpoint: {save_tag}")

                best_epoch_metrics = {
                    "train_loss": float(train_loss), "val_loss": float(val_loss),
                    "mae": float(val_mae), "mse": float(val_mse),
                    "psnr": float(val_psnr), "rmse": float(val_rmse), "ssim": float(val_ssim),
                }
                best_epoch_index = epoch + 1
            # ====== 改动点结束 ======

            # >>> NEW: 每个 epoch 的可视化（仅 logger；指标提升或到达周期）
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

                    # 反归一化（若需要），否则直接用当前张量
                    vis_in = vis_batch['input']
                    vis_pred_v, vis_y_v, vis_in_v = vis_pred, vis_y, vis_in
                    vis_in_v = input_inverse_transform(vis_in)
                    vis_pred_v = output_inverse_transform(vis_pred)
                    vis_y_v = output_inverse_transform(vis_y)

                    num_samples = min(4, vis_pred.shape[0])
                    vis_dir = results_dir / f"vis_epoch_{epoch+1}"
                    visualize_results(
                        vis_in_v, vis_y_v, vis_pred_v,
                        save_dir=vis_dir, max_samples=num_samples,
                        tb_writer=tb_writer,
                        wandb_run=(wandb if wandb is not None else None),
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
                        wandb_run=(wandb if wandb is not None else None),
                        global_step=epoch,
                    )
                except Exception as _viz_err:
                    print(f"[Viz] Skipped visualization due to error: {_viz_err}")

            if early_stopper is not None and early_stopper.step(val_loss, epoch):
                if is_logger:
                    print("[EarlyStop] 触发提前停止。")
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
                tb_writer.add_hparams(hparam_summary, metric_summary)
                print("已写入hparams")
    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        # 正常收尾
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

    # 互斥：setting_path 或 config_path 二选一
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--setting_path", type=str, default=None,
                   help="指向包含 args.json & config.json 的目录（从已有实验恢复配置）")
    g.add_argument("--config_path", type=str, default=None,
                   help="直接使用配置 JSON 路径（覆盖 config 类）")

    p.add_argument("--deepspeed_config", type=str, default="scripts/ds_zero3_bf16_offload.json", help="Path to ds_config.json")

    # 可能需要的运行时数据路径
    p.add_argument("--zarr_path", type=str, default=None)
    p.add_argument("--status_json", type=str, default=None)

    # resume / encoder
    p.add_argument("--resume_path", type=str, default=None, help="DeepSpeed checkpoint dir to resume")
    p.add_argument("--encoder_path", type=str, default=None, help="预训练 encoder 权重（可选）")

    # get_seismic_config 可能用到的参数（保留少量）
    p.add_argument("--distributed", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--local_rank", type=int, default=-1, help="provided by deepspeed/torchrun")

    return p


def main():
    args = build_arg_parser().parse_args()
    # 常见加速/稳定环境变量（按需保留）
    os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 可选：避免 Triton autotune 缓存在 NFS 导致退出卡顿（根据你机器目录调整）
    cache_dir = os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.triton_cache"))
    os.makedirs(cache_dir, exist_ok=True)

    # 规范化路径
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
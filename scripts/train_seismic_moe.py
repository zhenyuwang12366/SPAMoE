# scripts/train_seismic_moe.py
"""
使用MOE（Mixture of Experts）架构训练地震数据的神经算子模型
支持分布式训练 + DeepSpeed
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

# >>> DeepSpeed（仅导入，不影响非 DS 路径）
try:
    import deepspeed
except Exception:
    deepspeed = None

# 添加项目根目录到路径
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
        raise ValueError(f"无法从 {router_path} 提取可用的 state_dict。")

    router_prefixes = ("router.", "gate.", "routing.", "router_net.", "router_module.")
    router_sd = _filter_state_by_prefix(sd, router_prefixes)

    missing, unexpected = model.load_state_dict(router_sd, strict=False)
    if is_logger:
        print(f"[Router] 已从 {router_path} 加载路由器权重（过滤前缀: {router_prefixes}）。")
        if missing:
            print(f"[Router] 缺失参数（可能是非路由器键或名称不匹配）：{missing}")
        if unexpected:
            print(f"[Router] 未使用参数（可能是检查点里包含了非路由器键）：{unexpected}")

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

    # <<< 新增：DeepSpeed 必须把当前进程绑定到自己的 GPU >>>
    if use_deepspeed:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    
    # ======================
    # 数据加载（Zarr 或文件）
    # ======================
    if args.zarr_path is not None:
        json_path = getattr(args, 'status_json', None)
        assert json_path is not None, "使用zarr数据集格式，需要指定归一化统计量json"

        zarr_path = getattr(args, 'zarr_path', None)
        assert zarr_path is not None, "使用zarr数据格式时请在 args.zarr_path 指定路径"

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
            raise ValueError("不支持的 family")

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
            print(f"数据集总大小: {dataset_size}")
            print(f"训练集大小: {train_size}")
            print(f"验证集大小: {val_size}")

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
                print(f"警告：训练集和验证集有{len(overlap)}个重叠样本！")
            else:
                print("验证成功：训练集和验证集没有重叠样本")
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

    # 形状检查
    sample_batch = next(iter(train_loader))
    if is_logger:
        input_shape = sample_batch['input'].shape
        output_shape = sample_batch['output'].shape
        print(f"输入张量形状: {input_shape}")
        print(f"输出张量形状: {output_shape}")
        if len(input_shape) < 3 or len(input_shape) > 4:
            print(f"警告：输入形状不符合预期（3D/4D），实际为{len(input_shape)}D")
        if len(output_shape) < 3:
            print(f"警告：输出形状不符合预期，应为3D或更高维张量，实际为{len(output_shape)}D")

    in_channels = sample_batch['input'].shape[1]
    config.in_channels = in_channels

    if is_logger:
        print(f"更新后的输入通道数: {config.in_channels}")
        print(f"输出通道数: {config.out_channels}")
        print(f"隐藏通道数: {config.hidden_channels}")
        print(f"use_moe = False时专家数量: {len(config.expert_configs)}")

    # ======================
    # 构建 Encoder
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
                print(f"[Encoder] 已从 {args.encoder_path} 加载预训练权重。")
                if missing:
                    print(f"[Encoder] 缺失参数: {missing}")
                if unexpected:
                    print(f"[Encoder] 未使用参数: {unexpected}")
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

        # ====== moe_in_channels 的确定 ======
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
            print(f"Encoder 输出通道数(供 MoE 使用): {moe_in_channels}")
    else:
        moe_in_channels = config.in_channels
        if is_logger:
            print("[Encoder] use_encoder=False，直接将原始输入送入 MoE。")

    config.moe_in_channels = moe_in_channels

    # ======================
    # 构建 Experts + MOE
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
            alpha=0.1,
        )

    # ======================
    # EMO + DDP / DeepSpeed
    # ======================
    if use_deepspeed:
        if deepspeed is None:
            raise RuntimeError("未安装 deepspeed，但传入了 --use_deepspeed")
        emo_model = EMO(encoder_model, moe_model, pass_encoder_logits_as_weights=True)
        model = emo_model  # 先占位，初始化后会替换为 DeepSpeedEngine
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
            print("[Encoder] 已冻结 encoder 参数。")

    # ======================
    # 优化器 & 调度器
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
            raise ValueError("需提供 --deepspeed_config")
        ds_cfg_path = Path(args.ds_config)
        if not ds_cfg_path.exists():
            raise ValueError(f"DeepSpeed 配置文件不存在: {ds_cfg_path}")

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
            print("[DeepSpeed] 已启用 DeepSpeed 训练。")
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
    # 损失函数
    # ======================
    lambda_grad_l1 = float(getattr(config, "lambda_grad_l1", 0.0))
    lambda_fourier_mag_l1 = float(getattr(config, "lambda_fourier_mag_l1", 0.0))
    lambda_ce = float(getattr(config, "lambda_ce", 0.0))

    base_loss: Callable = L1L2Loss(config.lambda_g1v, config.lambda_g2v).to(device)
    grad_loss_module = SobelLoss().to(device) if lambda_grad_l1 > 0 else None
    fourier_loss_module = FourierMag_L1().to(device) if lambda_fourier_mag_l1 > 0 else None

    # 训练分类 + 回归（encoder logits 参与 CE）
    if config.train_encoder:
        def criterion(pred: torch.Tensor, gt: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
            # 在损失阶段禁用 AMP，并统一到 fp32 计算，避免混精度导致的 dtype 冲突/数值不稳
            with torch.amp.autocast(device_type=pred.device.type, enabled=False):
                pred32 = pred.float()
                gt32 = gt.float()

                # base: 必须返回 Tensor 分量 {"loss", "l1", "l2"}
                loss_dict = base_loss(pred32, gt32)
                total_loss_t = loss_dict["loss"]  # Tensor (标量)

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

                # Cross Entropy（同样在 fp32 下计算）
                ce_val_t = pred32.new_zeros(())
                if logits is not None and labels is not None:
                    ce_val_t = F.cross_entropy(logits.float(), labels.long(), reduction="mean")
                    total_loss_t = total_loss_t + lambda_ce * ce_val_t

            # 返回两套（*_t: 反传用；无后缀: 日志用 float）
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
            # 在损失阶段禁用 AMP，并统一到 fp32 计算
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
    # 运行目录 & 日志
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
    # 最佳模型保存路径
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
    # Resume（与保存逻辑严格对偶；不对 moe 分开加载）
    # ======================
    start_epoch = 0
    if hasattr(args, "resume_path") and args.resume_path is not None and os.path.exists(args.resume_path):
        ckpt_path = args.resume_path
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # 兼容 DDP(EMO) / EMO / DeepSpeedEngine
        target_model = model.module if hasattr(model, "module") else model
        target_moe = getattr(target_model, "moe", None)

        # ---------- 1) DeepSpeed 引擎状态（若保存时用 engine.save_checkpoint） ----------
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
                    print(f"[DeepSpeed] 已加载引擎状态: dir={resume_dir}, tag={ds_tag}, load_path={load_path}")
            else:
                if is_logger:
                    print("[DeepSpeed][WARN] 未能推断 tag；若保存用过 engine.save_checkpoint，请提供 --resume_tag=best 或 last-epN")

        # ---------- 2) 解析保存文件中各分量（仅按你的保存逻辑对偶恢复） ----------
        model_check   = checkpoint.get("model_state_dict", None)       # 只有某些场景会存在
        router_check  = checkpoint.get("router_state_dict", None)      # 非 DS + "all" 场景保存
        encoder_check = checkpoint.get("encoder_state_dict", None)     # 两边都可能存在
        expert_check  = checkpoint.get("expert_state_dict", None)      # 仅“单专家模式”可选存在（有时单独文件）

        # 情况 A：非 DeepSpeed 且 experts_name_str == "all"
        # 保存端：删除了 model_state_dict，仅保存 router_state_dict (+ encoder)
        if (not use_deepspeed) and (experts_name_str == "all"):
            if router_check is not None and (target_moe is not None) and hasattr(target_moe, "router") and (target_moe.router is not None):
                missing, unexpected = target_moe.router.load_state_dict(router_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Resume][all][router] 缺失参数: {missing}, 多余参数: {unexpected}")
            else:
                if is_logger:
                    print("[Resume][all][router] ckpt 未含 router_state_dict（或模型无 moe.router），跳过 Router 恢复")

            # Encoder：若未指定外部 encoder_path，则从 ckpt 恢复
            if hasattr(target_model, "encoder") and (encoder_check is not None) and not getattr(args, "encoder_path", None):
                missing, unexpected = load_encoder_weights(target_model.encoder, encoder_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Resume][all][encoder] 缺失参数: {missing}, 多余参数: {unexpected}")
            elif encoder_check is not None and getattr(args, "encoder_path", None) and is_logger:
                print(f"[Resume][all][encoder] 检测到 --encoder_path，跳过从 ckpt 恢复 encoder")

        else:
            # 情况 B：其余所有场景（包括 DeepSpeed 或 非 DS 的普通/单专家/非 all）
            # 保存端一定/常常包含整模 model_state_dict（DeepSpeed: 附加文件里；非 DS：默认就有）
            if model_check is not None:
                missing, unexpected = target_model.load_state_dict(model_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Resume][generic][model] 缺失参数: {missing}, 多余参数: {unexpected}")
            else:
                if is_logger:
                    print("[Resume][generic] ckpt 未包含 model_state_dict，跳过整模恢复（与保存逻辑一致）")

            # 单专家模式下：如果你另存了专家专属文件，可通过 --resume_expert_path 再细粒度覆盖
            if (experts_name is not None and len(experts_name) == 1 and experts_name[0] != "all"):
                # 1) 从 ckpt 主文件里恢复到整模（上面已做）
                # 2) 可选：再覆盖 experts[0]
                resume_expert_path = getattr(args, "resume_expert_path", None)
                if resume_expert_path and os.path.exists(resume_expert_path):
                    try:
                        expert_blob = torch.load(resume_expert_path, map_location="cpu", weights_only=False)
                        expert_sd = expert_blob.get("expert_state_dict", None)
                        if expert_sd is not None and (target_moe is not None) and hasattr(target_moe, "experts") and len(target_moe.experts) > 0:
                            missing, unexpected = target_moe.experts[0].load_state_dict(expert_sd, strict=False)
                            if is_logger and (missing or unexpected):
                                print(f"[Resume][single-expert][experts[0]] 缺失参数: {missing}, 多余参数: {unexpected}")
                            if is_logger:
                                print(f"[Resume][single-expert] 已从 {resume_expert_path} 覆盖 experts[0]")
                        elif is_logger:
                            print(f"[Resume][single-expert] {resume_expert_path} 未含 expert_state_dict 或模型无 experts[0]")
                    except Exception as e:
                        if is_logger:
                            print(f"[Resume][single-expert] 载入 resume_expert_path 失败：{e}")
                elif expert_check is not None and (target_moe is not None) and hasattr(target_moe, "experts") and len(target_moe.experts) > 0:
                    # 若主 ckpt 就带了 expert_state_dict（DeepSpeed 附件场景）
                    missing, unexpected = target_moe.experts[0].load_state_dict(expert_check, strict=False)
                    if is_logger and (missing or unexpected):
                        print(f"[Resume][single-expert][experts[0]] 缺失参数: {missing}, 多余参数: {unexpected}")

            # Encoder：DeepSpeed 与普通模式都可存在
            if hasattr(target_model, "encoder") and (encoder_check is not None) and not getattr(args, "encoder_path", None):
                missing, unexpected = load_encoder_weights(target_model.encoder, encoder_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Resume][generic][encoder] 缺失参数: {missing}, 多余参数: {unexpected}")
            elif encoder_check is not None and getattr(args, "encoder_path", None) and is_logger:
                print(f"[Resume][generic][encoder] 检测到 --encoder_path，跳过从 ckpt 恢复 encoder")

            # Router：即便不是 "all"，若 ckpt 附带了 router_state_dict 也可尝试恢复（保存时允许存在）
            if router_check is not None and (target_moe is not None) and hasattr(target_moe, "router") and (target_moe.router is not None):
                try:
                    missing, unexpected = target_moe.router.load_state_dict(router_check, strict=False)
                    if is_logger and (missing or unexpected):
                        print(f"[Resume][generic][router] 缺失参数: {missing}, 多余参数: {unexpected}")
                except Exception as e:
                    if is_logger:
                        print(f"[Resume][generic][router] 跳过 router_state_dict 加载：{e}")

        # ---------- 3) Optimizer（仅非 DeepSpeed 会保存在 ckpt 里） ----------
        if (not use_deepspeed) and (optimizer is not None):
            try:
                opt_state = checkpoint.get("optimizer_state_dict", None)
                if opt_state:
                    optimizer.load_state_dict(opt_state)
                else:
                    if is_logger:
                        print("[Resume][optimizer] ckpt 未包含 optimizer_state_dict，跳过")
            except Exception as e:
                if is_logger:
                    print(f"[Resume][optimizer] 跳过 optimizer_state_dict 加载：{e}")

        # ---------- 4) 训练元信息 ----------
        data_dict   = checkpoint.get("data_dict", None)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if start_epoch < 0:
            start_epoch = 0

        if is_logger:
            print(f"==> 成功从 {ckpt_path} 恢复，start_epoch = {start_epoch}")
    else:
        if is_logger:
            print("未提供 resume 路径，或路径无效，将从头开始训练。")

    amp_enabled = bool(use_amp) and (device.type == 'cuda')
    if use_deepspeed:
        model.zero_grad()
    else:
        optimizer.zero_grad(set_to_none=True)

    # ========= 关键：初始化“最佳指标缓存” =========
    best_epoch_metrics = None
    best_epoch_index = 0

    # ======================
    # 核心训练循环
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
                # ==== 关键：把 DeepSpeed 引擎传入 ====
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
            print("已写入hparams")
    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    return model, best_val_loss


def run_inference(n_args):
    """
    推理流程（与训练流程严格对偶）：
      1) 载入训练期 args/config，并用 CLI 覆盖
      2) 构建数据管线（Zarr + 归一化）
      3) 构建 Encoder + MOE -> EMO（参数与训练完全一致）
      4) 权重加载：
         - 'all'：仅加载 router（以及必须提供 encoder_path）；专家从目录 load
         - 其他：从 checkpoint 加载 MoE（model_state_dict 或 router_state_dict），Encoder 优先 encoder_path
      5) AMP(bf16) 推理、指标统计、可视化/类型预测保存/频域分析
    """
    import tqdm
    from argparse import Namespace
    from torchvision.transforms import Compose

    # ===== 读取训练期保存的 args/config，并被 CLI 覆盖 =====
    setting_dir = Path(getattr(n_args, "setting_path", ""))
    if not setting_dir:
        raise ValueError("推理需要 --setting_path（包含训练时保存的 args.json 与 config.json）")
    if not setting_dir.exists():
        raise ValueError(f"--setting_path 不存在: {setting_dir}")

    args_path   = setting_dir / "args.json"
    config_path = setting_dir / "config.json"
    if not args_path.exists():
        raise ValueError(f"缺少训练时保存的参数文件: {args_path}")
    if not config_path.exists():
        raise ValueError(f"缺少训练时保存的配置文件: {config_path}")

    with open(args_path, "r", encoding="utf-8") as f:
        stored_args_dict = json.load(f)
    with open(config_path, "r", encoding="utf-8") as f:
        stored_config_dict = json.load(f)

    # CLI 高优先级覆盖
    runtime_args_dict = dict(stored_args_dict)
    for key, value in vars(n_args).items():
        if value is not None:
            runtime_args_dict[key] = value
    runtime_args_dict["mode"] = "inference"
    runtime_args = Namespace(**runtime_args_dict)

    # ===== 初始化 config / 运行环境（与训练一致） =====
    config, runtime_ctx = get_seismic_config(runtime_args)
    device       = runtime_ctx["device"]
    is_logger    = runtime_ctx["is_logger"]
    world_size   = runtime_ctx["world_size"]
    local_rank   = runtime_ctx["local_rank"]
    experts_name = runtime_ctx["experts_name"]
    experts_name_str = runtime_ctx["experts_name_str"]

    # 回填训练时的 config 字段（保持一致）
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

    # ===== 归一化统计（优先 --status_json；否则 checkpoint 内的 data_dict）=====
    checkpoint = None
    if experts_name_str != "all":
        model_path = Path(getattr(runtime_args, "model_path", ""))
        if not model_path.exists():
            raise ValueError(f"[MoE] 常规推理需要提供模型文件 --model_path，未找到: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

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
        hint = "（ALL 模式下通常无整体 checkpoint，请通过 --status_json 提供归一化统计量）" if experts_name_str == "all" else ""
        raise ValueError(f"无法获取归一化统计量，请提供 --status_json 或确保 checkpoint 中包含 data_dict。{hint}")

    # ===== 数据变换 & Zarr DataLoader（与训练相同的对数 + MinMax）=====
    k_value = float(getattr(runtime_args, "k", 1.0))
    input_transform = Compose([
        T.LogTransform(k=k_value),
        T.MinMaxNormalize(T.log_transform(stats_dict["input_min"], k=k_value),
                          T.log_transform(stats_dict["input_max"], k=k_value)),
    ])
    output_transform = Compose([
        T.MinMaxNormalize(stats_dict["output_min"], stats_dict["output_max"])
    ])
    input_inverse_transform = Compose([
        T.InverseMinMaxNormalize(T.log_transform(stats_dict["input_min"], k=k_value),
                                 T.log_transform(stats_dict["input_max"], k=k_value)),
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
        raise ValueError("推理需要指定 zarr 数据集路径 (--zarr_path)")
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise ValueError(f"zarr 数据文件不存在: {zarr_path}")

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

    # 分布式/单机（与训练一致）
    if getattr(runtime_args, "distributed", False) and world_size > 1:
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
            batch_size=int(config.test_batch_size),
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
            batch_size=int(config.test_batch_size),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    if len(test_loader) == 0:
        raise RuntimeError("测试数据加载器为空，请检查数据集配置。")

    # ===== 构建 EMO（Encoder + MOE），与训练完全一致的参数 =====
    # 探测 in_channels
    sample_batch = next(iter(test_loader))
    config.in_channels = int(sample_batch["input"].shape[1])

    # --- Encoder ---
    encoder_model = None
    if getattr(config, "use_encoder", False):
        num_types = int(getattr(config, "v_type_num", 10) or 10)
        type_act  = "identity" if getattr(config, "train_encoder", False) else "softmax"
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

        # Encoder 权重优先级：--encoder_path > checkpoint.encoder_state_dict > 随机
        if experts_name_str == "all":
            enc_ckpt = getattr(runtime_args, "encoder_path", None)
            if not enc_ckpt:
                raise ValueError("[Encoder] experts_name_str='all' 模式下必须提供 --encoder_path")
            missing, unexpected = load_encoder_weights(encoder_model, enc_ckpt, map_location=device, strict=False)
            if is_logger:
                print(f"[Encoder] (ALL) 已从 {enc_ckpt} 加载推理权重。")
                if missing:    print(f"[Encoder] 缺失参数: {missing}")
                if unexpected: print(f"[Encoder] 未使用参数: {unexpected}")
        else:
            enc_ckpt = getattr(runtime_args, "encoder_path", None)
            if enc_ckpt:
                missing, unexpected = load_encoder_weights(encoder_model, enc_ckpt, map_location=device, strict=False)
                if is_logger:
                    print(f"[Encoder] 已从 {enc_ckpt} 加载推理权重。")
                    if missing:    print(f"[Encoder] 缺失参数: {missing}")
                    if unexpected: print(f"[Encoder] 未使用参数: {unexpected}")
            else:
                encoder_state = checkpoint.get("encoder_state_dict") if isinstance(checkpoint, dict) else None
                if encoder_state is not None:
                    missing, unexpected = load_encoder_weights(encoder_model, encoder_state, map_location=device, strict=False)
                    if is_logger and (missing or unexpected):
                        print(f"[Encoder] 从 checkpoint 加载编码器: 缺失 {missing}, 多余 {unexpected}")
                elif is_logger:
                    print("[Encoder] 未提供 encoder_path，且 checkpoint 中缺少 encoder_state_dict，将使用随机初始化的编码器。")

        # 决定 moe_in_channels（用一次 probe）
        with torch.no_grad():
            probe_inputs = sample_batch["input"].to(device, non_blocking=True)
            encoder_probe, _, _ = encoder_model(probe_inputs)
        config.moe_in_channels = int(encoder_probe.shape[1])
        del encoder_probe
    else:
        config.moe_in_channels = config.in_channels
        if is_logger:
            print("[Encoder] use_encoder=False，推理阶段直接使用原始输入。")
    
    # --- Experts ---
    if experts_name_str == "all":
        experts_dir = getattr(config, "use_experts_path", None) or getattr(runtime_args, "experts_path", None)
        if not experts_dir:
            raise ValueError("[Experts] experts_name_str='all' 模式下必须提供专家权重目录（config.use_experts_path 或 --experts_path）")
        if not Path(experts_dir).exists():
            raise ValueError(f"[Experts] 专家权重目录不存在: {experts_dir}")

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
            print(f"[Experts] (ALL) 已从目录加载专家: {experts_dir}")
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
        use_expert_memory_proxy=config.use_gpu_proxy,   # 与训练一致
        use_encoder=getattr(config, "use_encoder", False),
        device=device,                                  # 与训练保持同构
    )

    # --- EMO 封装 ---
    emo = EMO(encoder_model, moe, pass_encoder_logits_as_weights=True).to(device)
    emo.eval()

    # ===== 权重加载（严格对偶保存逻辑）=====
    if experts_name_str == "all":
        # ALL 模式：专家已经从目录加载，这里只需要加载 router 权重
        router_path = getattr(runtime_args, "router_path", None)
        if not router_path:
            raise ValueError("[Router] experts_name_str='all' 模式下必须提供 --router_path")
        _load_router_weights(emo.moe, router_path, map_location=device, is_logger=is_logger)
    else:
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint 不是字典，无法解析 state_dict。")

        # 解析 checkpoint 中的各个部分
        model_check   = checkpoint.get("model_state_dict", None)
        router_check  = checkpoint.get("router_state_dict", None)
        expert_check  = checkpoint.get("expert_state_dict", None)

        # 1) 优先尝试整模恢复（model_state_dict）
        if model_check is not None:
            missing, unexpected = emo.moe.load_state_dict(model_check, strict=False)
            if is_logger and (missing or unexpected):
                print(f"[MoE][generic][model] 缺失参数: {missing}, 多余参数: {unexpected}")
            elif is_logger:
                print(f"[MoE][generic][model] 已从 checkpoint 加载 model_state_dict")
        elif router_check is not None:
            # 2) 如果没有整模，只能加载 router_state_dict
            if hasattr(emo.moe, "router") and emo.moe.router is not None:
                missing, unexpected = emo.moe.router.load_state_dict(router_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[MoE][generic][router-only] 缺失参数: {missing}, 多余参数: {unexpected}")
                elif is_logger:
                    print(f"[MoE][generic][router-only] 已从 checkpoint 加载 router_state_dict")
            else:
                raise ValueError("[MoE][generic] checkpoint 仅包含 router_state_dict，但模型中不存在 moe.router")
        else:
            raise ValueError("checkpoint 中既无 model_state_dict 也无 router_state_dict，无法恢复 MoE。")

        # 3) 单专家模式：若 ckpt 带 expert_state_dict，可进一步覆盖 experts[0]
        if experts_name is not None and isinstance(experts_name, (list, tuple)) \
                and len(experts_name) == 1 and experts_name[0] != "all":
            if expert_check is not None and hasattr(emo.moe, "experts") and len(emo.moe.experts) > 0:
                missing, unexpected = emo.moe.experts[0].load_state_dict(expert_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[MoE][single-expert][experts[0]] 缺失参数: {missing}, 多余参数: {unexpected}")
                elif is_logger:
                    print(f"[MoE][single-expert] 已从 checkpoint 覆盖 experts[0]")
            elif expert_check is not None and is_logger:
                print("[MoE][single-expert] checkpoint 带 expert_state_dict 但模型无 experts[0]，跳过")

        # 4) 如果 checkpoint 同时有 model_state_dict 和 router_state_dict，可以选择性再覆盖一次 router
        if router_check is not None and hasattr(emo.moe, "router") and emo.moe.router is not None:
            try:
                missing, unexpected = emo.moe.router.load_state_dict(router_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[MoE][generic][router] 额外覆盖 router：缺失参数: {missing}, 多余参数: {unexpected}")
                elif is_logger:
                    print(f"[MoE][generic][router] 已从 checkpoint 覆盖 router_state_dict")
            except Exception as e:
                if is_logger:
                    print(f"[MoE][generic][router] 覆盖 router_state_dict 失败：{e}")

    # ===== 推理循环（AMP bf16，与训练相同的 dtype 策略）=====
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

    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc=f"推理中({eval_split})", disable=not is_logger):
            inputs  = batch["input"].to(device, non_blocking=True)
            targets = batch.get("output")
            if targets is None:
                continue
            targets = targets.to(device, non_blocking=True).to(dtype=torch.float32)

            if amp_enabled:
                with torch.amp.autocast(device_type=device.type, enabled=True, dtype=torch.bfloat16):
                    preds, aux, enc_weights = emo(inputs)
            else:
                preds, aux, enc_weights = emo(inputs)

            preds = preds.to(dtype=torch.float32)

            # 指标累计（与训练的 metrics 模块一致）
            mse_sum  += metrics_module.calculate_mse(preds, targets)
            mae_sum  += metrics_module.calculate_mae(preds, targets)
            psnr_sum += metrics_module.calculate_psnr(preds, targets)
            rmse_sum += metrics_module.calculate_rmse(preds, targets)
            ssim_sum += metrics_module.calculate_ssim(preds, targets)
            batch_count += 1

            # 仅采样一次可视化样本
            if visual_payload is None and is_logger:
                inputs_cpu  = inputs.detach().cpu()
                targets_cpu = targets.detach().cpu()
                preds_cpu   = preds.detach().cpu()
                if input_inverse_transform is not None:
                    inputs_cpu = input_inverse_transform(inputs_cpu)
                if output_inverse_transform is not None:
                    preds_cpu   = output_inverse_transform(preds_cpu)
                    targets_cpu = output_inverse_transform(targets_cpu)
                logits_cpu = enc_weights.detach().cpu() if enc_weights is not None else None

                # 若启用 encoder，额外导出中间编码特征
                if encoder_model is not None:
                    if amp_enabled:
                        with torch.amp.autocast(device_type=device.type, enabled=True, dtype=torch.bfloat16):
                            encoded, _, _ = encoder_model(inputs)
                    else:
                        encoded, _, _ = encoder_model(inputs)
                    encoded_cpu = encoded.detach().cpu()
                else:
                    encoded_cpu = None

                visual_payload = {
                    "inputs": inputs_cpu,
                    "targets": targets_cpu,
                    "preds": preds_cpu,
                    "encoded": encoded_cpu,
                    "logits": logits_cpu,
                    "batch": batch,
                }

    if batch_count == 0:
        raise RuntimeError("测试数据集中没有可评估的样本。")

    mse  = mse_sum  / batch_count
    mae  = mae_sum  / batch_count
    psnr = psnr_sum / batch_count
    rmse = rmse_sum / batch_count
    ssim = ssim_sum / batch_count

    if is_logger:
        print(f"推理指标: MSE={mse:.6f} | MAE={mae:.6f} | PSNR={psnr:.4f} | RMSE={rmse:.6f} | SSIM={ssim:.6f}")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"  Inference |        -        |        -        | {mae:.6f} | {mse:.6f} | {psnr:.4f} | {rmse:.6f} | {ssim:.6f} |\n")

    # ===== 可视化输出（与训练日志一致）=====
    if visual_payload is not None:
        inputs_vis  = visual_payload["inputs"]
        targets_vis = visual_payload["targets"]
        preds_vis   = visual_payload["preds"]
        logits_vis  = visual_payload["logits"]
        encoded_vis = visual_payload["encoded"]
        vis_batch   = visual_payload["batch"]
        num_samples = min(4, inputs_vis.shape[0])

        visualize_results(
            inputs_vis,
            targets_vis,
            preds_vis,
            save_dir=img_path,
            max_samples=num_samples,
        )

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

        analyze_fourier_domain(
            inputs_vis,
            targets_vis,
            preds_vis,
            save_dir=img_path,
            max_samples=num_samples,
        )

        visualize_encoded(
            encoded_vis,
            save_dir=img_path,
            max_samples=num_samples,
            selection='l2',
        )

    if test_sampler is not None and hasattr(test_sampler, "set_epoch"):
        test_sampler.set_epoch(0)

    print(f"推理完成！结果保存在: {results_dir}")

if __name__ == '__main__':
    args = build_argparser_and_parse()
    if args.mode in ('train', 'train_encoder'):
        run_training(args)
    elif args.mode == 'inference':
        run_inference(args)
    else:
        raise ValueError(f"不支持的运行模式: {args.mode}")
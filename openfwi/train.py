import os
import sys
import re
import math
import json
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from datetime import datetime
import tqdm as _tqdm

# ============== Project deps (kept consistent with moe facade) ==============
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralop.data.datasets import ZarrSeismicDataset
from neuralop.data.datasets.seismic_dataset import SeismicDataProcessor
from neuralop.data.dataloader.zarr_seismic_dataloader import build_loaders

from scripts.scheduler import WarmupMultiStepLR
from config.seismic_moe_config import SeismicMOEConfig
from utils import SeismicMetrics, EarlyStopping, plot_loss_curve, analyze_fourier_domain, visualize_results
import scripts.transforms as T
import network  # provides InversionNet, FCN4_Deep_Resize_2 (UPFWI), Discriminator
import neuralop.mpu.comm as comm
from neuralop.training import setup

try:
    import deepspeed
except Exception:
    deepspeed = None

# Optional patch for spectral conv (no-op if not present)
try:
    from neuralop.layers.spectral_convolution import SpectralConv

    patch_spectral_conv_forward(SpectralConv)  # type: ignore
except Exception:
    pass


# ------------------------------
# Config helper
# ------------------------------
def get_seismic_config(args):
    """
    构建用于 plain/GAN 训练管线的配置（SeismicMOEConfig）与运行时上下文（runtime_ctx）。
    """
    cfg = SeismicMOEConfig()
    # 设置随机种子
    cfg.distributed.seed = args.seed

    # 启用分布式训练
    if args.distributed:
        cfg.distributed.use_distributed = True
        device, is_logger = setup(cfg)
    else:
        device, is_logger = setup(cfg)

    local_rank = comm.get_local_rank()
    global_rank = comm.get_global_rank()
    world_size = comm.get_world_size()

    # —— 数据/任务基本设置 —— #
    cfg.family = str(getattr(args, "family", "all"))
    cfg.channel_dim = int(getattr(cfg, "channel_dim", 1))
    cfg.out_channels = int(getattr(cfg, "out_channels", 1))

    # —— 训练超参 —— #
    cfg.epochs = int(getattr(args, "epochs", 200))
    cfg.learning_rate = float(getattr(args, "learning_rate", 1e-4))
    cfg.weight_decay = float(getattr(args, "weight_decay", 1e-2))
    cfg.batch_size = int(getattr(args, "batch_size", 64))
    cfg.test_batch_size = int(getattr(args, "test_batch_size", 64))

    # —— 日志/输出 —— #
    cfg.output_dir = str(getattr(args, "output_dir", "./outputs"))
    cfg.log_root = str(getattr(args, "log_root", "./runs"))
    cfg.save_every = int(getattr(args, "save_every", 50))
    cfg.log_every = int(getattr(args, "log_every", 50))

    # —— AMP/梯度裁剪 —— #
    cfg.use_amp = bool(getattr(args, "use_amp", False))
    cfg.max_norm = float(getattr(args, "max_norm", 0.0))

    # —— 早停 —— #
    cfg.early_stop = bool(getattr(args, "early_stop", False))
    cfg.early_stop_patience = int(getattr(args, "early_stop_patience", 20))
    cfg.early_stop_min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
    cfg.early_stop_warmup_epochs = int(getattr(args, "early_stop_warmup_epochs", 10))

    # —— 学习率调度 —— #
    cfg.lr_milestones = list(getattr(args, "lr_milestones", []))
    cfg.lr_gamma = float(getattr(args, "lr_gamma", 0.1))
    cfg.lr_warmup_epochs = float(getattr(args, "lr_warmup_epochs", 0.0))
    cfg.concat_channels = False

    runtime_ctx = {
        "device": device,
        "is_logger": is_logger,
        "world_size": world_size,
        "local_rank": local_rank,
    }

    return cfg, runtime_ctx


def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


class TransformedSubset(Subset):
    def __init__(self, dataset, transform=None):
        if hasattr(dataset, "indices"):
            super().__init__(dataset.dataset, dataset.indices)
        else:
            super().__init__(dataset, list(range(len(dataset))))
        self.transform = transform

    def _get_single(self, index: int):
        sample = self.dataset[self.indices[index]]
        sample = {**sample, "idx": index}
        return self.transform(sample) if self.transform else sample

    def __getitem__(self, index):
        if isinstance(index, list):
            return [self._get_single(i) for i in index]
        return self._get_single(index)

    def __getitems__(self, idx: list[int]):
        return [self._get_single(index) for index in idx]


class PlainAdapter(nn.Module):
    """Wrap arbitrary generator (InversionNet or UPFWI) to match (pred, aux, logits) interface."""

    def __init__(self, generator: nn.Module):
        super().__init__()
        self.generator = generator

    def forward(self, x, use_amp: bool = False):
        y = self.generator(x)
        return y, None, None


# ------------------------------
# Data builders (same as moe facade)
# ------------------------------
def build_zarr_loaders_like_moe(args, config, world_size, local_rank, is_logger):
    assert args.zarr_path is not None, "需要 --zarr_path"
    assert args.status_json is not None, "需要 --status_json（归一化统计量）"

    with open(args.status_json, "r", encoding="utf-8") as f:
        stats = json.load(f)
    data_dict = stats["overall"] if config.family == "all" else stats["per_type"][config.family]

    input_transform = Compose(
        [
            T.LogTransform(k=args.k),
            T.MinMaxNormalize(
                T.log_transform(data_dict["input_min"], k=args.k),
                T.log_transform(data_dict["input_max"], k=args.k),
            ),
        ]
    )
    output_transform = Compose(
        [
            T.MinMaxNormalize(data_dict["output_min"], data_dict["output_max"]),
        ]
    )
    input_inverse_transform = Compose(
        [
            T.InverseMinMaxNormalize(
                T.log_transform(data_dict["input_min"], k=args.k),
                T.log_transform(data_dict["input_max"], k=args.k),
            ),
            T.InverseLogTransform(k=args.k),
        ]
    )
    output_inverse_transform = Compose(
        [
            T.InverseMinMaxNormalize(data_dict["output_min"], data_dict["output_max"]),
        ]
    )

    processor = SeismicDataProcessor(
        input_transform=input_transform,
        output_transform=output_transform,
        channel_dim=config.channel_dim,
        config=config,
    )

    train_ds = ZarrSeismicDataset(
        zarr_path=args.zarr_path,
        split="train",
        input_transform=None,
        output_transform=None,
        expect_input_shape=(5, 1000, 70),
        to_float32=True,
    )
    val_ds = ZarrSeismicDataset(
        zarr_path=args.zarr_path,
        split="val",
        input_transform=None,
        output_transform=None,
        expect_input_shape=(5, 1000, 70),
        to_float32=True,
    )

    train_ds_t = TransformedSubset(train_ds, processor)
    val_ds_t = TransformedSubset(val_ds, processor)

    train_loader, val_loader, train_sampler, val_sampler = build_loaders(
        args=args,
        config=config,
        train_dataset_with_transform=train_ds_t,
        val_dataset_with_transform=val_ds_t,
        chunks=32,
        world_size=world_size,
        local_rank=local_rank,
    )

    if is_logger:
        pf = getattr(train_loader, "prefetch_factor", None)
        if pf is not None:
            print(f"prefetch_factor={pf}")
    return (
        train_loader,
        val_loader,
        train_sampler,
        val_sampler,
        input_inverse_transform,
        output_inverse_transform,
        data_dict,
    )


# ------------------------------
# Losses
# ------------------------------
class GenCriterion(nn.Module):
    """L1 + L2 + optional adversarial ( -E[D(G(x))] )."""

    def __init__(self, lambda_g1v: float, lambda_g2v: float, lambda_adv: float = 0.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.lambda_g1v = float(lambda_g1v)
        self.lambda_g2v = float(lambda_g2v)
        self.lambda_adv = float(lambda_adv)

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor, d_model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        l1 = self.l1(pred, gt)
        l2 = self.l2(pred, gt)
        loss = self.lambda_g1v * l1 + self.lambda_g2v * l2
        adv = pred.new_zeros(())
        if (d_model is not None) and (self.lambda_adv > 0.0):
            adv = -torch.mean(d_model(pred))
            loss = loss + self.lambda_adv * adv
        return {
            "loss_t": loss,
            "loss": loss.detach(),
            "l1": l1.detach(),
            "l2": l2.detach(),
            "adv": adv.detach(),
        }


class WGANGPCriterion(nn.Module):
    """Wasserstein GP for discriminator.
    Given real=gt, fake=pred (no grad), returns loss for D (to MINIMIZE):
      loss_D = (E[D(fake)] - E[D(real)]) + lambda_gp * GP
    """

    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = float(lambda_gp)

    @staticmethod
    def _grad_penalty(d_model: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        batch = real.shape[0]
        device = real.device
        alpha = torch.rand(batch, 1, 1, 1, device=device, dtype=real.dtype)
        inter = alpha * real + (1 - alpha) * fake
        inter.requires_grad_(True)
        d_inter = d_model(inter)
        ones = torch.ones_like(d_inter)
        grads = torch.autograd.grad(
            outputs=d_inter,
            inputs=inter,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads = grads.reshape(batch, -1)
        gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
        return gp

    def forward(
        self, real: torch.Tensor, fake_detached: torch.Tensor, d_model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        d_real = d_model(real)
        d_fake = d_model(fake_detached)
        loss_diff = torch.mean(d_fake) - torch.mean(d_real)
        gp = self._grad_penalty(d_model, real, fake_detached)
        loss_d = loss_diff + self.lambda_gp * gp
        return {"loss_t": loss_d, "loss_diff": loss_diff.detach(), "loss_gp": gp.detach()}


# ------------------------------
# Validation
# ------------------------------
@torch.no_grad()
def evaluate_epoch(model, val_loader, device, criterion, metrics: "SeismicMetrics", use_amp: bool, is_logger: bool):
    model.eval()
    pbar = _tqdm.tqdm(val_loader, desc="Val", disable=not is_logger)
    vloss = mse = mae = psnr = rmse = ssim = 0.0
    n = 0
    for batch in pbar:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["output"].to(device, non_blocking=True).to(dtype=torch.float32)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            pred, _, _ = model(x, use_amp=use_amp)
        pack = criterion(pred.to(torch.float32), y)
        vloss += float(pack["loss"])
        mse += float(metrics.calculate_mse(pred, y))
        mae += float(metrics.calculate_mae(pred, y))
        psnr += float(metrics.calculate_psnr(pred, y))
        rmse += float(metrics.calculate_rmse(pred, y))
        ssim += float(metrics.calculate_ssim(pred, y))
        n += 1
    n = max(1, n)
    return {
        "val_loss": vloss / n,
        "mse": mse / n,
        "mae": mae / n,
        "psnr": psnr / n,
        "rmse": rmse / n,
        "ssim": ssim / n,
    }


# ------------------------------
# Core: run_training (supports plain or GAN)
# ------------------------------
def run_training(args):
    # Backend & config
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    config, runtime_ctx = get_seismic_config(args)
    device = runtime_ctx["device"]
    is_logger = runtime_ctx["is_logger"]
    world_size = runtime_ctx["world_size"]
    local_rank = runtime_ctx["local_rank"]

    if bool(getattr(args, "use_deepspeed", False)):
        assert deepspeed is not None, "未安装 DeepSpeed"
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    # Data
    (
        train_loader,
        val_loader,
        train_sampler,
        val_sampler,
        input_inv,
        output_inv,
        data_dict,
    ) = build_zarr_loaders_like_moe(args, config, world_size, local_rank, is_logger)

    # Detect channels
    sample_batch = next(iter(train_loader))
    config.in_channels = int(sample_batch["input"].shape[1])
    if is_logger:
        print(f"in_channels={config.in_channels}, out_channels={config.out_channels}")

    # ========= 构建 Generator & Discriminator =========
    gen_key = str(getattr(args, "generator", "InversionNet"))
    if gen_key == "UPFWI":
        gen_ctor = network.model_dict["UPFWI"]
        gen_kwargs = {
            "ratio": getattr(args, "sample_spatial", 1.0),
            "upsample_mode": getattr(args, "up_mode", None),
        }
    elif gen_key == "InversionNet":
        gen_ctor = network.model_dict["InversionNet"]
        gen_kwargs = {
            "sample_spatial": getattr(args, "sample_spatial", 1.0),
        }
    else:
        raise ValueError(f"未知生成器：{gen_key}；可选 InversionNet/UPFWI")

    generator = gen_ctor(**gen_kwargs).to(device)
    model_core = PlainAdapter(generator)

    # Optional Discriminator (GAN)
    use_gan = bool(getattr(args, "use_gan", False))
    if use_gan:
        disc_key = str(getattr(args, "discriminator", "Discriminator"))
        assert disc_key in network.model_dict, f"未知判别器：{disc_key}"
        discriminator = network.model_dict[disc_key]().to(device)
    else:
        discriminator = None

    # ========= 分布式封装：全部使用 PyTorch DDP =========
    ddp_use = bool(config.distributed.use_distributed) and (world_size > 1)

    if use_gan:
        # GAN + WGAN-GP：这里也使用 PyTorch 自带 DDP
        if ddp_use:
            if is_logger:
                print("[GAN] 使用 torch.nn.parallel.DistributedDataParallel 封装 G 和 D")
            model = torch.nn.parallel.DistributedDataParallel(
                model_core.to(device),
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=False,
            )
            discriminator = torch.nn.parallel.DistributedDataParallel(
                discriminator.to(device),
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=False,
            )
        else:
            model = model_core.to(device)
            # discriminator 已经 to(device)
    else:
        # 非 GAN：沿用 torch DDP
        if ddp_use:
            if is_logger:
                print("[Info] 非 GAN 训练：使用 torch.nn.parallel.DistributedDataParallel")
            model = torch.nn.parallel.DistributedDataParallel(
                model_core.to(device),
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=False,
            )
        else:
            model = model_core.to(device)

    # ========= Optimizers =========
    if use_gan:
        lr_g = float(getattr(args, "lr_g", args.learning_rate)) * max(1, world_size)
        lr_d = float(getattr(args, "lr_d", args.learning_rate)) * max(1, world_size)
        optimizer_g = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr_g,
            betas=(0.0, 0.9),
            weight_decay=args.weight_decay,
        )
        optimizer_d = torch.optim.AdamW(
            [p for p in (discriminator.parameters() if discriminator is not None else []) if p.requires_grad],
            lr=lr_d,
            betas=(0.0, 0.9),
            weight_decay=args.weight_decay,
        )
    else:
        lr = float(config.learning_rate) * (
            math.sqrt(world_size) if (ddp_use and world_size > 2) else 1.0
        )
        optimizer_g = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=getattr(config, "weight_decay", 1e-2),
        )
        optimizer_d = None

    # ========= Schedulers (per-iter) =========
    steps_per_epoch = max(1, len(train_loader))
    warmup_iters = args.lr_warmup_epochs * steps_per_epoch
    lr_milestones = [steps_per_epoch * m for m in args.lr_milestones]

    sched_g = WarmupMultiStepLR(
        optimizer_g,
        milestones=lr_milestones,
        gamma=args.lr_gamma,
        warmup_iters=warmup_iters,
        warmup_factor=1e-5,
    )
    if use_gan:
        sched_d = WarmupMultiStepLR(
            optimizer_d,
            milestones=lr_milestones,
            gamma=args.lr_gamma,
            warmup_iters=warmup_iters,
            warmup_factor=1e-5,
        )
    else:
        sched_d = None

    # ========= Criteria =========
    g_crit = GenCriterion(
        args.lambda_g1v,
        args.lambda_g2v,
        lambda_adv=(args.lambda_adv if use_gan else 0.0),
    )
    d_crit = WGANGPCriterion(lambda_gp=(args.lambda_gp if use_gan else 10.0)) if use_gan else None

    # ========= AMP 设置 =========
    if use_gan:
        amp_enabled = False
        if is_logger:
            print("[Info] use_gan=True ⇒ 关闭 AMP 以保证 WGAN-GP 稳定性")
    else:
        amp_enabled = bool(getattr(config, "use_amp", False)) and device.type == "cuda"

    # ========= Logging & dirs =========
    def _slugify(text: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in str(text))

    run_group = _slugify(config.family or "all")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = "_".join(
        [
            _slugify(gen_key),
            ("GAN" if use_gan else "PLAIN"),
            f"lr{getattr(args,'learning_rate',1e-4):g}",
            f"bs{config.batch_size}",
            timestamp,
        ]
    )
    output_root = Path(config.output_dir) / f"seismic_plain_{run_group}"
    results_dir = output_root / run_name
    tb_dir = Path(getattr(config, "log_root", "./runs")) / run_group / run_name
    if is_logger:
        results_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                vars(config) if not hasattr(config, "to_dict") else config.to_dict(),
                f,
                indent=2,
                default=str,
            )
    tb_writer = SummaryWriter(log_dir=str(tb_dir)) if is_logger else None

    # ========= Resume =========
    start_epoch = 0
    best_val = float("inf")
    if getattr(args, "resume_path", None) and os.path.exists(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location="cpu", weights_only=False)
        target = model.module if hasattr(model, "module") else model
        if "model_state_dict" in ckpt:
            missing, unexpected = target.load_state_dict(ckpt["model_state_dict"], strict=False)
            if is_logger and (missing or unexpected):
                print(f"[Resume] miss={missing} unexpected={unexpected}")
        if use_gan and ("discriminator_state_dict" in ckpt) and (discriminator is not None):
            disc_tgt = discriminator.module if hasattr(discriminator, "module") else discriminator
            disc_tgt.load_state_dict(ckpt["discriminator_state_dict"], strict=False)
        if ("optimizer_g" in ckpt) and (optimizer_g is not None):
            try:
                optimizer_g.load_state_dict(ckpt["optimizer_g"])
            except Exception as e:
                if is_logger:
                    print(f"[Resume][optG] skip: {e}")
        if use_gan and ("optimizer_d" in ckpt) and (optimizer_d is not None):
            try:
                optimizer_d.load_state_dict(ckpt["optimizer_d"])
            except Exception as e:
                if is_logger:
                    print(f"[Resume][optD] skip: {e}")
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_loss", best_val))
        if is_logger:
            print(f"[Resume] start_epoch={start_epoch}, best={best_val:.6f}")

    # Metrics & early stop
    metrics = SeismicMetrics()
    early_stopper = (
        EarlyStopping(
            patience=getattr(config, "early_stop_patience", 20),
            min_delta=getattr(config, "early_stop_min_delta", 0.0),
            warmup_epochs=getattr(config, "early_stop_warmup_epochs", 10),
            mode="min",
        )
        if getattr(config, "early_stop", False)
        else None
    )

    n_critic = int(getattr(args, "n_critic", 5)) if use_gan else 1

    # ===== Training loop =====
    try:
        for epoch in range(start_epoch, config.epochs):
            if ddp_use and hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "set_epoch"
            ):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            if use_gan and discriminator is not None:
                if isinstance(discriminator, nn.parallel.DistributedDataParallel):
                    discriminator.module.train()
                else:
                    discriminator.train()

            running = {
                "g_loss": 0.0,
                "l1": 0.0,
                "l2": 0.0,
                "adv": 0.0,
                "d_diff": 0.0,
                "d_gp": 0.0,
                "seen": 0,
            }
            pbar = _tqdm.tqdm(train_loader, desc=f"Train {epoch}", disable=not is_logger)
            for it, batch in enumerate(pbar):
                x = batch["input"].to(device, non_blocking=True)
                y = batch["output"].to(device, non_blocking=True).to(dtype=torch.float32)

                # ---- D update ----
                if use_gan and (discriminator is not None):
                    optimizer_d.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        with torch.amp.autocast(
                            device_type=device.type,
                            enabled=amp_enabled,
                            dtype=torch.bfloat16,
                        ):
                            pred_detach, _, _ = model(x, use_amp=amp_enabled)
                            pred_detach = pred_detach.to(torch.float32)
                    d_pack = d_crit(y, pred_detach.detach(), discriminator)
                    d_pack["loss_t"].backward()
                    optimizer_d.step()
                    running["d_diff"] += float(d_pack["loss_diff"]) * x.size(0)
                    running["d_gp"] += float(d_pack["loss_gp"]) * x.size(0)

                # ---- G update (every n_critic) ----
                do_g = ((it + 1) % n_critic == 0) or ((it + 1) == len(train_loader))
                if do_g:
                    optimizer_g.zero_grad(set_to_none=True)
                    with torch.amp.autocast(
                        device_type=device.type,
                        enabled=amp_enabled,
                        dtype=torch.bfloat16,
                    ):
                        pred, _, _ = model(x, use_amp=amp_enabled)
                        pack = g_crit(
                            pred.to(torch.float32),
                            y,
                            (discriminator if use_gan else None),
                        )
                    pack["loss_t"].backward()
                    if hasattr(torch.nn.utils, "clip_grad_norm_") and getattr(
                        config, "max_norm", 0.0
                    ) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
                    optimizer_g.step()

                    running["g_loss"] += float(pack["loss"]) * x.size(0)
                    running["l1"] += float(pack["l1"]) * x.size(0)
                    running["l2"] += float(pack["l2"]) * x.size(0)
                    running["adv"] += float(pack["adv"]) * x.size(0)

                # Log
                running["seen"] += x.size(0)
                if is_logger and tb_writer and (
                    it % max(1, getattr(config, "log_every", 50)) == 0
                ):
                    step = epoch * len(train_loader) + it
                    if use_gan:
                        tb_writer.add_scalar("train/diff", float(d_pack["loss_diff"]), step)
                        tb_writer.add_scalar("train/gp", float(d_pack["loss_gp"]), step)
                    if do_g:
                        tb_writer.add_scalar("train/l1", float(pack["l1"]), step)
                        tb_writer.add_scalar("train/l2", float(pack["l2"]), step)
                        tb_writer.add_scalar("train/g_loss", float(pack["loss"]), step)

                # Step schedulers per iter
                if sched_g is not None:
                    sched_g.step()
                if use_gan and (sched_d is not None):
                    sched_d.step()

            # ------- validate -------
            val_stats = evaluate_epoch(model, val_loader, device, g_crit, metrics, amp_enabled, is_logger)
            val_loss = val_stats["val_loss"]
            if is_logger:
                g_loss_avg = running["g_loss"] / max(1, running["seen"])
                print(
                    f"[Epoch {epoch}] train_g_loss={g_loss_avg:.6f}  "
                    f"val_loss={val_loss:.6f}  PSNR={val_stats['psnr']:.4f}  "
                    f"SSIM={val_stats['ssim']:.4f}"
                )
                if tb_writer:
                    tb_writer.add_scalar("val/loss", val_loss, (epoch + 1) * len(train_loader))
                    tb_writer.add_scalar(
                        "val/psnr", val_stats["psnr"], (epoch + 1) * len(train_loader)
                    )
                    tb_writer.add_scalar(
                        "val/ssim", val_stats["ssim"], (epoch + 1) * len(train_loader)
                    )

            # Save best & periodic
            improved = val_loss < best_val
            if improved and is_logger:
                best_val = val_loss
                to_save = {
                    "model_state_dict": (
                        model.module.state_dict()
                        if hasattr(model, "module")
                        else model.state_dict()
                    ),
                    "discriminator_state_dict": (
                        discriminator.module.state_dict()
                        if (use_gan and discriminator is not None and hasattr(discriminator, "module"))
                        else (discriminator.state_dict() if (use_gan and discriminator is not None) else None)
                    ),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": (
                        optimizer_d.state_dict()
                        if (use_gan and (optimizer_d is not None))
                        else None
                    ),
                    "epoch": epoch,
                    "best_loss": best_val,
                    "data_dict": data_dict,
                }
                results_dir.mkdir(parents=True, exist_ok=True)
                torch.save(to_save, results_dir / "best_model.pt")

            if ((epoch + 1) % max(1, getattr(config, "save_every", 50)) == 0) and is_logger:
                to_save = {
                    "model_state_dict": (
                        model.module.state_dict()
                        if hasattr(model, "module")
                        else model.state_dict()
                    ),
                    "discriminator_state_dict": (
                        discriminator.module.state_dict()
                        if (use_gan and discriminator is not None and hasattr(discriminator, "module"))
                        else (discriminator.state_dict() if (use_gan and discriminator is not None) else None)
                    ),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": (
                        optimizer_d.state_dict()
                        if (use_gan and (optimizer_d is not None))
                        else None
                    ),
                    "epoch": epoch,
                    "best_loss": best_val,
                    "data_dict": data_dict,
                }
                torch.save(to_save, results_dir / f"model_{epoch+1}.pt")

            # Early stop
            if (early_stopper is not None) and early_stopper.step(val_loss):
                if is_logger:
                    print(f"[EarlyStop] best={best_val:.6f}")
                break

        if is_logger:
            print(f"VAL_LOSS:{best_val}")
    finally:
        if tb_writer:
            tb_writer.flush()
            tb_writer.close()

    return model, best_val, results_dir


# ------------------------------
# Inference
# ------------------------------
@torch.no_grad()
def run_inference(args):
    # Backend
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    config, runtime_ctx = get_seismic_config(args)
    device = runtime_ctx["device"]
    is_logger = runtime_ctx["is_logger"]
    world_size = runtime_ctx["world_size"]
    local_rank = runtime_ctx["local_rank"]

    (
        train_loader,
        val_loader,
        train_sampler,
        val_sampler,
        input_inv,
        output_inv,
        data_dict,
    ) = build_zarr_loaders_like_moe(args, config, world_size, local_rank, is_logger)

    # Build generator (same as train)
    gen_key = str(getattr(args, "generator", "InversionNet"))
    if gen_key == "UPFWI":
        gen_ctor = network.model_dict["UPFWI"]
        gen_kwargs = {
            "ratio": getattr(args, "sample_spatial", 1.0),
            "upsample_mode": getattr(args, "up_mode", None),
        }
    elif gen_key == "InversionNet":
        gen_ctor = network.model_dict["InversionNet"]
        gen_kwargs = {
            "sample_spatial": getattr(args, "sample_spatial", 1.0),
        }
    else:
        raise ValueError(f"未知生成器：{gen_key}；可选 InversionNet/UPFWI")

    generator = gen_ctor(**gen_kwargs).to(device)
    model = PlainAdapter(generator).to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint
    assert ckpt_path and os.path.exists(ckpt_path), f"checkpoint 不存在: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    (model.module if hasattr(model, "module") else model).load_state_dict(state, strict=False)

    # Choose split
    infer_split = getattr(args, "infer_split", "val")
    loader = val_loader if infer_split != "train" else train_loader

    # Metrics
    metrics = SeismicMetrics()

    # Output dirs
    save_dir = Path(getattr(args, "save_dir", "./infer_outputs"))
    vis_dir = save_dir / "vis"
    fft_dir = save_dir / "fft"
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    fft_dir.mkdir(parents=True, exist_ok=True)

    amp_enabled = bool(getattr(args, "use_amp", False)) and (device.type == "cuda")

    mse_sum = mae_sum = psnr_sum = rmse_sum = ssim_sum = 0.0
    n = 0
    max_vis = int(getattr(args, "max_vis", 8))
    dump_npy = bool(getattr(args, "dump_npy", False))

    model.eval()
    for bi, batch in enumerate(_tqdm.tqdm(loader, desc="Infer", disable=not is_logger)):
        x = batch["input"].to(device, non_blocking=True)
        y = batch["output"].to(device, non_blocking=True).to(dtype=torch.float32)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.bfloat16):
            pred, _, _ = model(x, use_amp=amp_enabled)
        # Metrics on normalized space
        mse_sum += float(metrics.calculate_mse(pred, y))
        mae_sum += float(metrics.calculate_mae(pred, y))
        psnr_sum += float(metrics.calculate_psnr(pred, y))
        rmse_sum += float(metrics.calculate_rmse(pred, y))
        ssim_sum += float(metrics.calculate_ssim(pred, y))
        n += 1

        # Visualize (inverse-transform to original scale)
        x_v = input_inv(x) if input_inv else x
        if output_inv := output_inv if callable(output_inv) else None:
            pass
        pred_v = output_inv(pred) if output_inv else pred
        target_v = output_inv(y) if output_inv else y

        num_samples = min(max_vis, x_v.shape[0])
        visualize_results(
            x_v.to(torch.float32),
            target_v.to(torch.float32),
            pred_v.to(torch.float32),
            save_dir=str(vis_dir),
            max_samples=num_samples,
            tb_writer=None,
            wandb_run=None,
            global_step=bi,
        )
        analyze_fourier_domain(
            x_v.to(torch.float32),
            target_v.to(torch.float32),
            pred_v.to(torch.float32),
            save_dir=str(fft_dir),
            max_samples=num_samples,
            tb_writer=None,
            wandb_run=None,
            global_step=bi,
        )

        if dump_npy:
            np.save(save_dir / f"inputs_b{bi}.npy", x_v.detach().cpu().numpy())
            np.save(save_dir / f"targets_b{bi}.npy", target_v.detach().cpu().numpy())
            np.save(save_dir / f"preds_b{bi}.npy", pred_v.detach().cpu().numpy())

    n = max(1, n)
    out_metrics = {
        "MSE": mse_sum / n,
        "MAE": mae_sum / n,
        "PSNR": psnr_sum / n,
        "RMSE": rmse_sum / n,
        "SSIM": ssim_sum / n,
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(out_metrics, f, indent=2)
    if is_logger:
        print("[Inference metrics]", out_metrics)


# ------------------------------
# CLI
# ------------------------------
def build_argparser_and_parse():
    import argparse

    p = argparse.ArgumentParser(
        "Plain Generator Training/Inference on moe-like pipeline with optional WGAN-GP"
    )

    # Device/Distributed
    p.add_argument("--device", default="cuda")
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--dist-url", default="env://")
    p.add_argument("--world-size", type=int, default=1)

    # IO & Zarr
    p.add_argument("--zarr_path", type=str, required=True)
    p.add_argument("--status_json", type=str, required=True)
    p.add_argument("--family", type=str, required=True)
    p.add_argument("--k", type=float, default=1.0)

    # Generator
    p.add_argument(
        "--generator",
        type=str,
        default="InversionNet",
        choices=["InversionNet", "UPFWI"],
    )
    p.add_argument("--up_mode", type=str, default=None)
    p.add_argument("--sample_spatial", type=float, default=1.0)

    # Plain training hyperparams
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--test_batch_size", type=int, default=64)

    p.add_argument("--lr_milestones", nargs="*", type=int, default=[])
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--lr_warmup_epochs", type=float, default=0.0)

    # Loss weights (plain)
    p.add_argument("--lambda_g1v", type=float, default=1.0)
    p.add_argument("--lambda_g2v", type=float, default=1.0)

    # AMP/Grad
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--max_norm", type=float, default=0.0)

    # Logs/Output
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--log_root", type=str, default="./runs")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--log_every", type=int, default=50)

    # Early stop
    p.add_argument("--early_stop", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--early_stop_warmup_epochs", type=int, default=10)

    # DeepSpeed (optional)
    p.add_argument("--use_deepspeed", action="store_true")
    p.add_argument("--ds_config", type=str, default="")

    # Dataloader
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # Mode
    p.add_argument("--mode", type=str, default="train", choices=["train", "inference"])
    p.add_argument("--resume_path", type=str, default=None)

    # Inference options
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--infer_split",
        type=str,
        default="val",
        choices=["train", "val"],
    )
    p.add_argument("--save_dir", type=str, default="./infer_outputs")
    p.add_argument("--dump_npy", action="store_true")
    p.add_argument("--max_vis", type=int, default=8)

    # ===== GAN options =====
    p.add_argument("--use_gan", action="store_true")
    p.add_argument("--discriminator", type=str, default="Discriminator")
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--lambda_adv", type=float, default=1.0)
    p.add_argument("--lambda_gp", type=float, default=10.0)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)

    args = p.parse_args()
    return args


# ------------------------------
# Entry
# ------------------------------
if __name__ == "__main__":
    args = build_argparser_and_parse()
    if args.mode == "train":
        run_training(args)
    else:
        run_inference(args)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Eval-only script for PDEBench-style datasets (burgers1d/navier2d/darcy2d/pipe/airfoil/plasticity).
Loads a trained checkpoint, runs metrics on val/test split, and optionally saves visualizations.
Training-time logging and visualization live in train_pde.py; this script only prints key metrics and writes them to JSON.
"""

import argparse
import json
from pathlib import Path

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose

# ========= In-repo path setup =========
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neuralop.mpu.comm as comm
from neuralop.training import setup
import scripts.transforms as T
from neuralop.data.datasets.pde_dataset import PDEBenchDataset
from neuralop.models.afreqmoe import AdaptiveFreqMoE  # noqa: F401  # keep import available
from neuralop.layers.spectral_convolution import SpectralConv
from config.distributed import DistributedConfig

from utils import (
    SeismicMetrics,
    patch_spectral_conv_forward,
)
from pde.train_pde import (
    PDEBenchConfig,
    build_emo_model,
    evaluate,
    is_main_process,
)
from utils.plot_fig import (
    analyze_fourier_domain,
    visualize_encoded,
    visualize_error_heatmap,
    visualize_results,
    visualize_routed_bands,
    visualize_router_selection_from_stats,
    visualize_pde_style,
    visualize_expert_freq_preference_from_router,
)

# SpectralConv forward patch for compatibility
patch_spectral_conv_forward(SpectralConv)


# ===========================================================
# 1. Argument parser
# ===========================================================

def build_argparser():
    parser = argparse.ArgumentParser("PDEBench EMO test script")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["navier", "darcy", "pipe", "airfoil", "plasticity"],
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("../pdebench_data"),
    )
    parser.add_argument(
        "--status_json",
        type=Path,
        default=Path("../pde_status.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trained checkpoint path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Eval split to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable AMP.",
    )
    parser.add_argument(
        "--no_amp",
        dest="use_amp",
        action="store_false",
    )
    parser.set_defaults(use_amp=True)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--band_sharpness",
        type=float,
        default=None,
        help="AFreqMoE soft band sharpness (None = use checkpoint config).",
    )
    parser.add_argument(
        "--freq_affinity_sharpness",
        type=float,
        default=None,
        help="AFreqMoE frequency-preference match sharpness (None = use checkpoint config).",
    )
    parser.add_argument(
        "--disable_soft_bands",
        action="store_true",
        help="Ablation: disable soft bands; use hard segmentation.",
    )
    parser.add_argument(
        "--disable_freq_attn",
        action="store_true",
        help="Ablation: disable frequency-domain self-attention.",
    )
    parser.add_argument(
        "--disable_band_mixing",
        action="store_true",
        help="Ablation: disable band-mixed inputs (each expert sees its band only).",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("./results_pdebench_test"),
        help="Directory to save metrics / visualizations.",
    )
    parser.add_argument(
        "--vis_every",
        type=int,
        default=1,
        help="Visualization frequency (for epoch index in eval loop).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable DDP-style multi-GPU test.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
    )
    return parser


# ===========================================================
# 2. Build DataLoader (single val/test split)
# ===========================================================

def build_eval_loader(cfg: PDEBenchConfig, args):
    """
    Build a single DataLoader for val/test split with the same normalization
    as train_pde.py.
    """
    status_path = Path(args.status_json).expanduser()
    if not status_path.exists():
        raise FileNotFoundError(
            f"PDEBench normalization stats file not found: {status_path}. "
            "Generate *_stats.json first via convert_pdebench_to_emo_format.py."
        )

    with open(status_path, "r", encoding="utf-8") as f:
        stats_dict = json.load(f)

    k_value = float(getattr(cfg, "k", 1.0))
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

    dataset = PDEBenchDataset(
        task=cfg.task,
        root=cfg.data_root,
        split=args.split,
        input_transform=input_transform,
        output_transform=output_transform,
    )

    if args.distributed and args.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False if sampler is None else False,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )

    return loader


# ===========================================================
# 3. main
# ===========================================================

def main():
    parser = build_argparser()
    args = parser.parse_args()

    # --- Basic RNG seeds & TF32 ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # === Load config from checkpoint and restore PDEBenchConfig safely ===
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("config", {}) or {}

    print(f"Loaded ckpt from {args.checkpoint}, epoch={ckpt.get('epoch')}, "
      f"best_mse={ckpt.get('best_mse')}, best_l2r={ckpt.get('best_l2r')}")
    
    # 1) Start from defaults, then override from ckpt (avoid raw dict for distributed)
    cfg = PDEBenchConfig()
    for k, v in ckpt_cfg.items():
        if not hasattr(cfg, k):
            continue
        if k == "distributed" and isinstance(v, dict):
            # Restore DistributedConfig safely from dict
            cfg.distributed = DistributedConfig(**v)
        else:
            setattr(cfg, k, v)

    # 2) Override runtime fields (task, data, batch, AMP, paths, etc.)
    cfg.task = args.task
    cfg.data_root = str(args.data_root)
    cfg.batch_size = args.batch_size
    cfg.test_batch_size = args.batch_size
    cfg.use_amp = args.use_amp
    cfg.amp_dtype = args.amp_dtype
    cfg.save_dir = str(args.save_dir)
    cfg.num_workers = args.num_workers
    cfg.distributed.seed = args.seed
    # Override AFreqMoE ablation flags when provided on CLI
    if args.band_sharpness is not None:
        cfg.band_sharpness = args.band_sharpness
    if args.freq_affinity_sharpness is not None:
        cfg.freq_affinity_sharpness = args.freq_affinity_sharpness
    if args.disable_soft_bands:
        cfg.use_soft_bands = False
    if args.disable_freq_attn:
        cfg.enable_freq_attn = False
    if args.disable_band_mixing:
        cfg.enable_band_mixing = False
    # Override vis_every for testing; avoid train-saved 0 blocking visualization
    cfg.vis_every = args.vis_every
    # Ensure cfg.epochs >= 1 for evaluate(epoch=0) reuse (even if unused here)
    if cfg.epochs <= 0:
        cfg.epochs = 1

    # === Init distributed/device, aligned with train ===
    if args.distributed:
        cfg.distributed.use_distributed = True
    device, is_logger = setup(cfg)
    cfg.is_logger = is_logger

    # Sync rank / world_size / local_rank
    args.rank = comm.get_global_rank()
    args.world_size = comm.get_world_size()
    args.local_rank = comm.get_local_rank()

    if is_main_process(cfg):
        args.save_dir.mkdir(parents=True, exist_ok=True)

    # === Data ===
    val_loader = build_eval_loader(cfg, args)

    # Infer channel count and spatial size from one sample
    try:
        sample = val_loader.dataset[0]
    except Exception:
        sample = next(iter(val_loader))

    cfg.in_channels = int(sample["input"].shape[0])
    cfg.out_channels = int(sample["output"].shape[0])
    cfg.img_size = tuple(sample["output"].shape[-2:])
    cfg.expert_configs[2]["default_in_shape"] = cfg.img_size

    # === Model ===
    emo = build_emo_model(cfg, device)
    cfg.has_complex_params = any(torch.is_complex(p) for p in emo.parameters())
    cfg.amp_enabled = cfg.use_amp and device.type == "cuda" and not cfg.has_complex_params
    if cfg.use_amp and cfg.has_complex_params and is_main_process(cfg):
        print("[AMP] Detected complex parameters (e.g., Fourier-domain weights); disabling AMP/GradScaler.")

    # DDP is optional: evaluate already all_reduces; mirror train if desired, then unwrap for eval.
    if args.distributed and args.world_size > 1:
        emo = nn.parallel.DistributedDataParallel(
            emo,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
            static_graph=False,
            gradient_as_bucket_view=True,
        )

    # === Load weights from checkpoint ===
    model_state = ckpt.get("model_state_dict", None)
    if model_state is None:
        raise RuntimeError("Checkpoint has no 'model_state_dict' key.")

    target_emo = emo.module if hasattr(emo, "module") else emo
    missing, unexpected = target_emo.load_state_dict(model_state, strict=False)
    print(f"missing keys: {missing}, unexpected keys: {unexpected}")
   
    # === Metrics & Eval ===
    metrics = SeismicMetrics()
    mse, mae, psnr, rmse, ssim, l2r, vis_payload = evaluate(
        emo,
        val_loader,
        device,
        metrics,
        cfg,
        args,
        epoch=0,
    )

    results = {
        "mse": float(mse),
        "mae": float(mae),
        "psnr": float(psnr),
        "rmse": float(rmse),
        "ssim": float(ssim),
        "l2_relative_error": float(l2r),
        "checkpoint": str(args.checkpoint),
        "task": cfg.task,
        "split": args.split,
    }

    # === Visualization ===
    if is_main_process(cfg):
        vis_dir = args.save_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_sample = vis_payload or {}
        inputs = vis_sample.get("inputs")
        targets = vis_sample.get("targets")
        preds = vis_sample.get("preds")
        encoded = vis_sample.get("encoded")

        # Denormalize + baseline visualization
        if inputs is not None and targets is not None and preds is not None:
            inv_in = getattr(cfg, "input_inverse_transform", None)
            inv_out = getattr(cfg, "output_inverse_transform", None)
            in_v = inv_in(inputs) if inv_in is not None else inputs
            tgt_v = inv_out(targets) if inv_out is not None else targets
            pred_v = inv_out(preds) if inv_out is not None else preds

            print(f"shapes for visualization:, int: {in_v.shape}, tgt: {tgt_v.shape}, pred: {pred_v.shape}")
            
            visualize_results(
                in_v,
                tgt_v,
                pred_v,
                save_dir=vis_dir,
                max_samples=min(4, in_v.shape[0]),
                task=cfg.task,
            )
            analyze_fourier_domain(
                in_v,
                tgt_v,
                pred_v,
                save_dir=vis_dir,
                max_samples=min(4, in_v.shape[0]),
                task=cfg.task,
            )
            visualize_error_heatmap(
                tgt_v,
                pred_v,
                save_dir=vis_dir,
                max_samples=min(4, in_v.shape[0]),
            )
            visualize_pde_style(
                in_v,
                tgt_v,
                pred_v,
                save_dir=vis_dir,
                task=cfg.task,
            )
            
        # Encoded-feature visualization
        if encoded is not None:
            visualize_encoded(
                encoded,
                save_dir=vis_dir,
                max_samples=min(4, encoded.shape[0]),
                selection="l2",
            )

        # Router / band visualization (AF-MoE only)
        base_emo = target_emo

        if isinstance(base_emo.moe, AdaptiveFreqMoE):
            router_vis_dir = vis_dir / "router"
            router_vis_dir.mkdir(parents=True, exist_ok=True)

            router_stats = None
            # 1) Router selection stats
            try:
                router_stats = base_emo.moe.get_router_stats()
                visualize_router_selection_from_stats(
                    router_stats,
                    save_dir=router_vis_dir,
                    epoch=0,
                    router_name=getattr(cfg, "router_type", "sar"),
                    tb_writer=None,
                    wandb_run=None,
                    global_step=None,
                )
            except Exception as e:
                print(f"[RouterVis] Band/expert selection visualization failed: {e}")

            # 2) Per-band routed outputs
            try:
                routed_bands = base_emo.moe.get_last_routed_bands()
                if routed_bands is not None:
                    band_centers = router_stats.get("band_centers") if router_stats else None
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
            except Exception as e:
                print(f"[RouterVis] routed_bands visualization failed: {e}")
            # 3) Per-expert frequency preference
            try:
                visualize_expert_freq_preference_from_router(
                    base_emo.moe.router,
                    save_dir=router_vis_dir,
                    epoch=0,
                    tb_writer=None,
                    wandb_run=None,
                    global_step=None,
                )
            except Exception as e:
                print(f"[RouterVis] Expert frequency-preference visualization failed: {e}")
                
        # Save metrics to JSON
        metrics_path = args.save_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Also print to terminal
        print(json.dumps(results, indent=2))

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

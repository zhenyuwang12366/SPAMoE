# scripts/infer_marmousi2.py
"""
Marmousi2 inference script (no tiling / sharding).
- Compute normalization statistics on Marmousi data
- Interpolate Marmousi waveforms to the training input spatial size
- Run full-field inference with EMO
- Interpolate outputs back to Marmousi GT spatial size
- Fully compatible with EMO + MoE
- FP32 inference only; no AMP
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torchvision.transforms import Compose
from argparse import Namespace

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# In-project modules
import scripts.transforms as T
from neuralop.models import MOEOperator, ExpertFactory
from neuralop.models.encoder import get_encoder
from neuralop.models.EMO import EMO
from neuralop.data.datasets.seismic_dataset import SeismicDataProcessor
from neuralop.utils import *
from utils import *


# ===============================
# Utilities
# ===============================

def compute_data_stats(arr_in: np.ndarray, arr_out: np.ndarray):
    """Compute input/output normalization stats on Marmousi data."""
    data_dict = {
        "input_min": float(np.min(arr_in)),
            "input_max": float(np.max(arr_in)),
        "input_mean": float(np.mean(arr_in)),
        "input_std": float(np.std(arr_in)),
        "output_min": float(np.min(arr_out)),
        "output_max": float(np.max(arr_out)),
        "output_mean": float(np.mean(arr_out)),
        "output_std": float(np.std(arr_out)),
    }
    return data_dict


def interpolate_to_target(tensor: torch.Tensor, target_shape: tuple[int, int]):
    """
    Bilinear interpolation to target size (H, W).
    Input: [B, C, H, W]
    Output: [B, C, target_H, target_W]
    """
    return F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)


def convert_keys_to_int(obj):
    """config.json keys may be string digits; normalize them to int where possible."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            try:
                k_int = int(k)
            except (ValueError, TypeError):
                k_int = k
            new_dict[k_int] = convert_keys_to_int(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_keys_to_int(i) for i in obj]
    else:
        return obj


# ===============================
# Main inference (no tiling)
# ===============================

def infer_marmousi2(n_args):
    """
    Full Marmousi2 inference pipeline (no tiling):
      - Load training args/config from setting_path
      - Load Marmousi seis / Ip npy files
      - Compute data_dict normalization stats on Marmousi
      - Interpolate Marmousi waveforms to training input spatial size
      - Run EMO full-field inference for a coarse velocity map
      - Interpolate coarse map back to Marmousi GT size
      - Compute metrics and visualization
    """
    # ===== Load saved training args/config; CLI overrides apply on top =====
    setting_dir = Path(getattr(n_args, "setting_path", ""))
    if not setting_dir:
        raise ValueError(
            "Inference requires --setting_path (directory with saved args.json and config.json from training)"
        )
    if not setting_dir.exists():
        raise ValueError(f"--setting_path does not exist: {setting_dir}")

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

    # CLI overrides for training args
    runtime_args_dict = dict(stored_args_dict)
    for key, value in vars(n_args).items():
        if value is not None:
            runtime_args_dict[key] = value
    runtime_args_dict["mode"] = "inference"
    runtime_args = Namespace(**runtime_args_dict)

    # ===== Init config / runtime =====
    config, runtime_ctx = get_seismic_config(runtime_args)
    stored_config_dict = convert_keys_to_int(stored_config_dict)

    # Merge fields from saved training config
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

    device = runtime_ctx["device"]
    is_logger = runtime_ctx["is_logger"]

    # -------------------------
    # Load Marmousi data
    # -------------------------
    seis = np.load(n_args.seis_path)  # waveforms
    gt = np.load(n_args.gt_path)      # GT velocity / Ip

    seis_t = torch.from_numpy(seis).float().to(device)
    gt_t = torch.from_numpy(gt).float().to(device)

    # Normalize to [B, C, H, W]
    if seis_t.ndim == 3:
        seis_t = seis_t.unsqueeze(0)       # [1,H,W] -> [1,1,H,W]
    if seis_t.ndim == 2:
        seis_t = seis_t.unsqueeze(0).unsqueeze(0)
    if gt_t.ndim == 3:
        gt_t = gt_t.unsqueeze(0)
    if gt_t.ndim == 2:
        gt_t = gt_t.unsqueeze(0).unsqueeze(0)

    if seis_t.shape[0] != 1:
        raise ValueError(
            f"This script supports single-sample inference only; got batch={seis_t.shape[0]}"
        )

    # -------------------------
    # Stats + transforms (on Marmousi)
    # -------------------------
    data_dict = compute_data_stats(seis, gt)
    k = float(getattr(n_args, "k", 1.0))

    input_transform = Compose([
        T.LogTransform(k=k),
        T.MinMaxNormalize(
            T.log_transform(data_dict["input_min"], k=k),
            T.log_transform(data_dict["input_max"], k=k),
        ),
    ])
    output_inverse_transform = Compose([
        T.InverseMinMaxNormalize(
            data_dict["output_min"], data_dict["output_max"]
        )
    ])

    # Keep interface consistent (data_processor itself is unused here)
    _ = SeismicDataProcessor(
        input_transform=input_transform,
        output_transform=None,
        channel_dim=seis_t.shape[1],
        config=config,
    )

    seis_t = input_transform(seis_t)   # normalized Marmousi waveforms

    # -------------------------
    # Build EMO (encoder + MoE)
    # -------------------------
    # --- Encoder ---
    encoder_model = None
    if getattr(config, "use_encoder", False):
        encoder_model = get_encoder(
            in_channels=seis_t.shape[1],
            out_channels=128,
            num_types=int(getattr(config, "v_type_num", 10) or 10),
            type_act="softmax",
            backbone=config.backbone,
        ).to(device)
        encoder_model.eval()
        for p in encoder_model.parameters():
            p.requires_grad_(False)

        enc_path = getattr(n_args, "encoder_path", None)
        if enc_path and os.path.exists(enc_path):
            missing, unexpected = load_encoder_weights(
                encoder_model, enc_path, map_location=device, strict=False
            )
            if is_logger:
                print(f"[Encoder] Loaded: {enc_path}")
                if missing:
                    print(f"[Encoder] Missing: {missing}")
                if unexpected:
                    print(f"[Encoder] Unexpected: {unexpected}")

        # Forward once only to infer moe_in_channels; not used for main pass
        with torch.no_grad():
            feat, _, _ = encoder_model(seis_t)
        moe_in_channels = feat.shape[1]
    else:
        moe_in_channels = seis_t.shape[1]

    # --- Experts ---
    if getattr(config, "use_moe", False) and getattr(config, "use_experts_path", None):
        experts = load_moe_experts(
            experts_config=getattr(config, "load_expert_configs", config.expert_configs),
            in_channels=moe_in_channels,
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
            in_channels=moe_in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels,
        )

    # --- MoE + EMO ---
    moe = MOEOperator(
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
        use_encoder=getattr(config, "use_encoder", False),
        device=device,
    )
    emo = EMO(encoder_model, moe, pass_encoder_logits_as_weights=True).to(device)
    emo.eval()

    # -------------------------
    # Load weights (subset aligned with training resume)
    # -------------------------
    model_path = getattr(n_args, "model_path", None)
    if not model_path or not os.path.exists(model_path):
        raise ValueError(f"Model checkpoint not found: {model_path}")

    # Load on CPU then move to device to reduce GPU memory fragmentation
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # No DDP/DeepSpeed wrapper at inference; emo is the final model
    target_model = emo
    target_moe = getattr(target_model, "moe", None)

    # Restore experts_name / experts_name_str from config / runtime_ctx
    experts_name = getattr(config, "experts_name", None)
    if experts_name is None:
        experts_name = runtime_ctx.get("experts_name", None)

    experts_name_str = None
    if experts_name is not None:
        if isinstance(experts_name, (list, tuple)):
            experts_name_str = "_".join(map(str, experts_name))
        else:
            experts_name_str = str(experts_name)

    # Parse checkpoint components
    model_check   = checkpoint.get("model_state_dict", None)
    router_check  = checkpoint.get("router_state_dict", None)
    encoder_check = checkpoint.get("encoder_state_dict", None)
    expert_check  = checkpoint.get("expert_state_dict", None)  # optional in single-expert mode

    # Track external encoder path override
    enc_path_override = getattr(n_args, "encoder_path", None)

    # ======================================================
    # Case A: experts_name_str == "all" and ckpt has only router_state_dict
    #         (training saved router_state_dict (+ encoder_state_dict) only)
    # ======================================================
    if (experts_name_str == "all") and (model_check is None) and (router_check is not None):
        # 1) Restore router
        if (target_moe is not None) and hasattr(target_moe, "router") and (target_moe.router is not None):
            missing, unexpected = target_moe.router.load_state_dict(router_check, strict=False)
            if is_logger and (missing or unexpected):
                print(f"[Infer][all][router] missing keys: {missing}, unexpected keys: {unexpected}")
            elif is_logger:
                print(f"[Infer][all][router] restored router_state_dict from {model_path}")
        elif is_logger:
            print("[Infer][all][router] no moe.router on model; skipping router restore")

        # 2) Restore encoder (unless --encoder_path is set)
        if hasattr(target_model, "encoder") and (target_model.encoder is not None) \
                and (encoder_check is not None) and not enc_path_override:
            missing, unexpected = load_encoder_weights(
                target_model.encoder, encoder_check, strict=False
            )
            if is_logger and (missing or unexpected):
                print(f"[Infer][all][encoder] missing keys: {missing}, unexpected keys: {unexpected}")
            elif is_logger:
                print("[Infer][all][encoder] restored encoder_state_dict from checkpoint")
        elif encoder_check is not None and enc_path_override and is_logger:
            print("[Infer][all][encoder] --encoder_path set; using external encoder, skipping ckpt encoder")

    # ======================================================
    # Case B: other setups (non-all, or all with model_state_dict in ckpt)
    # ======================================================
    else:
        # 1) Prefer full-model load (model_state_dict)
        if model_check is not None:
            missing, unexpected = target_model.load_state_dict(model_check, strict=False)
            if is_logger and (missing or unexpected):
                print(f"[Infer][generic][model] missing keys: {missing}, unexpected keys: {unexpected}")
            elif is_logger:
                print(f"[Infer][generic][model] restored model_state_dict from {model_path}")
        elif is_logger:
            print("[Infer][generic] checkpoint has no model_state_dict; skipping full-model restore")

        # 2) Single-expert: expert_state_dict can overwrite experts[0]
        if experts_name is not None and isinstance(experts_name, (list, tuple)) \
                and len(experts_name) == 1 and experts_name[0] != "all":
            if expert_check is not None and (target_moe is not None) \
                    and hasattr(target_moe, "experts") and len(target_moe.experts) > 0:
                missing, unexpected = target_moe.experts[0].load_state_dict(expert_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Infer][single-expert][experts[0]] missing keys: {missing}, unexpected keys: {unexpected}")
                elif is_logger:
                    print("[Infer][single-expert] overwrote experts[0] from checkpoint")
            elif is_logger and expert_check is not None:
                print("[Infer][single-expert] checkpoint has expert_state_dict but no experts[0]; skipping")

        # 3) Encoder: restore from ckpt if no external encoder_path
        if hasattr(target_model, "encoder") and (target_model.encoder is not None) \
                and (encoder_check is not None) and not enc_path_override:
            missing, unexpected = load_encoder_weights(
                target_model.encoder, encoder_check, strict=False
            )
            if is_logger and (missing or unexpected):
                print(f"[Infer][generic][encoder] missing keys: {missing}, unexpected keys: {unexpected}")
            elif is_logger:
                print("[Infer][generic][encoder] restored encoder_state_dict from checkpoint")
        elif encoder_check is not None and enc_path_override and is_logger:
            print("[Infer][generic][encoder] --encoder_path set; using external encoder, skipping ckpt encoder")

        # 4) Router: optional second pass if router_state_dict is present
        if router_check is not None and (target_moe is not None) \
                and hasattr(target_moe, "router") and (target_moe.router is not None):
            try:
                missing, unexpected = target_moe.router.load_state_dict(router_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Infer][generic][router] missing keys: {missing}, unexpected keys: {unexpected}")
                elif is_logger:
                    print("[Infer][generic][router] overwrote router_state_dict from checkpoint")
            except Exception as e:
                if is_logger:
                    print(f"[Infer][generic][router] skipped router_state_dict load: {e}")

    # -------------------------
    # Core: interpolate to training input size, full-field inference
    # -------------------------

    # 1) Training input spatial size (H_train, W_train)
    #    Prefer config; change field names here if your config uses others
    h_train = getattr(config, "input_height", None)
    w_train = getattr(config, "input_width", None)

    if h_train is not None and w_train is not None:
        train_input_shape = (int(h_train), int(w_train))
    else:
        # Fallback if config omits training input spatial size: set (H, W) used when training EMO
        # e.g. OpenFWI often uses (1000, 70) (time samples x receivers)
        train_input_shape = (1000, 70)   # <<< edit to match your training setup

    if is_logger:
        print(f"[Infer] interpolating Marmousi waveforms to training input size: {train_input_shape}")

    seis_interp = interpolate_to_target(seis_t, train_input_shape)  # [1,C,H_train,W_train]

    # 2) Full-field inference
    with torch.no_grad():
        pred_coarse, _, _ = emo(seis_interp)   # e.g. [1,1,70,70] or [1,1,H',W']

    # 3) Upsample coarse output to Marmousi GT size
    target_output_shape = (gt_t.shape[-2], gt_t.shape[-1])  # e.g. (13601, 2801)
    if is_logger:
        print(f"[Infer] interpolating coarse output to GT size: {target_output_shape}")

    pred_full = interpolate_to_target(pred_coarse, target_output_shape)
    pred_full = output_inverse_transform(pred_full)

    # -------------------------
    # Save results and metrics
    # -------------------------
    results_dir = Path(getattr(n_args, "output_dir", "./marmousi_results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "marmousi_pred.npy", pred_full.detach().cpu().numpy())

    metrics = SeismicMetrics()
    mse = metrics.calculate_mse(pred_full, gt_t)
    mae = metrics.calculate_mae(pred_full, gt_t)
    psnr = metrics.calculate_psnr(pred_full, gt_t)
    rmse = metrics.calculate_rmse(pred_full, gt_t)
    ssim = metrics.calculate_ssim(pred_full, gt_t)

    print(f"[Metrics] MSE={mse:.6f}, MAE={mae:.6f}, PSNR={psnr:.4f}, RMSE={rmse:.6f}, SSIM={ssim:.6f}")

    visualize_results(
        seis_t.detach().cpu(),   # Marmousi waveforms after normalization
        gt_t.detach().cpu(),
        pred_full.detach().cpu(),
        save_dir=results_dir,
        max_samples=1,
    )
    analyze_fourier_domain(
        seis_t.detach().cpu(),
        gt_t.detach().cpu(),
        pred_full.detach().cpu(),
        save_dir=results_dir,
        max_samples=1,
    )

    print(f"Inference finished; results saved under {results_dir}")


# ===============================
# CLI entry
# ===============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Marmousi2 inference script (no tiling)")
    parser.add_argument("--seis_path", type=str,
                        default="./marmousi/marmousi_synthetic_seismic.npy",
                        help="Path to seismic waveform .npy")
    parser.add_argument("--gt_path", type=str,
                        default="./marmousi/marmousi_Ip_model.npy",
                        help="Path to velocity / Ip ground-truth .npy")
    parser.add_argument("--model_path", type=str, required=True,
                        help="MoE checkpoint path from training")
    parser.add_argument("--encoder_path", type=str, default=None,
                        help="Encoder weights path (if encoder was used in training)")
    parser.add_argument("--output_dir", type=str, default="./marmousi_results",
                        help="Output directory")
    parser.add_argument("--k", type=float, default=1.0,
                        help="LogTransform k parameter")
    parser.add_argument("--use_moe", action="store_true",
                        help="Enable MoE (keep consistent with training config)")
    parser.add_argument("--use_encoder", action="store_true",
                        help="Use encoder (keep consistent with training config)")
    parser.add_argument("--setting_path", type=str, required=True,
                        help="Training run directory (contains args.json and config.json)")
    parser.add_argument("--use_experts_path", type=str, default="../other_experts",
                        help="Expert weights directory (if required by config)")

    args = parser.parse_args()
    infer_marmousi2(args)

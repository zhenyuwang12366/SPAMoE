# scripts/infer_marmousi2.py
"""
Marmousi2 推理脚本
- 自动计算数据统计量
- 自动进行插值、分块、拼接
- 完全兼容 EMO + MoE 架构
- 无需 AMP，仅 FP32 推理
"""

import os
import sys
import json
import math
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from argparse import Namespace
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
import scripts.transforms as T
from neuralop.models import MOEOperator, ExpertFactory
from neuralop.models.encoder import get_encoder
from neuralop.models.EMO import EMO
from neuralop.data.datasets.seismic_dataset import SeismicDataProcessor
from neuralop.utils import *
from utils import *

# ===============================
# 数据处理与分块函数
# ===============================

def compute_data_stats(arr_in: np.ndarray, arr_out: np.ndarray):
    """计算输入和输出的归一化统计量"""
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
    """插值到目标尺寸 (height, width)"""
    return F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)


def make_patches(tensor, patch_size=(1000, 350), halo=(200, 25)):
    """
    输入: [1,1,H,W]
    输出: patch 列表，每个 [1,1,1000,350]
    """
    _, _, H, W = tensor.shape
    ph, pw = patch_size
    hh, hw = halo
    stride_h = ph - 2 * hh
    stride_w = pw - 2 * hw

    patches, coords = [], []
    y = 0
    while y < H:
        if y + ph > H:
            y = H - ph
        x = 0
        while x < W:
            if x + pw > W:
                x = W - pw
            patch = tensor[:, :, y:y+ph, x:x+pw]
            patches.append(patch)
            coords.append((y, x))
            if x + pw >= W:
                break
            x += stride_w
        if y + ph >= H:
            break
        y += stride_h
    return patches, coords


def stitch_patches(patches, coords, full_shape, patch_size=(1000, 350), halo=(200, 25)):
    """根据 halo 融合分块，重建整图"""
    _, _, H, W = full_shape
    ph, pw = patch_size
    hh, hw = halo
    result = torch.zeros(full_shape, dtype=patches[0].dtype, device=patches[0].device)
    weight = torch.zeros_like(result)

    for patch, (y, x) in zip(patches, coords):
        yh0, yh1 = y + hh, y + ph - hh
        xw0, xw1 = x + hw, x + pw - hw
        core = patch[:, :, hh:ph-hh, hw:pw-hw]
        result[:, :, yh0:yh1, xw0:xw1] += core
        weight[:, :, yh0:yh1, xw0:xw1] += 1.0

    weight = torch.where(weight == 0, torch.ones_like(weight), weight)
    return result / weight


# ===============================
# 主推理流程
# ===============================

def infer_marmousi2(n_args):
    """
    对 Marmousi2 执行完整推理流程
    """
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
    
    experts_name_str = runtime_ctx["experts_name_str"]
    device = runtime_ctx["device"]
    is_logger = runtime_ctx["is_logger"]

    # -------------------------
    # 加载数据
    # -------------------------
    seis = np.load(n_args.seis_path)  # [1,1,2721,701]
    gt = np.load(n_args.gt_path)      # [1,1,13601,2801]
    seis_t = torch.from_numpy(seis).float().to(device).unsqueeze(0)
    gt_t = torch.from_numpy(gt).float().to(device).unsqueeze(0)

    # -------------------------
    # 计算数据统计量 + 变换
    # -------------------------
    data_dict = compute_data_stats(seis, gt)
    k = getattr(n_args, "k", 1.0)

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
    data_processor = SeismicDataProcessor(
        input_transform=input_transform,
        output_transform=None,
        channel_dim=1,
        config=config,
    )

    seis_t = input_transform(seis_t)
    gt_t = output_inverse_transform(gt_t)

    # -------------------------
    # 构建模型（Encoder + MoE + EMO）
    # -------------------------
    # --- Encoder ---
    encoder_model = None
    if getattr(config, "use_encoder", False):
        encoder_model = get_encoder(
            in_channels=1,
            out_channels=128,
            num_types=10,
            type_act="softmax",
            backbone=config.backbone,
        ).to(device)
        encoder_model.eval()
        for p in encoder_model.parameters():
            p.requires_grad_(False)

        enc_path = getattr(args, "encoder_path", None)
        if enc_path and os.path.exists(enc_path):
            missing, unexpected = load_encoder_weights(encoder_model, enc_path, map_location=device, strict=False)
            if is_logger:
                print(f"[Encoder] Loaded: {enc_path}")
                if missing: print(f"Missing: {missing}")
                if unexpected: print(f"Unexpected: {unexpected}")

        with torch.no_grad():
            feat, _, _ = encoder_model(seis_t)
        moe_in_channels = feat.shape[1]
    else:
        moe_in_channels = 1

    # --- Experts ---
    if getattr(config, "use_moe", False):
        experts = load_moe_experts(
            experts_config=config.expert_configs,
            in_channels=moe_in_channels,
            out_channels=1,
            hidden_channels=config.hidden_channels,
            model_path=config.use_experts_path,
            is_specific=config.is_specific,
            map_location=device,
            type_dict=config.type_id,
            moe_mode=config.moe_mode,
        )
    else:
        experts = ExpertFactory.create_expert_ensemble(
            expert_configs=config.expert_configs,
            in_channels=moe_in_channels,
            out_channels=1,
            hidden_channels=config.hidden_channels,
        )

    # --- MoE + EMO ---
    moe = MOEOperator(
        experts=experts,
        in_channels=moe_in_channels,
        out_channels=1,
        hidden_channels=config.hidden_channels,
        top_k=config.top_k,
        router_hidden_dim=config.router_hidden_dim,
        router_type=config.router_type,
        moe_mode=config.moe_mode,
        use_expert_memory_proxy=False,
        use_encoder=getattr(config, "use_encoder", False),
        device=device,
    )
    emo = EMO(encoder_model, moe, pass_encoder_logits_as_weights=True).to(device)
    emo.eval()

    model_path = getattr(args, "model_path", None)
    ckpt = torch.load(model_path, map_location=device)
    if "model_state_dict" in ckpt:
        emo.moe.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "router_state_dict" in ckpt:
        emo.moe.router.load_state_dict(ckpt["router_state_dict"], strict=False)
    else:
        raise ValueError("Checkpoint中未找到model_state_dict或router_state_dict")

    # -------------------------
    # 插值到可分块尺寸
    # -------------------------
    target_input_shape = (3000, 1050)
    seis_interp = interpolate_to_target(seis_t, target_input_shape)

    # -------------------------
    # 分块推理
    # -------------------------
    patches, coords = make_patches(seis_interp, patch_size=(1000, 350), halo=(200, 25))
    preds = []
    for p in tqdm(patches, desc="Patch 推理中"):
        with torch.no_grad():
            out, _, _ = emo(p)
            preds.append(out)

    # 拼接整图
    pred_full = stitch_patches(preds, coords, (1, 1) + target_input_shape, (1000, 350), (200, 25))
    pred_full = interpolate_to_target(pred_full, (13601, 2801))
    pred_full = output_inverse_transform(pred_full)

    # -------------------------
    # 保存结果与指标
    # -------------------------
    results_dir = Path(getattr(args, "output_dir", "./marmousi_results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / "marmousi_pred.npy", pred_full.cpu().numpy())

    metrics = SeismicMetrics()
    mse = metrics.calculate_mse(pred_full, gt_t)
    mae = metrics.calculate_mae(pred_full, gt_t)
    psnr = metrics.calculate_psnr(pred_full, gt_t)
    rmse = metrics.calculate_rmse(pred_full, gt_t)
    ssim = metrics.calculate_ssim(pred_full, gt_t)
    print(f"[Metrics] MSE={mse:.6f}, MAE={mae:.6f}, PSNR={psnr:.4f}, RMSE={rmse:.6f}, SSIM={ssim:.6f}")

    visualize_results(seis_t.cpu(), gt_t.cpu(), pred_full.cpu(), save_dir=results_dir, max_samples=1)
    analyze_fourier_domain(seis_t.cpu(), gt_t.cpu(), pred_full.cpu(), save_dir=results_dir, max_samples=1)

    print(f"推理完成，结果保存在 {results_dir}")

# ===============================
# CLI 入口
# ===============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Marmousi2 推理脚本")
    parser.add_argument("--seis_path", type=str, default="./marmousi/marmousi_synthetic_seismic.npy", help="地震波形 numpy 文件路径")
    parser.add_argument("--gt_path", type=str, default="./marmousi/marmousi_Ip_model.npy", help="速度图 ground truth 文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="MoE 模型权重路径")
    parser.add_argument("--encoder_path", type=str, default=None, help="Encoder 权重路径（可选）")
    parser.add_argument("--output_dir", type=str, default="./marmousi_results", help="输出目录")
    parser.add_argument("--k", type=float, default=1.0, help="LogTransform 参数")
    parser.add_argument("--use_moe", action="store_true", help="是否启用 MoE")
    parser.add_argument("--use_encoder", action="store_true", help="是否使用 encoder")
    parser.add_argument("--setting_path", type=str, default=None)
    parser.add_argument("--use_experts_path", type=str, default="../other_experts")
    args = parser.parse_args()

    infer_marmousi2(args)
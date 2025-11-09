# scripts/infer_marmousi2.py
"""
Marmousi2 推理脚本（无分片版）
- 在 Marmousi 数据上计算归一化统计量
- 通过插值将 Marmousi 波形对齐到训练时的输入尺寸
- 使用 EMO 进行整图推理
- 将输出再插值回 Marmousi GT 的空间尺寸
- 完全兼容 EMO + MoE 架构
- 无需 AMP，仅 FP32 推理
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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 项目内模块
import scripts.transforms as T
from neuralop.models import MOEOperator, ExpertFactory
from neuralop.models.encoder import get_encoder
from neuralop.models.EMO import EMO
from neuralop.data.datasets.seismic_dataset import SeismicDataProcessor
from neuralop.utils import *
from utils import *


# ===============================
# 工具函数
# ===============================

def compute_data_stats(arr_in: np.ndarray, arr_out: np.ndarray):
    """在 Marmousi 数据上计算输入和输出的归一化统计量"""
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
    双线性插值到目标尺寸 (H, W)
    输入: [B,C,H,W]
    输出: [B,C,target_H,target_W]
    """
    return F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)


def convert_keys_to_int(obj):
    """config.json 中 key 可能是字符串数字，这里统一转成 int"""
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
# 主推理流程（无分片）
# ===============================

def infer_marmousi2(n_args):
    """
    对 Marmousi2 执行完整推理流程（不分片）：
      - 从 setting_path 载入训练时的 args/config
      - 加载 Marmousi seis/Ip npy
      - 在 Marmousi 上计算 data_dict 归一化统计量
      - 将 Marmousi 波形插值到“训练输入空间尺寸”
      - 使用 EMO 整图推理，得到粗分辨率速度图
      - 将粗分辨率速度图插值回 Marmousi GT 尺寸
      - 计算指标 & 可视化
    """
    # ===== 读取训练期保存的 args/config，并被 CLI 覆盖 =====
    setting_dir = Path(getattr(n_args, "setting_path", ""))
    if not setting_dir:
        raise ValueError("推理需要 --setting_path（包含训练时保存的 args.json 与 config.json）")
    if not setting_dir.exists():
        raise ValueError(f"--setting_path 不存在: {setting_dir}")

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

    # CLI 覆盖训练 args
    runtime_args_dict = dict(stored_args_dict)
    for key, value in vars(n_args).items():
        if value is not None:
            runtime_args_dict[key] = value
    runtime_args_dict["mode"] = "inference"
    runtime_args = Namespace(**runtime_args_dict)

    # ===== 初始化 config / 运行环境 =====
    config, runtime_ctx = get_seismic_config(runtime_args)
    stored_config_dict = convert_keys_to_int(stored_config_dict)

    # 回填训练 config 字段
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
    # 加载 Marmousi 数据
    # -------------------------
    seis = np.load(n_args.seis_path)  # 波形
    gt = np.load(n_args.gt_path)      # 真实速度/Ip

    seis_t = torch.from_numpy(seis).float().to(device)
    gt_t = torch.from_numpy(gt).float().to(device)

    # 统一到 [B,C,H,W]
    if seis_t.ndim == 3:
        seis_t = seis_t.unsqueeze(0)       # [1,H,W] -> [1,1,H,W]
    if seis_t.ndim == 2:
        seis_t = seis_t.unsqueeze(0).unsqueeze(0)
    if gt_t.ndim == 3:
        gt_t = gt_t.unsqueeze(0)
    if gt_t.ndim == 2:
        gt_t = gt_t.unsqueeze(0).unsqueeze(0)

    if seis_t.shape[0] != 1:
        raise ValueError(f"当前脚本只支持单样本推理，收到 batch={seis_t.shape[0]}")

    # -------------------------
    # 计算数据统计量 + 变换（在 Marmousi 上）
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

    # 保持接口一致（虽然这里只用不到 data_processor 本身）
    _ = SeismicDataProcessor(
        input_transform=input_transform,
        output_transform=None,
        channel_dim=seis_t.shape[1],
        config=config,
    )

    seis_t = input_transform(seis_t)   # 归一化后的 Marmousi 波形

    # -------------------------
    # 构建 EMO (Encoder + MoE)
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

        # 这里只是为了确定 moe_in_channels，不参与真正推理
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
    # 加载权重（对齐训练 resume 逻辑的子集）
    # -------------------------
    model_path = getattr(n_args, "model_path", None)
    if not model_path or not os.path.exists(model_path):
        raise ValueError(f"模型文件不存在: {model_path}")

    # 用 CPU 加载，再 to(device)，避免显存碎片
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # 推理阶段不会再有 DDP/DeepSpeed 包壳，emo 就是最终模型
    target_model = emo
    target_moe = getattr(target_model, "moe", None)

    # 从 config / runtime_ctx 中恢复 experts_name / experts_name_str
    experts_name = getattr(config, "experts_name", None)
    if experts_name is None:
        experts_name = runtime_ctx.get("experts_name", None)

    experts_name_str = None
    if experts_name is not None:
        if isinstance(experts_name, (list, tuple)):
            experts_name_str = "_".join(map(str, experts_name))
        else:
            experts_name_str = str(experts_name)

    # 从 ckpt 中解析各个分量
    model_check   = checkpoint.get("model_state_dict", None)
    router_check  = checkpoint.get("router_state_dict", None)
    encoder_check = checkpoint.get("encoder_state_dict", None)
    expert_check  = checkpoint.get("expert_state_dict", None)  # 单专家模式可选

    # 便于下面判断 encoder 是否用外部路径覆盖
    enc_path_override = getattr(n_args, "encoder_path", None)

    # ======================================================
    # 情况 A：experts_name_str == "all" 且 ckpt 里只有 router_state_dict
    #         —— 训练端只保存了 router_state_dict (+ encoder_state_dict)
    # ======================================================
    if (experts_name_str == "all") and (model_check is None) and (router_check is not None):
        # 1) 恢复 Router
        if (target_moe is not None) and hasattr(target_moe, "router") and (target_moe.router is not None):
            missing, unexpected = target_moe.router.load_state_dict(router_check, strict=False)
            if is_logger and (missing or unexpected):
                print(f"[Infer][all][router] 缺失参数: {missing}, 多余参数: {unexpected}")
            elif is_logger:
                print(f"[Infer][all][router] 已从 {model_path} 恢复 router_state_dict")
        elif is_logger:
            print("[Infer][all][router] 模型中不存在 moe.router，跳过 Router 恢复")

        # 2) 恢复 Encoder（前提：没指定 --encoder_path）
        if hasattr(target_model, "encoder") and (target_model.encoder is not None) \
                and (encoder_check is not None) and not enc_path_override:
            missing, unexpected = load_encoder_weights(
                target_model.encoder, encoder_check, strict=False
            )
            if is_logger and (missing or unexpected):
                print(f"[Infer][all][encoder] 缺失参数: {missing}, 多余参数: {unexpected}")
            elif is_logger:
                print(f"[Infer][all][encoder] 已从 ckpt 恢复 encoder_state_dict")
        elif encoder_check is not None and enc_path_override and is_logger:
            print(f"[Infer][all][encoder] 检测到 --encoder_path，优先使用外部 encoder，跳过 ckpt encoder")

    # ======================================================
    # 情况 B：其它普通场景（包括非 all 模式，或 all 但 ckpt 中带 model_state_dict）
    # ======================================================
    else:
        # 1) 优先尝试整模恢复（model_state_dict）
        if model_check is not None:
            missing, unexpected = target_model.load_state_dict(model_check, strict=False)
            if is_logger and (missing or unexpected):
                print(f"[Infer][generic][model] 缺失参数: {missing}, 多余参数: {unexpected}")
            elif is_logger:
                print(f"[Infer][generic][model] 已从 {model_path} 恢复 model_state_dict")
        elif is_logger:
            print("[Infer][generic] ckpt 未包含 model_state_dict，跳过整模恢复")

        # 2) 单专家模式：若 ckpt 里带 expert_state_dict，可进一步覆盖 experts[0]
        if experts_name is not None and isinstance(experts_name, (list, tuple)) \
                and len(experts_name) == 1 and experts_name[0] != "all":
            if expert_check is not None and (target_moe is not None) \
                    and hasattr(target_moe, "experts") and len(target_moe.experts) > 0:
                missing, unexpected = target_moe.experts[0].load_state_dict(expert_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Infer][single-expert][experts[0]] 缺失参数: {missing}, 多余参数: {unexpected}")
                elif is_logger:
                    print(f"[Infer][single-expert] 已从 ckpt 覆盖 experts[0]")
            elif is_logger and expert_check is not None:
                print("[Infer][single-expert] ckpt 带 expert_state_dict 但模型无 experts[0]，跳过")

        # 3) Encoder：如果没用外部 encoder_path，就尝试从 ckpt 恢复
        if hasattr(target_model, "encoder") and (target_model.encoder is not None) \
                and (encoder_check is not None) and not enc_path_override:
            missing, unexpected = load_encoder_weights(
                target_model.encoder, encoder_check, strict=False
            )
            if is_logger and (missing or unexpected):
                print(f"[Infer][generic][encoder] 缺失参数: {missing}, 多余参数: {unexpected}")
            elif is_logger:
                print(f"[Infer][generic][encoder] 已从 ckpt 恢复 encoder_state_dict")
        elif encoder_check is not None and enc_path_override and is_logger:
            print(f"[Infer][generic][encoder] 检测到 --encoder_path，优先使用外部 encoder，跳过 ckpt encoder")

        # 4) Router：如果 ckpt 里额外带了 router_state_dict，可以再尝试覆盖一次
        if router_check is not None and (target_moe is not None) \
                and hasattr(target_moe, "router") and (target_moe.router is not None):
            try:
                missing, unexpected = target_moe.router.load_state_dict(router_check, strict=False)
                if is_logger and (missing or unexpected):
                    print(f"[Infer][generic][router] 缺失参数: {missing}, 多余参数: {unexpected}")
                elif is_logger:
                    print(f"[Infer][generic][router] 已从 ckpt 覆盖 router_state_dict")
            except Exception as e:
                if is_logger:
                    print(f"[Infer][generic][router] 跳过 router_state_dict 加载：{e}")

    # -------------------------
    # 关键部分：整图插值到“训练输入尺寸”，整图推理
    # -------------------------

    # 1) 确定训练时的输入空间尺寸 (H_train, W_train)
    #    优先从 config 里读；如果你的 config 里有别的字段名，改成对应的即可
    h_train = getattr(config, "input_height", None)
    w_train = getattr(config, "input_width", None)

    if h_train is not None and w_train is not None:
        train_input_shape = (int(h_train), int(w_train))
    else:
        # fallback：如果 config 没有明确定义训练输入空间尺寸，
        # 就手动填写你训练 EMO 时使用的 (H, W)
        # 例如：OpenFWI 通常是 (1000, 70)（时间步 1000，接收点 70）
        train_input_shape = (1000, 70)   # <<< 按你真实的训练设置改掉这一行

    if is_logger:
        print(f"[Infer] 将 Marmousi 波形插值到训练输入尺寸: {train_input_shape}")

    seis_interp = interpolate_to_target(seis_t, train_input_shape)  # [1,C,H_train,W_train]

    # 2) 整图推理
    with torch.no_grad():
        pred_coarse, _, _ = emo(seis_interp)   # 例如 [1,1,70,70] 或 [1,1,H',W']

    # 3) 将粗分辨率输出插值回 Marmousi GT 尺寸
    target_output_shape = (gt_t.shape[-2], gt_t.shape[-1])  # 如 (13601, 2801)
    if is_logger:
        print(f"[Infer] 将粗分辨率输出插值到 GT 尺寸: {target_output_shape}")

    pred_full = interpolate_to_target(pred_coarse, target_output_shape)
    pred_full = output_inverse_transform(pred_full)

    # -------------------------
    # 保存结果与指标
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
        seis_t.detach().cpu(),   # 原始 Marmousi（已归一化）的波形
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

    print(f"推理完成，结果保存在 {results_dir}")


# ===============================
# CLI 入口
# ===============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Marmousi2 推理脚本（无分片版）")
    parser.add_argument("--seis_path", type=str,
                        default="./marmousi/marmousi_synthetic_seismic.npy",
                        help="地震波形 numpy 文件路径")
    parser.add_argument("--gt_path", type=str,
                        default="./marmousi/marmousi_Ip_model.npy",
                        help="速度/Ip ground truth 文件路径")
    parser.add_argument("--model_path", type=str, required=True,
                        help="MoE 模型权重路径（训练得到的 checkpoint）")
    parser.add_argument("--encoder_path", type=str, default=None,
                        help="Encoder 权重路径（如果训练时用了 encoder）")
    parser.add_argument("--output_dir", type=str, default="./marmousi_results",
                        help="输出目录")
    parser.add_argument("--k", type=float, default=1.0,
                        help="LogTransform 参数")
    parser.add_argument("--use_moe", action="store_true",
                        help="是否启用 MoE（保持与训练配置一致即可）")
    parser.add_argument("--use_encoder", action="store_true",
                        help="是否使用 encoder（保持与训练配置一致即可）")
    parser.add_argument("--setting_path", type=str, required=True,
                        help="训练时产生的目录（包含 args.json 和 config.json）")
    parser.add_argument("--use_experts_path", type=str, default="../other_experts",
                        help="专家权重目录（若 config 中需要）")

    args = parser.parse_args()
    infer_marmousi2(args)
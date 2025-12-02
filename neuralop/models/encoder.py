# -*- coding: utf-8 -*-
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ===== 可选依赖：safetensors =====
_HAS_SAFE = False
try:
    from safetensors.torch import load_file as safe_load_file
    _HAS_SAFE = True
except Exception:
    _HAS_SAFE = False

# ===== 可选：timm 的 pos_embed 插值工具（优先用官方） =====
_timm_resize_pos_embed = None
try:
    from timm.models.vision_transformer import resize_pos_embed as _timm_resize_pos_embed
except Exception:
    _timm_resize_pos_embed = None


# =========================
# 工具函数（通用）
# =========================

def _strip_prefix(sd: Dict[str, torch.Tensor],
                  prefixes=('module.', 'backbone.', 'model.', 'encoder.')) -> Dict[str, torch.Tensor]:
    """去除常见的 state_dict 键名前缀"""
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _drop_head(sd: Dict[str, torch.Tensor],
               drops=('head.', 'fc.', 'fc_norm.', 'classifier.')) -> Dict[str, torch.Tensor]:
    """去除分类头等下游层权重"""
    return {k: v for k, v in sd.items() if not any(k.startswith(d) for d in drops)}


def _naive_resize_pos_embed(src: torch.Tensor,
                            dst_shape: torch.Size,
                            num_tokens: int = 1) -> torch.Tensor:
    """
    兜底版 pos_embed 插值：src, dst 形状均为 [1, N, D]
    仅在 timm 官方函数不可用时调用。
    """
    assert src.dim() == 3 and dst_shape[0] == 1 and src.size(-1) == dst_shape[-1], \
        f"pos_embed shape mismatch: src={tuple(src.shape)}, dst={tuple(dst_shape)}"

    N_src = src.shape[1]
    N_dst = dst_shape[1]
    D = src.shape[2]

    if N_src == N_dst:
        return src

    # 将前 num_tokens（通常是 CLS）剥离，其余当作网格插值
    tok_src = src[:, :num_tokens] if num_tokens > 0 else None
    grid_src = src[:, num_tokens:] if num_tokens > 0 else src

    tok_dst = torch.empty((1, num_tokens, D), device=src.device, dtype=src.dtype) if num_tokens > 0 else None
    grid_dst_len = N_dst - num_tokens

    # 尝试从长度近似方形恢复网格
    gsrc = grid_src.shape[1]
    hsrc = int(round(gsrc ** 0.5))
    wsrc = gsrc // hsrc
    if hsrc * wsrc != gsrc:
        grid_dst = F.interpolate(
            grid_src.transpose(1, 2), size=grid_dst_len, mode='linear', align_corners=False
        ).transpose(1, 2)
    else:
        # 目标网格（根据长度开方）
        hdst = int(round(grid_dst_len ** 0.5))
        wdst = grid_dst_len // hdst
        if hdst * wdst != grid_dst_len:
            grid_dst = F.interpolate(
                grid_src.transpose(1, 2), size=grid_dst_len, mode='linear', align_corners=False
            ).transpose(1, 2)
        else:
            # [1, HW, D] -> [1, D, H, W]
            grid_src_2d = grid_src.view(1, hsrc, wsrc, D).permute(0, 3, 1, 2).contiguous()
            grid_dst_2d = F.interpolate(grid_src_2d, size=(hdst, wdst), mode='bilinear', align_corners=False)
            grid_dst = grid_dst_2d.permute(0, 2, 3, 1).contiguous().view(1, hdst * wdst, D)

    if num_tokens > 0:
        if tok_src is not None:
            tok_dst = tok_src[:, :num_tokens]
        out = torch.cat([tok_dst, grid_dst], dim=1)
    else:
        out = grid_dst
    return out


def resize_pos_embed(src: torch.Tensor, dst_like: torch.Tensor, num_tokens: int = 1) -> torch.Tensor:
    """
    统一入口：优先用 timm 官方的 resize_pos_embed，失败时用兜底版。
    """
    if _timm_resize_pos_embed is not None:
        try:
            return _timm_resize_pos_embed(src, dst_like, num_tokens=num_tokens)
        except Exception:
            pass
    return _naive_resize_pos_embed(src, dst_like.shape, num_tokens=num_tokens)


def _maybe_resize_pos_embed(model: nn.Module, sd: Dict[str, torch.Tensor]):
    """pos_embed 形状不匹配时进行插值（ViT 专用）"""
    if 'pos_embed' in sd and hasattr(model, 'pos_embed'):
        if sd['pos_embed'].shape != model.pos_embed.shape:
            sd['pos_embed'] = resize_pos_embed(sd['pos_embed'], model.pos_embed, num_tokens=1)


def _unfreeze_last_n_layers(module: nn.Module, n: int = 2) -> None:
    """允许微调 backbone 的最后 n 层（ViT 常见为 blocks 的最后几层与 norm）。"""
    if n <= 0:
        return
    blocks = getattr(module, "blocks", None)
    if isinstance(blocks, (nn.ModuleList, list)) and len(blocks) > 0:
        for block in blocks[-n:]:
            block.requires_grad_(True)
        if hasattr(module, "norm"):
            module.norm.requires_grad_(True)
        return
    children = list(module.children())
    if not children:
        return
    for child in children[-n:]:
        child.requires_grad_(True)


# =========================
# 轻量稳定层：通道 RMS 归一化（用于 ConvNeXt→MoE 对接）
# =========================

class ChannelRMSNorm2d(nn.Module):
    """
    对 [B, C, H, W] 做每通道 RMSNorm：y = x / sqrt(mean(x^2)+eps)
    不含仿射参数；可选 clamp 限幅，防止进入 MoE 时数值过大。
    """
    def __init__(self, eps: float = 1e-6, clamp_value: Optional[float] = None):
        super().__init__()
        self.eps = eps
        self.clamp_value = clamp_value

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        rms = (x.pow(2).mean(dim=(2, 3), keepdim=True) + self.eps).sqrt()
        y = x / rms
        if self.clamp_value is not None:
            y = torch.clamp(y, -self.clamp_value, self.clamp_value)
        return y


# =========================
# ViT（DINOv3）专用适配
# =========================

def _adapt_in_chans_vit(model: nn.Module, in_chans: int):
    """把 ViT 的 patch_embed 卷积输入通道适配为 in_chans（常见从 3→1 或 3→k）"""
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
        proj = model.patch_embed.proj
        if isinstance(proj, nn.Conv2d) and proj.in_channels != in_chans:
            new_proj = nn.Conv2d(in_chans, proj.out_channels, proj.kernel_size,
                                 proj.stride, proj.padding, bias=False)
            with torch.no_grad():
                w = proj.weight  # [out, Cin_old, kh, kw]
                if in_chans == 1:
                    w = w.mean(dim=1, keepdim=True)  # RGB → 灰度均值
                else:
                    w = w.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1)
                new_proj.weight.copy_(w)
            model.patch_embed.proj = new_proj


def remap_rgb_to_gray(state_dict: Dict[str, torch.Tensor],
                      key: str = "patch_embed.proj.weight",
                      in_chans: int = 1) -> Dict[str, torch.Tensor]:
    """将权重中的 RGB 卷积核 (3通道) 映射为灰度卷积核 (1通道) 或扩展到任意通道数"""
    if key not in state_dict:
        return state_dict
    w = state_dict[key]
    if not isinstance(w, torch.Tensor):
        w = torch.tensor(w)
    Cin = w.shape[1]
    if Cin == in_chans:
        return state_dict
    with torch.no_grad():
        if in_chans == 1:
            w_new = w.mean(dim=1, keepdim=True)
        else:
            w_new = w.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1)
    state_dict[key] = w_new
    return state_dict

def _load_local_weights_into_timm_vit(model_name: str, ckpt_path: str, in_chans: int) -> nn.Module:
    """
    加载本地 ViT-DINOv3 权重（支持 .safetensors/.pth），并自动做：
    - 去头/去前缀
    - RGB→灰度/多通道映射
    - pos_embed 插值
    - 某些 dinov3 qkv 变体 fallback（例如 vit_base_patch16_dinov3_qkvb）
    """
    assert os.path.isfile(ckpt_path), f"checkpoint not found: {ckpt_path}"

    if ckpt_path.endswith('.safetensors'):
        assert _HAS_SAFE, "需要 pip install safetensors"
        raw = safe_load_file(ckpt_path)  # Dict[str, Tensor]
    else:
        raw = torch.load(ckpt_path, map_location='cpu')

    sd = _drop_head(_strip_prefix(raw))
    sd = remap_rgb_to_gray(sd, key="patch_embed.proj.weight", in_chans=in_chans)

    candidates = [model_name]
    # 常见 qkvb 变体兜底
    if model_name.startswith('vit_base_patch16_dinov3'):
        candidates.append('vit_base_patch16_dinov3_qkvb')
    if model_name.startswith('vit_small_patch16_dinov3'):
        candidates.append('vit_small_patch16_dinov3_qkvb')

    last_err = None
    for arch in candidates:
        try:
            model = timm.create_model(
                arch,
                pretrained=False,
                num_classes=0,
                global_pool='',
                in_chans=in_chans,
            )
            _adapt_in_chans_vit(model, in_chans)
            tmp = dict(sd)
            _maybe_resize_pos_embed(model, tmp)
            missing, unexpected = model.load_state_dict(tmp, strict=False)
            print(f"[Encoder_Dino] loaded '{arch}' | missing={len(missing)}, unexpected={len(unexpected)}")
            return model
        except RuntimeError as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load local weights into {candidates}. Last error: {last_err}")


# =========================
# ConvNeXt（DINOv3）专用适配
# =========================

def _remap_convnext_stem_weight(sd: Dict[str, torch.Tensor], in_chans: int) -> Dict[str, torch.Tensor]:
    """
    将 state_dict 中 ConvNeXt 的 stem.0.weight 从 3 通道映射到 in_chans 通道：
      - in_chans == 1: 对输入通道做均值 -> [out,1,kh,kw]
      - in_chans  > 1: 先均值到 1 再 repeat 到 in_chans
    """
    key = "stem.0.weight"
    if key not in sd:
        return sd
    w = sd[key]  # [out, Cin, kh, kw]（通常 Cin=3）
    if not isinstance(w, torch.Tensor):
        w = torch.tensor(w)
    Cin = w.shape[1]
    if Cin == in_chans:
        return sd
    with torch.no_grad():
        w1 = w.mean(dim=1, keepdim=True)  # [out,1,kh,kw]
        if in_chans == 1:
            w_new = w1
        else:
            w_new = w1.repeat(1, in_chans, 1, 1)
    sd[key] = w_new
    return sd

def _adapt_stem_in_chans_convnext(model: nn.Module, in_chans: int):
    """
    运行时把 ConvNeXt 的 stem 卷积从 3 通道适配到 in_chans。
    """
    conv = None
    if hasattr(model, 'stem'):
        if isinstance(model.stem, nn.Sequential) and len(model.stem) > 0 and isinstance(model.stem[0], nn.Conv2d):
            conv = model.stem[0]
        elif isinstance(model.stem, nn.Conv2d):
            conv = model.stem
    if conv is None or conv.in_channels == in_chans:
        return

    new_conv = nn.Conv2d(in_chans, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, bias=False)
    with torch.no_grad():
        w = conv.weight  # [out, Cin_old, kh, kw]
        w1 = w.mean(dim=1, keepdim=True)
        if in_chans == 1:
            w_new = w1
        else:
            w_new = w1.repeat(1, in_chans, 1, 1)
        new_conv.weight.copy_(w_new)

    if isinstance(model.stem, nn.Sequential) and isinstance(model.stem[0], nn.Conv2d):
        model.stem[0] = new_conv
    elif isinstance(model.stem, nn.Conv2d):
        model.stem = new_conv


def _load_local_weights_into_timm_convnext(model_name: str, ckpt_path: str, in_chans: int) -> nn.Module:
    """
    加载本地 ConvNeXt-DINOv3 权重（.safetensors/.pth）并解决 in_chans 不匹配导致的 size mismatch。
    步骤：
      1) 读取 ckpt，去头/去前缀；
      2) 先对 sd 的 stem.0.weight 做 3→in_chans 通道映射；
      3) 构建模型(in_chans=in_chans) 并直接 load；
      4) 若仍异常，则退化为：in_chans=3 构建→load→再把 stem 改成 in_chans。
    """
    assert os.path.isfile(ckpt_path), f"checkpoint not found: {ckpt_path}"

    # 1) 读权重
    if ckpt_path.endswith('.safetensors'):
        assert _HAS_SAFE, "需要 pip install safetensors"
        raw = safe_load_file(ckpt_path)  # Dict[str, Tensor]
    else:
        raw = torch.load(ckpt_path, map_location='cpu')

    sd = _drop_head(_strip_prefix(raw))

    # 2) 预先把 stem.0.weight 映射到 in_chans，避免 size mismatch
    sd = _remap_convnext_stem_weight(sd, in_chans=in_chans)

    # 3) 直接按 in_chans 构建并加载
    try:
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool='',
            in_chans=in_chans,
        )
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[Encoder_ConvNeXt] loaded '{model_name}' (in_chans={in_chans}) | missing={len(missing)}, unexpected={len(unexpected)}")
        return model

    except RuntimeError as e:
        # 4) 兜底：先按 3 通道构建并加载，再把 stem 改到 in_chans
        print(f"[Encoder_ConvNeXt] fallback due to: {e}")
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool='',
            in_chans=3,
        )
        missing, unexpected = model.load_state_dict(_drop_head(_strip_prefix(raw)), strict=False)
        _adapt_stem_in_chans_convnext(model, in_chans=in_chans)
        print(f"[Encoder_ConvNeXt] loaded with fallback and adapted stem to in_chans={in_chans} | missing={len(missing)}, unexpected={len(unexpected)}")
        return model


# =========================
# 编码器实现（ViT / ConvNeXt）
# =========================

class Encoder_Dino(nn.Module):
    """
    ViT（DINOv3）编码器
    输入:  x [B, in_chans, H, W]（可非方形）
    输出:
      - latent_map:  [B, out_channels, 70, 70]
      - type_weight: [B, num_types]
      - global_repr: [B, D] (CLS + REG-mean)
    """
    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,  # 若提供本地权重则优先使用
        in_chans: int = 1,
        out_channels: int = 64,
        num_types: int = 3,
        img_size: tuple = (70,70),
        patch_size: int = 16,
        mode: str = "pad_then_interp",  # "pad_then_interp" | "resize1120"
        type_act: str = "softmax"       # "softmax" | "sigmoid" | "identity"
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_types = num_types
        self.target_h, self.target_w = img_size
        self.patch_size = patch_size
        self.mode = mode

        # 1) backbone
        if checkpoint_path:
            base_arch = model_name.split('.')[0] if '.lvd' in model_name else model_name
            self.backbone = _load_local_weights_into_timm_vit(base_arch, checkpoint_path, in_chans=in_chans)
        else:
            self.backbone = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,
                global_pool="",
            )
            _adapt_in_chans_vit(self.backbone, in_chans)

        self.embed_dim = getattr(self.backbone, "num_features", None) or getattr(self.backbone, "embed_dim", None)
        if self.embed_dim is None:
            raise RuntimeError("无法从 ViT backbone 获取 embedding 维度。")

        # 2) D→C 的 1×1 投影（把 token 网格投影到所需通道）
        self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1, bias=False)

        # 3) 类型预测头（基于 CLS+REG-mean 的全局表征）
        self.type_head = nn.Linear(self.embed_dim, num_types)
        if type_act == "softmax":
            self.type_act = nn.Softmax(dim=-1)
        elif type_act == "sigmoid":
            self.type_act = nn.Sigmoid()
        elif type_act == "identity":
            self.type_act = nn.Identity()
        else:
            raise ValueError(f"Unsupported type_act: {type_act}")
        
        self.cast_to_fp32_for_moe = False

    @torch.no_grad()
    def _preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        B, C, H, W = x.shape
        if self.mode == "resize1120":
            Ht = self.target_h * self.patch_size  # 70*16=1120
            Wt = self.target_w * self.patch_size
            x_resized = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
            return x_resized, self.target_h, self.target_w

        # pad 到 patch_size 的倍数（不拉伸）
        H_pad = (H + self.patch_size - 1) // self.patch_size * self.patch_size
        W_pad = (W + self.patch_size - 1) // self.patch_size * self.patch_size
        pad_h, pad_w = H_pad - H, W_pad - W
        x_padded = F.pad(x, pad=(0, pad_w, 0, pad_h), mode="reflect")
        Hg, Wg = H_pad // self.patch_size, W_pad // self.patch_size
        return x_padded, Hg, Wg

    def forward(self, x: torch.Tensor):
        B = x.size(0)

        # 1) 预处理到 ViT 几何
        x_proc, Hg, Wg = self._preprocess(x)

        # 2) 提取 tokens
        feats = self.backbone.forward_features(x_proc)

        # 兼容 timm 的不同返回形式
        if isinstance(feats, dict):
            tokens_full = feats.get("tokens", None)
            if tokens_full is not None:
                # [B, 1+num_reg+num_patch, D]
                cls_token = tokens_full[:, 0]       # [B, D]
                patch_tokens = tokens_full[:, 1:]   # 可能含 REG
            else:
                patch_tokens = feats.get("x_norm_patchtokens", None)
                if patch_tokens is None:
                    raise RuntimeError("未找到 patch tokens。")
                cls_token = feats.get("x_norm_clstoken", None)
                if cls_token is None:
                    raise RuntimeError("未找到 cls token。")
        else:
            tokens = feats                           # [B, 1+N', D]
            cls_token = tokens[:, 0]
            patch_tokens = tokens[:, 1:]

        # 2.1) 识别 & 切除 REG tokens（常见 dinov3/EVA 在末尾追加）
        N, D = patch_tokens.size(1), patch_tokens.size(2)
        HgWg = Hg * Wg

        n_reg = getattr(self.backbone, "num_register_tokens", 0)
        if n_reg == 0 and N > HgWg:
            extra = N - HgWg
            if 0 < extra <= 8:
                n_reg = extra

        reg_tokens = None
        if n_reg > 0 and N >= HgWg + n_reg:
            reg_tokens = patch_tokens[:, -n_reg:, :]   # [B, R, D]
            patch_tokens = patch_tokens[:, :HgWg, :]   # [B, HgWg, D]
            N = patch_tokens.size(1)

        # 2.2) CLS + REG-mean 融合（全局表征）
        if reg_tokens is not None:
            global_repr = torch.cat([cls_token.unsqueeze(1), reg_tokens], dim=1).mean(dim=1)  # [B, D]
        else:
            global_repr = cls_token  # 无 REG 时退化为 CLS-only

        # 3) 还原 patch 网格 → [B, D, Ht, Wt]
        if N != HgWg:
            W_try = int(round(N / Hg))
            if Hg * W_try == N:
                Hg_eff, Wg_eff = Hg, W_try
            else:
                Wg_eff = int(round(N ** 0.5))
                Hg_eff = N // Wg_eff
                if Hg_eff * Wg_eff != N:
                    raise RuntimeError(f"N={N} 不能 reshape 为网格 (Hg={Hg}, Wg={Wg}).")
        else:
            Hg_eff, Wg_eff = Hg, Wg

        feat_grid = patch_tokens.view(B, Hg_eff, Wg_eff, D).permute(0, 3, 1, 2).contiguous()  # [B, D, Ht, Wt]

        # 4) D→C，必要时插值到 70×70
        feat_proj = self.proj(feat_grid)  # [B, C, Ht, Wt]
        if (Hg_eff, Wg_eff) != (self.target_h, self.target_w):
            latent_map = F.interpolate(feat_proj, size=(self.target_h, self.target_w),
                                       mode="bilinear", align_corners=False)
        else:
            latent_map = feat_proj

        # 5) 全局表征 → 类型权重
        type_weight = self.type_act(self.type_head(global_repr))  # [B, num_types]
        return latent_map, type_weight, global_repr


class Encoder_ConvNeXt(nn.Module):
    """
    ConvNeXt（DINOv3）编码器（使用 timm 的 forward_head 得到分类 logits）
    输入:  x [B, in_chans, H, W]
    输出:
      - latent_map:  [B, out_channels, 70, 70]     # 由最后stage特征投影+插值得到
      - type_weight: [B, num_types]                # 由 timm.forward_head 计算出的 logits → 激活
      - global_repr: [B, D_last]                   # GAP 后的全局向量（pre_logits）
    """
    def __init__(
        self,
        model_name: str = "convnext_tiny.dinov3_lvd1689m",
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        in_chans: int = 1,
        out_channels: int = 64,
        num_types: int = 3,
        img_size: int = 70,
        type_act: str = "softmax",   # "softmax" | "sigmoid" | "identity"
        use_timm_head: bool = True,  # 关键：走 timm 的 forward_head 产出分类 logits
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_types = num_types
        self.target_h = self.target_w = int(img_size)
        self.use_timm_head = use_timm_head and (num_types is not None) and (num_types > 0)

        # 1) backbone：若要用 timm 的 head，就让 timm 知道 num_classes & global_pool
        if checkpoint_path:
            self.backbone = _load_local_weights_into_timm_convnext(
                model_name, checkpoint_path, in_chans
            )
        else:
            self.backbone = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=(num_types if self.use_timm_head else 0),
                global_pool=("avg" if self.use_timm_head else ""),
            )
            _adapt_stem_in_chans_convnext(self.backbone, in_chans)

        # 通道数（最后stage）
        self.embed_dim = getattr(self.backbone, "num_features", None) \
                         or getattr(self.backbone, "num_classes", None)
        if self.embed_dim is None:
            raise RuntimeError("无法从 ConvNeXt backbone 获取 embedding 维度。")

        # 2) D→C 投影给下游
        self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1, bias=False)

        # 3) 激活函数（timm 的 forward_head 给 logits，这里再变成权重）
        if type_act == "softmax":
            self.type_act = nn.Softmax(dim=-1)
        elif type_act == "sigmoid":
            self.type_act = nn.Sigmoid()
        elif type_act == "identity":
            self.type_act = nn.Identity()
        else:
            raise ValueError(f"Unsupported type_act: {type_act}")

        # ========== 新增：稳定对接 MoE 的关键模块 ==========
        # 对投影后的 [B,C,H,W] 做通道 RMS 归一化 + 限幅（避免频域/小波算子被大幅值击穿）
        self.post_norm = ChannelRMSNorm2d(eps=1e-5, clamp_value=5.0)
        # 路由温度缩放（降低/提升 logits 尖锐度；<1 更尖锐，>1 更平坦）
        self.router_temp = nn.Parameter(torch.tensor(0.7), requires_grad=False)
        # 在 AMP 下把进入 MoE 的 latent_map 铸为 fp32，避免 BF16/FP16 造成的路由/算子数值不稳
        self.cast_to_fp32_for_moe = True

        # 如果你不走 timm 头，也给一个无参 GAP 以便导出 global_repr
        self._gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor):
        # A) 最后 stage 特征（/32），形状 [B, C=embed_dim, H32, W32]
        feat = self.backbone.forward_features(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        assert feat.dim() == 4

        # B) 分类分支：走 timm 的 forward_head
        if self.use_timm_head:
            global_repr = self.backbone.forward_head(feat, pre_logits=True)     # [B, D]
            logits = self.backbone.forward_head(feat, pre_logits=False)         # [B, num_types]
            # === 温度缩放（数值稳定&可控尖锐度） ===
            temp = torch.clamp(self.router_temp, min=0.25, max=4.0)
            logits = logits / temp
            type_weight = self.type_act(logits)
        else:
            global_repr = self._gap(feat).flatten(1)                            # [B, D]
            type_weight = None

        # C) 特征 → latent_map（投影 + 归一化 + 插值到 70×70）
        feat_proj = self.proj(feat)                                # [B, out_channels, H32, W32]
        feat_proj = self.post_norm(feat_proj)                      # ★ 关键：RMSNorm + clamp
        # 可选：再乘一个缩放系数（遇到极端不稳时可打开）
        # feat_proj = feat_proj * 0.7

        if (feat_proj.shape[-2], feat_proj.shape[-1]) != (self.target_h, self.target_w):
            latent_map = F.interpolate(
                feat_proj, size=(self.target_h, self.target_w),
                mode="bilinear", align_corners=False
            )
        else:
            latent_map = feat_proj

        # === AMP 安全铸型：只把进 MoE 的 latent_map 转为 fp32 ===
        if self.cast_to_fp32_for_moe and latent_map.dtype != torch.float32:
            latent_map = latent_map.float()

        return latent_map, type_weight, global_repr


# =========================
# 统一工厂
# =========================

def get_encoder(
    in_channels: int = 1,
    out_channels: Optional[int] = None,
    num_types: int = 3,
    type_act: str = 'softmax',
    backbone: str = 'vit',   # 'vit' | 'convnext_tiny'
    img_size: tuple = (70,70),
) -> nn.Module:
    """
    统一工厂：
      - backbone='vit'            -> Encoder_Dino（ViT-S/16 默认）
      - backbone='convnext_tiny'  -> Encoder_ConvNeXt(tiny)
    """
    if out_channels is None:
        out_channels = 64

    if backbone == 'vit':
        model_name = 'vit_small_patch16_dinov3.lvd1689m'
        local_ckpt_path = "pretrain_weight/vit_small_patch16_dinov3.lvd1689m.safetensors"
        use_local = (local_ckpt_path is not None) and os.path.isfile(local_ckpt_path)
        enc = Encoder_Dino(
            model_name=model_name,
            pretrained=not use_local,
            checkpoint_path=local_ckpt_path if use_local else None,
            in_chans=in_channels,
            out_channels=out_channels,
            num_types=num_types,
            img_size=img_size,
            patch_size=16,
            mode="pad_then_interp",
            type_act=type_act,
        )
        # 需要微调时可解开：
        # _unfreeze_last_n_layers(enc.backbone, n=2)
        return enc

    elif backbone == 'convnext_tiny':
        model_name = 'convnext_tiny.dinov3_lvd1689m'
        local_ckpt_path = "pretrain_weight/convnext_tiny.dinov3_lvd1689m.safetensors"
        use_local = (local_ckpt_path is not None) and os.path.isfile(local_ckpt_path)
        enc = Encoder_ConvNeXt(
            model_name=model_name,
            pretrained=not use_local,
            checkpoint_path=local_ckpt_path if use_local else None,
            in_chans=in_channels,
            out_channels=out_channels,
            num_types=num_types,
            img_size=img_size,
            type_act=type_act,
            use_timm_head=True,
        )
        # 根据需要微调稳定参数
        enc.router_temp.data.fill_(0.7)   
        enc.post_norm.clamp_value = 5.0   
        enc.cast_to_fp32_for_moe = False
        # 如果想轻微微调最后 stage：
        # for p in enc.backbone.stages[-1].parameters(): p.requires_grad_(True)
        return enc
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
# -*- coding: utf-8 -*-
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ===== Optional: safetensors =====
_HAS_SAFE = False
try:
    from safetensors.torch import load_file as safe_load_file
    _HAS_SAFE = True
except Exception:
    _HAS_SAFE = False

# ===== Optional: timm resize_pos_embed (prefer official) =====
_timm_resize_pos_embed = None
try:
    from timm.models.vision_transformer import resize_pos_embed as _timm_resize_pos_embed
except Exception:
    _timm_resize_pos_embed = None


# =========================
# Shared helpers
# =========================

def _strip_prefix(sd: Dict[str, torch.Tensor],
                  prefixes=('module.', 'backbone.', 'model.', 'encoder.')) -> Dict[str, torch.Tensor]:
    """Strip common prefixes from state_dict keys."""
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
    """Drop classifier head and other downstream weights."""
    return {k: v for k, v in sd.items() if not any(k.startswith(d) for d in drops)}


def _naive_resize_pos_embed(src: torch.Tensor,
                            dst_shape: torch.Size,
                            num_tokens: int = 1) -> torch.Tensor:
    """
    Fallback pos_embed resize; src and dst shapes are [1, N, D].
    Used only when timm's helper is unavailable.
    """
    assert src.dim() == 3 and dst_shape[0] == 1 and src.size(-1) == dst_shape[-1], \
        f"pos_embed shape mismatch: src={tuple(src.shape)}, dst={tuple(dst_shape)}"

    N_src = src.shape[1]
    N_dst = dst_shape[1]
    D = src.shape[2]

    if N_src == N_dst:
        return src

    # Strip leading num_tokens (usually CLS); interpolate the grid part
    tok_src = src[:, :num_tokens] if num_tokens > 0 else None
    grid_src = src[:, num_tokens:] if num_tokens > 0 else src

    tok_dst = torch.empty((1, num_tokens, D), device=src.device, dtype=src.dtype) if num_tokens > 0 else None
    grid_dst_len = N_dst - num_tokens

    # Recover a near-square grid from token count
    gsrc = grid_src.shape[1]
    hsrc = int(round(gsrc ** 0.5))
    wsrc = gsrc // hsrc
    if hsrc * wsrc != gsrc:
        grid_dst = F.interpolate(
            grid_src.transpose(1, 2), size=grid_dst_len, mode='linear', align_corners=False
        ).transpose(1, 2)
    else:
        # Target grid from sqrt of length
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
    Prefer timm resize_pos_embed; fall back to _naive_resize_pos_embed.
    """
    if _timm_resize_pos_embed is not None:
        try:
            return _timm_resize_pos_embed(src, dst_like, num_tokens=num_tokens)
        except Exception:
            pass
    return _naive_resize_pos_embed(src, dst_like.shape, num_tokens=num_tokens)


def _maybe_resize_pos_embed(model: nn.Module, sd: Dict[str, torch.Tensor]):
    """Interpolate pos_embed when shapes disagree (ViT)."""
    if 'pos_embed' in sd and hasattr(model, 'pos_embed'):
        if sd['pos_embed'].shape != model.pos_embed.shape:
            sd['pos_embed'] = resize_pos_embed(sd['pos_embed'], model.pos_embed, num_tokens=1)


def _unfreeze_last_n_layers(module: nn.Module, n: int = 2) -> None:
    """Unfreeze the last n backbone blocks (and norm), typical ViT fine-tuning."""
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
# Light stability: channel RMS norm (ConvNeXt -> MoE)
# =========================

class ChannelRMSNorm2d(nn.Module):
    """
    Per-channel RMSNorm on [B, C, H, W]: y = x / sqrt(mean(x^2)+eps).
    No affine; optional clamp before MoE for stability.
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
# ViT (DINOv3) adapters
# =========================

def _adapt_in_chans_vit(model: nn.Module, in_chans: int):
    """Adapt ViT patch_embed conv input channels to in_chans (e.g. 3 -> 1 or 3 -> k)."""
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
        proj = model.patch_embed.proj
        if isinstance(proj, nn.Conv2d) and proj.in_channels != in_chans:
            new_proj = nn.Conv2d(in_chans, proj.out_channels, proj.kernel_size,
                                 proj.stride, proj.padding, bias=False)
            with torch.no_grad():
                w = proj.weight  # [out, Cin_old, kh, kw]
                if in_chans == 1:
                    w = w.mean(dim=1, keepdim=True)  # RGB to grayscale mean
                else:
                    w = w.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1)
                new_proj.weight.copy_(w)
            model.patch_embed.proj = new_proj


def remap_rgb_to_gray(state_dict: Dict[str, torch.Tensor],
                      key: str = "patch_embed.proj.weight",
                      in_chans: int = 1) -> Dict[str, torch.Tensor]:
    """Map RGB conv weights (3 ch) to grayscale (1 ch) or repeat to in_chans."""
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
    Load local ViT-DINOv3 weights (.safetensors/.pth) with:
    - head/prefix stripping
    - RGB to gray / multi-channel remap
    - pos_embed resize fallback
    - dinov3 qkv variants (e.g. vit_base_patch16_dinov3_qkvb)
    """
    assert os.path.isfile(ckpt_path), f"checkpoint not found: {ckpt_path}"

    if ckpt_path.endswith('.safetensors'):
        assert _HAS_SAFE, "pip install safetensors is required"
        raw = safe_load_file(ckpt_path)  # Dict[str, Tensor]
    else:
        raw = torch.load(ckpt_path, map_location='cpu')

    sd = _drop_head(_strip_prefix(raw))
    sd = remap_rgb_to_gray(sd, key="patch_embed.proj.weight", in_chans=in_chans)

    candidates = [model_name]
    # Common qkvb variant fallback
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
# ConvNeXt (DINOv3) adapters
# =========================

def _remap_convnext_stem_weight(sd: Dict[str, torch.Tensor], in_chans: int) -> Dict[str, torch.Tensor]:
    """
    Map ConvNeXt stem.0.weight in state_dict from 3 channels to in_chans:
      - in_chans == 1: channel mean -> [out,1,kh,kw]
      - in_chans  > 1: mean to 1 then repeat -> in_chans
    """
    key = "stem.0.weight"
    if key not in sd:
        return sd
    w = sd[key]  # [out, Cin, kh, kw] (typically Cin=3)
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
    """At runtime adapt ConvNeXt stem conv from 3 channels to in_chans."""
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
    Load local ConvNeXt-DINOv3 weights; fix in_chans mismatches (size mismatch).
    Steps:
      1) read ckpt, strip head/prefix
      2) remap stem.0.weight 3 -> in_chans
      3) build with in_chans and load
      4) on failure: build in_chans=3, load, then adapt stem to in_chans
    """
    assert os.path.isfile(ckpt_path), f"checkpoint not found: {ckpt_path}"

    # 1) load checkpoint
    if ckpt_path.endswith('.safetensors'):
        assert _HAS_SAFE, "pip install safetensors is required"
        raw = safe_load_file(ckpt_path)  # Dict[str, Tensor]
    else:
        raw = torch.load(ckpt_path, map_location='cpu')

    sd = _drop_head(_strip_prefix(raw))

    # 2) remap stem.0.weight early to avoid size mismatch
    sd = _remap_convnext_stem_weight(sd, in_chans=in_chans)

    # 3) build with in_chans and load
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
        # 4) fallback: build with 3 ch, load, adapt stem to in_chans
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
# Encoders (ViT / ConvNeXt)
# =========================

class Encoder_Dino(nn.Module):
    """
    ViT (DINOv3) encoder.
    Inputs:  x [B, in_chans, H, W] (H/W need not match)
    Outputs:
      - latent_map:  [B, out_channels, 70, 70]
      - type_weight: [B, num_types]
      - global_repr: [B, D] (CLS + REG-mean)
    """
    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,  # if set, load local weights first
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
            raise RuntimeError("Cannot infer embedding dim from ViT backbone.")

        # 2) 1x1 D->C projection (token grid to target channels)
        self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1, bias=False)

        # 3) type head on global repr (CLS + REG mean)
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

        # pad to multiple of patch_size (no stretch)
        H_pad = (H + self.patch_size - 1) // self.patch_size * self.patch_size
        W_pad = (W + self.patch_size - 1) // self.patch_size * self.patch_size
        pad_h, pad_w = H_pad - H, W_pad - W
        x_padded = F.pad(x, pad=(0, pad_w, 0, pad_h), mode="reflect")
        Hg, Wg = H_pad // self.patch_size, W_pad // self.patch_size
        return x_padded, Hg, Wg

    def forward(self, x: torch.Tensor):
        B = x.size(0)

        # 1) preprocess for ViT geometry
        x_proc, Hg, Wg = self._preprocess(x)

        # 2) extract tokens
        feats = self.backbone.forward_features(x_proc)

        # timm may return dict or tensor
        if isinstance(feats, dict):
            tokens_full = feats.get("tokens", None)
            if tokens_full is not None:
                # [B, 1+num_reg+num_patch, D]
                cls_token = tokens_full[:, 0]       # [B, D]
                patch_tokens = tokens_full[:, 1:]   # may include REG tokens
            else:
                patch_tokens = feats.get("x_norm_patchtokens", None)
                if patch_tokens is None:
                    raise RuntimeError("patch tokens not found.")
                cls_token = feats.get("x_norm_clstoken", None)
                if cls_token is None:
                    raise RuntimeError("cls token not found.")
        else:
            tokens = feats                           # [B, 1+N', D]
            cls_token = tokens[:, 0]
            patch_tokens = tokens[:, 1:]

        # 2.1) detect and strip REG tokens (often appended in dinov3/EVA)
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

        # 2.2) CLS + mean(REG) global representation
        if reg_tokens is not None:
            global_repr = torch.cat([cls_token.unsqueeze(1), reg_tokens], dim=1).mean(dim=1)  # [B, D]
        else:
            global_repr = cls_token  # no REG: CLS-only

        # 3) patch grid -> [B, D, Ht, Wt]
        if N != HgWg:
            W_try = int(round(N / Hg))
            if Hg * W_try == N:
                Hg_eff, Wg_eff = Hg, W_try
            else:
                Wg_eff = int(round(N ** 0.5))
                Hg_eff = N // Wg_eff
                if Hg_eff * Wg_eff != N:
                    raise RuntimeError(f"N={N} cannot reshape to grid (Hg={Hg}, Wg={Wg}).")
        else:
            Hg_eff, Wg_eff = Hg, Wg

        feat_grid = patch_tokens.view(B, Hg_eff, Wg_eff, D).permute(0, 3, 1, 2).contiguous()  # [B, D, Ht, Wt]

        # 4) D->C; interpolate to 70x70 if needed
        feat_proj = self.proj(feat_grid)  # [B, C, Ht, Wt]
        if (Hg_eff, Wg_eff) != (self.target_h, self.target_w):
            latent_map = F.interpolate(feat_proj, size=(self.target_h, self.target_w),
                                       mode="bilinear", align_corners=False)
        else:
            latent_map = feat_proj

        # 5) global repr -> type weights
        type_weight = self.type_act(self.type_head(global_repr))  # [B, num_types]
        return latent_map, type_weight, global_repr


class Encoder_ConvNeXt(nn.Module):
    """
    ConvNeXt (DINOv3) encoder; uses timm forward_head for classification logits.
    Input:  x [B, in_chans, H, W]
    Outputs:
      - latent_map:  [B, out_channels, 70, 70]     # last-stage features, proj + interp
      - type_weight: [B, num_types]                # logits from timm.forward_head -> activation
      - global_repr: [B, D_last]                   # GAP / pre_logits vector
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
        use_timm_head: bool = True,  # use timm forward_head for class logits
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_types = num_types
        self.target_h = self.target_w = int(img_size)
        self.use_timm_head = use_timm_head and (num_types is not None) and (num_types > 0)

        # 1) backbone: for timm head, pass num_classes & global_pool
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

        # channels at last stage
        self.embed_dim = getattr(self.backbone, "num_features", None) \
                         or getattr(self.backbone, "num_classes", None)
        if self.embed_dim is None:
            raise RuntimeError("Cannot infer embedding dim from ConvNeXt backbone.")

        # 2) D->C projection downstream
        self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1, bias=False)

        # 3) activation on logits from timm head -> weights
        if type_act == "softmax":
            self.type_act = nn.Softmax(dim=-1)
        elif type_act == "sigmoid":
            self.type_act = nn.Sigmoid()
        elif type_act == "identity":
            self.type_act = nn.Identity()
        else:
            raise ValueError(f"Unsupported type_act: {type_act}")

        # MoE stability: channel RMS + clamp after projection
        self.post_norm = ChannelRMSNorm2d(eps=1e-5, clamp_value=5.0)
        # router temperature on logits (<1 sharper, >1 flatter)
        self.router_temp = nn.Parameter(torch.tensor(0.7), requires_grad=False)
        # cast latent_map to fp32 before MoE under AMP (BF16/FP16 stability)
        self.cast_to_fp32_for_moe = True

        # parameter-free GAP for global_repr when timm head is off
        self._gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor):
        # A) last stage (/32), [B, C=embed_dim, H32, W32]
        feat = self.backbone.forward_features(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        assert feat.dim() == 4

        # B) classification via timm forward_head
        if self.use_timm_head:
            global_repr = self.backbone.forward_head(feat, pre_logits=True)     # [B, D]
            logits = self.backbone.forward_head(feat, pre_logits=False)         # [B, num_types]
            # temperature scaling for stability / sharpness
            temp = torch.clamp(self.router_temp, min=0.25, max=4.0)
            logits = logits / temp
            type_weight = self.type_act(logits)
        else:
            global_repr = self._gap(feat).flatten(1)                            # [B, D]
            type_weight = None

        # C) latent_map: project, normalize, interp to target size
        feat_proj = self.proj(feat)                                # [B, out_channels, H32, W32]
        feat_proj = self.post_norm(feat_proj)                      # Key: RMSNorm + clamp
        # Optional: extra scale if still unstable
        # feat_proj = feat_proj * 0.7

        if (feat_proj.shape[-2], feat_proj.shape[-1]) != (self.target_h, self.target_w):
            latent_map = F.interpolate(
                feat_proj, size=(self.target_h, self.target_w),
                mode="bilinear", align_corners=False
            )
        else:
            latent_map = feat_proj

        # AMP: cast latent_map to fp32 before MoE only
        if self.cast_to_fp32_for_moe and latent_map.dtype != torch.float32:
            latent_map = latent_map.float()

        return latent_map, type_weight, global_repr


# =========================
# Factory
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
    Factory:
      - backbone='vit'            -> Encoder_Dino (ViT-S/16 default)
      - backbone='convnext_tiny'  -> Encoder_ConvNeXt (tiny)
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
        # Uncomment to fine-tune last blocks:
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
        # Tune stability hyperparameters if needed
        enc.router_temp.data.fill_(0.7)   
        enc.post_norm.clamp_value = 5.0   
        enc.cast_to_fp32_for_moe = False
        # Optional: lightly unfreeze last stage
        # for p in enc.backbone.stages[-1].parameters(): p.requires_grad_(True)
        return enc
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
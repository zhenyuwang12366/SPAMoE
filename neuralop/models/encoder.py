import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Any, Union
from timm.models.vision_transformer import resize_pos_embed

try:
    from safetensors.torch import load_file as load_safetensors
    _HAS_SAFE = True
except Exception:
    _HAS_SAFE = False
    
class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int]):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Encoder_CNN(nn.Module):
    """
    CNN encoder that compresses raw seismic gathers into a fixed 70×70 spatial feature grid.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        out_channels: int | None = None,
        target_size: tuple[int, int] = (70, 70),
    ) -> None:
        super().__init__()
        hidden_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 4]
        strides = [(2, 2), (2, 2), (2, 1), (2, 1)]
        layers = []
        channels = [in_channels, *hidden_channels]
        for idx, stride in enumerate(strides):
            layers.append(_ConvBlock(channels[idx], channels[idx + 1], stride=stride))
        self.backbone = nn.Sequential(*layers)
        self.hidden_out_channels = hidden_channels[-1]
        self.out_channels = out_channels if out_channels is not None else self.hidden_out_channels
        if self.out_channels != self.hidden_out_channels:
            self.projector = nn.Sequential(
                nn.Conv2d(self.hidden_out_channels, self.hidden_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.hidden_out_channels),
                nn.GELU(),
                nn.Conv2d(self.hidden_out_channels, self.out_channels, kernel_size=1, bias=True),
            )
        else:
            self.projector = nn.Identity()
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got shape {tuple(x.shape)}")
        x = self.backbone(x)
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        x = self.projector(x)
        return x

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


def _maybe_resize_pos_embed(model: nn.Module, sd: Dict[str, torch.Tensor]):
    """pos_embed 形状不匹配时进行插值"""
    if 'pos_embed' in sd and hasattr(model, 'pos_embed'):
        if sd['pos_embed'].shape != model.pos_embed.shape:
            sd['pos_embed'] = resize_pos_embed(sd['pos_embed'], model.pos_embed, num_tokens=1)


def _adapt_in_chans(model: nn.Module, in_chans: int):
    """把 patch_embed 卷积输入通道适配为 in_chans（常见从3→1）"""
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
        proj = model.patch_embed.proj
        if proj.in_channels != in_chans:
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


def _unfreeze_last_n_layers(module: nn.Module, n: int = 2) -> None:
    """允许微调 backbone 的最后 n 层，默认取 transformer blocks 最后两层。"""
    if n <= 0:
        return

    blocks = getattr(module, "blocks", None)
    if isinstance(blocks, (nn.ModuleList, list)) and len(blocks) > 0:
        for block in blocks[-n:]:
            block.requires_grad_(True)
        # 常规 ViT 末尾还有一个 LayerNorm，需要一起解冻
        if hasattr(module, "norm"):
            module.norm.requires_grad_(True)
        return

    children = list(module.children())
    if not children:
        return
    for child in children[-n:]:
        child.requires_grad_(True)


def remap_rgb_to_gray(state_dict: Dict[str, torch.Tensor],
                      key: str = "patch_embed.proj.weight",
                      in_chans: int = 1) -> Dict[str, torch.Tensor]:
    """将权重中的 RGB 卷积核 (3通道) 映射为灰度卷积核 (1通道) 或扩展到任意通道数"""
    if key not in state_dict:
        return state_dict

    w = state_dict[key]  # [out, Cin, kh, kw]
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


def _load_local_weights_into_timm(model_name: str, ckpt_path: str, in_chans: int) -> nn.Module:
    """
    加载本地权重（支持 .safetensors/.pth），并自动做：
    - 去头/去前缀
    - RGB→灰度均值映射（3→1）
    - pos_embed 插值
    - dinov3 qkv 变体 fallback
    """
    assert os.path.isfile(ckpt_path), f"checkpoint not found: {ckpt_path}"

    if ckpt_path.endswith('.safetensors'):
        assert _HAS_SAFE, "需要 pip install safetensors"
        raw = load_safetensors(ckpt_path, device='cpu')
    else:
        raw = torch.load(ckpt_path, map_location='cpu')

    sd = _drop_head(_strip_prefix(raw))
    sd = remap_rgb_to_gray(sd, key="patch_embed.proj.weight", in_chans=in_chans)

    candidates = [model_name]
    if model_name.startswith('vit_base_patch16_dinov3'):
        candidates.append('vit_base_patch16_dinov3_qkvb')

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
            _adapt_in_chans(model, in_chans)
            tmp = dict(sd)
            _maybe_resize_pos_embed(model, tmp)
            missing, unexpected = model.load_state_dict(tmp, strict=False)
            print(f"[Encoder_Dino] loaded '{arch}' | missing={len(missing)}, unexpected={len(unexpected)}")
            return model
        except RuntimeError as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load local weights into {candidates}. Last error: {last_err}")

class Encoder_Dino(nn.Module):
    """
    输入:  x [B, in_chans, H=1000, W=350]（或任意尺寸）
    输出:
      - latent_map:  [B, out_channels, 70, 70]
      - type_weight: [B, num_types]
      - global_repr: [B, D] (CLS + REG-mean)
    """
    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        checkpoint_path: str | None = None,  # 若提供本地权重则优先使用
        in_chans: int = 1,
        out_channels: int = 64,
        num_types: int = 3,
        img_size: int = 70,
        patch_size: int = 16,
        mode: str = "pad_then_interp",  # "pad_then_interp" | "resize1120"
        type_act: str = "softmax"       # "softmax" | "sigmoid" | "identity"
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_types = num_types
        self.target_h = self.target_w = int(img_size)
        self.patch_size = patch_size
        self.mode = mode

        # 1) backbone
        if checkpoint_path:
            base_arch = model_name.split('.')[0] if '.lvd' in model_name else model_name
            self.backbone = _load_local_weights_into_timm(base_arch, checkpoint_path, in_chans=in_chans)
        else:
            self.backbone = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,
                global_pool="",
            )
            _adapt_in_chans(self.backbone, in_chans)
        # Freeze backbone parameters to exclude them from training updates
        # self.backbone.requires_grad_(False)
        # # 解冻最后两层用于微调
        # _unfreeze_last_n_layers(self.backbone, n=2)

        self.embed_dim = getattr(self.backbone, "num_features", None) or getattr(self.backbone, "embed_dim", None)
        if self.embed_dim is None:
            raise RuntimeError("无法从 backbone 获取 embedding 维度。")

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

        # 优先用模型属性；否则用 N-HgWg 推断
        n_reg = getattr(self.backbone, "num_register_tokens", 0)
        if n_reg == 0 and N > HgWg:
            extra = N - HgWg
            if 0 < extra <= 8:
                n_reg = extra

        reg_tokens = None
        if n_reg > 0 and N >= HgWg + n_reg:
            # 若你的实现把 REG 放在前面，改成 [:, :n_reg, :] 和 [:, n_reg:n_reg+HgWg, :]
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
            # 兜底（几乎不会触发）：尝试近似网格
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

def get_encoder(in_channels: int = 1, out_channels: int | None = None, num_types: int = 3) -> Encoder_Dino:
    """
    供训练脚本调用的工厂方法
    """
    if out_channels is None:
        out_channels = 64

    # ★ 如果你已经把权重下载到本地，请把路径改成你实际的文件：
    local_ckpt = "/data1/home/teacher/teacher_s/t108790/weights/vit_small_patch16_dinov3.lvd1689m.safetensors"
    use_local = os.path.isfile(local_ckpt)

    model = Encoder_Dino(
        model_name='vit_small_patch16_dinov3.lvd1689m',
        pretrained=not use_local,                 # 有本地就不用在线预训练
        checkpoint_path=local_ckpt if use_local else None,
        in_chans=in_channels,
        out_channels=out_channels,
        num_types=num_types,
        img_size=70,
        patch_size=16,
        mode="pad_then_interp",
        type_act="softmax",
    )
    return model

# -*- coding: utf-8 -*-
"""
WNO2d (DWT/DTCWT) + WNO3d (DWT) 一体化实现
- 与 wavelet_conv.py 的 WaveConv2d / WaveConv2dCwt / WaveConv3d 完全对口
- 小波路径禁用 AMP（fp16/bf16 下更稳），其余层可使用 AMP
- 每层结构：x ← K(x) + W(x)；最后一层不激活；Channel-MLP 走残差
- 2^L 对齐的样本级 padding（右/下/后补齐），退出时精确裁剪
"""

import contextlib
from typing import Tuple, List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.cuda.amp as amp
except ImportError:
    amp = None

Number = Union[float, int]

# === 你的工程内模块 ===
from ..layers.embeddings import GridEmbeddingND
from ..layers.wavelet_conv import WaveConv2d, WaveConv2dCwt, WaveConv3d
from ..layers.padding import DomainPadding
from ..layers.channel_mlp import ChannelMLP
from .base_model import BaseModel


# -----------------------------
# 工具: 固定到 2^L 倍数的 padding + 精确裁剪
# -----------------------------
def _ceil_to(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m

def pad_to_pow2_hw(x: torch.Tensor, levels: Tuple[int, ...], mode: str) -> Tuple[torch.Tensor, dict]:
    """
    2D: 只对 H,W 进行右/下补齐，使其能被 2^L 整除；返回补齐后的张量与 meta。
    x: [B,C,H,W]
    """
    assert x.dim() == 4, f"pad_to_pow2_hw 仅支持 4D 张量，收到 {x.shape}"
    H, W = x.shape[-2:]
    multiple = 2 ** max(levels)
    Ht, Wt = _ceil_to(H, multiple), _ceil_to(W, multiple)
    pad = (0, Wt - W, 0, Ht - H)   # (left,right,top,bottom) -> 只补右/下
    x_pad = F.pad(x, pad, mode=mode) if (pad[1] or pad[3]) else x
    meta = {"orig_hw": (H, W)}
    return x_pad, meta

def unpad_from_meta_hw(x: torch.Tensor, meta: dict) -> torch.Tensor:
    H, W = meta["orig_hw"]
    return x[..., :H, :W]

def pad_to_pow2_dhw(x: torch.Tensor, levels: Tuple[int, ...], mode: str = "symmetric"):
    """
    3D: 只对 D,H,W 进行右/后/下补齐，使其能被 2^L 整除；返回补齐张量与 meta。
    x: [B,C,D,H,W]
    """
    assert x.dim() == 5, f"pad_to_pow2_dhw 仅支持 5D 张量，收到 {x.shape}"
    D, H, W = x.shape[-3:]
    multiple = 2 ** max(levels)
    Dt, Ht, Wt = _ceil_to(D, multiple), _ceil_to(H, multiple), _ceil_to(W, multiple)
    # pad 顺序: (W_left, W_right, H_top, H_bottom, D_front, D_back)
    pad = (0, Wt - W, 0, Ht - H, 0, Dt - D)   # 只补右/下/后
    x_pad = F.pad(x, pad, mode=mode) if (pad[1] or pad[3] or pad[5]) else x
    meta = {"orig_dhw": (D, H, W)}
    return x_pad, meta

def unpad_from_meta_dhw(x: torch.Tensor, meta: dict) -> torch.Tensor:
    D, H, W = meta["orig_dhw"]
    return x[..., :D, :H, :W]


# -----------------------------
# 归一化工厂: 小 batch 稳定优先 -> GroupNorm
# -----------------------------
def make_norm(num_channels: int, n_dim: int, use_group_norm: bool = True, num_groups: int = 32):
    if use_group_norm:
        return nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels)
    else:
        if n_dim == 1:
            return nn.BatchNorm1d(num_channels)
        elif n_dim == 2:
            return nn.BatchNorm2d(num_channels)
        else:
            return nn.BatchNorm3d(num_channels)


# -----------------------------
# 2D: WNOBlock（DWT / DTCWT 按 conv_kind 切换）
# -----------------------------
class WNOBlock2d(nn.Module):
    """
    2D 小波神经算子块:
      - 每层: x ← K(x) + W(x)；最后一层不激活；Channel-MLP 残差
      - 仅小波路径禁用 AMP（fp32）
      - 进入/离开块时，样本级右下 pad 到 2^L 倍数并精确裁剪
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_levels: Tuple[int, int],           # 例如 (2,2)
        base_size: Tuple[int, int],          # 训练主分辨率 (H,W)
        conv_kind: str = "dwt",              # "dwt" | "dtcwt"
        # DWT 参数
        wavelet: str = 'db6',
        dwt_mode: str = 'symmetric',         # 官方 DWT 默认
        # DTCWT 参数
        biort: Optional[str] = None,
        qshift: Optional[str] = None,
        # 堆叠与非线性
        n_layers: int = 1,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0.0,
        channel_mlp_expansion: float = 0.5,
        non_linearity=F.gelu,
    ):
        super().__init__()
        assert conv_kind in ("dwt", "dtcwt"), f"conv_kind 必须是 'dwt' 或 'dtcwt'，收到 {conv_kind}"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_levels = n_levels
        self.level = max(n_levels)          # 小波分解级数
        self.conv_kind = conv_kind
        self.wavelet = wavelet
        self.dwt_mode = dwt_mode
        self.biort = biort
        self.qshift = qshift
        self.n_layers = n_layers
        self.non_linearity = non_linearity
        self.base_size = list(base_size)    # WaveConv2d/WaveConv2dCwt 需要 list

        # 小波卷积 K
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            if self.conv_kind == "dwt":
                self.convs.append(
                    WaveConv2d(
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        level=self.level,
                        size=self.base_size,
                        wavelet=self.wavelet,
                        mode=self.dwt_mode
                    )
                )
            else:
                assert self.biort is not None and self.qshift is not None, \
                    "使用 DTCWT 时必须提供 biort 和 qshift"
                self.convs.append(
                    WaveConv2dCwt(
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        level=self.level,
                        size=self.base_size,
                        wavelet1=self.biort,
                        wavelet2=self.qshift
                    )
                )

        # 1×1 像素域 W
        self.w_local = nn.ModuleList([
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=True)
            for _ in range(n_layers)
        ])

        # 通道 MLP（残差）
        self.use_channel_mlp = use_channel_mlp
        if use_channel_mlp:
            self.channel_mlps = nn.ModuleList([
                ChannelMLP(
                    in_channels=self.out_channels,
                    hidden_channels=max(1, int(round(self.out_channels * channel_mlp_expansion))),
                    dropout=channel_mlp_dropout,
                    n_dim=2,
                ) for _ in range(n_layers)
            ])
            self.channel_mlp_skips = nn.ModuleList([
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=True)
                for _ in range(n_layers)
            ])
        else:
            self.channel_mlps = None
            self.channel_mlp_skips = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 仅小波路径禁用 AMP，保证 DWT/IDWT / DTCWT 数值稳定
        amp_ctx = torch.amp.autocast(enabled=False) if torch.cuda.is_available() else contextlib.nullcontext()
        with amp_ctx:
            pad_mode = 'symmetric' if self.conv_kind == 'dwt' else 'reflect'
            x, meta = pad_to_pow2_hw(x, self.n_levels, mode=pad_mode)

            for i in range(self.n_layers):
                kx = self.convs[i](x)            # K(x)
                wx = self.w_local[i](x)          # W(x)
                x = kx + wx

                if i != self.n_layers - 1:
                    x = self.non_linearity(x)

                if self.use_channel_mlp:
                    mlp_out = self.channel_mlps[i](x)
                    skip = self.channel_mlp_skips[i](x)
                    x = self.non_linearity(mlp_out + skip)

            x = unpad_from_meta_hw(x, meta)
        return x


# -----------------------------
# 2D: 顶层 WNO2d（DWT / DTCWT）
# -----------------------------
class WNO2d(BaseModel):
    """
    2D 小波神经算子:
      - conv_kind='dwt' -> 调 WaveConv2d（DWT）
      - conv_kind='dtcwt' -> 调 WaveConv2dCwt（DTCWT）
      输入: [B, C_in, H, W] （若使用 grid 位置嵌入，会在通道维追加 2 个坐标通道）
      输出: [B, C_out, H, W]
    """
    def __init__(
        self,
        n_levels_height: int,
        n_levels_width: int,
        hidden_channels: int,
        base_size: Tuple[int, int],              # (H,W) 训练主分辨率
        in_channels: int = 3,
        out_channels: int = 1,
        conv_kind: str = 'dwt',                  # 'dwt' | 'dtcwt'
        # DWT 参数
        wavelet: str = 'db6',
        dwt_mode: str = 'symmetric',
        # DTCWT 参数
        biort: Optional[str] = None,
        qshift: Optional[str] = None,
        # 结构
        n_layers: int = 4,
        positional_embedding: Union[str, nn.Module] = "grid",
        domain_padding: Union[Number, List[Number], None] = None,
        domain_padding_mode: str = "symmetric",
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0.0,
        channel_mlp_expansion: float = 0.5,
        non_linearity = F.gelu,
        use_amp: bool = True,
        use_group_norm: bool = True,
        dropout_rate: float = 0.10,
        lifting_channels: Optional[int] = None,
        projection_channels: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        assert conv_kind in ("dwt", "dtcwt")
        self.n_levels = (n_levels_height, n_levels_width)
        self.level = max(self.n_levels)
        self.conv_kind = conv_kind
        self.wavelet = wavelet
        self.dwt_mode = dwt_mode
        self.biort = biort
        self.qshift = qshift

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.use_amp = bool(use_amp and torch.cuda.is_available())
        self.dropout_rate = float(dropout_rate)
        self.base_size = base_size

        # 位置嵌入（坐标归一化到 [0,1]）
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0., 1.]] * 2
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=2,
                grid_boundaries=spatial_grid_boundaries
            )
            lifted_in_channels = self.in_channels + 2
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
            lifted_in_channels = self.in_channels + 2
        elif positional_embedding is None:
            self.positional_embedding = None
            lifted_in_channels = self.in_channels
        else:
            raise ValueError(f"无效的位置嵌入类型: {positional_embedding}")

        # 物理域 padding（可选）
        self.domain_padding = DomainPadding(domain_padding, domain_padding_mode) if domain_padding is not None else None

        # 提升
        self.lifting = ChannelMLP(
            in_channels=lifted_in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=(2 * hidden_channels if (lifting_channels is None) else lifting_channels),
            n_layers=2,
            n_dim=2
        )
        self.lifting_norm = make_norm(self.hidden_channels, 2, use_group_norm)
        self.lifting_drop = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        # 主干（K+W）
        self.wno_blocks = WNOBlock2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            n_levels=self.n_levels,
            base_size=self.base_size,
            conv_kind=self.conv_kind,
            wavelet=self.wavelet,
            dwt_mode=self.dwt_mode,
            biort=self.biort,
            qshift=self.qshift,
            n_layers=self.n_layers,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
        )

        # 投影
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=(2 * hidden_channels if (projection_channels is None) else projection_channels),
            n_layers=2,
            n_dim=2
        )
        self.projection_norm = make_norm(self.out_channels, 2, use_group_norm)
        self.projection_drop = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

    # 外层可用 AMP；小波路径内部已禁用
    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int, ...]] = None, **kwargs) -> torch.Tensor:
        if self.use_amp and amp is not None:
            with torch.amp.autocast(enabled=True):
                return self._forward_impl(x, output_shape=output_shape, **kwargs)
        else:
            return self._forward_impl(x, output_shape=output_shape, **kwargs)

    def _forward_impl(self, x: torch.Tensor, output_shape: Optional[Tuple[int, ...]] = None, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        original_hw = (H, W)

        # 位置嵌入
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # 物理域 padding（可选）
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        # 提升
        x = self.lifting(x)
        x = self.lifting_norm(x)
        x = self.lifting_drop(x)

        # 主干（内部自带 2^L pad/unpad & fp32）
        x = self.wno_blocks(x)

        # 投影
        x = self.projection(x)
        x = self.projection_norm(x)
        x = self.projection_drop(x)

        # 物理域裁剪
        if self.domain_padding is not None:
            target_hw = output_shape[2:] if (output_shape is not None and len(output_shape) >= 4) else original_hw
            x = self.domain_padding.unpad(x, target_hw)

        # 强制输出尺寸（若上游接口要求）
        default_output_shape = kwargs.get('default_output_shape', None)
        if default_output_shape is not None and x.shape[2:] != tuple(default_output_shape):
            x = F.interpolate(x, size=default_output_shape, mode='bilinear', align_corners=True)

        return x


# -----------------------------
# 3D: WNOBlock（DWT）
# -----------------------------
class WNOBlock3d(nn.Module):
    """
    3D 小波神经算子块 (DWT):
      - 每层: x ← K(x) + W(x)；最后一层不激活；Channel-MLP 残差
      - 仅小波路径禁用 AMP（fp32）
      - 进入/离开块时，样本级右/后/下 pad 到 2^L 倍数并精确裁剪
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_levels: Tuple[int, int, int],     # 例如 (2,2,2)
        base_size: Tuple[int, int, int],    # 训练主分辨率 (D,H,W)
        wavelet: str = 'db6',
        dwt_mode: str = 'symmetric',
        n_layers: int = 1,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0.0,
        channel_mlp_expansion: float = 0.5,
        non_linearity=F.gelu,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_levels = n_levels
        self.level = max(n_levels)
        self.wavelet = wavelet
        self.dwt_mode = dwt_mode
        self.n_layers = n_layers
        self.non_linearity = non_linearity
        self.base_size = list(base_size)     # WaveConv3d 需要 list

        # 小波卷积 K（对口 WaveConv3d）
        self.convs = nn.ModuleList([
            WaveConv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                level=self.level,
                size=self.base_size,
                wavelet=self.wavelet,
                mode=self.dwt_mode
            ) for _ in range(n_layers)
        ])

        # 1×1×1 像素域 W
        self.w_local = nn.ModuleList([
            nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, bias=True)
            for _ in range(n_layers)
        ])

        # 通道 MLP（残差）
        self.use_channel_mlp = use_channel_mlp
        if use_channel_mlp:
            self.channel_mlps = nn.ModuleList([
                ChannelMLP(
                    in_channels=self.out_channels,
                    hidden_channels=max(1, int(round(self.out_channels * channel_mlp_expansion))),
                    dropout=channel_mlp_dropout,
                    n_dim=3,
                ) for _ in range(n_layers)
            ])
            self.channel_mlp_skips = nn.ModuleList([
                nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1, bias=True)
                for _ in range(n_layers)
            ])
        else:
            self.channel_mlps = None
            self.channel_mlp_skips = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amp_ctx = torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else contextlib.nullcontext()
        with amp_ctx:
            x, meta = pad_to_pow2_dhw(x, self.n_levels, mode=self.dwt_mode)

            for i in range(self.n_layers):
                kx = self.convs[i](x)            # K(x)
                wx = self.w_local[i](x)          # W(x)
                x = kx + wx

                if i != self.n_layers - 1:
                    x = self.non_linearity(x)

                if self.use_channel_mlp:
                    mlp_out = self.channel_mlps[i](x)
                    skip = self.channel_mlp_skips[i](x)
                    x = self.non_linearity(mlp_out + skip)

            x = unpad_from_meta_dhw(x, meta)
        return x


# -----------------------------
# 3D: 顶层 WNO3d（DWT）
# -----------------------------
class WNO3d(BaseModel):
    """
    3D 小波神经算子 (DWT):
      输入: [B, C_in, D, H, W] （若使用 grid 位置嵌入，会在通道维追加 3 个坐标通道）
      输出: [B, C_out, D, H, W]
    """
    def __init__(
        self,
        n_levels_depth: int,
        n_levels_height: int,
        n_levels_width: int,
        hidden_channels: int,
        base_size: Tuple[int, int, int],        # (D,H,W)
        in_channels: int = 3,
        out_channels: int = 1,
        wavelet: str = 'db6',
        dwt_mode: str = 'symmetric',
        n_layers: int = 4,
        positional_embedding: Union[str, nn.Module] = "grid",
        domain_padding: Union[Number, List[Number], None] = None,
        domain_padding_mode: str = "symmetric",
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0.0,
        channel_mlp_expansion: float = 0.5,
        non_linearity = F.gelu,
        use_amp: bool = True,
        use_group_norm: bool = True,
        dropout_rate: float = 0.10,
        lifting_channels: Optional[int] = None,
        projection_channels: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.n_levels = (n_levels_depth, n_levels_height, n_levels_width)
        self.level = max(self.n_levels)
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.use_amp = bool(use_amp and torch.cuda.is_available())
        self.dropout_rate = float(dropout_rate)
        self.base_size = base_size
        self.wavelet = wavelet
        self.dwt_mode = dwt_mode

        # 位置嵌入：坐标归一化到 [0,1]
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0., 1.]] * 3
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=3,
                grid_boundaries=spatial_grid_boundaries
            )
            lifted_in_channels = self.in_channels + 3
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
            lifted_in_channels = self.in_channels + 3
        elif positional_embedding is None:
            self.positional_embedding = None
            lifted_in_channels = self.in_channels
        else:
            raise ValueError(f"无效的位置嵌入类型: {positional_embedding}")

        # 可选“物理域”padding（与你的数据边界条件相关；与 2^L pad 独立）
        self.domain_padding = DomainPadding(domain_padding, domain_padding_mode) if domain_padding is not None else None

        # 提升
        self.lifting = ChannelMLP(
            in_channels=lifted_in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=(2 * hidden_channels if (lifting_channels is None) else lifting_channels),
            n_layers=2,
            n_dim=3
        )
        self.lifting_norm = make_norm(self.hidden_channels, 3, use_group_norm)
        self.lifting_drop = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        # 主干（3D DWT 路径）
        self.wno_blocks = WNOBlock3d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            n_levels=self.n_levels,
            base_size=self.base_size,
            wavelet=self.wavelet,
            dwt_mode=self.dwt_mode,
            n_layers=self.n_layers,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
        )

        # 投影
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=(2 * hidden_channels if (projection_channels is None) else projection_channels),
            n_layers=2,
            n_dim=3
        )
        self.projection_norm = make_norm(self.out_channels, 3, use_group_norm)
        self.projection_drop = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

    # 外层仍可使用 AMP；小波路径内部已禁用
    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int, ...]] = None, **kwargs) -> torch.Tensor:
        if self.use_amp and amp is not None:
            with torch.amp.autocast(enabled=True):
                return self._forward_impl(x, output_shape=output_shape, **kwargs)
        else:
            return self._forward_impl(x, output_shape=output_shape, **kwargs)

    def _forward_impl(self, x: torch.Tensor, output_shape: Optional[Tuple[int, ...]] = None, **kwargs) -> torch.Tensor:
        B, C, D, H, W = x.shape
        original_dhw = (D, H, W)

        # 位置嵌入
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # 物理域 padding（可选）
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        # 提升
        x = self.lifting(x)
        x = self.lifting_norm(x)
        x = self.lifting_drop(x)

        # 主干（内部自带 2^L pad/unpad & fp32）
        x = self.wno_blocks(x)

        # 投影
        x = self.projection(x)
        x = self.projection_norm(x)
        x = self.projection_drop(x)

        # 物理域裁剪
        if self.domain_padding is not None:
            target_dhw = output_shape[2:] if (output_shape is not None and len(output_shape) >= 5) else original_dhw
            x = self.domain_padding.unpad(x, target_dhw)

        # 若需要强制输出到某尺寸
        default_output_shape = kwargs.get('default_output_shape', None)
        if default_output_shape is not None and x.shape[2:] != tuple(default_output_shape):
            x = F.interpolate(x, size=default_output_shape, mode='trilinear', align_corners=True)

        return x
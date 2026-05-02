# -*- coding: utf-8 -*-
"""
Unified WNO2d (DWT/DTCWT) + WNO3d (DWT) implementation.
"""

import contextlib
from typing import Tuple, List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

Number = Union[float, int]

# === Project-local modules ===
from ..layers.embeddings import GridEmbeddingND
from ..layers.wavelet_conv import WaveConv2d, WaveConv2dCwt, WaveConv3d
from ..layers.padding import DomainPadding
from ..layers.channel_mlp import ChannelMLP
from .base_model import BaseModel

# -----------------------------
# Normalization: favor GroupNorm for small-batch stability
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
# 2D: WNOBlock (DWT vs DTCWT via conv_kind)
# -----------------------------
class WNOBlock2d(nn.Module):
    """
    2D wavelet neural operator block:
      - Each layer: x <- K(x) + W(x); no activation on last layer; Channel-MLP residual
      - Wavelet path runs in fp32 without AMP; rest uses outer AMP dtype (bf16/fp16)
      - Per-sample pad/crop to multiples of 2^L (handled inside WaveConv)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_levels: Tuple[int, int],           # e.g. (2, 2)
        base_size: Tuple[int, int],          # primary (H, W)
        conv_kind: str = "dwt",              # "dwt" | "dtcwt"
        # DWT
        wavelet: str = 'db6',
        dwt_mode: str = 'symmetric',         # default DWT padding
        # DTCWT
        biort: Optional[str] = None,
        qshift: Optional[str] = None,
        # Depth / nonlinearity
        n_layers: int = 1,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0.0,
        channel_mlp_expansion: float = 0.5,
        non_linearity=F.gelu,
    ):
        super().__init__()
        assert conv_kind in ("dwt", "dtcwt"), f"conv_kind must be 'dwt' or 'dtcwt', got {conv_kind}"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_levels = n_levels
        self.level = max(n_levels)          # wavelet decomposition depth
        self.conv_kind = conv_kind
        self.wavelet = wavelet
        self.dwt_mode = dwt_mode
        self.biort = biort
        self.qshift = qshift
        self.n_layers = n_layers
        self.non_linearity = non_linearity
        self.base_size = list(base_size)    # WaveConv2d / WaveConv2dCwt expect list

        # Wavelet conv K
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
                    "biort and qshift are required for DTCWT"
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

        # 1x1 pixel-domain W
        self.w_local = nn.ModuleList([
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=True)
            for _ in range(n_layers)
        ])

        # Channel MLP (residual)
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
        # Wavelet path: fp32, autocast off for DWT/IDWT / DTCWT stability
        orig_dtype = x.dtype

        for i in range(self.n_layers):
            # Wavelet branch K(x): fp32, autocast off
            with torch.autocast(device_type="cuda", enabled=False):
                x32 = x.to(torch.float32)
                kx32 = self.convs[i](x32)     # WaveConv aligns float32 filters when using pytorch_wavelets
            kx = kx32.to(dtype=orig_dtype)     # back to outer AMP dtype

            # 1x1 branch W(x) in outer AMP dtype
            wx = self.w_local[i](x)

            # Fusion
            x = kx + wx

            if i != self.n_layers - 1:
                x = self.non_linearity(x)

            if self.use_channel_mlp:
                mlp_out = self.channel_mlps[i](x)
                skip = self.channel_mlp_skips[i](x)
                x = self.non_linearity(mlp_out + skip)

        return x


# -----------------------------
# 2D: top-level WNO2d (DWT / DTCWT)
# -----------------------------
class WNO2d(BaseModel):
    """
    2D wavelet neural operator:
      - conv_kind='dwt' -> WaveConv2d (DWT)
      - conv_kind='dtcwt' -> WaveConv2dCwt (DTCWT)
    Input: [B, C_in, H, W] (grid embedding appends 2 coordinate channels)
    Output: [B, C_out, H, W]
    """
    def __init__(
        self,
        n_levels_height: int,
        n_levels_width: int,
        hidden_channels: int,
        base_size: Tuple[int, int],              # (H, W) primary resolution
        in_channels: int = 3,
        out_channels: int = 1,
        conv_kind: str = 'dwt',                  # 'dwt' | 'dtcwt'
        wavelet: str = 'db6',
        dwt_mode: str = 'symmetric',
        biort: Optional[str] = None,
        qshift: Optional[str] = None,
        n_layers: int = 4,
        positional_embedding: Union[str, nn.Module] = "grid",
        domain_padding: Union[Number, List[Number], None] = None,
        domain_padding_mode: str = "symmetric",
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0.0,
        channel_mlp_expansion: float = 0.5,
        non_linearity = F.gelu,
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
        self.dropout_rate = float(dropout_rate)
        self.base_size = base_size

        # Positional embedding (coords in [0,1])
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
            raise ValueError(f"Invalid positional_embedding: {positional_embedding}")

        # Optional physical-domain padding
        self.domain_padding = DomainPadding(domain_padding, domain_padding_mode) if domain_padding is not None else None

        # Lifting
        self.lifting = ChannelMLP(
            in_channels=lifted_in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=(2 * hidden_channels if (lifting_channels is None) else lifting_channels),
            n_layers=2,
            n_dim=2
        )
        self.lifting_norm = make_norm(self.hidden_channels, 2, use_group_norm)
        self.lifting_drop = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        # Backbone (K+W)
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

        # Projection
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=(2 * hidden_channels if (projection_channels is None) else projection_channels),
            n_layers=2,
            n_dim=2
        )
        self.projection_norm = make_norm(self.out_channels, 2, use_group_norm)
        self.projection_drop = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int, ...]] = None, **kwargs) -> torch.Tensor:
        return self._forward_impl(x, output_shape=output_shape, **kwargs)

    def _forward_impl(self, x: torch.Tensor, output_shape: Optional[Tuple[int, ...]] = None, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        original_hw = (H, W)

        # Positional embedding
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # Optional physical-domain padding
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        # Lifting
        x = self.lifting(x)
        x = self.lifting_norm(x)
        x = self.lifting_drop(x)

        # Backbone (internal 2^L pad/unpad; wavelet branch uses fp32, AMP off)
        x = self.wno_blocks(x)

        # Projection
        x = self.projection(x)
        x = self.projection_norm(x)
        x = self.projection_drop(x)

        # Unpad physical domain
        if self.domain_padding is not None:
            target_hw = output_shape[2:] if (output_shape is not None and len(output_shape) >= 4) else original_hw
            x = self.domain_padding.unpad(x, target_hw)

        # Enforce output size when requested by caller
        default_output_shape = kwargs.get('default_output_shape', None)
        if default_output_shape is not None and x.shape[2:] != tuple(default_output_shape):
            x = F.interpolate(x, size=default_output_shape, mode='bilinear', align_corners=True)

        return x


# -----------------------------
# 3D: WNOBlock (DWT)
# -----------------------------
class WNOBlock3d(nn.Module):
    """
    3D WNO block (DWT):
      - Each layer: x <- K(x) + W(x); no activation on last layer; Channel-MLP residual
      - Wavelet path: fp32, no AMP; rest keeps outer AMP dtype
      - Per-sample pad/crop to multiples of 2^L (inside WaveConv)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_levels: Tuple[int, int, int],     # e.g. (2,2,2)
        base_size: Tuple[int, int, int],    # primary (D,H,W)
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
        self.base_size = list(base_size)     # WaveConv3d expects list

        # Wavelet conv K (WaveConv3d)
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

        # 1x1x1 pixel-domain W
        self.w_local = nn.ModuleList([
            nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, bias=True)
            for _ in range(n_layers)
        ])

        # Channel MLP (residual)
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
        orig_dtype = x.dtype

        for i in range(self.n_layers):
            # Wavelet branch K(x): fp32, autocast off
            with torch.autocast(device_type="cuda", enabled=False):
                x32 = x.to(torch.float32)
                kx32 = self.convs[i](x32)
            kx = kx32.to(dtype=orig_dtype)

            # 1x1x1 branch W(x) in outer AMP dtype
            wx = self.w_local[i](x)

            x = kx + wx

            if i != self.n_layers - 1:
                x = self.non_linearity(x)

            if self.use_channel_mlp:
                mlp_out = self.channel_mlps[i](x)
                skip = self.channel_mlp_skips[i](x)
                x = self.non_linearity(mlp_out + skip)

        return x


# -----------------------------
# 3D: top-level WNO3d (DWT)
# -----------------------------
class WNO3d(BaseModel):
    """
    3D wavelet neural operator (DWT):
    Input: [B, C_in, D, H, W] (grid embedding appends 3 coordinate channels)
    Output: [B, C_out, D, H, W]
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
        self.dropout_rate = float(dropout_rate)
        self.base_size = base_size
        self.wavelet = wavelet
        self.dwt_mode = dwt_mode

        # Positional embedding: coords in [0,1]
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
            raise ValueError(f"Invalid positional_embedding: {positional_embedding}")

        # Optional physical-domain padding (BCs; separate from 2^L padding)
        self.domain_padding = DomainPadding(domain_padding, domain_padding_mode) if domain_padding is not None else None

        # Lifting
        self.lifting = ChannelMLP(
            in_channels=lifted_in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=(2 * hidden_channels if (lifting_channels is None) else lifting_channels),
            n_layers=2,
            n_dim=3
        )
        self.lifting_norm = make_norm(self.hidden_channels, 3, use_group_norm)
        self.lifting_drop = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        # Backbone (3D DWT stack)
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

        # Projection
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=(2 * hidden_channels if (projection_channels is None) else projection_channels),
            n_layers=2,
            n_dim=3
        )
        self.projection_norm = make_norm(self.out_channels, 3, use_group_norm)
        self.projection_drop = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

    # Outer forward may use AMP; wavelet internals disable it
    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int, ...]] = None, **kwargs) -> torch.Tensor:
        return self._forward_impl(x, output_shape=output_shape, **kwargs)
        
    def _forward_impl(self, x: torch.Tensor, output_shape: Optional[Tuple[int, ...]] = None, **kwargs) -> torch.Tensor:
        B, C, D, H, W = x.shape
        original_dhw = (D, H, W)

        # Positional embedding
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # Optional physical-domain padding
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        # Lifting
        x = self.lifting(x)
        x = self.lifting_norm(x)
        x = self.lifting_drop(x)

        # Backbone (internal 2^L pad/unpad; wavelet branch uses fp32, AMP off)
        x = self.wno_blocks(x)

        # Projection
        x = self.projection(x)
        x = self.projection_norm(x)
        x = self.projection_drop(x)

        # Unpad physical domain
        if self.domain_padding is not None:
            target_dhw = output_shape[2:] if (output_shape is not None and len(output_shape) >= 5) else original_dhw
            x = self.domain_padding.unpad(x, target_dhw)

        # Optional fixed output size
        default_output_shape = kwargs.get('default_output_shape', None)
        if default_output_shape is not None and x.shape[2:] != tuple(default_output_shape):
            x = F.interpolate(x, size=default_output_shape, mode='trilinear', align_corners=True)

        return x
"""
小波卷积层 - 高效Haar小波实现
使用纯Tensor操作，避免矩阵乘法导致的shape问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from numbers import Number
import numpy as np

# 导入混合精度训练相关模块
try:
    import torch.cuda.amp as amp
except ImportError:
    amp = None

from .base_spectral_conv import BaseSpectralConv

def _optimize_for_gpu(tensor, device=None, dtype=None):
    """GPU优化：确保张量在正确的设备和数据类型上"""
    if device is not None:
        tensor = tensor.to(device)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.contiguous()

class WaveletConv(BaseSpectralConv):
    """
    高效Haar小波卷积层 - 纯Tensor操作版本
    
    使用张量切片和einsum操作，避免复杂的矩阵乘法
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_levels: List[int],
        wavelet_type: str = 'haar',
        wavelet_filter: int = 2,
        resolution_scaling_factor: Optional[List[Number]] = None,
        max_n_levels: Optional[List[int]] = None,
        complex_data: bool = False,
        separable: bool = False,
        factorization: Optional[str] = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = 'factorized',
        decomposition_kwargs: dict = dict(),
        precision: str = 'full',
        fno_block_precision: str = 'full',
        ensure_even_shapes: bool = False,
        pad_mode: str = 'constant',
        adaptive_padding: bool = False,
        device = None,
        dtype = None,
        use_checkpoint: bool = False,
        use_amp: bool = True,
        **kwargs
    ):
        # 只传递BaseSpectralConv支持的参数
        super().__init__(device=device, dtype=dtype)
        
        # 保存参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_levels = n_levels
        self.complex_data = complex_data
        self.separable = separable
        self.factorization = factorization
        self.rank = rank
        self.fixed_rank_modes = fixed_rank_modes
        self.implementation = implementation
        self.decomposition_kwargs = decomposition_kwargs
        self.precision = precision
        self.fno_block_precision = fno_block_precision
        self.ensure_even_shapes = ensure_even_shapes
        self.pad_mode = pad_mode
        self.adaptive_padding = adaptive_padding
        self.use_checkpoint = use_checkpoint
        self.use_amp = use_amp
        
        # 设置设备和数据类型
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype or torch.float32
        
        # 确保权重在正确的设备上
        self.to(self.device)
        
        # 小波参数
        self.wavelet_type = wavelet_type
        self.wavelet_filter = wavelet_filter
        self.resolution_scaling_factor = resolution_scaling_factor
        self.max_n_levels = max_n_levels or n_levels
        
        # 初始化权重 - 使用高效的权重结构
        self._initialize_weights()
        
        # 性能优化设置
        self.use_checkpoint = use_checkpoint
        self.use_amp = use_amp and torch.cuda.is_available()
    
    def _initialize_weights(self):
        """初始化权重参数 - 高效版本"""
        if not self.complex_data:
            # 创建权重 [out_channels, in_channels, 4]，4是LL/LH/HL/HH
            self.weight = nn.Parameter(
                torch.empty(self.out_channels, self.in_channels, 4, device=self.device, dtype=self.dtype)
            )
            nn.init.xavier_normal_(self.weight)
            self.bias = nn.Parameter(torch.zeros(self.out_channels, 1, 1, device=self.device, dtype=self.dtype))
        else:
            # 复数情况
            self.weight_real = nn.Parameter(
                torch.empty(self.out_channels, self.in_channels, 4, device=self.device, dtype=self.dtype)
            )
            self.weight_imag = nn.Parameter(
                torch.empty(self.out_channels, self.in_channels, 4, device=self.device, dtype=self.dtype)
            )
            nn.init.xavier_normal_(self.weight_real)
            nn.init.xavier_normal_(self.weight_imag)
            self.bias = nn.Parameter(torch.zeros(self.out_channels, 1, 1, device=self.device, dtype=self.dtype))
        
        # 确保所有参数都在正确的设备上
        self.to(self.device)
    
    def haar_decompose_2d(self, x):
        """2D Haar小波分解 - 纯Tensor操作"""
        # 确保输入在正确的设备上
        device = x.device
        dtype = x.dtype
        
        # x: [B, C, H, W], H/W必须为偶数
        # 确保输入尺寸为偶数
        h, w = x.shape[2], x.shape[3]
        if h % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1), mode=self.pad_mode)
        if w % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0), mode=self.pad_mode)
        
        # 使用张量切片进行Haar分解
        LL = (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] + x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) * 0.5
        LH = (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] + x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) * 0.5
        HL = (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] - x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) * 0.5
        HH = (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] - x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) * 0.5
        
        # 确保所有张量在同一设备上
        LL = LL.to(device)
        LH = LH.to(device)
        HL = HL.to(device)
        HH = HH.to(device)
        
        # 返回 [B, C, 4, H//2, W//2]
        result = torch.stack([LL, LH, HL, HH], dim=2)
        return result.to(device)
    
    def haar_reconstruct_2d(self, coeffs):
        """2D Haar小波重构 - 纯Tensor操作"""
        # coeffs: [B, C, 4, H, W]
        device = coeffs.device
        dtype = coeffs.dtype
        
        LL, LH, HL, HH = coeffs.unbind(dim=2)
        B, C, H, W = LL.shape
        
        # 创建输出张量
        x = torch.zeros(B, C, H*2, W*2, device=device, dtype=dtype)
        
        # 使用张量切片进行Haar重构
        x[..., 0::2, 0::2] = (LL + LH + HL + HH) * 0.5
        x[..., 0::2, 1::2] = (LL - LH + HL - HH) * 0.5
        x[..., 1::2, 0::2] = (LL + LH - HL - HH) * 0.5
        x[..., 1::2, 1::2] = (LL - LH - HL + HH) * 0.5
        
        return x
    
    def haar_decompose_1d(self, x):
        """1D Haar小波分解 - 纯Tensor操作"""
        # x: [B, C, L], L必须为偶数
        if x.shape[-1] % 2 == 1:
            x = F.pad(x, (0, 1), mode=self.pad_mode)
        
        # 使用张量切片进行Haar分解
        L = (x[..., 0::2] + x[..., 1::2]) * 0.7071067811865475
        H = (x[..., 0::2] - x[..., 1::2]) * 0.7071067811865475
        
        # 返回 [B, C, 2, L//2]
        return torch.stack([L, H], dim=2)
    
    def haar_reconstruct_1d(self, coeffs):
        """1D Haar小波重构 - 纯Tensor操作"""
        # coeffs: [B, C, 2, L]
        L, H = coeffs.unbind(dim=2)
        B, C, length = L.shape
        
        # 创建输出张量
        x = torch.zeros(B, C, length*2, device=coeffs.device, dtype=coeffs.dtype)
        
        # 使用张量切片进行Haar重构
        x[..., 0::2] = (L + H) * 0.7071067811865475
        x[..., 1::2] = (L - H) * 0.7071067811865475
        
        return x
    
    def dwt_forward(self, x, levels=None):
        """离散小波变换 - 高效Haar版本"""
        if levels is None:
            levels = self.n_levels
        
        # 获取输入的空间维度
        spatial_dims = x.shape[2:]
        
        if len(spatial_dims) == 1:
            # 1D情况
            coeffs = self.haar_decompose_1d(x)
            return [coeffs]
        
        elif len(spatial_dims) == 2:
            # 2D情况
            coeffs = self.haar_decompose_2d(x)
            return [coeffs]
        
        else:
            # 3D或更高维情况 - 使用简单的下采样
            return [x[:, :, ::2, ::2] if len(spatial_dims) >= 2 else x[:, :, ::2]]
    
    def idwt_forward(self, coeffs, output_size):
        """逆离散小波变换 - 高效Haar版本"""
        if len(coeffs) == 1:
            coeff = coeffs[0]
            if coeff.shape[2] == 4:  # 2D情况
                return self.haar_reconstruct_2d(coeff)
            elif coeff.shape[2] == 2:  # 1D情况
                return self.haar_reconstruct_1d(coeff)
            else:
                return coeff
        
        # 多层情况 - 递归重构
        result = coeffs[-1]
        for i in range(len(coeffs) - 2, -1, -1):
            if result.shape[2] == 4:  # 2D情况
                result = self.haar_reconstruct_2d(result)
            elif result.shape[2] == 2:  # 1D情况
                result = self.haar_reconstruct_1d(result)
        
        return result
    
    def forward(self, x, output_shape=None):
        """前向传播 - 高效Haar版本"""
        # 使用混合精度训练
        if self.use_amp and amp is not None:
            with torch.amp.autocast('cuda'):
                return self._forward_impl(x, output_shape)
        else:
            return self._forward_impl(x, output_shape)
    
    def _forward_impl(self, x, output_shape=None):
        """前向传播实现 - 高效Haar版本"""
        # 确保输入是连续的
        x = _optimize_for_gpu(x, self.device, self.dtype)
        
        # 获取输入形状
        batch_size, in_channels, *spatial_dims = x.shape
        
        # 执行小波变换
        coeffs = self.dwt_forward(x, self.n_levels)
        
        # 应用权重 - 使用einsum进行高效卷积
        if not self.complex_data:
            weight = self.weight
            bias = self.bias
        else:
            weight = torch.complex(self.weight_real, self.weight_imag)
            bias = self.bias
        
        # 批量处理所有系数
        out_coeffs = []
        for coeff in coeffs:
            # 使用einsum进行高效的权重应用
            # coeff: [B, C, 4, H, W] 或 [B, C, 2, L]
            # weight: [out_channels, in_channels, 4] 或 [out_channels, in_channels, 2]
            # 输出: [B, out_channels, H, W] 或 [B, out_channels, L]
            
            if coeff.shape[2] == 4:  # 2D情况
                out = torch.einsum('ocf,bcfxy->boxy', weight, coeff) + bias
            else:  # 1D情况
                out = torch.einsum('ocf,bcfx->box', weight, coeff) + bias
            
            out_coeffs.append(out)
        
        # 执行逆小波变换
        if output_shape is None:
            output_shape = spatial_dims
        
        result = self.idwt_forward(out_coeffs, output_shape)
        
        # 确保输出形状正确 - 修复尺寸不匹配问题
        if len(result.shape) >= 2 and result.shape[-len(output_shape):] != output_shape:
            # 使用正确的插值方式 - 只传递空间维度
            if len(output_shape) == 2:
                result = F.interpolate(result, size=output_shape, mode='bilinear', align_corners=False)
            elif len(output_shape) == 1:
                result = F.interpolate(result, size=output_shape, mode='linear', align_corners=False)
            else:
                result = F.interpolate(result, size=output_shape, mode='nearest')
        
        return result
    
    def _dwt_1d(self, x, levels):
        """一维离散小波变换 - 高效Haar版本"""
        return self.haar_decompose_1d(x)
    
    def _dwt_2d(self, x, levels_h, levels_w):
        """二维离散小波变换 - 高效Haar版本"""
        return self.haar_decompose_2d(x)
    
    def _dwt_3d(self, x, levels_d, levels_h, levels_w):
        """三维离散小波变换 - 高效Haar版本"""
        # 3D情况暂时使用简单的下采样
        return x[:, :, ::2, ::2, ::2] if len(x.shape) == 5 else x[:, :, ::2, ::2] 
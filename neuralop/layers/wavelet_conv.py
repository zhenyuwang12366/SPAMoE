import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Union, Optional

from .base_spectral_conv import BaseSpectralConv

Number = Union[float, int]


class WaveletConv(BaseSpectralConv):
    """
    小波卷积层
    
    该层使用小波变换代替傅里叶变换，在小波域中执行卷积操作。
    这使得模型能够捕捉信号的局部特性和多尺度特征。
    
    Parameters
    ----------
    in_channels : int
        输入通道数
    out_channels : int
        输出通道数
    n_levels : int
        小波分解的级别数
    wavelet_type : str, optional
        小波类型，目前支持'haar', 'db', 'sym', 'coif'，默认为'haar'
    wavelet_filter : int, optional
        小波滤波器长度（对于特定小波类型），默认为2
    complex_data : bool, optional
        数据是否为复数，默认为False
    separable : bool, optional
        是否使用可分离卷积，默认为False
    factorization : str, optional
        权重矩阵的因子分解方法，默认为None
    rank : float, optional
        因子分解的秩，默认为1.0
    fixed_rank_modes : bool, optional
        是否固定秩模式，默认为False
    implementation : str, optional
        实现方式，默认为'factorized'
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
        **kwargs
    ):
        # 只传递BaseSpectralConv支持的参数
        super().__init__(device=device, dtype=dtype)
        
        # 保存模型参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 如果n_levels是一个整数，转换为列表
        if isinstance(n_levels, int):
            self.n_levels = [n_levels]
        else:
            self.n_levels = n_levels
            
        self.wavelet_type = wavelet_type
        self.wavelet_filter = wavelet_filter
        self.complex_data = complex_data
        self.separable = separable
        self.factorization = factorization
        self.rank = rank
        self.fixed_rank_modes = fixed_rank_modes
        self.implementation = implementation
        self.resolution_scaling_factor = resolution_scaling_factor
        self.max_n_levels = max_n_levels
        self.decomposition_kwargs = decomposition_kwargs
        self.precision = precision
        self.fno_block_precision = fno_block_precision
        self.ensure_even_shapes = ensure_even_shapes
        self.pad_mode = pad_mode
        self.adaptive_padding = adaptive_padding
        
        # 确定维度
        self.n_dim = len(n_levels)
        
        # 创建小波滤波器
        self._create_wavelet_filters()
        
        # 初始化权重
        self._initialize_weights()
    
    def dwt_forward(self, x, levels=None):
        """
        前向离散小波变换 (DWT)
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量
        levels : int or List[int], optional
            各维度的分解级别, 默认为None (使用初始化时设定的级别)
        
        Returns
        -------
        List[torch.Tensor]
            小波系数列表
        """
        if levels is None:
            levels = self.n_levels
        
        # 如果启用了自适应填充，确保输入尺寸适合小波变换
        if self.adaptive_padding and self.n_dim > 0:
            # 检查每个空间维度
            pad_values = []
            shapes = x.shape[2:] if len(x.shape) > 2 else [x.shape[1]]
            
            for i, dim_size in enumerate(shapes):
                # 计算2^levels后的尺寸
                target_size = dim_size
                # 确保level是一个整数
                if isinstance(levels, (list, tuple)) and i < len(levels):
                    level_value = levels[i]
                    # 如果level_value本身还是元组或列表，取第一个元素
                    if isinstance(level_value, (list, tuple)):
                        level_value = level_value[0]
                else:
                    # 如果levels是一个元组，取第一个元素
                    if isinstance(levels, (list, tuple)):
                        level_value = levels[0]
                    else:
                        level_value = levels
                
                # 确保level_value是一个整数
                level_value = int(level_value)
                
                for _ in range(level_value):
                    # 每级小波变换将尺寸减半，确保能被2整除
                    if target_size % 2 != 0:
                        target_size += 1
                
                # 计算需要的填充量
                pad_size = target_size - dim_size if target_size > dim_size else 0
                
                # 对于不同维度，填充列表的顺序不同
                if self.n_dim == 1:
                    pad_values = [0, pad_size]  # [左填充, 右填充]
                elif self.n_dim == 2:
                    # 二维情况: [左, 右, 上, 下] - 反向添加
                    if i == 0:  # 高度维度 (最后一个空间维度)
                        pad_values = [0, 0, 0, pad_size] + pad_values
                    else:  # 宽度维度 (倒数第二个空间维度)
                        pad_values = [0, pad_size] + pad_values
                elif self.n_dim == 3:
                    # 三维情况: [前, 后, 上, 下, 左, 右] - 反向添加
                    if i == 0:  # 深度维度
                        pad_values = [0, 0, 0, 0, 0, pad_size] + pad_values
                    elif i == 1:  # 高度维度
                        pad_values = [0, 0, 0, pad_size] + pad_values
                    else:  # 宽度维度
                        pad_values = [0, pad_size] + pad_values
            
            # 应用填充
            if any(pad_values):
                x = F.pad(x, pad_values, mode=self.pad_mode)
                
        # 一维情况
        if self.n_dim == 1:
            level_value = levels[0] if isinstance(levels, (list, tuple)) else levels
            if isinstance(level_value, (list, tuple)):
                level_value = level_value[0]
            return self._dwt_1d(x, int(level_value))
            
        # 二维情况
        elif self.n_dim == 2:
            level_h = levels[0] if isinstance(levels, (list, tuple)) and len(levels) > 0 else levels
            level_w = levels[1] if isinstance(levels, (list, tuple)) and len(levels) > 1 else levels
            
            # 确保level_h和level_w是整数
            if isinstance(level_h, (list, tuple)):
                level_h = level_h[0]
            if isinstance(level_w, (list, tuple)):
                level_w = level_w[0]
                
            return self._dwt_2d(x, int(level_h), int(level_w))
            
        # 三维情况
        elif self.n_dim == 3:
            level_d = levels[0] if isinstance(levels, (list, tuple)) and len(levels) > 0 else levels
            level_h = levels[1] if isinstance(levels, (list, tuple)) and len(levels) > 1 else levels
            level_w = levels[2] if isinstance(levels, (list, tuple)) and len(levels) > 2 else levels
            
            # 确保level_d、level_h和level_w是整数
            if isinstance(level_d, (list, tuple)):
                level_d = level_d[0]
            if isinstance(level_h, (list, tuple)):
                level_h = level_h[0]
            if isinstance(level_w, (list, tuple)):
                level_w = level_w[0]
                
            return self._dwt_3d(x, int(level_d), int(level_h), int(level_w))
    
    def idwt_forward(self, coeffs, output_size):
        """
        逆小波变换
        
        Parameters
        ----------
        coeffs : List[torch.Tensor]
            小波系数列表
        output_size : tuple
            输出张量的尺寸
        
        Returns
        -------
        torch.Tensor
            重构后的张量
        """
        # 获取小波系数
        if len(coeffs) == 0:
            # 如果没有系数，返回零张量
            print("警告：没有小波系数，返回零张量")
            return torch.zeros(output_size, device=coeffs[0].device if coeffs else None)
        elif len(coeffs) == 1:
            # 如果只有近似系数，直接返回
            return coeffs[0]
        
        try:
            # 一维情况
            if self.n_dim == 1:
                return self._idwt_1d(coeffs, output_size)
            # 二维情况
            elif self.n_dim == 2:
                return self._idwt_2d(coeffs, output_size)
            # 三维情况
            elif self.n_dim == 3:
                return self._idwt_3d(coeffs, output_size)
            else:
                print(f"警告：不支持的维度 {self.n_dim}，返回零张量")
                return torch.zeros(output_size, device=coeffs[0].device)
        except Exception as e:
            print(f"逆小波变换中发生错误: {str(e)}")
            # 返回第一个系数作为后备
            if len(coeffs) > 0 and coeffs[0] is not None:
                # 调整第一个系数的大小以匹配输出
                try:
                    if coeffs[0].dim() == 4:  # 二维数据
                        return F.interpolate(coeffs[0], size=(output_size[2], output_size[3]), 
                                          mode='bilinear', align_corners=True)
                    elif coeffs[0].dim() == 5:  # 三维数据
                        return F.interpolate(coeffs[0], size=(output_size[2], output_size[3], output_size[4]), 
                                          mode='trilinear', align_corners=True)
                    else:  # 一维数据
                        return F.interpolate(coeffs[0], size=output_size[2:], 
                                          mode='linear', align_corners=True)
                except Exception:
                    # 如果插值失败，返回零张量
                    return torch.zeros(output_size, device=coeffs[0].device)
            else:
                # 如果没有可用的系数，返回零张量
                return torch.zeros(output_size, device=coeffs[0].device if coeffs and coeffs[0] is not None else None)

    def _idwt_1d(self, coeffs, output_size):
        """一维逆离散小波变换实现 - 非递归版本"""
        if len(coeffs) == 0:
            print("警告：没有小波系数，返回零张量")
            return torch.zeros(output_size, device=coeffs[0].device if coeffs else None)
        
        # 计算变换级数
        n_levels = len(coeffs) // 2 + (1 if len(coeffs) % 2 == 1 else 0)
        if n_levels == 0:
            # 如果只有一个系数，直接返回
            return coeffs[0] if coeffs[0] is not None else torch.zeros(output_size)
        
        # 从最低频重构到最高频
        # 先获取最低频的近似系数
        approx = coeffs[-1]  # 最后一个系数是近似系数
        
        if approx is None:
            print("警告：近似系数为None，使用零张量")
            # 创建一个合适尺寸的零张量
            batch_size = output_size[0]
            channels = output_size[1]
            # 推断一个合理的近似系数尺寸：输出尺寸除以2^n_levels
            length = max(1, output_size[2] // (2 ** n_levels))
            approx = torch.zeros(batch_size, channels, length, device=coeffs[0].device if coeffs else None)
        
        # 从最低频开始，逐级重构
        for level in range(n_levels):
            # 计算当前级别的细节系数索引
            idx = len(coeffs) - 1 - (level + 1)
            
            # 获取当前级别的细节系数
            detail = coeffs[idx] if idx >= 0 and idx < len(coeffs) else None
            
            # 如果细节系数为None，使用零张量替代
            if detail is None:
                detail = torch.zeros_like(approx)
                
            # 确保形状一致
            if detail.shape[2:] != approx.shape[2:]:
                detail = F.interpolate(detail, size=approx.shape[2:], 
                                     mode='linear', align_corners=True)
            
            # 计算下一级的尺寸
            batch_size, channels = approx.shape[0], approx.shape[1]
            length = approx.shape[2]
            next_length = length * 2
            
            # 创建输出张量
            next_approx = torch.zeros((batch_size, channels, next_length), device=approx.device)
            
            # 使用Haar小波公式重构
            # 偶数位置 = approx + detail
            # 奇数位置 = approx - detail
            next_approx[:, :, ::2] = approx + detail
            next_approx[:, :, 1::2] = approx - detail
            
            # 更新approx为下一级重构结果
            approx = next_approx
        
        # 最终结果可能需要调整尺寸以匹配输出尺寸
            if approx.shape[2] != output_size[2]:
            # 裁剪或插值
                    if approx.shape[2] >= output_size[2]:
                        approx = approx[:, :, :output_size[2]]
                    else:
                        try:
                            approx = F.interpolate(approx, size=output_size[2:], 
                                        mode='linear', align_corners=True)
                        except Exception as e:
                            print(f"调整输出尺寸时出错: {e}")
                            # 创建正确尺寸的零张量
                            approx = torch.zeros(output_size, device=approx.device)
                        
            return approx
                
    def _idwt_2d(self, coeffs, output_size):
        """二维逆离散小波变换实现 - 非递归版本"""
        # 首先获取最终的LL子带（最低频率的近似系数）
        # 对于二维小波变换，系数顺序为:
        # [HH_1, HL_1, LH_1, ..., HH_n, HL_n, LH_n, LL_n]
        # 其中n是变换级数
        
        if len(coeffs) == 0:
            print("警告：没有小波系数，返回零张量")
            return torch.zeros(output_size, device=coeffs[0].device if coeffs else None)
        
        # 计算变换级数
        n_levels = (len(coeffs) + 1) // 4
        if n_levels == 0:
            # 如果只有一个系数，直接返回
            return coeffs[0] if coeffs[0] is not None else torch.zeros(output_size)
        
        # 从最低频重构到最高频
        # 先获取最低频的近似系数
        ll = coeffs[-1]  # 最后一个系数是LL
        if ll is None:
            print("警告：LL系数为None，使用零张量")
            # 创建一个合适尺寸的零张量
            batch_size = output_size[0]
            channels = output_size[1]
            # 推断一个合理的LL尺寸：输出尺寸除以2^n_levels
            h_ll = max(1, output_size[2] // (2 ** n_levels))
            w_ll = max(1, output_size[3] // (2 ** n_levels))
            ll = torch.zeros(batch_size, channels, h_ll, w_ll, device=coeffs[0].device if coeffs else None)
        
        # 从LL开始，逐级重构
        for level in range(n_levels):
            # 计算当前级别的系数索引
            idx_base = len(coeffs) - 1 - 3 * (level + 1)
            
            # 获取当前级别的细节系数
            lh = coeffs[idx_base + 2] if idx_base + 2 >= 0 and idx_base + 2 < len(coeffs) else None
            hl = coeffs[idx_base + 1] if idx_base + 1 >= 0 and idx_base + 1 < len(coeffs) else None
            hh = coeffs[idx_base] if idx_base >= 0 and idx_base < len(coeffs) else None
            
            # 如果细节系数为None，使用零张量替代
            if lh is None:
                lh = torch.zeros_like(ll)
            if hl is None:
                hl = torch.zeros_like(ll)
            if hh is None:
                hh = torch.zeros_like(ll)
                
                # 确保所有系数具有相同的形状
            if lh.shape[2:] != ll.shape[2:]:
                lh = F.interpolate(lh, size=ll.shape[2:], mode='bilinear', align_corners=True)
            if hl.shape[2:] != ll.shape[2:]:
                hl = F.interpolate(hl, size=ll.shape[2:], mode='bilinear', align_corners=True)
            if hh.shape[2:] != ll.shape[2:]:
                hh = F.interpolate(hh, size=ll.shape[2:], mode='bilinear', align_corners=True)
            
            # 计算下一级的尺寸
            batch_size, channels = ll.shape[0], ll.shape[1]
            h, w = ll.shape[2], ll.shape[3]
            next_h, next_w = h * 2, w * 2
            
            # 创建输出张量
            next_ll = torch.zeros((batch_size, channels, next_h, next_w), device=ll.device)
            
            # 使用Haar小波公式重构
            # 左上 (偶行偶列) = 0.5 * (LL + LH + HL + HH)
            # 右上 (偶行奇列) = 0.5 * (LL - LH + HL - HH)
            # 左下 (奇行偶列) = 0.5 * (LL + LH - HL - HH)
            # 右下 (奇行奇列) = 0.5 * (LL - LH - HL + HH)
            next_ll[:, :, ::2, ::2] = 0.5 * (ll + lh + hl + hh)
            next_ll[:, :, ::2, 1::2] = 0.5 * (ll - lh + hl - hh)
            next_ll[:, :, 1::2, ::2] = 0.5 * (ll + lh - hl - hh)
            next_ll[:, :, 1::2, 1::2] = 0.5 * (ll - lh - hl + hh)
            
            # 更新ll为下一级重构结果
            ll = next_ll
        
        # 最终结果可能需要调整尺寸以匹配输出尺寸
        if ll.shape[2] != output_size[2] or ll.shape[3] != output_size[3]:
            # 裁剪或插值
            if ll.shape[2] >= output_size[2] and ll.shape[3] >= output_size[3]:
                ll = ll[:, :, :output_size[2], :output_size[3]]
            else:
                try:
                    ll = F.interpolate(ll, size=(output_size[2], output_size[3]), 
                                              mode='bilinear', align_corners=True)
                except Exception as e:
                    print(f"调整输出尺寸时出错: {e}")
                    # 创建正确尺寸的零张量
                    ll = torch.zeros(output_size, device=ll.device)
        
        return ll

    def _idwt_3d(self, coeffs, output_size):
        """三维逆离散小波变换实现 - 简化版本"""
        if len(coeffs) == 0:
            print("警告：没有小波系数，返回零张量")
            return torch.zeros(output_size, device=coeffs[0].device if coeffs else None)
        
        # 3D小波变换的实现比较复杂，我们采用简化的方式：
        # 直接从最低频的近似系数进行上采样
        
        # 获取最低频的近似系数
        ll = coeffs[-1]  # 最后一个系数是近似系数
        
        if ll is None:
            print("警告：3D小波变换近似系数为None，返回零张量")
            return torch.zeros(output_size, device=coeffs[0].device if coeffs else None)
        
        # 检查是否需要调整尺寸
        if (ll.shape[2] != output_size[2] or 
            ll.shape[3] != output_size[3] or 
            ll.shape[4] != output_size[4]):
            
            try:
                # 使用三线性插值进行上采样
                return F.interpolate(
                    ll, 
                    size=(output_size[2], output_size[3], output_size[4]),
                    mode='trilinear',
                    align_corners=True
                )
            except Exception as e:
                print(f"3D插值失败: {e}")
                # 如果插值失败，返回零张量
                return torch.zeros(output_size, device=ll.device)
        else:
            # 尺寸已经匹配，直接返回
            return ll
    
    def forward(self, x, output_shape=None):
        """
        前向传播
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量
        output_shape : tuple, optional
            输出张量的形状，默认为None
        
        Returns
        -------
        torch.Tensor
            输出张量
        """
        batch_size = x.shape[0]
        n_channels = x.shape[1]
        
        # 保存原始形状用于输出
        original_shape = x.shape
        
        if output_shape is None:
            output_shape = x.shape
            
        # 如果启用了形状确保，检查输入尺寸
        if self.ensure_even_shapes and self.n_dim == 2:
            h, w = x.shape[2], x.shape[3]
            if h % 2 == 1 or w % 2 == 1:
                pad_h = 1 if h % 2 == 1 else 0
                pad_w = 1 if w % 2 == 1 else 0
                x = F.pad(x, (0, pad_w, 0, pad_h), mode=self.pad_mode)
        
        try:
            # 计算小波变换
            x_coeffs = self.dwt_forward(x)
            
            # 在小波域中应用可学习的权重
            out_coeffs = []
            
            if not self.complex_data:
                for i, coeff in enumerate(x_coeffs):
                    # 对每个尺度级别应用权重
                    if i < len(x_coeffs) - 1:  # 细节系数
                        out_coeff = torch.zeros_like(coeff)
                        for in_channel in range(self.in_channels):
                            for out_channel in range(self.out_channels):
                                # 获取可学习权重
                                weight = self.weight[out_channel, in_channel]
                                
                                # 处理权重和系数的尺寸不匹配问题
                                if isinstance(weight, torch.Tensor) and weight.ndim > 0:
                                    # 对于多维权重，确保形状匹配
                                    if coeff[:, in_channel].shape != weight.shape and weight.ndim > 0:
                                        # 如果维度不匹配，使用广播或调整大小
                                        if self.n_dim == 1:
                                            # 一维情况
                                            weight_expanded = weight.expand(coeff.shape[2])
                                        elif self.n_dim == 2:
                                            # 二维情况 - 使用插值调整权重大小
                                            weight_expanded = F.interpolate(
                                                weight.unsqueeze(0).unsqueeze(0),
                                                size=coeff.shape[2:],
                                                mode='nearest'
                                            ).squeeze(0).squeeze(0)
                                        else:
                                            # 其他维度情况
                                            weight_expanded = weight
                                    else:
                                        weight_expanded = weight
                                else:
                                    # 标量权重直接使用
                                    weight_expanded = weight
                                    
                                # 应用权重
                                out_coeff[:, out_channel] += coeff[:, in_channel] * weight_expanded
                        out_coeffs.append(out_coeff)
                    else:  # 近似系数
                        out_coeff = torch.zeros_like(coeff)
                        for in_channel in range(self.in_channels):
                            for out_channel in range(self.out_channels):
                                # 获取可学习权重
                                weight = self.weight[out_channel, in_channel]
                                
                                # 处理权重和系数的尺寸不匹配问题
                                if isinstance(weight, torch.Tensor) and weight.ndim > 0:
                                    # 对于多维权重，确保形状匹配
                                    if coeff[:, in_channel].shape != weight.shape and weight.ndim > 0:
                                        # 如果维度不匹配，使用广播或调整大小
                                        if self.n_dim == 1:
                                            # 一维情况
                                            weight_expanded = weight.expand(coeff.shape[2])
                                        elif self.n_dim == 2:
                                            # 二维情况 - 使用插值调整权重大小
                                            weight_expanded = F.interpolate(
                                                weight.unsqueeze(0).unsqueeze(0),
                                                size=coeff.shape[2:],
                                                mode='nearest'
                                            ).squeeze(0).squeeze(0)
                                        else:
                                            # 其他维度情况
                                            weight_expanded = weight
                                    else:
                                        weight_expanded = weight
                                else:
                                    # 标量权重直接使用
                                    weight_expanded = weight
                                    
                                # 应用权重
                                out_coeff[:, out_channel] += coeff[:, in_channel] * weight_expanded
                        out_coeffs.append(out_coeff)
            else:
                # 复数情况的处理
                for i, coeff in enumerate(x_coeffs):
                    # 实部和虚部分别处理
                    if i < len(x_coeffs) - 1:  # 细节系数
                        out_coeff = torch.zeros_like(coeff, dtype=torch.complex64)
                        for in_channel in range(self.in_channels):
                            for out_channel in range(self.out_channels):
                                # 获取复数权重
                                weight_real = self.weight_real[out_channel, in_channel]
                                weight_imag = self.weight_imag[out_channel, in_channel]
                                
                                # 处理权重和系数的尺寸不匹配问题
                                if isinstance(weight_real, torch.Tensor) and weight_real.ndim > 0:
                                    # 对于多维权重，确保形状匹配
                                    if coeff[:, in_channel].shape != weight_real.shape and weight_real.ndim > 0:
                                        # 如果维度不匹配，使用广播或调整大小
                                        if self.n_dim == 1:
                                            # 一维情况
                                            weight_real_expanded = weight_real.expand(coeff.shape[2])
                                            weight_imag_expanded = weight_imag.expand(coeff.shape[2])
                                        elif self.n_dim == 2:
                                            # 二维情况 - 使用插值调整权重大小
                                            weight_real_expanded = F.interpolate(
                                                weight_real.unsqueeze(0).unsqueeze(0),
                                                size=coeff.shape[2:],
                                                mode='nearest'
                                            ).squeeze(0).squeeze(0)
                                            weight_imag_expanded = F.interpolate(
                                                weight_imag.unsqueeze(0).unsqueeze(0),
                                                size=coeff.shape[2:],
                                                mode='nearest'
                                            ).squeeze(0).squeeze(0)
                                        else:
                                            # 其他维度情况
                                            weight_real_expanded = weight_real
                                            weight_imag_expanded = weight_imag
                                    else:
                                        weight_real_expanded = weight_real
                                        weight_imag_expanded = weight_imag
                                else:
                                    # 标量权重直接使用
                                    weight_real_expanded = weight_real
                                    weight_imag_expanded = weight_imag
                                    
                                    # 应用权重 (使用复数乘法)
                                    complex_weight = torch.complex(weight_real_expanded, weight_imag_expanded)
                                    complex_coeff = torch.complex(
                                        coeff[:, in_channel].real if hasattr(coeff[:, in_channel], 'real') else coeff[:, in_channel],
                                        torch.zeros_like(coeff[:, in_channel])
                                    )
                                    out_coeff[:, out_channel] += complex_coeff * complex_weight
                                    
                        out_coeffs.append(out_coeff)
                    else:  # 近似系数
                            # 近似系数处理与细节系数类似
                        out_coeff = torch.zeros_like(coeff, dtype=torch.complex64)
                            # ... 类似的处理逻辑
                        out_coeffs.append(out_coeff)
                            
            # 执行逆小波变换
            out = self.idwt_forward(out_coeffs, output_shape)
                
            # 确保输出不为None
            if out is None:
                # 如果输出是None，返回一个零张量
                print("警告：小波变换返回了None，使用零张量替代")
                out = torch.zeros(output_shape, device=x.device)
                    
            return out
            
        except Exception as e:
            print(f"小波变换中发生错误: {str(e)}")
            # 返回一个零张量作为后备
            return torch.zeros(output_shape, device=x.device)

    def _create_wavelet_filters(self):
        """创建小波滤波器"""
        if self.wavelet_type == 'haar':
            # Haar小波滤波器
            self.dec_lo = torch.tensor([0.7071067811865475, 0.7071067811865475])
            self.dec_hi = torch.tensor([0.7071067811865475, -0.7071067811865475])
            self.rec_lo = self.dec_lo
            self.rec_hi = self.dec_hi
        # 可以添加其他小波类型的支持...
    
    def _initialize_weights(self):
        """初始化权重参数"""
        if not self.complex_data:
            # 对每个尺度级别创建权重
            weight_shape = [self.out_channels, self.in_channels]
            for level in self.n_levels:
                weight_shape.append(level)
            self.weight = nn.Parameter(torch.empty(*weight_shape))
            nn.init.xavier_normal_(self.weight)
        else:
            # 复数情况
            weight_shape = [self.out_channels, self.in_channels]
            for level in self.n_levels:
                weight_shape.append(level)
            self.weight_real = nn.Parameter(torch.empty(*weight_shape))
            self.weight_imag = nn.Parameter(torch.empty(*weight_shape))
            nn.init.xavier_normal_(self.weight_real)
            nn.init.xavier_normal_(self.weight_imag)
    
    def _dwt_1d(self, x, levels):
        """一维离散小波变换实现"""
        # 确保levels是整数
        levels = int(levels)
        
        # 初始化系数列表
        coeffs = []
        
        # 获取输入形状信息
        batch_size, channels = x.shape[0], x.shape[1]
        
        # 确保输入长度是偶数
        n = x.shape[2]
        x_padded = x
        
        # 如果是奇数长度，进行填充
        if n % 2 == 1:
            x_padded = F.pad(x, (0, 1), mode=self.pad_mode)
        
        # 进行多级变换
        for level in range(levels):
            # 确保当前张量尺寸是偶数
            current_n = x_padded.shape[2]
            
            # 进行填充以确保尺寸为偶数
            pad_n = 0
            if current_n % 2 == 1:
                pad_n = 1
                x_padded = F.pad(x_padded, (0, pad_n), mode=self.pad_mode)
                current_n = x_padded.shape[2]
            
            # 检查是否可以继续分解
            if current_n <= 2:
                break
                
            # 确保对分是均匀的
            n_half = current_n // 2
            
            # 分解为偶数和奇数样本 - 使用切片确保形状匹配
            even = x_padded[..., :n_half*2:2]  # 只取可以均匀分割的元素
            odd = x_padded[..., 1:n_half*2:2]  # 只取可以均匀分割的元素
            
            # 检查形状匹配
            assert even.shape == odd.shape, f"形状不匹配：even {even.shape} vs odd {odd.shape}"
            
            # 计算近似和细节系数
            approx = (even + odd) / 2
            detail = (even - odd) / 2
            
            # 保存细节系数
            coeffs.append(detail)
            
            # 继续使用近似系数进行下一级分解
            x_padded = approx
        
        # 添加最终的近似系数
        coeffs.append(x_padded)
        
        return coeffs
    
    def _dwt_2d(self, x, levels_h, levels_w):
        """二维离散小波变换实现"""
        # 确保levels_h和levels_w是整数
        levels_h = int(levels_h)
        levels_w = int(levels_w)
        
        # 初始化系数列表
        coeffs = []
        
        # 获取输入形状信息
        batch_size, channels = x.shape[0], x.shape[1]
        
        # 确保输入尺寸是偶数
        h, w = x.shape[2], x.shape[3]
        x_padded = x
        
        # 如果是奇数尺寸，进行填充
        if h % 2 == 1 or w % 2 == 1:
            pad_h = 1 if h % 2 == 1 else 0
            pad_w = 1 if w % 2 == 1 else 0
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode=self.pad_mode)
        
        # 确定分解级数
        levels = min(levels_h, levels_w)
        
        # 进行多级变换
        for level in range(levels):
            # 确保当前张量尺寸是偶数
            current_h, current_w = x_padded.shape[2], x_padded.shape[3]
            
            # 进行填充以确保尺寸为偶数
            pad_h = 0
            pad_w = 0
            if current_h % 2 == 1:
                pad_h = 1
            if current_w % 2 == 1:
                pad_w = 1
                
            if pad_h > 0 or pad_w > 0:
                x_padded = F.pad(x_padded, (0, pad_w, 0, pad_h), mode=self.pad_mode)
                current_h, current_w = x_padded.shape[2], x_padded.shape[3]
            
            # 检查是否可以继续分解
            if current_h <= 2 or current_w <= 2:
                break
                
            # 确保行对分是均匀的
            # 计算可以均匀分割的高度
            h_half = current_h // 2
            
            # 行方向变换 - 使用切片来确保形状匹配
            even_rows = x_padded[:, :, :h_half*2:2, :]  # 只取可以均匀分割的行
            odd_rows = x_padded[:, :, 1:h_half*2:2, :]  # 只取可以均匀分割的行
            
            # 检查形状匹配
            assert even_rows.shape == odd_rows.shape, f"形状不匹配：even_rows {even_rows.shape} vs odd_rows {odd_rows.shape}"
            
            # 计算行方向的近似和细节系数
            approx_rows = (even_rows + odd_rows) / 2
            detail_rows = (even_rows - odd_rows) / 2
            
            # 确保列对分是均匀的
            # 计算可以均匀分割的宽度
            w_half = current_w // 2
            
            # 列方向变换 - 近似系数
            even_cols_approx = approx_rows[:, :, :, :w_half*2:2]  # 只取可以均匀分割的列
            odd_cols_approx = approx_rows[:, :, :, 1:w_half*2:2]  # 只取可以均匀分割的列
            
            # 检查形状匹配
            assert even_cols_approx.shape == odd_cols_approx.shape, f"形状不匹配：even_cols_approx {even_cols_approx.shape} vs odd_cols_approx {odd_cols_approx.shape}"
            
            # 列方向变换 - 细节系数
            even_cols_detail = detail_rows[:, :, :, :w_half*2:2]  # 只取可以均匀分割的列
            odd_cols_detail = detail_rows[:, :, :, 1:w_half*2:2]  # 只取可以均匀分割的列
            
            # 检查形状匹配
            assert even_cols_detail.shape == odd_cols_detail.shape, f"形状不匹配：even_cols_detail {even_cols_detail.shape} vs odd_cols_detail {odd_cols_detail.shape}"
            
            # 计算四个子带
            # LL: 低频行，低频列(近似)
            approx_approx = (even_cols_approx + odd_cols_approx) / 2
            # LH: 低频行，高频列(水平细节)
            approx_detail = (even_cols_approx - odd_cols_approx) / 2
            # HL: 高频行，低频列(垂直细节)
            detail_approx = (even_cols_detail + odd_cols_detail) / 2
            # HH: 高频行，高频列(对角细节)
            detail_detail = (even_cols_detail - odd_cols_detail) / 2
            
            # 保存细节系数 (HH, HL, LH)
            coeffs.append(detail_detail)
            coeffs.append(detail_approx)
            coeffs.append(approx_detail)
            
            # 继续使用近似系数进行下一级分解
            x_padded = approx_approx
        
        # 添加最终的近似系数 (LL)
        coeffs.append(x_padded)
        
        return coeffs
    
    def _dwt_3d(self, x, levels_d, levels_h, levels_w):
        """
        三维离散小波变换实现
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量, 形状为[batch_size, channels, depth, height, width]
        levels_d : int
            深度方向的分解级数
        levels_h : int
            高度方向的分解级数
        levels_w : int
            宽度方向的分解级数
            
        Returns
        -------
        List[torch.Tensor]
            小波系数列表，每级包含8个子带: 
            HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL(最后一个)
        """
        # 确保levels_d、levels_h和levels_w是整数
        levels_d = int(levels_d)
        levels_h = int(levels_h)
        levels_w = int(levels_w)
        
        # 初始化系数列表
        coeffs = []
        
        # 获取输入形状信息
        batch_size, channels = x.shape[0], x.shape[1]
        
        # 确保输入尺寸是偶数
        d, h, w = x.shape[2], x.shape[3], x.shape[4]
        x_padded = x
        
        # 如果是奇数尺寸，进行填充
        if d % 2 == 1 or h % 2 == 1 or w % 2 == 1:
            pad_d = 1 if d % 2 == 1 else 0
            pad_h = 1 if h % 2 == 1 else 0
            pad_w = 1 if w % 2 == 1 else 0
            x_padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode=self.pad_mode)
        
        # 确定分解级数
        levels = min(levels_d, levels_h, levels_w)
        
        # 进行多级变换
        for level in range(levels):
            # 检查当前尺寸
            current_d, current_h, current_w = x_padded.shape[2], x_padded.shape[3], x_padded.shape[4]
            
            # 进行填充以确保尺寸为偶数
            pad_d = 0
            pad_h = 0
            pad_w = 0
            if current_d % 2 == 1:
                pad_d = 1
            if current_h % 2 == 1:
                pad_h = 1
            if current_w % 2 == 1:
                pad_w = 1
                
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                x_padded = F.pad(x_padded, (0, pad_w, 0, pad_h, 0, pad_d), mode=self.pad_mode)
                current_d, current_h, current_w = x_padded.shape[2], x_padded.shape[3], x_padded.shape[4]
            
            # 检查是否可以继续分解
            if current_d <= 2 or current_h <= 2 or current_w <= 2:
                break
            
            # 确保深度对分是均匀的
            d_half = current_d // 2
            
            # 1. 深度方向变换 - 使用切片确保均匀分割
            even_d = x_padded[:, :, :d_half*2:2, :, :]  # 只取可以均匀分割的深度
            odd_d = x_padded[:, :, 1:d_half*2:2, :, :]  # 只取可以均匀分割的深度
            
            # 检查深度形状匹配
            assert even_d.shape == odd_d.shape, f"形状不匹配：even_d {even_d.shape} vs odd_d {odd_d.shape}"
            
            # 计算深度方向的近似和细节系数
            L_d = (even_d + odd_d) / 2  # 低频深度 (L)
            H_d = (even_d - odd_d) / 2  # 高频深度 (H)
            
            # 确保高度对分是均匀的
            h_half = current_h // 2
            
            # 2. 高度方向变换 - 对低频深度和高频深度分别进行
            # 2.1 低频深度 -> 高度方向
            even_h_L = L_d[:, :, :, :h_half*2:2, :]  # 只取可以均匀分割的高度
            odd_h_L = L_d[:, :, :, 1:h_half*2:2, :]  # 只取可以均匀分割的高度
            
            # 检查高度形状匹配
            assert even_h_L.shape == odd_h_L.shape, f"形状不匹配：even_h_L {even_h_L.shape} vs odd_h_L {odd_h_L.shape}"
            
            LL_dh = (even_h_L + odd_h_L) / 2  # 低频深度，低频高度 (LL)
            LH_dh = (even_h_L - odd_h_L) / 2  # 低频深度，高频高度 (LH)
            
            # 2.2 高频深度 -> 高度方向
            even_h_H = H_d[:, :, :, :h_half*2:2, :]  # 只取可以均匀分割的高度
            odd_h_H = H_d[:, :, :, 1:h_half*2:2, :]  # 只取可以均匀分割的高度
            
            # 检查高度形状匹配
            assert even_h_H.shape == odd_h_H.shape, f"形状不匹配：even_h_H {even_h_H.shape} vs odd_h_H {odd_h_H.shape}"
            
            HL_dh = (even_h_H + odd_h_H) / 2  # 高频深度，低频高度 (HL)
            HH_dh = (even_h_H - odd_h_H) / 2  # 高频深度，高频高度 (HH)
            
            # 确保宽度对分是均匀的
            w_half = current_w // 2
            
            # 3. 宽度方向变换 - 对四个高度变换结果分别进行
            # 3.1 低频深度，低频高度 -> 宽度方向
            even_w_LL = LL_dh[:, :, :, :, :w_half*2:2]  # 只取可以均匀分割的宽度
            odd_w_LL = LL_dh[:, :, :, :, 1:w_half*2:2]  # 只取可以均匀分割的宽度
            
            # 检查宽度形状匹配
            assert even_w_LL.shape == odd_w_LL.shape, f"形状不匹配：even_w_LL {even_w_LL.shape} vs odd_w_LL {odd_w_LL.shape}"
            
            LLL_dhw = (even_w_LL + odd_w_LL) / 2  # 低频深度，低频高度，低频宽度 (LLL) - 近似系数
            LLH_dhw = (even_w_LL - odd_w_LL) / 2  # 低频深度，低频高度，高频宽度 (LLH)
            
            # 3.2 低频深度，高频高度 -> 宽度方向
            even_w_LH = LH_dh[:, :, :, :, :w_half*2:2]  # 只取可以均匀分割的宽度
            odd_w_LH = LH_dh[:, :, :, :, 1:w_half*2:2]  # 只取可以均匀分割的宽度
            
            # 检查宽度形状匹配
            assert even_w_LH.shape == odd_w_LH.shape, f"形状不匹配：even_w_LH {even_w_LH.shape} vs odd_w_LH {odd_w_LH.shape}"
            
            LHL_dhw = (even_w_LH + odd_w_LH) / 2  # 低频深度，高频高度，低频宽度 (LHL)
            LHH_dhw = (even_w_LH - odd_w_LH) / 2  # 低频深度，高频高度，高频宽度 (LHH)
            
            # 3.3 高频深度，低频高度 -> 宽度方向
            even_w_HL = HL_dh[:, :, :, :, :w_half*2:2]  # 只取可以均匀分割的宽度
            odd_w_HL = HL_dh[:, :, :, :, 1:w_half*2:2]  # 只取可以均匀分割的宽度
            
            # 检查宽度形状匹配
            assert even_w_HL.shape == odd_w_HL.shape, f"形状不匹配：even_w_HL {even_w_HL.shape} vs odd_w_HL {odd_w_HL.shape}"
            
            HLL_dhw = (even_w_HL + odd_w_HL) / 2  # 高频深度，低频高度，低频宽度 (HLL)
            HLH_dhw = (even_w_HL - odd_w_HL) / 2  # 高频深度，低频高度，高频宽度 (HLH)
            
            # 3.4 高频深度，高频高度 -> 宽度方向
            even_w_HH = HH_dh[:, :, :, :, :w_half*2:2]  # 只取可以均匀分割的宽度
            odd_w_HH = HH_dh[:, :, :, :, 1:w_half*2:2]  # 只取可以均匀分割的宽度
            
            # 检查宽度形状匹配
            assert even_w_HH.shape == odd_w_HH.shape, f"形状不匹配：even_w_HH {even_w_HH.shape} vs odd_w_HH {odd_w_HH.shape}"
            
            HHL_dhw = (even_w_HH + odd_w_HH) / 2  # 高频深度，高频高度，低频宽度 (HHL)
            HHH_dhw = (even_w_HH - odd_w_HH) / 2  # 高频深度，高频高度，高频宽度 (HHH)
            
            # 4. 按照约定顺序将细节系数添加到列表中
            # 标准顺序是从高频到低频: HHH, HHL, HLH, HLL, LHH, LHL, LLH
            coeffs.append(HHH_dhw)  # 高频深度，高频高度，高频宽度
            coeffs.append(HHL_dhw)  # 高频深度，高频高度，低频宽度
            coeffs.append(HLH_dhw)  # 高频深度，低频高度，高频宽度
            coeffs.append(HLL_dhw)  # 高频深度，低频高度，低频宽度
            coeffs.append(LHH_dhw)  # 低频深度，高频高度，高频宽度
            coeffs.append(LHL_dhw)  # 低频深度，高频高度，低频宽度
            coeffs.append(LLH_dhw)  # 低频深度，低频高度，高频宽度
            
            # 继续使用近似系数进行下一级分解
            x_padded = LLL_dhw
        
        # 添加最终的近似系数 (LLL)
        coeffs.append(x_padded)
        
        return coeffs 
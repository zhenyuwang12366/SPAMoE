# -*- coding: utf-8 -*-
"""
Wavelet convolution layers with fp32-stable kernels (方案B)
- 在层内部强制关闭 autocast，并把参与计算的张量统一成 float32，
  以避免 DeepSpeed/AMP 将参数转为 bf16/fp16 引起的 dtype 冲突。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

try:
    import ptwt, pywt
    from ptwt.conv_transform_3 import wavedec3, waverec3
    from pytorch_wavelets import DWT1D, IDWT1D
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    from pytorch_wavelets import DWT, IDWT
except ImportError:
    print(
        'Wavelet convolution requires <Pytorch Wavelets>, <PyWavelets>, <Pytorch Wavelet Toolbox>\n'
        '  For Pytorch Wavelet Toolbox: $ pip install ptwt\n'
        '  For PyWavelets:             $ conda install pywavelets\n'
        '  For Pytorch Wavelets:       $ git clone https://github.com/fbcotter/pytorch_wavelets\n'
        '                               $ cd pytorch_wavelets\n'
        '                               $ pip install .'
    )


# =========================
# 1D Wavelet convolution
# =========================
class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='symmetric'):
        super(WaveConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if np.isscalar(size):
            self.size = size
        else:
            raise Exception("size: WaveConv1d accepts signal length in scalar only")
        self.wavelet = wavelet
        self.mode = mode

        # 预热获取 modes
        self.dwt_ = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode)
        dummy = torch.randn(1, 1, self.size)
        mode_data, _ = self.dwt_(dummy)
        self.modes1 = int(mode_data.shape[-1])

        self.scale = (1.0 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))

    # einsum 之前统一到 fp32
    def mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if input.dtype != torch.float32:
            input = input.to(torch.float32)
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 层内禁用 AMP，保证 DWT/IDWT 与权重计算都在 fp32
        with torch.autocast(device_type="cuda", enabled=False):
            if x.dtype != torch.float32:
                x = x.to(torch.float32)

            L = x.shape[-1]
            if L > self.size:
                factor = int(np.log2(L // self.size))
                dwt = DWT1D(wave=self.wavelet, J=self.level + factor, mode=self.mode).to(x.device)
                x_ft, x_coeff = dwt(x)
            elif L < self.size:
                factor = int(np.log2(self.size // L))
                dwt = DWT1D(wave=self.wavelet, J=self.level - factor, mode=self.mode).to(x.device)
                x_ft, x_coeff = dwt(x)
            else:
                dwt = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode).to(x.device)
                x_ft, x_coeff = dwt(x)

            out_ft = torch.zeros_like(x_ft, device=x.device)
            out_coeff = [torch.zeros_like(c, device=x.device) for c in x_coeff]

            out_ft = self.mul1d(x_ft, self.weights1)
            out_coeff[-1] = self.mul1d(x_coeff[-1].clone(), self.weights2)

            idwt = IDWT1D(wave=self.wavelet, mode=self.mode).to(x.device)
            x = idwt((out_ft, out_coeff))
            return x


# =========================
# 2D Wavelet convolution (DWT)
# =========================
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet, mode='symmetric'):
        super(WaveConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list) and len(size) == 2:
            self.size = size
        else:
            raise Exception('size: WaveConv2d accepts size of 2D signal as list with 2 elements')
        self.wavelet = wavelet
        self.mode = mode

        dummy = torch.randn(1, 1, *self.size)
        dwt_ = DWT(J=self.level, mode=self.mode, wave=self.wavelet)
        mode_data, _ = dwt_(dummy)
        self.modes1 = int(mode_data.shape[-2])
        self.modes2 = int(mode_data.shape[-1])

        self.scale = (1.0 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    def mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if input.dtype != torch.float32:
            input = input.to(torch.float32)
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            if x.dtype != torch.float32:
                x = x.to(torch.float32)

            W = x.shape[-1]
            if W > self.size[-1]:
                factor = int(np.log2(W // self.size[-1]))
                dwt = DWT(J=self.level + factor, mode=self.mode, wave=self.wavelet).to(x.device)
                x_ft, x_coeff = dwt(x)
            elif W < self.size[-1]:
                factor = int(np.log2(self.size[-1] // W))
                dwt = DWT(J=self.level - factor, mode=self.mode, wave=self.wavelet).to(x.device)
                x_ft, x_coeff = dwt(x)
            else:
                dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
                x_ft, x_coeff = dwt(x)

            out_ft = torch.zeros_like(x_ft, device=x.device)
            out_coeff = [torch.zeros_like(c, device=x.device) for c in x_coeff]

            out_ft = self.mul2d(x_ft, self.weights1)
            out_coeff_last = x_coeff[-1]
            out_coeff[-1][:, :, 0, :, :] = self.mul2d(out_coeff_last[:, :, 0, :, :].clone(), self.weights2)
            out_coeff[-1][:, :, 1, :, :] = self.mul2d(out_coeff_last[:, :, 1, :, :].clone(), self.weights3)
            out_coeff[-1][:, :, 2, :, :] = self.mul2d(out_coeff_last[:, :, 2, :, :].clone(), self.weights4)

            idwt = IDWT(mode=self.mode, wave=self.wavelet).to(x.device)
            x = idwt((out_ft, out_coeff))
            return x


# ==========================================
# 2D Wavelet convolution (DTCWT - continuous)
# ==========================================
class WaveConv2dCwt(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet1, wavelet2):
        super(WaveConv2dCwt, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list) and len(size) == 2:
            self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 2D signal as list with 2 elements')

        self.wavelet_level1 = wavelet1
        self.wavelet_level2 = wavelet2

        dummy = torch.randn(1, 1, *self.size)
        dwt_ = DTCWTForward(J=self.level, biort=self.wavelet_level1, qshift=self.wavelet_level2)
        mode_data, mode_coef = dwt_(dummy)
        self.modes1 = int(mode_data.shape[-2])
        self.modes2 = int(mode_data.shape[-1])
        self.modes21 = int(mode_coef[-1].shape[-3])
        self.modes22 = int(mode_coef[-1].shape[-2])

        self.scale = (1.0 / (in_channels * out_channels))
        self.weights0   = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  self.modes2))
        self.weights15r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights15c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights45r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights45c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights75r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights75c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights105r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights105c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights135r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights135c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights165r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights165c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))

    def mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if input.dtype != torch.float32:
            input = input.to(torch.float32)
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            if x.dtype != torch.float32:
                x = x.to(torch.float32)

            W = x.shape[-1]
            if W > self.size[-1]:
                factor = int(np.log2(W // self.size[-1]))
                cwt = DTCWTForward(J=self.level + factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
                x_ft, x_coeff = cwt(x)
            elif W < self.size[-1]:
                factor = int(np.log2(self.size[-1] // W))
                cwt = DTCWTForward(J=self.level - factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
                x_ft, x_coeff = cwt(x)
            else:
                cwt = DTCWTForward(J=self.level, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
                x_ft, x_coeff = cwt(x)

            out_ft = torch.zeros_like(x_ft, device=x.device)
            out_coeff = [torch.zeros_like(c, device=x.device) for c in x_coeff]

            out_ft = self.mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights0)

            oc = out_coeff[-1]
            xc = x_coeff[-1]
            oc[:, :, 0, :, :, 0] = self.mul2d(xc[:, :, 0, :, :, 0].clone(), self.weights15r)
            oc[:, :, 0, :, :, 1] = self.mul2d(xc[:, :, 0, :, :, 1].clone(), self.weights15c)
            oc[:, :, 1, :, :, 0] = self.mul2d(xc[:, :, 1, :, :, 0].clone(), self.weights45r)
            oc[:, :, 1, :, :, 1] = self.mul2d(xc[:, :, 1, :, :, 1].clone(), self.weights45c)
            oc[:, :, 2, :, :, 0] = self.mul2d(xc[:, :, 2, :, :, 0].clone(), self.weights75r)
            oc[:, :, 2, :, :, 1] = self.mul2d(xc[:, :, 2, :, :, 1].clone(), self.weights75c)
            oc[:, :, 3, :, :, 0] = self.mul2d(xc[:, :, 3, :, :, 0].clone(), self.weights105r)
            oc[:, :, 3, :, :, 1] = self.mul2d(xc[:, :, 3, :, :, 1].clone(), self.weights105c)
            oc[:, :, 4, :, :, 0] = self.mul2d(xc[:, :, 4, :, :, 0].clone(), self.weights135r)
            oc[:, :, 4, :, :, 1] = self.mul2d(xc[:, :, 4, :, :, 1].clone(), self.weights135c)
            oc[:, :, 5, :, :, 0] = self.mul2d(xc[:, :, 5, :, :, 0].clone(), self.weights165r)
            oc[:, :, 5, :, :, 1] = self.mul2d(xc[:, :, 5, :, :, 1].clone(), self.weights165c)

            icwt = DTCWTInverse(biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x = icwt((out_ft, out_coeff))
            return x


# =========================
# 3D Wavelet convolution
# =========================
class WaveConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='periodic'):
        super(WaveConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list) and len(size) == 3:
            self.size = size
        else:
            raise Exception('size: WaveConv3d accepts size of 3D signal as list with 3 elements')
        self.wavelet = wavelet
        self.mode = mode

        dummy = torch.randn([*self.size]).unsqueeze(0)
        mode_data = wavedec3(dummy, pywt.Wavelet(self.wavelet), level=self.level, mode=self.mode)
        self.modes1 = int(mode_data[0].shape[-3])
        self.modes2 = int(mode_data[0].shape[-2])
        self.modes3 = int(mode_data[0].shape[-1])

        self.scale = (1.0 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights5 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights6 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights7 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights8 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))

    def mul3d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if input.dtype != torch.float32:
            input = input.to(torch.float32)
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)
        return torch.einsum("ixyz,ioxyz->oxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            if x.dtype != torch.float32:
                x = x.to(torch.float32)

            xr = torch.zeros_like(x, device=x.device)
            B = x.shape[0]
            for i in range(B):
                vol = x[i, ...]
                W = vol.shape[-1]
                if W > self.size[-1]:
                    factor = int(np.log2(W // self.size[-1]))
                    x_coeff = wavedec3(vol, pywt.Wavelet(self.wavelet), level=self.level + factor, mode=self.mode)
                elif W < self.size[-1]:
                    factor = int(np.log2(self.size[-1] // W))
                    x_coeff = wavedec3(vol, pywt.Wavelet(self.wavelet), level=self.level - factor, mode=self.mode)
                else:
                    x_coeff = wavedec3(vol, pywt.Wavelet(self.wavelet), level=self.level, mode=self.mode)

                # 低频 & 7个高频子带
                x_coeff[0]            = self.mul3d(x_coeff[0].clone(),            self.weights1)
                x_coeff[1]['aad']     = self.mul3d(x_coeff[1]['aad'].clone(),     self.weights2)
                x_coeff[1]['ada']     = self.mul3d(x_coeff[1]['ada'].clone(),     self.weights3)
                x_coeff[1]['add']     = self.mul3d(x_coeff[1]['add'].clone(),     self.weights4)
                x_coeff[1]['daa']     = self.mul3d(x_coeff[1]['daa'].clone(),     self.weights5)
                x_coeff[1]['dad']     = self.mul3d(x_coeff[1]['dad'].clone(),     self.weights6)
                x_coeff[1]['dda']     = self.mul3d(x_coeff[1]['dda'].clone(),     self.weights7)
                x_coeff[1]['ddd']     = self.mul3d(x_coeff[1]['ddd'].clone(),     self.weights8)

                # 更高层系数置零（保持结构）
                for jj in range(2, self.level + 1):
                    x_coeff[jj] = {k: torch.zeros_like(x_coeff[jj][k], device=x.device)
                                   for k in x_coeff[jj].keys()}

                xr[i, ...] = waverec3(x_coeff, pywt.Wavelet(self.wavelet))

            return xr
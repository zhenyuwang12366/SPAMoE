import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Callable
import math

class LinearMix(nn.Module):
    """
    普通线性映射

    Args:
        input (torch.Tensor): 形状为B,k,1,h,w
    """
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
        self.linear = nn.LazyLinear(self.output_channels, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bsz, k, _, h, w = input.shape
        input = input.view(bsz, k, -1).permute(0, 2, 1).reshape(-1, k) # 把batch和空间合并的常见写法，方便其他统一操作
        output = self.linear(input) # b*h*w, 1
        output = output.view(bsz, h, w, self.output_channels).permute(0, 3, 1, 2)
        return output

class MeanMix(nn.Module):
    """默认均值聚合

    输入：b,k,1,h,w
    
    """
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bsz, k, _, h, w = input.shape
        output = input.mean(dim=1) # b, 1, h, w
        output = output.expand((bsz, self.output_channels, h, w))
        return output

class SumMix(nn.Module):
    """求和聚合

    输入：b,k,1,h,w
    """
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bsz, k, _, h, w = input.shape
        output = input.sum(dim=1)
        output = output.expand((bsz, self.output_channels, h, w))
        return output

class AttentionMix(nn.Module):
    """注意力聚合

    输入: b, k, 1, h, w
    """
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
        self.attn = None
        self.linear = nn.LazyLinear(self.output_channels, bias=False)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bsz, k, _, h, w = input.shape
        self.attn = nn.MultiheadAttention(k, num_heads = 4, batch_first=True)
        input = input.view(bsz, k, -1).permute(0, 2, 1).reshape(-1, k) # b*h*w, k
        output, _ = self.attn(
            input, input, input
        )
        output = self.linear(output) # b*h*w, c
        output = output.view(bsz, h, w, self.output_channels).permute(0, 3, 1, 2)
        
        return output
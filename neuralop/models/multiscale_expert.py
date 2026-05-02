import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional

from .base_model import BaseModel
from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import GridEmbeddingND
from ..layers.resample import resample


class MultiscaleExpert(BaseModel, name='MultiscaleExpert'):
    """
    多尺度专家模型
    
    该模型能够在多个尺度上处理输入，捕捉不同尺度的特征并融合它们。
    特别适合处理具有多尺度特性的物理系统。
    
    Parameters
    ----------
    base_model : nn.Module
        基础模型，将在不同尺度上应用
    in_channels : int
        输入通道数
    out_channels : int
        输出通道数
    hidden_channels : int
        隐藏层通道数
    n_scales : int, optional
        尺度数量，默认为3
    scale_factors : List[float], optional
        每个尺度的缩放因子，默认为[1.0, 0.5, 0.25]
    fusion_type : str, optional
        尺度融合方式，可选'adaptive'或'fixed'，默认为'adaptive'
    positional_embedding : str or nn.Module, optional
        位置嵌入类型，默认为'grid'
    """
    def __init__(
        self,
        base_model: nn.Module,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_scales: int = 3,
        scale_factors: List[float] = None,
        fusion_type: str = 'adaptive',
        positional_embedding: Union[str, nn.Module] = "grid",
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_scales = n_scales
        
        # 确定每个尺度的缩放因子
        if scale_factors is None:
            self.scale_factors = [1.0]
            for i in range(1, n_scales):
                self.scale_factors.append(self.scale_factors[-1] * 0.5)
        else:
            self.scale_factors = scale_factors[:n_scales]
            
        # 位置嵌入
        # 假设基础模型接受的输入维度与位置嵌入相同
        if hasattr(base_model, 'n_dim'):
            self.n_dim = base_model.n_dim
        else:
            # 默认为2D
            self.n_dim = 2
            
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0., 1.]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=self.n_dim,
                grid_boundaries=spatial_grid_boundaries
            )
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding is None:
            self.positional_embedding = None
        else:
            raise ValueError(f"无效的位置嵌入类型: {positional_embedding}")
        
        # 对每个尺度创建一个基础模型实例
        self.models = nn.ModuleList()
        for i in range(n_scales):
            # 克隆基础模型的参数
            if i == 0:
                # 第一个尺度使用原始模型
                self.models.append(base_model)
            else:
                # 创建基础模型的新实例
                self.models.append(type(base_model)(**{
                    k: v for k, v in base_model.__dict__.items() 
                    if not k.startswith('_') and not callable(v)
                }))
        
        # 尺度融合机制
        self.fusion_type = fusion_type
        if fusion_type == 'adaptive':
            # 自适应融合：学习不同尺度的权重
            self.fusion_weights = nn.Parameter(torch.ones(n_scales) / n_scales)
            
            # 尺度特征融合层
            self.fusion_layer = nn.Sequential(
                nn.Conv2d(out_channels * n_scales, hidden_channels, 1) 
                if self.n_dim == 2 else 
                nn.Conv3d(out_channels * n_scales, hidden_channels, 1) 
                if self.n_dim == 3 else
                nn.Conv1d(out_channels * n_scales, hidden_channels, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, out_channels, 1) 
                if self.n_dim == 2 else 
                nn.Conv3d(hidden_channels, out_channels, 1) 
                if self.n_dim == 3 else
                nn.Conv1d(hidden_channels, out_channels, 1)
            )
        elif fusion_type == 'fixed':
            # 固定权重融合
            self.register_buffer('fusion_weights', torch.ones(n_scales) / n_scales)
        else:
            raise ValueError(f"无效的融合类型: {fusion_type}")
            
    def forward(self, x, output_shape=None, **kwargs):
        """
        前向传播
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量
        output_shape : tuple, optional
            输出形状，默认为None
        
        Returns
        -------
        torch.Tensor
            输出张量
        """
        # 保存原始形状
        original_shape = x.shape
        
        # 应用位置嵌入
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        
        # 在不同尺度上处理输入
        scale_outputs = []
        for i, model in enumerate(self.models):
            # 缩放输入
            if self.scale_factors[i] != 1.0:
                # 计算缩放后的空间尺寸
                scale_shape = list(original_shape)
                for d in range(2, 2 + self.n_dim):
                    scale_shape[d] = int(scale_shape[d] * self.scale_factors[i])
                
                # 缩放输入
                scale_x = resample(x, scale_shape)
            else:
                scale_x = x
                
            # 应用模型
            scale_out = model(scale_x, **kwargs)
            
            # 如果需要，将输出调整回原始尺寸
            if self.scale_factors[i] != 1.0 and (output_shape is None or output_shape == original_shape):
                scale_out = resample(scale_out, original_shape)
            
            scale_outputs.append(scale_out)
        
        # 融合不同尺度的输出
        if self.fusion_type == 'adaptive':
            # 应用可学习的权重并融合
            norm_weights = F.softmax(self.fusion_weights, dim=0)
            
            # 将所有尺度的输出拼接在一起
            # 假设所有输出的空间尺寸相同
            concat_output = torch.cat(scale_outputs, dim=1)
            
            # 通过融合层
            output = self.fusion_layer(concat_output)
        else:  # 'fixed'
            # 使用固定权重融合
            output = sum(w * out for w, out in zip(self.fusion_weights, scale_outputs))
        
        return output 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional

from .base_model import BaseModel
from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import GridEmbeddingND
from ..layers.fno_block import FNOBlocks
from ..layers.spectral_convolution import SpectralConv
from ..layers.resample import resample
from ..layers.padding import DomainPadding


class MultiscaleNO(BaseModel, name='MultiscaleNO'):
    """
    多尺度神经算子
    
    这是一个原生支持多尺度特征提取和融合的神经算子模型。
    与简单地在不同尺度上应用相同模型不同，该模型在每个尺度上有专门设计的处理分支。
    
    Parameters
    ----------
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
    n_modes_list : List[Tuple[int, ...]], optional
        每个尺度上的傅里叶模式数，默认为None
    n_layers : int, optional
        每个尺度分支的层数，默认为4
    fusion_mode : str, optional
        尺度融合模式，可选'adaptive'或'hierarchical'，默认为'hierarchical'
    scale_connection : str, optional
        尺度间连接方式，可选'cascade'或'independent'，默认为'cascade'
    lifting_channels : int, optional
        提升层通道数，默认为None（使用2*hidden_channels）
    projection_channels : int, optional
        投影层通道数，默认为None（使用2*hidden_channels）
    positional_embedding : Union[str, nn.Module], optional
        位置嵌入类型，默认为'grid'
    non_linearity : callable, optional
        非线性激活函数，默认为F.gelu
    domain_padding : Union[float, List[float]], optional
        域填充比例，默认为None
    domain_padding_mode : str, optional
        域填充模式，默认为'symmetric'
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_scales: int = 3,
        scale_factors: List[float] = None,
        n_modes_list: List[Tuple[int, ...]] = None,
        n_layers: int = 4,
        fusion_mode: str = 'hierarchical',
        scale_connection: str = 'cascade',
        lifting_channels: int = None,
        projection_channels: int = None,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity = F.gelu,
        domain_padding: Union[float, List[float]] = None,
        domain_padding_mode: str = 'symmetric',
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_scales = n_scales
        self.fusion_mode = fusion_mode
        self.scale_connection = scale_connection
        self.non_linearity = non_linearity
        
        # 确定每个尺度的缩放因子
        if scale_factors is None:
            self.scale_factors = [1.0]
            for i in range(1, n_scales):
                self.scale_factors.append(self.scale_factors[-1] * 0.5)
        else:
            self.scale_factors = scale_factors[:n_scales]
        
        # 确定输入数据维度（假设为2D）
        self.n_dim = 2
        
        # 根据维度确定每个尺度的模式数
        if n_modes_list is None:
            # 默认设置：较低分辨率的尺度使用较少的模式
            if self.n_dim == 1:
                self.n_modes_list = [(min(16, int(32 * sf)),) for sf in self.scale_factors]
            elif self.n_dim == 2:
                self.n_modes_list = [(min(16, int(32 * sf)), min(16, int(32 * sf))) for sf in self.scale_factors]
            elif self.n_dim == 3:
                self.n_modes_list = [(min(8, int(16 * sf)), min(8, int(16 * sf)), min(8, int(16 * sf))) for sf in self.scale_factors]
        else:
            self.n_modes_list = n_modes_list[:n_scales]
        
        # 设置提升和投影通道数
        self.lifting_channels = lifting_channels if lifting_channels is not None else 4 * hidden_channels
        self.projection_channels = projection_channels if projection_channels is not None else 4 * hidden_channels
        
        # 位置嵌入
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
            
        # 如果使用位置嵌入，增加输入通道数
        if self.positional_embedding is not None:
            pos_emb_channels = self.n_dim
        else:
            pos_emb_channels = 0
        
        # 域填充
        if domain_padding is not None:
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode
            )
        else:
            self.domain_padding = None
        
        # 每个尺度的提升层 - 将输入映射到各自尺度的特征空间
        self.scale_liftings = nn.ModuleList([
            ChannelMLP(
                in_channels=in_channels + pos_emb_channels,
                out_channels=hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim
            ) for _ in range(n_scales)
        ])
        
        # 每个尺度的特征提取块
        self.scale_blocks = nn.ModuleList()
        
        for i in range(n_scales):
            # 每个尺度使用不同的傅里叶模式数
            self.scale_blocks.append(
                FNOBlocks(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    n_modes=self.n_modes_list[i],
                    n_layers=n_layers,
                    non_linearity=non_linearity,
                    **kwargs
                )
            )
        
        # 尺度融合机制
        if fusion_mode == 'adaptive':
            # 自适应融合：学习不同尺度的权重
            self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)
            
            # 尺度特征融合层
            self.fusion_layer = nn.Sequential(
                nn.Conv2d(hidden_channels * n_scales, hidden_channels, 1) 
                if self.n_dim == 2 else 
                nn.Conv3d(hidden_channels * n_scales, hidden_channels, 1) 
                if self.n_dim == 3 else
                nn.Conv1d(hidden_channels * n_scales, hidden_channels, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 1) 
                if self.n_dim == 2 else 
                nn.Conv3d(hidden_channels, hidden_channels, 1) 
                if self.n_dim == 3 else
                nn.Conv1d(hidden_channels, hidden_channels, 1)
            )
        elif fusion_mode == 'hierarchical':
            # 层次融合：从粗到细逐级融合
            self.scale_fusion = nn.ModuleList()
            
            for i in range(n_scales - 1):
                self.scale_fusion.append(
                    nn.Conv2d(hidden_channels * 2, hidden_channels, 1) 
                    if self.n_dim == 2 else 
                    nn.Conv3d(hidden_channels * 2, hidden_channels, 1) 
                    if self.n_dim == 3 else
                    nn.Conv1d(hidden_channels * 2, hidden_channels, 1)
                )
        else:
            raise ValueError(f"无效的融合模式: {fusion_mode}")
        
        # 投影层 - 将融合后的特征映射到输出空间
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim
        )
    
    def forward(self, x, output_shape=None, **kwargs):
        """
        前向传播
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状为 [batch_size, in_channels, ...]
        output_shape : tuple, optional
            期望的输出形状，默认为None
        
        Returns
        -------
        torch.Tensor
            输出张量，形状为 [batch_size, out_channels, ...]
        """
        # 保存原始形状
        original_shape = x.shape
        
        # 应用位置嵌入
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        
        # 应用域填充
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        
        # 处理每个尺度
        scale_features = []
        
        for i in range(self.n_scales):
            # 缩放输入到当前尺度
            if self.scale_factors[i] != 1.0:
                # 计算缩放后的空间尺寸
                scale_shape = list(x.shape)
                for d in range(2, 2 + self.n_dim):
                    scale_shape[d] = int(scale_shape[d] * self.scale_factors[i])
                
                # 缩放输入 - 使用缩放因子而不是目标形状
                spatial_dims = list(range(2, 2 + self.n_dim))
                scale_x = resample(x, self.scale_factors[i], axis=spatial_dims, output_shape=scale_shape[2:2+self.n_dim])
            else:
                scale_x = x
            
            # 应用当前尺度的提升层
            scale_x = self.scale_liftings[i](scale_x)
            
            # 应用当前尺度的特征提取块
            scale_x = self.scale_blocks[i](scale_x)
            
            # 保存当前尺度的特征
            scale_features.append(scale_x)
        
        # 融合不同尺度的特征
        if self.fusion_mode == 'adaptive':
            # 将所有尺度的特征调整为相同的空间尺寸（使用最高分辨率）
            aligned_features = []
            
            for i, feat in enumerate(scale_features):
                if self.scale_factors[i] != 1.0:
                    # 计算调整到第一个尺度（通常是最高分辨率）的缩放因子
                    target_shape = scale_features[0].shape[2:2+self.n_dim]
                    current_shape = feat.shape[2:2+self.n_dim]
                    scale_factors = [target_shape[j] / current_shape[j] for j in range(self.n_dim)]
                    
                    # 使用缩放因子进行重采样
                    spatial_dims = list(range(2, 2 + self.n_dim))
                    if len(scale_factors) == 1:
                        aligned_feat = resample(feat, scale_factors[0], axis=spatial_dims, output_shape=target_shape)
                    else:
                        aligned_feat = resample(feat, scale_factors, axis=spatial_dims, output_shape=target_shape)
                else:
                    aligned_feat = feat
                
                aligned_features.append(aligned_feat)
            
            # 应用可学习的权重
            norm_weights = F.softmax(self.scale_weights, dim=0)
            
            # 拼接特征
            concat_features = torch.cat(aligned_features, dim=1)
            
            # 通过融合层
            fused_features = self.fusion_layer(concat_features)
            
        elif self.fusion_mode == 'hierarchical':
            # 从最粗尺度开始，逐步融合到最细尺度
            # 先将所有特征调整到相应的尺度
            aligned_features = []
            
            for i, feat in enumerate(scale_features):
                # 从粗到细的顺序处理，所以这里需要倒序
                idx = self.n_scales - 1 - i
                
                if self.scale_factors[idx] != 1.0:
                    # 对于粗尺度，上采样到下一个较细的尺度
                    if idx < self.n_scales - 1:
                        # 计算目标形状和缩放因子
                        target_shape = scale_features[idx + 1].shape[2:2+self.n_dim]
                        current_shape = feat.shape[2:2+self.n_dim]
                        scale_factors = [target_shape[j] / current_shape[j] for j in range(self.n_dim)]
                        
                        # 使用缩放因子进行重采样
                        spatial_dims = list(range(2, 2 + self.n_dim))
                        if len(scale_factors) == 1:
                            aligned_feat = resample(feat, scale_factors[0], axis=spatial_dims, output_shape=target_shape)
                        else:
                            aligned_feat = resample(feat, scale_factors, axis=spatial_dims, output_shape=target_shape)
                    else:
                        # 最细尺度保持原样
                        aligned_feat = feat
                else:
                    aligned_feat = feat
                
                aligned_features.append(aligned_feat)
            
            # 逆序，从最粗到最细
            aligned_features = aligned_features[::-1]
            
            # 层次融合
            current = aligned_features[0]
            
            for i in range(self.n_scales - 1):
                # 确保空间维度匹配
                next_feat = aligned_features[i + 1]
                
                # 检查空间维度是否匹配
                if any(current.shape[2+j] != next_feat.shape[2+j] for j in range(self.n_dim)):
                    # 如果不匹配，将下一个特征调整到当前特征的形状
                    target_shape = current.shape[2:2+self.n_dim]
                    current_shape = next_feat.shape[2:2+self.n_dim]
                    scale_factors = [target_shape[j] / current_shape[j] for j in range(self.n_dim)]
                    
                    # 使用缩放因子进行重采样
                    spatial_dims = list(range(2, 2 + self.n_dim))
                    if len(scale_factors) == 1:
                        next_feat = resample(next_feat, scale_factors[0], axis=spatial_dims, output_shape=target_shape)
                    else:
                        next_feat = resample(next_feat, scale_factors, axis=spatial_dims, output_shape=target_shape)
                
                # 拼接当前特征和下一个尺度的特征
                concat = torch.cat([current, next_feat], dim=1)
                
                # 通过融合层
                current = self.scale_fusion[i](concat)
                current = self.non_linearity(current)
            
            fused_features = current
        
        # 应用投影层得到最终输出
        output = self.projection(fused_features)
        
        # 如果使用了域填充，需要裁剪回原始尺寸
        if self.domain_padding is not None:
            # 确定需要裁剪的形状
            if output_shape is not None:
                crop_shape = output_shape[2:]
            else:
                crop_shape = original_shape[2:]
                
            output = self.domain_padding.unpad(output, crop_shape)
        
        return output


class MultiscaleNO1d(MultiscaleNO):
    """1D多尺度神经算子"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_scales: int = 3,
        scale_factors: List[float] = None,
        n_modes_list: List[Tuple[int]] = None,
        n_layers: int = 4,
        fusion_mode: str = 'hierarchical',
        **kwargs
    ):
        # 设置默认的n_modes_list
        if n_modes_list is None:
            if scale_factors is None:
                scale_factors = [1.0, 0.5, 0.25][:n_scales]
            n_modes_list = [(min(32, int(32 * sf)),) for sf in scale_factors[:n_scales]]
        
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_scales=n_scales,
            scale_factors=scale_factors,
            n_modes_list=n_modes_list,
            n_layers=n_layers,
            fusion_mode=fusion_mode,
            **kwargs
        )
        self.n_dim = 1


class MultiscaleNO2d(MultiscaleNO):
    """2D多尺度神经算子"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_scales: int = 3,
        scale_factors: List[float] = None,
        n_modes_list: List[Tuple[int, int]] = None,
        n_layers: int = 4,
        fusion_mode: str = 'hierarchical',
        **kwargs
    ):
        # 设置默认的n_modes_list
        if n_modes_list is None:
            if scale_factors is None:
                scale_factors = [1.0, 0.5, 0.25][:n_scales]
            n_modes_list = [(min(16, int(32 * sf)), min(16, int(32 * sf))) for sf in scale_factors[:n_scales]]
        
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_scales=n_scales,
            scale_factors=scale_factors,
            n_modes_list=n_modes_list,
            n_layers=n_layers,
            fusion_mode=fusion_mode,
            **kwargs
        )
        self.n_dim = 2


class MultiscaleNO3d(MultiscaleNO):
    """3D多尺度神经算子"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_scales: int = 3,
        scale_factors: List[float] = None,
        n_modes_list: List[Tuple[int, int, int]] = None,
        n_layers: int = 4,
        fusion_mode: str = 'hierarchical',
        **kwargs
    ):
        # 设置默认的n_modes_list
        if n_modes_list is None:
            if scale_factors is None:
                scale_factors = [1.0, 0.5, 0.25][:n_scales]
            n_modes_list = [(min(8, int(16 * sf)), min(8, int(16 * sf)), min(8, int(16 * sf))) for sf in scale_factors[:n_scales]]
        
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_scales=n_scales,
            scale_factors=scale_factors,
            n_modes_list=n_modes_list,
            n_layers=n_layers,
            fusion_mode=fusion_mode,
            **kwargs
        )
        self.n_dim = 3 
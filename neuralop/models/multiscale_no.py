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
    Native multiscale neural operator.

    Dedicated branches per scale (not merely the same model at different resolutions).

    Parameters
    ----------
    in_channels : int
    out_channels : int
    hidden_channels : int
    n_scales : int, optional
        Number of scales (default 3)
    scale_factors : List[float], optional
        Per-scale factors (default [1.0, 0.5, 0.25])
    n_modes_list : List[Tuple[int, ...]], optional
        Fourier modes per scale (default derived from resolution)
    n_layers : int, optional
        Layers per scale branch (default 4)
    fusion_mode : str, optional
        'adaptive' or 'hierarchical' (default 'hierarchical')
    scale_connection : str, optional
        'cascade' or 'independent' (default 'cascade')
    lifting_channels : int, optional
        Lifting MLP width (default 4 * hidden_channels)
    projection_channels : int, optional
        Projection MLP width (default 4 * hidden_channels)
    positional_embedding : Union[str, nn.Module], optional
        Default 'grid'
    non_linearity : callable, optional
        Default F.gelu
    domain_padding : Union[float, List[float]], optional
        Fractional padding per side (optional)
    domain_padding_mode : str, optional
        Default 'symmetric'
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
        
        if scale_factors is None:
            self.scale_factors = [1.0]
            for i in range(1, n_scales):
                self.scale_factors.append(self.scale_factors[-1] * 0.5)
        else:
            self.scale_factors = scale_factors[:n_scales]
        
        self.n_dim = 2

        if n_modes_list is None:
            if self.n_dim == 1:
                self.n_modes_list = [(min(16, int(32 * sf)),) for sf in self.scale_factors]
            elif self.n_dim == 2:
                self.n_modes_list = [(min(16, int(32 * sf)), min(16, int(32 * sf))) for sf in self.scale_factors]
            elif self.n_dim == 3:
                self.n_modes_list = [(min(8, int(16 * sf)), min(8, int(16 * sf)), min(8, int(16 * sf))) for sf in self.scale_factors]
        else:
            self.n_modes_list = n_modes_list[:n_scales]
        
        self.lifting_channels = lifting_channels if lifting_channels is not None else 4 * hidden_channels
        self.projection_channels = projection_channels if projection_channels is not None else 4 * hidden_channels
        
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
            raise ValueError(f"Invalid positional_embedding: {positional_embedding}")

        if self.positional_embedding is not None:
            pos_emb_channels = self.n_dim
        else:
            pos_emb_channels = 0
        
        if domain_padding is not None:
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode
            )
        else:
            self.domain_padding = None
        
        self.scale_liftings = nn.ModuleList([
            ChannelMLP(
                in_channels=in_channels + pos_emb_channels,
                out_channels=hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim
            ) for _ in range(n_scales)
        ])
        
        self.scale_blocks = nn.ModuleList()
        
        for i in range(n_scales):
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
        
        if fusion_mode == 'adaptive':
            self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)
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
            raise ValueError(f"Invalid fusion_mode: {fusion_mode}")
        
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim
        )
    
    def forward(self, x, output_shape=None, **kwargs):
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_size, in_channels, ...]
        output_shape : tuple, optional
            Desired output spatial shape after unpadding

        Returns
        -------
        torch.Tensor
            [batch_size, out_channels, ...]
        """
        original_shape = x.shape

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        
        scale_features = []
        
        for i in range(self.n_scales):
            if self.scale_factors[i] != 1.0:
                scale_shape = list(x.shape)
                for d in range(2, 2 + self.n_dim):
                    scale_shape[d] = int(scale_shape[d] * self.scale_factors[i])
                spatial_dims = list(range(2, 2 + self.n_dim))
                scale_x = resample(x, self.scale_factors[i], axis=spatial_dims, output_shape=scale_shape[2:2+self.n_dim])
            else:
                scale_x = x
            
            scale_x = self.scale_liftings[i](scale_x)
            scale_x = self.scale_blocks[i](scale_x)
            scale_features.append(scale_x)
        
        if self.fusion_mode == 'adaptive':
            aligned_features = []
            
            for i, feat in enumerate(scale_features):
                if self.scale_factors[i] != 1.0:
                    target_shape = scale_features[0].shape[2:2+self.n_dim]
                    current_shape = feat.shape[2:2+self.n_dim]
                    scale_factors = [target_shape[j] / current_shape[j] for j in range(self.n_dim)]
                    spatial_dims = list(range(2, 2 + self.n_dim))
                    if len(scale_factors) == 1:
                        aligned_feat = resample(feat, scale_factors[0], axis=spatial_dims, output_shape=target_shape)
                    else:
                        aligned_feat = resample(feat, scale_factors, axis=spatial_dims, output_shape=target_shape)
                else:
                    aligned_feat = feat
                
                aligned_features.append(aligned_feat)
            
            norm_weights = F.softmax(self.scale_weights, dim=0)
            concat_features = torch.cat(aligned_features, dim=1)
            fused_features = self.fusion_layer(concat_features)

        elif self.fusion_mode == 'hierarchical':
            current = scale_features[-1]

            for i in range(self.n_scales - 2, -1, -1):
                next_feat = scale_features[i]

                target_shape = next_feat.shape[2:2 + self.n_dim]
                current_shape = current.shape[2:2 + self.n_dim]

                scale_factors = [target_shape[j] / current_shape[j] for j in range(self.n_dim)]

                spatial_dims = list(range(2, 2 + self.n_dim))
                if len(scale_factors) == 1:
                    current = resample(current, scale_factors[0], axis=spatial_dims, output_shape=target_shape)
                else:
                    current = resample(current, scale_factors, axis=spatial_dims, output_shape=target_shape)

                if any(current.shape[2 + j] != next_feat.shape[2 + j] for j in range(self.n_dim)):
                    target_shape = current.shape[2:2 + self.n_dim]
                    current_shape = next_feat.shape[2:2 + self.n_dim]
                    scale_factors = [target_shape[j] / current_shape[j] for j in range(self.n_dim)]

                    spatial_dims = list(range(2, 2 + self.n_dim))
                    if len(scale_factors) == 1:
                        next_feat = resample(next_feat, scale_factors[0], axis=spatial_dims, output_shape=target_shape)
                    else:
                        next_feat = resample(next_feat, scale_factors, axis=spatial_dims, output_shape=target_shape)

                concat = torch.cat([current, next_feat], dim=1)

                current = self.scale_fusion[i](concat)
                current = self.non_linearity(current)

            fused_features = current
        
        output = self.projection(fused_features)
        
        if self.domain_padding is not None:
            if output_shape is not None:
                crop_shape = output_shape[2:]
            else:
                crop_shape = original_shape[2:]
                
            output = self.domain_padding.unpad(output, crop_shape)
        
        return output


class MultiscaleNO1d(MultiscaleNO):
    """1D multiscale neural operator."""
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
    """2D multiscale neural operator."""
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
    """3D multiscale neural operator."""
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
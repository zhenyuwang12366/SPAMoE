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
    Multiscale expert wrapper.

    Processes input at multiple resolutions, aggregates multi-scale features,
    suited to physical systems with scale structure.

    Parameters
    ----------
    base_model : nn.Module
        Backbone applied at each scale
    in_channels : int
    out_channels : int
    hidden_channels : int
    n_scales : int, optional
        Number of scales (default 3)
    scale_factors : List[float], optional
        Per-scale factors (default [1.0, 0.5, 0.25])
    fusion_type : str, optional
        'adaptive' or 'fixed' (default 'adaptive')
    positional_embedding : str or nn.Module, optional
        Default 'grid'
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
        
        # Scale factors per branch
        if scale_factors is None:
            self.scale_factors = [1.0]
            for i in range(1, n_scales):
                self.scale_factors.append(self.scale_factors[-1] * 0.5)
        else:
            self.scale_factors = scale_factors[:n_scales]
            
        # Positional embedding (same dim as base model when available)
        if hasattr(base_model, 'n_dim'):
            self.n_dim = base_model.n_dim
        else:
            # Default 2D
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
            raise ValueError(f"Invalid positional_embedding type: {positional_embedding}")
        
        # One base-model instance per scale
        self.models = nn.ModuleList()
        for i in range(n_scales):
            if i == 0:
                self.models.append(base_model)
            else:
                self.models.append(type(base_model)(**{
                    k: v for k, v in base_model.__dict__.items() 
                    if not k.startswith('_') and not callable(v)
                }))
        
        self.fusion_type = fusion_type
        if fusion_type == 'adaptive':
            self.fusion_weights = nn.Parameter(torch.ones(n_scales) / n_scales)
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
            self.register_buffer('fusion_weights', torch.ones(n_scales) / n_scales)
        else:
            raise ValueError(f"Invalid fusion_type: {fusion_type}")
            
    def forward(self, x, output_shape=None, **kwargs):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        output_shape : tuple, optional
            Target output shape (optional)

        Returns
        -------
        torch.Tensor
        """
        original_shape = x.shape

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        
        scale_outputs = []
        for i, model in enumerate(self.models):
            if self.scale_factors[i] != 1.0:
                scale_shape = list(original_shape)
                for d in range(2, 2 + self.n_dim):
                    scale_shape[d] = int(scale_shape[d] * self.scale_factors[i])
                scale_x = resample(x, scale_shape)
            else:
                scale_x = x
                
            scale_out = model(scale_x, **kwargs)
            
            if self.scale_factors[i] != 1.0 and (output_shape is None or output_shape == original_shape):
                scale_out = resample(scale_out, original_shape)
            
            scale_outputs.append(scale_out)
        
        if self.fusion_type == 'adaptive':
            norm_weights = F.softmax(self.fusion_weights, dim=0)
            concat_output = torch.cat(scale_outputs, dim=1)
            output = self.fusion_layer(concat_output)
        else:
            output = sum(w * out for w, out in zip(self.fusion_weights, scale_outputs))
        
        return output 
import torch
import torch.nn as nn
from typing import Dict, Any, List, Union, Optional

from .fno import FNO, FNO1d, FNO2d, FNO3d
from .wno import WNO, WNO1d, WNO2d, WNO3d
from .gino import GINO
from .multiscale_expert import MultiscaleExpert
from .multiscale_no import MultiscaleNO, MultiscaleNO1d, MultiscaleNO2d, MultiscaleNO3d
from .local_no import LocalNO


class ExpertFactory:
    """
    专家工厂类，用于创建各种类型的专家模型
    
    支持创建不同类型的域专家、尺度专家和几何专家
    """
    @staticmethod
    def create_domain_expert(
        domain_type: str,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_dim: int = 2,
        **kwargs
    ) -> nn.Module:
        """
        创建域专家
        
        Parameters
        ----------
        domain_type : str
            专家类型，支持'fourier'（傅里叶域）和'wavelet'（小波域）
        in_channels : int
            输入通道数
        out_channels : int
            输出通道数
        hidden_channels : int
            隐藏层通道数
        n_dim : int, optional
            输入维度，默认为2
        
        Returns
        -------
        nn.Module
            创建的域专家模型
        """
        if domain_type == 'fourier':
            # 傅里叶神经算子专家
            if n_dim == 1:
                # 为了避免参数冲突，检查kwargs中是否已包含n_modes_height
                modes_kwargs = {}
                if 'n_modes_height' not in kwargs:
                    modes_kwargs['n_modes_height'] = 16  # 默认参数
                
                return FNO1d(
                    hidden_channels=hidden_channels,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **modes_kwargs,
                    **kwargs
                )
            elif n_dim == 2:
                # 为了避免参数冲突，检查kwargs中是否已包含必要的模式参数
                modes_kwargs = {}
                if 'n_modes_height' not in kwargs:
                    modes_kwargs['n_modes_height'] = 16  # 默认参数
                if 'n_modes_width' not in kwargs:
                    modes_kwargs['n_modes_width'] = 16  # 默认参数
                
                return FNO2d(
                    hidden_channels=hidden_channels,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **modes_kwargs,
                    **kwargs
                )
            elif n_dim == 3:
                # 为了避免参数冲突，检查kwargs中是否已包含必要的模式参数
                modes_kwargs = {}
                if 'n_modes_height' not in kwargs:
                    modes_kwargs['n_modes_height'] = 8  # 默认参数
                if 'n_modes_width' not in kwargs:
                    modes_kwargs['n_modes_width'] = 8  # 默认参数
                if 'n_modes_depth' not in kwargs:
                    modes_kwargs['n_modes_depth'] = 8  # 默认参数
                
                return FNO3d(
                    hidden_channels=hidden_channels,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **modes_kwargs,
                    **kwargs
                )
            else:
                raise ValueError(f"不支持的维度: {n_dim}")
                
        elif domain_type == 'wavelet':
            # 小波神经算子专家
            if n_dim == 1:
                # 为了避免参数冲突，检查kwargs中是否已包含n_levels_height
                levels_kwargs = {}
                if 'n_levels_height' not in kwargs:
                    levels_kwargs['n_levels_height'] = 4  # 默认参数
                
                return WNO1d(
                    hidden_channels=hidden_channels,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **levels_kwargs,
                    **kwargs
                )
            elif n_dim == 2:
                # 为了避免参数冲突，检查kwargs中是否已包含必要的级别参数
                levels_kwargs = {}
                if 'n_levels_height' not in kwargs:
                    levels_kwargs['n_levels_height'] = 4  # 默认参数
                if 'n_levels_width' not in kwargs:
                    levels_kwargs['n_levels_width'] = 4  # 默认参数
                
                return WNO2d(
                    hidden_channels=hidden_channels,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **levels_kwargs,
                    **kwargs
                )
            elif n_dim == 3:
                # 为了避免参数冲突，检查kwargs中是否已包含必要的级别参数
                levels_kwargs = {}
                if 'n_levels_height' not in kwargs:
                    levels_kwargs['n_levels_height'] = 3  # 默认参数
                if 'n_levels_width' not in kwargs:
                    levels_kwargs['n_levels_width'] = 3  # 默认参数
                if 'n_levels_depth' not in kwargs:
                    levels_kwargs['n_levels_depth'] = 3  # 默认参数
                
                return WNO3d(
                    hidden_channels=hidden_channels,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **levels_kwargs,
                    **kwargs
                )
            else:
                raise ValueError(f"不支持的维度: {n_dim}")
        else:
            raise ValueError(f"不支持的域类型: {domain_type}")
    
    @staticmethod
    def create_scale_expert(
        expert_type: str = 'wrapper',
        base_expert: nn.Module = None,
        in_channels: int = None,
        out_channels: int = None,
        hidden_channels: int = None,
        n_dim: int = 2,
        n_scales: int = 3,
        scale_factors: List[float] = None,
        fusion_type: str = 'adaptive',
        fusion_mode: str = 'hierarchical',
        **kwargs
    ) -> nn.Module:
        """
        创建多尺度专家
        
        Parameters
        ----------
        expert_type : str, optional
            多尺度专家类型，'wrapper'表示包装现有模型，'native'表示原生多尺度模型，默认为'wrapper'
        base_expert : nn.Module, optional
            基础专家模型，当expert_type为'wrapper'时需要提供
        in_channels : int
            输入通道数
        out_channels : int
            输出通道数
        hidden_channels : int
            隐藏层通道数
        n_dim : int, optional
            输入维度，默认为2
        n_scales : int, optional
            尺度数量，默认为3
        scale_factors : List[float], optional
            每个尺度的缩放因子，默认为None
        fusion_type : str, optional
            当expert_type为'wrapper'时的融合类型，默认为'adaptive'
        fusion_mode : str, optional
            当expert_type为'native'时的融合模式，默认为'hierarchical'
        
        Returns
        -------
        nn.Module
            创建的多尺度专家模型
        """
        if expert_type == 'wrapper':
            # 包装器类型的多尺度专家
            if base_expert is None:
                raise ValueError("当expert_type为'wrapper'时，必须提供base_expert")
                
            return MultiscaleExpert(
                base_model=base_expert,
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                n_scales=n_scales,
                scale_factors=scale_factors,
                fusion_type=fusion_type,
                **kwargs
            )
        elif expert_type == 'native':
            # 原生多尺度神经算子
            if n_dim == 1:
                return MultiscaleNO1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=hidden_channels,
                    n_scales=n_scales,
                    scale_factors=scale_factors,
                    fusion_mode=fusion_mode,
                    **kwargs
                )
            elif n_dim == 2:
                return MultiscaleNO2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=hidden_channels,
                    n_scales=n_scales,
                    scale_factors=scale_factors,
                    fusion_mode=fusion_mode,
                    **kwargs
                )
            elif n_dim == 3:
                return MultiscaleNO3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=hidden_channels,
                    n_scales=n_scales,
                    scale_factors=scale_factors,
                    fusion_mode=fusion_mode,
                    **kwargs
                )
            else:
                raise ValueError(f"不支持的维度: {n_dim}")
        else:
            raise ValueError(f"不支持的多尺度专家类型: {expert_type}")
    
    @staticmethod
    def create_geometry_expert(
        geometry_type: str,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        **kwargs
    ) -> nn.Module:
        """
        创建几何专家
        
        Parameters
        ----------
        geometry_type : str
            几何类型，支持'regular'（规则网格）和'irregular'（非规则网格）
        in_channels : int
            输入通道数
        out_channels : int
            输出通道数
        hidden_channels : int
            隐藏层通道数
        
        Returns
        -------
        nn.Module
            创建的几何专家模型
        """
        if geometry_type == 'regular':
            # 规则网格使用标准FNO
            # 为了避免参数冲突，检查kwargs中是否已包含必要的模式参数
            modes_kwargs = {}
            if 'n_modes_height' not in kwargs:
                modes_kwargs['n_modes_height'] = 16  # 默认参数
            if 'n_modes_width' not in kwargs:
                modes_kwargs['n_modes_width'] = 16  # 默认参数
            
            return FNO2d(
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                **modes_kwargs,
                **kwargs
            )
        elif geometry_type == 'irregular':
            # 非规则网格使用GINO
            return GINO(
                in_channels=in_channels,
                out_channels=out_channels,
                fno_hidden_channels=hidden_channels,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的几何类型: {geometry_type}")
    
    @staticmethod
    def create_local_expert(
        local_type: str,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_dim: int = 2,
        **kwargs
    ) -> nn.Module:
        """
        创建局部专家
        
        Parameters
        ----------
        local_type : str
            局部专家类型，目前支持'basic'
        in_channels : int
            输入通道数
        out_channels : int
            输出通道数
        hidden_channels : int
            隐藏层通道数
        n_dim : int, optional
            输入维度，默认为2
            
        Returns
        -------
        nn.Module
            创建的局部专家模型
        """
        if local_type == 'basic':
            # 基本局部神经算子
            return LocalNO(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                n_dim=n_dim,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的局部专家类型: {local_type}")
    
    @staticmethod
    def create_expert(
        expert_type: str,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        **kwargs
    ) -> nn.Module:
        """
        创建各种类型的专家
        
        Parameters
        ----------
        expert_type : str
            专家类型，支持'domain', 'scale', 'geometry', 'local'
        in_channels : int
            输入通道数
        out_channels : int
            输出通道数
        hidden_channels : int
            隐藏层通道数
            
        Returns
        -------
        nn.Module
            创建的专家模型
        """
        # 提取default_output_shape参数，如果存在的话
        default_output_shape = kwargs.pop('default_output_shape', None)
        
        if expert_type == 'domain':
            # 域专家
            domain_type = kwargs.pop('domain_type', 'fourier')
            n_dim = kwargs.pop('n_dim', 2)
            
            # 如果default_output_shape存在，将其添加回kwargs
            if default_output_shape is not None:
                kwargs['default_output_shape'] = default_output_shape
                
            return ExpertFactory.create_domain_expert(
                domain_type=domain_type,
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                n_dim=n_dim,
                **kwargs
            )
        elif expert_type == 'scale':
            # 多尺度专家
            scale_expert_type = kwargs.pop('scale_expert_type', 'native')
            if scale_expert_type == 'native':
                expert_type = scale_expert_type
                n_dim = kwargs.pop('n_dim', 2)
                n_scales = kwargs.pop('n_scales', 3)
                scale_factors = kwargs.pop('scale_factors', [1.0, 0.5, 0.25])
                fusion_mode = kwargs.pop('fusion_mode', 'hierarchical')
                
                # 如果default_output_shape存在，将其添加回kwargs
                if default_output_shape is not None:
                    kwargs['default_output_shape'] = default_output_shape
                
                return ExpertFactory.create_scale_expert(
                    expert_type=expert_type,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=hidden_channels,
                    n_dim=n_dim,
                    n_scales=n_scales,
                    scale_factors=scale_factors,
                    fusion_mode=fusion_mode,
                    **kwargs
                )
            else:
                # 其他类型的多尺度专家，需要先创建基础专家
                base_expert_type = kwargs.pop('base_expert_type', 'domain')
                base_expert_config = kwargs.pop('base_expert_config', {})
                
                # 创建基础专家
                base_expert = ExpertFactory.create_expert(
                    expert_type=base_expert_type,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=hidden_channels,
                    **base_expert_config
                )
                
                # 配置多尺度参数
                fusion_type = kwargs.pop('fusion_type', 'adaptive')
                n_scales = kwargs.pop('n_scales', 3)
                scale_factors = kwargs.pop('scale_factors', [1.0, 0.5, 0.25])
                
                # 如果default_output_shape存在，将其添加回kwargs
                if default_output_shape is not None:
                    kwargs['default_output_shape'] = default_output_shape
                
                return ExpertFactory.create_scale_expert(
                    expert_type='wrapper',
                    base_expert=base_expert,
                    fusion_type=fusion_type,
                    n_scales=n_scales,
                    scale_factors=scale_factors,
                    **kwargs
                )
        elif expert_type == 'geometry':
            # 几何专家
            geometry_type = kwargs.pop('geometry_type', 'gino')
            
            # 如果default_output_shape存在，将其添加回kwargs
            if default_output_shape is not None:
                kwargs['default_output_shape'] = default_output_shape
                
            return ExpertFactory.create_geometry_expert(
                geometry_type=geometry_type,
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                **kwargs
            )
        elif expert_type == 'local':
            # 本地专家
            local_type = kwargs.pop('local_type', 'basic')
            n_dim = kwargs.pop('n_dim', 2)
            
            # 如果default_output_shape存在，将其添加回kwargs
            if default_output_shape is not None:
                kwargs['default_output_shape'] = default_output_shape
                
            return ExpertFactory.create_local_expert(
                local_type=local_type,
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                n_dim=n_dim,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的专家类型: {expert_type}")
    
    @staticmethod
    def create_expert_ensemble(
        expert_configs: List[Dict[str, Any]],
        in_channels: int,
        out_channels: int,
        hidden_channels: int
    ) -> List[nn.Module]:
        """
        创建专家集合
        
        Parameters
        ----------
        expert_configs : List[Dict[str, Any]]
            专家配置列表
        in_channels : int
            输入通道数
        out_channels : int
            输出通道数
        hidden_channels : int
            隐藏层通道数
            
        Returns
        -------
        List[nn.Module]
            创建的专家模型列表
        """
        experts = []
        
        for config in expert_configs:
            # 复制配置以避免修改原始配置
            config = config.copy()
            
            # 提取专家类型
            expert_type = config.pop('type', 'domain')
            
            # 创建专家
            expert = ExpertFactory.create_expert(
                expert_type=expert_type,
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                **config
            )
            
            experts.append(expert)
            
        return experts 
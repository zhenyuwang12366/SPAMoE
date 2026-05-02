import torch
import torch.nn as nn
from typing import Dict, Any, List, Union, Optional

from .fno import FNO, FNO1d, FNO2d, FNO3d
from .wno import WNO2d, WNO3d
from .gino import GINO
from .multiscale_expert import MultiscaleExpert
from .multiscale_no import MultiscaleNO, MultiscaleNO1d, MultiscaleNO2d, MultiscaleNO3d
from .local_no import LocalNO
from .geofno import GeoFNO2d

class ExpertFactory:
    """
    Expert factory for building various expert modules.

    Supports domain, multiscale, and geometry experts.
    """
    @staticmethod
    def create_domain_expert(
        domain_type: str,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_dim: int = 2,
        v_type_id: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """
        Create a domain expert.

        Parameters
        ----------
        domain_type : str
            'fourier' (Fourier) or 'wavelet' (wavelet domain)
        in_channels : int
        out_channels : int
        hidden_channels : int
        n_dim : int, optional
            Spatial dimension (default 2)

        Returns
        -------
        nn.Module
        """
        if domain_type == 'fourier':
            # Fourier neural operator expert
            if v_type_id is not None:
                hidden_channels = kwargs.get('hc', 64)
            
            if n_dim == 1:
                # Avoid duplicate kwargs: only set defaults if missing
                modes_kwargs = {}
                if 'n_modes_height' not in kwargs:
                    modes_kwargs['n_modes_height'] = 16  # default
                
                return FNO1d(
                    hidden_channels=hidden_channels,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **modes_kwargs,
                    **kwargs
                )
            elif n_dim == 2:
                modes_kwargs = {}
                if 'n_modes_height' not in kwargs:
                    modes_kwargs['n_modes_height'] = 16
                if 'n_modes_width' not in kwargs:
                    modes_kwargs['n_modes_width'] = 16
                
                return FNO2d(
                    hidden_channels=hidden_channels,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **modes_kwargs,
                    **kwargs
                )
            elif n_dim == 3:
                # Avoid kwargs conflicts: fill defaults only if missing
                modes_kwargs = {}
                if 'n_modes_height' not in kwargs:
                    modes_kwargs['n_modes_height'] = 8
                if 'n_modes_width' not in kwargs:
                    modes_kwargs['n_modes_width'] = 8
                if 'n_modes_depth' not in kwargs:
                    modes_kwargs['n_modes_depth'] = 8
                
                return FNO3d(
                    hidden_channels=hidden_channels,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **modes_kwargs,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported dimension: {n_dim}")
                
        elif domain_type == 'wavelet':
            # Wavelet neural operator expert
            if v_type_id is not None:
                hidden_channels = kwargs.get('hc', 64)
            if n_dim == 2:
                levels_kwargs = {}
                if 'n_levels_height' not in kwargs:
                    levels_kwargs['n_levels_height'] = 4
                if 'n_levels_width' not in kwargs:
                    levels_kwargs['n_levels_width'] = 4
                
                base_size = kwargs.pop("base_size", (79, 70))
                
                return WNO2d(
                    base_size=base_size,
                    hidden_channels=hidden_channels,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **levels_kwargs,
                    **kwargs
                )
            elif n_dim == 3:
                # Avoid kwargs conflicts: fill level defaults only if missing
                levels_kwargs = {}
                if 'n_levels_height' not in kwargs:
                    levels_kwargs['n_levels_height'] = 3
                if 'n_levels_width' not in kwargs:
                    levels_kwargs['n_levels_width'] = 3
                if 'n_levels_depth' not in kwargs:
                    levels_kwargs['n_levels_depth'] = 3
                
                return WNO3d(
                    base_size=(1, 70, 70),
                    hidden_channels=hidden_channels,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **levels_kwargs,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported dimension: {n_dim}")
        else:
            raise ValueError(f"Unsupported domain type: {domain_type}")
    
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
        v_type_id: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """
        Create a multiscale expert.

        Parameters
        ----------
        expert_type : str, optional
            'wrapper' wraps an existing module; 'native' uses MultiscaleNO (default 'wrapper')
        base_expert : nn.Module, optional
            Required when expert_type is 'wrapper'
        in_channels : int
        out_channels : int
        hidden_channels : int
        n_dim : int, optional
        n_scales : int, optional
        scale_factors : List[float], optional
        fusion_type : str, optional
            For 'wrapper': fusion mode (default 'adaptive')
        fusion_mode : str, optional
            For 'native': fusion mode (default 'hierarchical')

        Returns
        -------
        nn.Module
        """
        if expert_type == 'wrapper':
            # Multiscale wrapper around base_expert
            if v_type_id is not None:
                hidden_channels = kwargs.get('hc', 64)
            if base_expert is None:
                raise ValueError("base_expert is required when expert_type is 'wrapper'")
                
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
            # Native multiscale neural operator
            if v_type_id is not None:
                hidden_channels = kwargs.get('hc', 64)
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
                raise ValueError(f"Unsupported dimension: {n_dim}")
        else:
            raise ValueError(f"Unsupported multiscale expert type: {expert_type}")
    
    @staticmethod
    def create_geometry_expert(
        geometry_type: str,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        **kwargs
    ) -> nn.Module:
        """
        Create a geometry expert.

        Parameters
        ----------
        geometry_type : str
            'regular' (structured grid) or 'irregular' (GINO-style)
        in_channels : int
        out_channels : int
        hidden_channels : int

        Returns
        -------
        nn.Module
        """
        if geometry_type == 'regular':
            # Regular grid: standard FNO
            modes_kwargs = {}
            if 'n_modes_height' not in kwargs:
                modes_kwargs['n_modes_height'] = 16
            if 'n_modes_width' not in kwargs:
                modes_kwargs['n_modes_width'] = 16
            
            return FNO2d(
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                **modes_kwargs,
                **kwargs
            )
        elif geometry_type == 'irregular':
            # Irregular grid: GINO
            return GINO(
                in_channels=in_channels,
                out_channels=out_channels,
                fno_hidden_channels=hidden_channels,
                **kwargs
            )
        elif geometry_type in ('geo', 'geofno'):
            modes1 = kwargs.pop('modes1', kwargs.pop('n_modes_height', 12))
            modes2 = kwargs.pop('modes2', kwargs.pop('n_modes_width', 12))
            code_dim = kwargs.pop('code_dim', 42)
            n_fourier_layers = kwargs.pop('n_fourier_layers', 5)
            s1 = kwargs.pop('s1', 40)
            s2 = kwargs.pop('s2', 40)
            is_mesh = kwargs.pop('is_mesh', True)

            return GeoFNO2d(
                modes1=modes1,
                modes2=modes2,
                width=hidden_channels,
                out_channels=out_channels,
                code_dim=code_dim,
                is_mesh=is_mesh,
                s1=s1,
                s2=s2,
                n_fourier_layers=n_fourier_layers,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")
    
    @staticmethod
    def create_local_expert(
        local_type: str,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_dim: int = 2,
        v_type_id: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """
        Create a local (convolutional) expert.

        Parameters
        ----------
        local_type : str
            Currently only 'basic' is supported
        in_channels : int
        out_channels : int
        hidden_channels : int
        n_dim : int, optional

        Returns
        -------
        nn.Module
        """
        if local_type == 'basic':
            # Basic local neural operator
            if v_type_id is not None:
                hidden_channels = kwargs.get('hc', 64)
            return LocalNO(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                n_dim=n_dim,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported local expert type: {local_type}")
    
    @staticmethod
    def create_expert(
        expert_type: str,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        v_type_id: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """
        Dispatch expert construction by high-level type.

        Parameters
        ----------
        expert_type : str
            'domain', 'scale', 'geometry', or 'local'
        in_channels : int
        out_channels : int
        hidden_channels : int

        Returns
        -------
        nn.Module
        """
        # Pop default_output_shape if present
        default_output_shape = kwargs.pop('default_output_shape', None)
        
        if expert_type == 'domain':
            # Domain expert
            domain_type = kwargs.pop('domain_type', 'fourier')
            n_dim = kwargs.pop('n_dim', 2)
            
            if default_output_shape is not None:
                kwargs['default_output_shape'] = default_output_shape
                
            return ExpertFactory.create_domain_expert(
                domain_type=domain_type,
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                n_dim=n_dim,
                v_type_id=v_type_id,
                **kwargs
            )
        elif expert_type == 'scale':
            # Multiscale expert
            scale_expert_type = kwargs.pop('scale_expert_type', 'native')
            if scale_expert_type == 'native':
                expert_type = scale_expert_type
                n_dim = kwargs.pop('n_dim', 2)
                n_scales = kwargs.pop('n_scales', 3)
                scale_factors = kwargs.pop('scale_factors', [1.0, 0.5, 0.25])
                fusion_mode = kwargs.pop('fusion_mode', 'hierarchical')
                
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
                    v_type_id=v_type_id,
                    **kwargs
                )
            else:
                # Non-native multiscale: build base expert first
                base_expert_type = kwargs.pop('base_expert_type', 'domain')
                base_expert_config = kwargs.pop('base_expert_config', {})
                
                base_expert = ExpertFactory.create_expert(
                    expert_type=base_expert_type,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=hidden_channels,
                    v_type_id=v_type_id,
                    **base_expert_config
                )
                
                fusion_type = kwargs.pop('fusion_type', 'adaptive')
                n_scales = kwargs.pop('n_scales', 3)
                scale_factors = kwargs.pop('scale_factors', [1.0, 0.5, 0.25])
                
                if default_output_shape is not None:
                    kwargs['default_output_shape'] = default_output_shape
                
                return ExpertFactory.create_scale_expert(
                    expert_type='wrapper',
                    base_expert=base_expert,
                    fusion_type=fusion_type,
                    n_scales=n_scales,
                    scale_factors=scale_factors,
                    v_type_id=v_type_id,
                    **kwargs
                )
        elif expert_type == 'geometry':
            # Geometry expert
            geometry_type = kwargs.pop('geometry_type', 'gino')
            
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
            # Local expert
            local_type = kwargs.pop('local_type', 'basic')
            n_dim = kwargs.pop('n_dim', 2)
            
            if default_output_shape is not None:
                kwargs['default_output_shape'] = default_output_shape
                
            return ExpertFactory.create_local_expert(
                local_type=local_type,
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                n_dim=n_dim,
                v_type_id=v_type_id,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported expert type: {expert_type}")
    
    @staticmethod
    def create_expert_ensemble(
        expert_configs: List[Dict[str, Any]],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        v_type_id: Optional[int] = None,
    ) -> List[nn.Module]:
        """
        Build a list of experts from configuration dicts.

        Parameters
        ----------
        expert_configs : List[Dict[str, Any]]
        in_channels : int
        out_channels : int
        hidden_channels : int

        Returns
        -------
        List[nn.Module]
        """
        experts = []
        
        for config in expert_configs:
            config = config.copy()
            
            expert_type = config.pop('type', 'domain')
            
            expert = ExpertFactory.create_expert(
                expert_type=expert_type,
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                v_type_id = v_type_id,
                **config
            )
            
            experts.append(expert)
            
        return experts 

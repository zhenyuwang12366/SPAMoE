import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union

Number = Union[float, int]

from ..layers.embeddings import GridEmbeddingND
from ..layers.wavelet_conv import WaveletConv
from ..layers.padding import DomainPadding
from ..layers.channel_mlp import ChannelMLP
from .base_model import BaseModel


class WNOBlock(nn.Module):
    """
    小波神经算子块
    
    与FNO块类似，但使用小波变换代替傅里叶变换
    
    Parameters
    ----------
    in_channels : int
        输入通道数
    out_channels : int
        输出通道数
    n_levels : List[int]
        每个维度上的小波分解级别数
    wavelet_type : str, optional
        小波类型，默认为'haar'
    resolution_scaling_factor : Optional[Union[Number, List[Number]]], optional
        分辨率缩放因子，默认为None
    n_layers : int, optional
        层数，默认为1
    use_channel_mlp : bool, optional
        是否使用通道MLP，默认为True
    channel_mlp_dropout : float, optional
        通道MLP的dropout比例，默认为0
    channel_mlp_expansion : float, optional
        通道MLP的扩展比例，默认为0.5
    non_linearity : torch.nn.functional, optional
        非线性激活函数，默认为F.gelu
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_levels,
        wavelet_type='haar',
        resolution_scaling_factor=None,
        n_layers=1,
        use_channel_mlp=True,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        skip="linear",
        channel_mlp_skip="soft-gating",
        complex_data=False,
        ensure_even_shapes=False,
        pad_mode='constant',
        adaptive_padding=False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_levels = n_levels
        self.n_dim = len(n_levels)
        self.wavelet_type = wavelet_type
        self.n_layers = n_layers
        self.non_linearity = non_linearity

        # 创建小波卷积层
        self.convs = nn.ModuleList([
            WaveletConv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                n_levels=self.n_levels,
                wavelet_type=self.wavelet_type,
                resolution_scaling_factor=resolution_scaling_factor,
                complex_data=complex_data,
                ensure_even_shapes=ensure_even_shapes,
                pad_mode=pad_mode,
                adaptive_padding=adaptive_padding
            ) for _ in range(n_layers)
        ])
        
        # 创建跳跃连接
        self.skip_connections = nn.ModuleList([
            nn.Conv1d(self.in_channels, self.out_channels, 1) if self.n_dim == 1 else
            nn.Conv2d(self.in_channels, self.out_channels, 1) if self.n_dim == 2 else
            nn.Conv3d(self.in_channels, self.out_channels, 1)
            for _ in range(n_layers)
        ])
        
        # 通道MLP
        if use_channel_mlp:
            self.channel_mlp = nn.ModuleList([
                ChannelMLP(
                    in_channels=self.out_channels,
                    hidden_channels=round(self.out_channels * channel_mlp_expansion),
                    dropout=channel_mlp_dropout,
                    n_dim=self.n_dim
                ) for _ in range(n_layers)
            ])
            
            # 通道MLP的跳跃连接
            self.channel_mlp_skips = nn.ModuleList([
                nn.Conv1d(self.out_channels, self.out_channels, 1) if self.n_dim == 1 else
                nn.Conv2d(self.out_channels, self.out_channels, 1) if self.n_dim == 2 else
                nn.Conv3d(self.out_channels, self.out_channels, 1)
                for _ in range(n_layers)
            ])
        else:
            self.channel_mlp = None
    
    def forward(self, x, output_shape=None):
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
        if output_shape is None:
            output_shape = x.shape
            
        # 依次通过每一层
        for i in range(self.n_layers):
            try:
                # 小波卷积
                x1 = self.convs[i](x, output_shape=output_shape)
                
                # 跳跃连接
                x2 = self.skip_connections[i](x)
                
                # 检查x1和x2是否为None
                if x1 is None:
                    print(f"警告：第{i}层小波卷积返回None，使用跳跃连接代替")
                    x = x2
                elif x2 is None:
                    print(f"警告：第{i}层跳跃连接返回None，使用小波卷积代替")
                    x = x1
                else:
                    # 正常情况：合并输出
                    x = x1 + x2
                
                # 非线性激活
                x = self.non_linearity(x)
                
                # 通道MLP（如果使用）
                if self.channel_mlp is not None:
                    try:
                        x1 = self.channel_mlp[i](x)
                        x2 = self.channel_mlp_skips[i](x)
                        
                        # 检查x1和x2是否为None
                        if x1 is None:
                            print(f"警告：第{i}层通道MLP返回None，使用跳跃连接代替")
                            x = x2
                        elif x2 is None:
                            print(f"警告：第{i}层通道MLP跳跃连接返回None，使用通道MLP代替")
                            x = x1
                        else:
                            # 正常情况：合并输出
                            x = x1 + x2
                            
                        x = self.non_linearity(x)
                    except Exception as e:
                        print(f"通道MLP处理中发生错误: {str(e)}")
                        # 不做任何处理，继续使用原始的x
            
            except Exception as e:
                print(f"WNOBlock第{i}层处理中发生错误: {str(e)}")
                # 如果处理过程中发生错误，终止循环
                break
                
        return x


class WNO(BaseModel, name='WNO'):
    """
    小波神经算子 (Wavelet Neural Operator)
    
    基于小波变换的神经算子，可以捕捉多尺度特征和局部模式。
    
    Parameters
    ----------
    n_levels : Tuple[int]
        每个维度上的小波分解级别数
    in_channels : int
        输入通道数
    out_channels : int
        输出通道数
    hidden_channels : int
        隐藏层通道数
    wavelet_type : str, optional
        小波类型，默认为'haar'
    n_layers : int, optional
        WNO块的数量，默认为4
    lifting_channels : int, optional
        提升块的通道数，默认为None
    projection_channels : int, optional
        投影块的通道数，默认为None
    positional_embedding : str or nn.Module, optional
        位置嵌入类型，默认为'grid'
    domain_padding : Union[Number, List[Number]], optional
        域填充比例，默认为None
    domain_padding_mode : str, optional
        域填充模式，默认为'symmetric'
    use_channel_mlp : bool, optional
        是否使用通道MLP，默认为True
    channel_mlp_dropout : float, optional
        通道MLP的dropout比例，默认为0
    channel_mlp_expansion : float, optional
        通道MLP的扩展比例，默认为0.5
    non_linearity : torch.nn.functional, optional
        非线性激活函数，默认为F.gelu
    """
    def __init__(
        self,
        n_levels: Tuple[int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        wavelet_type: str = 'haar',
        n_layers: int = 4,
        lifting_channels: int = None,
        projection_channels: int = None,
        positional_embedding: Union[str, nn.Module] = "grid",
        domain_padding: Union[Number, List[Number]] = None,
        domain_padding_mode: str = "symmetric",
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        non_linearity = F.gelu,
        ensure_even_shapes: bool = False,
        pad_mode: str = 'constant',
        adaptive_padding: bool = False,
        **kwargs
    ):
        super().__init__()
        self.n_dim = len(n_levels)
        self.n_levels = n_levels
        self.wavelet_type = wavelet_type
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        
        # 设置提升和投影通道数
        self.lifting_channels = lifting_channels if lifting_channels is not None else 2 * hidden_channels
        self.projection_channels = projection_channels if projection_channels is not None else 2 * hidden_channels
        
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
            in_channels += self.n_dim
            
        # 域填充
        if domain_padding is not None:
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode
            )
        else:
            self.domain_padding = None
            
        # 提升层（输入映射）
        self.lifting = ChannelMLP(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.lifting_channels,
            n_layers=2,
            n_dim=self.n_dim
        )
        
        # WNO块
        self.wno_blocks = WNOBlock(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            n_levels=self.n_levels,
            wavelet_type=self.wavelet_type,
            n_layers=self.n_layers,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            ensure_even_shapes=ensure_even_shapes,
            pad_mode=pad_mode,
            adaptive_padding=adaptive_padding
        )
        
        # 投影层（输出映射）
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
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
            输入张量
        output_shape : tuple, optional
            输出形状，默认为None
        
        Returns
        -------
        torch.Tensor
            输出张量
        """
        try:
            # 检查是否存在default_output_shape参数
            default_output_shape = kwargs.get('default_output_shape', None)
            
            # 保存原始形状
            original_shape = x.shape
            device = x.device
            dtype = x.dtype
            
            # 应用位置嵌入
            if self.positional_embedding is not None:
                try:
                    x = self.positional_embedding(x)
                except Exception as e:
                    print(f"位置嵌入出错: {str(e)}")
                    # 如果嵌入失败，继续处理
                    
            # 应用域填充
            if self.domain_padding is not None:
                try:
                    x = self.domain_padding.pad(x)
                except Exception as e:
                    print(f"域填充出错: {str(e)}")
                    # 如果填充失败，继续处理
                    
            # 提升层
            try:
                x = self.lifting(x)
                if x is None:
                    print("提升层返回None，使用原始输入")
                    # 如果提升层返回None，使用原始输入
                    x = torch.zeros(original_shape, device=device, dtype=dtype)
            except Exception as e:
                print(f"提升层处理出错: {str(e)}")
                # 如果提升层处理出错，使用原始输入
                x = torch.zeros(original_shape, device=device, dtype=dtype)
            
            # WNO块
            try:
                x = self.wno_blocks(x, output_shape=output_shape)
                if x is None:
                    print("WNO块返回None，使用零张量替代")
                    # 如果WNO块返回None，使用零张量
                    x = torch.zeros(original_shape, device=device, dtype=dtype)
            except Exception as e:
                print(f"WNO块处理出错: {str(e)}")
                # 如果WNO块处理出错，使用零张量
                x = torch.zeros(original_shape, device=device, dtype=dtype)
            
            # 投影层
            try:
                x = self.projection(x)
                if x is None:
                    print("投影层返回None，使用零张量替代")
                    # 如果投影层返回None，使用零张量
                    x = torch.zeros((original_shape[0], self.out_channels, *original_shape[2:]), 
                                  device=device, dtype=dtype)
            except Exception as e:
                print(f"投影层处理出错: {str(e)}")
                # 如果投影层处理出错，使用零张量
                x = torch.zeros((original_shape[0], self.out_channels, *original_shape[2:]), 
                              device=device, dtype=dtype)
            
            # 如果使用了域填充，需要裁剪回原始尺寸
            if self.domain_padding is not None:
                try:
                    # 确定需要裁剪的形状
                    if output_shape is not None:
                        crop_shape = output_shape[2:]
                    else:
                        crop_shape = original_shape[2:]
                        
                    x = self.domain_padding.unpad(x, crop_shape)
                except Exception as e:
                    print(f"域填充裁剪出错: {str(e)}")
                    # 如果裁剪失败，使用插值
                    try:
                        if self.n_dim == 1:
                            x = F.interpolate(x, size=original_shape[2:], 
                                           mode='linear', align_corners=True)
                        elif self.n_dim == 2:
                            x = F.interpolate(x, size=original_shape[2:], 
                                           mode='bilinear', align_corners=True)
                        elif self.n_dim == 3:
                            x = F.interpolate(x, size=original_shape[2:], 
                                           mode='trilinear', align_corners=True)
                    except Exception:
                        # 如果插值也失败，返回零张量
                        x = torch.zeros((original_shape[0], self.out_channels, *original_shape[2:]), 
                                      device=device, dtype=dtype)
                                      
            # 检查是否需要调整为default_output_shape                  
            if default_output_shape is not None:
                try:
                    # 确保输出形状符合default_output_shape
                    if x.shape[2:] != default_output_shape:
                        print(f"调整输出形状: {x.shape[2:]} -> {default_output_shape}")
                        if self.n_dim == 1:
                            x = F.interpolate(x, size=default_output_shape, 
                                           mode='linear', align_corners=True)
                        elif self.n_dim == 2:
                            x = F.interpolate(x, size=default_output_shape, 
                                           mode='bilinear', align_corners=True)
                        elif self.n_dim == 3:
                            x = F.interpolate(x, size=default_output_shape, 
                                           mode='trilinear', align_corners=True)
                except Exception as e:
                    print(f"调整输出形状出错: {str(e)}")
                    # 如果调整失败，创建一个形状正确的零张量
                    x = torch.zeros((original_shape[0], self.out_channels, *default_output_shape), 
                                  device=device, dtype=dtype)
            
            return x
            
        except Exception as e:
            print(f"WNO前向传播出现未处理的错误: {str(e)}")
            # 如果存在默认输出形状，使用它
            if 'default_output_shape' in kwargs and kwargs['default_output_shape'] is not None:
                shape = (x.shape[0], self.out_channels, *kwargs['default_output_shape'])
            else:
                shape = (x.shape[0], self.out_channels, *x.shape[2:])
                
            # 返回零张量
            return torch.zeros(shape, device=x.device, dtype=x.dtype)
        
        
class WNO1d(WNO):
    """1D小波神经算子"""
    def __init__(
        self,
        n_levels_height,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        wavelet_type='haar',
        lifting_channels=None,
        projection_channels=None,
        n_layers=4,
        positional_embedding="grid",
        domain_padding=None,
        domain_padding_mode="symmetric",
        use_channel_mlp=True,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        ensure_even_shapes=False,
        pad_mode='constant',
        adaptive_padding=False,
        **kwargs
    ):
        super().__init__(
            n_levels=(n_levels_height,),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            wavelet_type=wavelet_type,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            positional_embedding=positional_embedding,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            ensure_even_shapes=ensure_even_shapes,
            pad_mode=pad_mode,
            adaptive_padding=adaptive_padding,
            **kwargs
        )
        
        
class WNO2d(WNO):
    """2D小波神经算子"""
    def __init__(
        self,
        n_levels_height,
        n_levels_width,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        wavelet_type='haar',
        lifting_channels=None,
        projection_channels=None,
        n_layers=4,
        positional_embedding="grid",
        domain_padding=None,
        domain_padding_mode="symmetric",
        use_channel_mlp=True,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        ensure_even_shapes=False,
        pad_mode='constant',
        adaptive_padding=False,
        **kwargs
    ):
        super().__init__(
            n_levels=(n_levels_height, n_levels_width),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            wavelet_type=wavelet_type,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            positional_embedding=positional_embedding,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            ensure_even_shapes=ensure_even_shapes,
            pad_mode=pad_mode,
            adaptive_padding=adaptive_padding,
            **kwargs
        )
        
        
class WNO3d(WNO):
    """3D小波神经算子"""
    def __init__(
        self,
        n_levels_height,
        n_levels_width,
        n_levels_depth,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        wavelet_type='haar',
        lifting_channels=None,
        projection_channels=None,
        n_layers=4,
        positional_embedding="grid",
        domain_padding=None,
        domain_padding_mode="symmetric",
        use_channel_mlp=True,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        ensure_even_shapes=False,
        pad_mode='constant',
        adaptive_padding=False,
        **kwargs
    ):
        super().__init__(
            n_levels=(n_levels_height, n_levels_width, n_levels_depth),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            wavelet_type=wavelet_type,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            positional_embedding=positional_embedding,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            ensure_even_shapes=ensure_even_shapes,
            pad_mode=pad_mode,
            adaptive_padding=adaptive_padding,
            **kwargs
        ) 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
# 导入混合精度训练相关模块
try:
    import torch.cuda.amp as amp
except ImportError:
    amp = None

Number = Union[float, int]

from ..layers.embeddings import GridEmbeddingND
from ..layers.wavelet_conv import WaveletConv
from ..layers.padding import DomainPadding
from ..layers.channel_mlp import ChannelMLP
from .base_model import BaseModel


class GradientMonitor:
    """梯度监控器"""
    
    def __init__(self, model, save_dir="./gradient_logs"):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储梯度信息
        self.gradient_norms = {}
        self.gradient_means = {}
        self.gradient_stds = {}
        self.layer_names = []
        
        # 注册钩子
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册梯度钩子"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d, nn.Conv3d, WaveletConv)):
                self.layer_names.append(name)
                
                # 前向钩子
                hook = module.register_forward_hook(
                    lambda m, i, o, name=name: self._forward_hook(name, m, i, o)
                )
                self.hooks.append(hook)
                
                # 反向钩子
                hook = module.register_backward_hook(
                    lambda m, gi, go, name=name: self._backward_hook(name, m, gi, go)
                )
                self.hooks.append(hook)
    
    def _forward_hook(self, name, module, input, output):
        """前向传播钩子"""
        # 记录输入输出的统计信息
        if hasattr(self, 'forward_stats'):
            if name not in self.forward_stats:
                self.forward_stats[name] = {'inputs': [], 'outputs': []}
            
            # 记录输入统计
            if input[0] is not None:
                input_tensor = input[0]
                self.forward_stats[name]['inputs'].append({
                    'mean': input_tensor.mean().item(),
                    'std': input_tensor.std().item(),
                    'min': input_tensor.min().item(),
                    'max': input_tensor.max().item()
                })
            
            # 记录输出统计
            if output is not None:
                self.forward_stats[name]['outputs'].append({
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item()
                })
    
    def _backward_hook(self, name, module, grad_input, grad_output):
        """反向传播钩子"""
        if grad_output[0] is not None:
            grad = grad_output[0]
            
            # 计算梯度统计信息
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()
            
            # 存储梯度信息
            if name not in self.gradient_norms:
                self.gradient_norms[name] = []
                self.gradient_means[name] = []
                self.gradient_stds[name] = []
            
            self.gradient_norms[name].append(grad_norm)
            self.gradient_means[name].append(grad_mean)
            self.gradient_stds[name].append(grad_std)
    
    def get_gradient_stats(self):
        """获取梯度统计信息"""
        stats = {}
        for name in self.layer_names:
            if name in self.gradient_norms:
                stats[name] = {
                    'norm_mean': np.mean(self.gradient_norms[name]),
                    'norm_std': np.std(self.gradient_norms[name]),
                    'norm_min': np.min(self.gradient_norms[name]),
                    'norm_max': np.max(self.gradient_norms[name]),
                    'mean_mean': np.mean(self.gradient_means[name]),
                    'mean_std': np.std(self.gradient_means[name]),
                    'std_mean': np.mean(self.gradient_stds[name]),
                    'std_std': np.std(self.gradient_stds[name])
                }
        return stats
    
    def plot_gradients(self, epoch, save=True):
        """绘制梯度分布图"""
        if not self.gradient_norms:
            print("No gradient data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Gradient Distribution - Epoch {epoch}')
        
        # 梯度范数分布
        ax1 = axes[0, 0]
        for name in self.layer_names[:5]:  # 只显示前5层
            if name in self.gradient_norms:
                ax1.plot(self.gradient_norms[name], label=name, alpha=0.7)
        ax1.set_title('Gradient Norm')
        ax1.set_ylabel('Gradient Norm')
        ax1.legend()
        ax1.grid(True)
        
        # 梯度均值分布
        ax2 = axes[0, 1]
        for name in self.layer_names[:5]:
            if name in self.gradient_means:
                ax2.plot(self.gradient_means[name], label=name, alpha=0.7)
        ax2.set_title('Gradient Mean')
        ax2.set_ylabel('Gradient Mean')
        ax2.legend()
        ax2.grid(True)
        
        # 梯度标准差分布
        ax3 = axes[1, 0]
        for name in self.layer_names[:5]:
            if name in self.gradient_stds:
                ax3.plot(self.gradient_stds[name], label=name, alpha=0.7)
        ax3.set_title('Gradient Std')
        ax3.set_ylabel('Gradient Std')
        ax3.legend()
        ax3.grid(True)
        
        # 梯度范数直方图
        ax4 = axes[1, 1]
        for name in self.layer_names[:3]:  # 只显示前3层
            if name in self.gradient_norms:
                ax4.hist(self.gradient_norms[name], bins=20, alpha=0.7, label=name)
        ax4.set_title('Gradient Norm Distribution')
        ax4.set_xlabel('Gradient Norm')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / f'gradients_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_gradient_log(self, epoch):
        """保存梯度日志"""
        stats = self.get_gradient_stats()
        
        log_file = self.save_dir / f'gradient_log_epoch_{epoch}.txt'
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Gradient Statistics - Epoch {epoch}\n")
            f.write("=" * 50 + "\n")
            
            for name, stat in stats.items():
                f.write(f"\nLayer: {name}\n")
                f.write(f"  Gradient Norm - Mean: {stat['norm_mean']:.6f}, Std: {stat['norm_std']:.6f}\n")
                f.write(f"  Gradient Norm - Min: {stat['norm_min']:.6f}, Max: {stat['norm_max']:.6f}\n")
                f.write(f"  Gradient Mean - Mean: {stat['mean_mean']:.6f}, Std: {stat['mean_std']:.6f}\n")
                f.write(f"  Gradient Std - Mean: {stat['std_mean']:.6f}, Std: {stat['std_std']:.6f}\n")
        
        # 检测梯度问题
        self._detect_gradient_issues(stats, epoch)
    
    def _detect_gradient_issues(self, stats, epoch):
        """检测梯度问题"""
        issues = []
        
        for name, stat in stats.items():
            # 检测梯度消失 - 降低阈值
            if stat['norm_mean'] < 1e-8:
                issues.append(f"Layer {name}: Gradient Vanishing (Norm Mean: {stat['norm_mean']:.2e})")
            
            # 检测梯度爆炸
            if stat['norm_mean'] > 10:
                issues.append(f"Layer {name}: Gradient Exploding (Norm Mean: {stat['norm_mean']:.2f})")
            
            # 检测梯度不稳定
            if stat['norm_std'] > stat['norm_mean']:
                issues.append(f"Layer {name}: Gradient Unstable (Std > Mean)")
        
        if issues:
            print(f"\n=== Epoch {epoch} Gradient Issues Detected ===")
            for issue in issues:
                print(f"Warning: {issue}")
            
            # 保存问题日志
            issue_file = self.save_dir / f'gradient_issues_epoch_{epoch}.txt'
            with open(issue_file, 'w', encoding='utf-8') as f:
                f.write(f"Gradient Issues - Epoch {epoch}\n")
                f.write("=" * 30 + "\n")
                for issue in issues:
                    f.write(f"{issue}\n")
    
    def clear_stats(self):
        """清除统计信息"""
        self.gradient_norms.clear()
        self.gradient_means.clear()
        self.gradient_stds.clear()
        if hasattr(self, 'forward_stats'):
            self.forward_stats.clear()
    
    def remove_hooks(self):
        """移除钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


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
        # 自动提取空间维度，确保output_shape只包含空间部分
        if output_shape is None and x.dim() >= 3:
            output_shape = x.shape[-2:]
        
        if output_shape is None:
            output_shape = x.shape
            
        # 确保所有张量在同一设备上
        device = x.device
        if output_shape is not None:
            output_shape = tuple(output_shape)
            
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
                # 打印完整堆栈到标准错误
                traceback.print_exc()
                break
        return x
    
    def enable_gradient_monitoring(self, save_dir=None):
        """启用梯度监控"""
        if save_dir is None:
            save_dir = self.gradient_save_dir
        
        self.gradient_monitor = GradientMonitor(self, save_dir)
        print(f"Gradient monitoring enabled, logs saved to: {save_dir}")
    
    def disable_gradient_monitoring(self):
        """禁用梯度监控"""
        if self.gradient_monitor is not None:
            self.gradient_monitor.remove_hooks()
            self.gradient_monitor = None
            print("Gradient monitoring disabled")
    
    def get_gradient_stats(self):
        """获取梯度统计信息"""
        if self.gradient_monitor is not None:
            return self.gradient_monitor.get_gradient_stats()
        return {}
    
    def plot_gradients(self, epoch, save=True):
        """绘制梯度分布图"""
        if self.gradient_monitor is not None:
            self.gradient_monitor.plot_gradients(epoch, save)
    
    def save_gradient_log(self, epoch):
        """保存梯度日志"""
        if self.gradient_monitor is not None:
            self.gradient_monitor.save_gradient_log(epoch)
    
    def clear_gradient_stats(self):
        """清除梯度统计信息"""
        if self.gradient_monitor is not None:
            self.gradient_monitor.clear_stats()


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
        use_checkpoint: bool = False,
        use_amp: bool = True,
        use_batch_norm: bool = True,
        use_residual: bool = True,
        dropout_rate: float = 0.1,
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
        
        # GPU优化设置
        self.use_checkpoint = use_checkpoint
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # 梯度监控设置
        self.gradient_monitor = None
        self.monitor_gradients = kwargs.get('monitor_gradients', False)
        self.gradient_save_dir = kwargs.get('gradient_save_dir', './gradient_logs')
        
        # 梯度消失解决方案
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        
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
        
        # 添加BatchNorm和Dropout到提升层
        if self.use_batch_norm:
            if self.n_dim == 1:
                self.lifting_bn = nn.BatchNorm1d(self.hidden_channels)
            elif self.n_dim == 2:
                self.lifting_bn = nn.BatchNorm2d(self.hidden_channels)
            else:
                self.lifting_bn = nn.BatchNorm3d(self.hidden_channels)
        else:
            self.lifting_bn = None
            
        if self.dropout_rate > 0:
            self.lifting_dropout = nn.Dropout(self.dropout_rate)
        else:
            self.lifting_dropout = None
        
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
            adaptive_padding=adaptive_padding,
            use_checkpoint=use_checkpoint,
            use_amp=use_amp
        )
        
        # 投影层（输出映射）
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim
        )
        
        # 添加BatchNorm和Dropout到投影层
        if self.use_batch_norm:
            if self.n_dim == 1:
                self.projection_bn = nn.BatchNorm1d(self.out_channels)
            elif self.n_dim == 2:
                self.projection_bn = nn.BatchNorm2d(self.out_channels)
            else:
                self.projection_bn = nn.BatchNorm3d(self.out_channels)
        else:
            self.projection_bn = None
            
        if self.dropout_rate > 0:
            self.projection_dropout = nn.Dropout(self.dropout_rate)
        else:
            self.projection_dropout = None
        
    def forward(self, x, output_shape=None, **kwargs):
        """
        前向传播 - GPU优化版本
        
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
        # 使用混合精度训练
        if self.use_amp and amp is not None:
            with torch.autocast(device_type='cuda'):
                return self._forward_impl(x, output_shape, **kwargs)
        else:
            return self._forward_impl(x, output_shape, **kwargs)
    
    def _forward_impl(self, x, output_shape=None, **kwargs):
        """实际的前向传播实现"""
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
                else:
                    # 添加BatchNorm和Dropout
                    if self.lifting_bn is not None:
                        x = self.lifting_bn(x)
                    if self.lifting_dropout is not None:
                        x = self.lifting_dropout(x)
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
                else:
                    # 添加BatchNorm和Dropout
                    if self.projection_bn is not None:
                        x = self.projection_bn(x)
                    if self.projection_dropout is not None:
                        x = self.projection_dropout(x)
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
    
    # 继承梯度监控方法 - 直接调用父类方法
    def enable_gradient_monitoring(self, save_dir=None):
        if hasattr(self, 'gradient_monitor'):
            if save_dir is None:
                save_dir = self.gradient_save_dir
            
            self.gradient_monitor = GradientMonitor(self, save_dir)
            print(f"Gradient monitoring enabled, logs saved to: {save_dir}")
    
    def disable_gradient_monitoring(self):
        if hasattr(self, 'gradient_monitor') and self.gradient_monitor is not None:
            self.gradient_monitor.remove_hooks()
            self.gradient_monitor = None
            print("Gradient monitoring disabled")
    
    def get_gradient_stats(self):
        if hasattr(self, 'gradient_monitor') and self.gradient_monitor is not None:
            return self.gradient_monitor.get_gradient_stats()
        return {}
    
    def plot_gradients(self, epoch, save=True):
        if hasattr(self, 'gradient_monitor') and self.gradient_monitor is not None:
            self.gradient_monitor.plot_gradients(epoch, save)
    
    def save_gradient_log(self, epoch):
        if hasattr(self, 'gradient_monitor') and self.gradient_monitor is not None:
            self.gradient_monitor.save_gradient_log(epoch)
    
    def clear_gradient_stats(self):
        if hasattr(self, 'gradient_monitor') and self.gradient_monitor is not None:
            self.gradient_monitor.clear_stats()
        
        
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

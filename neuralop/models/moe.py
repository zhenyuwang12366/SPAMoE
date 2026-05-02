import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from typing import List, Dict, Union, Tuple, Callable, Optional
import numpy as np

from .base_model import BaseModel
from ..layers.channel_mlp import ChannelMLP
from .task_router import TaskAwareRouter
from ..layers.spectral_convolution import SpectralConv
from .expert_factory import ExpertFactory


class Router(nn.Module):
    """
    路由器模块，负责将输入分配给最合适的专家
    
    Parameters
    ----------
    input_dim : int
        输入特征维度
    num_experts : int
        专家数量
    hidden_dim : int, optional
        路由器隐藏层维度
    top_k : int, optional
        选择前k个专家
    noisy_gating : bool, optional
        是否使用噪声门控机制
    """
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 256,
        top_k: int = 2,
        noisy_gating: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noisy_gating = noisy_gating

        # 路由器网络
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
    def forward(self, x):
        """
        计算每个专家的路由权重
        
        Parameters
        ----------
        x : torch.Tensor
            输入特征
        
        Returns
        -------
        dispatched_input : torch.Tensor
            分配后的输入
        routing_weights : torch.Tensor
            路由权重
        expert_indices : torch.Tensor
            选择的专家索引
        """
        # 计算路由权重
        logits = self.router(x)
        
        if self.noisy_gating and self.training:
            # 训练时添加噪声以增加探索性
            noise = torch.randn_like(logits) * 1.0
            logits = logits + noise
            
        # 使用Softmax获取专家权重
        routing_weights = F.softmax(logits, dim=-1)
        
        # 选择top-k专家
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # 归一化top-k权重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_weights, top_k_indices
    
class ContinuousCRF(nn.Module):
    def __init__(self, window_size=5, num_iterations=5, lambda_feat=1.0, lambda_pos=1.0):
        super().__init__()
        self.window_size = window_size
        self.num_iterations = num_iterations
        self.lambda_feat = lambda_feat
        self.lambda_pos = lambda_pos

        # 可学习参数：pairwise 权重 w ≥ 0
        self.w = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, unary_pred, features):
        """
        unary_pred: CNN 输出速度预测 [B, 1, H, W]
        features: 最后一层 decoder 特征图 [B, C, H, W]
        """
        B, _, H, W = unary_pred.shape
        _, C, _, _ = features.shape
        device = unary_pred.device

        # 初始 mean field 分布
        mu = unary_pred.clone()
        for _ in range(self.num_iterations):
            mu_new = torch.zeros_like(mu)

            for dx in range(-self.window_size, self.window_size + 1):
                for dy in range(-self.window_size, self.window_size + 1):
                    if dx == 0 and dy == 0:
                        continue

                    # 平移 feature 和 mu
                    f_shift = self.shift_tensor(features, dx, dy)
                    mu_shift = self.shift_tensor(mu, dx, dy)

                    # 计算 feature 相似度
                    diff_feat = (features - f_shift).pow(2).sum(dim=1, keepdim=True)
                    diff_pos = (dx**2 + dy**2)

                    weight = torch.exp(-self.lambda_feat * diff_feat - self.lambda_pos * diff_pos)
                    mu_new += weight * mu_shift

            # 归一化更新
            norm_factor = 1 + self.w * (2 * self.window_size + 1)**2
            mu = (unary_pred + self.w * mu_new) / norm_factor

        return mu  # CRF refined prediction

    def shift_tensor(self, x, dx, dy):
        B, C, H, W = x.shape
        shifted = torch.zeros_like(x)

        # 计算有效区域的起止 index，让 src 和 tgt 尺寸一致
        if dx >= 0:
            src_x1, src_x2 = 0, H - dx
            tgt_x1, tgt_x2 = dx, H
        else:
            src_x1, src_x2 = -dx, H
            tgt_x1, tgt_x2 = 0, H + dx

        if dy >= 0:
            src_y1, src_y2 = 0, W - dy
            tgt_y1, tgt_y2 = dy, W
        else:
            src_y1, src_y2 = -dy, W
            tgt_y1, tgt_y2 = 0, W + dy

        # 确保 source 和 target 区域 shape 相同
        shifted[:, :, tgt_x1:tgt_x2, tgt_y1:tgt_y2] = x[:, :, src_x1:src_x2, src_y1:src_y2]
        return shifted

class MOEOperator(BaseModel, name='MOE'):
    """
    多专家混合神经算子 (Mixture of Experts Neural Operator)
    
    该模型集成了多种不同类型的神经算子专家，包括不同域专家、尺度专家和几何专家。
    
    Parameters
    ----------
    experts : List[nn.Module]
        专家模型列表
    in_channels : int
        输入通道数
    out_channels : int
        输出通道数
    hidden_channels : int
        隐藏层通道数
    top_k : int, optional
        每次选择的专家数量，默认为2
    noisy_gating : bool, optional
        是否使用噪声门控，默认为True
    fusion_type : str, optional
        专家输出融合方式，可选'linear'或'attention'，默认为'linear'
    router_hidden_dim : int, optional
        路由器隐藏层维度，默认为256
    router_type : str, optional
        路由器类型，可选'basic'或'task_aware'，默认为'basic'
    task_dim : int, optional
        任务特征维度，当router_type为'task_aware'时有效，默认为0
    routing_mode : str, optional
        路由模式，当router_type为'task_aware'时有效，默认为'input'
    """
    def __init__(
        self,
        experts: List[nn.Module],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        top_k: int = 2,
        noisy_gating: bool = True,
        fusion_type: str = 'linear',
        router_hidden_dim: int = 256,
        router_type: str = 'basic',
        task_dim: int = 0,
        routing_mode: str = 'input',
        **kwargs
    ):
        super().__init__()
        
        # 保存参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.fusion_type = fusion_type
        self.router_type = router_type
        
        # CRF
        self.CCrf = ContinuousCRF()
        
        # CRF使用的特征图字典
        self.feature_map = {}
        
        # 专家列表
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        
        # 确保top_k不超过专家数量
        self.top_k = min(self.top_k, self.num_experts)
        
        # 路由器，用于分配输入给合适的专家
        if router_type == 'basic':
            self.router = Router(
                input_dim=in_channels,
                num_experts=self.num_experts,
                hidden_dim=router_hidden_dim,
                top_k=self.top_k,
                noisy_gating=noisy_gating
            )
        elif router_type == 'task_aware':
            self.router = TaskAwareRouter(
                input_dim=in_channels,
                task_dim=task_dim,
                num_experts=self.num_experts,
                hidden_dim=router_hidden_dim,
                top_k=self.top_k,
                noisy_gating=noisy_gating,
                routing_mode=routing_mode
            )
        else:
            raise ValueError(f"不支持的路由器类型: {router_type}")
            
        # 输出融合层
        if fusion_type == 'linear':
            self.fusion = nn.Linear(self.out_channels, self.out_channels)
        elif fusion_type == 'attention':
            self.fusion = nn.MultiheadAttention(
                embed_dim=self.out_channels,
                num_heads=4,
                batch_first=True
            )
        else:
            raise ValueError(f"未支持的融合类型: {fusion_type}")
            
        # 添加空间到时间-偏移的转换网络
        # 这个网络将把形状为[B, C, H, W]的空间表示转换为形状为[B, C, T, R]的时间-偏移表示
        # 其中B是批次大小，C是通道数，H和W是空间维度，T是时间步数，R是接收器数量
        # self.space_to_time_projection = nn.Sequential(
        #     # 第一层：特征提取和通道扩展
        #     nn.Conv2d(out_channels, out_channels * 4, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     # 第二层：开始增加时间维度
        #     nn.ConvTranspose2d(out_channels * 4, out_channels * 4, 
        #                       kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
        #     nn.BatchNorm2d(out_channels * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     # 第三层：继续增加时间维度
        #     nn.ConvTranspose2d(out_channels * 4, out_channels * 2, 
        #                       kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
        #     nn.BatchNorm2d(out_channels * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     # 第四层：继续增加时间维度
        #     nn.ConvTranspose2d(out_channels * 2, out_channels * 2, 
        #                       kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
        #     nn.BatchNorm2d(out_channels * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     # 第五层：继续增加时间维度
        #     nn.ConvTranspose2d(out_channels * 2, out_channels, 
        #                       kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
        #     nn.BatchNorm2d(out_channels),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     # 最终调整层：精确调整到目标形状
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.Upsample(size=(1000, 70), mode='bilinear', align_corners=True)
        # )
        # [B, C, T, R] -> [B, C, H, W] <-> [B, 1, 256, 256] -> [B, 1, 70, 70]
        self.time_to_space_projection = nn.Sequential(
            # R不变，压缩时间维度：1000 -> 500
            nn.Conv2d(1, 32, kernel_size=3, stride=(2,1), padding=1), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 500 -> 250
            nn.Conv2d(32, 64, kernel_size=3, stride=(2,1), padding=1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 250 -> 125
            nn.Conv2d(64, 128, kernel_size=3, stride=(2,1), padding=1), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 125 -> 70
            nn.Conv2d(128, 128, kernel_size=4, stride=(1,1), padding=0),  
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, stride=(1,1), padding=1),   
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=(1,1), padding=1),     
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 最后一次卷积对齐为 (70, 70)
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=0),  
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, 1, kernel_size=3, stride=(1,1), padding=1),
            nn.Upsample(size=(70, 70), mode='bilinear', align_corners=True)
        )
        # 注册钩子函数
        self.time_to_space_projection[-2].register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module: nn.Module, input: tuple[torch.Tensor], output: torch.Tensor):
        features = input[0].detach()
        features = F.interpolate(features, size=(70, 70), mode='bilinear', align_corners=True)
        self.feature_map['feature'] = features
       
    def forward(self, x, task_features=None, **kwargs):
        """
        前向传播
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量
        task_features : torch.Tensor, optional
            任务特征，当router_type为'task_aware'时使用
        
        Returns
        -------
        torch.Tensor
            输出张量
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 提取输入特征进行路由
        x_flat = x.view(batch_size, -1, self.in_channels).mean(dim=1)
        
        # 计算路由权重和选择的专家
        if self.router_type == 'basic':
            routing_weights, expert_indices = self.router(x_flat)
        else:  # task_aware
            routing_weights, expert_indices = self.router(x_flat, task_features)
        
        # 初始化输出和形状信息收集
        expert_outputs_collection = []
        
        # 调试信息：记录每个专家的输出形状
        expert_shapes = {}
        
        # 对每个选定的专家
        for k in range(self.top_k):
            # 获取当前专家索引和权重
            indices = expert_indices[:, k]
            weights = routing_weights[:, k]
            
            # 每个样本分别通过不同的专家
            batch_expert_outputs = []
            
            # 运行所有专家得到输出
            for b in range(batch_size):
                expert_idx = indices[b].item()
                try:
                    # 进行映射
                    x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
                    
                    # 调用专家模型
                    expert_output = self.experts[expert_idx](x[b:b+1], **kwargs)
                    
                    # 检查专家输出是否为None
                    if expert_output is None:
                        print(f"警告：专家{expert_idx}返回None，使用零张量代替")
                        # 使用零张量作为替代
                        expert_output = torch.zeros(
                            1, self.out_channels, *x.shape[2:], 
                            device=device, dtype=x.dtype
                        )
                        
                    # 记录专家输出形状（用于调试）
                    if expert_idx not in expert_shapes:
                        expert_shapes[expert_idx] = []
                    expert_shapes[expert_idx].append(tuple(expert_output.shape))
                    
                    batch_expert_outputs.append(expert_output)
                    
                except Exception as e:
                    print(f"专家{expert_idx}处理样本{b}时发生错误：{str(e)}")
                    # 使用零张量作为替代
                    zero_output = torch.zeros(
                        1, self.out_channels, *x.shape[2:], 
                        device=device, dtype=x.dtype
                    )
                    batch_expert_outputs.append(zero_output)
            
            # 收集这个专家的所有输出和对应权重
            expert_outputs_collection.append((batch_expert_outputs, weights))
        
        # 打印调试信息
        print("专家输出形状:")
        for expert_idx, shapes in expert_shapes.items():
            print(f"专家 {expert_idx}: {shapes}")
        
        # 分析阶段：收集所有输出的形状信息
        output_shapes = []
        for batch_outputs, _ in expert_outputs_collection:
            for output in batch_outputs:
                if output.dim() >= 4:  # 确保有足够的维度
                    output_shapes.append(output.shape[2:])  # 收集空间维度形状
        
        # 如果没有有效输出，返回零张量
        if not output_shapes:
            return torch.zeros(batch_size, self.out_channels, *x.shape[2:], device=device)
        
        # 决策阶段：确定统一的目标形状
        # 对于每个空间维度，选择最常见的大小
        from collections import Counter
        
        # 获取所有形状的维度数
        ndims = [len(shape) for shape in output_shapes]
        if len(set(ndims)) > 1:
            # 如果维度数不一致，选择最常见的维度数
            target_ndim = Counter(ndims).most_common(1)[0][0]
            # 只保留具有目标维度数的形状
            output_shapes = [shape for shape in output_shapes if len(shape) == target_ndim]
        
        # 为每个维度确定目标大小
        target_shape = []
        for dim_idx in range(len(output_shapes[0])):
            dim_sizes = [shape[dim_idx] for shape in output_shapes]
            target_size = Counter(dim_sizes).most_common(1)[0][0]
            target_shape.append(target_size)
        
        target_shape = tuple(target_shape)
        print(f"目标形状: {target_shape}")
        
        # 处理每个专家的输出
        outputs = []
        for batch_outputs, weights in expert_outputs_collection:
            # 调整每个样本的输出
            adjusted_batch_outputs = []
            for i, output in enumerate(batch_outputs):
                # 检查形状是否需要调整
                if output.dim() >= 4 and output.shape[2:] != target_shape:
                    print(f"调整输出形状: {output.shape[2:]} -> {target_shape}")
                    # 使用插值调整大小
                    try:
                        # 对于2D和3D输入，使用适当的插值模式
                        mode = 'trilinear' if len(target_shape) == 3 else 'bilinear' if len(target_shape) == 2 else 'linear'
                        adjusted_output = F.interpolate(
                            output,
                            size=target_shape,
                            mode=mode,
                            align_corners=True
                        )
                        adjusted_batch_outputs.append(adjusted_output)
                    except Exception as e:
                        print(f"插值失败: {e}")
                        print(f"输入形状: {output.shape}, 目标形状: {target_shape}")
                        # 如果插值失败，创建一个形状正确的零张量
                        adjusted_output = torch.zeros(
                            output.shape[0], output.shape[1], *target_shape,
                            device=output.device, dtype=output.dtype
                        )
                        adjusted_batch_outputs.append(adjusted_output)
                else:
                    adjusted_batch_outputs.append(output)
            
            # 检查调整后的形状是否一致
            shapes_after_adjustment = [out.shape for out in adjusted_batch_outputs]
            print(f"调整后形状: {shapes_after_adjustment}")
            
            try:
                # 合并这个专家的所有样本输出
                expert_output = torch.cat(adjusted_batch_outputs, dim=0)
                
                # 应用路由权重
                weighted_output = expert_output * weights.view(batch_size, 1, 1, 1)
                outputs.append(weighted_output)
            except Exception as e:
                print(f"合并输出失败: {e}")
                print(f"调整后的输出形状: {[out.shape for out in adjusted_batch_outputs]}")
                # 跳过这个专家
                continue
        
        # 如果没有有效输出，返回零张量
        if not outputs:
            print("没有有效的专家输出，返回零张量")
            return torch.zeros(batch_size, self.out_channels, *target_shape, device=device)
        
        # 合并所有专家的输出
        combined_output = sum(outputs)
        
        # 应用融合层
        if self.fusion_type == 'linear':
            # 线性融合
            shape = combined_output.shape
            combined_output = combined_output.view(batch_size, -1, self.out_channels)
            combined_output = self.fusion(combined_output)
            combined_output = combined_output.view(*shape)
        elif self.fusion_type == 'attention':
            # 注意力融合
            shape = combined_output.shape
            combined_output = combined_output.view(batch_size, -1, self.out_channels)
            combined_output, _ = self.fusion(
                combined_output, combined_output, combined_output
            )
            combined_output = combined_output.view(*shape)
        
        # 检查是否需要调整输出形状以匹配目标形状
        # 地震数据的目标形状通常为 [batch_size, num_sources, time_steps, num_receivers]
        # 其中 num_sources=5, time_steps=1000, num_receivers=70
        # if self.out_channels == 1 and combined_output.shape[2] != 1000:
        #     # 使用可学习的转换网络将空间表示转换为时间-偏移表示
        #     combined_output = self.space_to_time_projection(combined_output)
        if self.out_channels == 1 and combined_output.shape[2] != 70:
            # print("ok \n")
            combined_output = self.time_to_space_projection(combined_output)
        
        #CCrf
        last_output = self.CCrf(combined_output, self.feature_map['feature'])
        
        return last_output
    
    def get_expert_distribution(self, x, task_features=None):
        """
        获取专家分配分布
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量
        task_features : torch.Tensor, optional
            任务特征，当router_type为'task_aware'时使用
            
        Returns
        -------
        torch.Tensor
            专家分配分布
        """
        batch_size = x.shape[0]
        
        # 提取输入特征进行路由
        x_flat = x.view(batch_size, -1, self.in_channels).mean(dim=1)
        
        # 获取专家分配分布
        if hasattr(self.router, 'get_expert_distribution'):
            # 对于TaskAwareRouter
            return self.router.get_expert_distribution(x_flat, task_features)
        else:
            # 对于基本Router，手动计算分布
            logits = self.router.router(x_flat)
            return F.softmax(logits, dim=-1)
    
    def save_experts(self, save_dir):
        """
        保存每个专家模型到指定目录
        
        Parameters
        ----------
        save_dir : str
            保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存每个专家
        for i, expert in enumerate(self.experts):
            expert_path = save_dir / f"expert_{i}.pt"
            torch.save(expert.state_dict(), expert_path)
        
        # 保存专家元数据
        metadata = {
            'num_experts': self.num_experts,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'hidden_channels': self.hidden_channels,
            'router_type': self.router_type
        }
        torch.save(metadata, save_dir / "metadata.pt")
    
    def load_experts(self, load_dir):
        """
        从指定目录加载专家模型
        
        Parameters
        ----------
        load_dir : str
            加载目录
        """
        load_dir = Path(load_dir)
        
        # 验证目录存在
        if not load_dir.exists():
            raise ValueError(f"专家目录不存在: {load_dir}")
        
        # 加载元数据
        metadata_path = load_dir / "metadata.pt"
        if not metadata_path.exists():
            raise ValueError(f"元数据文件不存在: {metadata_path}")
        
        metadata = torch.load(metadata_path)
        
        # 验证专家数量
        if metadata['num_experts'] != self.num_experts:
            raise ValueError(f"专家数量不匹配: 期望 {self.num_experts}，实际 {metadata['num_experts']}")
        
        # 加载每个专家
        for i, expert in enumerate(self.experts):
            expert_path = load_dir / f"expert_{i}.pt"
            if not expert_path.exists():
                raise ValueError(f"专家文件不存在: {expert_path}")
            
            expert.load_state_dict(torch.load(expert_path))
        
        print(f"成功加载 {self.num_experts} 个专家模型") 
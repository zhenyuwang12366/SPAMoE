import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple, Optional

class TaskAwareRouter(nn.Module):
    """
    任务感知路由器，能够根据输入数据特征和任务特征进行路由
    
    Parameters
    ----------
    input_dim : int
        输入特征维度
    task_dim : int
        任务特征维度，如果为0则不使用任务特征
    num_experts : int
        专家数量
    hidden_dim : int, optional
        路由器隐藏层维度
    top_k : int, optional
        选择前k个专家
    noisy_gating : bool, optional
        是否使用噪声门控机制
    routing_mode : str, optional
        路由模式，可选'input'（仅使用输入特征）, 'task'（仅使用任务特征）, 'both'（同时使用输入和任务特征）
    """
    def __init__(
        self,
        input_dim: int,
        task_dim: int = 0,
        num_experts: int = 4,
        hidden_dim: int = 256,
        top_k: int = 2,
        noisy_gating: bool = True,
        routing_mode: str = 'both',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.task_dim = task_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noisy_gating = noisy_gating
        self.routing_mode = routing_mode
        
        # 验证路由模式
        if routing_mode not in ['input', 'task', 'both']:
            raise ValueError(f"不支持的路由模式: {routing_mode}")
        
        # 如果使用任务特征但未提供任务维度，则报错
        if routing_mode in ['task', 'both'] and task_dim <= 0:
            raise ValueError(f"当routing_mode为'{routing_mode}'时，task_dim必须大于0")
        
        # 根据路由模式确定输入维度
        router_input_dim = 0
        if routing_mode in ['input', 'both']:
            router_input_dim += input_dim
        if routing_mode in ['task', 'both']:
            router_input_dim += task_dim
        
        # 路由器网络
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, task_features=None):
        """
        计算每个专家的路由权重
        
        Parameters
        ----------
        x : torch.Tensor
            输入特征，形状为 [batch_size, input_dim]
        task_features : torch.Tensor, optional
            任务特征，形状为 [batch_size, task_dim]，当routing_mode为'task'或'both'时必须提供
            
        Returns
        -------
        routing_weights : torch.Tensor
            路由权重，形状为 [batch_size, top_k]
        expert_indices : torch.Tensor
            选择的专家索引，形状为 [batch_size, top_k]
        """
        batch_size = x.shape[0]
        
        # 验证输入
        if self.routing_mode in ['task', 'both'] and task_features is None:
            raise ValueError(f"当routing_mode为'{self.routing_mode}'时，必须提供task_features")
        
        # 准备路由器输入
        router_input = []
        if self.routing_mode in ['input', 'both']:
            # 如果x是多维张量，将其展平为2D
            if x.dim() > 2:
                x_flat = x.view(batch_size, -1)
                # 如果展平后的维度太大，可以进行池化或特征提取
                if x_flat.shape[1] > self.input_dim:
                    # 简单的平均池化
                    x_flat = x_flat.view(batch_size, -1, self.input_dim).mean(dim=1)
            else:
                x_flat = x
            router_input.append(x_flat)
            
        if self.routing_mode in ['task', 'both'] and task_features is not None:
            router_input.append(task_features)
        
        # 合并输入特征
        router_input = torch.cat(router_input, dim=1)
        
        # 计算路由权重
        logits = self.router(router_input)
        
        if self.noisy_gating and self.training:
            # 训练时添加噪声以增加探索性
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise
            
        # 使用Softmax获取专家权重
        routing_weights = F.softmax(logits, dim=-1)
        
        # 选择top-k专家
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # 归一化top-k权重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_weights, top_k_indices
    
    def get_expert_distribution(self, x, task_features=None):
        """
        获取专家分配分布
        
        Parameters
        ----------
        x : torch.Tensor
            输入特征
        task_features : torch.Tensor, optional
            任务特征
            
        Returns
        -------
        expert_distribution : torch.Tensor
            专家分配分布，形状为 [batch_size, num_experts]
        """
        batch_size = x.shape[0]
        
        # 准备路由器输入
        router_input = []
        if self.routing_mode in ['input', 'both']:
            # 如果x是多维张量，将其展平为2D
            if x.dim() > 2:
                x_flat = x.view(batch_size, -1)
                # 如果展平后的维度太大，可以进行池化或特征提取
                if x_flat.shape[1] > self.input_dim:
                    # 简单的平均池化
                    x_flat = x_flat.view(batch_size, -1, self.input_dim).mean(dim=1)
            else:
                x_flat = x
            router_input.append(x_flat)
            
        if self.routing_mode in ['task', 'both'] and task_features is not None:
            router_input.append(task_features)
        
        # 合并输入特征
        router_input = torch.cat(router_input, dim=1)
        
        # 计算路由权重
        logits = self.router(router_input)
        
        # 使用Softmax获取专家权重
        routing_weights = F.softmax(logits, dim=-1)
        
        return routing_weights 
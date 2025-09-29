import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from typing import List, Dict, Union, Tuple, Callable, Optional
import numpy as np
from torchvision import models
from collections import Counter

from .base_model import BaseModel
from ..layers.channel_mlp import ChannelMLP
from .task_router import TaskAwareRouter
from ..layers.spectral_convolution import SpectralConv
from .expert_factory import ExpertFactory
from .task_dependent_router import TaskDependentRouter
from .SWActivate import GroupActMerge, SWActMerge
from .merge_processer import MeanMix, SumMix, LinearMix, AttentionMix

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
        alpha: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noisy_gating = noisy_gating
        self.alpha = alpha # aux负载均衡损失因子
        
        # 路由器网络
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
    def forward(self, x):
        """
        返回:
        top_k_weights: [B, k]
        top_k_indices: [B, k]
        w_weights:     [B, w_k] or None
        w_indices:     [B, w_k] or None
        aux_loss:      scalar tensor or None( eval )
        """
        logits = self.router(x)                             # [B, N]

        if self.noisy_gating and self.training:
            noise_scale = 1.0
            logits = logits + torch.randn_like(logits) * noise_scale

        routing_weights = F.softmax(logits, dim=-1)         # [B, N]

        # --- top-k 主专家 ---
        top_k = int(self.top_k)
        num_experts = int(self.num_experts)
        assert num_experts > 0 and 1 <= top_k <= num_experts

        top_k_weights, top_k_indices = torch.topk(
            routing_weights, k=top_k, dim=-1, largest=True, sorted=True
        )                                                   # [B, k], [B, k]

        # 归一化
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # --- 弱激活专家（从“未被 top-k 选中”的剩余里挑） ---
        w_top_k = num_experts - top_k
        if w_top_k > 0:
            # 构造与 routing_weights 同形状的选择掩码: 被 top-k 选中的位置为 True
            B, N = routing_weights.shape
            selected_mask = torch.zeros(B, N, dtype=torch.bool, device=routing_weights.device)
            # scatter_ 支持 bool，按行把 top_k_indices 位置置 True
            selected_mask.scatter_(dim=1, index=top_k_indices, src=torch.ones_like(top_k_indices, dtype=torch.bool))

            # 把已选中的位置屏蔽为 -inf，从“剩余位置”再取最大的 w_top_k
            remaining = routing_weights.clone().masked_fill(selected_mask, float('-inf'))

            # 若 num_experts == top_k，这里全是 -inf；但我们有 w_top_k>0 的判断保证不会走到这里
            w_weights, w_indices = torch.topk(remaining, k=w_top_k, dim=-1, largest=True, sorted=True)
            # 将 -inf 安全归一化（如果极端情况下全是 -inf，会得到 nan，故先替换为 0）
            w_weights = torch.where(torch.isfinite(w_weights), w_weights, torch.zeros_like(w_weights))
            w_weights = w_weights / w_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            w_weights, w_indices = None, None

        # --- aux（负载均衡损失） ---
        if self.training and getattr(self, "alpha", 0.0) > 0.0:
            if top_k_indices.numel() == 0 or routing_weights.numel() == 0:
                # 空 batch 情况：返回零标量，保持图连接，避免 DDP 不一致
                aux_loss = routing_weights.sum() * 0.0
            else:
                idx = top_k_indices.reshape(-1).to(dtype=torch.long)
                ce = F.one_hot(idx, num_classes=num_experts).float().mean(dim=0)  # [N]
                Pi = routing_weights.mean(dim=0)                                  # [N]
                fi = ce * float(num_experts)                                      # [N]
                aux_loss = (Pi * fi).sum() * float(self.alpha)
        else:
            aux_loss = None

        return top_k_weights, top_k_indices, w_weights, w_indices, aux_loss
    
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
        router_type: str = 'basic',  # 'basic' | 'task_aware' | 'adamv'
        task_dim: int = 0,
        routing_mode: str = 'input',
        is_logger: bool = False,
        v_type_num: Optional[int] = None,
        batch_size: int = 0,
        **kwargs
    ):
        super().__init__()
        self.config = kwargs

        # 保存参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.top_k = top_k
        
        if v_type_num is None:
            self.v_type_num = 5 if self.config.get('is_specific', True) else 3
        else:
            self.v_type_num = v_type_num
            
        # 专家列表
        self.experts = nn.ModuleList(experts)
        if self.config.get('is_classier', False):
            if self.config.get('is_specific', True):
                self.num_experts = int(len(experts) / self.v_type_num)
            else:
                self.num_experts = int(len(experts) / self.v_type_num)
        else:
            self.num_experts = len(experts)

        self.noisy_gating = noisy_gating
        self.fusion_type = fusion_type
        self.router_type = router_type

        # 确保 top_k 不超过专家数量
        self.top_k = min(self.top_k, self.num_experts)
        # 得出弱专家组专家数
        self.w_k = max(0, self.num_experts - top_k)
        
        # -------- 分类器骨干（ImageNet 预训练 ResNet50，B*1*H*W）--------
        if self.config.get('is_classier', False):
            self.classier = models.resnet50(pretrained=True)
            old_conv = self.classier.conv1
            new_conv = nn.Conv2d(
                1,
                old_conv.out_channels,
                old_conv.kernel_size,
                old_conv.stride,
                old_conv.padding,
                bias=False
            )
            with torch.no_grad():
                w = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(w)
            self.classier.conv1 = new_conv

            in_features = self.classier.fc.in_features
            
            self.fc = nn.Linear(in_features, self.v_type_num)
            
            self.classier.fc = nn.Identity()
        else:
            pass

        # -------- 路由器 --------
        if self.num_experts > 1:
            self.alpha: float = 0.1
        else:
            self.alpha: float = 0.0
        if self.router_type == 'task_aware' and self.fusion_type == 'swa':
            raise ValueError("task_aware 路由当前不支持 'swa' 融合（缺少弱组）。")
        if router_type == 'basic':
            self.router = Router(
                input_dim=in_channels,
                num_experts=self.num_experts,
                hidden_dim=router_hidden_dim,
                top_k=self.top_k,
                noisy_gating=noisy_gating,
                alpha = self.alpha,
            )
        elif router_type == 'task_aware':
            self.router = TaskAwareRouter(
                input_dim=in_channels,
                task_dim=task_dim,
                num_experts=self.num_experts,
                hidden_dim=router_hidden_dim,
                top_k=self.top_k,
                noisy_gating=noisy_gating,
                routing_mode=routing_mode,
            )
        elif router_type == 'adamv':
            self.router = TaskDependentRouter(
                input_dim=in_channels,
                num_experts=self.num_experts,
                hidden_dim=router_hidden_dim,
                bsz=batch_size,
                noisy_gating=noisy_gating,
                alpha = self.alpha,
            )
        else:
            raise ValueError(f"不支持的路由器类型: {router_type}")

        # -------- s_processor / w_processor --------
        for t in ['s_processor', 'w_processor']:
            cfg_key = f'{t}_type'
            module = None
            if self.config.get(cfg_key, None) == 'linear':
                module = nn.Linear(self.out_channels, self.out_channels)
            elif self.config.get(cfg_key, None) == 'attn':
                module = AttentionMix(self.out_channels)
            elif self.config.get(cfg_key, None) == 'mean':
                module = MeanMix(self.out_channels)
            elif self.config.get(cfg_key, None) == 'sum':
                module = SumMix(self.out_channels)
            setattr(self, t, module)
        
        # -------- 融合层 --------
        if fusion_type == 'linear':
            self.fusion = nn.Linear(self.out_channels, self.out_channels)
        elif fusion_type == 'attention':
            self.fusion = nn.MultiheadAttention(
                embed_dim=self.out_channels,
                num_heads=4,
                batch_first=True
            )
        elif fusion_type == 'swa':
            self.s_act = GroupActMerge(processor=self.s_processor)
            self.w_act = GroupActMerge(processor=self.w_processor)
            self.sw_act = SWActMerge(beta=self.config.get('beta', 0.5))
        else:
            raise ValueError(f"未支持的融合类型: {fusion_type}")

        # -------- 时域→空域投影（B,1,T,R -> B,1,70,70）--------
        # 如用 AMP，调试阶段可把 inplace=False 更稳
        self.time_to_space_projection = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=4, stride=(1, 1), padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 1, kernel_size=3, stride=(1, 1), padding=1),
            nn.Upsample(size=(70, 70), mode='bilinear', align_corners=True)
        )

        # -------- CRF 与特征 Map --------
        self.CCrf = ContinuousCRF()
        self.feature_map = {}
        self.time_to_space_projection[-2].register_forward_hook(self.hook_fn)

        self.is_logger = is_logger

        # ========= 统一初始化：放在最后 =========
        self.reset_parameters_()

    # --------------- 初始化策略 ---------------
    def reset_parameters_(self):
        """
        - Conv2d(LeakyReLU 0.2) → Kaiming normal (fan_in, a=0.2)
        - BatchNorm2d → gamma=1, beta=0
        - Linear → Xavier uniform
        - Router 的最后一层 Linear 权重与偏置置零（初始均匀路由）
        - fusion_type='linear' 时：若方阵，初始化为近似恒等；否则 Xavier
        - 预训练 ResNet50：仅初始化新建的 fc（conv1 已做均值拷贝）
        - Experts/CRF/SWActMerge/GroupActMerge 等按通用规则或调用其内置 reset
        """
        neg_slope = 0.2

        def init_module(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=neg_slope, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 1) 通用初始化：time_to_space, CRF, experts, s/w processor
        self.time_to_space_projection.apply(init_module)

        if hasattr(self, 'CCrf') and self.CCrf is not None:
            # 若 CCrf 内部包含可学习层
            try:
                self.CCrf.apply(init_module)
            except Exception:
                pass

        for e in self.experts:
            if hasattr(e, 'reset_parameters'):
                e.reset_parameters()
            else:
                e.apply(init_module)

        for t in ['s_processor', 'w_processor']:
            mod = getattr(self, t, None)
            if mod is not None:
                try:
                    mod.apply(init_module)
                except Exception:
                    pass

        # 2) Router：先通用初始化，再把“最后一层 Linear”置零，得到“均匀起步”
        if hasattr(self, 'router') and self.router is not None:
            self.router.apply(init_module)
            last_linear = None
            for m in self.router.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

        # 3) 融合层
        if self.fusion_type == 'linear':
            if isinstance(self.fusion, nn.Linear) and self.fusion.weight.shape[0] == self.fusion.weight.shape[1]:
                with torch.no_grad():
                    self.fusion.weight.zero_()
                    eye = torch.eye(self.fusion.weight.shape[0], device=self.fusion.weight.device)
                    self.fusion.weight.add_(eye)
                    if self.fusion.bias is not None:
                        self.fusion.bias.zero_()
            else:
                init_module(self.fusion)
        elif self.fusion_type == 'attention':
            # PyTorch 默认初始化已较稳，通常不改
            pass
        elif self.fusion_type == 'swa':
            if hasattr(self, 's_act') and self.s_act is not None:
                try:
                    self.s_act.apply(init_module)
                except Exception:
                    pass
            if hasattr(self, 'w_act') and self.w_act is not None:
                try:
                    self.w_act.apply(init_module)
                except Exception:
                    pass
            if hasattr(self, 'sw_act') and self.sw_act is not None:
                try:
                    self.sw_act.apply(init_module)
                except Exception:
                    pass

        # 4) ResNet50 新建 fc
        if hasattr(self, 'classier'):
            if hasattr(self, 'fc'):
                nn.init.xavier_uniform_(self.fc.weight, gain=1.0)
                if self.fc.bias is not None:
                    nn.init.zeros_(self.fc.bias)

    # --------------- Hook：为 CRF 准备特征图 ---------------
    def hook_fn(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
        features = input[0].detach()
        features = F.interpolate(features, size=(70, 70), mode='bilinear', align_corners=True)
        self.feature_map['feature'] = features

    # --------------- 前向传播 ---------------
    def forward(self, x, task_features=None, **kwargs) -> Tuple[torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device

        if self.config.get('is_classier', False):
            feats = self.classier(x)
            type_logits = self.fc(feats)  # B*v_type/B*v_type
            type_weights = torch.softmax(type_logits, dim=-1)
        else:
            type_weights = None

        # x: [B,1,1000,350] -> 对 C H W 做均值
        x_flat = x.mean(dim=(2, 3)).view(batch_size, -1)  # [B, in_channels]

        # 路由
        if self.router_type == 'basic':
            s_weights, s_indices, w_weights, w_indices, aux_loss = self.router(x_flat)
        elif self.router_type == 'adamv':
            s_weights, s_indices, w_weights, w_indices, aux_loss, new_k = self.router(x_flat)
            self.top_k = new_k
            self.w_k = max(0, self.num_experts - self.top_k)
        else:  # task_aware
            s_weights, s_indices = self.router(x_flat, task_features)
            w_weights = w_indices = None
            aux_loss = None

        s_outputs = self._process_activation_group(
            x=x,
            expert_indices=s_indices,
            routing_weights=s_weights,
            k=self.top_k,
            batch_size=batch_size,
            device=device,
            type_weights=type_weights,
            kwargs=kwargs,
        )

        if self.fusion_type == 'swa':
            w_outputs = self._process_activation_group(
                x=x,
                expert_indices=w_indices,
                routing_weights=w_weights,
                k=self.w_k,
                batch_size=batch_size,
                device=device,
                type_weights=type_weights,
                kwargs=kwargs,
            )

        # 合并专家输出
        if self.fusion_type == 'swa':
            s_combined = torch.stack(s_outputs, dim=1)  # (B, k, 1, h, w)
            w_combined = torch.stack(w_outputs, dim=1)
            s_merged = self.s_act(s_combined)
            w_merged = self.w_act(w_combined)
            combined_output = self.sw_act(s_merged, w_merged)
        else:
            # 线性/注意力两种都先把 s_outputs 加权求和得到 combined_output
            combined_output = sum(s_outputs)

        # 融合层
        if self.fusion_type == 'linear':
            shape = combined_output.shape
            combined_output = combined_output.view(batch_size, -1, self.out_channels)
            combined_output = self.fusion(combined_output)
            combined_output = combined_output.view(*shape)
        elif self.fusion_type == 'attention':
            shape = combined_output.shape
            combined_output = combined_output.view(batch_size, -1, self.out_channels)
            combined_output, _ = self.fusion(
                combined_output, combined_output, combined_output
            )
            combined_output = combined_output.view(*shape)
        # 'swa' 已在上面合并

        # 调整输出尺寸到 (B,1,70,70)
        if self.out_channels == 1 and combined_output.shape[2] != 70:
            combined_output = self.time_to_space_projection(combined_output)
        else:
            if 'feature' not in self.feature_map:
                self.feature_map['feature'] = F.interpolate(
                    combined_output.detach(),
                    size=(70, 70),
                    mode='bilinear',
                    align_corners=True
                )

        # CRF
        last_output: torch.Tensor = self.CCrf(combined_output, self.feature_map['feature'])
        return last_output, aux_loss
    
    def _process_activation_group(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,   # [B, top_k] —— 每个样本被路由到的组/专家ID
        routing_weights: torch.Tensor,  # [B, top_k] —— 对应的路由权重
        k: int,                         # 选择的 top-k（传入值）
        batch_size: int,
        device,
        type_weights: Optional[torch.Tensor] = None,  # [B, T] —— 速度图类型权重(T=5或3)，None 表示不使用分组结构
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        计算 Router 选出的 top-k 组/专家的输出，并乘以对应的路由权重。
        返回长度为 top_k 的列表，每个元素形状为 [B, C, H, W]。

        两种模式：
        1) 分组模式（type_weights != None）：
        - 每个“组 g”包含 T 个子模型（速度图类型 0..T-1）
        - 对组内子模型输出按 type_weights[b, t] 加权求和，得到该组在样本 b 的输出
        - 再乘以该组的路由权重 routing_weights[b, k_idx]

        2) 直接专家模式（type_weights == None）：
        - Router 直接把每个样本路由到某个专家（不区分组内子模型）
        - 乘以对应路由权重

        额外特性：
        - 自动统一空间分辨率到最常见的目标形状（避免不同专家输出形状不一致）
        - 所有异常（专家抛错/None、插值失败、cat失败）都有零张量兜底
        """
        
        top_k = k  # 统一命名
        C = self.out_channels
        T = self.v_type_num

        # ==========（可选）统一输入尺寸给专家；默认关闭以保留原始 T×R 语义 ==========
        if self.config.get('resize_input_for_experts', False):
            try:
                x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
            except Exception as e:
                if self.is_logger:
                    print(f"[WARN] 输入插值到(256,256)失败：{e}，保持原始形状 {tuple(x.shape)}")

        def _zeros_like_x(bsz: int, channels: int, like: torch.Tensor) -> torch.Tensor:
            return torch.zeros(bsz, channels, *like.shape[2:], device=like.device, dtype=like.dtype)

        def _safe_call_expert(expert_idx: int, x_slice: torch.Tensor) -> torch.Tensor:
            """
            安全地调用 self.experts[expert_idx](x_slice, **kwargs)
            若异常或返回 None，用零张量兜底，形状为 [1, C, H, W]
            """
            try:
                out = self.experts[expert_idx](x_slice, **kwargs)
                if out is None:
                    if self.is_logger:
                        print(f"[WARN] 专家 {expert_idx} 返回 None，使用零张量代替")
                    return _zeros_like_x(1, C, x_slice)
                return out
            except Exception as e:
                if self.is_logger:
                    print(f"[ERROR] 专家 {expert_idx} 处理异常：{e}，使用零张量代替")
                return _zeros_like_x(1, C, x_slice)

        def _most_common_target_shape(all_shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
            """
            从收集的空间形状中选出维度数最常见、且每个维度大小最常见的 target_shape
            """
            if not all_shapes:
                # 回退：使用当前 x 的空间维度
                return tuple(x.shape[2:])

            ndims = [len(s) for s in all_shapes]
            # 选择最常见的维度数
            target_ndim = Counter(ndims).most_common(1)[0][0]
            shapes_same_ndim = [s for s in all_shapes if len(s) == target_ndim]
            # 在每个维度上采用最常见的大小
            target = []
            for d in range(target_ndim):
                dim_sizes = [s[d] for s in shapes_same_ndim]
                target.append(Counter(dim_sizes).most_common(1)[0][0])
            return tuple(target)

        def _resize_to(out: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
            """
            将 out 插值到 target_shape，保持 [B, C, ...] 结构
            """
            if out.dim() < 4 or out.shape[2:] == target_shape:
                return out
            try:
                if len(target_shape) == 3:
                    mode = 'trilinear'
                elif len(target_shape) == 2:
                    mode = 'bilinear'
                else:
                    mode = 'linear'
                return F.interpolate(out, size=target_shape, mode=mode, align_corners=True)
            except Exception as e:
                if self.is_logger:
                    print(f"[ERROR] 插值失败: {e}；用零张量兜底。in={tuple(out.shape)} target={target_shape}")
                return torch.zeros(out.shape[0], out.shape[1], *target_shape, device=out.device, dtype=out.dtype)

        def _apply_sample_weights(batch_tensor: torch.Tensor, sample_wts: torch.Tensor) -> torch.Tensor:
            """
            batch_tensor: [B, C, H, W]
            sample_wts  : [B] 或 [B, 1]
            返回逐样本广播乘权的结果
            """
            if sample_wts.dim() == 1:
                sample_wts = sample_wts.view(-1, 1, 1, 1)
            elif sample_wts.dim() == 2 and sample_wts.shape[1] == 1:
                sample_wts = sample_wts.view(-1, 1, 1, 1)
            return batch_tensor * sample_wts

        collected: List[Tuple[List[torch.Tensor], torch.Tensor]] = []
        # 仅用于日志：记录每个（组, 子模型）或专家的输出形状
        shape_log = {}

        if type_weights is not None:
            # ------------------------- 分组模式（组内 T 子模型按类型权重加权） -------------------------
            T = self.v_type_num

            for k_idx in range(top_k):
                group_ids: torch.Tensor = expert_indices[:, k_idx]   # [B]
                group_wts: torch.Tensor = routing_weights[:, k_idx]  # [B]
                batch_outputs: List[torch.Tensor] = []

                for b in range(batch_size):
                    g = int(group_ids[b].item())  # 组ID
                    weighted_sum = None
                    # 组内 T 个子模型（速度图类型 0..T-1），全局索引：g*T + t
                    for t in range(T):
                        expert_idx = g * T + t
                        out_bt = _safe_call_expert(expert_idx, x[b:b+1])  # [1, C, h, w]

                        # 日志：记录 (g,t) 的输出形状
                        if self.is_logger:
                            shape_log.setdefault(("group", g, t), []).append(tuple(out_bt.shape))

                        # 乘以类型权重（标量）
                        tw = type_weights[b, t].view(1, 1, 1, 1)  # [1,1,1,1]
                        out_bt = out_bt * tw

                        weighted_sum = out_bt if weighted_sum is None else (weighted_sum + out_bt)

                    # 该样本在组 g 内的加权和
                    batch_outputs.append(weighted_sum)

                collected.append((batch_outputs, group_wts))

        else:
            # ------------------------- 直接专家模式（无分组） -------------------------
            for k_idx in range(top_k):
                indices: torch.Tensor = expert_indices[:, k_idx]   # [B]
                weights: torch.Tensor = routing_weights[:, k_idx]  # [B]
                batch_outputs: List[torch.Tensor] = []

                for b in range(batch_size):
                    expert_idx = int(indices[b].item())
                    out_bt = _safe_call_expert(expert_idx, x[b:b+1])  # [1, C, h, w]

                    if self.is_logger:
                        shape_log.setdefault(("expert", expert_idx), []).append(tuple(out_bt.shape))

                    batch_outputs.append(out_bt)

                collected.append((batch_outputs, weights))

        # ----------------------------- 形状对齐：确定目标空间大小 -----------------------------
        # 收集所有输出的空间形状
        all_spatial_shapes: List[Tuple[int, ...]] = []
        for batch_outputs, _ in collected:
            for out in batch_outputs:
                if out is not None and out.dim() >= 4:
                    all_spatial_shapes.append(tuple(out.shape[2:]))

        if not all_spatial_shapes:
            # 所有输出都无效时兜底：返回 top_k 个 [B, C, Hx, Wx] 的零张量（Hx,Wx 取自 x）
            if self.is_logger:
                print("[WARN] 没有有效输出，返回零张量列表")
            return [torch.zeros(batch_size, C, *x.shape[2:], device=device, dtype=x.dtype) for _ in range(top_k)]

        target_shape = _most_common_target_shape(all_spatial_shapes)
        if self.is_logger:
            print(f"[INFO] 目标形状（对齐用）: {target_shape}")

        # ----------------------------- 对齐 + 按样本路由权重加权 -----------------------------
        outputs: List[torch.Tensor] = []
        for batch_outputs, sample_wts in collected:
            adjusted = []
            for out in batch_outputs:
                out_adj = _resize_to(out, target_shape)
                adjusted.append(out_adj)

            # 尝试合并为 [B, C, H, W]
            try:
                stacked = torch.cat(adjusted, dim=0)  # [B, C, H, W]
            except Exception as e:
                if self.is_logger:
                    bad_shapes = [tuple(a.shape) for a in adjusted]
                    print(f"[ERROR] 合并组/专家输出失败：{e}；形状列表：{bad_shapes}；跳过该条目")
                # 跳过这个 k_idx
                continue

            weighted = _apply_sample_weights(stacked, sample_wts)  # [B, C, H, W]
            outputs.append(weighted)

        if not outputs:
            if self.is_logger:
                print("[WARN] 没有有效的组/专家输出，返回零张量列表")
            return [torch.zeros(batch_size, C, *target_shape, device=device, dtype=x.dtype) for _ in range(top_k)]

        if self.is_logger and len(shape_log) > 0:
            # 打印若干条样例形状以辅助调试（避免刷屏）
            print("[DEBUG] 专家/组内子模型输出形状样例（最多每类展示3条）：")
            shown = 0
            for key, shapes in shape_log.items():
                print(f"  {key}: {shapes[:3]}{' ...' if len(shapes) > 3 else ''}")
                shown += 1
                if shown >= 20:  # 控制日志长度
                    print("  ...（更多省略）")
                    break

        return outputs
            
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

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Union, Tuple, Callable, Optional, Any
from collections import Counter

from .base_model import BaseModel
from .task_router import TaskAwareRouter
from ..layers.spectral_convolution import SpectralConv
from .task_dependent_router import TaskDependentRouter
from .SWActivate import GroupActMerge, SWActMerge
from .merge_processer import MeanMix, SumMix, LinearMix, AttentionMix
from torch.nn.parameter import UninitializedParameter
from .expert_memory_proxy import ExpertMemoryProxy

# ============================== Router ==============================
class Router(nn.Module):
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
        self.alpha = alpha
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        logits = self.router(x)
        if self.noisy_gating and self.training:
            logits = logits + torch.randn_like(logits)
        routing_weights = F.softmax(logits, dim=-1)

        top_k = int(self.top_k)
        num_experts = int(self.num_experts)
        assert num_experts > 0 and 1 <= top_k <= num_experts

        top_k_weights, top_k_indices = torch.topk(
            routing_weights, k=top_k, dim=-1, largest=True, sorted=True
        )
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        w_top_k = num_experts - top_k
        if w_top_k > 0:
            B, N = routing_weights.shape
            selected_mask = torch.zeros(B, N, dtype=torch.bool, device=routing_weights.device)
            selected_mask.scatter_(dim=1, index=top_k_indices, src=torch.ones_like(top_k_indices, dtype=torch.bool))
            remaining = routing_weights.clone().masked_fill(selected_mask, float('-inf'))
            w_weights, w_indices = torch.topk(remaining, k=w_top_k, dim=-1, largest=True, sorted=True)
            w_weights = torch.where(torch.isfinite(w_weights), w_weights, torch.zeros_like(w_weights))
            w_weights = w_weights / w_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            w_weights, w_indices = None, None

        if self.training and getattr(self, "alpha", 0.0) > 0.0:
            if top_k_indices.numel() == 0 or routing_weights.numel() == 0:
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


# ============================== ContinuousCRF（可选） ==============================
class ContinuousCRF(nn.Module):
    def __init__(self, window_size=5, num_iterations=5, lambda_feat=1.0, lambda_pos=1.0):
        super().__init__()
        self.window_size = window_size
        self.num_iterations = num_iterations
        self.lambda_feat = lambda_feat
        self.lambda_pos = lambda_pos
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, unary_pred, features):
        B, _, H, W = unary_pred.shape
        _, C, _, _ = features.shape
        device = unary_pred.device

        mu = unary_pred.clone()
        for _ in range(self.num_iterations):
            mu_new = torch.zeros_like(mu)
            for dx in range(-self.window_size, self.window_size + 1):
                for dy in range(-self.window_size, self.window_size + 1):
                    if dx == 0 and dy == 0: continue
                    f_shift = self.shift_tensor(features, dx, dy)
                    mu_shift = self.shift_tensor(mu, dx, dy)
                    diff_feat = (features - f_shift).pow(2).sum(dim=1, keepdim=True)
                    diff_pos = (dx**2 + dy**2)
                    weight = torch.exp(-self.lambda_feat * diff_feat - self.lambda_pos * diff_pos)
                    mu_new += weight * mu_shift
            norm_factor = 1 + self.w * (2 * self.window_size + 1)**2
            mu = (unary_pred + self.w * mu_new) / norm_factor
        return mu

    def shift_tensor(self, x, dx, dy):
        B, C, H, W = x.shape
        shifted = torch.zeros_like(x)

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

        shifted[:, :, tgt_x1:tgt_x2, tgt_y1:tgt_y2] = x[:, :, src_x1:src_x2, src_y1:src_y2]
        return shifted


# ============================== 轻量编码块（可选） ==============================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, ch, expansion=2):
        super().__init__()
        mid = ch * expansion // 2
        self.f = nn.Sequential(
            ConvBNAct(ch, mid, k=1, s=1, p=0),
            ConvBNAct(mid, mid, k=3, s=1, p=1),
            nn.Conv2d(mid, ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.act(self.f(x) + x)

class AdaptiveDownsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv_21 = ConvBNAct(ch_in, ch_out, k=3, s=(2,1), p=1)
        self.conv_12 = ConvBNAct(ch_in, ch_out, k=3, s=(1,2), p=1)
        self.conv_22 = ConvBNAct(ch_in, ch_out, k=3, s=(2,2), p=1)

    def forward(self, x):
        B, C, H, W = x.shape
        aspect = H / max(W, 1)
        if aspect > 1.5: return self.conv_21(x)
        if aspect < 1 / 1.5: return self.conv_12(x)
        return self.conv_22(x)

class TimeSpaceProjectorFlexible(nn.Module):
    def __init__(self, base_ch=32, bottleneck_depth=3, min_side_target=96):
        super().__init__()
        self.min_side_target = min_side_target
        self.stem = nn.Sequential(
            ConvBNAct(1, base_ch,   k=3, s=1, p=1),
            ConvBNAct(base_ch, base_ch, k=3, s=1, p=1),
        )
        down_channels = [
            (base_ch, base_ch * 2),
            (base_ch * 2, base_ch * 4),
            (base_ch * 4, base_ch * 4),
            (base_ch * 4, base_ch * 4),
            (base_ch * 4, base_ch * 4),
        ]
        self.down_stages = nn.ModuleList(AdaptiveDownsample(cin, cout) for cin, cout in down_channels)
        self.min_required_down_stages = 2
        ch = base_ch*4
        self.bottleneck = nn.Sequential(*[ResidualBlock(ch, expansion=2) for _ in range(bottleneck_depth)])
        self.head = nn.Sequential(
            ConvBNAct(ch, ch//2, k=3, s=1, p=1),
            ConvBNAct(ch//2, ch//4, k=3, s=1, p=1),
            nn.Conv2d(ch//4, 1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.out_pool = nn.AdaptiveAvgPool2d((70, 70))

    def forward(self, x):
        x = self.stem(x)
        H, W = x.shape[-2:]
        need_force_down = min(H, W) > self.min_side_target
        for idx, d in enumerate(self.down_stages):
            if min(H, W) <= self.min_side_target and not need_force_down:
                break
            x = d(x)
            H, W = x.shape[-2:]
            if need_force_down and idx + 1 >= self.min_required_down_stages:
                need_force_down = False
        x = self.bottleneck(x)
        x = self.head(x)
        x = self.out_pool(x)
        return x


# ============================== MOEOperator ==============================
class MOEOperator(BaseModel, name='MOE'):
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
        moe_mode: str = 'standard',
        **kwargs
    ):
        super().__init__()
        self.config = kwargs
        self.moe_mode = moe_mode
        self.config.setdefault('moe_mode', moe_mode)
        
        valid_modes = {'standard', 'group', 'velocity_type'}
        if self.moe_mode not in valid_modes:
            raise ValueError(f"Unsupported moe_mode: {self.moe_mode}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.top_k = top_k

        if v_type_num is None:
            self.v_type_num = 10 if self.config.get('is_specific', True) else 3
        else:
            self.v_type_num = v_type_num

        if self.moe_mode == 'velocity_type' and (self.v_type_num is None or self.v_type_num <= 0):
            self.v_type_num = len(experts)

        self.experts = nn.ModuleList(experts)
        # >>> 显存管理透明代理（开关可由 config 控制）
        use_proxy = self.config.get('use_expert_memory_proxy', True)
        self.expert_proxy: Optional[ExpertMemoryProxy] = None
        if use_proxy:
            self.expert_proxy = ExpertMemoryProxy(
                experts=list(self.experts),
                device=self.config.get('device', 'cuda'),
                cache_size=self.config.get('expert_cache_size', 2),
                amp_dtype=torch.bfloat16,
            )
        
        # 派生属性
        if self.moe_mode == 'velocity_type':
            self.num_experts = len(experts)
            self.types_per_group = 0
        elif self.moe_mode == 'group':
            self.types_per_group = int(self.v_type_num)
            if self.types_per_group <= 0:
                raise ValueError("group 模式需要正整数的 v_type_num。")
            if len(experts) % self.types_per_group != 0 and is_logger:
                print(f"[WARN] 专家数量 {len(experts)} 不能被每组类型数 {self.types_per_group} 整除，将按整除结果截断。")
            self.num_experts = max(1, len(experts) // self.types_per_group)
        else:
            self.num_experts = len(experts)
            self.types_per_group = 0

        self.noisy_gating = noisy_gating
        self.fusion_type = fusion_type
        self.router_type = router_type
        self._type_weight_warned = False

        if self.moe_mode == 'velocity_type':
            self.router_type = 'velocity_type'

        if self.moe_mode != 'velocity_type':
            self.top_k = min(self.top_k, self.num_experts)
            self.w_k = max(0, self.num_experts - top_k)

            if self.num_experts > 1:
                self.alpha: float = 0.1
            else:
                self.alpha: float = 0.0

            if self.router_type == 'task_aware' and self.fusion_type == 'swa':
                raise ValueError("task_aware 路由当前不支持 'swa' 融合。")

            if router_type == 'basic':
                self.router = Router(
                    input_dim=in_channels,
                    num_experts=self.num_experts,
                    hidden_dim=router_hidden_dim,
                    top_k=self.top_k,
                    noisy_gating=noisy_gating,
                    alpha=self.alpha,
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
                    alpha=self.alpha,
                )
            else:
                raise ValueError(f"不支持的路由器类型: {router_type}")

            # s_processor / w_processor
            for t in ['s_processor', 'w_processor']:
                cfg_key = f'{t}_type'
                module = None
                k = self.top_k if t == 's_processor' else self.w_k
                if self.config.get(cfg_key, None) == 'linear':
                    module = LinearMix(k, self.out_channels)
                elif self.config.get(cfg_key, None) == 'attention':
                    module = AttentionMix(
                        input_resolution=256, patch_size=16, in_channels=self.in_channels,out_channels=self.in_channels,
                        width=512, layers=6, heads=8, num_experts=k,
                        use_cls_expert=False,
                    )
                elif self.config.get(cfg_key, None) == 'mean':
                    module = MeanMix(self.out_channels)
                elif self.config.get(cfg_key, None) == 'sum':
                    module = SumMix(self.out_channels)
                setattr(self, t, module)

            # 融合层
            if fusion_type == 'linear':
                self.fusion = LinearMix(self.top_k, self.out_channels)
            elif fusion_type == 'attention':
                self.fusion = AttentionMix(
                    input_resolution=256, patch_size=16, in_channels=self.in_channels, out_channels=self.in_channels,
                    width=512, layers=6, heads=8, num_experts=self.top_k,
                    use_cls_expert=False,
                )
            elif fusion_type == 'swa':
                self.s_act = GroupActMerge(processor=self.s_processor)
                self.w_act = GroupActMerge(processor=self.w_processor)
                self.sw_act = SWActMerge(beta=self.config.get('beta', 0.5))
            elif fusion_type == 'basic':
                self.fusion = SumMix(self.in_channels)
            else:
                raise ValueError(f"未支持的融合类型: {fusion_type}")
        else:
            self.top_k = 0
            self.w_k = 0
            self.alpha = 0.0
            self.router = None
            for attr in ['s_processor', 'w_processor', 'fusion', 's_act', 'w_act', 'sw_act']:
                setattr(self, attr, None)

        self.is_logger = is_logger
        self.reset_parameters_()

    # ---------------- Forward ----------------
    def forward(self, x, class_weights, task_features=None, **kwargs) -> Tuple[torch.Tensor]:
        if self.moe_mode == 'velocity_type':
            return self._forward_velocity_type(x, class_weights, **kwargs)

        batch_size = x.shape[0]
        device = x.device

        if self.moe_mode == 'group':
            if class_weights is None:
                type_weights = None
                if self.is_logger and not self._type_weight_warned:
                    print("[WARN] 未提供类型权重，分组模式将退化为普通专家模式。")
                    self._type_weight_warned = True
            else:
                type_weights = class_weights
        else:
            type_weights = None
            if class_weights is not None and self.is_logger and not self._type_weight_warned:
                print("[WARN] 当前 moe_mode 非 'group'，忽略 encoder 的类型权重。")
                self._type_weight_warned = True

        x_flat = x.mean(dim=(2, 3)).view(batch_size, -1)

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

        s_combined = torch.stack(s_outputs, dim=1)  # (B, k, C, h, w)
        if self.fusion_type == 'swa':
            w_combined = torch.stack(w_outputs, dim=1)

        if self.fusion_type == 'swa':
            s_combined = self.s_act(s_outputs)
            w_combined = self.w_act(w_outputs)
            combined_output = self.sw_act(s_combined, w_combined)
        else:
            combined_output = self.fusion(s_combined)

        if combined_output.shape[2] != 70 or combined_output.shape[3] != 70:
            combined_output = F.interpolate(combined_output, size=(70, 70), mode='bilinear', align_corners=False)
        
        return combined_output, aux_loss

    def _most_common_shape(
        self,
        all_shapes: List[Tuple[int, ...]],
        default_shape: Optional[Tuple[int, ...]] = None
    ) -> Tuple[int, ...]:
        """
        从收集到的空间形状中选出 target_shape。
        若 all_shapes 为空则回退到 default_shape（若提供）或 x 的空间形状。
        """
        if not all_shapes:
            return tuple(default_shape)
        ndims = [len(s) for s in all_shapes]
        target_ndim = Counter(ndims).most_common(1)[0][0]
        shapes_same_ndim = [s for s in all_shapes if len(s) == target_ndim]
        target = []
        for d in range(target_ndim):
            dim_sizes = [s[d] for s in shapes_same_ndim]
            target.append(Counter(dim_sizes).most_common(1)[0][0])
        return tuple(target)
    
    def _resize_to_shape(
        self, 
        out: torch.Tensor, 
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        if out is None or out.dim() < 4 or out.shape[2:] == target_shape:
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
            print(f"[ERROR] 插值失败: {e}；用零张量兜底。in={tuple(out.shape)} target={target_shape}")
            return torch.zeros(out.shape[0], out.shape[1], *target_shape, device=out.device, dtype=out.dtype)
    
    def _forward_velocity_type(self, x: torch.Tensor, class_weights: Optional[torch.Tensor], **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = x.shape[0]
        device = x.device

        num_available = len(self.experts)
        if num_available == 0:
            raise RuntimeError("velocity_type 模式需要至少一个专家。")

        weights = class_weights
        if weights is None:
            if self.is_logger:
                print("[WARN] type_weights 未提供，使用均匀权重。")
            num_types = min(self.v_type_num or num_available, num_available)
            weights = torch.ones(batch_size, num_types, device=device, dtype=x.dtype) / float(num_types)
        else:
            if weights.dim() == 1:
                weights = weights.view(1, -1).expand(batch_size, -1)
            elif weights.dim() == 2 and weights.size(0) == 1 and batch_size > 1:
                weights = weights.expand(batch_size, -1)
            elif weights.dim() == 2 and weights.size(0) != batch_size:
                raise ValueError(f"type_weights batch 大小 {weights.size(0)} 与输入 {batch_size} 不匹配")
            elif weights.dim() != 2:
                raise ValueError(f"Unsupported type_weights shape: {tuple(weights.shape)}")
            weights = weights.to(device=device, dtype=x.dtype)

        expected_types = self.v_type_num or weights.size(1)
        total_experts = num_available
        num_types = min(expected_types, weights.size(1), total_experts)
        if num_types <= 0:
            raise ValueError("velocity_type 模式需要至少一个专家与对应类型权重。")
        if num_types < weights.size(1) and self.is_logger:
            print(f"[WARN] type_weights 列数 {weights.size(1)} 超过可用专家数量 {total_experts}，仅使用前 {num_types} 个。")

        weights = weights[:, :num_types]
        C = self.out_channels if self.out_channels > 0 else x.shape[1]

        def _zeros_like() -> torch.Tensor:
            spatial = x.shape[2:] if x.dim() > 2 else (1, 1)
            return torch.zeros(batch_size, C, *spatial, device=device, dtype=x.dtype)

        outputs: List[torch.Tensor] = []
        spatial_shapes: List[Tuple[int, ...]] = []

        for idx in range(num_types):
            try:
                if self.expert_proxy is None:
                    out = self.experts[idx](x, **kwargs)
                else:
                    out = self.expert_proxy.forward_expert(idx, x, **kwargs)
                if out is None:
                    raise RuntimeError("expert returned None")
            except Exception as exc:
                if self.is_logger:
                    print(f"[WARN] 速度类型专家 {idx} 前向失败: {exc}")
                out = _zeros_like()

            if not torch.is_tensor(out):
                if self.is_logger:
                    print(f"[WARN] 速度类型专家 {idx} 返回非张量，使用零张量代替。")
                out = _zeros_like()

            if out.dim() == 3: out = out.unsqueeze(1)
            elif out.dim() == 2: out = out.view(batch_size, -1, 1, 1)
            elif out.dim() < 2: out = out.view(batch_size, 1, 1, 1)
            elif out.dim() > 4: out = out.view(out.size(0), out.size(1), out.size(2), -1)
            if out.size(0) != batch_size:
                if self.is_logger:
                    print(f"[WARN] 专家 {idx} 输出 batch 大小 {out.size(0)} 异常，使用零张量替换。")
                out = _zeros_like()

            outputs.append(out)
            spatial_shapes.append(tuple(out.shape[2:]))

        if not outputs:
            combined = _zeros_like()
            return combined, None

        target_shape = self._most_common_shape(spatial_shapes, default_shape=tuple(outputs[0].shape[2:]))
        aligned = [self._resize_to_shape(out, target_shape) for out in outputs]
        stacked = torch.stack(aligned, dim=1)  # [B, T, C, H, W]
        weight_tensor = weights.view(batch_size, num_types, 1, 1, 1)
        combined = (stacked * weight_tensor).sum(dim=1)
        
        if combined.shape[2] != 70 or combined.shape[3] != 70:
            combined = F.interpolate(combined, size=(70, 70), mode='bilinear', align_corners=False)
        
        return combined, None

    # ---------------- 工具：按专家聚合 & 回填 ----------------
    def _pack_by_expert(self, indices_1d: torch.Tensor, x: torch.Tensor):
        """
        indices_1d: [B] 每个样本的 expert_idx（已是 experts 的直接索引）
        x:          [B, C, H, W]
        返回:
          routed: {expert_idx -> x_sub_cat [n_e, C, H, W]}
          meta  : {expert_idx -> (positions_list, sizes_list)}  # 回填需要
        """
        routed, meta = {}, {}
        for pos, eid in enumerate(indices_1d.tolist()):
            x_slice = x[pos:pos+1]  # [1,C,H,W]
            if eid not in routed:
                routed[eid] = [x_slice]
                meta[eid] = ([pos], [1])
            else:
                routed[eid].append(x_slice)
                meta[eid][0].append(pos)
                meta[eid][1].append(1)
        for eid in routed:
            routed[eid] = torch.cat(routed[eid], dim=0).contiguous()
        return routed, meta

    def _scatter_back(self,
                      y_by_expert: Dict[int, torch.Tensor],
                      meta,
                      B: int,
                      C: int,
                      shape_hw: Tuple[int, int],
                      device,
                      dtype):
        """
        y_by_expert: {expert_idx: y_sub_cat [n_e, C, H, W]}
        meta       : _pack_by_expert 返回的 meta
        返回: y_full [B, C, H, W]
        """
        H, W = shape_hw
        y_full = torch.zeros(B, C, H, W, device=device, dtype=dtype)
        for eid, y_sub in y_by_expert.items():
            positions, _ = meta[eid]
            for i, pos in enumerate(positions):
                y_full[pos:pos+1] = y_sub[i:i+1]
        return y_full

    # ---------------- 核心：批处理版激活组计算 ----------------
    def _process_activation_group(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,   # [B, top_k]
        routing_weights: torch.Tensor,  # [B, top_k] 或 None
        k: int,
        batch_size: int,
        device,
        type_weights: Optional[torch.Tensor] = None,  # [B, T] 或 None
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        要点：
        1) 按专家聚合，一次前向（forward_many 或 per-expert 批处理）；
        2) 像老版本一样：把“形状对齐 + 按样本路由加权”放在最后统一执行；
        3) 提示信息与老版本一致，且 print(..., flush=True) 及时打印。
        返回：长度为 top_k 的列表，每个 [B,C,H,W]。
        """
        # ------------------ 打印封装（及时 flush） ------------------
        def _log(msg: str):
            if getattr(self, "is_logger", False):
                print(msg, flush=True)

        # ------------------ 基本校验与准备 ------------------
        top_k = int(k)
        if top_k <= 0:
            return []

        if expert_indices is None:
            raise ValueError("expert_indices 为空，无法获取路由结果。")
        if expert_indices.size(1) < top_k:
            raise ValueError(f"expert_indices 的列数不足以支持 top_k={top_k}")

        if routing_weights is None:
            routing_weights = torch.ones(batch_size, top_k, device=device, dtype=x.dtype)
        elif routing_weights.size(1) < top_k:
            raise ValueError(f"routing_weights 的列数不足以支持 top_k={top_k}")
        else:
            routing_weights = routing_weights.to(device=device, dtype=x.dtype)

        C = self.out_channels if getattr(self, "out_channels", 0) > 0 else x.size(1)
        total_experts = len(self.experts)
        grouped_mode = (getattr(self, "moe_mode", None) == "group") and (type_weights is not None)

        # ---- 分组一致性与类型权重准备 ----
        if grouped_mode:
            if type_weights.dim() == 1:
                type_weights = type_weights.view(1, -1).expand(batch_size, -1)
            if type_weights.size(0) != batch_size:
                raise ValueError(f"type_weights batch 维度不一致: {type_weights.size(0)} vs {batch_size}")

            T = getattr(self, "types_per_group", None)
            if T is None:
                _log("[WARN] 未设置 types_per_group，降级为直接模式。")
                grouped_mode = False
                type_weights = None
            elif type_weights.size(1) != T:
                _log(f"[WARN] type_weights 列={type_weights.size(1)} 与组内专家数 T={T} 不一致，降级为直接模式。")
                grouped_mode = False
                type_weights = None
            else:
                expected_total = self.num_experts * T
                experts_per_group = total_experts // max(1, self.num_experts)
                if (expected_total == 0 or total_experts < expected_total
                    or total_experts % T != 0 or experts_per_group != T):
                    _log(f"[WARN] 专家数量 {total_experts} 与分组结构不匹配（期望 {expected_total}），降级为直接模式。")
                    grouped_mode = False
                    type_weights = None
                else:
                    type_weights = type_weights.to(device=device, dtype=x.dtype)

        # ------------------ 工具函数（与老版本语义一致） ------------------
        def _zeros_like_x(bsz: int, channels: int, like: torch.Tensor) -> torch.Tensor:
            return torch.zeros(bsz, channels, *like.shape[2:], device=like.device, dtype=like.dtype)

        def _most_common_target_shape(
            all_shapes: List[Tuple[int, ...]],
            default_shape: Optional[Tuple[int, ...]] = None
        ) -> Tuple[int, ...]:
            """
            从收集到的空间形状中选出 target_shape。
            若 all_shapes 为空则回退到 default_shape（若提供）或 x 的空间形状。
            """
            if not all_shapes:
                return tuple(default_shape) if default_shape is not None else tuple(x.shape[2:])
            ndims = [len(s) for s in all_shapes]
            target_ndim = Counter(ndims).most_common(1)[0][0]
            shapes_same_ndim = [s for s in all_shapes if len(s) == target_ndim]
            target = []
            for d in range(target_ndim):
                dim_sizes = [s[d] for s in shapes_same_ndim]
                target.append(Counter(dim_sizes).most_common(1)[0][0])
            return tuple(target)

        def _resize_to(out: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
            if out is None or out.dim() < 4 or out.shape[2:] == target_shape:
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
                _log(f"[ERROR] 插值失败: {e}；用零张量兜底。in={tuple(out.shape)} target={target_shape}")
                return torch.zeros(out.shape[0], out.shape[1], *target_shape, device=out.device, dtype=out.dtype)

        def _apply_sample_weights(batch_tensor: torch.Tensor, sample_wts: torch.Tensor) -> torch.Tensor:
            if sample_wts.dim() == 1:
                sample_wts = sample_wts.view(-1, 1, 1, 1)
            elif sample_wts.dim() == 2 and sample_wts.shape[1] == 1:
                sample_wts = sample_wts.view(-1, 1, 1, 1)
            return batch_tensor * sample_wts

        # 形状日志，与老版本保持一致风格（只存 shape，避免持有显存）
        shape_log: Dict[Any, List[Tuple[int, ...]]] = {}

        # ------------------ 主过程：逐 k 计算“未对齐、未乘路由权重”的逐样本输出 ------------------
        # 每个 k_idx -> (List[torch.Tensor: [1,C,h,w] * B], sample_weights: [B])
        collected: List[Tuple[List[torch.Tensor], torch.Tensor]] = []

        for k_idx in range(top_k):
            indices_k: torch.Tensor = expert_indices[:, k_idx]   # [B]
            weights_k: torch.Tensor = routing_weights[:, k_idx]  # [B]
            weights_k = weights_k.to(device=device, dtype=x.dtype)

            if not grouped_mode:
                # -------- 直接专家模式：按专家聚合、一次前向、scatter 回填为 [B,C,h,w]，但先不乘路由权重 --------
                routed, meta = self._pack_by_expert(indices_k, x)

                if not routed:
                    # 这一列 k 没有任何路由命中
                    batch_outputs = [_zeros_like_x(1, C, x[b:b+1]) for b in range(batch_size)]
                    collected.append((batch_outputs, weights_k))
                    continue

                # 前向（代理 or 直连），带兜底
                try:
                    if getattr(self, "expert_proxy", None) is not None:
                        y_by_eid = self.expert_proxy.forward_many(routed)
                    else:
                        y_by_eid = {}
                        for eid, x_sub in routed.items():
                            try:
                                y_by_eid[eid] = self.experts[eid](x_sub, **kwargs)
                            except Exception as e:
                                _log(f"[ERROR] 专家 {eid} 处理异常：{e}，使用零张量代替")
                                y_by_eid[eid] = _zeros_like_x(x_sub.size(0), C, x_sub)
                                import traceback
                                traceback.print_exc()
                                exit(0)
                except Exception as e:
                    _log(f"[ERROR] forward_many 异常：{e}；该列以零张量兜底")
                    batch_outputs = [_zeros_like_x(1, C, x[b:b+1]) for b in range(batch_size)]
                    collected.append((batch_outputs, weights_k))
                    continue

                # 形状日志（每个专家一条）
                for eid, y in y_by_eid.items():
                    if y is not None:
                        shape_log.setdefault(("expert", int(eid)), []).append(tuple(y.shape))

                # 暂不全局对齐：先局部统一到本列最常见形状，便于 scatter
                spatial_shapes = [tuple(y.shape[2:]) for y in y_by_eid.values() if y is not None and y.dim() >= 4]
                target_shape_local = _most_common_target_shape(spatial_shapes, default_shape=tuple(x.shape[2:]))
                for eid in list(y_by_eid.keys()):
                    y_by_eid[eid] = _resize_to(y_by_eid[eid], target_shape_local).contiguous()

                y_full = self._scatter_back(y_by_eid, meta, batch_size, C, target_shape_local, device, x.dtype)

                # 拆为逐样本 [1,C,h,w]（后续统一全局对齐）
                batch_outputs = [y_full[b:b+1] for b in range(batch_size)]
                collected.append((batch_outputs, weights_k))

            else:
                # -------- 分组模式：把 (group, t) 展开为具体专家，合一批处理；先按 type_weights 加权到样本，不乘路由权重 --------
                T = self.types_per_group
                routed_dict: Dict[int, torch.Tensor] = {}
                meta_dict: Dict[int, Tuple[List[int], List[int]]] = {}

                group_ids = indices_k  # [B]
                for b in range(batch_size):
                    g = int(group_ids[b].item())
                    if not (0 <= g < self.num_experts):
                        continue
                    for t in range(T):
                        eid = g * T + t
                        if eid >= total_experts:
                            continue
                        x_slice = x[b:b+1]
                        if eid not in routed_dict:
                            routed_dict[eid] = [x_slice]
                            meta_dict[eid] = ([b], [1])
                        else:
                            routed_dict[eid].append(x_slice)
                            meta_dict[eid][0].append(b)
                            meta_dict[eid][1].append(1)

                if not routed_dict:
                    batch_outputs = [_zeros_like_x(1, C, x[b:b+1]) for b in range(batch_size)]
                    collected.append((batch_outputs, weights_k))
                    continue

                # 拼接 mini-batch
                for eid in routed_dict:
                    routed_dict[eid] = torch.cat(routed_dict[eid], dim=0).contiguous()

                # 前向（代理 or 直连），带兜底
                try:
                    if getattr(self, "expert_proxy", None) is not None:
                        y_by_eid = self.expert_proxy.forward_many(routed_dict)
                    else:
                        y_by_eid = {}
                        for eid, x_sub in routed_dict.items():
                            try:
                                y_by_eid[eid] = self.experts[eid](x_sub, **kwargs)
                            except Exception as e:
                                _log(f"[ERROR] 专家 {eid} 处理异常：{e}，使用零张量代替")
                                y_by_eid[eid] = _zeros_like_x(x_sub.size(0), C, x_sub)
                except Exception as e:
                    _log(f"[ERROR] forward_many 异常：{e}；该列以零张量兜底")
                    batch_outputs = [_zeros_like_x(1, C, x[b:b+1]) for b in range(batch_size)]
                    collected.append((batch_outputs, weights_k))
                    continue

                # 形状日志（按组内 t 记录）
                for eid, y in y_by_eid.items():
                    if y is not None:
                        g = int(eid // T); t = int(eid % T)
                        shape_log.setdefault(("group", g, t), []).append(tuple(y.shape))

                # 局部统一到本列最常见形状，便于逐样本累加
                spatial_shapes = [tuple(y.shape[2:]) for y in y_by_eid.values() if y is not None and y.dim() >= 4]
                target_shape_local = _most_common_target_shape(spatial_shapes, default_shape=tuple(x.shape[2:]))

                for eid in list(y_by_eid.keys()):
                    y_by_eid[eid] = _resize_to(y_by_eid[eid], target_shape_local).contiguous()

                H, W = target_shape_local
                y_group = torch.zeros(batch_size, C, H, W, device=device, dtype=x.dtype)

                # 向量化回填：对同一 eid 的所有样本一次性 index_add_
                for eid, y_sub in y_by_eid.items():
                    positions, _ = meta_dict[eid]         # List[int]，长度 N
                    if len(positions) == 0:
                        continue
                    idx = torch.tensor(positions, device=device, dtype=torch.long)  # [N]
                    local_t = eid % T
                    tw = type_weights[idx, local_t].view(-1, 1, 1, 1)               # [N,1,1,1]
                    # y_sub: [N,C,H,W]（已被局部对齐）
                    y_group.index_add_(0, idx, y_sub * tw)

                # 拆成逐样本 [1,C,h,w]；后续再统一“全局对齐 + 按样本路由加权”
                batch_outputs = [y_group[b:b+1] for b in range(batch_size)]
                collected.append((batch_outputs, weights_k))

        # ------------------ 统一形状对齐 + 统一按样本路由加权（与老版本一致） ------------------
        # 收集所有输出的空间形状
        all_spatial_shapes: List[Tuple[int, ...]] = []
        for batch_outputs, _ in collected:
            for out in batch_outputs:
                if out is not None and out.dim() >= 4:
                    all_spatial_shapes.append(tuple(out.shape[2:]))

        if not all_spatial_shapes:
            _log("[WARN] 没有有效输出，返回零张量列表")
            return [torch.zeros(batch_size, C, *x.shape[2:], device=device, dtype=x.dtype) for _ in range(top_k)]

        target_shape = _most_common_target_shape(all_spatial_shapes)
        _log(f"[INFO] 目标形状（对齐用）: {target_shape}")

        outputs: List[torch.Tensor] = []
        for batch_outputs, sample_wts in collected:
            adjusted = []
            for out in batch_outputs:
                out_adj = _resize_to(out, target_shape)
                adjusted.append(out_adj)

            # 合并为 [B,C,H,W]
            try:
                stacked = torch.cat(adjusted, dim=0)  # [B,C,H,W]
            except Exception as e:
                bad_shapes = [tuple(a.shape) for a in adjusted]
                _log(f"[ERROR] 合并组/专家输出失败：{e}；形状列表：{bad_shapes}；跳过该条目")
                # 跳过这个 k_idx
                continue

            weighted = _apply_sample_weights(stacked, sample_wts)  # 逐样本广播乘权
            outputs.append(weighted)

        if not outputs:
            _log("[WARN] 没有有效的组/专家输出，返回零张量列表")
            return [torch.zeros(batch_size, C, *target_shape, device=device, dtype=x.dtype) for _ in range(top_k)]

        if len(shape_log) > 0:
            _log("[DEBUG] 专家/组内子模型输出形状样例（最多每类展示3条）：")
            shown = 0
            for key, shapes in shape_log.items():
                _log(f"  {key}: {shapes[:3]}{' ...' if len(shapes) > 3 else ''}")
                shown += 1
                if shown >= 20:
                    _log("  ... ")
                    break

        return outputs

    def reset_parameters_(self):
        neg_slope = 0.2

        def _init_linear(layer: nn.Linear):
            weight = getattr(layer, 'weight', None)
            if isinstance(weight, UninitializedParameter):
                return
            if weight is not None:
                nn.init.xavier_uniform_(weight, gain=1.0)
            bias = getattr(layer, 'bias', None)
            if bias is not None:
                nn.init.zeros_(bias)

        def _init_basic(module: nn.Module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=neg_slope, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None: nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None: nn.init.ones_(module.weight)
                if module.bias   is not None: nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                _init_linear(module)
            elif isinstance(module, SpectralConv):
                if hasattr(module, 'reset_parameters'): module.reset_parameters()

        def _apply_basic(module: Optional[nn.Module]):
            if module is not None: module.apply(_init_basic)

        def _init_composite(module: Optional[nn.Module]):
            if module is None: return
            if hasattr(module, 'reset_parameters_'):
                module.reset_parameters_()
            elif hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            else:
                module.apply(_init_basic)

        if getattr(self, 'CCrf', None) is not None:
            if hasattr(self.CCrf, 'apply'): self.CCrf.apply(_init_basic)
            if hasattr(self.CCrf, 'w'): nn.init.constant_(self.CCrf.w, 1.0)

        for expert in self.experts:
            if hasattr(expert, 'reset_parameters'): expert.reset_parameters()
            elif hasattr(expert, 'reset_parameters_'): expert.reset_parameters_()
            else: expert.apply(_init_basic)

        if getattr(self, 'router', None) is not None:
            self.router.apply(_init_basic)
            last_linear = None
            for module in self.router.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
            if last_linear is not None and not isinstance(getattr(last_linear, 'weight', None), UninitializedParameter):
                nn.init.zeros_(last_linear.weight)
                if last_linear.bias is not None: nn.init.zeros_(last_linear.bias)

        for name in ('s_processor', 'w_processor'):
            _init_composite(getattr(self, name, None))

        if self.fusion_type == 'linear':
            _init_composite(getattr(self, 'fusion', None))
        elif self.fusion_type == 'attention':
            if hasattr(self, 'fusion') and hasattr(self.fusion, 'reset_parameters'):
                self.fusion.reset_parameters()
        elif self.fusion_type == 'swa':
            if getattr(self, 's_act', None) is not None:
                _init_composite(getattr(self.s_act, 'processor', None))
            if getattr(self, 'w_act', None) is not None:
                _init_composite(getattr(self.w_act, 'processor', None))

        if getattr(self, 'config', {}).get('is_classifier', False) and hasattr(self, 'fc'):
            _init_linear(self.fc)

    def get_expert_distribution(self, x, task_features=None):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1, self.in_channels).mean(dim=1)
        if hasattr(self.router, 'get_expert_distribution'):
            return self.router.get_expert_distribution(x_flat, task_features)
        else:
            logits = self.router.router(x_flat)
            return F.softmax(logits, dim=-1)

    def save_experts(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, expert in enumerate(self.experts):
            expert_path = save_dir / f"expert_{i}.pt"
            torch.save(expert.state_dict(), expert_path)
        metadata = {
            'num_experts': self.num_experts,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'hidden_channels': self.hidden_channels,
            'router_type': self.router_type
        }
        torch.save(metadata, save_dir / "metadata.pt")

    def load_experts(self, load_dir):
        load_dir = Path(load_dir)
        if not load_dir.exists():
            raise ValueError(f"专家目录不存在: {load_dir}")
        metadata_path = load_dir / "metadata.pt"
        if not metadata_path.exists():
            raise ValueError(f"元数据文件不存在: {metadata_path}")
        metadata = torch.load(metadata_path)
        if metadata['num_experts'] != self.num_experts:
            raise ValueError(f"专家数量不匹配: 期望 {self.num_experts}，实际 {metadata['num_experts']}")
        for i, expert in enumerate(self.experts):
            expert_path = load_dir / f"expert_{i}.pt"
            if not expert_path.exists():
                raise ValueError(f"专家文件不存在: {expert_path}")
            expert.load_state_dict(torch.load(expert_path))
        print(f"成功加载 {self.num_experts} 个专家模型")
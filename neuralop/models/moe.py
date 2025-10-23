# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Union, Tuple, Callable, Optional
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

        # 专家列表（索引即身份）
        self.experts = nn.ModuleList(experts)

        # >>> 显存管理透明代理（开关可由 config 控制）
        use_proxy = self.config.get('use_expert_memory_proxy', True)
        self.expert_proxy: Optional[ExpertMemoryProxy] = None
        if use_proxy:
            self.expert_proxy = ExpertMemoryProxy(
                experts=list(self.experts),
                device=self.config.get('device', 'cuda'),
                cache_size=self.config.get('expert_cache_size', 2),
                amp_dtype=(torch.float16 if self.config.get('proxy_fp16', True) else None),
                convert_param_dtype_on_gpu=self.config.get('proxy_convert_param_dtype', True),
                safety_ratio=self.config.get('proxy_safety_ratio', 1.2),
                measure_on_first_use=self.config.get('proxy_measure_on_first_use', True),
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
                    module = LinearMix(k, self.in_channels)
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
                self.fusion = LinearMix(self.top_k, self.in_channels)
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

        self.proj = nn.Conv2d(in_channels, 1, 1)
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
        combined_output = self.proj(combined_output)
        return combined_output, aux_loss

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
        combined = self.proj(combined)
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
    @torch.no_grad()
    def _process_activation_group(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,   # [B, top_k]
        routing_weights: torch.Tensor,  # [B, top_k]
        k: int,
        batch_size: int,
        device,
        type_weights: Optional[torch.Tensor] = None,  # [B, T] 或 None
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        返回长度为 k 的列表，每个元素是 [B, C, H, W]（已经乘以该列的 routing 权重）。
        - 直接专家模式（无 type_weights）：每列 k_idx：
            1) 按 expert_idx 聚合 -> {eid: [n_e,C,H,W]}
            2) 一次 forward_many
            3) 回填到 [B,C,H,W] 并乘以 routing 权重
        - 分组模式（有 type_weights, 组内 T 专家 × 类型权重）：
            1) 将 (group_id, t) 展开为 eid=group_id*T+t，按 eid 聚合
            2) 一次 forward_many
            3) 对每个样本按 t 用 type_weights[b,t] 累加，再乘以该列 routing 权重
        """
        top_k = int(k)
        if top_k <= 0:
            return []

        if expert_indices is None:
            raise ValueError("expert_indices 为空，无法获取路由结果。")

        if routing_weights is None:
            routing_weights = torch.ones(batch_size, top_k, device=device, dtype=x.dtype)
        else:
            routing_weights = routing_weights.to(device=device, dtype=x.dtype)

        C = self.out_channels if self.out_channels > 0 else x.size(1)
        total_experts = len(self.experts)
        grouped_mode = (self.moe_mode == 'group') and (type_weights is not None)

        # 检查分组模式一致性
        if grouped_mode:
            if type_weights.dim() == 1:
                type_weights = type_weights.view(1, -1).expand(batch_size, -1)
            if type_weights.size(0) != batch_size:
                raise ValueError(f"type_weights batch 维度不一致: {type_weights.size(0)} vs {batch_size}")
            T = self.types_per_group
            if type_weights.size(1) != T:
                if self.is_logger:
                    print(f"[WARN] type_weights 列={type_weights.size(1)} 与组内专家数 T={T} 不一致，降级为直接模式。")
                grouped_mode = False
                type_weights = None
            else:
                expected_total = self.num_experts * T
                experts_per_group = total_experts // max(1, self.num_experts)
                if (expected_total == 0 or total_experts < expected_total
                    or total_experts % T != 0 or experts_per_group != T):
                    if self.is_logger:
                        print(f"[WARN] 专家数量 {total_experts} 与分组结构不匹配（期望 {expected_total}），降级为直接模式。")
                    grouped_mode = False
                    type_weights = None
                else:
                    type_weights = type_weights.to(device=device, dtype=x.dtype)

        outputs_per_k: List[torch.Tensor] = []

        for k_idx in range(top_k):
            indices_k: torch.Tensor = expert_indices[:, k_idx]   # [B]
            weights_k: torch.Tensor = routing_weights[:, k_idx]  # [B]
            weights_k = weights_k.to(device=device, dtype=x.dtype)

            if not grouped_mode:
                # -------- 直接专家模式：一次 per-expert mini-batch 前向 --------
                routed, meta = self._pack_by_expert(indices_k, x)

                if not routed:
                    outputs_per_k.append(torch.zeros(batch_size, C, *x.shape[2:], device=device, dtype=x.dtype))
                    continue

                # 一次前向（代理 or 直连）
                if getattr(self, "expert_proxy", None) is not None:
                    y_by_eid = self.expert_proxy.forward_many(routed)
                else:
                    y_by_eid = {eid: self.experts[eid](x_sub, **kwargs) for eid, x_sub in routed.items()}

                # 统一空间形状
                spatial_shapes = [tuple(y.shape[2:]) for y in y_by_eid.values() if y.dim() >= 4]
                target_shape = self._most_common_shape(spatial_shapes, default_shape=tuple(x.shape[2:]))
                for eid in list(y_by_eid.keys()):
                    y_by_eid[eid] = self._resize_to_shape(y_by_eid[eid], target_shape).contiguous()

                # 回填为 [B,C,H,W]
                y_full = self._scatter_back(y_by_eid, meta, batch_size, C, target_shape, device, x.dtype)
                # 逐样本乘以该列的 routing 权重
                y_full = y_full * weights_k.view(-1, 1, 1, 1)
                outputs_per_k.append(y_full)

            else:
                # -------- 分组模式：把 (group, t) 展开为具体专家，合一批处理 --------
                T = self.types_per_group
                routed: Dict[int, torch.Tensor] = {}
                meta: Dict[int, Tuple[List[int], List[int]]] = {}

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
                        if eid not in routed:
                            routed[eid] = [x_slice]
                            meta[eid] = ([b], [1])
                        else:
                            routed[eid].append(x_slice)
                            meta[eid][0].append(b)
                            meta[eid][1].append(1)

                if not routed:
                    outputs_per_k.append(torch.zeros(batch_size, C, *x.shape[2:], device=device, dtype=x.dtype))
                    continue

                # 拼接 mini-batch
                for eid in routed:
                    routed[eid] = torch.cat(routed[eid], dim=0).contiguous()

                # 一次前向（代理 or 直连）
                if getattr(self, "expert_proxy", None) is not None:
                    y_by_eid = self.expert_proxy.forward_many(routed)
                else:
                    y_by_eid = {eid: self.experts[eid](x_sub, **kwargs) for eid, x_sub in routed.items()}

                # 统一空间形状
                spatial_shapes = [tuple(y.shape[2:]) for y in y_by_eid.values() if y.dim() >= 4]
                target_shape = self._most_common_shape(spatial_shapes, default_shape=tuple(x.shape[2:]))
                for eid in list(y_by_eid.keys()):
                    y_by_eid[eid] = self._resize_to_shape(y_by_eid[eid], target_shape).contiguous()

                # 组内按 t 加权累加到样本位
                H, W = target_shape
                y_group = torch.zeros(batch_size, C, H, W, device=device, dtype=x.dtype)
                for eid, y_sub in y_by_eid.items():
                    positions, _ = meta[eid]
                    local_t = eid % T
                    for i, pos in enumerate(positions):
                        tw = type_weights[pos, local_t]   # 标量
                        y_group[pos:pos+1] += y_sub[i:i+1] * tw

                # 乘以该列的路由权重
                y_group = y_group * weights_k.view(-1, 1, 1, 1)
                outputs_per_k.append(y_group)

        return outputs_per_k

    # ---------------- 其它工具 & 初始化 ----------------
    @staticmethod
    def _most_common_shape(shapes: List[Tuple[int, ...]], default_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if not shapes:
            return default_shape
        ndims = [len(s) for s in shapes]
        target_ndim = Counter(ndims).most_common(1)[0][0]
        candidates = [s for s in shapes if len(s) == target_ndim]
        result = []
        for dim in range(target_ndim):
            dim_sizes = [s[dim] for s in candidates]
            result.append(Counter(dim_sizes).most_common(1)[0][0])
        return tuple(result)

    def _resize_to_shape(self, tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        if tensor.dim() < 4 or tensor.shape[2:] == target_shape:
            return tensor
        try:
            if len(target_shape) == 2:
                mode = 'bilinear'
            elif len(target_shape) == 3:
                mode = 'trilinear'
            else:
                mode = 'linear'
            return F.interpolate(tensor, size=target_shape, mode=mode, align_corners=False)
        except Exception as exc:
            if self.is_logger:
                print(f"[WARN] 插值到 {target_shape} 失败：{exc}，使用零张量替代。")
            return tensor.new_zeros(tensor.size(0), tensor.size(1), *target_shape)

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
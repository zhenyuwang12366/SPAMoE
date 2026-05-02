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
        # Optional transparent GPU memory proxy (toggle via config)
        use_proxy = self.config.get('use_expert_memory_proxy', False)
        self.expert_proxy: Optional[ExpertMemoryProxy] = None
        if use_proxy:
            self.expert_proxy = ExpertMemoryProxy(
                experts=list(self.experts),
                device=self.config.get('device', None),
                cache_size=self.config.get('expert_cache_size', 2),
                amp_dtype=torch.bfloat16,
            )
        
        # Derived attributes
        if self.moe_mode == 'velocity_type':
            self.num_experts = len(experts)
            self.types_per_group = 0
        elif self.moe_mode == 'group':
            self.types_per_group = int(self.v_type_num)
            if self.types_per_group <= 0:
                raise ValueError("group mode requires a positive integer v_type_num.")
            if len(experts) % self.types_per_group != 0 and is_logger:
                print(f"[WARN] Number of experts {len(experts)} is not divisible by types_per_group {self.types_per_group}; truncating.")
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
                raise ValueError("task_aware router does not support 'swa' fusion.")

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
                raise ValueError(f"Unsupported router type: {router_type}")

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

            # fusion modules
            if fusion_type == 'linear':
                self.fusion = LinearMix(self.top_k, self.out_channels)
            elif fusion_type == 'attention':
                self.fusion = AttentionMix(
                    input_resolution=256, patch_size=16, in_channels=self.in_channels, out_channels=self.out_channels,
                    width=512, layers=6, heads=8, num_experts=self.top_k,
                    use_cls_expert=False,
                )
            elif fusion_type == 'swa':
                self.s_act = GroupActMerge(processor=self.s_processor)
                self.w_act = GroupActMerge(processor=self.w_processor)
                self.sw_act = SWActMerge(beta=self.config.get('beta', 0.5))
            elif fusion_type == 'basic':
                self.fusion = SumMix(self.out_channels)
            else:
                raise ValueError(f"Unsupported fusion type: {fusion_type}")
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
                    print("[WARN] No type weights provided; group mode falls back to vanilla experts.")
                    self._type_weight_warned = True
            else:
                type_weights = class_weights
        else:
            type_weights = None
            if class_weights is not None and self.is_logger and not self._type_weight_warned:
                print("[WARN] moe_mode is not 'group'; ignoring encoder type weights.")
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
            # pass stacked Tensor (not list)
            s_combined = self.s_act(s_combined)
            w_combined = self.w_act(w_combined)
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
        Pick target_shape from collected spatial shapes.
        If all_shapes is empty, use default_shape when provided, else x.spatial shape.
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
            print(f"[ERROR] Interpolation failed: {e}; using zeros. in={tuple(out.shape)} target={target_shape}")
            return torch.zeros(out.shape[0], out.shape[1], *target_shape, device=out.device, dtype=out.dtype)
    
    def _forward_velocity_type(self, x: torch.Tensor, class_weights: Optional[torch.Tensor], **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = x.shape[0]
        device = x.device

        num_available = len(self.experts)
        if num_available == 0:
            raise RuntimeError("velocity_type mode needs at least one expert.")

        weights = class_weights
        if weights is None:
            if self.is_logger:
                print("[WARN] type_weights missing; using uniform weights.")
            num_types = min(self.v_type_num or num_available, num_available)
            weights = torch.ones(batch_size, num_types, device=device, dtype=x.dtype) / float(num_types)
        else:
            if weights.dim() == 1:
                weights = weights.view(1, -1).expand(batch_size, -1)
            elif weights.dim() == 2 and weights.size(0) == 1 and batch_size > 1:
                weights = weights.expand(batch_size, -1)
            elif weights.dim() == 2 and weights.size(0) != batch_size:
                raise ValueError(f"type_weights batch size {weights.size(0)} does not match input batch {batch_size}")
            elif weights.dim() != 2:
                raise ValueError(f"Unsupported type_weights shape: {tuple(weights.shape)}")
            weights = weights.to(device=device, dtype=x.dtype)

        expected_types = self.v_type_num or weights.size(1)
        total_experts = num_available
        num_types = min(expected_types, weights.size(1), total_experts)
        if num_types <= 0:
            raise ValueError("velocity_type mode needs at least one expert and matching type weights.")
        if num_types < weights.size(1) and self.is_logger:
            print(f"[WARN] type_weights has {weights.size(1)} columns but only {total_experts} experts; using first {num_types}.")

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
                    print(f"[WARN] velocity-type expert {idx} forward failed: {exc}")
                out = _zeros_like()

            if not torch.is_tensor(out):
                if self.is_logger:
                    print(f"[WARN] velocity-type expert {idx} returned non-tensor; using zeros.")
                out = _zeros_like()

            if out.dim() == 3: out = out.unsqueeze(1)
            elif out.dim() == 2: out = out.view(batch_size, -1, 1, 1)
            elif out.dim() < 2: out = out.view(batch_size, 1, 1, 1)
            elif out.dim() > 4: out = out.view(out.size(0), out.size(1), out.size(2), -1)
            if out.size(0) != batch_size:
                if self.is_logger:
                    print(f"[WARN] expert {idx} output batch size {out.size(0)} mismatch; replacing with zeros.")
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

    # ---------------- Pack by expert & scatter back ----------------
    def _pack_by_expert(self, indices_1d: torch.Tensor, x: torch.Tensor):
        """
        indices_1d: [B] per-sample expert_idx (indices into self.experts)
        x:          [B, C, H, W]
        Returns:
          routed: {expert_idx -> x_sub_cat [n_e, C, H, W]}
          meta  : {expert_idx -> (positions_list, sizes_list)}  # for scatter_back
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
        meta       : meta from _pack_by_expert
        Returns: y_full [B, C, H, W]
        """
        H, W = shape_hw
        y_full = torch.zeros(B, C, H, W, device=device, dtype=dtype)
        for eid, y_sub in y_by_expert.items():
            positions, _ = meta[eid]
            for i, pos in enumerate(positions):
                y_full[pos:pos+1] = y_sub[i:i+1]
        return y_full

    # ---------------- Batched activation-group processing ----------------
    def _process_activation_group(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,   # [B, top_k]
        routing_weights: torch.Tensor,  # [B, top_k] or None
        k: int,
        batch_size: int,
        device,
        type_weights: Optional[torch.Tensor] = None,  # [B, T] or None
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Summary:
        1) Batch by expert, one forward (forward_many or per-expert).
        2) Match legacy: defer shape alignment + per-sample routing weights to the end.
        3) Logging matches legacy with print(..., flush=True).
        Returns: list of length top_k, each [B,C,H,W].
        """
        # ------------------ logging helper (flush) ------------------
        def _log(msg: str):
            if getattr(self, "is_logger", False):
                print(msg, flush=True)

        # ------------------ validation & setup ------------------
        top_k = int(k)
        if top_k <= 0:
            return []

        if expert_indices is None:
            raise ValueError("expert_indices is empty; cannot route.")
        if expert_indices.size(1) < top_k:
            raise ValueError(f"expert_indices has fewer columns than top_k={top_k}")

        if routing_weights is None:
            routing_weights = torch.ones(batch_size, top_k, device=device, dtype=x.dtype)
        elif routing_weights.size(1) < top_k:
            raise ValueError(f"routing_weights has fewer columns than top_k={top_k}")
        else:
            routing_weights = routing_weights.to(device=device, dtype=x.dtype)

        C = self.out_channels if getattr(self, "out_channels", 0) > 0 else x.size(1)
        total_experts = len(self.experts)
        grouped_mode = (getattr(self, "moe_mode", None) == "group") and (type_weights is not None)

        # ---- Grouping checks & type weights ----
        if grouped_mode:
            if type_weights.dim() == 1:
                type_weights = type_weights.view(1, -1).expand(batch_size, -1)
            if type_weights.size(0) != batch_size:
                raise ValueError(f"type_weights batch dim mismatch: {type_weights.size(0)} vs {batch_size}")

            T = getattr(self, "types_per_group", None)
            if T is None:
                _log("[WARN] types_per_group unset; falling back to direct mode.")
                grouped_mode = False
                type_weights = None
            elif type_weights.size(1) != T:
                _log(f"[WARN] type_weights columns={type_weights.size(1)} != in-group experts T={T}; falling back to direct mode.")
                grouped_mode = False
                type_weights = None
            else:
                expected_total = self.num_experts * T
                experts_per_group = total_experts // max(1, self.num_experts)
                if (expected_total == 0 or total_experts < expected_total
                    or total_experts % T != 0 or experts_per_group != T):
                    _log(f"[WARN] expert count {total_experts} incompatible with group layout (expected {expected_total}); falling back to direct mode.")
                    grouped_mode = False
                    type_weights = None
                else:
                    type_weights = type_weights.to(device=device, dtype=x.dtype)

        # ------------------ inner helpers (legacy semantics) ------------------
        def _zeros_like_x(bsz: int, channels: int, like: torch.Tensor) -> torch.Tensor:
            return torch.zeros(bsz, channels, *like.shape[2:], device=like.device, dtype=like.dtype)

        def _most_common_target_shape(
            all_shapes: List[Tuple[int, ...]],
            default_shape: Optional[Tuple[int, ...]] = None
        ) -> Tuple[int, ...]:
            """
            Pick target_shape from collected spatial sizes.
            If all_shapes is empty, use default_shape if given else x.spatial shape.
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
                _log(f"[ERROR] Interpolation failed: {e}; using zeros. in={tuple(out.shape)} target={target_shape}")
                return torch.zeros(out.shape[0], out.shape[1], *target_shape, device=out.device, dtype=out.dtype)

        def _apply_sample_weights(batch_tensor: torch.Tensor, sample_wts: torch.Tensor) -> torch.Tensor:
            if sample_wts.dim() == 1:
                sample_wts = sample_wts.view(-1, 1, 1, 1)
            elif sample_wts.dim() == 2 and sample_wts.shape[1] == 1:
                sample_wts = sample_wts.view(-1, 1, 1, 1)
            return batch_tensor * sample_wts

        # Shape log: store shapes only (avoid holding tensors)
        shape_log: Dict[Any, List[Tuple[int, ...]]] = {}

        # ------------------ main loop: per k, outputs before align / route multiply ------------------
        # each k_idx -> (List of per-sample [1,C,h,w], sample_weights [B])
        collected: List[Tuple[List[torch.Tensor], torch.Tensor]] = []

        for k_idx in range(top_k):
            indices_k: torch.Tensor = expert_indices[:, k_idx]   # [B]
            weights_k: torch.Tensor = routing_weights[:, k_idx]  # [B]
            weights_k = weights_k.to(device=device, dtype=x.dtype)

            if not grouped_mode:
                # -------- Direct expert mode: pack by expert, forward, scatter; no route mult yet --------
                routed, meta = self._pack_by_expert(indices_k, x)

                if not routed:
                    # no route hits for this k column
                    batch_outputs = [_zeros_like_x(1, C, x[b:b+1]) for b in range(batch_size)]
                    collected.append((batch_outputs, weights_k))
                    continue

                # Forward (proxy or direct) with fallbacks
                try:
                    if getattr(self, "expert_proxy", None) is not None:
                        y_by_eid = self.expert_proxy.forward_many(routed)
                    else:
                        y_by_eid = {}
                        for eid, x_sub in routed.items():
                            try:
                                y_by_eid[eid] = self.experts[eid](x_sub, **kwargs)
                            except Exception as e:
                                _log(f"[ERROR] expert {eid} failed: {e}; using zeros")
                                y_by_eid[eid] = _zeros_like_x(x_sub.size(0), C, x_sub)
                                import traceback
                                traceback.print_exc()
                                exit(0)
                except Exception as e:
                    _log(f"[ERROR] forward_many failed: {e}; zero fallback for this column")
                    batch_outputs = [_zeros_like_x(1, C, x[b:b+1]) for b in range(batch_size)]
                    collected.append((batch_outputs, weights_k))
                    continue

                # Per-expert shape log
                for eid, y in y_by_eid.items():
                    if y is not None:
                        shape_log.setdefault(("expert", int(eid)), []).append(tuple(y.shape))

                # Local align to modal shape in this column before scatter
                spatial_shapes = [tuple(y.shape[2:]) for y in y_by_eid.values() if y is not None and y.dim() >= 4]
                target_shape_local = _most_common_target_shape(spatial_shapes, default_shape=tuple(x.shape[2:]))
                for eid in list(y_by_eid.keys()):
                    y_by_eid[eid] = _resize_to(y_by_eid[eid], target_shape_local).contiguous()

                y_full = self._scatter_back(y_by_eid, meta, batch_size, C, target_shape_local, device, x.dtype)

                # Split to per-sample [1,C,h,w]; global align later
                batch_outputs = [y_full[b:b+1] for b in range(batch_size)]
                collected.append((batch_outputs, weights_k))

            else:
                # -------- Group mode: expand (group,t) to experts; type_weights on samples; no route mult yet --------
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

                # Concat mini-batches per expert
                for eid in routed_dict:
                    routed_dict[eid] = torch.cat(routed_dict[eid], dim=0).contiguous()

                # Forward (proxy or direct) with fallbacks
                try:
                    if getattr(self, "expert_proxy", None) is not None:
                        y_by_eid = self.expert_proxy.forward_many(routed_dict)
                    else:
                        y_by_eid = {}
                        for eid, x_sub in routed_dict.items():
                            try:
                                y_by_eid[eid] = self.experts[eid](x_sub, **kwargs)
                            except Exception as e:
                                _log(f"[ERROR] expert {eid} failed: {e}; using zeros")
                                y_by_eid[eid] = _zeros_like_x(x_sub.size(0), C, x_sub)
                                import traceback
                                traceback.print_exc()
                                exit(0)
                except Exception as e:
                    _log(f"[ERROR] forward_many failed: {e}; zero fallback for this column")
                    batch_outputs = [_zeros_like_x(1, C, x[b:b+1]) for b in range(batch_size)]
                    collected.append((batch_outputs, weights_k))
                    continue

                # Shape log per (group, t)
                for eid, y in y_by_eid.items():
                    if y is not None:
                        g = int(eid // T); t = int(eid % T)
                        shape_log.setdefault(("group", g, t), []).append(tuple(y.shape))

                # Local unify for this column
                spatial_shapes = [tuple(y.shape[2:]) for y in y_by_eid.values() if y is not None and y.dim() >= 4]
                target_shape_local = _most_common_target_shape(spatial_shapes, default_shape=tuple(x.shape[2:]))

                for eid in list(y_by_eid.keys()):
                    y_by_eid[eid] = _resize_to(y_by_eid[eid], target_shape_local).contiguous()

                H, W = target_shape_local
                y_group = torch.zeros(batch_size, C, H, W, device=device, dtype=x.dtype)

                # Vectorized fill: index_add_ for all samples of same eid
                for eid, y_sub in y_by_eid.items():
                    positions, _ = meta_dict[eid]         # List[int], length N
                    if len(positions) == 0:
                        continue
                    idx = torch.tensor(positions, device=device, dtype=torch.long)  # [N]
                    local_t = eid % T
                    tw = type_weights[idx, local_t].view(-1, 1, 1, 1)               # [N,1,1,1]
                    # y_sub: [N,C,H,W] (locally aligned)
                    y_group.index_add_(0, idx, y_sub * tw)

                # Per-sample [1,C,h,w]; then global align + route weights
                batch_outputs = [y_group[b:b+1] for b in range(batch_size)]
                collected.append((batch_outputs, weights_k))

        # ------------------ global align + per-sample routing weights (legacy) ------------------
        # Collect spatial shapes from all outputs
        all_spatial_shapes: List[Tuple[int, ...]] = []
        for batch_outputs, _ in collected:
            for out in batch_outputs:
                if out is not None and out.dim() >= 4:
                    all_spatial_shapes.append(tuple(out.shape[2:]))

        if not all_spatial_shapes:
            _log("[WARN] no valid outputs; returning zero tensors")
            return [torch.zeros(batch_size, C, *x.shape[2:], device=device, dtype=x.dtype) for _ in range(top_k)]

        target_shape = _most_common_target_shape(all_spatial_shapes)
        _log(f"[INFO] target shape for alignment: {target_shape}")

        outputs: List[torch.Tensor] = []
        for batch_outputs, sample_wts in collected:
            adjusted = []
            for out in batch_outputs:
                out_adj = _resize_to(out, target_shape)
                adjusted.append(out_adj)

            # stack -> [B,C,H,W]
            try:
                stacked = torch.cat(adjusted, dim=0)  # [B,C,H,W]
            except Exception as e:
                bad_shapes = [tuple(a.shape) for a in adjusted]
                _log(f"[ERROR] merge expert outputs failed: {e}; shapes: {bad_shapes}; skipping")
                # skip this k_idx
                continue

            weighted = _apply_sample_weights(stacked, sample_wts)  # broadcast per-sample weights
            outputs.append(weighted)

        if not outputs:
            _log("[WARN] no valid grouped/expert outputs; returning zero tensors")
            return [torch.zeros(batch_size, C, *target_shape, device=device, dtype=x.dtype) for _ in range(top_k)]

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
            raise ValueError(f"Expert directory does not exist: {load_dir}")
        metadata_path = load_dir / "metadata.pt"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file missing: {metadata_path}")
        metadata = torch.load(metadata_path)
        if metadata['num_experts'] != self.num_experts:
            raise ValueError(f"Expert count mismatch: expected {self.num_experts}, got {metadata['num_experts']}")
        for i, expert in enumerate(self.experts):
            expert_path = load_dir / f"expert_{i}.pt"
            if not expert_path.exists():
                raise ValueError(f"Expert checkpoint missing: {expert_path}")
            expert.load_state_dict(torch.load(expert_path))
        print(f"Loaded {self.num_experts} expert checkpoints")
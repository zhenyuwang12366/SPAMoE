# neuralop/models/EMO.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Dict


class EMO(nn.Module):
    """
    Encoder–Mixture-of-Operators (EMO)

    将 encoder 与 moe 打包成一个统一模块，保持你现有训练/评估流程不变：
      - 兼容当前调用：preds, aux = model(encoded, weights)
      - 也支持端到端：preds, aux = model.forward_raw(inputs, use_amp=..., dtype=...)

    Checkpoint 兼容性：
      - state_dict() / load_state_dict() 仅作用于 MoE（与原来保存/恢复一致）
      - encoder_state_dict() / load_encoder_state_dict() 专管编码器权重
    """

    def __init__(
        self,
        encoder: Optional[nn.Module],
        moe: nn.Module,
        *,
        pass_encoder_logits_as_weights: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.moe = moe
        self._pass_encoder_logits_as_weights = bool(pass_encoder_logits_as_weights)

    # ---------------------------
    # 读写代理（保持原训练流程可见性）
    # ---------------------------
    @property
    def router(self) -> nn.Module:
        return self.moe.router

    @property
    def router_type(self) -> str:
        return self.moe.router_type

    @property
    def experts(self):
        return self.moe.experts

    @property
    def in_channels(self) -> Optional[int]:
        return getattr(self.moe, "in_channels", None)

    @property
    def out_channels(self) -> Optional[int]:
        return getattr(self.moe, "out_channels", None)

    @property
    def hidden_channels(self) -> Optional[int]:
        return getattr(self.moe, "hidden_channels", None)

    # ---------------------------
    # 便捷：端到端(raw inputs)推理/训练
    # ---------------------------
    @torch.no_grad()
    def encode_only(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        只做编码，返回 (encoded, weights/logits)。
        """
        if self.encoder is None:
            return inputs, None
        encoded, weights, _ = self.encoder(inputs)
        return encoded, weights

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        force_dtype: Optional[torch.dtype] = torch.float32,
        **moe_kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        直接给原始输入：自动走 encoder -> moe
        - use_amp: 是否在 encoder+moe 中使用 autocast
        - amp_dtype: autocast dtype（默认 bfloat16）
        - force_dtype: 在送入 moe 前强制转换的 dtype（默认 float32；None 表示不转换）
        """
        if self.encoder is None:
            encoded, weights = inputs, None
        else:
            if use_amp:
                with torch.amp.autocast(device_type=inputs.device.type, enabled=True, dtype=amp_dtype):
                    encoded, enc_weights, _ = self.encoder(inputs)
            else:
                encoded, enc_weights, _ = self.encoder(inputs)
            weights = enc_weights if self._pass_encoder_logits_as_weights else None

        if use_amp:
            with torch.amp.autocast(device_type=encoded.device.type, enabled=True, dtype=amp_dtype):
                preds, aux = self.moe(encoded, weights, **moe_kwargs)
        else:
            preds, aux = self.moe(encoded, weights, **moe_kwargs)
        return preds, aux, enc_weights

    # ---------------------------
    # Checkpoint 兼容性：与原流程一致
    # ---------------------------
    def state_dict(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        仅导出 MoE 的状态字典，兼容原有：
            checkpoint['model_state_dict'] <- moe.state_dict()
        """
        return self.moe.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """
        仅加载 MoE 的权重。
        """
        missing, unexpected = self.moe.load_state_dict(state_dict, strict=strict)
        return missing, unexpected

    # 编码器权重单独管理
    def encoder_state_dict(self, *args, **kwargs) -> Optional[Dict[str, torch.Tensor]]:
        if self.encoder is None:
            return None
        return self.encoder.state_dict(*args, **kwargs)

    def load_encoder_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        if self.encoder is None:
            raise RuntimeError("No encoder inside EMO to load state_dict into.")
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=strict)
        return missing, unexpected

    # ---------------------------
    # 冻结/解冻便捷函数（可选）
    # ---------------------------
    def freeze_encoder(self) -> None:
        if self.encoder is not None:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

    def unfreeze_encoder(self) -> None:
        if self.encoder is not None:
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            self.encoder.train()
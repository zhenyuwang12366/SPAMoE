# neuralop/models/EMO.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Dict


class EMO(nn.Module):
    """
    Encoder–Mixture-of-Operators (EMO)

    Wraps encoder and MoE in one module while keeping your existing train/eval flow:
      - Current-style call: preds, aux = model(encoded, weights)
      - End-to-end: preds, aux = model.forward_raw(inputs, use_amp=..., dtype=...)

    Checkpoint compatibility:
      - state_dict() / load_state_dict() only touch the MoE (same as before)
      - encoder_state_dict() / load_encoder_state_dict() handle encoder weights only
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
        if encoder is None:
            self.moe_amp = True
        else:
            self.moe_amp = False if encoder.cast_to_fp32_for_moe == True else True

    # ---------------------------
    # Read/write proxies (same visibility as original training flow)
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
    # Convenience: end-to-end (raw inputs) train/infer
    # ---------------------------
    @torch.no_grad()
    def encode_only(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode only; returns (encoded, weights/logits).
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
        **moe_kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Raw inputs: runs encoder -> MoE automatically.
        - use_amp: whether to use autocast in encoder+MoE
        - amp_dtype: autocast dtype (default bfloat16)
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
            with torch.amp.autocast(device_type=encoded.device.type, enabled=self.moe_amp, dtype=amp_dtype):
                preds, aux = self.moe(encoded, weights, **moe_kwargs)
        else:
            preds, aux = self.moe(encoded, weights, **moe_kwargs)
        return preds, aux, weights

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Callable
import math
from collections import OrderedDict

class LinearMix(nn.Module):
    """
    Standard linear map over expert logits.

    Args:
        input (torch.Tensor): shape [B, k, 1, h, w]
    """
    def __init__(
        self, 
        num_experts: int, 
        output_channels,
        ):
        super().__init__()
        self.output_channels = output_channels
        self.num_experts = num_experts
        self.linear = nn.Linear(self.num_experts, self.output_channels, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bsz, k, _, h, w = input.shape
        input = input.view(bsz, k, -1).permute(0, 2, 1).reshape(-1, k)  # (B*H*W, k); merge batch and space for shared ops
        output = self.linear(input) # b*h*w, k -> b*h*w, 1
        output = output.view(bsz, h, w, self.output_channels).permute(0, 3, 1, 2)
        return output

class MeanMix(nn.Module):
    """Mean aggregation over experts.

    Input shape: [b, k, 1, h, w]
    """
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bsz, k, _, h, w = input.shape
        output = input.mean(dim=1) # b, 1, h, w
        output = output.expand((bsz, self.output_channels, h, w))
        return output

class SumMix(nn.Module):
    """Sum aggregation over experts.

    Input shape: [b, k, 1, h, w]
    """
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bsz, k, _, h, w = input.shape
        output = input.sum(dim=1)
        output = output.expand((bsz, self.output_channels, h, w))
        return output

# class AttentionMix(nn.Module):
#     """Attention-based aggregation.

#     Input: [b, k, 1, h, w]
#     """
#     def __init__(self, output_channels):
#         super().__init__()
#         self.output_channels = output_channels
#         self.attn = None
#         self.linear = nn.LazyLinear(self.output_channels, bias=False)
        
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         bsz, k, _, h, w = input.shape
#         if self.attn is None:
#             self.attn = nn.MultiheadAttention(k, num_heads = 4, batch_first=True)
#         input = input.view(bsz, k, -1).permute(0, 2, 1).reshape(-1, k) # b*h*w, k
#         output, _ = self.attn(
#             input, input, input
#         )
#         output = self.linear(output) # b*h*w, c
#         output = output.view(bsz, h, w, self.output_channels).permute(0, 3, 1, 2)
        
#         return output

class LayerNorm(nn.LayerNorm):
    """
    LayerNorm subclass that runs the normalization in float32 for fp16 inputs.
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """
    QuickGELU: a simpler, faster variant of GELU (x * sigmoid(1.702 * x)).
    """

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Residual block with multi-head self-attention and a feed-forward MLP.
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        """
        Args:
            d_model: Hidden feature dimension.
            n_head: Number of attention heads.
            attn_mask: Optional attention mask.
        """
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),  # expand 4x
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))  # project back
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    Transformer stack built from ResidualAttentionBlock layers.
    """

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        """
        Args:
            width: Hidden dimension.
            layers: Number of blocks.
            heads: Attention heads per block.
            attn_mask: Optional mask shared by all blocks.
        """
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class AttentionMix(nn.Module):
    """
    1) Patch-embed each expert's feature map to `width`.
    2) Add patch positions (ViT-style).
    3) Reshape [B, P, K, width] -> [B*P, K, width] and run Transformer along expert dim K.
    4) Pool over K -> [B*P, width], restore grid [B, width, H', W'], upsample to input resolution.
    """

    def __init__(
        self,
        input_resolution: int,   # Square H=W
        patch_size: int,         # patch size (= stride)
        in_channels: int,        # channels per expert output (e.g. 1 or 64)
        width: int,              # Transformer width (ViT-style)
        layers: int,
        heads: int,
        num_experts: int,        # number of experts K
        out_channels: int = 1,
        use_cls_expert: bool = False,  # optional learnable cls token on the expert axis
    ):
        super().__init__()
        assert input_resolution % patch_size == 0, "input_resolution must be divisible by patch_size"
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.num_experts = num_experts
        self.width = width
        self.use_cls_expert = use_cls_expert

        # Shared patch embedding: [B, C, H, W] -> [B, width, H', W']
        self.patch_embed = nn.Conv2d(in_channels, width, kernel_size=patch_size, stride=patch_size, bias=False)

        grid = input_resolution // patch_size
        num_patches = grid * grid
        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches, width))

        if use_cls_expert:
            self.cls_expert = nn.Parameter(scale * torch.randn(width))

        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(
            width=width, layers=layers, heads=heads
        )

        self.ln_post = nn.LayerNorm(width)

        self.reconstruct = nn.ConvTranspose2d(width, out_channels,
                                              kernel_size=patch_size, stride=patch_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        if self.use_cls_expert:
            nn.init.trunc_normal_(self.cls_expert, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, experts_out):
        """
        experts_out: Tensor [B, K, 1, H, W] with H=W=input_resolution per expert slice [B, C_in, H, W].
        Returns fused [B, out_channels, H, W].
        """
        K = self.num_experts
        assert isinstance(experts_out, torch.Tensor) and experts_out.shape[1] == K
        experts_out = torch.unbind(experts_out, dim=1)
        
        # Patch embedding per expert -> [B, D, H', W'] -> [B, P, D]
        embeds = []
        for x in experts_out:
            e = self.patch_embed(x)               # [B, D, H', W']
            e = e.flatten(2).transpose(1, 2)      # [B, P, D]
            embeds.append(e)
        # [B, P, K, D]
        X = torch.stack(embeds, dim=2)

        B, P, K_, D = X.shape
        assert K_ == K
        H_, W_ = self.input_resolution // self.patch_size, self.input_resolution // self.patch_size

        # Add patch positions (ViT-style), broadcast over experts
        pos = self.positional_embedding.to(X.dtype)              # [P, D]
        X = X + pos[None, :, None, :]

        # Pre-norm, reshape to [B*P, K(+cls), D] for expert-axis attention
        X = self.ln_pre(X)
        X = X.reshape(B * P, K, D)                               # [B*P, K, D]

        if self.use_cls_expert:
            # Prepend cls token on expert axis: [B*P, 1, D]
            cls = self.cls_expert.to(X.dtype)[None, None, :].expand(B * P, 1, D)
            X = torch.cat([cls, X], dim=1)                       # [B*P, K+1, D]

        X = self.transformer(X)                                   # [B*P, K(±1), D]

        if self.use_cls_expert:
            fused = X[:, 0, :]                                   # cls token
        else:
            fused = X.sum(dim=1)                                 # sum over experts
        fused = self.ln_post(fused)                              # [B*P, D]

        # Restore patch grid and upsample to full resolution
        fused = fused.view(B, P, D).transpose(1, 2).contiguous() # [B, D, P]
        fused = fused.view(B, D, H_, W_)                         # [B, D, H', W']
        out   = self.reconstruct(fused)                          # [B, out_channels, H, W]
        return out
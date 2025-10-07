import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Callable
import math
from collections import OrderedDict

class LinearMix(nn.Module):
    """
    普通线性映射

    Args:
        input (torch.Tensor): 形状为B,k,1,h,w
    """
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
        self.linear = nn.LazyLinear(self.output_channels, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bsz, k, _, h, w = input.shape
        input = input.view(bsz, k, -1).permute(0, 2, 1).reshape(-1, k) # 把batch和空间合并的常见写法，方便其他统一操作
        output = self.linear(input) # b*h*w, 1
        output = output.view(bsz, h, w, self.output_channels).permute(0, 3, 1, 2)
        return output

class MeanMix(nn.Module):
    """默认均值聚合

    输入：b,k,1,h,w
    
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
    """求和聚合

    输入：b,k,1,h,w
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
#     """注意力聚合

#     输入: b, k, 1, h, w
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
    LayerNorm 的子类，用于处理 fp16（半精度浮点数）。
    """

    def forward(self, x: torch.Tensor):
        # 保存原始数据类型
        orig_type = x.dtype
        # 将输入转换为 float32 类型，执行 LayerNorm 操作
        ret = super().forward(x.type(torch.float32))
        # 恢复原始数据类型
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """
    实现 QuickGELU 激活函数，比标准 GELU 更简单且计算效率更高。
    """

    def forward(self, x: torch.Tensor):
        # QuickGELU 激活函数公式：x * sigmoid(1.702 * x)
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    实现一个残差注意力模块，包括多头注意力机制和前馈网络（MLP）。
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        """
        初始化函数。
        参数：
        - d_model: 模型的隐藏层维度（即特征向量的维度）。
        - n_head: 多头注意力的头数。
        - attn_mask: 注意力掩码，可选。
        """
        super().__init__()

        # 多头注意力模块
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # 第一个 LayerNorm
        self.ln_1 = LayerNorm(d_model)
        # 前馈网络（MLP），包括线性层、QuickGELU 激活和另一层线性变换
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),  # 扩展特征维度 4 倍
            ("gelu", QuickGELU()),  # QuickGELU 激活
            ("c_proj", nn.Linear(d_model * 4, d_model))  # 恢复到原始特征维度
        ]))
        # 第二![](https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2024LLM/training/01.png)个 LayerNorm
        self.ln_2 = LayerNorm(d_model)
        # 注意力掩码
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        """
        计算多头注意力。
        """
        # 如果存在注意力掩码，将其转换为与输入相同的数据类型和设备
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 调用多头注意力，返回注意力结果（不需要权重）
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """
        前向传播。
        """
        # 1. 多头注意力：先进行 LayerNorm，再计算注意力，并与输入 x 相加（残差连接）
        x = x + self.attention(self.ln_1(x))
        # 2. 前馈网络：先进行 LayerNorm，再通过 MLP，并与输入 x 相加（残差连接）
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    基于 ResidualAttentionBlock 的 Transformer 模块。
    """

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        """
        初始化函数。
        参数：
        - width: 模型的隐藏层维度。
        - layers: Transformer 层数。
        - heads: 多头注意力的头数。
        - attn_mask: 注意力掩码。
        """
        super().__init__()
        self.width = width
        self.layers = layers
        # 堆叠多个 ResidualAttentionBlock，构成完整的 Transformer
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        """
        前向传播。
        """
        return self.resblocks(x)

class AttentionMix(nn.Module):
    """
    1) 对每个专家的特征做 patch embedding -> width
    2) 对 patch 维加位置嵌入（与 ViT 一致）
    3) 将 [B, P, K, width] -> [B*P, K, width]，在专家维 K 上做 Transformer
    4) 聚合专家 K -> [B*P, width]，再还原成 [B, width, H', W'] 并上采回原分辨率
    """

    def __init__(
        self,
        input_resolution: int,   # 输入图像的 H=W 分辨率（方形）
        patch_size: int,         # patch 尺寸（= stride）
        in_channels: int,        # 每个专家输出的通道数（例如 1 或 64）
        width: int,              # Transformer 的隐藏维（ViT 的 width）
        layers: int,             # Transformer 层数
        heads: int,              # 多头数
        num_experts: int,        # 专家数量 K
        out_channels: int = 1,   # 最终输出通道（例如 1）
        use_cls_expert: bool = False,  # 是否引入“cls expert”作为可学习的专家 token
    ):
        super().__init__()
        assert input_resolution % patch_size == 0, "input_resolution 必须能被 patch_size 整除"
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.num_experts = num_experts
        self.width = width
        self.use_cls_expert = use_cls_expert

        # 1) 对每个专家共享的 patch embedding：Conv2d(kernel=stride=patch)
        #    把 [B, C, H, W] -> [B, D=width, H', W']
        self.patch_embed = nn.Conv2d(in_channels, width, kernel_size=patch_size, stride=patch_size, bias=False)

        # 2) 位置嵌入（对 patch 维，与 ViT 一致）
        grid = input_resolution // patch_size
        num_patches = grid * grid
        # ViT 的初始化风格
        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches, width))

        # （可选）cls expert：在专家维前面加一个“可学习专家 token”，专门用于聚合
        if use_cls_expert:
            self.cls_expert = nn.Parameter(scale * torch.randn(width))

        # 输入归一化（与 ViT 类似）
        self.ln_pre = nn.LayerNorm(width)

        # 3) Transformer 编码器（在专家维 K 上运行；序列长度=K 或 K+1）
        self.transformer = Transformer(
            width=width, layers=layers, heads=heads
        )

        # 输出归一化
        self.ln_post = nn.LayerNorm(width)

        # 4) 上采样把 patch 网格还原为输入分辨率
        #    [B, D, H', W'] -> [B, out_channels, H, W]
        self.reconstruct = nn.ConvTranspose2d(width, out_channels,
                                              kernel_size=patch_size, stride=patch_size)

        # 参数初始化（ViT/Transformer 常见做法）
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        if self.use_cls_expert:
            nn.init.trunc_normal_(self.cls_expert, std=0.02)
        # 更细的 init（如 Linear/Conv 权重）按需添加
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
        experts_out: Tensor，b,k,1,h,w
            每个张量形状 [B, C_in, H, W]，H=W=input_resolution
        返回：
            重建后的融合结果 [B, out_channels, H, W]
        """
        K = self.num_experts
        assert isinstance(experts_out, torch.Tensor) and experts_out.shape[1] == K
        experts_out = torch.unbind(experts_out, dim=1)
        
        # 1) 每个专家的特征做 patch embedding -> [B, D, H', W'] -> [B, P, D]
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

        # 2) 给 patch 维加位置嵌入（与 ViT 一致），广播到每个专家
        pos = self.positional_embedding.to(X.dtype)              # [P, D]
        X = X + pos[None, :, None, :]

        # 3) 前归一化（与 ViT 一致），再重排为 [B*P, K(±1), D] 做“专家维”注意力
        X = self.ln_pre(X)
        X = X.reshape(B * P, K, D)                               # [B*P, K, D]

        if self.use_cls_expert:
            # 在专家维前拼一个 cls expert： [B*P, 1, D]
            cls = self.cls_expert.to(X.dtype)[None, None, :].expand(B * P, 1, D)
            X = torch.cat([cls, X], dim=1)                       # [B*P, K+1, D]

        # 4) Transformer（L=K 或 K+1）
        X = self.transformer(X)                                   # [B*P, K(±1), D]

        # 5) 聚合专家维
        if self.use_cls_expert:
            fused = X[:, 0, :]                                   # 取 cls expert
        else:
            fused = X.sum(dim=1)                                # 求和聚合 K 专家
        fused = self.ln_post(fused)                              # [B*P, D]

        # 6) 复原成 patch 网格，再上采回原分辨率
        fused = fused.view(B, P, D).transpose(1, 2).contiguous() # [B, D, P]
        fused = fused.view(B, D, H_, W_)                         # [B, D, H', W']
        out   = self.reconstruct(fused)                          # [B, out_channels, H, W]
        return out
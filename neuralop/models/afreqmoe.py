import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ======================
#   频率网格 & 软频带
# ======================

def make_radius_grid(H: int, W: int) -> torch.Tensor:
    """
    构造以频谱中心为圆心的半径网格 r \in [0, 1]，形状 [H, W]
    """
    ys = torch.linspace(-1, 1, steps=H)
    xs = torch.linspace(-1, 1, steps=W)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)    # [H, W], 半径 0 ~ sqrt(2)
    r = r / r.max().clamp_min(1e-6)  # 归一化到 [0, 1]
    return r


def make_soft_bands(
    r: torch.Tensor,
    num_bands: int = 4,
    sharpness: float = 20.0,
) -> List[torch.Tensor]:
    """
    根据半径 r \in [0,1] 生成 num_bands 个 soft mask：
    - 每个频带中心在 [0,1] 内均匀分布（避开两端）
    - 形式接近 Gaussian(r - center)

    返回：
        bands: list of [1, 1, H, W]，从低频 -> 高频
    """
    H, W = r.shape
    centers = torch.linspace(0.0, 1.0, steps=num_bands + 2)[1:-1]  # 避开 0 和 1
    bands: List[torch.Tensor] = []
    for c in centers:
        band = torch.exp(-sharpness * (r - c) ** 2)  # Gaussian-like
        bands.append(band.view(1, 1, H, W))
    return bands


# ============================
#   频谱注意力路由器（Router）
# ============================

class SpectralAttentionRouter(nn.Module):
    """
    输入空间域特征 x [B, C, H, W]：
      1) 先做 FFT 得到频谱 Ftt；
      2) 在频域幅度上做简单空间注意力，得到 F_refined；
      3) 用 F_refined 的全局信息生成 num_heads 个“专家 gate”；
      4) 同时根据半径 r 把频谱切成 num_heads 个 soft band，并 IFFT 回时空域。

    输出：
      - routed:        list of [B, C, H, W]，每个元素对应一个频带特征（低 -> 高频）
      - top_k_weights: [B, top_k]，对专家的权重（归一化）
      - top_k_indices: [B, top_k]，被选中的专家索引（0 ~ num_heads-1）
      - aux_loss:      标量或 None，用于 MoE 负载均衡
    """
    def __init__(
        self, 
        C: int, 
        num_heads: int = 4,   # 这里其实是专家数 num_experts
        top_k: int = 2,
        alpha: float = 0.0,
        band_sharpness: float = 20.0,
    ):
        super().__init__()
        assert 1 <= top_k <= num_heads
        self.top_k = top_k
        self.alpha = alpha
        self.num_heads = num_heads
        self.band_sharpness = band_sharpness

        # 频域 attention：qkv + proj
        self.qkv = nn.Conv2d(C, C * 3, 1)
        self.proj = nn.Conv2d(C, C, 1)

        # 频域特征 -> 专家权重
        self.freq_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # [B,C,H,W] -> [B,C,1,1]
            nn.Flatten(1),             # [B,C]
            nn.Linear(C, num_heads)    # [B,num_heads]
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # ===== 1. 频域变换 =====
        Ftt = torch.fft.fft2(x, dim=(-2, -1))                     # [B,C,H,W] 复数
        F_amp = (Ftt.real ** 2 + Ftt.imag ** 2).sqrt()            # [B, C, H, W]

        # ===== 2. 频域注意力（简单空间注意力）=====
        q, k, v = self.qkv(F_amp).chunk(3, dim=1)                 # 各 [B,C,H,W]

        energy = (q * k).sum(1, keepdim=True) / (C ** 0.5)        # [B,1,H,W]
        attn = torch.softmax(energy.view(B, 1, -1), dim=-1)       # 在 HW 上做 softmax
        attn = attn.view(B, 1, H, W)                              # [B,1,H,W]

        attn_v = attn * v                                         # [B,C,H,W]
        F_refined = self.proj(attn_v)                             # [B,C,H,W]

        # ===== 3. 专家门控（基于频域编码）=====
        gates = torch.softmax(self.freq_proj(F_refined), dim=-1)  # [B,num_heads]
        top_k_weights, top_k_indices = torch.topk(
            gates, k=self.top_k, dim=-1
        )  # [B,top_k], [B,top_k]

        # 归一化 top-k 权重
        top_k_weights = top_k_weights / top_k_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        # ===== 4. 负载均衡 aux_loss =====
        if self.alpha > 0.0:
            if top_k_indices.numel() == 0 or gates.numel() == 0:
                aux_loss = gates.sum() * 0.0   # 保证梯度图不被打断
            else:
                # freq: 每个专家在 top-k 中被选中的频率分布
                idx = top_k_indices.reshape(-1).to(torch.long)
                freq = F.one_hot(idx, num_classes=self.num_heads).float().mean(dim=0)  # [num_heads]
                # prob: gate 的平均分布
                prob = gates.mean(dim=0)                                              # [num_heads]

                # 目标可以是 freq 或者均匀分布
                target = freq
                # target = torch.full_like(freq, 1.0 / self.num_heads)

                aux_loss = F.mse_loss(prob, target) * float(self.alpha)
        else:
            aux_loss = None
        
        # ===== 5. 输出每个频段的 ifft 特征 =====
        # 注意这里让频带数量 = num_heads，以便用专家索引直接索引频带
        r = make_radius_grid(H, W).to(x.device)
        bands = make_soft_bands(r, num_bands=self.num_heads, sharpness=self.band_sharpness)
        routed = [
            torch.fft.ifft2(Ftt * band, dim=(-2, -1)).real
            for band in bands
        ]  # list of [B,C,H,W]，索引 0..num_heads-1 对应低→高频

        return routed, top_k_weights, top_k_indices, aux_loss


# ============================
#   自适应频段 MoE 组合模块
# ============================

class AdaptiveFreqMoE(nn.Module):
    """
    自适应频段 MoE：
    - experts: 按 [低频专家, 中低频专家, 中高频专家, 高频专家] 排列，例如 [FNO, MNO, LNO, WNO]
    - router: 负责在频谱上做注意力 & 产生专家 gate & 频段特征
    - forward:
        对每个样本 b：
          - router 给出 top_k_indices[b] & top_k_weights[b]
          - 激活对应专家 e_idx，并将该专家负责的频带 routed_bands[e_idx][b] 作为输入
          - 按 top_k_weights 做加权求和
    """
    def __init__(
        self, 
        experts: List[nn.Module],
        in_channels: int,
        topk: int,
        alpha: float = 0.0,
        band_sharpness: float = 20.0,
    ):
        super().__init__()
        # 注册为 ModuleList，确保参数被 optimizer 管理
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(self.experts)
        self.top_k = topk

        self.router = SpectralAttentionRouter(
            C=in_channels,
            num_heads=self.num_experts,
            top_k=topk,
            alpha=alpha,
            band_sharpness=band_sharpness,
        )
    
    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        返回:
            y:        [B, Cout, H', W']  MoE 组合输出
            aux_loss: router 的正则（可直接加到总 loss 里）
        """
        routed_bands, top_k_weights, top_k_indices, aux_loss = self.router(x)
        B = x.size(0)
        outputs = []

        for b in range(B):
            y_b = 0
            for j in range(self.top_k):
                e_idx = top_k_indices[b, j].item()          # 哪个专家 (0..num_experts-1)
                w = top_k_weights[b, j].view(1, 1, 1, 1)    # 权重 broadcast

                # 选对应频带特征作为该专家输入
                expert_input = routed_bands[e_idx][b:b+1]   # [1,C,H,W]
                expert_out = self.experts[e_idx](expert_input)  # [1,Cout,H',W']

                y_b = y_b + w * expert_out

            outputs.append(y_b)

        y = torch.cat(outputs, dim=0)   # [B,Cout,H',W']
        return y, aux_loss
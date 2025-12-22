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
    这里默认配合 fftshift 后的频谱使用：
        - (H//2, W//2) 附近是 DC 位置
        - 半径越大，频率越高
    """
    ys = torch.linspace(-1.0, 1.0, steps=H)
    xs = torch.linspace(-1.0, 1.0, steps=W)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")      # [H,W]
    r = torch.sqrt(xx**2 + yy**2)                       # [0, sqrt(2)]
    r = r / r.max().clamp_min(1e-6)                     # 归一化到 [0,1]
    return r


def make_soft_bands(
    r: torch.Tensor,
    num_bands: int = 3,
    sharpness: float = 20.0,
) -> List[torch.Tensor]:
    """
    根据半径 r \in [0,1] 生成 num_bands 个 soft mask：
    - 每个频带中心在 [0,1] 内均匀分布（避开两端）
    - mask 形状近似 Gaussian(r - center)

    返回：
        bands: list of [1,1,H,W]，从低频 -> 高频
    """
    H, W = r.shape
    centers = torch.linspace(0.0, 1.0, steps=num_bands)
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
      2) 对频谱做 fftshift，使 DC 移到中心，方便用半径划分频带；
      3) 在频域幅度上做简单空间注意力，得到 F_refined；
      4) 用 F_refined 的全局信息生成 num_heads 个“专家 gate”；
      5) 根据半径 r 把频谱切成 num_heads 个 soft band（同心圆）；
      6) 为每个专家引入一个可学习的“频率偏好” expert_freq[i]∈[0,1]，
         按 expert_freq vs band_centers 的相似度对各个 band 特征加权，
         得到每个专家自己的频带混合输入。

    输出：
      - routed:        list of [B, C, H, W]，每个元素对应一个“专家输入特征”（已经按频率偏好混合好）
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
        freq_affinity_sharpness: float = 10.0,  # 专家频率偏好的“锐度”
        use_soft_bands: bool = True,            # 消融：False -> 硬划分频带
        enable_freq_attn: bool = True,          # 消融：False -> 不做频率自注意力
        enable_band_mixing: bool = True,        # 消融：False -> 不做频带混合输入（专家=对应频带）
        routing_mode: str = "learned",          # "learned" | "uniform" | "random"
    ):
        super().__init__()
        assert 1 <= top_k <= num_heads
        self.top_k = top_k
        self.alpha = alpha
        self.num_heads = num_heads
        self.band_sharpness = band_sharpness
        self.freq_affinity_sharpness = freq_affinity_sharpness
        self.use_soft_bands = use_soft_bands
        self.enable_freq_attn = enable_freq_attn
        self.enable_band_mixing = enable_band_mixing
        self.routing_mode = routing_mode

        # 频域 attention：qkv + proj（纯实数卷积，只吃幅度谱）
        self.qkv = nn.Conv2d(C, C * 3, 1)
        self.proj = nn.Conv2d(C, C, 1)

        # 频域特征 -> 专家权重（全局 gate，和频率偏好解耦）
        self.freq_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # [B,C,H,W] -> [B,C,1,1]
            nn.Flatten(1),             # [B,C]
            nn.Linear(C, num_heads)    # [B,num_heads]
        )

        # ==== 频率相关参数 & 统计 ====

        # 频带中心（在 r∈[0,1] 上均匀），长度 = num_heads（= num_bands）
        centers = torch.linspace(0.0, 1.0, steps=num_heads)  # [num_heads]
        self.register_buffer("band_centers", centers)

        # 每个专家的“频率偏好” meta-tag，初始化为均匀分布在 [0,1] 上
        # 形状 [num_heads]，可学习
        self.expert_freq = nn.Parameter(torch.linspace(0.0, 1.0, steps=num_heads))

        # 专家被选中次数（按 top-k 展开）
        self.register_buffer("expert_select_counts", torch.zeros(num_heads))
        # 累计 batch 数（可选，用来算平均 gate）
        self.register_buffer("num_batches_tracked", torch.tensor(0.0))
        # 最近一次 forward 的平均 gate 分布（仅用于分析/可视化）
        self.register_buffer("last_avg_gates", torch.zeros(num_heads))

        # 可选：缓存最近一次的“原始 band 特征”，用于可视化
        self.last_band_feats = None  # list[Tensor[B,C,H,W]]

    # ==== 统计控制 ====
    @torch.no_grad()
    def reset_stats(self):
        """重置路由统计量（在新一轮训练/实验前调用）"""
        self.expert_select_counts.zero_()
        self.num_batches_tracked.zero_()
        self.last_avg_gates.zero_()

    @torch.no_grad()
    def _update_stats(self, gates: torch.Tensor, top_k_indices: torch.Tensor):
        """
        内部使用：在 forward 里调用，用 top-k 结果和 gate 更新统计数据。
        gates:         [B,num_heads]
        top_k_indices: [B,top_k]
        """
        B = gates.size(0)
        idx = top_k_indices.reshape(-1)  # [B*top_k]
        one_hot = F.one_hot(idx, num_classes=self.num_heads).float().sum(dim=0)  # [num_heads]
        self.expert_select_counts += one_hot

        avg_g = gates.mean(dim=0)  # [num_heads]
        self.last_avg_gates = avg_g.detach()
        self.num_batches_tracked += 1.0

    @torch.no_grad()
    def get_stats(self) -> dict:
        """
        返回一个 dict，直接可以喂给可视化函数：
            - band_centers:         [num_heads]
            - expert_freq:          [num_heads]
            - band_select_counts:   [num_heads] （这里与 expert_select_counts 相同）
            - expert_select_counts: [num_heads]
            - avg_gates:            [num_heads] 最近一次的 gate 平均分布
        """
        return {
            "band_centers":         self.band_centers.detach().cpu().tolist(),
            "expert_freq":          self.expert_freq.detach().cpu().tolist(),
            "band_select_counts":   self.expert_select_counts.detach().cpu().tolist(),
            "expert_select_counts": self.expert_select_counts.detach().cpu().tolist(),
            "avg_gates":            self.last_avg_gates.detach().cpu().tolist(),
            "num_batches_tracked":  float(self.num_batches_tracked.item()),
        }

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # ===== 1. 频域变换 + fftshift，使 DC 到中心 =====
        Ftt = torch.fft.fft2(x, dim=(-2, -1))                   # [B,C,H,W], DC 在 (0,0)
        Ftt_shift = torch.fft.fftshift(Ftt, dim=(-2, -1))       # DC -> 中心

        # 在“中心坐标系”的频谱上做幅度与注意力
        F_amp = (Ftt_shift.real ** 2 + Ftt_shift.imag ** 2).sqrt()  # [B,C,H,W]

        # ===== 2. 频域注意力（简单空间注意力）=====
        if self.enable_freq_attn:
            q, k, v = self.qkv(F_amp).chunk(3, dim=1)               # 各 [B,C,H,W]

            energy = (q * k).sum(1, keepdim=True) / (C ** 0.5)      # [B,1,H,W]
            attn = torch.softmax(energy.view(B, 1, -1), dim=-1)     # 在 HW 上 softmax
            attn = attn.view(B, 1, H, W)                            # [B,1,H,W]

            attn_v = attn * v                                       # [B,C,H,W]
            F_refined = self.proj(attn_v)                           # [B,C,H,W]
        else:
            # 消融：直接使用幅度谱作为路由特征
            F_refined = F_amp

        # ===== 3. 全局专家门控（基于频域编码）=====
        if self.routing_mode == "uniform":
            gates = torch.full((B, self.num_heads), 1.0 / self.num_heads, device=x.device, dtype=F_refined.dtype)
            top_k_indices = torch.arange(self.num_heads, device=x.device).unsqueeze(0).repeat(B, 1)
            top_k_indices = top_k_indices[:, : self.top_k]
            top_k_weights = torch.full((B, self.top_k), 1.0 / self.top_k, device=x.device, dtype=F_refined.dtype)
        elif self.routing_mode == "random":
            gates = torch.full((B, self.num_heads), 1.0 / self.num_heads, device=x.device, dtype=F_refined.dtype)
            rand_idx = torch.rand(B, self.num_heads, device=x.device)
            top_k_indices = torch.argsort(rand_idx, dim=-1, descending=True)[:, : self.top_k]
            top_k_weights = torch.full((B, self.top_k), 1.0 / self.top_k, device=x.device, dtype=F_refined.dtype)
        else:
            gates = torch.softmax(self.freq_proj(F_refined), dim=-1)  # [B,num_heads]
            top_k_weights, top_k_indices = torch.topk(
                gates, k=self.top_k, dim=-1
            )  # [B,top_k], [B,top_k]

        # 归一化 top-k 权重
        top_k_weights = top_k_weights / top_k_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        # 更新路由统计
        self._update_stats(gates.detach(), top_k_indices.detach())

        # ===== 4. 负载均衡 aux_loss =====
        if self.alpha > 0.0:
            if top_k_indices.numel() == 0 or gates.numel() == 0:
                aux_loss = gates.sum() * 0.0   # 保证图不被打断
            else:
                # freq: 每个专家在 top-k 中被选中的频率分布
                idx = top_k_indices.reshape(-1).to(torch.long)
                freq = F.one_hot(idx, num_classes=self.num_heads).float().mean(dim=0)  # [num_heads]
                # prob: gate 的平均分布
                prob = gates.mean(dim=0)                                              # [num_heads]

                # target 可以取 freq 或均匀分布，看你实验
                target = freq
                # target = torch.full_like(freq, 1.0 / self.num_heads)

                aux_loss = F.mse_loss(prob, target) * float(self.alpha)
        else:
            aux_loss = None
        
        # ===== 5. 在“已 shift 的频谱”上做真正的同心圆分频 =====
        r = make_radius_grid(H, W).to(x.device)                 # 以中心为圆心, r∈[0,1]
        if self.use_soft_bands:
            band_masks = make_soft_bands(
                r, num_bands=self.num_heads, sharpness=self.band_sharpness
            )  # list(len=num_heads) of [1,1,H,W]
        else:
            band_masks = self._make_hard_bands(r)  # list(len=num_heads) of [1,1,H,W]

        band_feats: List[torch.Tensor] = []
        for mask in band_masks:
            # 与频谱 dtype 对齐（complex32/64 都可以）
            mask = mask.to(Ftt_shift.real.dtype).to(Ftt_shift.device)
            F_band_shift = Ftt_shift * mask                          # 在中心坐标系掩码
            F_band = torch.fft.ifftshift(F_band_shift, dim=(-2, -1)) # shift 回原坐标
            x_band = torch.fft.ifft2(F_band, dim=(-2, -1)).real      # 回到空间域
            band_feats.append(x_band)                                # [B,C,H,W]

        # 缓存一下原始 band 特征（纯频带分解），方便可视化
        self.last_band_feats = [bf.detach() for bf in band_feats]

        # ===== 6. 基于“专家频率偏好 vs 频带中心”的 soft compatibility，
        #          为每个专家构造自己的频带混合输入
        # ------------------------------------------------
        # expert_freq:   [E]，可学习 meta-tag
        # band_centers:  [B]（这里 B == E == num_heads）
        # compat[e,b] = -lambda * (f_e - c_b)^2
        # band_weights[e,b] = softmax_b compat[e,b]
        # expert_input[e] = sum_b band_weights[e,b] * band_feats[b]
        # ------------------------------------------------
        expert_freq = self.expert_freq           # [E]
        band_centers = self.band_centers         # [B] (==E)

        routed: List[torch.Tensor] = []
        if self.enable_band_mixing:
            diff = expert_freq[:, None] - band_centers[None, :]           # [E,B]
            compat = -self.freq_affinity_sharpness * (diff ** 2)          # [E,B]
            band_weights = F.softmax(compat, dim=-1)                      # [E,B]

            # 堆叠 band_feats: [num_bands, B, C, H, W]
            band_stack = torch.stack(band_feats, dim=0)

            for e in range(self.num_heads):
                # band_weights[e]: [num_bands]
                w_e = band_weights[e].view(-1, 1, 1, 1, 1)  # [num_bands,1,1,1,1]
                # 对所有 band 做加权和 -> [B,C,H,W]
                mixed = (w_e * band_stack).sum(dim=0)
                routed.append(mixed)
        else:
            # 消融：不做频带混合，专家直接对应各自的频带
            routed = band_feats

        # routed: list(len=num_heads) of [B,C,H,W]，
        # 现在代表“每个专家的频带混合输入”，而不是简单的“频带本身”
        return routed, top_k_weights, top_k_indices, aux_loss

    def _make_hard_bands(self, r: torch.Tensor) -> List[torch.Tensor]:
        """
        消融用：将每个 (H,W) 位置硬分配到最近的频带中心，返回 one-hot 掩码。
        """
        # r: [H,W], band_centers: [E]
        dist2 = (r.unsqueeze(0) - self.band_centers.view(-1, 1, 1)) ** 2  # [E,H,W]
        assign = dist2.argmin(dim=0)                                     # [H,W]
        bands: List[torch.Tensor] = []
        for i in range(self.num_heads):
            mask = (assign == i).float().unsqueeze(0).unsqueeze(0)       # [1,1,H,W]
            bands.append(mask)
        return bands


# ============================
#   自适应频段 MoE 组合模块
# ============================

class AdaptiveFreqMoE(nn.Module):
    """
    自适应频段 MoE（基于频率偏好的专家）：
    - experts: 任意顺序的专家列表（例如 [FNO, MNO, LNO, WNO]）
    - router: 负责在频谱上做注意力 & 产生专家 gate & 频带特征 & 频率偏好混合
    - forward:
        对每个样本 b：
          - router 给出 top_k_indices[b] & top_k_weights[b]
          - 激活对应专家 e_idx，并将该专家对应的“频带混合输入” routed[e_idx][b] 作为输入
          - 按 top_k_weights 做加权求和
    """
    def __init__(
        self, 
        experts: List[nn.Module],
        in_channels: int,
        topk: int,
        alpha: float = 0.0,
        band_sharpness: float = 20.0,
        freq_affinity_sharpness: float = 10.0,
        use_soft_bands: bool = True,
        enable_freq_attn: bool = True,
        enable_band_mixing: bool = True,
        routing_mode: str = "learned",
    ):
        super().__init__()
        # 注册为 ModuleList，确保参数被 optimizer 管理
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(self.experts)
        self.top_k = topk
        self.router_type = "sar"
        
        self.router = SpectralAttentionRouter(
            C=in_channels,
            num_heads=self.num_experts,         # 这里同时作为“专家数 & 频带数”
            top_k=topk,
            alpha=alpha,
            band_sharpness=band_sharpness,
            freq_affinity_sharpness=freq_affinity_sharpness,
            use_soft_bands=use_soft_bands,
            enable_freq_attn=enable_freq_attn,
            enable_band_mixing=enable_band_mixing,
            routing_mode=routing_mode,
        )

        # 最近一次 forward 的 routed 缓存（这里是“专家输入特征”）
        self._last_routed_bands = None   # list[Tensor[B,C,H,W]] on CPU

    # ==== 辅助接口 ====
    @torch.no_grad()
    def get_router_stats(self) -> dict:
        """
        直接转发 router.get_stats()，供可视化调用。
        """
        if hasattr(self.router, "get_stats"):
            return self.router.get_stats()
        return {}

    @torch.no_grad()
    def get_last_routed_bands(self):
        """
        返回最近一次 forward 时缓存的 routed（专家输入特征，CPU tensor 列表），
        可直接丢给可视化函数。
        """
        return self._last_routed_bands

    @torch.no_grad()
    def get_last_raw_bands(self):
        """
        返回最近一次 forward 时 router 里的原始 band_feats（频带分解结果），
        如果需要看“纯频带”而不是“专家输入混合”。
        """
        if getattr(self.router, "last_band_feats", None) is None:
            return None
        return [bf.cpu() for bf in self.router.last_band_feats]

    @torch.no_grad()
    def reset_router_stats(self):
        """
        重置 router 内部统计（训练新阶段/新实验前调用）。
        """
        if hasattr(self.router, "reset_stats"):
            self.router.reset_stats()

    def forward(self, x: torch.Tensor, weights=None, **kwargs):
        """
        x: [B, C, H, W]
        返回:
            y:        [B, Cout, H', W']  MoE 组合输出
            aux_loss: router 的正则（可直接加到总 loss 里）
        """
        routed_bands, top_k_weights, top_k_indices, aux_loss = self.router(x)
        B = x.size(0)
        outputs = []

        # 缓存 routed_bands（此时是“专家输入特征”）
        with torch.no_grad():
            self._last_routed_bands = [rb.detach().cpu() for rb in routed_bands]

        for b in range(B):
            y_b = 0
            for j in range(self.top_k):
                e_idx = top_k_indices[b, j].item()          # 哪个专家 (0..num_experts-1)
                w = top_k_weights[b, j].view(1, 1, 1, 1)    # 权重 broadcast

                # 选该专家对应的频率偏好混合特征作为输入
                expert_input = routed_bands[e_idx][b:b+1]   # [1,C,H,W]
                expert_out = self.experts[e_idx](expert_input, **kwargs) \
                    if callable(getattr(self.experts[e_idx], "forward", None)) \
                    else self.experts[e_idx](expert_input)

                y_b = y_b + w * expert_out

            outputs.append(y_b)

        y = torch.cat(outputs, dim=0)   # [B,Cout,H',W']
        return y, aux_loss

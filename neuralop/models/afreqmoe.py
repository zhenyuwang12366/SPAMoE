import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ======================
#   Frequency grid & soft bands
# ======================

def make_radius_grid(H: int, W: int) -> torch.Tensor:
    """
    Build a radius grid r in [0, 1] centered on the spectrum, shape [H, W].
    Intended for use with fftshifted spectra:
        - (H//2, W//2) is near DC
        - larger radius means higher frequency
    """
    ys = torch.linspace(-1.0, 1.0, steps=H)
    xs = torch.linspace(-1.0, 1.0, steps=W)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")      # [H,W]
    r = torch.sqrt(xx**2 + yy**2)                       # [0, sqrt(2)]
    r = r / r.max().clamp_min(1e-6)                     # normalize to [0,1]
    return r


def make_soft_bands(
    r: torch.Tensor,
    num_bands: int = 3,
    sharpness: float = 20.0,
) -> List[torch.Tensor]:
    """
    From radius r in [0,1], build num_bands soft masks:
    - band centers uniform in [0,1] (avoiding endpoints)
    - mask shape ~ Gaussian(r - center)

    Returns:
        bands: list of [1,1,H,W], low -> high frequency
    """
    H, W = r.shape
    centers = torch.linspace(0.0, 1.0, steps=num_bands)
    bands: List[torch.Tensor] = []
    for c in centers:
        band = torch.exp(-sharpness * (r - c) ** 2)  # Gaussian-like
        bands.append(band.view(1, 1, H, W))
    return bands


# ============================
#   Spectral attention router
# ============================

class SpectralAttentionRouter(nn.Module):
    """
    Spatial-domain input x [B, C, H, W]:
      1) FFT to spectrum Ftt
      2) fftshift so DC is centered for radial band splitting
      3) simple spatial attention on spectral magnitude -> F_refined
      4) global info from F_refined -> num_heads expert gates
      5) split spectrum into num_heads soft bands (concentric) by radius r
      6) learnable frequency preference expert_freq[i] in [0,1] for each expert;
         weight band features by similarity of expert_freq vs band_centers
         to form each expert's band-mixed input.

    Outputs:
      - routed:        list of [B, C, H, W], one entry per expert input (mixed by frequency preference)
      - top_k_weights: [B, top_k], normalized expert weights
      - top_k_indices: [B, top_k], selected expert indices (0 .. num_heads-1)
      - aux_loss:      scalar or None for MoE load balancing
    """
    def __init__(
        self, 
        C: int, 
        num_heads: int = 3,   # effectively num_experts here
        top_k: int = 2,
        alpha: float = 0.0,
        band_sharpness: float = 20.0,
        freq_affinity_sharpness: float = 10.0,  # sharpness of expert frequency preference
        use_soft_bands: bool = True,            # ablation: False -> hard band split
        enable_freq_attn: bool = True,          # ablation: False -> no frequency self-attention
        enable_band_mixing: bool = True,        # ablation: False -> no band mixing (expert = its band)
        enable_band_decomposition: bool = True,  # ablation: enable band decomposition
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
        self.enable_band_decomposition = enable_band_decomposition
        
        # Frequency-domain attention: qkv + proj (real convs on magnitude only)
        self.qkv = nn.Conv2d(C, C * 3, 1)
        self.proj = nn.Conv2d(C, C, 1)

        # Spectral features -> expert weights (global gate, decoupled from freq preference)
        self.freq_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # [B,C,H,W] -> [B,C,1,1]
            nn.Flatten(1),             # [B,C]
            nn.Linear(C, num_heads)    # [B,num_heads]
        )

        # ==== Frequency-related parameters & stats ====

        # Band centers uniform on r in [0,1], length = num_heads (= num_bands)
        centers = torch.linspace(0.0, 1.0, steps=num_heads)  # [num_heads]
        self.register_buffer("band_centers", centers)

        # Learnable frequency preference meta-tag per expert, init uniform on [0,1]
        # shape [num_heads]
        self.expert_freq = nn.Parameter(torch.linspace(0.0, 1.0, steps=num_heads))

        # Expert selection counts (expanded for top-k)
        self.register_buffer("expert_select_counts", torch.zeros(num_heads))
        # Batches seen (optional, for average gate)
        self.register_buffer("num_batches_tracked", torch.tensor(0.0))
        # Last forward average gate distribution (analysis / viz only)
        self.register_buffer("last_avg_gates", torch.zeros(num_heads))

        # Optional: cache last raw band features for visualization
        self.last_band_feats = None  # list[Tensor[B,C,H,W]]

    # ==== Stats control ====
    @torch.no_grad()
    def reset_stats(self):
        """Reset router statistics (call before a new train run / experiment)."""
        self.expert_select_counts.zero_()
        self.num_batches_tracked.zero_()
        self.last_avg_gates.zero_()

    @torch.no_grad()
    def _update_stats(self, gates: torch.Tensor, top_k_indices: torch.Tensor):
        """
        Internal: called from forward to update stats from top-k and gates.
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
        Dict ready for plotting:
            - band_centers:         [num_heads]
            - expert_freq:          [num_heads]
            - band_select_counts:   [num_heads] (same as expert_select_counts here)
            - expert_select_counts: [num_heads]
            - avg_gates:            [num_heads] last gate mean
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

        # ===== 1. Spectrum + fftshift, DC to center =====
        Ftt = torch.fft.fft2(x, dim=(-2, -1))                   # [B,C,H,W], DC at (0,0)
        Ftt_shift = torch.fft.fftshift(Ftt, dim=(-2, -1))       # DC -> center

        # Magnitude and attention in centered coordinates
        F_amp = (Ftt_shift.real ** 2 + Ftt_shift.imag ** 2).sqrt()  # [B,C,H,W]

        # ===== 2. Frequency-domain attention (simple spatial attention) =====
        if self.enable_freq_attn:
            q, k, v = self.qkv(F_amp).chunk(3, dim=1)               # each [B,C,H,W]

            energy = (q * k).sum(1, keepdim=True) / (C ** 0.5)      # [B,1,H,W]
            attn = torch.softmax(energy.view(B, 1, -1), dim=-1)   # softmax over HW
            attn = attn.view(B, 1, H, W)                            # [B,1,H,W]

            attn_v = attn * v                                       # [B,C,H,W]
            F_refined = self.proj(attn_v)                           # [B,C,H,W]
        else:
            # ablation: use magnitude spectrum as routing features
            F_refined = F_amp

        # ===== 3. Global expert gating (spectral encoding) =====
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

        # normalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        # update routing stats
        self._update_stats(gates.detach(), top_k_indices.detach())

        # ===== 4. load-balancing aux_loss =====
        if self.alpha > 0.0:
            if top_k_indices.numel() == 0 or gates.numel() == 0:
                aux_loss = gates.sum() * 0.0   # keep graph connected
            else:
                # freq: selection frequency per expert in top-k
                idx = top_k_indices.reshape(-1).to(torch.long)
                freq = F.one_hot(idx, num_classes=self.num_heads).float().mean(dim=0)  # [num_heads]
                # prob: mean gate distribution
                prob = gates.mean(dim=0)                                              # [num_heads]

                # target can be freq or uniform, depending on experiment
                target = freq
                # target = torch.full_like(freq, 1.0 / self.num_heads)

                aux_loss = F.mse_loss(prob, target) * float(self.alpha)
        else:
            aux_loss = None
        
        if self.enable_band_decomposition is False:
            # ablation: no band split, pass x to each expert
            routed = [x for _ in range(self.num_heads)]
            return routed, top_k_weights, top_k_indices, aux_loss
        
        # ===== 5. Concentric band split on shifted spectrum =====
        r = make_radius_grid(H, W).to(x.device)                 # center-based, r in [0,1]
        if self.use_soft_bands:
            band_masks = make_soft_bands(
                r, num_bands=self.num_heads, sharpness=self.band_sharpness
            )  # list(len=num_heads) of [1,1,H,W]
        else:
            band_masks = self._make_hard_bands(r)  # list(len=num_heads) of [1,1,H,W]

        band_feats: List[torch.Tensor] = []
        for mask in band_masks:
            # match spectrum dtype (complex32/64)
            mask = mask.to(Ftt_shift.real.dtype).to(Ftt_shift.device)
            F_band_shift = Ftt_shift * mask                          # mask in centered coords
            F_band = torch.fft.ifftshift(F_band_shift, dim=(-2, -1)) # shift back
            x_band = torch.fft.ifft2(F_band, dim=(-2, -1)).real      # spatial domain
            band_feats.append(x_band)                                # [B,C,H,W]

        # cache raw band features (pure band decomposition) for visualization
        self.last_band_feats = [bf.detach() for bf in band_feats]

        # ===== 6. Soft compatibility: expert frequency preference vs band centers
        #          -> per-expert band-mixed input
        # ------------------------------------------------
        # expert_freq:   [E], learnable meta-tag
        # band_centers:  [E] with E == num_heads
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

            # stack band_feats: [num_bands, B, C, H, W]
            band_stack = torch.stack(band_feats, dim=0)

            for e in range(self.num_heads):
                # band_weights[e]: [num_bands]
                w_e = band_weights[e].view(-1, 1, 1, 1, 1)  # [num_bands,1,1,1,1]
                # weighted sum over bands -> [B,C,H,W]
                mixed = (w_e * band_stack).sum(dim=0)
                routed.append(mixed)
        else:
            # ablation: no mixing; expert i = band i
            routed = band_feats

        # routed: list(len=num_heads) of [B,C,H,W],
        # each entry is the expert's band-mixed input, not a raw band alone
        return routed, top_k_weights, top_k_indices, aux_loss

    def _make_hard_bands(self, r: torch.Tensor) -> List[torch.Tensor]:
        """
        Ablation: hard-assign each (H,W) to nearest band center; return one-hot masks.
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
#   Adaptive frequency-band MoE
# ============================

class AdaptiveFreqMoE(nn.Module):
    """
    Adaptive frequency-band MoE (frequency-preference experts):
    - experts: arbitrary list (e.g. [FNO, MNO, LNO, WNO])
    - router: spectral attention, expert gates, band features, frequency-preference mixing
    - forward:
        per sample b:
          - router yields top_k_indices[b] & top_k_weights[b]
          - activate expert e_idx with mixed input routed[e_idx][b]
          - weighted sum by top_k_weights
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
        enable_band_decomposition: bool = True,
        routing_mode: str = "learned",
    ):
        super().__init__()
        # ModuleList so parameters are registered for the optimizer
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(self.experts)
        self.top_k = topk
        self.router_type = "sar"
        
        self.router = SpectralAttentionRouter(
            C=in_channels,
            num_heads=self.num_experts,         # expert count == band count
            top_k=topk,
            alpha=alpha,
            band_sharpness=band_sharpness,
            freq_affinity_sharpness=freq_affinity_sharpness,
            use_soft_bands=use_soft_bands,
            enable_freq_attn=enable_freq_attn,
            enable_band_mixing=enable_band_mixing,
            enable_band_decomposition=enable_band_decomposition,
            routing_mode=routing_mode,
        )

        # last forward routed cache (expert input features)
        self._last_routed_bands = None   # list[Tensor[B,C,H,W]] on CPU

    # ==== helpers ====
    @torch.no_grad()
    def get_router_stats(self) -> dict:
        """
        Forward to router.get_stats() for visualization.
        """
        if hasattr(self.router, "get_stats"):
            return self.router.get_stats()
        return {}

    @torch.no_grad()
    def get_last_routed_bands(self):
        """
        Last cached routed bands (expert inputs, CPU tensor list), for plotting.
        """
        return self._last_routed_bands

    @torch.no_grad()
    def get_last_raw_bands(self):
        """
        Last router band_feats (pure band decomposition) if you want raw bands
        instead of expert-mixed inputs.
        """
        if getattr(self.router, "last_band_feats", None) is None:
            return None
        return [bf.cpu() for bf in self.router.last_band_feats]

    @torch.no_grad()
    def reset_router_stats(self):
        """
        Reset router internal stats (new training stage / experiment).
        """
        if hasattr(self.router, "reset_stats"):
            self.router.reset_stats()

    def forward(self, x: torch.Tensor, weights=None, **kwargs):
        """
        x: [B, C, H, W]
        Returns:
            y:        [B, Cout, H', W']  MoE output
            aux_loss: router regularizer (add to total loss)
        """
        routed_bands, top_k_weights, top_k_indices, aux_loss = self.router(x)
        B = x.size(0)
        outputs = []

        # cache routed_bands (expert input features)
        with torch.no_grad():
            self._last_routed_bands = [rb.detach().cpu() for rb in routed_bands]

        for b in range(B):
            y_b = 0
            for j in range(self.top_k):
                e_idx = top_k_indices[b, j].item()          # expert index (0..num_experts-1)
                w = top_k_weights[b, j].view(1, 1, 1, 1)    # weight for broadcast

                # mixed feature for this expert's frequency preference
                expert_input = routed_bands[e_idx][b:b+1]   # [1,C,H,W]
                expert_out = self.experts[e_idx](expert_input, **kwargs) \
                    if callable(getattr(self.experts[e_idx], "forward", None)) \
                    else self.experts[e_idx](expert_input)

                y_b = y_b + w * expert_out

            outputs.append(y_b)

        y = torch.cat(outputs, dim=0)   # [B,Cout,H',W']
        return y, aux_loss

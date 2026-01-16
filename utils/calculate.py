import torch.nn.functional as F
import torch
import numpy as np
from .pytorch_ssim import SSIM
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

class SeismicMetrics:
    """
    地震数据评估指标
    约定：所有指标在 CPU + float32 下计算，避免 AMP/bfloat16 导致的 dtype 冲突。
    """
    DEFAULT_BANDS = ((0.0, 0.30), (0.30, 0.65), (0.65, 1.01))

    @staticmethod
    def _to_cpu_f32(x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            x = x.detach().cpu()
        else:
            x = x.detach()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return x

    @staticmethod
    def calculate_mse(pred, target):
        pred = SeismicMetrics._to_cpu_f32(pred)
        target = SeismicMetrics._to_cpu_f32(target)
        return F.mse_loss(pred, target).item()

    @staticmethod
    def calculate_mae(pred, target):
        pred = SeismicMetrics._to_cpu_f32(pred)
        target = SeismicMetrics._to_cpu_f32(target)
        return F.l1_loss(pred, target).item()

    @staticmethod
    def calculate_rmse(pred, target):
        pred = SeismicMetrics._to_cpu_f32(pred)
        target = SeismicMetrics._to_cpu_f32(target)
        mse = F.mse_loss(pred, target)
        return torch.sqrt(mse).item()

    @staticmethod
    def calculate_relative_l2(pred, target, eps=1e-12):
        """
        Relative L2 = ||pred - target||_2 / ||target||_2
        CPU + fp32 统一计算
        """
        pred = SeismicMetrics._to_cpu_f32(pred)
        target = SeismicMetrics._to_cpu_f32(target)

        numerator = torch.norm(pred - target, p=2)
        denominator = torch.norm(target, p=2)

        # 避免除零
        denom = max(float(denominator.item()), eps)

        return float((numerator.item() / denom))

    @staticmethod
    def calculate_psnr(pred, target, data_range=None):
        pred = SeismicMetrics._to_cpu_f32(pred)
        target = SeismicMetrics._to_cpu_f32(target)

        if data_range is None:
            data_range = target.max() - target.min()

        if isinstance(data_range, torch.Tensor):
            data_range = data_range.detach().cpu().item()
        data_range = float(data_range)

        eps = 1e-12
        data_range = max(data_range, eps)

        mse = float(F.mse_loss(pred, target).item())
        mse = max(mse, eps)

        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
        return psnr

    @staticmethod
    def calculate_ssim(pred, target, window_size: int = 11):
        pred = SeismicMetrics._to_cpu_f32(pred)
        target = SeismicMetrics._to_cpu_f32(target)

        pred_01 = pred / 2.0 + 0.5
        target_01 = target / 2.0 + 0.5

        ssim_loss = SSIM(window_size=window_size).to(device=pred_01.device, dtype=torch.float32)

        with torch.amp.autocast(device_type="cpu", enabled=False):
            ssim_val = ssim_loss(target_01, pred_01)

        return float(ssim_val.detach().cpu().item())

    @staticmethod
    def calculate_freq_band_metrics(
        pred: torch.Tensor,
        target: torch.Tensor,
        bands=DEFAULT_BANDS,
        eps: float = 1e-12,
    ):
        """
        在频域按照半径划分的 band 评估误差（默认 low/mid/high 三段）：
          - 对 pred/target 做 2D FFT + fftshift
          - 使用圆环 mask 做 band-pass，ifft 回空间域计算相对 L2 / MAE
          - 额外返回目标/预测在该频段的能量占比
        """
        pred = SeismicMetrics._to_cpu_f32(pred)
        target = SeismicMetrics._to_cpu_f32(target)
        if pred.shape != target.shape:
            raise ValueError(f"pred/target 形状不一致: {pred.shape} vs {target.shape}")

        _, _, H, W = pred.shape
        device = pred.device

        ys = torch.linspace(-1.0, 1.0, steps=H, device=device)
        xs = torch.linspace(-1.0, 1.0, steps=W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        r = torch.sqrt(xx**2 + yy**2)
        r = r / r.max().clamp_min(eps)

        masks = []
        for lo, hi in bands:
            mask = ((r >= float(lo)) & (r < float(hi))).to(torch.float32)
            masks.append(mask.view(1, 1, H, W))

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        pred_fft = torch.fft.fftshift(pred_fft, dim=(-2, -1))
        target_fft = torch.fft.fftshift(target_fft, dim=(-2, -1))

        total_pred_energy = torch.sum(torch.abs(pred_fft) ** 2).clamp_min(eps)
        total_tgt_energy = torch.sum(torch.abs(target_fft) ** 2).clamp_min(eps)

        band_labels = ["low", "mid", "high"] if len(bands) == 3 else None
        metrics = {}

        for idx, (lo, hi) in enumerate(bands):
            name = band_labels[idx] if band_labels and idx < len(band_labels) else f"band{idx}"
            mask = masks[idx]

            pred_band_fft = pred_fft * mask
            tgt_band_fft = target_fft * mask

            pred_band = torch.fft.ifft2(torch.fft.ifftshift(pred_band_fft, dim=(-2, -1)), dim=(-2, -1)).real
            tgt_band = torch.fft.ifft2(torch.fft.ifftshift(tgt_band_fft, dim=(-2, -1)), dim=(-2, -1)).real

            diff = pred_band - tgt_band
            rel_l2 = torch.norm(diff) / torch.norm(tgt_band).clamp_min(eps)
            mae = torch.mean(torch.abs(diff))

            pred_energy = torch.sum(torch.abs(pred_band_fft) ** 2)
            tgt_energy = torch.sum(torch.abs(tgt_band_fft) ** 2)

            metrics[name] = {
                "rel_l2": float(rel_l2.item()),
                "mae": float(mae.item()),
                "pred_energy_ratio": float((pred_energy / total_pred_energy).item()),
                "tgt_energy_ratio": float((tgt_energy / total_tgt_energy).item()),
                "band_range": (float(lo), float(hi)),
            }
        return metrics

    # ==== 统一指标计算接口 ====
    def __call__(self, pred, target):
        mse  = self.calculate_mse(pred, target)
        mae  = self.calculate_mae(pred, target)
        rmse = self.calculate_rmse(pred, target)
        psnr = self.calculate_psnr(pred, target)
        ssim_val = self.calculate_ssim(pred, target)
        rel_l2 = self.calculate_relative_l2(pred, target)

        return {
            "loss": float(mse),
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "psnr": float(psnr),
            "ssim": float(ssim_val),
            "relative_l2": float(rel_l2),
        }
        
def _radius_grid(H: int, W: int) -> np.ndarray:
    """
    返回 fftshift 后频谱中心为原点的归一化半径网格 r in [0,1]，shape [H,W]
    """
    cy = H // 2
    cx = W // 2
    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rr.max() if rr.max() > 0 else 1.0
    return (rr / rmax).astype(np.float32)


def _make_band_masks(
    H: int,
    W: int,
    low_band: Tuple[float, float] = (0.05, 0.30),
    high_band: Tuple[float, float] = (0.40, 0.85),
) -> Tuple[np.ndarray, np.ndarray]:
    r = _radius_grid(H, W)
    low = (r >= low_band[0]) & (r < low_band[1])
    high = (r >= high_band[0]) & (r < high_band[1])
    return low, high


def _fft_power(u: np.ndarray) -> np.ndarray:
    """
    u: [H,W] float
    return: power spectrum |FFTshift(FFT2(u))|^2, shape [H,W]
    """
    fu = np.fft.fft2(u)
    fu = np.fft.fftshift(fu)
    ps = np.abs(fu) ** 2
    return ps.astype(np.float64)


def _safe_mean(x: List[float]) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.mean(np.asarray(x, dtype=np.float64)))


def _quantiles(x: List[float], qs=(0.05, 0.50, 0.95)) -> Dict[str, float]:
    if len(x) == 0:
        return {f"p{int(q*100):02d}": float("nan") for q in qs}
    arr = np.asarray(x, dtype=np.float64)
    out = {}
    for q in qs:
        out[f"p{int(q*100):02d}"] = float(np.quantile(arr, q))
    return out


def _pearson_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """
    a,b: [H,W]
    """
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float((a @ b) / denom)


@dataclass
class SpectralAccumulator:
    """
    用于“假设检验”的全 batch 统计器。

    每个 sample 记录：
      - 对 u_front（encoder读出 / 插值输入）计算 E_H, E_L, HF(u_front)
      - 对 y_pred, y_gt 计算 E_H/E_L/HF
      - 计算若干“可用来验证 A1-A3 的代理量”
    """
    name: str = "exp"
    eps: float = 1e-12
    low_band: Tuple[float, float] = (0.05, 0.30)
    high_band: Tuple[float, float] = (0.40, 0.85)

    # 每条是 dict
    records: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.records is None:
            self.records = []

    def add_sample(
        self,
        u_front: Optional[np.ndarray],  # encoder读出 or 插值结果；允许 None（比如没encoder时）
        y_pred: np.ndarray,
        y_gt: np.ndarray,
        sample_id: Optional[int] = None,
    ) -> None:
        """
        u_front: [H,W] or None
        y_pred/y_gt: [H,W]
        """
        H, W = y_gt.shape[-2], y_gt.shape[-1]
        low_mask, high_mask = _make_band_masks(H, W, self.low_band, self.high_band)

        # gt
        ps_gt = _fft_power(y_gt)
        E_L_gt = float(ps_gt[low_mask].sum())
        E_H_gt = float(ps_gt[high_mask].sum())
        HF_gt = float(E_H_gt / (E_L_gt + self.eps))

        # pred
        ps_pr = _fft_power(y_pred)
        E_L_pr = float(ps_pr[low_mask].sum())
        E_H_pr = float(ps_pr[high_mask].sum())
        HF_pr = float(E_H_pr / (E_L_pr + self.eps))

        # front
        if u_front is not None:
            ps_u = _fft_power(u_front)
            E_L_u = float(ps_u[low_mask].sum())
            E_H_u = float(ps_u[high_mask].sum())
            HF_u = float(E_H_u / (E_L_u + self.eps))
            corr_u_gt_spec = _pearson_corr(np.log(ps_u + 1.0), np.log(ps_gt + 1.0))
        else:
            E_L_u = float("nan")
            E_H_u = float("nan")
            HF_u = float("nan")
            corr_u_gt_spec = float("nan")

        # ====== 假设检验用的“代理量” ======
        # A1（高频非收缩）：希望 E_H(u_front) / E_H(gt) 不小（接近 1）
        a1_ratio_h = (E_H_u / (E_H_gt + self.eps)) if np.isfinite(E_H_u) else float("nan")

        # A2（低频可控）：希望 E_L(u_front) / E_L(gt) 在同一数量级（~1）
        a2_ratio_l = (E_L_u / (E_L_gt + self.eps)) if np.isfinite(E_L_u) else float("nan")

        # A3（下游有界响应）：我们无法直接拿到理论 m,M
        # 但可用“前端到输出的频带增益”作为经验界：
        #   gH = E_H(pred)/E_H(u_front), gL = E_L(pred)/E_L(u_front)
        # 若 gH,gL 在数据集上“波动范围有限”，就支撑 A3。
        if np.isfinite(E_H_u) and E_H_u > 0:
            gH = float(E_H_pr / (E_H_u + self.eps))
        else:
            gH = float("nan")
        if np.isfinite(E_L_u) and E_L_u > 0:
            gL = float(E_L_pr / (E_L_u + self.eps))
        else:
            gL = float("nan")

        # 额外：预测与GT频谱相关
        corr_pr_gt_spec = _pearson_corr(np.log(ps_pr + 1.0), np.log(ps_gt + 1.0))

        rec = {
            "sample_id": int(sample_id) if sample_id is not None else None,
            "H": int(H),
            "W": int(W),

            "E_L_u": E_L_u, "E_H_u": E_H_u, "HF_u": HF_u,
            "E_L_pred": E_L_pr, "E_H_pred": E_H_pr, "HF_pred": HF_pr,
            "E_L_gt": E_L_gt, "E_H_gt": E_H_gt, "HF_gt": HF_gt,

            "a1_ratio_h": a1_ratio_h,
            "a2_ratio_l": a2_ratio_l,
            "gain_H_pred_over_u": gH,
            "gain_L_pred_over_u": gL,

            "spec_corr_u_gt": corr_u_gt_spec,
            "spec_corr_pred_gt": corr_pr_gt_spec,
        }
        self.records.append(rec)

    def summary(self) -> Dict[str, Any]:
        """
        输出：
          - 每个关键指标的 mean/p05/p50/p95
          - 用于 A3 的经验 m_hat/M_hat（基于增益分位数）
        """
        keys = [
            "a1_ratio_h", "a2_ratio_l",
            "gain_H_pred_over_u", "gain_L_pred_over_u",
            "HF_u", "HF_pred", "HF_gt",
            "spec_corr_u_gt", "spec_corr_pred_gt",
        ]

        out: Dict[str, Any] = {"name": self.name, "count": len(self.records)}
        for k in keys:
            vals = [r[k] for r in self.records if r.get(k) is not None and np.isfinite(r[k])]
            out[k] = {
                "mean": _safe_mean(vals),
                **_quantiles(vals, qs=(0.05, 0.50, 0.95)),
            }

        # A3：经验下界/上界（把 m,M 理解为增益的某种“保守界”）
        # 例如用 5% 分位当 m_hat，95% 分位当 M_hat（仅作经验佐证）
        gH_vals = [r["gain_H_pred_over_u"] for r in self.records if np.isfinite(r["gain_H_pred_over_u"])]
        gL_vals = [r["gain_L_pred_over_u"] for r in self.records if np.isfinite(r["gain_L_pred_over_u"])]

        def _bounds(vals: List[float]) -> Dict[str, float]:
            if len(vals) == 0:
                return {"m_hat": float("nan"), "M_hat": float("nan")}
            arr = np.asarray(vals, dtype=np.float64)
            return {
                "m_hat": float(np.quantile(arr, 0.05)),
                "M_hat": float(np.quantile(arr, 0.95)),
            }

        out["A3_empirical_bounds"] = {
            "gain_H": _bounds(gH_vals),
            "gain_L": _bounds(gL_vals),
        }
        return out
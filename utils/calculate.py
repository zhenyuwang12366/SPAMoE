import torch.nn.functional as F
import torch
import numpy as np
from .pytorch_ssim import SSIM

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

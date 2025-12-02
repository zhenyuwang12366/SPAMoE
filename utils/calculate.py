import torch.nn.functional as F
import torch
import numpy as np
from .pytorch_ssim import SSIM

class SeismicMetrics:
    """
    地震数据评估指标
    约定：所有指标在 CPU + float32 下计算，避免 AMP/bfloat16 导致的 dtype 冲突。
    """
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
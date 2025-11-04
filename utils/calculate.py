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
        # 统一搬到 CPU 并转为 float32
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
    def calculate_psnr(pred, target, data_range=None):
        """
        计算峰值信噪比：CPU + fp32
        """
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
        """
        SSIM：CPU + fp32，且在 SSIM 内部禁用 AMP，确保 conv2d 的权重与输入 dtype/device 完全一致。
        同时把 [-1,1] 归一化到 [0,1] 再计算（与原实现等价）。
        """
        pred = SeismicMetrics._to_cpu_f32(pred)
        target = SeismicMetrics._to_cpu_f32(target)

        # 归一化到 [0,1]
        pred_01 = pred / 2.0 + 0.5
        target_01 = target / 2.0 + 0.5

        # 在 CPU+fp32 下构造 SSIM（内部会根据输入动态匹配 window 的 dtype/device）
        ssim_loss = SSIM(window_size=window_size).to(device=pred_01.device, dtype=torch.float32)

        with torch.amp.autocast(device_type="cpu", enabled=False):
            ssim_val = ssim_loss(target_01, pred_01)

        # SSIM 实现返回的是标量 tensor（fp32），这里转 float
        return float(ssim_val.detach().cpu().item())

    # ==== 统一指标计算接口 ====
    def __call__(self, pred, target):
        """
        验证阶段统一接口：返回 {'loss','mae','mse','psnr','rmse','ssim'} 字典
        """
        mse  = self.calculate_mse(pred, target)
        mae  = self.calculate_mae(pred, target)
        rmse = self.calculate_rmse(pred, target)
        psnr = self.calculate_psnr(pred, target)
        ssim_val = self.calculate_ssim(pred, target)

        return {
            "loss": float(mse),
            "mae": float(mae),
            "mse": float(mse),
            "psnr": float(psnr),
            "rmse": float(rmse),
            "ssim": float(ssim_val),
        }
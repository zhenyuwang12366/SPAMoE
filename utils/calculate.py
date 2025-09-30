import torch.nn.functional as F
import torch
import numpy as np

class SeismicMetrics:
    """
    地震数据评估指标
    """
    @staticmethod
    def calculate_mse(pred, target):
        """计算均方误差"""
        return F.mse_loss(pred, target).item()
    
    @staticmethod
    def calculate_mae(pred, target):
        """计算平均绝对误差"""
        return F.l1_loss(pred, target).item()
    
    @staticmethod
    def calculate_psnr(pred, target, data_range=None):
        """计算峰值信噪比"""
        # 确保张量在CPU上
        if pred.is_cuda:
            pred = pred.detach().cpu()
        if target.is_cuda:
            target = target.detach().cpu()
            
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

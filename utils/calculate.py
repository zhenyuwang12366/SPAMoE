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
        
        # 确保data_range也在CPU上
        if isinstance(data_range, torch.Tensor) and data_range.is_cuda:
            data_range = data_range.detach().cpu()
        
        mse = F.mse_loss(pred, target).item()
        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
        return psnr

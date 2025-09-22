from pytorch_msssim import ssim
import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    """
    L1 + L2 + (可选) SSIM + (可选) gradient/edge loss
    参数:
        lambda_l1, lambda_l2 : L1、L2 权重
        lambda_ssim          : SSIM 权重 (默认0表示不用)
        lambda_grad          : 边缘梯度权重 (默认0表示不用)
    """
    def __init__(self,
                 lambda_l1: float = 0.3,
                 lambda_l2: float = 0.3,
                 lambda_ssim: float = 0.0,
                 lambda_grad: float = 0.0):
        super().__init__()
        self.lambda_l1   = lambda_l1
        self.lambda_l2   = lambda_l2
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def gradient_loss(self, pred, gt):
        # 简单 Sobel 一阶梯度差
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dx_gt   = gt[:, :, :, 1:]   - gt[:, :, :, :-1]
        dy_gt   = gt[:, :, 1:, :]   - gt[:, :, :-1, :]
        return (self.l1(dx_pred, dx_gt) + self.l1(dy_pred, dy_gt)) * 0.5

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        loss_l1 = self.l1(pred, gt)
        loss_l2 = self.l2(pred, gt)
        loss = self.lambda_l1 * loss_l1 + self.lambda_l2 * loss_l2

        # 可选 SSIM（越接近1越好，这里取 1-SSIM 作为损失）
        if self.lambda_ssim > 0:
            ssim_loss = 1 - ssim(pred, gt, data_range=1.0, size_average=True)
            loss += self.lambda_ssim * ssim_loss
        else:
            ssim_loss = torch.tensor(0.0, device=pred.device)

        # 可选梯度损失
        if self.lambda_grad > 0:
            grad_loss = self.gradient_loss(pred, gt)
            loss += self.lambda_grad * grad_loss
        else:
            grad_loss = torch.tensor(0.0, device=pred.device)

        # 返回总损失及各分量，方便日志记录
        return {
            "loss": loss,
            "l1": loss_l1.detach(),
            "l2": loss_l2.detach(),
            "ssim": ssim_loss.detach(),
            "grad": grad_loss.detach()
        }

class L1L2Loss(nn.Module):
    """
    组合 L1 + L2 损失
    loss = λ1 * L1Loss + λ2 * MSELoss

    Parameters
    ----------
    lambda_g1v : float
        L1 (MAE) 损失权重
    lambda_g2v : float
        L2 (MSE) 损失权重
    """
    def __init__(self, lambda_g1v: float = 1.0, lambda_g2v: float = 1.0):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.l1 = nn.L1Loss()   # MAE
        self.l2 = nn.MSELoss()  # MSE

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        pred : Tensor
            模型预测值
        gt : Tensor
            真实标签
        返回一个 dict:
            {
              "loss": 组合总损失 (tensor),
              "l1":   单独的 L1 损失 (tensor, 已 detach),
              "l2":   单独的 L2 损失 (tensor, 已 detach)
            }
        """
        loss_l1 = self.l1(pred, gt)
        loss_l2 = self.l2(pred, gt)
        total   = self.lambda_g1v * loss_l1 + self.lambda_g2v * loss_l2
        return {
            "loss": total,
            "l1": loss_l1.detach(),
            "l2": loss_l2.detach()
        }
    
from __future__ import annotations

from typing import Optional, Sequence

from pytorch_msssim import ssim
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "L1L2Loss",
    "GradL1",
    "FourierMagL1",
    "SobelLoss",
    "FourierMag_L1",
    "RelativeL2",
    "PDECombinedLoss",
]


class L1L2Loss(nn.Module):
    """λ1·L1 + λ2·L2 (returns individual components for logging)."""

    def __init__(self, lambda_g1v: float = 1.0, lambda_g2v: float = 1.0):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        loss_l1 = self.l1(pred, gt)
        loss_l2 = self.l2(pred, gt)
        total = self.lambda_g1v * loss_l1 + self.lambda_g2v * loss_l2
        return {
            "loss": total,
            "l1": loss_l1.detach(),
            "l2": loss_l2.detach(),
        }


class GradL1(nn.Module):
    """Sobel/Scharr gradient-magnitude L1 distance."""

    def __init__(
        self,
        kernel: str = "sobel",
        spacing: Optional[Sequence[float]] = None,
        reduction: str = "mean",
        padding_mode: str = "replicate",
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction
        self.padding_mode = padding_mode
        self.spacing = spacing

        if kernel == "sobel":
            kx = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        elif kernel == "scharr":
            kx = torch.tensor([[3., 0., -3.], [10., 0., -10.], [3., 0., -3.]])
        else:
            raise ValueError("kernel must be 'sobel' or 'scharr'")
        ky = kx.t()
        norm = kx.abs().sum()
        self.register_buffer("kx", kx.view(1, 1, 3, 3) / norm)
        self.register_buffer("ky", ky.view(1, 1, 3, 3) / norm)

    def _conv(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        weight = kernel.expand(channels, 1, 3, 3)
        if self.padding_mode == "zeros":
            x_padded = x
            padding = 1
        else:
            x_padded = F.pad(x, (1, 1, 1, 1), mode=self.padding_mode)
            padding = 0
        return F.conv2d(x_padded, weight, padding=padding, groups=channels)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gx_p, gy_p = self._conv(pred, self.kx), self._conv(pred, self.ky)
        gx_g, gy_g = self._conv(gt, self.kx), self._conv(gt, self.ky)

        if self.spacing is not None:
            dz, dx = self.spacing
            gx_p, gx_g = gx_p / dx, gx_g / dx
            gy_p, gy_g = gy_p / dz, gy_g / dz

        mag_p = torch.sqrt(gx_p.square() + gy_p.square() + 1e-12)
        mag_g = torch.sqrt(gx_g.square() + gy_g.square() + 1e-12)
        return F.l1_loss(mag_p, mag_g, reduction=self.reduction)


class FourierMagL1(nn.Module):
    """L1 distance between Fourier magnitudes, with optional high-frequency boost."""

    def __init__(
        self,
        dims: Sequence[int] = (-2, -1),
        reduction: str = "mean",
        fft_norm: Optional[str] = "ortho",
        use_window: bool = True,
        highfreq_gamma: float = 0.0,
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.dims = tuple(dims)
        self.reduction = reduction
        self.fft_norm = fft_norm
        self.use_window = use_window
        self.highfreq_gamma = float(highfreq_gamma)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_window:
            _, _, H, W = x.shape
            wy = torch.hann_window(H, dtype=x.dtype, device=x.device)
            wx = torch.hann_window(W, dtype=x.dtype, device=x.device)
            window = (wy[:, None] * wx[None, :]).view(1, 1, H, W)
            x = x * window
        return x - x.mean(dim=self.dims, keepdim=True)

    def _radial_weight(self, shape: Sequence[int], device, dtype) -> torch.Tensor:
        H, W = shape
        fy = torch.fft.fftfreq(H, d=1.0, device=device)
        fx = torch.fft.rfftfreq(W, d=1.0, device=device)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        r = torch.sqrt(yy.square() + xx.square()) / 0.5
        if self.highfreq_gamma > 0:
            return r.clamp_(0, 1).pow(self.highfreq_gamma).to(dtype)
        return torch.ones_like(r, dtype=dtype)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if pred.shape != gt.shape:
            raise ValueError("pred and gt must have identical shapes")
        if pred.ndim != 4:
            raise ValueError("expected 4D tensors [B,C,H,W]")

        pred = self._prep(pred)
        gt = self._prep(gt)

        F_pred = torch.fft.rfftn(pred, dim=self.dims, norm=self.fft_norm)
        F_gt = torch.fft.rfftn(gt, dim=self.dims, norm=self.fft_norm)

        mag_pred = F_pred.abs()
        mag_gt = F_gt.abs()

        H, W = pred.shape[-2:]
        weight = self._radial_weight((H, W), pred.device, pred.dtype)
        weight = weight.view(1, 1, H, weight.shape[-1])

        diff = (mag_pred - mag_gt).abs() * weight
        if self.reduction == "mean":
            return diff.mean()
        if self.reduction == "sum":
            return diff.sum()
        return diff


class SobelLoss(nn.Module):
    """Wrapper returning a dict so training code can log grad penalties easily."""

    def __init__(
        self,
        kernel: str = "sobel",
        spacing: Optional[Sequence[float]] = None,
        reduction: str = "mean",
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.impl = GradL1(kernel=kernel, spacing=spacing, reduction=reduction, padding_mode=padding_mode)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        loss_val = self.impl(pred, gt)
        return {"loss": loss_val}


class FourierMag_L1(nn.Module):
    """Wrapper returning a dict for compatibility with existing training loops."""

    def __init__(
        self,
        dims: Sequence[int] = (-2, -1),
        reduction: str = "mean",
        fft_norm: Optional[str] = "ortho",
        use_window: bool = True,
        highfreq_gamma: float = 0.0,
    ):
        super().__init__()
        self.impl = FourierMagL1(
            dims=dims,
            reduction=reduction,
            fft_norm=fft_norm,
            use_window=use_window,
            highfreq_gamma=highfreq_gamma,
        )

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        loss_val = self.impl(pred, gt)
        return {"loss": loss_val}

class RelativeL2(nn.Module):
    """
    Relative L2 loss:
        rel_L2 = ||pred - gt||_2 / (||gt||_2 + eps)

    - By default reduces over the batch with mean; suitable as the primary loss for PDEBench / Neural Operator.
    """

    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if pred.shape != gt.shape:
            raise ValueError("pred and gt must have identical shapes")

        B = pred.size(0)
        pred_flat = pred.reshape(B, -1).float()
        gt_flat = gt.reshape(B, -1).float()

        diff = pred_flat - gt_flat
        diff_norm = torch.norm(diff, p=2, dim=1)
        gt_norm = torch.norm(gt_flat, p=2, dim=1)

        rel = diff_norm / (gt_norm + self.eps)

        if self.reduction == "mean":
            return rel.mean()
        if self.reduction == "sum":
            return rel.sum()
        return rel  # shape [B]

class PDECombinedLoss(nn.Module):
    """
    Combined loss for PDE / Neural Operator training:

        loss = λ_rel * RelativeL2
             + λ_grad * GradL1(∇pred, ∇gt)         (optional)
             + λ_fourier * FourierMagL1(|F(pred)|, |F(gt)|) (optional)

    Returns a dict for convenient logging in training scripts:
        {
            "loss":    total_loss,
            "rel":     rel_l2.detach(),
            "grad":    grad_val.detach(),     # zero scalar if grad term disabled
            "fourier": fourier_val.detach(),  # zero scalar if Fourier term disabled
        }
    """

    def __init__(
        self,
        lambda_rel: float = 1.0,
        lambda_grad: float = 0.0,
        lambda_fourier: float = 0.0,
        # Grad term settings
        grad_kernel: str = "sobel",
        grad_reduction: str = "mean",
        grad_padding_mode: str = "replicate",
        grad_spacing: Optional[Sequence[float]] = None,
        # Fourier term settings
        fourier_dims: Sequence[int] = (-2, -1),
        fourier_reduction: str = "mean",
        fourier_fft_norm: Optional[str] = "ortho",
        fourier_use_window: bool = True,
        fourier_highfreq_gamma: float = 0.0,
    ):
        super().__init__()
        self.lambda_rel = float(lambda_rel)
        self.lambda_grad = float(lambda_grad)
        self.lambda_fourier = float(lambda_fourier)

        # Primary loss: Relative L2
        self.rel_loss = RelativeL2(reduction="mean")

        # Optional gradient loss
        self.grad_loss = (
            SobelLoss(
                kernel=grad_kernel,
                spacing=grad_spacing,
                reduction=grad_reduction,
                padding_mode=grad_padding_mode,
            )
            if self.lambda_grad > 0.0
            else None
        )

        # Optional frequency-domain loss
        self.fourier_loss = (
            FourierMag_L1(
                dims=fourier_dims,
                reduction=fourier_reduction,
                fft_norm=fourier_fft_norm,
                use_window=fourier_use_window,
                highfreq_gamma=fourier_highfreq_gamma,
            )
            if self.lambda_fourier > 0.0
            else None
        )

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        # Primary loss: Relative L2
        rel_val = self.rel_loss(pred, gt)
        total = self.lambda_rel * rel_val

        # Gradient loss
        if self.grad_loss is not None:
            grad_dict = self.grad_loss(pred, gt)
            grad_val = grad_dict["loss"]
            total = total + self.lambda_grad * grad_val
        else:
            grad_val = pred.new_zeros(())

        # Fourier-domain loss
        if self.fourier_loss is not None:
            fourier_dict = self.fourier_loss(pred, gt)
            fourier_val = fourier_dict["loss"]
            total = total + self.lambda_fourier * fourier_val
        else:
            fourier_val = pred.new_zeros(())

        return {
            "loss":    total,
            "rel":     float(rel_val.detach().item()),
            "grad":    float(grad_val.detach().item()),
            "fourier": float(fourier_val.detach().item()),
        }
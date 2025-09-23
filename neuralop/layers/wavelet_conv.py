"""
Wavelet convolution layer backed by pytorch-wavelets.
Supports configurable wavelet types while keeping the original API.
"""

import inspect
from numbers import Number
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import (
        DWT1DForward,
        DWT1DInverse,
        DWTForward,
        DWTInverse,
    )
    try:  # 3D is optional in pytorch-wavelets
        from pytorch_wavelets import DWT3DForward, DWT3DInverse
    except ImportError:  # pragma: no cover - optional dependency
        DWT3DForward = None
        DWT3DInverse = None
    _HAS_PYTORCH_WAVELETS = True
except ImportError:  # pragma: no cover - library not installed
    DWT1DForward = DWT1DInverse = DWTForward = DWTInverse = None
    DWT3DForward = DWT3DInverse = None
    _HAS_PYTORCH_WAVELETS = False

try:
    import torch.cuda.amp as amp
except ImportError:  # pragma: no cover - torch without cuda
    amp = None

from .base_spectral_conv import BaseSpectralConv


def _optimize_for_gpu(tensor: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
    """Move tensor to device/dtype and ensure contiguous memory."""
    if device is not None:
        tensor = tensor.to(device)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.contiguous()


_ORIENTATION_BY_DIM = {1: 1, 2: 3, 3: 7}
_PAD_MODE_MAP = {
    "constant": "zero",
    "zeros": "zero",
    "zero": "zero",
    "replicate": "replicate",
    "reflection": "reflect",
    "reflect": "reflect",
    "periodic": "periodization",
    "periodization": "periodization",
    "symmetric": "symmetric",
}


class WaveletConv(BaseSpectralConv):
    """Wavelet convolution using pytorch-wavelets DWT/IDWT implementations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_levels: Sequence[int],
        wavelet_type: str = "haar",
        wavelet_filter: int = 2,
        resolution_scaling_factor: Optional[Sequence[Number]] = None,
        max_n_levels: Optional[Sequence[int]] = None,
        complex_data: bool = False,
        separable: bool = False,
        factorization: Optional[str] = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: Optional[dict] = None,
        precision: str = "full",
        fno_block_precision: str = "full",
        ensure_even_shapes: bool = False,
        pad_mode: str = "constant",
        adaptive_padding: bool = False,
        device=None,
        dtype=None,
        use_checkpoint: bool = False,
        use_amp: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(device=device, dtype=dtype)

        if not _HAS_PYTORCH_WAVELETS:  # pragma: no cover - runtime safeguard
            raise ImportError(
                "WaveletConv requires the 'pytorch-wavelets' package. "
                "Install it with `pip install pytorch-wavelets`."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_levels = tuple(int(level) for level in n_levels)
        self.n_dim = len(self.n_levels)
        if self.n_dim not in (1, 2, 3):
            raise ValueError(f"WaveletConv only supports 1D/2D/3D data, got {self.n_dim}D input.")

        self.wavelet_type = wavelet_type
        self.wavelet_filter = wavelet_filter
        self.resolution_scaling_factor = resolution_scaling_factor
        self.max_n_levels = tuple(int(level) for level in max_n_levels) if max_n_levels else self.n_levels
        self.complex_data = complex_data
        self.separable = separable
        self.factorization = factorization
        self.rank = rank
        self.fixed_rank_modes = fixed_rank_modes
        self.implementation = implementation
        self.decomposition_kwargs = decomposition_kwargs or {}
        self.precision = precision
        self.fno_block_precision = fno_block_precision
        self.ensure_even_shapes = ensure_even_shapes
        self.pad_mode = pad_mode
        self.adaptive_padding = adaptive_padding
        self.use_checkpoint = use_checkpoint

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.cfloat if complex_data else torch.float32)
        self.param_dtype = torch.float32 if torch.is_complex_dtype(self.dtype) else self.dtype
        self.use_amp = use_amp and torch.cuda.is_available()

        self.wavelet_mode = _PAD_MODE_MAP.get(self.pad_mode, self.pad_mode)
        self.J = max(1, min(self.n_levels))
        self.n_orient = _ORIENTATION_BY_DIM[self.n_dim]

        self._build_wavelet_ops()
        self._idwt_accepts_size = "size" in inspect.signature(self.idwt.forward).parameters
        self._initialize_parameters()
        self.to(device=self.device)

    def _build_wavelet_ops(self) -> None:
        if self.n_dim == 1:
            self.dwt = DWT1DForward(J=self.J, wave=self.wavelet_type, mode=self.wavelet_mode, **self.decomposition_kwargs)
            self.idwt = DWT1DInverse(wave=self.wavelet_type, mode=self.wavelet_mode)
        elif self.n_dim == 2:
            self.dwt = DWTForward(J=self.J, wave=self.wavelet_type, mode=self.wavelet_mode, **self.decomposition_kwargs)
            self.idwt = DWTInverse(wave=self.wavelet_type, mode=self.wavelet_mode)
        else:  # 3D
            if DWT3DForward is None or DWT3DInverse is None:
                raise ImportError("3D wavelet convolutions require pytorch-wavelets with 3D support installed.")
            self.dwt = DWT3DForward(J=self.J, wave=self.wavelet_type, mode=self.wavelet_mode, **self.decomposition_kwargs)
            self.idwt = DWT3DInverse(wave=self.wavelet_type, mode=self.wavelet_mode)

    def _initialize_parameters(self) -> None:
        if not self.complex_data:
            self.weight_low = nn.Parameter(
                torch.empty(self.out_channels, self.in_channels, device=self.device, dtype=self.param_dtype)
            )
            self.weight_high = nn.Parameter(
                torch.empty(
                    self.out_channels,
                    self.in_channels,
                    self.n_orient,
                    device=self.device,
                    dtype=self.param_dtype,
                )
            )
            nn.init.xavier_normal_(self.weight_low)
            nn.init.xavier_normal_(self.weight_high)
        else:
            self.weight_low_real = nn.Parameter(
                torch.empty(self.out_channels, self.in_channels, device=self.device, dtype=self.param_dtype)
            )
            self.weight_low_imag = nn.Parameter(
                torch.empty(self.out_channels, self.in_channels, device=self.device, dtype=self.param_dtype)
            )
            self.weight_high_real = nn.Parameter(
                torch.empty(
                    self.out_channels,
                    self.in_channels,
                    self.n_orient,
                    device=self.device,
                    dtype=self.param_dtype,
                )
            )
            self.weight_high_imag = nn.Parameter(
                torch.empty(
                    self.out_channels,
                    self.in_channels,
                    self.n_orient,
                    device=self.device,
                    dtype=self.param_dtype,
                )
            )
            nn.init.xavier_normal_(self.weight_low_real)
            nn.init.xavier_normal_(self.weight_low_imag)
            nn.init.xavier_normal_(self.weight_high_real)
            nn.init.xavier_normal_(self.weight_high_imag)

        self.bias = nn.Parameter(torch.zeros(self.out_channels, device=self.device, dtype=self.param_dtype))

    def forward(self, x: torch.Tensor, output_shape: Optional[Sequence[int]] = None) -> torch.Tensor:
        if self.use_amp and amp is not None:
            with torch.amp.autocast():
                return self._forward_impl(x, output_shape)
        return self._forward_impl(x, output_shape)

    def _forward_impl(self, x: torch.Tensor, output_shape: Optional[Sequence[int]]) -> torch.Tensor:
        x = _optimize_for_gpu(x, self.device, None if self.complex_data else self.dtype)
        original_shape = x.shape[2:]
        target_shape = tuple(int(dim) for dim in (output_shape or original_shape))

        if not self.complex_data:
            yl, yh = self.dwt(x)
            yl_out, yh_out = self._apply_weights_to_coeffs(yl, yh)
            result = self._idwt_with_size(yl_out, yh_out, target_shape)
        else:
            yl_real, yh_real = self.dwt(x.real)
            yl_imag, yh_imag = self.dwt(x.imag)
            (
                yl_real_out,
                yh_real_out,
                yl_imag_out,
                yh_imag_out,
            ) = self._apply_complex_weights_to_coeffs(yl_real, yh_real, yl_imag, yh_imag)
            result_real = self._idwt_with_size(yl_real_out, yh_real_out, target_shape)
            result_imag = self._idwt_with_size(yl_imag_out, yh_imag_out, target_shape)
            result = torch.complex(result_real, result_imag)

        if result.shape[-len(target_shape):] != target_shape:
            interp_mode = "bilinear" if self.n_dim == 2 else "linear" if self.n_dim == 1 else "trilinear"
            result = F.interpolate(result, size=target_shape, mode=interp_mode, align_corners=False)

        return result

    def _idwt_with_size(
        self, yl: Optional[torch.Tensor], yh: Sequence[torch.Tensor], target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        if self._idwt_accepts_size:
            return self.idwt((yl, yh), size=target_shape)
        return self.idwt((yl, yh))

    def _apply_weights_to_coeffs(
        self,
        yl: Optional[torch.Tensor],
        yh: Sequence[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        bias_low = self.bias.view(1, -1, *([1] * self.n_dim))
        bias_high = self.bias.view(1, -1, 1, *([1] * self.n_dim))

        yl_out = None
        if yl is not None:
            yl_out = torch.einsum("oc,bc...->bo...", self.weight_low, yl) + bias_low

        yh_out: List[torch.Tensor] = []
        for coeff in yh:
            transformed = torch.einsum("ocp,bcp...->bop...", self.weight_high, coeff) + bias_high
            yh_out.append(transformed)

        return yl_out, yh_out

    def _apply_complex_weights_to_coeffs(
        self,
        yl_real: Optional[torch.Tensor],
        yh_real: Sequence[torch.Tensor],
        yl_imag: Optional[torch.Tensor],
        yh_imag: Sequence[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor], List[torch.Tensor]]:
        bias_low = self.bias.view(1, -1, *([1] * self.n_dim))
        bias_high = self.bias.view(1, -1, 1, *([1] * self.n_dim))

        yl_real_out = yl_imag_out = None
        if yl_real is not None and yl_imag is not None:
            real_real = torch.einsum("oc,bc...->bo...", self.weight_low_real, yl_real)
            imag_imag = torch.einsum("oc,bc...->bo...", self.weight_low_imag, yl_imag)
            yl_real_out = real_real - imag_imag + bias_low

            real_imag = torch.einsum("oc,bc...->bo...", self.weight_low_real, yl_imag)
            imag_real = torch.einsum("oc,bc...->bo...", self.weight_low_imag, yl_real)
            yl_imag_out = real_imag + imag_real

        yh_real_out: List[torch.Tensor] = []
        yh_imag_out: List[torch.Tensor] = []
        for coeff_real, coeff_imag in zip(yh_real, yh_imag):
            real_real = torch.einsum("ocp,bcp...->bop...", self.weight_high_real, coeff_real)
            imag_imag = torch.einsum("ocp,bcp...->bop...", self.weight_high_imag, coeff_imag)
            yh_real_out.append(real_real - imag_imag + bias_high)

            real_imag = torch.einsum("ocp,bcp...->bop...", self.weight_high_real, coeff_imag)
            imag_real = torch.einsum("ocp,bcp...->bop...", self.weight_high_imag, coeff_real)
            yh_imag_out.append(real_imag + imag_real)

        return yl_real_out, yh_real_out, yl_imag_out, yh_imag_out

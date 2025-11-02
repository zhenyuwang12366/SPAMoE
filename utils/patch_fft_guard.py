# patch_fft_guard.py
import functools
import torch

def fft_guard_forward(force_mixed_if_half: bool = True):
    """
    用于 SpectralConv.forward 的装饰器工厂：
    - 关闭 autocast，避免外层 AMP 把张量降到 bf16/fp16
    - 将输入 x 提升到 float32（若为复数则 complex64），确保 FFT 可用
    - 可选：若模块精度模式为 'half'，临时改为 'mixed'（FFT 后再半精度），调用后恢复
    """
    def deco(forward_fn):
        @functools.wraps(forward_fn)
        def wrapper(self, x, *args, **kwargs):
            # 记录并必要时临时覆写 fno_block_precision
            old_prec = getattr(self, "fno_block_precision", "full")
            if force_mixed_if_half and old_prec == "half":
                setattr(self, "fno_block_precision", "mixed")

            try:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    # 提升输入精度（避免 rfftn/fftn 在 bf16/fp16 崩溃）
                    if torch.is_tensor(x):
                        if x.is_complex():
                            if x.dtype != torch.complex64 and x.dtype != torch.complex128:
                                x = x.to(torch.complex64)
                        else:
                            if x.dtype != torch.float32:
                                x = x.to(torch.float32)
                    # 调用原始 forward（其内部逻辑完全不改）
                    return forward_fn(self, x, *args, **kwargs)
            finally:
                # 恢复精度设置
                if force_mixed_if_half:
                    setattr(self, "fno_block_precision", old_prec)
        return wrapper
    return deco


def patch_spectral_conv_forward(SpectralConvClass):
    """
    对传入的 SpectralConv 类做猴子补丁：用 fft_guard_forward 包裹其 forward。
    使用方法：
        from neuralop.layers.spectral_convolution import SpectralConv
        from patch_fft_guard import patch_spectral_conv_forward
        patch_spectral_conv_forward(SpectralConv)
    """
    SpectralConvClass.forward = fft_guard_forward(force_mixed_if_half=True)(SpectralConvClass.forward)
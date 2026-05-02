# patch_fft_guard.py
import functools
import torch

def fft_guard_forward(force_mixed_if_half: bool = True):
    """
    Decorator factory for SpectralConv.forward:
    - Disable autocast so outer AMP does not cast tensors to bf16/fp16
    - Promote input x to float32 (or complex64 if complex) so FFT is stable
    - Optional: if module precision mode is 'half', temporarily switch to 'mixed' (half after FFT), restore after
    """
    def deco(forward_fn):
        @functools.wraps(forward_fn)
        def wrapper(self, x, *args, **kwargs):
            # Record / temporarily override fno_block_precision
            old_prec = getattr(self, "fno_block_precision", "full")
            if force_mixed_if_half and old_prec == "half":
                setattr(self, "fno_block_precision", "mixed")

            try:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    # Promote input dtype (avoid rfftn/fftn failing on bf16/fp16)
                    if torch.is_tensor(x):
                        if x.is_complex():
                            if x.dtype != torch.complex64 and x.dtype != torch.complex128:
                                x = x.to(torch.complex64)
                        else:
                            if x.dtype != torch.float32:
                                x = x.to(torch.float32)
                    # Call original forward (logic unchanged)
                    return forward_fn(self, x, *args, **kwargs)
            finally:
                # Restore precision setting
                if force_mixed_if_half:
                    setattr(self, "fno_block_precision", old_prec)
        return wrapper
    return deco


def patch_spectral_conv_forward(SpectralConvClass):
    """
    Monkey-patch SpectralConv.forward with fft_guard_forward.

    Example:
        from neuralop.layers.spectral_convolution import SpectralConv
        from patch_fft_guard import patch_spectral_conv_forward
        patch_spectral_conv_forward(SpectralConv)
    """
    SpectralConvClass.forward = fft_guard_forward(force_mixed_if_half=True)(SpectralConvClass.forward)
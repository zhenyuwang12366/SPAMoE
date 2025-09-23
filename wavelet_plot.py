import numpy as np
import matplotlib.pyplot as plt
import pywt
import dtcwt

catlogs = ['coif4','db4','db8','sym4','coif5','sym8']
for catlog in catlogs:
    # 选择 db2 小波
    wavelet = pywt.Wavelet(catlog)
    # 返回离散化后的尺度函数 φ(t) 和小波函数 ψ(t)
    # level=10 表示迭代次数，越大曲线越平滑
    phi, psi, x = wavelet.wavefun(level=10)

    plt.figure(figsize=(8, 4))
    plt.plot(x, phi, label="Scaling function φ(t)", color='steelblue')
    plt.plot(x, psi, label="Wavelet function ψ(t)", color='tomato')
    plt.title(f"Daubechies {catlog} Wavelet Basis Functions")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/Users/kanameyuiki/{catlog}.png')
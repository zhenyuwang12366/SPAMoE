# -*- coding: utf-8 -*-
"""
Radar chart for SSIM (↑) across 10 categories:
Ours(EMO) vs InversionNet vs VelocityGAN vs UPFWI
依赖：matplotlib、numpy
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_multi_radar(categories, model2scores, 
                     title="SSIM by Category — Ours vs Baselines",
                     outfile="radar_ssim_multi.png",
                     rmin=None, rmax=None):
    """
    categories: list[str], 维度名称（10 个类别）
    model2scores: dict[str, list[float]], 每个模型 10 维 SSIM
    rmin/rmax: 可选的半径范围（默认自动根据数据决定）
    """
    # 基本检查
    N = len(categories)
    assert N >= 3, "雷达图至少需要 3 个维度"
    for name, vals in model2scores.items():
        assert len(vals) == N, f"{name} 维度不匹配（期望 {N}）"

    # 角度（闭合）
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # 自动半径范围（留出可视化余量）
    all_vals = np.concatenate([np.asarray(v) for v in model2scores.values()])
    _min, _max = float(np.min(all_vals)), float(np.max(all_vals))
    if rmin is None: rmin = max(0.0, round(_min - 0.02, 2))
    if rmax is None: rmax = min(1.0, round(_max + 0.02, 2))
    if rmax <= rmin: rmax = rmin + 0.05

    # 颜色与样式（可扩展）
    palette = {
        "Ours (SA-EMO)"   : "#E84C3D",  # 红
        "InversionNet" : "#1F77B4",  # 蓝
        "VelocityGAN"  : "#2CA02C",  # 绿
        "UPFWI"        : "#FF7F0E",  # 橙
    }
    line_styles = {
        "Ours (SA-EMO)"   : "-",
        "InversionNet" : "--",
        "VelocityGAN"  : "-.",
        "UPFWI"        : ":",
    }
    alphas = {
        "Ours (SA-EMO)"   : 0.25,
        "InversionNet" : 0.15,
        "VelocityGAN"  : 0.15,
        "UPFWI"        : 0.15,
    }

    # 画布
    fig = plt.figure(figsize=(7, 7), dpi=300)
    ax = plt.subplot(111, polar=True)
    ax.set_title(title, fontsize=12, pad=16)

    # 径向刻度
    ax.set_rlabel_position(90)
    ax.set_ylim(rmin, rmax)
    ticks = np.linspace(rmin, rmax, 5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.2f}" for t in ticks], fontsize=8, color="#555")

    # 维度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)

    # 逐模型绘制
    for name, vals in model2scores.items():
        vals_ = list(vals) + [vals[0]]
        color = palette.get(name, None)
        ls    = line_styles.get(name, "-")
        a     = alphas.get(name, 0.15)

        ax.plot(angles, vals_, color=color, linewidth=2.0 if "Ours" in name else 1.8,
                linestyle=ls, label=name)
        ax.fill(angles, vals_, color=color, alpha=a)

    # 网格/外观
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.spines["polar"].set_alpha(0.4)

    # 图例
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12),
              frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    # 10 个类别（示例，可改）
    categories = [
        "CurveVelA","CurveVelB","FlatVelA","FlatVelB",
        "CurveFaultA","CurveFaultB","FlatFaultA",
        "FlatFaultB","StyleA","StyleB"
    ]

    # ======= 把下面的示例值换成你的 SSIM（0~1） =======
    model2scores = {
        "Ours (SA-EMO)"  : [0.9564, 0.8750, 0.9999, 0.9949, 0.9897, 0.7753, 0.9963, 0.9262, 0.9604, 0.9345],
        "InversionNet": [0.8074, 0.6727, 0.9895, 0.9461, 0.9566, 0.6163, 0.9766, 0.7268, 0.8859, 0.6314],
        "VelocityGAN" : [0.8624, 0.7111, 0.9916, 0.9521, 0.9613, 0.5996, 0.9313, 0.7476, 0.8883, 0.6953],
        "UPFWI"       : [0.8443, 0.6614, 0.9563, 0.8774, 0.9495, 0.3941, 0.9340, 0.6937, 0.7846, 0.6102],
    }
    # ==============================================

    plot_multi_radar(categories, model2scores,
                     title="SSIM — Ours(EMO) vs InversionNet / VelocityGAN / UPFWI",
                     outfile="radar_ssim_multi.png")
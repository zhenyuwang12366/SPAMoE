# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.style.use('seaborn-v0_8-muted')
plt.rcParams['font.serif'] = ['DejaVu Sans']

font = FontProperties(size=6)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def draw_single_radar(algs, stats_ls, labels, rgrid_ticks, title, outfile):
    """绘制一张雷达图（一个组一张图）"""
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles_closed = np.concatenate((angles, [angles[0]]))

    fig = plt.figure()
    fig.set_facecolor('#FFFFFF')
    fig.subplots_adjust(wspace=0.5, hspace=0.20, top=0.85, bottom=0.05)

    ax = fig.add_subplot(111, polar=True)
    lines = []
    for i, stats in enumerate(stats_ls):
        vals = np.array(stats, dtype=float)
        vals_closed = np.concatenate((vals, [vals[0]]))
        line, = ax.plot(angles_closed, vals_closed, linewidth=0.75,
                        color=colors[i], label=algs[i])
        lines.append(line)
        ax.fill(angles_closed, vals_closed, alpha=0.10, color=colors[i])

    ax.set_rgrids(rgrid_ticks, font=FontProperties(size=6), color='grey')
    ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties=font,
                      fontname='DejaVu Sans')
    ax.spines['polar'].set_visible(False)
    # ax.set_title(title, fontsize=8, loc="center", y=-0.25,
    #              fontname='Times New Roman')
    for txt in ax.get_xticklabels():
        x, y = txt.get_position()
        txt.set_position((x, y - 0.01))
    ax.legend(handles=lines, loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=False, frameon=True,
              prop=FontProperties(size=6, family='DejaVu Sans'),
              framealpha=0.1)

    plt.savefig(outfile, format='pdf', dpi=500, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {outfile}")
    plt.close(fig)


# =========================
# 数据区（与你原来的保持一致）
# =========================

labels = ["CurveVelA","CurveVelB","FlatVelA","FlatVelB",
        "CurveFaultA","CurveFaultB","FlatFaultA",
        "FlatFaultB","StyleA","StyleB"]

# 图1：Closed-source LMMs
algs1 = ["Ours (SA-EMO)","InversionNet","VelocityGAN","UPFWI"]
stats_ls1 = [
    [0.9564, 0.8750, 0.9999, 0.9949, 0.9897, 0.7753, 0.9963, 0.9262, 0.9604, 0.9345],
    [0.8074, 0.6727, 0.9895, 0.9461, 0.9566, 0.6163, 0.9766, 0.7268, 0.8859, 0.6314],
    [0.8624, 0.7111, 0.9916, 0.9521, 0.9613, 0.5996, 0.9313, 0.7476, 0.8883, 0.6953],
    [0.8443, 0.6614, 0.9563, 0.8774, 0.9495, 0.3941, 0.9340, 0.6937, 0.7846, 0.6102],
]
rgrids1 = [0.25, 0.50, 0.75, 1.0]

# 逐一出图（一个组一张图）
draw_single_radar(algs1, stats_ls1, labels, rgrids1,
                  title="Comparative Evaluation of Model Performance across Geological Domains",
                  outfile="radar_seismic.pdf")
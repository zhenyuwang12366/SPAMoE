import os
import io
import re
import json
import time
from pathlib import Path
from typing import Optional, Dict, Union, Any, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.fft import fft2, fftshift
from PIL import Image


# ============================================================
#  通用工具：Figure → ndarray；保存 & 日志 (TB / WandB)
# ============================================================

def _fig_to_arrays(fig, dpi=200):
    """
    将 Figure 转为：
      - arr_chw: (3,H,W) 供 TensorBoard add_image 使用
      - arr_hwc: (H,W,3) 供 wandb.Image 使用
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    arr_hwc = np.array(im)                         # (H,W,3), uint8
    arr_chw = np.transpose(arr_hwc, (2, 0, 1))     # (3,H,W)
    return arr_chw, arr_hwc


def _log_figure(
    fig,
    save_path,
    tb_writer=None,
    tb_tag=None,
    step=None,
    wandb_run=None,
    wb_key=None,
    dpi=200,
):
    """
    保存 + 同步到 TensorBoard 和 W&B。三者都可选。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    arr_chw, arr_hwc = _fig_to_arrays(fig, dpi=dpi)

    if tb_writer is not None and tb_tag is not None:
        tb_writer.add_image(tb_tag, arr_chw, global_step=step)

    if wandb_run is not None and wb_key is not None:
        try:
            import wandb
            wandb_run.log({wb_key: wandb.Image(arr_hwc)}, step=step)
        except Exception as e:
            print(f"[warn] wandb log 失败：{e}")


# ============================================================
# 1. 类型预测日志保存
# ============================================================

def save_type_predictions_txt(
    logits: Optional[torch.Tensor],
    batch: Dict[str, Any],
    save_dir: Union[str, Path],
    epoch: int,
    config=None,                # 直接传 config（含 type_id_specific）
    filename: str = "type_predictions.txt",
    append: bool = True,
    is_logger: bool = False,
) -> Path:
    """
    将预测类型（未softmax的原始logits）与真实类型标签保存到本地txt文件。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_path = save_dir / filename

    # ---- 自动反转 type_id_specific: str->int  →  int->str ----
    id2name = {}
    if hasattr(config, "type_id_specific"):
        id2name = {v: k for k, v in getattr(config, "type_id_specific").items()}

    # ---- 批次大小 ----
    B = None
    if isinstance(batch, dict):
        for k in ('input', 'label', 'input_file'):
            if k in batch:
                v = batch[k]
                if isinstance(v, torch.Tensor):
                    B = v.shape[0]
                    break
                elif isinstance(v, (list, tuple)):
                    B = len(v)
                    break
    if B is None and isinstance(logits, torch.Tensor):
        B = logits.shape[0]
    if B is None:
        B = 0

    # ---- 预测结果（未softmax，仅argmax）----
    if isinstance(logits, torch.Tensor):
        pred_prob = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(pred_prob, dim=-1).detach().cpu().tolist()
        pred_prob = pred_prob.detach().cpu().tolist()
    else:
        pred_ids = [None] * B
        pred_prob = [None] * B

    # ---- 真实标签 ----
    true_ids = [None] * B
    if isinstance(batch, dict) and ('label' in batch):
        t = batch['label']
        if isinstance(t, torch.Tensor):
            true_ids = t.detach().cpu().tolist()
        elif isinstance(t, (list, tuple)):
            true_ids = list(t)

    # ---- 文件名（可选）----
    file_names = ["" for _ in range(B)]
    if isinstance(batch, dict) and ('input_file' in batch):
        f = batch['input_file']
        if isinstance(f, torch.Tensor):
            file_names = [str(x) for x in f]
        elif isinstance(f, (list, tuple)):
            file_names = list(f)

    # ---- 写入 txt ----
    mode = "a" if append else "w"
    try:
        with open(txt_path, mode, encoding="utf-8") as f:
            f.write(f"{'='*80}\n")
            f.write(f"Epoch {epoch+1} Type Prediction Results (raw logits)\n")
            f.write(f"{'='*80}\n")
            for i in range(B):
                pid = pred_ids[i]
                tid = true_ids[i] if i < len(true_ids) else None
                pname = id2name.get(pid, "N/A") if pid is not None else "N/A"
                tname = id2name.get(tid, "N/A") if tid is not None else "N/A"
                logits_row = pred_prob[i] if (pred_prob[i] is not None) else "None"
                fname = file_names[i] if i < len(file_names) else ""

                f.write(f"[{i:02d}] {fname}\n")
                f.write(f"  Pred Type: {pname} (id={pid})\n")
                f.write(f"  True Type: {tname} (id={tid})\n")
                f.write(f"  Pred Logits: {logits_row}\n\n")

        if is_logger:
            print(f"[Visualization] Type predictions saved to: {txt_path}")
    except Exception as e:
        if is_logger:
            print(f"[Visualization] Failed to save type predictions: {e}")

    return txt_path


# ============================================================
# 2. Loss 曲线（从日志解析）
# ============================================================

def plot_loss_curve(log_file, save_path=None):
    """
    从日志文件中解析并绘制 Train Loss 和 Val Loss 曲线
    自动适配是否存在 CE 列
    """
    with open(log_file, "r", encoding="utf-8") as f:
        text = f.read()

    pattern_with_ce = re.compile(
        r"""^\s*(\d+)\s*\|\s*                     # Epoch
            (\d+(?:\.\d+)?)\s*\|\s*              # Train Loss
            (\d+(?:\.\d+)?)\s*\|\s*              # Val Loss
            (\d+(?:\.\d+)?)\s*\|\s*              # MAE
            (\d+(?:\.\d+)?)\s*\|\s*              # MSE
            ([+-]?\d+(?:\.\d+)?)\s*\|\s*         # PSNR
            (\d+(?:\.\d+)?)\s*\|\s*              # RMSE
            (\d+(?:\.\d+)?)\s*\|\s*              # SSIM
            (\d+(?:\.\d+)?)\s*\|?\s*$            # CE（可选）
        """, re.MULTILINE | re.VERBOSE
    )

    pattern_without_ce = re.compile(
        r"""^\s*(\d+)\s*\|\s*                     # Epoch
            (\d+(?:\.\d+)?)\s*\|\s*              # Train Loss
            (\d+(?:\.\d+)?)\s*\|\s*              # Val Loss
            (\d+(?:\.\d+)?)\s*\|\s*              # MAE
            (\d+(?:\.\d+)?)\s*\|\s*              # MSE
            ([+-]?\d+(?:\.\d+)?)\s*\|\s*         # PSNR
            (\d+(?:\.\d+)?)\s*\|\s*              # RMSE
            (\d+(?:\.\d+)?)\s*\|?\s*$            # SSIM
        """, re.MULTILINE | re.VERBOSE
    )

    rows = [m.groups() for m in pattern_with_ce.finditer(text)]
    if rows:
        columns = ["Epoch", "Train Loss", "Val Loss", "MAE", "MSE", "PSNR", "RMSE", "SSIM", "CE"]
    else:
        rows = [m.groups() for m in pattern_without_ce.finditer(text)]
        columns = ["Epoch", "Train Loss", "Val Loss", "MAE", "MSE", "PSNR", "RMSE", "SSIM"]

    if not rows:
        raise ValueError("日志格式不匹配，请检查 log_file 是否包含正确的数值行")

    df = pd.DataFrame(rows, columns=columns).astype(float)
    df_grouped = df.groupby("Epoch").mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(df_grouped["Epoch"], df_grouped["Train Loss"], label="Train Loss", lw=2)
    plt.plot(df_grouped["Epoch"], df_grouped["Val Loss"], label="Val Loss", lw=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"保存曲线到 {save_path}")
        plt.close()
    else:
        plt.show()


# ============================================================
# 3. 输入/预测/真值可视化（空间域）
#    + 通用辅助函数
# ============================================================

def _squeeze_leading_ones(t: torch.Tensor) -> torch.Tensor:
    """去掉前面多余 size=1 的维度，例如 [1,C,H,W] -> [C,H,W]."""
    while t.dim() > 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def _ensure_2d(t: torch.Tensor) -> np.ndarray:
    """确保拿到 2D ndarray，用在“最终帧”这种场景。"""
    t = _squeeze_leading_ones(t)
    if t.dim() == 2:
        return t.cpu().numpy()
    if t.dim() == 3 and t.shape[0] == 1:
        return t[0].cpu().numpy()
    raise ValueError(f"Expected 2D or [1,H,W], got shape {list(t.shape)}")


def _to_channel_list_2d(t: torch.Tensor):
    """
    将张量转成若干 2D 通道列表（适用于非时间序列任务）：
      - [H,W]        -> [ (ndarray,) ]
      - [C,H,W]      -> [ (ndarray_0, ndarray_1, ... ) ]
    """
    t = _squeeze_leading_ones(t)
    if t.dim() == 2:
        return [t.cpu().numpy()]
    if t.dim() == 3:
        return [t[c].cpu().numpy() for c in range(t.shape[0])]
    # 尽量 squeeze 再试一次
    t2 = t.squeeze()
    if t2.dim() == 2:
        return [t2.cpu().numpy()]
    raise ValueError(f"Cannot convert tensor with shape {list(t.shape)} to [C,H,W] style.")


def visualize_results(
    inputs,
    targets,
    predictions,
    save_dir='./results',
    max_samples=4,
    tb_writer=None,
    wandb_run=None,
    global_step=None,
    log_prefix='vis',
    task: str = 'generic',   # 'navier' | 'plasticity' | 其他
):
    """
    可视化输入 / 输出（空间域）：

    - 时间序列任务（navier, plasticity）：
        * navier: 输入选取若干时间帧，输出取最终时间步
        * plasticity: 输入为参数场，输出为 4*T 的展开张量，取最终时间步
    - 其他任务（pipe / airfoil / darcy / seismic / generic）：
        * 输入所有通道：每个通道一个子图
        * 输出所有通道：target/pred 每个通道一个子图

    要求：
        * input 通道共用一套归一化 (norm_in)，但每个子图有自己的颜色条
        * target + pred 通道共用一套归一化 (norm_out)，但每个子图有自己的颜色条
        * 颜色条放在各自子图旁边
    """
    os.makedirs(save_dir, exist_ok=True)

    n_samples = min(inputs.shape[0], max_samples)

    for i in range(n_samples):
        inp = inputs[i].detach().cpu()
        tgt = targets[i].detach().cpu()
        pred = predictions[i].detach().cpu()

        fig = None

        # ==============================
        # 1) navier: u(t,x,y)，有时间维
        # ==============================
        if task == 'navier':
            inp = _squeeze_leading_ones(inp)   # [T_in,H,W]
            tgt = _squeeze_leading_ones(tgt)   # [T_out,H,W]
            pred = _squeeze_leading_ones(pred)

            if inp.dim() != 3 or tgt.dim() != 3 or pred.dim() != 3:
                print(f"[visualize_results][navier] Unexpected dims for sample {i}, skip.")
                continue

            T_in, H, W = inp.shape
            T_out = tgt.shape[0]

            # 输入：3 个关键帧
            if T_in >= 3:
                idx_in = [0, T_in // 2, T_in - 1]
            else:
                idx_in = list(range(T_in))
            input_snaps = [inp[t].numpy() for t in idx_in]

            # 输出：最终时间步
            tgt_last = tgt[-1].numpy()
            pred_last = pred[-1].numpy()
            err_last = np.abs(pred_last - tgt_last)

            vmin = float(tgt_last.min())
            vmax = float(tgt_last.max())
            norm = Normalize(vmin=vmin, vmax=vmax)

            fig, axes = plt.subplots(
                2, 3,
                figsize=(15, 8),
                constrained_layout=True
            )

            # 第一行：输入关键帧
            for k in range(3):
                ax = axes[0, k]
                if k < len(idx_in):
                    im = ax.imshow(input_snaps[k], cmap='viridis', aspect='auto')
                    ax.set_title(f'Input u(t={idx_in[k]})', fontsize=10)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])

            # 第二行：最终帧 target/pred/error
            ax_t = axes[1, 0]
            im_t = ax_t.imshow(tgt_last, cmap='jet', norm=norm, aspect='auto')
            ax_t.set_title(f'Target u(t={T_out-1})', fontsize=10)
            plt.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)
            ax_t.set_xticks([]); ax_t.set_yticks([])

            ax_p = axes[1, 1]
            im_p = ax_p.imshow(pred_last, cmap='jet', norm=norm, aspect='auto')
            ax_p.set_title(f'Pred u(t={T_out-1})', fontsize=10)
            plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)
            ax_p.set_xticks([]); ax_p.set_yticks([])

            ax_e = axes[1, 2]
            im_e = ax_e.imshow(err_last, cmap='magma', aspect='auto')
            ax_e.set_title('Abs error (final step)', fontsize=10)
            plt.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04)
            ax_e.set_xticks([]); ax_e.set_yticks([])

        # ==============================
        # 2) plasticity: 输出 4*T 展平，有时间维
        # ==============================
        elif task == 'plasticity':
            # 输入为参数场
            param_field = _ensure_2d(inp)  # -> [H,W]

            tgt = _squeeze_leading_ones(tgt)   # [4*T,H,W]
            pred = _squeeze_leading_ones(pred)

            if tgt.dim() != 3 or pred.dim() != 3:
                print(f"[visualize_results][plasticity] Unexpected dims for sample {i}, skip.")
                continue
            C, H, W = tgt.shape
            if C % 4 != 0:
                print(f"[visualize_results][plasticity] C={C} not divisible by 4, skip sample {i}")
                continue

            T = C // 4
            t_last = T - 1

            tgt_last = []
            pred_last = []
            for comp in range(4):
                idx = comp * T + t_last
                tgt_last.append(tgt[idx].numpy())
                pred_last.append(pred[idx].numpy())

            all_vals = np.concatenate([np.stack(tgt_last), np.stack(pred_last)])
            vmin = float(all_vals.min())
            vmax = float(all_vals.max())
            norm = Normalize(vmin=vmin, vmax=vmax)

            fig, axes = plt.subplots(
                3, 4,
                figsize=(18, 10),
                constrained_layout=True
            )

            # 第 0 行：param + 空白
            ax_p = axes[0, 0]
            im_param = ax_p.imshow(param_field, cmap='viridis', aspect='auto')
            ax_p.set_title('Input parameter', fontsize=10)
            plt.colorbar(im_param, ax=ax_p, fraction=0.046, pad=0.04)
            ax_p.set_xticks([]); ax_p.set_yticks([])

            for c in range(1, 4):
                axes[0, c].axis('off')

            # 第 1 行：4 个 target 分量
            for comp in range(4):
                ax = axes[1, comp]
                im_t = ax.imshow(tgt_last[comp], cmap='jet', norm=norm, aspect='auto')
                ax.set_title(f'Target comp{comp} (t={t_last})', fontsize=10)
                plt.colorbar(im_t, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks([]); ax.set_yticks([])

            # 第 2 行：4 个 pred 分量
            for comp in range(4):
                ax = axes[2, comp]
                im_p = ax.imshow(pred_last[comp], cmap='jet', norm=norm, aspect='auto')
                ax.set_title(f'Pred comp{comp} (t={t_last})', fontsize=10)
                plt.colorbar(im_p, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks([]); ax.set_yticks([])

        # ==============================
        # 3) 其他任务：所有通道展开（seismic 也走这里）
        # ==============================
        else:
            in_chs = _to_channel_list_2d(inp)   # List[np.ndarray], 每个 [H,W]
            tgt_chs = _to_channel_list_2d(tgt)
            pred_chs = _to_channel_list_2d(pred)

            C_in = len(in_chs)
            C_tgt = len(tgt_chs)
            C_pred = len(pred_chs)
            C_max = max(C_in, C_tgt, C_pred)

            if C_max == 0:
                print(f"[visualize_results][{task}] No channels for sample {i}, skip.")
                continue

            # ---- 统一 input 的归一化 ----
            if C_in > 0:
                in_vals = np.concatenate([c.ravel() for c in in_chs])
                vmin_in = float(in_vals.min())
                vmax_in = float(in_vals.max())
                norm_in = Normalize(vmin=vmin_in, vmax=vmax_in)
            else:
                norm_in = None

            # ---- 统一 target + pred 的归一化 ----
            out_arrays = []
            if C_tgt > 0:
                out_arrays.append(np.concatenate([c.ravel() for c in tgt_chs]))
            if C_pred > 0:
                out_arrays.append(np.concatenate([c.ravel() for c in pred_chs]))
            if len(out_arrays) > 0:
                out_vals = np.concatenate(out_arrays)
                vmin_out = float(out_vals.min())
                vmax_out = float(out_vals.max())
                norm_out = Normalize(vmin=vmin_out, vmax=vmax_out)
            else:
                norm_out = None

            # ====== 3 行：inputs / targets / preds ======
            rows, cols = 3, C_max
            fig, axes = plt.subplots(
                rows, cols,
                figsize=(3.2 * cols, 9.0),
                constrained_layout=True
            )

            # ---- 关键修复：强制 axes 变成 (rows, cols) 形状，避免 C_max=1 时变成 1D ----
            axes = np.array(axes)
            if axes.ndim == 0:
                # 单个 Axes 对象 -> (1,1)
                axes = axes[None, None]
            elif axes.ndim == 1:
                # 1D 情况有两种：
                #   - rows==1, cols>1  -> shape (cols,)
                #   - rows>1, cols==1  -> shape (rows,)
                if rows == 1 and cols > 1:
                    axes = axes[None, :]        # (cols,) -> (1, cols)
                elif cols == 1 and rows > 1:
                    axes = axes[:, None]        # (rows,) -> (rows, 1)

            assert axes.shape[0] == rows and axes.shape[1] == cols, \
                f"axes shape {axes.shape} != ({rows},{cols})"

            # row 0: inputs
            for c in range(C_max):
                ax = axes[0, c]
                if c < C_in:
                    im = ax.imshow(
                        in_chs[c],
                        cmap='viridis',
                        norm=norm_in,
                        aspect='auto'
                    )
                    ax.set_title(f'Input ch{c}', fontsize=10)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])

            # row 1: targets
            for c in range(C_max):
                ax = axes[1, c]
                if c < C_tgt:
                    im = ax.imshow(
                        tgt_chs[c],
                        cmap='jet',
                        norm=norm_out,
                        aspect='auto'
                    )
                    ax.set_title(f'Target ch{c}', fontsize=10)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])

            # row 2: predictions
            for c in range(C_max):
                ax = axes[2, c]
                if c < C_pred:
                    im = ax.imshow(
                        pred_chs[c],
                        cmap='jet',
                        norm=norm_out,
                        aspect='auto'
                    )
                    ax.set_title(f'Pred ch{c}', fontsize=10)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])

        # ==== 统一的日志保存部分 ====
        if fig is not None:
            png_path = os.path.join(save_dir, f'sample_{i}_{task}.png')
            tb_tag = f"{log_prefix}/{task}/sample_{i}/composed"
            wb_key = f"{log_prefix}/{task}/sample_{i}/figure"

            _log_figure(
                fig,
                png_path,
                tb_writer=tb_writer,
                tb_tag=tb_tag,
                step=global_step,
                wandb_run=wandb_run,
                wb_key=wb_key,
                dpi=300,
            )

            plt.close(fig)


# ============================================================
# 4. Fourier 频域分析（空间 + 频谱，支持时间序列任务）
# ============================================================

def analyze_fourier_domain(
    inputs,
    targets,
    predictions,
    save_dir='./results',
    max_samples=4,
    tb_writer=None,
    wandb_run=None,
    global_step=None,
    log_prefix='vis',
    task: str = 'generic',   # 'navier' | 'plasticity' | 其他
):
    """
    分析输入与输出在傅里叶域的特点：

    - 时间序列任务（navier, plasticity）：
        * 选定若干关键帧/最终帧，分别作为“输入面板”和“输出面板”
    - 其他任务：
        * Inputs：所有通道 → spatial_inputs
        * Outputs：Target/Pred 所有通道 → spatial_outputs

    空间域可视化：
        * Input 面板使用统一归一化 norm_in，但每个子图各自有一条颜色条
        * Target + Pred 面板使用统一归一化 norm_out，但每个子图各自有一条颜色条

    频域可视化：
        * 所有 PSD 面板使用同一 log10 归一化，但每个子图各自有一条颜色条
    """
    os.makedirs(save_dir, exist_ok=True)
    n_samples = min(inputs.shape[0], max_samples)

    def compute_hf_lf_ratio(psd, low_band=(0.05, 0.3), high_band=(0.4, 0.85)):
        h, w = psd.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_norm = r / (np.sqrt(cy ** 2 + cx ** 2) + 1e-12)
        lf_mask = (r_norm >= low_band[0]) & (r_norm < low_band[1])
        hf_mask = (r_norm >= high_band[0]) & (r_norm < high_band[1])
        low_e = float(np.sum(psd[lf_mask]))
        high_e = float(np.sum(psd[hf_mask]))
        if low_e < 1e-12:
            return 0.0
        hf_ratio = high_e / (low_e + 1e-12)
        return float(np.clip(hf_ratio, 1e-6, 1e3))

    def _fft2_psd(field_2d: np.ndarray) -> np.ndarray:
        return np.abs(fftshift(fft2(field_2d))) ** 2

    for i in range(n_samples):
        try:
            print(f"[Fourier] Processing sample {i}, task={task}...")

            inp = inputs[i].detach().cpu()
            tgt = targets[i].detach().cpu()
            pred = predictions[i].detach().cpu()

            # 分开维护：输入面板 / 输出面板（target + pred）
            spatial_inputs: List[tuple[str, np.ndarray]] = []
            spatial_outputs: List[tuple[str, np.ndarray]] = []
            psd_panels: List[tuple[str, np.ndarray]] = []

            main_tgt_field = None
            main_pred_field = None

            # ===================== navier =====================
            if task == 'navier':
                inp = _squeeze_leading_ones(inp)   # [T_in,H,W]
                tgt = _squeeze_leading_ones(tgt)   # [T_out,H,W]
                pred = _squeeze_leading_ones(pred)

                if inp.dim() != 3 or tgt.dim() != 3 or pred.dim() != 3:
                    print(f"[Fourier][navier] Unexpected dims, skip sample {i}")
                    continue

                T_in, H, W = inp.shape
                T_out = tgt.shape[0]

                # 输入 3 帧
                if T_in >= 3:
                    idx_in = [0, T_in // 2, T_in - 1]
                else:
                    idx_in = list(range(T_in))
                for t_idx in idx_in:
                    spatial_inputs.append((f"Input u(t={t_idx})", inp[t_idx].numpy()))

                # 输出最终帧
                tgt_last = tgt[-1].numpy()
                pred_last = pred[-1].numpy()
                main_tgt_field = tgt_last
                main_pred_field = pred_last
                spatial_outputs.append((f"Target u(t={T_out-1})", tgt_last))
                spatial_outputs.append((f"Pred u(t={T_out-1})", pred_last))

            # ===================== plasticity =====================
            elif task == 'plasticity':
                # 输入参数场
                param_field = _ensure_2d(inp)
                spatial_inputs.append(("Input parameter", param_field))

                tgt = _squeeze_leading_ones(tgt)   # [4*T,H,W]
                pred = _squeeze_leading_ones(pred)

                if tgt.dim() != 3 or pred.dim() != 3:
                    print(f"[Fourier][plasticity] Unexpected dims, skip sample {i}")
                    continue
                C, H, W = tgt.shape
                if C % 4 != 0:
                    print(f"[Fourier][plasticity] C={C} not divisible by 4, skip sample {i}")
                    continue

                T = C // 4
                t_last = T - 1

                tgt_last = []
                pred_last = []
                for comp in range(4):
                    idx = comp * T + t_last
                    t_field = tgt[idx].numpy()
                    p_field = pred[idx].numpy()
                    tgt_last.append(t_field)
                    pred_last.append(p_field)
                    spatial_outputs.append((f"Target comp{comp}(t={t_last})", t_field))
                    spatial_outputs.append((f"Pred comp{comp}(t={t_last})", p_field))

                # 主评估：comp0 的最终帧
                main_tgt_field = tgt_last[0]
                main_pred_field = pred_last[0]

            # ===================== 其他任务：所有通道 =====================
            else:
                in_chs = _to_channel_list_2d(inp)
                tgt_chs = _to_channel_list_2d(tgt)
                pred_chs = _to_channel_list_2d(pred)

                for c, ch in enumerate(in_chs):
                    spatial_inputs.append((f"Input ch{c}", ch))
                for c, ch in enumerate(tgt_chs):
                    spatial_outputs.append((f"Target ch{c}", ch))
                for c, ch in enumerate(pred_chs):
                    spatial_outputs.append((f"Pred ch{c}", ch))

                if len(tgt_chs) == 0 or len(pred_chs) == 0:
                    print(f"[Fourier][{task}] No target/pred channels, skip sample {i}")
                    continue
                main_tgt_field = tgt_chs[0]
                main_pred_field = pred_chs[0]

            if main_tgt_field is None or main_pred_field is None:
                print(f"[Fourier] main fields missing, skip sample {i}")
                continue

            # ====== 计算 PSD ======
            all_spatial_panels = spatial_inputs + spatial_outputs
            for title, field in all_spatial_panels:
                psd_panels.append((title, _fft2_psd(field)))

            tgt_psd = _fft2_psd(main_tgt_field)
            pred_psd = _fft2_psd(main_pred_field)

            # ====== 频谱指标 ======
            h, w = tgt_psd.shape
            cy, cx = h // 2, w // 2

            idx_t = np.unravel_index(np.argmax(tgt_psd), tgt_psd.shape)
            idx_p = np.unravel_index(np.argmax(pred_psd), pred_psd.shape)
            tgt_domf = float(np.hypot(idx_t[0] - cy, idx_t[1] - cx))
            pred_domf = float(np.hypot(idx_p[0] - cy, idx_p[1] - cx))

            tgt_hf_ratio = compute_hf_lf_ratio(tgt_psd)
            pred_hf_ratio = compute_hf_lf_ratio(pred_psd)

            tgt_psd_mean, tgt_psd_max = float(np.mean(tgt_psd)), float(np.max(tgt_psd))
            pred_psd_mean, pred_psd_max = float(np.mean(pred_psd)), float(np.max(pred_psd))

            tgt_flat = tgt_psd.flatten()
            pred_flat = pred_psd.flatten()
            spectral_corr = float(np.corrcoef(tgt_flat, pred_flat)[0, 1])

            mse = float(np.mean((main_tgt_field - main_pred_field) ** 2))
            mae = float(np.mean(np.abs(main_tgt_field - main_pred_field)))

            # ====== 统一色标：空间输入 / 空间输出 / 频谱 ======
            # 空间输入
            if len(spatial_inputs) > 0:
                in_vals = np.concatenate([f.ravel() for _, f in spatial_inputs])
                vmin_in = float(in_vals.min())
                vmax_in = float(in_vals.max())
                norm_in = Normalize(vmin=vmin_in, vmax=vmax_in)
            else:
                norm_in = None

            # 空间输出（target + pred）
            if len(spatial_outputs) > 0:
                out_vals = np.concatenate([f.ravel() for _, f in spatial_outputs])
                vmin_out = float(out_vals.min())
                vmax_out = float(out_vals.max())
                norm_out = Normalize(vmin=vmin_out, vmax=vmax_out)
            else:
                norm_out = None

            # PSD 范围（log10）
            all_psd_vals = np.concatenate([p.ravel() for _, p in psd_panels])
            vmin_freq = np.log10(all_psd_vals.min() + 1e-10)
            vmax_freq = np.log10(all_psd_vals.max() + 1e-10)

            # ====== 作图：上空间，下频谱 ======
            n_panel = len(all_spatial_panels)
            cols = n_panel
            rows = 2

            fig, axes = plt.subplots(
                rows, cols,
                figsize=(3.2 * cols, 6.5),
                constrained_layout=True
            )
            axes = np.atleast_2d(axes)

            for idx, (title, field) in enumerate(all_spatial_panels):
                is_input = idx < len(spatial_inputs)

                # ---- 空间域 ----
                ax_sp = axes[0, idx]
                if is_input:
                    im_sp = ax_sp.imshow(
                        field,
                        cmap='viridis',
                        norm=norm_in,
                        aspect='auto'
                    )
                else:
                    im_sp = ax_sp.imshow(
                        field,
                        cmap='viridis',
                        norm=norm_out,
                        aspect='auto'
                    )

                ax_sp.set_title(title, fontsize=9)
                ax_sp.set_xticks([])
                ax_sp.set_yticks([])
                # 每个空间图一个 colorbar
                plt.colorbar(im_sp, ax=ax_sp, fraction=0.046, pad=0.04)

                # ---- PSD ----
                title_psd, psd = psd_panels[idx]
                ax_ps = axes[1, idx]
                im_ps = ax_ps.imshow(
                    np.log10(psd + 1e-10),
                    cmap='viridis',
                    vmin=vmin_freq,
                    vmax=vmax_freq,
                    aspect='auto'
                )
                ax_ps.set_title(f"{title_psd} PSD", fontsize=9)
                ax_ps.set_xticks([])
                ax_ps.set_yticks([])
                # 每个 PSD 图一个 colorbar
                plt.colorbar(im_ps, ax=ax_ps, fraction=0.046, pad=0.04)

            # ====== 保存图像 ======
            png_path = os.path.join(save_dir, f'fourier_analysis_{task}_sample_{i}.png')
            tb_tag = f"{log_prefix}/{task}/sample_{i}/fourier_composed"
            wb_key = f"{log_prefix}/{task}/sample_{i}/fourier_figure"

            _log_figure(
                fig,
                png_path,
                tb_writer=tb_writer,
                tb_tag=tb_tag,
                step=global_step,
                wandb_run=wandb_run,
                wb_key=wb_key,
                dpi=300,
            )
            plt.close(fig)

            # ====== 保存数值结果 ======
            analysis_results = {
                'sample_id': i,
                'task': task,
                'spatial_stats': {
                    'tgt_mean': float(np.mean(main_tgt_field)),
                    'tgt_std': float(np.std(main_tgt_field)),
                    'pred_mean': float(np.mean(main_pred_field)),
                    'pred_std': float(np.std(main_pred_field)),
                },
                'spectral_stats': {
                    'tgt_psd_mean': tgt_psd_mean,
                    'tgt_psd_max': tgt_psd_max,
                    'pred_psd_mean': pred_psd_mean,
                    'pred_psd_max': pred_psd_max,
                },
                'frequency_characteristics': {
                    'tgt_dominant_freq': tgt_domf,
                    'pred_dominant_freq': pred_domf,
                    'tgt_hf_ratio': tgt_hf_ratio,
                    'pred_hf_ratio': pred_hf_ratio,
                },
                'similarity_metrics': {
                    'spectral_correlation': spectral_corr,
                    'mse': mse,
                    'mae': mae,
                },
            }

            np.save(
                os.path.join(save_dir, f'fourier_analysis_{task}_sample_{i}.npy'),
                analysis_results
            )

            txt_path = os.path.join(save_dir, f'fourier_analysis_{task}_sample_{i}.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + f"\n傅里叶频域分析结果 - 任务 {task} - 样本 {i}\n" + "=" * 80 + "\n\n")
                f.write(json.dumps(analysis_results, indent=2, ensure_ascii=False))
                f.write(f"\n完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n")

            print(f"   [{task}] Sample {i}: HF/LF tgt={tgt_hf_ratio:.4f}, "
                  f"pred={pred_hf_ratio:.4f}, domf tgt={tgt_domf:.1f}, "
                  f"pred={pred_domf:.1f}, spec_corr={spectral_corr:.4f}")
            print(f"   Results saved to: {txt_path}")
            print("-" * 80)

        except Exception as e:
            print(f"[Fourier] 样本 {i} 分析失败: {e}")
            continue


# ============================================================
# 5. Encoder 输出特征可视化（空间 + 频谱）
# ============================================================

def visualize_encoded(
    encoded,
    save_dir='./results/encoded_vis',
    max_samples=4,
    channels=None,
    topk=4,
    selection='variance',    # 'variance' | 'l2' | 'random'
    norm_mode='percentile',  # 'percentile' | 'minmax'
    p_low=1.0,
    p_high=99.0,
    tb_writer=None,
    wandb_run=None,
    global_step=None,
    log_prefix='vis/encoded',
    # 频谱设置
    use_window=True,
    r_dc=0.02,
    lf_band=(0.05, 0.30),
    hf_band=(0.40, 0.85),
    unify_freq_clim_across_batch=False,
    log_callback=None,
):
    """
    可视化 Encoder 输出特征并进行空间/频谱统计（稳健版）。
    输入:
        encoded: torch.Tensor [B, C, H, W]
    返回:
        {'meta': {...}, 'summary': {...}, 'per_sample': [ {...}, ... ]}
    """
    os.makedirs(save_dir, exist_ok=True)
    assert encoded.dim() == 4, f"encoded 应为 [B,C,H,W]，但收到 {tuple(encoded.shape)}"
    B, C, H, W = encoded.shape
    n_samples = min(B, max_samples)

    feat = encoded.detach().to('cpu')
    if channels is None:
        k = min(topk, C)
        if selection == 'random':
            perm = torch.randperm(C)
            channels = perm[:k].tolist()
        else:
            flat = feat.reshape(B, C, -1)
            if selection == 'variance':
                score = flat.var(dim=-1, unbiased=False).mean(dim=0)
            elif selection == 'l2':
                score = (flat.pow(2).sum(dim=-1)).mean(dim=0)
            else:
                raise ValueError(f"未知 selection: {selection}")
            top_idx = torch.topk(score, k=k, largest=True).indices
            channels = top_idx.tolist()
    else:
        channels = sorted(set(int(c) for c in channels if 0 <= int(c) < C))
        if len(channels) == 0:
            raise ValueError("channels 为空或越界")

    K = len(channels)

    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = np.sqrt((cy)**2 + (cx)**2)
    r_norm = rr / (rmax + 1e-12)

    def band_mask(r_lo, r_hi):
        return (r_norm >= r_lo) & (r_norm < r_hi)

    mask_valid = r_norm > float(r_dc)
    lf_mask = band_mask(float(lf_band[0]), float(lf_band[1]))
    hf_mask = band_mask(float(hf_band[0]), float(hf_band[1]))

    if use_window:
        wy = np.hanning(H)[:, None]
        wx = np.hanning(W)[None, :]
        win = (wy * wx).astype(np.float64)
    else:
        win = None

    all_log_psd_for_batch = []
    per_sample_summaries = []

    for i in range(n_samples):
        try:
            fmap = feat[i, channels].numpy()
            fig, axes = plt.subplots(2, K, figsize=(4.8 * K, 8), constrained_layout=True)
            if K == 1:
                axes = np.array([[axes[0]], [axes[1]]])

            ch_stats = []
            spatial_mat = np.zeros((K, H * W), dtype=np.float64)
            spectral_mat = np.zeros((K, H * W), dtype=np.float64)

            all_log_psd_list = []

            for j, ch in enumerate(channels):
                arr = fmap[j].astype(np.float64)

                mean_ = float(np.mean(arr))
                std_ = float(np.std(arr))
                min_ = float(np.min(arr))
                max_ = float(np.max(arr))
                l2_ = float(np.sqrt(np.sum(arr**2)) + 1e-12)

                if norm_mode == 'percentile':
                    lo = np.percentile(arr, p_low)
                    hi = np.percentile(arr, p_high)
                    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
                        lo, hi = float(min_), float(max_) if max_ > min_ else (min_, min_ + 1e-6)
                    vis_img = np.clip((arr - lo) / (hi - lo + 1e-12), 0.0, 1.0)
                    vmin, vmax = 0.0, 1.0
                    range_note = f"[{norm_mode}] lo={lo:.3e}, hi={hi:.3e}"
                elif norm_mode == 'minmax':
                    lo, hi = min_, max_
                    if hi <= lo:
                        hi = lo + 1e-6
                    vis_img = (arr - lo) / (hi - lo)
                    vmin, vmax = 0.0, 1.0
                    range_note = f"[{norm_mode}] min={min_:.3e}, max={max_:.3e}"
                else:
                    raise ValueError(f"未知 norm_mode: {norm_mode}")

                arr_in = arr * win if use_window else arr
                fft2c = fftshift(fft2(arr_in))
                psd = (np.abs(fft2c) ** 2).astype(np.float64)
                log_psd = np.log10(psd + 1e-10)
                all_log_psd_list.append(log_psd)

                psd_valid = psd.copy()
                psd_valid[~mask_valid] = -np.inf
                if not np.isfinite(psd_valid).any():
                    psd_valid = psd
                max_idx = np.unravel_index(int(np.nanargmax(psd_valid)), psd.shape)
                dom_r_px = float(rr[max_idx])
                dom_r_hat = float(r_norm[max_idx])

                low_e = float(psd[lf_mask].sum())
                high_e = float(psd[hf_mask].sum())
                hf_ratio = float(high_e / (low_e + 1e-12))
                total_energy_ch = float(psd.sum())

                im_spatial = axes[0, j].imshow(vis_img, cmap='viridis', aspect='auto',
                                               vmin=vmin, vmax=vmax)
                axes[0, j].set_title(
                    f'Encoder Feature | sample {i} | ch {ch}\n'
                    f'{range_note}\nμ={mean_:.3e}, σ={std_:.3e}, L2={l2_:.2e}'
                )
                axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
                plt.colorbar(im_spatial, ax=axes[0, j], fraction=0.046, pad=0.04) \
                    .set_label('Normalized activation')

                im_freq = axes[1, j].imshow(log_psd, cmap='viridis', aspect='auto')
                axes[1, j].set_title(
                    f'Power Spectrum (log10)\n'
                    f'r*={dom_r_hat:.2f}, HF/LF={hf_ratio:.3f}'
                )
                axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
                plt.colorbar(im_freq, ax=axes[1, j], fraction=0.046, pad=0.04) \
                    .set_label('log10(PSD)')

                ch_stats.append({
                    'channel': int(ch),
                    'spatial': {
                        'mean': mean_, 'std': std_, 'min': min_, 'max': max_, 'l2': l2_
                    },
                    'spectral': {
                        'psd_sum': total_energy_ch,
                        'psd_mean': float(np.mean(psd)),
                        'psd_max': float(np.max(psd)),
                        'dominant_radius_px': dom_r_px,
                        'dominant_radius_norm': dom_r_hat,
                        'low_energy': low_e,
                        'high_energy': high_e,
                        'hf_ratio': hf_ratio
                    }
                })

                spatial_mat[j, :] = arr.reshape(-1)
                spectral_mat[j, :] = log_psd.reshape(-1)

                if unify_freq_clim_across_batch:
                    all_log_psd_for_batch.append(log_psd)

            all_log_psd = np.stack(all_log_psd_list, axis=0)
            vmin_freq = float(np.percentile(all_log_psd, 1.0))
            vmax_freq = float(np.percentile(all_log_psd, 99.0))
            if not np.isfinite(vmin_freq):
                vmin_freq = float(np.min(all_log_psd))
            if (not np.isfinite(vmax_freq)) or vmax_freq <= vmin_freq:
                vmax_freq = vmin_freq + 1e-6
            for j in range(K):
                axes[1, j].images[0].set_clim(vmin=vmin_freq, vmax=vmax_freq)

            total_energy_selected = float(sum(cs['spectral']['psd_sum'] for cs in ch_stats)) + 1e-12
            for cs in ch_stats:
                cs['spectral']['energy_ratio_selected'] = float(cs['spectral']['psd_sum'] / total_energy_selected)

            def safe_corrcoef(mat):
                try:
                    C = np.corrcoef(mat)
                    if np.isnan(C).any():
                        C = np.nan_to_num(C, nan=0.0, posinf=0.0, negative_inf=0.0)
                except Exception:
                    C = np.zeros((mat.shape[0], mat.shape[0]), dtype=np.float64)
                return C

            spatial_corr = safe_corrcoef(spatial_mat)
            spectral_corr = safe_corrcoef(spectral_mat)

            png_path = os.path.join(save_dir, f'encoded_sample_{i}_ch_{"-".join(map(str,channels))}.png')
            if callable(log_callback):
                log_callback(
                    fig,
                    png_path,
                    tb_writer=tb_writer,
                    tb_tag=f"{log_prefix}/sample_{i}/encoded_composed",
                    step=global_step,
                    wandb_run=wandb_run,
                    wb_key=f"{log_prefix}/sample_{i}/encoded_figure",
                    dpi=300,
                )
            else:
                _log_figure(
                    fig,
                    png_path,
                    tb_writer=tb_writer,
                    tb_tag=f"{log_prefix}/sample_{i}/encoded_composed",
                    step=global_step,
                    wandb_run=wandb_run,
                    wb_key=f"{log_prefix}/sample_{i}/encoded_figure",
                    dpi=300,
                )
            plt.close(fig)

            analysis = {
                'sample_id': int(i),
                'shape': {'B': B, 'C_total': C, 'H': H, 'W': W},
                'selected_channels': [int(c) for c in channels],
                'norm_mode': norm_mode,
                'selection': selection,
                'stats_per_channel': ch_stats,
                'inter_channel': {
                    'spatial_corr': spatial_corr.tolist(),
                    'spectral_corr': spectral_corr.tolist()
                }
            }
            np.save(os.path.join(save_dir, f'encoded_analysis_sample_{i}.npy'), analysis)

            txt_path = os.path.join(save_dir, f'encoded_analysis_sample_{i}.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 90 + "\n")
                f.write(f"Encoder 特征可视化与频域分析 - 样本 {i}\n")
                f.write("=" * 90 + "\n\n")
                f.write(f"通道选择: {channels}  (mode={selection}, topk={K})\n")
                f.write(f"可视化归一化: {norm_mode} (p_low={p_low}, p_high={p_high})\n")
                f.write(f"频谱设置: use_window={use_window}, r_dc={r_dc}, "
                        f"lf_band={lf_band}, hf_band={hf_band}\n\n")

                for cs in ch_stats:
                    ch = cs['channel']
                    s = cs['spatial']
                    sp = cs['spectral']
                    f.write(f"[Channel {ch}]\n")
                    f.write(f"  空间: mean={s['mean']:.6e}, std={s['std']:.6e}, "
                            f"min={s['min']:.6e}, max={s['max']:.6e}, L2={s['l2']:.6e}\n")
                    f.write(f"  频谱: psd_mean={sp['psd_mean']:.6e}, psd_max={sp['psd_max']:.6e}, "
                            f"r*={sp['dominant_radius_norm']:.3f}, HF/LF={sp['hf_ratio']:.6f}, "
                            f"energy_ratio_selected={sp['energy_ratio_selected']:.6f}\n\n")

                f.write("空间域通道间相关矩阵（K×K）:\n")
                f.write(np.array2string(spatial_corr, formatter={'float_kind': lambda x: f"{x: .3f}"}))
                f.write("\n\n频谱域通道间相关矩阵（K×K，基于 log10(PSD)）:\n")
                f.write(np.array2string(spectral_corr, formatter={'float_kind': lambda x: f"{x: .3f}"}))
                f.write("\n\n")

                f.write("=" * 90 + "\n")
                f.write(f"保存图像: {png_path}\n")
                f.write(f"保存 NPY:  {os.path.join(save_dir, f'encoded_analysis_sample_{i}.npy')}\n")
                f.write(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 90 + "\n")

            print(f"[visualize_encoded] Sample {i} done.")
            print(f"  Selected channels: {channels}")
            for cs in ch_stats:
                ch = cs['channel']; sp = cs['spectral']; s = cs['spatial']
                print(f"    ch{ch:>3} | μ={s['mean']:.3e}, σ={s['std']:.3e}, "
                      f"L2={s['l2']:.2e} | r*={sp['dominant_radius_norm']:.2f}, "
                      f"HF/LF={sp['hf_ratio']:.3f}, E%={sp['energy_ratio_selected']:.3f}")
            print(f"  Spatial Corr (KxK): min={float(spatial_corr.min()):.3f}, max={float(spatial_corr.max()):.3f}")
            print(f"  Spectral Corr (KxK): min={float(spectral_corr.min()):.3f}, max={float(spectral_corr.max()):.3f}")
            print(f"  Saved: {png_path} | {txt_path}")
            print("-" * 90)

            per_sample_summaries.append({
                'sample_id': int(i),
                'selected_channels': [int(c) for c in channels],
                'spatial_corr_min': float(spatial_corr.min()),
                'spatial_corr_max': float(spatial_corr.max()),
                'spectral_corr_min': float(spectral_corr.min()),
                'spectral_corr_max': float(spectral_corr.max()),
            })

        except Exception as e:
            print(f"[visualize_encoded] 样本 {i} 可视化失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    meta = {
        'shape': {'B': B, 'C': C, 'H': H, 'W': W},
        'max_samples': n_samples,
        'selected_channels': [int(c) for c in channels],
        'selection': selection,
        'norm_mode': norm_mode,
        'p_low': p_low,
        'p_high': p_high,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'freq_settings': {
            'use_window': bool(use_window),
            'r_dc': float(r_dc),
            'lf_band': [float(lf_band[0]), float(lf_band[1])],
            'hf_band': [float(hf_band[0]), float(hf_band[1])],
        }
    }
    with open(os.path.join(save_dir, 'encoded_vis_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if unify_freq_clim_across_batch and len(all_log_psd_for_batch) > 0:
        all_log_psd_for_batch = np.stack(all_log_psd_for_batch, axis=0)
        vmin_b = float(np.percentile(all_log_psd_for_batch, 1.0))
        vmax_b = float(np.percentile(all_log_psd_for_batch, 99.0))
        meta['batch_freq_clim'] = {'vmin': vmin_b, 'vmax': vmax_b}

    print(f"[visualize_encoded] Done. samples={n_samples}, channels={channels}, savedir='{save_dir}'")

    summary = {
        'num_samples': n_samples,
        'channels': channels,
        'selection': selection,
        'norm_mode': norm_mode,
        'hf_lf_settings': {
            'r_dc': float(r_dc),
            'lf': [float(lf_band[0]), float(lf_band[1])],
            'hf': [float(hf_band[0]), float(hf_band[1])],
            'use_window': bool(use_window)
        }
    }
    return {'meta': meta, 'summary': summary, 'per_sample': per_sample_summaries}


# ============================================================
# 6. Router 频段选择 / Top-k 专家频率 可视化
# ============================================================

def visualize_router_selection(
    band_centers: np.ndarray,
    band_select_counts: np.ndarray,
    expert_select_counts: np.ndarray,
    save_path: str,
    router_name: str = "router",
    tb_writer=None,
    wandb_run=None,
    global_step=None,
    log_prefix: str = "router",
):
    """
    可视化 router 对频段的选择分布 & Top-k 专家被选中的次数。
    """
    band_centers = np.asarray(band_centers, dtype=float)
    band_select_counts = np.asarray(band_select_counts, dtype=float)
    expert_select_counts = np.asarray(expert_select_counts, dtype=float)

    num_bands = band_centers.shape[0]
    num_experts = expert_select_counts.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # 频段直方图
    axes[0].bar(np.arange(num_bands), band_select_counts)
    axes[0].set_xticks(np.arange(num_bands))
    axes[0].set_xticklabels([f"{c:.2f}" for c in band_centers], rotation=45)
    axes[0].set_xlabel("Normalized band center r*")
    axes[0].set_ylabel("Selections")
    axes[0].set_title(f"{router_name}: band selection histogram")

    # 专家选择次数
    axes[1].bar(np.arange(num_experts), expert_select_counts)
    axes[1].set_xticks(np.arange(num_experts))
    axes[1].set_xlabel("Expert index")
    axes[1].set_ylabel("Selections")
    axes[1].set_title(f"{router_name}: top-k expert selection count")

    tb_tag = f"{log_prefix}/selection_hist"
    wb_key = f"{log_prefix}/selection_hist"

    _log_figure(
        fig,
        save_path,
        tb_writer=tb_writer,
        tb_tag=tb_tag,
        step=global_step,
        wandb_run=wandb_run,
        wb_key=wb_key,
        dpi=300,
    )
    plt.close(fig)
    print(f"[RouterVis] selection hist saved to {save_path}")


def visualize_router_selection_from_stats(
    stats: Dict[str, Any],
    save_dir: Union[str, Path],
    epoch: Optional[int] = None,
    router_name: str = "router",
    tb_writer=None,
    wandb_run=None,
    global_step=None,
    log_prefix: str = "router",
):
    """
    兼容 router.get_stats() 的版本，从 stats 字典中抽取字段并调用 visualize_router_selection。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    band_centers = np.asarray(stats.get("band_centers", []), dtype=float)
    band_counts = np.asarray(stats.get("band_select_counts", []), dtype=float)
    expert_counts = np.asarray(stats.get("expert_select_counts", []), dtype=float)

    if band_centers.size == 0 or band_counts.size == 0 or expert_counts.size == 0:
        print("[RouterVis] stats 中缺少 band_centers/band_select_counts/expert_select_counts，跳过可视化。")
        return

    if band_centers.shape[0] != band_counts.shape[0]:
        print(f"[RouterVis] band_centers 与 band_select_counts 长度不一致: "
              f"{band_centers.shape[0]} vs {band_counts.shape[0]}，强制裁剪对齐。")
        n = min(band_centers.shape[0], band_counts.shape[0])
        band_centers = band_centers[:n]
        band_counts = band_counts[:n]

    if epoch is None:
        fname = "router_selection_hist.png"
    else:
        fname = f"router_selection_hist_epoch_{epoch+1}.png"

    save_path = save_dir / fname

    visualize_router_selection(
        band_centers,
        band_counts,
        expert_counts,
        save_path=str(save_path),
        router_name=router_name,
        tb_writer=tb_writer,
        wandb_run=wandb_run,
        global_step=global_step,
        log_prefix=log_prefix,
    )


# ============================================================
# 7. 可视化 routed_bands：每个专家实际吃到的特征
# ============================================================

def visualize_routed_bands(
    routed_bands: List[torch.Tensor],
    save_dir: Union[str, Path],
    sample_idx: int = 0,
    max_channels: int = 4,
    prefix: str = "routed_bands",
    band_centers: Optional[np.ndarray] = None,
    tb_writer=None,
    wandb_run=None,
    global_step=None,
    log_prefix: str = "router/routed",
    # 通道选择策略（参考 visualize_encoded）
    channels: Optional[List[int]] = None,
    topk: int = 4,
    selection: str = "variance",     # 'variance' | 'l2' | 'random'
    norm_mode: str = "percentile",   # 'percentile' | 'minmax'
    p_low: float = 1.0,
    p_high: float = 99.0,
):
    """
    可视化分频后的特征 routed_bands（空间域 + 频域）：
      - routed_bands: list of [B, C, H, W]，长度 = num_experts
      - 对于给定 sample_idx，从每个 expert 的输入中截取 [C, H, W]
      - 当 C 很大时，参考 visualize_encoded 的方式选择若干通道:
          * channels 显式指定
          * 否则根据 selection/topk 从该 expert 的所有通道中选出子集

      每个 expert 的图：
          * 上一行：空间域特征图（选出的 K 个通道）
          * 下一行：对应的频域 log10(PSD)
    """
    if not routed_bands:
        print("[RoutedBandsVis] routed_bands 为空，无法可视化。")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_experts = len(routed_bands)
    if band_centers is not None:
        band_centers = np.asarray(band_centers, dtype=float)

    for e_idx in range(num_experts):
        feat = routed_bands[e_idx]
        if not isinstance(feat, torch.Tensor):
            print(f"[RoutedBandsVis] routed_bands[{e_idx}] 不是 Tensor，实际类型 {type(feat)}，跳过。")
            continue
        if feat.ndim != 4:
            print(f"[RoutedBandsVis] routed_bands[{e_idx}] 维度不是 4D，实际为 {feat.shape}，跳过。")
            continue
        if sample_idx >= feat.size(0):
            print(f"[RoutedBandsVis] sample_idx={sample_idx} 超出 batch 大小 {feat.size(0)}，跳过该 expert。")
            continue

        # feat: [B, C, H, W]
        B, C, H, W = feat.shape

        # ===== 通道选择：参考 visualize_encoded =====
        if channels is None:
            k = min(topk, C, max_channels)
            if selection == "random":
                perm = torch.randperm(C)
                ch_idx = perm[:k].tolist()
            else:
                flat = feat.detach().reshape(B, C, -1).to("cpu")
                if selection == "variance":
                    score = flat.var(dim=-1, unbiased=False).mean(dim=0)   # [C]
                elif selection == "l2":
                    score = (flat.pow(2).sum(dim=-1)).mean(dim=0)          # [C]
                else:
                    raise ValueError(f"[RoutedBandsVis] 未知 selection: {selection}")
                top_idx = torch.topk(score, k=k, largest=True).indices
                ch_idx = top_idx.tolist()
        else:
            # 显式给定通道时：去重 + 边界裁剪
            ch_idx = sorted(set(int(c) for c in channels if 0 <= int(c) < C))
            if len(ch_idx) == 0:
                print(f"[RoutedBandsVis] 指定 channels 全部越界/为空，expert {e_idx} 跳过。")
                continue
            if len(ch_idx) > max_channels:
                ch_idx = ch_idx[:max_channels]

        K = len(ch_idx)
        if K == 0:
            print(f"[RoutedBandsVis] expert {e_idx} 没有可视化通道，跳过。")
            continue

        # 取指定样本： [C, H, W]
        feat_e = feat[sample_idx].detach().cpu()   # [C, H, W]

        # 两行：上 = 空间域；下 = 频域
        fig, axes = plt.subplots(
            2,
            K,
            figsize=(3.2 * K, 6.0),
            constrained_layout=True
        )

        # 兼容 K=1 的情况
        if K == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        # 收集所有 log10(PSD)，方便统一频谱色标
        all_log_psd_list: List[np.ndarray] = []

        for col_idx, ch in enumerate(ch_idx):
            img = feat_e[ch].numpy().astype(np.float64)

            # ===== 空间域归一化：参考 visualize_encoded =====
            mean_ = float(np.mean(img))
            std_ = float(np.std(img))
            min_ = float(np.min(img))
            max_ = float(np.max(img))

            if norm_mode == "percentile":
                lo = np.percentile(img, p_low)
                hi = np.percentile(img, p_high)
                if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
                    lo, hi = float(min_), float(max_) if max_ > min_ else (min_, min_ + 1e-6)
                vis_img = np.clip((img - lo) / (hi - lo + 1e-12), 0.0, 1.0)
                vmin, vmax = 0.0, 1.0
                range_note = f"[{norm_mode}] lo={lo:.3e}, hi={hi:.3e}"
            elif norm_mode == "minmax":
                lo, hi = min_, max_
                if hi <= lo:
                    hi = lo + 1e-6
                vis_img = (img - lo) / (hi - lo)
                vmin, vmax = 0.0, 1.0
                range_note = f"[{norm_mode}] min={min_:.3e}, max={max_:.3e}"
            else:
                raise ValueError(f"[RoutedBandsVis] 未知 norm_mode: {norm_mode}")

            # ---- 空间域图 ----
            ax_sp = axes[0, col_idx]
            im_sp = ax_sp.imshow(vis_img, origin="lower", aspect="auto",
                                 cmap="viridis", vmin=vmin, vmax=vmax)
            ax_sp.set_title(
                f"E{e_idx} sample {sample_idx} | ch={ch}\n"
                f"{range_note}\nμ={mean_:.3e}, σ={std_:.3e}",
                fontsize=9
            )
            ax_sp.axis("off")
            plt.colorbar(im_sp, ax=ax_sp, fraction=0.046, pad=0.04).set_label("Normalized act.")

            # ---- 频域：log10 Power Spectrum ----
            F = fftshift(fft2(img))
            psd = np.abs(F) ** 2
            log_psd = np.log10(psd + 1e-10)
            all_log_psd_list.append(log_psd)

            ax_fd = axes[1, col_idx]
            im_fd = ax_fd.imshow(log_psd, origin="lower", aspect="auto", cmap="viridis")
            ax_fd.set_title(f"ch={ch} | log10(PSD)", fontsize=9)
            ax_fd.axis("off")
            plt.colorbar(im_fd, ax=ax_fd, fraction=0.046, pad=0.04).set_label("log10(PSD)")

        # 统一当前 expert 内所有频谱的色标，使对比更直观
        if len(all_log_psd_list) > 0:
            all_log_psd = np.stack(all_log_psd_list, axis=0)
            vmin_freq = float(np.percentile(all_log_psd, 1.0))
            vmax_freq = float(np.percentile(all_log_psd, 99.0))
            if not np.isfinite(vmin_freq):
                vmin_freq = float(np.min(all_log_psd))
            if (not np.isfinite(vmax_freq)) or vmax_freq <= vmin_freq:
                vmax_freq = vmin_freq + 1e-6

            for col_idx in range(K):
                img_obj = axes[1, col_idx].images[0]
                img_obj.set_clim(vmin=vmin_freq, vmax=vmax_freq)

        # 图标题和文件名中带上 band center（如果有）
        if band_centers is not None and band_centers.shape[0] == num_experts:
            r_c = float(band_centers[e_idx])
            fig.suptitle(
                f"Expert E{e_idx} | band center r≈{r_c:.2f} | channels={ch_idx}",
                fontsize=10
            )
            fname = f"{prefix}_E{e_idx}_r{r_c:.2f}.png"
        else:
            fig.suptitle(f"Expert E{e_idx} | channels={ch_idx}", fontsize=10)
            fname = f"{prefix}_E{e_idx}.png"

        out_path = save_dir / fname

        tb_tag = f"{log_prefix}/E{e_idx}/sample_{sample_idx}"
        wb_key = f"{log_prefix}/E{e_idx}/sample_{sample_idx}"

        _log_figure(
            fig,
            str(out_path),
            tb_writer=tb_writer,
            tb_tag=tb_tag,
            step=global_step,
            wandb_run=wandb_run,
            wb_key=wb_key,
            dpi=300,
        )
        plt.close(fig)

        print(f"[RoutedBandsVis] 已保存专家 E{e_idx} 空间+频域特征图到 {out_path} | channels={ch_idx}")


# ============================================================
# 8. 误差热力图
# ============================================================

def visualize_error_heatmap(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    save_dir: str = "./results/error_maps",
    max_samples: int = 4,
    mode: str = "abs",          # "abs" | "signed" | "sq" | "rel"
    eps: float = 1e-8,
    tb_writer=None,
    wandb_run=None,
    global_step=None,
    log_prefix: str = "vis/error",
):
    """
    绘制预测误差的 heatmap，并同时展示 target / pred / error 三张图。
    """
    os.makedirs(save_dir, exist_ok=True)

    assert targets.shape[0] == predictions.shape[0], \
        f"targets and predictions batch size mismatch: {targets.shape[0]} vs {predictions.shape[0]}"
    B = targets.shape[0]
    n_samples = min(B, max_samples)

    for i in range(n_samples):
        try:
            tgt = targets[i].detach().cpu().numpy().squeeze()
            pred = predictions[i].detach().cpu().numpy().squeeze()

            if tgt.ndim != 2 or pred.ndim != 2:
                print(f"[ErrorHeatmap] sample {i}: 非 2D 数据，跳过（tgt.ndim={tgt.ndim}, pred.ndim={pred.ndim}）")
                continue

            diff = pred - tgt
            if mode == "abs":
                err = np.abs(diff)
                err_title = "|pred - target|"
                cmap = "viridis"
                vmin, vmax = float(err.min()), float(err.max())
            elif mode == "signed":
                err = diff
                err_title = "pred - target"
                cmap = "seismic"
                vmax = float(np.max(np.abs(err)))
                vmin = -vmax
            elif mode == "sq":
                err = diff ** 2
                err_title = "(pred - target)^2"
                cmap = "viridis"
                vmin, vmax = float(err.min()), float(err.max())
            elif mode == "rel":
                err = np.abs(diff) / (np.abs(tgt) + eps)
                err_title = "|pred - target| / (|target| + eps)"
                cmap = "viridis"
                vmin, vmax = float(err.min()), float(err.max())
            else:
                raise ValueError(f"Unknown error mode: {mode}")

            vmin_tp = min(float(tgt.min()), float(pred.min()))
            vmax_tp = max(float(tgt.max()), float(pred.max()))

            mae = float(np.mean(np.abs(diff)))
            mse = float(np.mean(diff ** 2))
            rmse = float(np.sqrt(mse + 1e-12))

            fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

            im0 = axes[0].imshow(tgt, cmap="jet", vmin=vmin_tp, vmax=vmax_tp, aspect="auto")
            axes[0].set_title(f"Target\nMAE={mae:.3e}, RMSE={rmse:.3e}")
            axes[0].set_xticks([]); axes[0].set_yticks([])
            cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            cbar0.set_label("Target value")

            im1 = axes[1].imshow(pred, cmap="jet", vmin=vmin_tp, vmax=vmax_tp, aspect="auto")
            axes[1].set_title("Prediction")
            axes[1].set_xticks([]); axes[1].set_yticks([])
            cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            cbar1.set_label("Prediction value")

            im2 = axes[2].imshow(err, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            axes[2].set_title(f"Error Heatmap ({mode})\n{err_title}")
            axes[2].set_xticks([]); axes[2].set_yticks([])
            cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            cbar2.set_label("Error")

            png_path = os.path.join(save_dir, f"error_sample_{i}_mode_{mode}.png")
            tb_tag = f"{log_prefix}/sample_{i}/error_{mode}"
            wb_key = f"{log_prefix}/sample_{i}/error_{mode}"

            _log_figure(
                fig,
                png_path,
                tb_writer=tb_writer,
                tb_tag=tb_tag,
                step=global_step,
                wandb_run=wandb_run,
                wb_key=wb_key,
                dpi=300,
            )
            plt.close(fig)

            stats = {
                "sample_id": int(i),
                "mode": mode,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "target_min": float(tgt.min()),
                "target_max": float(tgt.max()),
                "pred_min": float(pred.min()),
                "pred_max": float(pred.max()),
                "error_min": float(err.min()),
                "error_max": float(err.max()),
            }
            txt_path = os.path.join(save_dir, f"error_sample_{i}_mode_{mode}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"Error Heatmap Stats - sample {i}\n")
                f.write("=" * 80 + "\n\n")
                f.write(json.dumps(stats, indent=2, ensure_ascii=False))
                f.write("\n" + "=" * 80 + "\n")

            print(f"[ErrorHeatmap] sample {i} saved: {png_path}")

        except Exception as e:
            print(f"[ErrorHeatmap] sample {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue


# ============================================================
# 9. pde 风格的输出可视化（pred / gt / err / input）
# ============================================================

def visualize_pde_style(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    save_dir: str = "./results/pde_vis",
    sample_indices: Optional[List[int]] = None,
    grid_shape: Optional[tuple[int, int]] = None,
    cmap: str = "coolwarm",
    task: Optional[str] = None,   # "darcy" | "pipe" | "airfoil" | "navier" | "plasticity" | None
    tb_writer=None,
    wandb_run=None,
    global_step=None,
    log_prefix: str = "vis/lamo",
):
    """
    参考 LaMO 官方脚本的可视化方式（darcy/pipe/airfoil 等）：
      - 上/左：pred
      - 上/右：gt
      - 下/左：abs error
      - 下/右：input（若可绘制，否则留空）

    适用场景：
      * targets / predictions 为 2D 场或扁平向量（自动 sqrt 还原）。
      * inputs 若为网格场，会显示第一通道；否则跳过输入面板。
    """
    os.makedirs(save_dir, exist_ok=True)

    B = targets.shape[0]
    if sample_indices is None:
        sample_indices = list(range(min(4, B)))

    def _to_field(x: torch.Tensor) -> Optional[np.ndarray]:
        """将 x（[H,W] / [C,H,W] / [N]）转为 2D ndarray。"""
        x = x.detach().cpu().squeeze()
        if x.ndim == 2:
            return x.numpy()
        if x.ndim == 3:
            return x[0].numpy()
        if x.ndim == 1:
            n = x.numel()
            if grid_shape is not None:
                h, w = grid_shape
                if h * w == n:
                    return x.reshape(h, w).numpy()
            s = int(np.sqrt(n))
            if s * s == n:
                return x.reshape(s, s).numpy()
        return None

    for idx in sample_indices:
        if idx >= B:
            continue
        # ==========================
        # LaMO 原生分支：逐任务复刻
        # ==========================
        if task in {"darcy", "pipe", "airfoil", "navier", "ns", "plasticity"}:
            # --------- darcy (85x85) ---------
            if task == "darcy":
                tgt_f = _to_field(targets[idx]) or _to_field(targets[idx].reshape(85, 85))
                pred_f = _to_field(predictions[idx]) or _to_field(predictions[idx].reshape(85, 85))
                inp_f = _to_field(inputs[idx]) if inputs is not None else None
                if tgt_f is None or pred_f is None:
                    print(f"[LaMOVis] sample {idx}: darcy reshape 85x85 失败，跳过。")
                    continue
                err_f = pred_f - tgt_f
                vmin = float(min(tgt_f.min(), pred_f.min()))
                vmax = float(max(tgt_f.max(), pred_f.max()))

                def _save_single(arr, name, clim=None, cmap_local=cmap):
                    plt.figure()
                    plt.axis("off")
                    im = plt.imshow(arr, cmap=cmap_local, vmin=(clim[0] if clim else None), vmax=(clim[1] if clim else None))
                    plt.colorbar()
                    fname = os.path.join(save_dir, f"{name}_{idx}.pdf")
                    plt.savefig(fname, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    return fname

                p_path = _save_single(pred_f, "pred", clim=(vmin, vmax))
                g_path = _save_single(tgt_f, "gt", clim=(vmin, vmax))
                e_path = _save_single(err_f, "error", clim=(-5e-4, 5e-4))
                if inp_f is not None:
                    _save_single(inp_f, "input", cmap_local="coolwarm")
                print(f"[LaMOVis][darcy] sample {idx} saved: {p_path}, {g_path}, {e_path}")
                continue

            # --------- pipe (129x129) ---------
            if task == "pipe":
                # inputs: [B, 2, H, W]; preds/targets: [B, 1, H, W] -> 129x129
                x0 = inputs[idx, 0].detach().cpu().numpy().reshape(129, 129)
                x1 = inputs[idx, 1].detach().cpu().numpy().reshape(129, 129)
                tgt_f = targets[idx].detach().cpu().numpy().reshape(129, 129)
                pred_f = predictions[idx].detach().cpu().numpy().reshape(129, 129)
                err_f = pred_f - tgt_f

                def _save_mesh(z, name, clim=None):
                    plt.axis("off")
                    plt.pcolormesh(x0, x1, z, shading="auto", cmap="coolwarm")
                    plt.colorbar()
                    if clim:
                        plt.clim(*clim)
                    fname = os.path.join(save_dir, f"{name}_{idx}.pdf")
                    plt.savefig(fname, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    return fname

                _save_mesh(np.zeros_like(tgt_f), "input")
                p_path = _save_mesh(pred_f, "pred", clim=(0, 0.3))
                g_path = _save_mesh(tgt_f, "gt", clim=(0, 0.3))
                e_path = _save_mesh(err_f, "error", clim=(-0.02, 0.02))
                print(f"[LaMOVis][pipe] sample {idx} saved: {p_path}, {g_path}, {e_path}")
                continue

            # --------- airfoil (221x51 裁剪 140×35) ---------
            if task == "airfoil":
                x0 = inputs[idx, 0].detach().cpu().numpy().reshape(221, 51)[40:180, :35]
                x1 = inputs[idx, 1].detach().cpu().numpy().reshape(221, 51)[40:180, :35]
                tgt_f = targets[idx].detach().cpu().numpy().reshape(221, 51)[40:180, :35]
                pred_f = predictions[idx].detach().cpu().numpy().reshape(221, 51)[40:180, :35]
                err_f = pred_f - tgt_f

                def _save_mesh(z, name, clim=None):
                    plt.axis("off")
                    plt.pcolormesh(x0, x1, z, shading="auto", cmap="coolwarm")
                    plt.colorbar()
                    if clim:
                        plt.clim(*clim)
                    fname = os.path.join(save_dir, f"{name}_{idx}.pdf")
                    plt.savefig(fname, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    return fname

                _save_mesh(np.zeros_like(tgt_f), "input")
                p_path = _save_mesh(pred_f, "pred", clim=(0, 1.2))
                g_path = _save_mesh(tgt_f, "gt", clim=(0, 1.2))
                e_path = _save_mesh(err_f, "error", clim=(-0.2, 0.2))
                print(f"[LaMOVis][airfoil] sample {idx} saved: {p_path}, {g_path}, {e_path}")
                continue

            # --------- navier-stokes (64x64, 单帧展示) ---------
            if task in {"navier", "ns"}:
                # 预测通常是 rollout 后的单帧；若有时间维，取最后一帧
                pred_t = predictions[idx].detach().cpu().squeeze()
                tgt_t = targets[idx].detach().cpu().squeeze()
                if pred_t.ndim == 2:
                    pred_f = pred_t.numpy()
                elif pred_t.ndim == 3:
                    pred_f = pred_t[-1].numpy()
                else:
                    print(f"[LaMOVis][ns] sample {idx}: 形状不支持 {pred_t.shape}")
                    continue
                if tgt_t.ndim == 2:
                    tgt_f = tgt_t.numpy()
                elif tgt_t.ndim == 3:
                    tgt_f = tgt_t[-1].numpy()
                else:
                    print(f"[LaMOVis][ns] sample {idx}: 形状不支持 {tgt_t.shape}")
                    continue
                err_f = pred_f - tgt_f

                def _save_im(z, name, clim=None):
                    plt.figure()
                    plt.axis("off")
                    plt.imshow(z, cmap="coolwarm")
                    if clim:
                        plt.clim(*clim)
                    plt.colorbar()
                    fname = os.path.join(save_dir, f"{name}_{idx}.pdf")
                    plt.savefig(fname, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    return fname

                p_path = _save_im(pred_f, "pred", clim=(-3, 3))
                g_path = _save_im(tgt_f, "gt", clim=(-3, 3))
                e_path = _save_im(err_f, "error", clim=(-2, 2))
                print(f"[LaMOVis][ns] sample {idx} saved: {p_path}, {g_path}, {e_path}")
                continue

            # --------- plasticity (scatter) ---------
            if task == "plasticity":
                tgt_np = targets[idx].detach().cpu().numpy()  # [C,H,W], C=4*T
                pred_np = predictions[idx].detach().cpu().numpy()
                c, h, w = tgt_np.shape
                if c % 4 != 0:
                    print(f"[LaMOVis][plasticity] sample {idx}: channels {c} not divisible by 4, skip.")
                    continue
                T = c // 4
                tgt_last = tgt_np.reshape(4, T, h, w)[:, -1]   # [4,H,W]
                pred_last = pred_np.reshape(4, T, h, w)[:, -1]
                # 均匀网格坐标
                xs = np.linspace(0, 1, h)
                ys = np.linspace(0, 1, w)
                xx, yy = np.meshgrid(xs, ys, indexing="ij")
                # 使用所有 4 个分量的范数
                truth_du = np.linalg.norm(tgt_last, axis=0)
                pred_du = np.linalg.norm(pred_last, axis=0)

                def _save_scatter(vals, name, clim=None):
                    plt.axis("off")
                    plt.scatter(xx, yy, 10, vals, cmap="coolwarm")
                    plt.colorbar()
                    if clim:
                        plt.clim(*clim)
                    fname = os.path.join(save_dir, f"{name}_{idx}.pdf")
                    plt.savefig(fname, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    return fname

                g_path = _save_scatter(truth_du, "gt", clim=(0, 6))
                p_path = _save_scatter(pred_du, "pred", clim=(0, 6))
                e_path = _save_scatter(pred_du - truth_du, "error", clim=(-0.2, 0.2))
                print(f"[LaMOVis][plasticity] sample {idx} saved: {p_path}, {g_path}, {e_path}")
                continue

            # --------- elasticity (scatter) ---------
            if task == "elasticity":
                fx = inputs[idx].detach().cpu().numpy()
                tgt = targets[idx].detach().cpu().numpy()
                pred = predictions[idx].detach().cpu().numpy()
                tgt = tgt.reshape(-1)
                pred = pred.reshape(-1)

                def _save_scatter(vals, name, clim=None):
                    plt.axis("off")
                    plt.scatter(x=fx[:, 0], y=fx[:, 1], c=vals, s=10, cmap="coolwarm")
                    plt.colorbar()
                    if clim:
                        plt.clim(*clim)
                    fname = os.path.join(save_dir, f"{name}_{idx}.pdf")
                    plt.savefig(fname, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    return fname

                g_path = _save_scatter(tgt, "gt", clim=(0, 1000))
                p_path = _save_scatter(pred, "pred", clim=(0, 1000))
                e_path = _save_scatter(pred - tgt, "error", clim=(-8, 8))
                print(f"[LaMOVis][elasticity] sample {idx} saved: {p_path}, {g_path}, {e_path}")
                continue

        # ==========================
        # 默认通用 2×2 布局
        # ==========================
        tgt_f = _to_field(targets[idx])
        pred_f = _to_field(predictions[idx])
        inp_f = _to_field(inputs[idx]) if inputs is not None else None
        if tgt_f is None or pred_f is None:
            print(f"[LaMOVis] sample {idx}: 无法还原为 2D 场，跳过。")
            continue

        err_f = np.abs(pred_f - tgt_f)
        vmin = float(min(tgt_f.min(), pred_f.min()))
        vmax = float(max(tgt_f.max(), pred_f.max()))

        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        ax_pred, ax_tgt, ax_err, ax_inp = axes.flat

        im_p = ax_pred.imshow(pred_f, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax_pred.set_title("Prediction")
        ax_pred.axis("off")
        plt.colorbar(im_p, ax=ax_pred, fraction=0.046, pad=0.04)

        im_t = ax_tgt.imshow(tgt_f, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax_tgt.set_title("Ground Truth")
        ax_tgt.axis("off")
        plt.colorbar(im_t, ax=ax_tgt, fraction=0.046, pad=0.04)

        im_e = ax_err.imshow(err_f, cmap="magma", aspect="auto")
        ax_err.set_title("|Pred - GT|")
        ax_err.axis("off")
        plt.colorbar(im_e, ax=ax_err, fraction=0.046, pad=0.04)

        if inp_f is not None:
            im_i = ax_inp.imshow(inp_f, cmap="viridis", aspect="auto")
            ax_inp.set_title("Input (ch0)")
            ax_inp.axis("off")
            plt.colorbar(im_i, ax=ax_inp, fraction=0.046, pad=0.04)
        else:
            ax_inp.axis("off")
            ax_inp.set_title("No input to show")

        fname = os.path.join(save_dir, f"pde_vis_sample_{idx}.png")
        tb_tag = f"{log_prefix}/sample_{idx}"
        wb_key = f"{log_prefix}/sample_{idx}"

        _log_figure(
            fig,
            fname,
            tb_writer=tb_writer,
            tb_tag=tb_tag,
            step=global_step,
            wandb_run=wandb_run,
            wb_key=wb_key,
            dpi=300,
        )
        plt.close(fig)

        print(f"[LaMOVis] sample {idx} saved to {fname}")

# ============================================================
# 10. 专家频率偏好可视化（expert_freq + band_centers）
# ============================================================

def visualize_expert_freq_preference(
    expert_freq: np.ndarray,
    band_centers: np.ndarray,
    save_path: str,
    freq_affinity_sharpness: float = 10.0,
    router_name: str = "router",
    tb_writer=None,
    wandb_run=None,
    global_step=None,
    log_prefix: str = "router/freq_pref",
):
    """
    可视化“专家频率偏好”：
      - expert_freq:   [E]，每个专家的 meta 频率标量 ∈[0,1]
      - band_centers:  [B]，每个频段的中心半径 ∈[0,1]
    输出一张图，包含：
      1) 左：expert_freq 与 band_centers 在 [0,1] 上的散点分布
      2) 右：expert × band 的 soft 权重矩阵（高斯兼容度 + softmax）
    """
    expert_freq = np.asarray(expert_freq, dtype=float).reshape(-1)
    band_centers = np.asarray(band_centers, dtype=float).reshape(-1)

    E = expert_freq.shape[0]
    B = band_centers.shape[0]

    if E == 0 or B == 0:
        print("[FreqPrefVis] expert_freq 或 band_centers 为空，跳过可视化。")
        return

    # ===== 计算 expert–band 的 soft 权重（高斯兼容度 + softmax）=====
    # diff[e,b] = expert_freq[e] - band_centers[b]
    diff = expert_freq[:, None] - band_centers[None, :]          # [E,B]
    compat = -freq_affinity_sharpness * (diff ** 2)              # [E,B]

    # 手动 softmax，避免额外依赖
    compat_shift = compat - compat.max(axis=1, keepdims=True)    # 数值稳定
    w = np.exp(compat_shift)
    band_weights = w / (w.sum(axis=1, keepdims=True) + 1e-12)    # [E,B]

    # ===== 作图 =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # 1) 左：在 [0,1] 上画出 experts 和 bands 的位置
    ax0 = axes[0]
    xs_e = expert_freq
    ys_e = np.zeros_like(xs_e)

    xs_b = band_centers
    ys_b = np.ones_like(xs_b) * 0.15

    ax0.scatter(xs_e, ys_e, marker='o')
    ax0.scatter(xs_b, ys_b, marker='x')

    for i, x in enumerate(xs_e):
        ax0.text(float(x), float(ys_e[i]) + 0.02, f"E{i}", ha='center', fontsize=8)
    for j, x in enumerate(xs_b):
        ax0.text(float(x), float(ys_b[j]) + 0.02, f"B{j}", ha='center', fontsize=8)

    ax0.set_yticks([])
    ax0.set_xlim(0.0, 1.0)
    ax0.set_xlabel("normalized radius r* ∈ [0,1]")
    ax0.set_title(f"{router_name}: expert meta frequency vs band centers")

    # 2) 右：expert × band 的 soft 权重热力图
    ax1 = axes[1]
    im = ax1.imshow(band_weights, aspect='auto', origin='lower')
    ax1.set_xlabel("band index")
    ax1.set_ylabel("expert index")
    ax1.set_title(f"{router_name}: expert–band preference matrix")

    ax1.set_xticks(np.arange(B))
    ax1.set_xticklabels([f"{c:.2f}" for c in band_centers], rotation=45)
    ax1.set_yticks(np.arange(E))
    ax1.set_yticklabels([f"E{i}" for i in range(E)])

    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("preference weight")

    # ===== 保存 & 日志 =====
    tb_tag = f"{log_prefix}/freq_pref"
    wb_key = f"{log_prefix}/freq_pref"

    _log_figure(
        fig,
        save_path,
        tb_writer=tb_writer,
        tb_tag=tb_tag,
        step=global_step,
        wandb_run=wandb_run,
        wb_key=wb_key,
        dpi=300,
    )
    plt.close(fig)
    print(f"[FreqPrefVis] expert frequency preference saved to {save_path}")


def visualize_expert_freq_preference_from_router(
    router: Any,
    save_dir: Union[str, Path],
    epoch: Optional[int] = None,
    router_name: str = "router",
    freq_affinity_sharpness: Optional[float] = None,
    tb_writer=None,
    wandb_run=None,
    global_step=None,
    log_prefix: str = "router/freq_pref",
):
    """
    直接从 router 中取参数进行可视化：
      - 需要 router.expert_freq: [E]
      - 需要 router.band_centers: [B]
      - freq_affinity_sharpness:
          * 若为 None，则尝试从 router.freq_affinity_sharpness 读取，
            否则默认 10.0
    使用方式（例）：
        stats = moe.get_router_stats()
        visualize_expert_freq_preference_from_router(
            moe.router, './results/router_vis', epoch=epoch
        )
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(router, "expert_freq"):
        print("[FreqPrefVis] router 不包含属性 'expert_freq'，无法可视化频率偏好。")
        return
    if not hasattr(router, "band_centers"):
        print("[FreqPrefVis] router 不包含属性 'band_centers'，无法可视化频率偏好。")
        return

    expert_freq = router.expert_freq.detach().cpu().numpy()
    band_centers = router.band_centers.detach().cpu().numpy()

    if freq_affinity_sharpness is None:
        if hasattr(router, "freq_affinity_sharpness"):
            freq_affinity_sharpness = float(router.freq_affinity_sharpness)
        else:
            freq_affinity_sharpness = 10.0

    if epoch is None:
        fname = "router_expert_freq_pref.png"
    else:
        fname = f"router_expert_freq_pref_epoch_{epoch+1}.png"

    save_path = save_dir / fname

    visualize_expert_freq_preference(
        expert_freq=expert_freq,
        band_centers=band_centers,
        save_path=str(save_path),
        freq_affinity_sharpness=freq_affinity_sharpness,
        router_name=router_name,
        tb_writer=tb_writer,
        wandb_run=wandb_run,
        global_step=global_step,
        log_prefix=log_prefix,
    )
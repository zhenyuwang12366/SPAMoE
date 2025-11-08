import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import re
import pandas as pd
import torch
import os, io
import time
from pathlib import Path
from typing import Optional, Dict, Union, Any
import numpy as np
from scipy.fft import fft2, fftshift
from PIL import Image
import json

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

    Args:
        logits: Tensor[B, num_types] 或 None。未softmax的原始logits。
        batch:  DataLoader 的一个 batch，支持字段：
                - 'label': Tensor[B] 或 list[int]
                - 'input_file': list[str]（可选）
        save_dir: 当前 epoch 的输出目录，例如 results_dir / f"vis_epoch_{epoch+1}"
        epoch: 当前 epoch 编号
        config: 包含 `type_id_specific` (str→int) 字典的配置对象
        filename: 输出文件名
        append: 是否追加写入（True=追加）
        is_logger: 是否打印日志
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

import re
import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_curve(log_file, save_path=None):
    """
    从日志文件中解析并绘制 Train Loss 和 Val Loss 曲线
    自动适配是否存在 CE 列
    """
    with open(log_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 尝试两种日志模式：带 CE 和不带 CE
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

    # 优先匹配带 CE 的格式
    rows = [m.groups() for m in pattern_with_ce.finditer(text)]
    if rows:
        columns = ["Epoch", "Train Loss", "Val Loss", "MAE", "MSE", "PSNR", "RMSE", "SSIM", "CE"]
    else:
        rows = [m.groups() for m in pattern_without_ce.finditer(text)]
        columns = ["Epoch", "Train Loss", "Val Loss", "MAE", "MSE", "PSNR", "RMSE", "SSIM"]

    if not rows:
        raise ValueError("日志格式不匹配，请检查 log_file 是否包含正确的数值行")

    # 构建 DataFrame
    df = pd.DataFrame(rows, columns=columns).astype(float)
    df_grouped = df.groupby("Epoch").mean().reset_index()

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df_grouped["Epoch"], df_grouped["Train Loss"], label="Train Loss", lw=2)
    plt.plot(df_grouped["Epoch"], df_grouped["Val Loss"], label="Val Loss", lw=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f" 保存曲线到 {save_path}")
    else:
        plt.show()

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
    arr_chw  = np.transpose(arr_hwc, (2, 0, 1))    # (3,H,W)
    return arr_chw, arr_hwc

def _log_figure(fig, save_path, tb_writer=None, tb_tag=None, step=None,
                wandb_run=None, wb_key=None, dpi=200):
    """
    保存 + 同步到 TensorBoard 和 W&B。三者都可选。
    """
    # 1) 保存到本地
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # 2) 转成数组
    arr_chw, arr_hwc = _fig_to_arrays(fig, dpi=dpi)

    # 3) TensorBoard
    if tb_writer is not None and tb_tag is not None:
        tb_writer.add_image(tb_tag, arr_chw, global_step=step)

    # 4) Weights & Biases
    if wandb_run is not None and wb_key is not None:
        try:
            import wandb
            wandb_run.log({wb_key: wandb.Image(arr_hwc)}, step=step)
        except Exception as e:
            print(f"[warn] wandb log 失败：{e}")


def visualize_results(inputs,
                      targets,
                      predictions,
                      save_dir='./results',
                      max_samples=4,
                      # 新增：日志相关
                      tb_writer=None,         # torch.utils.tensorboard.SummaryWriter 实例或 None
                      wandb_run=None,         # wandb.run 实例或 None
                      global_step=None,       # 当前步数/epoch
                      log_prefix='vis'):
    """可视化地震数据和预测结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    n_samples = min(inputs.shape[0], max_samples)
    
    for i in range(n_samples):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)  
        
        input_tensor = inputs[i]
        while input_tensor.dim() > 3 and input_tensor.shape[0] == 1:
            input_tensor = input_tensor.squeeze(0)

        if input_tensor.dim() == 3:
            channels, height, width = input_tensor.shape
            if channels > 1:
                merged = input_tensor.permute(1, 0, 2).contiguous().view(height, channels * width)
                input_data = merged.cpu().numpy()
            else:
                input_data = input_tensor.squeeze(0).cpu().numpy()
        elif input_tensor.dim() == 2:
            input_data = input_tensor.cpu().numpy()
        else:
            input_data = input_tensor.squeeze().cpu().numpy()

        im0 = axes[0].imshow(input_data, cmap='viridis', aspect='auto')
        axes[0].set_title('inputs data')
        input_cbar = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        input_cbar.set_label('Amplitude')
        
        target_map = targets[i, 0].cpu().numpy()
        pred_map   = predictions[i, 0].cpu().numpy()
        vmin, vmax = float(target_map.min()), float(target_map.max())
        shared_norm = Normalize(vmin=vmin, vmax=vmax)

        im1 = axes[1].imshow(target_map, cmap='jet', norm=shared_norm, aspect='auto')
        axes[1].set_title('targets model')

        im2 = axes[2].imshow(pred_map, cmap='jet', norm=shared_norm, aspect='auto')
        axes[2].set_title('predictions model')

        cbar = fig.colorbar(im1, ax=axes[1:3], fraction=0.046, pad=0.04)
        cbar.set_label('Model inpedance')

        png_path = os.path.join(save_dir, f'sample_{i}.png')
        tb_tag   = f"{log_prefix}/sample_{i}/composed"
        wb_key   = f"{log_prefix}/sample_{i}/figure"
        _log_figure(fig, png_path,
                    tb_writer=tb_writer, tb_tag=tb_tag, step=global_step,
                    wandb_run=wandb_run, wb_key=wb_key, dpi=300)
        
        plt.close(fig)

def analyze_fourier_domain(inputs,
                           targets,
                           predictions,
                           save_dir='./results',
                           max_samples=4,
                           tb_writer=None,
                           wandb_run=None,
                           global_step=None,
                           log_prefix='vis'):
    """
    分析输入与输出速度波形图在傅里叶域的特点（最终整合版）
    - 修复 HF ratio = 0.00
    - 标题中加入 Dominant Frequency
    - 保留所有 WandB / TensorBoard / 文件保存逻辑
    """
    os.makedirs(save_dir, exist_ok=True)
    n_samples = min(inputs.shape[0], max_samples)

    # 改进版 HF/LF 比率计算函数（归一化环带）
    def compute_hf_lf_ratio(psd, low_band=(0.05, 0.3), high_band=(0.4, 0.85)):
        h, w = psd.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        r_norm = r / (np.sqrt(cy**2 + cx**2) + 1e-12)
        lf_mask = (r_norm >= low_band[0]) & (r_norm < low_band[1])
        hf_mask = (r_norm >= high_band[0]) & (r_norm < high_band[1])
        low_e = float(np.sum(psd[lf_mask]))
        high_e = float(np.sum(psd[hf_mask]))
        if low_e < 1e-12:
            return 0.0
        hf_ratio = high_e / (low_e + 1e-12)
        return float(np.clip(hf_ratio, 1e-6, 1e3))  # 防止0或无穷大

    for i in range(n_samples):
        try:
            print(f"Processing sample {i}...")

            input_data = inputs[i].detach().cpu().numpy().squeeze()
            target_data = targets[i].detach().cpu().numpy().squeeze()
            pred_data = predictions[i].detach().cpu().numpy().squeeze()

            if input_data.ndim != 2 or target_data.ndim != 2 or pred_data.ndim != 2:
                print(f"  Warning: Non-2D data detected, skipping sample {i}")
                continue

            # === FFT ===
            input_psd = np.abs(fftshift(fft2(input_data))) ** 2
            target_psd = np.abs(fftshift(fft2(target_data))) ** 2
            pred_psd = np.abs(fftshift(fft2(pred_data))) ** 2

            # === 空间与频谱统计 ===
            input_mean, input_std = float(np.mean(input_data)), float(np.std(input_data))
            target_mean, target_std = float(np.mean(target_data)), float(np.std(target_data))
            pred_mean, pred_std = float(np.mean(pred_data)), float(np.std(pred_data))

            input_psd_mean, input_psd_max = float(np.mean(input_psd)), float(np.max(input_psd))
            target_psd_mean, target_psd_max = float(np.mean(target_psd)), float(np.max(target_psd))
            pred_psd_mean, pred_psd_max = float(np.mean(pred_psd)), float(np.max(pred_psd))

            # === 主频率 ===
            h, w = input_psd.shape
            cy, cx = h // 2, w // 2
            idx_i = np.unravel_index(np.argmax(input_psd), input_psd.shape)
            idx_t = np.unravel_index(np.argmax(target_psd), target_psd.shape)
            idx_p = np.unravel_index(np.argmax(pred_psd), pred_psd.shape)
            input_domf = float(np.hypot(idx_i[0] - cy, idx_i[1] - cx))
            target_domf = float(np.hypot(idx_t[0] - cy, idx_t[1] - cx))
            pred_domf = float(np.hypot(idx_p[0] - cy, idx_p[1] - cx))

            # === 改进 HF/LF ===
            input_hf_ratio = compute_hf_lf_ratio(input_psd)
            target_hf_ratio = compute_hf_lf_ratio(target_psd)
            pred_hf_ratio = compute_hf_lf_ratio(pred_psd)

            # === 能量比例 ===
            total_energy = float(np.sum(input_psd) + np.sum(target_psd) + np.sum(pred_psd))
            input_energy_ratio = float(np.sum(input_psd) / total_energy)
            target_energy_ratio = float(np.sum(target_psd) / total_energy)
            pred_energy_ratio = float(np.sum(pred_psd) / total_energy)

            # === 可视化 ===
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            vmin_phys = min(target_data.min(), pred_data.min())
            vmax_phys = max(target_data.max(), pred_data.max())
            vmin_freq = np.log10(min(target_psd.min(), pred_psd.min()) + 1e-10)
            vmax_freq = np.log10(max(target_psd.max(), pred_psd.max()) + 1e-10)

            im1 = axes[0, 0].imshow(input_data, cmap='viridis')
            axes[0, 0].set_title(f'Input Seismic Data\nMean={input_mean:.3f}, Std={input_std:.3f}')
            plt.colorbar(im1, ax=axes[0, 0])

            im2 = axes[0, 1].imshow(target_data, cmap='jet', vmin=vmin_phys, vmax=vmax_phys)
            axes[0, 1].set_title(f'Target Velocity Model\nMean={target_mean:.3f}, Std={target_std:.3f}')
            plt.colorbar(im2, ax=axes[0, 1])

            im3 = axes[0, 2].imshow(pred_data, cmap='jet', vmin=vmin_phys, vmax=vmax_phys)
            axes[0, 2].set_title(f'Predicted Velocity Model\nMean={pred_mean:.3f}, Std={pred_std:.3f}')
            plt.colorbar(im3, ax=axes[0, 2])

            im4 = axes[1, 0].imshow(np.log10(input_psd + 1e-10), cmap='viridis')
            axes[1, 0].set_title(f'Input Power Spectrum (log)\nHF/LF={input_hf_ratio:.4f}, f*={input_domf:.1f}')
            plt.colorbar(im4, ax=axes[1, 0])

            im5 = axes[1, 1].imshow(np.log10(target_psd + 1e-10), cmap='viridis', vmin=vmin_freq, vmax=vmax_freq)
            axes[1, 1].set_title(f'Target Power Spectrum (log)\nHF/LF={target_hf_ratio:.4f}, f*={target_domf:.1f}')
            plt.colorbar(im5, ax=axes[1, 1])

            im6 = axes[1, 2].imshow(np.log10(pred_psd + 1e-10), cmap='viridis', vmin=vmin_freq, vmax=vmax_freq)
            axes[1, 2].set_title(f'Predicted Power Spectrum (log)\nHF/LF={pred_hf_ratio:.4f}, f*={pred_domf:.1f}')
            plt.colorbar(im6, ax=axes[1, 2])

            plt.tight_layout()

            png_path = os.path.join(save_dir, f'fourier_analysis_sample_{i}.png')
            tb_tag = f"{log_prefix}/sample_{i}/fourier_composed"
            wb_key = f"{log_prefix}/sample_{i}/fourier_figure"

            _log_figure(fig, png_path,
                        tb_writer=tb_writer, tb_tag=tb_tag, step=global_step,
                        wandb_run=wandb_run, wb_key=wb_key, dpi=300)
            plt.close(fig)

            # === 相似性 ===
            target_flat = target_psd.flatten()
            pred_flat = pred_psd.flatten()
            correlation = float(np.corrcoef(target_flat, pred_flat)[0, 1])
            mse = float(np.mean((target_data - pred_data) ** 2))
            mae = float(np.mean(np.abs(target_data - pred_data)))
            psd_correlation = correlation

            # === 保存结果 ===
            analysis_results = {
                'sample_id': i,
                'data_shapes': {'input': list(input_data.shape), 'target': list(target_data.shape), 'prediction': list(pred_data.shape)},
                'spatial_statistics': {'input_mean': input_mean, 'input_std': input_std,
                                       'target_mean': target_mean, 'target_std': target_std,
                                       'pred_mean': pred_mean, 'pred_std': pred_std},
                'spectral_statistics': {'input_psd_mean': input_psd_mean, 'input_psd_max': input_psd_max,
                                        'target_psd_mean': target_psd_mean, 'target_psd_max': target_psd_max,
                                        'pred_psd_mean': pred_psd_mean, 'pred_psd_max': pred_psd_max},
                'frequency_characteristics': {
                    'input_dominant_freq': input_domf, 'target_dominant_freq': target_domf, 'pred_dominant_freq': pred_domf,
                    'input_hf_ratio': input_hf_ratio, 'target_hf_ratio': target_hf_ratio, 'pred_hf_ratio': pred_hf_ratio
                },
                'energy_distribution': {'input_energy_ratio': input_energy_ratio,
                                        'target_energy_ratio': target_energy_ratio,
                                        'pred_energy_ratio': pred_energy_ratio},
                'similarity_metrics': {'spatial_correlation': correlation,
                                       'spectral_correlation': psd_correlation,
                                       'mse': mse, 'mae': mae}
            }

            np.save(os.path.join(save_dir, f'fourier_analysis_sample_{i}.npy'), analysis_results)

            txt_path = os.path.join(save_dir, f'fourier_analysis_sample_{i}.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + f"\n傅里叶频域分析结果 - 样本 {i}\n" + "="*80 + "\n\n")
                f.write(json.dumps(analysis_results, indent=2, ensure_ascii=False))
                f.write(f"\n完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n")

            print(f"   Sample {i}: HF/LF → In={input_hf_ratio:.4f}, Tar={target_hf_ratio:.4f}, Pred={pred_hf_ratio:.4f}")
            print(f"   DomFreq → In={input_domf:.1f}, Tar={target_domf:.1f}, Pred={pred_domf:.1f}")
            print(f"   Results saved to: {txt_path}")
            print("-" * 80)

        except Exception as e:
            print(f"样本 {i} 分析失败: {e}")
            continue

def visualize_encoded(
    encoded,
    save_dir='./results/encoded_vis',
    max_samples=4,
    channels=None,           # 指定通道索引，如 [0,3,7]；None 则自动选择
    topk=4,                  # 自动选择的通道数
    selection='variance',    # 'variance' | 'l2' | 'random'
    norm_mode='percentile',  # 'percentile' | 'minmax'（空间域可视化归一化）
    p_low=1.0, p_high=99.0,  # 分位数上下界（percentile 模式）

    # 日志（与现有接口统一）
    tb_writer=None,
    wandb_run=None,
    global_step=None,
    log_prefix='vis/encoded',

    # ==== 新增：频谱稳健设置 ====
    use_window=True,         # 2D Hann 窗抑制谱泄漏
    r_dc=0.02,               # 排除极低频/直流邻域（归一化半径）
    lf_band=(0.05, 0.30),    # 低频环带 [r1, r2]（归一化半径）
    hf_band=(0.40, 0.85),    # 高频环带 [r3, r4]（归一化半径）

    # 颜色范围：是否全 batch 统一频谱 clim（默认按样本统一即可）
    unify_freq_clim_across_batch=False,

    # 可注入自定义的图像记录回调，避免依赖全局 _log_figure
    log_callback=None,
):
    """
    可视化 Encoder 输出特征并进行空间/频谱统计（稳健版）。
    输入:
        encoded: torch.Tensor [B, C, H, W]
    输出:
        - 每个样本一张 2xK 图 (K=通道数)：第一行空间特征、第二行功率谱(log)
        - 每个样本对应 *.npy 与 *.txt 的统计文件
        - encoded_vis_meta.json 记录通道选择与设置
    额外改进:
        - 频谱前乘 2D Hann 窗（可关）
        - 主频在排除 DC 的有效区域内寻找
        - HF/LF 采用同心环带定义，更稳定
        - 仅一次 FFT，缓存 log-PSD，避免重复计算
        - 返回结构化 summary 便于上层日志/可视化
    返回:
        {'meta': {...}, 'summary': {...}, 'per_sample': [ {...}, ... ]}
    """
    import os, time, json, random
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from scipy.fft import fft2, fftshift

    os.makedirs(save_dir, exist_ok=True)
    assert encoded.dim() == 4, f"encoded 应为 [B,C,H,W]，但收到 {tuple(encoded.shape)}"
    B, C, H, W = encoded.shape
    n_samples = min(B, max_samples)

    # ---- 准备 CPU 数据，避免显存占用与梯度污染 ----
    feat = encoded.detach().to('cpu')  # [B,C,H,W]

    # ---- 通道选择：在整个 batch 上评分以稳定选择 ----
    if channels is None:
        k = min(topk, C)
        if selection == 'random':
            perm = torch.randperm(C)
            channels = perm[:k].tolist()
        else:
            flat = feat.reshape(B, C, -1)  # [B,C,HW]
            if selection == 'variance':
                score = flat.var(dim=-1, unbiased=False).mean(dim=0)  # [C]
            elif selection == 'l2':
                score = (flat.pow(2).sum(dim=-1)).mean(dim=0)         # [C]
            else:
                raise ValueError(f"未知 selection: {selection}")
            top_idx = torch.topk(score, k=k, largest=True).indices
            channels = top_idx.tolist()
    else:
        channels = sorted(set(int(c) for c in channels if 0 <= int(c) < C))
        if len(channels) == 0:
            raise ValueError("channels 为空或越界")

    K = len(channels)

    # ==== 频率网格（与 H,W 相关，样本内复用） ====
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = np.sqrt((cy)**2 + (cx)**2)  # 归一化半径的最大值近似
    r_norm = rr / (rmax + 1e-12)

    # 环带掩码函数
    def band_mask(r_lo, r_hi):
        return (r_norm >= r_lo) & (r_norm < r_hi)

    mask_valid = r_norm > float(r_dc)
    lf_mask = band_mask(float(lf_band[0]), float(lf_band[1]))
    hf_mask = band_mask(float(hf_band[0]), float(hf_band[1]))

    # 可选 2D Hann 窗
    if use_window:
        wy = np.hanning(H)[:, None]
        wx = np.hanning(W)[None, :]
        win = (wy * wx).astype(np.float64)
    else:
        win = None

    # ==== 若需要跨样本统一频谱 clim，先全收集 ====
    all_log_psd_for_batch = []  # 仅用于 unify_freq_clim_across_batch=True

    per_sample_summaries = []

    for i in range(n_samples):
        try:
            fmap = feat[i, channels].numpy()  # [K,H,W]

            # 计算布局
            fig, axes = plt.subplots(2, K, figsize=(4.8 * K, 8), constrained_layout=True)
            if K == 1:
                axes = np.array([[axes[0]], [axes[1]]])

            # 统计容器
            ch_stats = []
            # 相关矩阵数据
            spatial_mat = np.zeros((K, H * W), dtype=np.float64)
            spectral_mat = np.zeros((K, H * W), dtype=np.float64)

            # 频谱缓存（避免重复 FFT）
            all_log_psd_list = []

            # ===== 周期：通道 =====
            for j, ch in enumerate(channels):
                arr = fmap[j].astype(np.float64)  # [H,W]

                # --- 空间统计 ---
                mean_ = float(np.mean(arr))
                std_  = float(np.std(arr))
                min_  = float(np.min(arr))
                max_  = float(np.max(arr))
                l2_   = float(np.sqrt(np.sum(arr**2)) + 1e-12)

                # --- 归一化（仅用于显示） ---
                if norm_mode == 'percentile':
                    lo = np.percentile(arr, p_low)
                    hi = np.percentile(arr, p_high)
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
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

                # --- 频域（一次性 + 稳健） ---
                arr_in = arr * win if use_window else arr
                fft2c = fftshift(fft2(arr_in))
                psd = (np.abs(fft2c) ** 2).astype(np.float64)
                log_psd = np.log10(psd + 1e-10)
                all_log_psd_list.append(log_psd)

                # 主频（排除 DC 区域）
                psd_valid = psd.copy()
                psd_valid[~mask_valid] = -np.inf
                # 若全为 -inf，回退到全域
                if not np.isfinite(psd_valid).any():
                    psd_valid = psd
                max_idx = np.unravel_index(int(np.nanargmax(psd_valid)), psd.shape)
                dom_r_px = float(rr[max_idx])
                dom_r_hat = float(r_norm[max_idx])  # 0-1

                # HF/LF 能量
                low_e  = float(psd[lf_mask].sum())
                high_e = float(psd[hf_mask].sum())
                hf_ratio = float(high_e / (low_e + 1e-12))
                total_energy_ch = float(psd.sum())

                # --- 可视化：空间+频谱 ---
                im_spatial = axes[0, j].imshow(vis_img, cmap='viridis', aspect='auto',
                                               vmin=vmin, vmax=vmax)
                axes[0, j].set_title(
                    f'Encoder Feature | sample {i} | ch {ch}\n'
                    f'{range_note}\nμ={mean_:.3e}, σ={std_:.3e}, L2={l2_:.2e}'
                )
                axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
                plt.colorbar(im_spatial, ax=axes[0, j], fraction=0.046, pad=0.04)\
                    .set_label('Normalized activation')

                im_freq = axes[1, j].imshow(log_psd, cmap='viridis', aspect='auto')
                axes[1, j].set_title(
                    f'Power Spectrum (log10)\n'
                    f'r*={dom_r_hat:.2f}, HF/LF={hf_ratio:.3f}'
                )
                axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
                plt.colorbar(im_freq, ax=axes[1, j], fraction=0.046, pad=0.04)\
                    .set_label('log10(PSD)')

                # 统计持久化
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

                # 相关矩阵数据
                spatial_mat[j, :]  = arr.reshape(-1)
                spectral_mat[j, :] = log_psd.reshape(-1)

                # 跨 batch 统一 clim 的收集
                if unify_freq_clim_across_batch:
                    all_log_psd_for_batch.append(log_psd)

            # --- 统一频谱颜色范围（样本内） ---
            all_log_psd = np.stack(all_log_psd_list, axis=0)  # [K,H,W]
            vmin_freq = float(np.percentile(all_log_psd, 1.0))
            vmax_freq = float(np.percentile(all_log_psd, 99.0))
            if not np.isfinite(vmin_freq): vmin_freq = float(np.min(all_log_psd))
            if not np.isfinite(vmax_freq) or vmax_freq <= vmin_freq:
                vmax_freq = vmin_freq + 1e-6
            for j in range(K):
                axes[1, j].images[0].set_clim(vmin=vmin_freq, vmax=vmax_freq)

            # --- 能量占比（在所选K个通道内） ---
            total_energy_selected = float(sum(cs['spectral']['psd_sum'] for cs in ch_stats)) + 1e-12
            for cs in ch_stats:
                cs['spectral']['energy_ratio_selected'] = float(cs['spectral']['psd_sum'] / total_energy_selected)

            # --- 通道间相关矩阵 ---
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

            # --- 保存图像 ---
            png_path = os.path.join(save_dir, f'encoded_sample_{i}_ch_{"-".join(map(str,channels))}.png')
            if callable(log_callback):
                log_callback(fig, png_path,
                             tb_writer=tb_writer, tb_tag=f"{log_prefix}/sample_{i}/encoded_composed",
                             step=global_step, wandb_run=wandb_run, wb_key=f"{log_prefix}/sample_{i}/encoded_figure",
                             dpi=300)
            else:
                fig.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            # --- 组织结果字典 & 保存 NPY ---
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

            # --- 写 TXT 摘要 ---
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
                    s  = cs['spatial']
                    sp = cs['spectral']
                    f.write(f"[Channel {ch}]\n")
                    f.write(f"  空间: mean={s['mean']:.6e}, std={s['std']:.6e}, "
                            f"min={s['min']:.6e}, max={s['max']:.6e}, L2={s['l2']:.6e}\n")
                    f.write(f"  频谱: psd_mean={sp['psd_mean']:.6e}, psd_max={sp['psd_max']:.6e}, "
                            f"r*={sp['dominant_radius_norm']:.3f}, HF/LF={sp['hf_ratio']:.6f}, "
                            f"energy_ratio_selected={sp['energy_ratio_selected']:.6f}\n\n")

                f.write("空间域通道间相关矩阵（K×K）:\n")
                f.write(np.array2string(spatial_corr, formatter={'float_kind':lambda x: f"{x: .3f}"}))
                f.write("\n\n频谱域通道间相关矩阵（K×K，基于 log10(PSD)）:\n")
                f.write(np.array2string(spectral_corr, formatter={'float_kind':lambda x: f"{x: .3f}"}))
                f.write("\n\n")

                f.write("=" * 90 + "\n")
                f.write(f"保存图像: {png_path}\n")
                f.write(f"保存 NPY:  {os.path.join(save_dir, f'encoded_analysis_sample_{i}.npy')}\n")
                f.write(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 90 + "\n")

            # --- 终端摘要 ---
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
            import traceback; traceback.print_exc()
            continue

    # ---- 元数据记录（复现通道选择等） ----
    meta = {
        'shape': {'B': B, 'C': C, 'H': H, 'W': W},
        'max_samples': n_samples,
        'selected_channels': [int(c) for c in channels],
        'selection': selection,
        'topk': topk,
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

    # ---- 跨 batch 统一频谱 clim（可选，通常不必）----
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
            'r_dc': float(r_dc), 'lf': [float(lf_band[0]), float(lf_band[1])],
            'hf': [float(hf_band[0]), float(hf_band[1])], 'use_window': bool(use_window)
        }
    }
    return {'meta': meta, 'summary': summary, 'per_sample': per_sample_summaries}
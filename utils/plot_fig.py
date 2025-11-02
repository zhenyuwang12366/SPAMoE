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
                           # 新增：日志相关
                           tb_writer=None,
                           wandb_run=None,
                           global_step=None,
                           log_prefix='vis'):
    """
    分析输入和输出速度波形图在傅里叶域的特点 - 完全安全版本
    
    Parameters:
    -----------
    inputs : torch.Tensor
        输入地震数据 [B, C, H, W]
    targets : torch.Tensor  
        目标速度模型 [B, C, H, W]
    predictions : torch.Tensor
        预测速度模型 [B, C, H, W]
    save_dir : str
        保存目录
    max_samples : int
        最大分析样本数
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 限制样本数
    n_samples = min(inputs.shape[0], max_samples)
    
    for i in range(n_samples):
        try:
            print(f"Processing sample {i}...")
            
            # 安全地获取数据并转换为numpy
            input_tensor = inputs[i]
            target_tensor = targets[i] 
            pred_tensor = predictions[i]
            
            # 处理输入数据
            if input_tensor.dim() > 2:
                input_data = input_tensor[0].detach().cpu().numpy()
            else:
                input_data = input_tensor.detach().cpu().numpy()
                
            # 处理目标数据
            if target_tensor.dim() > 2:
                target_data = target_tensor[0].detach().cpu().numpy()
            else:
                target_data = target_tensor.detach().cpu().numpy()
                
            # 处理预测数据
            if pred_tensor.dim() > 2:
                pred_data = pred_tensor[0].detach().cpu().numpy()
            else:
                pred_data = pred_tensor.detach().cpu().numpy()
            
            # 确保数据是2D的
            while input_data.ndim > 2:
                input_data = input_data.squeeze()
            while target_data.ndim > 2:
                target_data = target_data.squeeze()
            while pred_data.ndim > 2:
                pred_data = pred_data.squeeze()
            
            print(f"  Data shapes: Input{input_data.shape}, Target{target_data.shape}, Pred{pred_data.shape}")
            
            # 确保所有数据都是2D的
            if input_data.ndim != 2 or target_data.ndim != 2 or pred_data.ndim != 2:
                print(f"  Warning: Non-2D data detected, skipping sample {i}")
                continue
            
            # 计算2D傅里叶变换
            input_fft = fft2(input_data)
            target_fft = fft2(target_data)
            pred_fft = fft2(pred_data)
            
            # 计算功率谱密度 (PSD)
            input_psd = np.abs(fftshift(input_fft))**2
            target_psd = np.abs(fftshift(target_fft))**2
            pred_psd = np.abs(fftshift(pred_fft))**2
            
            print(f"  PSD shapes: Input{input_psd.shape}, Target{target_psd.shape}, Pred{pred_psd.shape}")
            
            # 计算基本统计信息
            input_mean = float(np.mean(input_data))
            input_std = float(np.std(input_data))
            target_mean = float(np.mean(target_data))
            target_std = float(np.std(target_data))
            pred_mean = float(np.mean(pred_data))
            pred_std = float(np.std(pred_data))
            
            # 计算功率谱统计
            input_psd_mean = float(np.mean(input_psd))
            input_psd_max = float(np.max(input_psd))
            target_psd_mean = float(np.mean(target_psd))
            target_psd_max = float(np.max(target_psd))
            pred_psd_mean = float(np.mean(pred_psd))
            pred_psd_max = float(np.max(pred_psd))
            
            # 计算频率域特征 - 安全版本
            h, w = input_psd.shape
            center_h, center_w = h // 2, w // 2
            
            # 找到功率谱最大值的位置
            input_max_idx = np.unravel_index(np.argmax(input_psd), input_psd.shape)
            target_max_idx = np.unravel_index(np.argmax(target_psd), target_psd.shape)
            pred_max_idx = np.unravel_index(np.argmax(pred_psd), pred_psd.shape)
            
            # 计算主频率（相对于中心）
            input_dominant_freq = float(np.sqrt((input_max_idx[0] - center_h)**2 + (input_max_idx[1] - center_w)**2))
            target_dominant_freq = float(np.sqrt((target_max_idx[0] - center_h)**2 + (target_max_idx[1] - center_w)**2))
            pred_dominant_freq = float(np.sqrt((pred_max_idx[0] - center_h)**2 + (pred_max_idx[1] - center_w)**2))
            
            # 计算频谱能量分布
            total_energy = float(np.sum(input_psd) + np.sum(target_psd) + np.sum(pred_psd))
            input_energy_ratio = float(np.sum(input_psd) / total_energy) if total_energy > 0 else 0.0
            target_energy_ratio = float(np.sum(target_psd) / total_energy) if total_energy > 0 else 0.0
            pred_energy_ratio = float(np.sum(pred_psd) / total_energy) if total_energy > 0 else 0.0
            
            # 计算高频/低频能量比 - 完全安全版本
            try:
                # 创建距离矩阵
                y_coords, x_coords = np.ogrid[:h, :w]
                distances = np.sqrt((x_coords - center_w)**2 + (y_coords - center_h)**2)
                
                # 确保掩码尺寸正确
                low_freq_mask = distances < min(h, w) * 0.3
                high_freq_mask = distances > min(h, w) * 0.7
                
                # 计算能量
                input_low_energy = float(np.sum(input_psd[low_freq_mask]))
                input_high_energy = float(np.sum(input_psd[high_freq_mask]))
                input_hf_ratio = input_high_energy / (input_low_energy + 1e-10)
                
                target_low_energy = float(np.sum(target_psd[low_freq_mask]))
                target_high_energy = float(np.sum(target_psd[high_freq_mask]))
                target_hf_ratio = target_high_energy / (target_low_energy + 1e-10)
                
                pred_low_energy = float(np.sum(pred_psd[low_freq_mask]))
                pred_high_energy = float(np.sum(pred_psd[high_freq_mask]))
                pred_hf_ratio = pred_high_energy / (pred_low_energy + 1e-10)
                
            except Exception as mask_error:
                print(f"  Warning: Frequency mask calculation failed: {mask_error}")
                # 使用默认值
                input_hf_ratio = target_hf_ratio = pred_hf_ratio = 0.0
            
            # 创建可视化
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # === 统一颜色范围 ===
            # 原始/预测/目标的物理域
            vmin_phys = min(target_data.min(), pred_data.min())
            vmax_phys = max(target_data.max(), pred_data.max())

            # 傅里叶域（取log10前的PSD）
            vmin_freq = np.log10(min(target_psd.min(), pred_psd.min()) + 1e-10)
            vmax_freq = np.log10(max(target_psd.max(), pred_psd.max()) + 1e-10)

            # === 第一行：原始数据 ===
            im1 = axes[0, 0].imshow(input_data, cmap='viridis')
            axes[0, 0].set_title(f'Input Seismic Data\nMean: {input_mean:.3f}, Std: {input_std:.3f}')
            plt.colorbar(im1, ax=axes[0, 0])

            im2 = axes[0, 1].imshow(target_data, cmap='jet', vmin=vmin_phys, vmax=vmax_phys)
            axes[0, 1].set_title(f'Target Velocity Model\nMean: {target_mean:.3f}, Std: {target_std:.3f}')
            plt.colorbar(im2, ax=axes[0, 1])

            im3 = axes[0, 2].imshow(pred_data, cmap='jet', vmin=vmin_phys, vmax=vmax_phys)
            axes[0, 2].set_title(f'Predicted Velocity Model\nMean: {pred_mean:.3f}, Std: {pred_std:.3f}')
            plt.colorbar(im3, ax=axes[0, 2])

            # === 第二行：傅里叶域 ===
            im4 = axes[1, 0].imshow(np.log10(input_psd + 1e-10), cmap='viridis')
            axes[1, 0].set_title(f'Input Power Spectrum (log)\nMax Freq: {input_dominant_freq:.1f}, HF Ratio: {input_hf_ratio:.3f}')
            plt.colorbar(im4, ax=axes[1, 0])

            im5 = axes[1, 1].imshow(np.log10(target_psd + 1e-10), cmap='viridis', vmin=vmin_freq, vmax=vmax_freq)
            axes[1, 1].set_title(f'Target Power Spectrum (log)\nMax Freq: {target_dominant_freq:.1f}, HF Ratio: {target_hf_ratio:.3f}')
            plt.colorbar(im5, ax=axes[1, 1])

            im6 = axes[1, 2].imshow(np.log10(pred_psd + 1e-10), cmap='viridis', vmin=vmin_freq, vmax=vmax_freq)
            axes[1, 2].set_title(f'Predicted Power Spectrum (log)\nMax Freq: {pred_dominant_freq:.1f}, HF Ratio: {pred_hf_ratio:.3f}')
            plt.colorbar(im6, ax=axes[1, 2])

            plt.tight_layout()
            
            # === 保存 & 日志 ===
            png_path = os.path.join(save_dir, f'fourier_analysis_sample_{i}.png')
            tb_tag   = f"{log_prefix}/sample_{i}/fourier_composed"
            wb_key   = f"{log_prefix}/sample_{i}/fourier_figure"

            _log_figure(fig, png_path,
                        tb_writer=tb_writer, tb_tag=tb_tag, step=global_step,
                        wandb_run=wandb_run, wb_key=wb_key, dpi=300)
            
            plt.close(fig)
            
            # 计算相似性指标
            try:
                target_flat = target_psd.flatten()
                pred_flat = pred_psd.flatten()
                correlation = float(np.corrcoef(target_flat, pred_flat)[0, 1])
            except:
                correlation = 0.0
            
            # 计算预测误差
            mse = float(np.mean((target_data - pred_data)**2))
            mae = float(np.mean(np.abs(target_data - pred_data)))
            
            # 计算频谱相似性
            try:
                target_psd_flat = target_psd.flatten()
                pred_psd_flat = pred_psd.flatten()
                psd_correlation = float(np.corrcoef(target_psd_flat, pred_psd_flat)[0, 1])
            except:
                psd_correlation = 0.0
            
            # 保存分析结果
            analysis_results = {
                'sample_id': i,
                'data_shapes': {
                    'input': list(input_data.shape),
                    'target': list(target_data.shape),
                    'prediction': list(pred_data.shape)
                },
                'spatial_statistics': {
                    'input_mean': input_mean,
                    'input_std': input_std,
                    'target_mean': target_mean,
                    'target_std': target_std,
                    'pred_mean': pred_mean,
                    'pred_std': pred_std
                },
                'spectral_statistics': {
                    'input_psd_mean': input_psd_mean,
                    'input_psd_max': input_psd_max,
                    'target_psd_mean': target_psd_mean,
                    'target_psd_max': target_psd_max,
                    'pred_psd_mean': pred_psd_mean,
                    'pred_psd_max': pred_psd_max
                },
                'frequency_characteristics': {
                    'input_dominant_freq': input_dominant_freq,
                    'target_dominant_freq': target_dominant_freq,
                    'pred_dominant_freq': pred_dominant_freq,
                    'input_hf_ratio': input_hf_ratio,
                    'target_hf_ratio': target_hf_ratio,
                    'pred_hf_ratio': pred_hf_ratio
                },
                'energy_distribution': {
                    'input_energy_ratio': input_energy_ratio,
                    'target_energy_ratio': target_energy_ratio,
                    'pred_energy_ratio': pred_energy_ratio
                },
                'similarity_metrics': {
                    'spatial_correlation': correlation,
                    'spectral_correlation': psd_correlation,
                    'mse': mse,
                    'mae': mae
                }
            }
            
            # 保存为numpy文件
            np.save(os.path.join(save_dir, f'fourier_analysis_sample_{i}.npy'), analysis_results)
            
            # 保存数值数据到txt文件
            txt_file_path = os.path.join(save_dir, f'fourier_analysis_sample_{i}.txt')
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"傅里叶频域分析结果 - 样本 {i}\n")
                f.write("=" * 80 + "\n\n")
                
                # 数据形状信息
                f.write("1. 数据形状信息:\n")
                f.write("-" * 40 + "\n")
                f.write(f"输入数据形状: {input_data.shape}\n")
                f.write(f"目标数据形状: {target_data.shape}\n")
                f.write(f"预测数据形状: {pred_data.shape}\n\n")
                
                # 空间域统计信息
                f.write("2. 空间域统计信息:\n")
                f.write("-" * 40 + "\n")
                f.write(f"输入数据 - 均值: {input_mean:.6f}, 标准差: {input_std:.6f}\n")
                f.write(f"目标数据 - 均值: {target_mean:.6f}, 标准差: {target_std:.6f}\n")
                f.write(f"预测数据 - 均值: {pred_mean:.6f}, 标准差: {pred_std:.6f}\n\n")
                
                # 频谱域统计信息
                f.write("3. 频谱域统计信息:\n")
                f.write("-" * 40 + "\n")
                f.write(f"输入功率谱 - 均值: {input_psd_mean:.6e}, 最大值: {input_psd_max:.6e}\n")
                f.write(f"目标功率谱 - 均值: {target_psd_mean:.6e}, 最大值: {target_psd_max:.6e}\n")
                f.write(f"预测功率谱 - 均值: {pred_psd_mean:.6e}, 最大值: {pred_psd_max:.6e}\n\n")
                
                # 频率特征
                f.write("4. 频率特征:\n")
                f.write("-" * 40 + "\n")
                f.write(f"主频率 - 输入: {input_dominant_freq:.3f}, 目标: {target_dominant_freq:.3f}, 预测: {pred_dominant_freq:.3f}\n")
                f.write(f"高频/低频比 - 输入: {input_hf_ratio:.6f}, 目标: {target_hf_ratio:.6f}, 预测: {pred_hf_ratio:.6f}\n\n")
                
                # 能量分布
                f.write("5. 能量分布:\n")
                f.write("-" * 40 + "\n")
                f.write(f"总能量: {total_energy:.6e}\n")
                f.write(f"能量比例 - 输入: {input_energy_ratio:.6f}, 目标: {target_energy_ratio:.6f}, 预测: {pred_energy_ratio:.6f}\n\n")
                
                # 相似性指标
                f.write("6. 相似性指标:\n")
                f.write("-" * 40 + "\n")
                f.write(f"空间域相关系数: {correlation:.6f}\n")
                f.write(f"频谱域相关系数: {psd_correlation:.6f}\n")
                f.write(f"均方误差 (MSE): {mse:.8f}\n")
                f.write(f"平均绝对误差 (MAE): {mae:.8f}\n\n")
                
                # 详细数值数据
                f.write("7. 详细数值数据 (JSON格式):\n")
                f.write("-" * 40 + "\n")
                import json
                f.write(json.dumps(analysis_results, indent=2, ensure_ascii=False))
                f.write("\n\n")
                
                f.write("=" * 80 + "\n")
                f.write(f"分析完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n")
            
            # 打印详细结果
            print(f"Sample {i} Fourier Analysis Completed Successfully:")
            print(f"  Data Shapes: Input{input_data.shape}, Target{target_data.shape}, Pred{pred_data.shape}")
            print(f"  Spatial Stats - Input: μ={input_mean:.3f}, σ={input_std:.3f}")
            print(f"  Spatial Stats - Target: μ={target_mean:.3f}, σ={target_std:.3f}")
            print(f"  Spatial Stats - Pred: μ={pred_mean:.3f}, σ={pred_std:.3f}")
            print(f"  Spectral Stats - Input: μ={input_psd_mean:.2e}, max={input_psd_max:.2e}")
            print(f"  Spectral Stats - Target: μ={target_psd_mean:.2e}, max={target_psd_max:.2e}")
            print(f"  Spectral Stats - Pred: μ={pred_psd_mean:.2e}, max={pred_psd_max:.2e}")
            print(f"  Frequency Characteristics:")
            print(f"    Dominant Freq - Input: {input_dominant_freq:.1f}, Target: {target_dominant_freq:.1f}, Pred: {pred_dominant_freq:.1f}")
            print(f"    High-Freq Ratio - Input: {input_hf_ratio:.3f}, Target: {target_hf_ratio:.3f}, Pred: {pred_hf_ratio:.3f}")
            print(f"  Energy Distribution - Input: {input_energy_ratio:.3f}, Target: {target_energy_ratio:.3f}, Pred: {pred_energy_ratio:.3f}")
            print(f"  Similarity Metrics:")
            print(f"    Spatial Correlation: {correlation:.4f}")
            print(f"    Spectral Correlation: {psd_correlation:.4f}")
            print(f"    MSE: {mse:.6f}, MAE: {mae:.6f}")
            print(f"  Results saved to: {txt_file_path}")
            print("-" * 80)
            
        except Exception as e:
            print(f"样本 {i} 傅里叶分析失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
def analyze_fourier_domain_safe(inputs, targets, predictions, save_dir='./results', max_samples=4):
    """
    分析输入和输出速度波形图在傅里叶域的特点
    
    Parameters:
    -----------
    inputs : torch.Tensor
        输入地震数据 [B, C, H, W]
    targets : torch.Tensor  
        目标速度模型 [B, C, H, W]
    predictions : torch.Tensor
        预测速度模型 [B, C, H, W]
    save_dir : str
        保存目录
    max_samples : int
        最大分析样本数
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft2, fftshift, fftfreq
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 限制样本数
    n_samples = min(inputs.shape[0], max_samples)
    
    for i in range(n_samples):
        try:
            # 获取数据并转换为numpy
            if len(inputs[i].shape) > 2:
                input_data = inputs[i, 0].detach().cpu().numpy()
            else:
                input_data = inputs[i].detach().cpu().numpy()
                
            target_data = targets[i, 0].detach().cpu().numpy()
            pred_data = predictions[i, 0].detach().cpu().numpy()
            
            # 确保数据是2D的
            if input_data.ndim > 2:
                input_data = input_data.squeeze()
            if target_data.ndim > 2:
                target_data = target_data.squeeze()
            if pred_data.ndim > 2:
                pred_data = pred_data.squeeze()
            
            # 计算2D傅里叶变换
            input_fft = fft2(input_data)
            target_fft = fft2(target_data)
            pred_fft = fft2(pred_data)
            
            # 计算功率谱密度 (PSD)
            input_psd = np.abs(fftshift(input_fft))**2
            target_psd = np.abs(fftshift(target_fft))**2
            pred_psd = np.abs(fftshift(pred_fft))**2
            
            # 计算详细的傅里叶特征
            # 1. 基本统计信息
            input_mean = np.mean(input_data)
            input_std = np.std(input_data)
            target_mean = np.mean(target_data)
            target_std = np.std(target_data)
            pred_mean = np.mean(pred_data)
            pred_std = np.std(pred_data)
            
            # 2. 功率谱统计
            input_psd_mean = np.mean(input_psd)
            input_psd_max = np.max(input_psd)
            target_psd_mean = np.mean(target_psd)
            target_psd_max = np.max(target_psd)
            pred_psd_mean = np.mean(pred_psd)
            pred_psd_max = np.max(pred_psd)
            
            # 3. 频率域特征
            # 计算主频率（功率谱最大值对应的频率）
            h, w = input_psd.shape
            center_h, center_w = h // 2, w // 2
            
            # 找到功率谱最大值的位置
            input_max_idx = np.unravel_index(np.argmax(input_psd), input_psd.shape)
            target_max_idx = np.unravel_index(np.argmax(target_psd), target_psd.shape)
            pred_max_idx = np.unravel_index(np.argmax(pred_psd), pred_psd.shape)
            
            # 计算频率（相对于中心）
            input_dominant_freq = np.sqrt((input_max_idx[0] - center_h)**2 + (input_max_idx[1] - center_w)**2)
            target_dominant_freq = np.sqrt((target_max_idx[0] - center_h)**2 + (target_max_idx[1] - center_w)**2)
            pred_dominant_freq = np.sqrt((pred_max_idx[0] - center_h)**2 + (pred_max_idx[1] - center_w)**2)
            
            # 4. 频谱能量分布
            total_energy = np.sum(input_psd) + np.sum(target_psd) + np.sum(pred_psd)
            input_energy_ratio = np.sum(input_psd) / total_energy
            target_energy_ratio = np.sum(target_psd) / total_energy
            pred_energy_ratio = np.sum(pred_psd) / total_energy
            
            # 5. 高频/低频能量比 - 修复尺寸问题
            # 确保掩码的尺寸与功率谱数组的尺寸完全匹配
            y_coords, x_coords = np.ogrid[:h, :w]
            distances = np.sqrt((x_coords - center_w)**2 + (y_coords - center_h)**2)
            
            low_freq_mask = distances < min(h, w) * 0.3
            high_freq_mask = distances > min(h, w) * 0.7
            
            input_low_energy = np.sum(input_psd[low_freq_mask])
            input_high_energy = np.sum(input_psd[high_freq_mask])
            input_hf_ratio = input_high_energy / (input_low_energy + 1e-10)
            
            target_low_energy = np.sum(target_psd[low_freq_mask])
            target_high_energy = np.sum(target_psd[high_freq_mask])
            target_hf_ratio = target_high_energy / (target_low_energy + 1e-10)
            
            pred_low_energy = np.sum(pred_psd[low_freq_mask])
            pred_high_energy = np.sum(pred_psd[high_freq_mask])
            pred_hf_ratio = pred_high_energy / (pred_low_energy + 1e-10)
            
            # 简化的傅里叶分析 - 只做基本的可视化
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 第一行：原始数据
            im1 = axes[0, 0].imshow(input_data, cmap='viridis')
            axes[0, 0].set_title(f'Input Seismic Data\nMean: {input_mean:.3f}, Std: {input_std:.3f}')
            plt.colorbar(im1, ax=axes[0, 0])
            
            im2 = axes[0, 1].imshow(target_data, cmap='jet')
            axes[0, 1].set_title(f'Target Velocity Model\nMean: {target_mean:.3f}, Std: {target_std:.3f}')
            plt.colorbar(im2, ax=axes[0, 1])
            
            im3 = axes[0, 2].imshow(pred_data, cmap='jet')
            axes[0, 2].set_title(f'Predicted Velocity Model\nMean: {pred_mean:.3f}, Std: {pred_std:.3f}')
            plt.colorbar(im3, ax=axes[0, 2])
            
            # 第二行：傅里叶域
            im4 = axes[1, 0].imshow(np.log10(input_psd + 1e-10), cmap='viridis')
            axes[1, 0].set_title(f'Input Power Spectrum (log)\nMax Freq: {input_dominant_freq:.1f}, HF Ratio: {input_hf_ratio:.3f}')
            plt.colorbar(im4, ax=axes[1, 0])
            
            im5 = axes[1, 1].imshow(np.log10(target_psd + 1e-10), cmap='viridis')
            axes[1, 1].set_title(f'Target Power Spectrum (log)\nMax Freq: {target_dominant_freq:.1f}, HF Ratio: {target_hf_ratio:.3f}')
            plt.colorbar(im5, ax=axes[1, 1])
            
            im6 = axes[1, 2].imshow(np.log10(pred_psd + 1e-10), cmap='viridis')
            axes[1, 2].set_title(f'Predicted Power Spectrum (log)\nMax Freq: {pred_dominant_freq:.1f}, HF Ratio: {pred_hf_ratio:.3f}')
            plt.colorbar(im6, ax=axes[1, 2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'fourier_analysis_sample_{i}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # 计算基本统计信息
            target_flat = target_psd.flatten()
            pred_flat = pred_psd.flatten()
            correlation = np.corrcoef(target_flat, pred_flat)[0, 1]
            
            # 计算预测误差
            mse = np.mean((target_data - pred_data)**2)
            mae = np.mean(np.abs(target_data - pred_data))
            
            # 计算频谱相似性
            target_psd_flat = target_psd.flatten()
            pred_psd_flat = pred_psd.flatten()
            psd_correlation = np.corrcoef(target_psd_flat, pred_psd_flat)[0, 1]
            
            # 保存详细的分析结果
            analysis_results = {
                'sample_id': i,
                'data_shapes': {
                    'input': input_data.shape,
                    'target': target_data.shape,
                    'prediction': pred_data.shape
                },
                'spatial_statistics': {
                    'input_mean': input_mean,
                    'input_std': input_std,
                    'target_mean': target_mean,
                    'target_std': target_std,
                    'pred_mean': pred_mean,
                    'pred_std': pred_std
                },
                'spectral_statistics': {
                    'input_psd_mean': input_psd_mean,
                    'input_psd_max': input_psd_max,
                    'target_psd_mean': target_psd_mean,
                    'target_psd_max': target_psd_max,
                    'pred_psd_mean': pred_psd_mean,
                    'pred_psd_max': pred_psd_max
                },
                'frequency_characteristics': {
                    'input_dominant_freq': input_dominant_freq,
                    'target_dominant_freq': target_dominant_freq,
                    'pred_dominant_freq': pred_dominant_freq,
                    'input_hf_ratio': input_hf_ratio,
                    'target_hf_ratio': target_hf_ratio,
                    'pred_hf_ratio': pred_hf_ratio
                },
                'energy_distribution': {
                    'input_energy_ratio': input_energy_ratio,
                    'target_energy_ratio': target_energy_ratio,
                    'pred_energy_ratio': pred_energy_ratio
                },
                'similarity_metrics': {
                    'spatial_correlation': correlation,
                    'spectral_correlation': psd_correlation,
                    'mse': mse,
                    'mae': mae
                }
            }
            
            np.save(os.path.join(save_dir, f'fourier_analysis_sample_{i}.npy'), analysis_results)
            
            # 保存数值数据到txt文件
            txt_file_path = os.path.join(save_dir, f'fourier_analysis_sample_{i}.txt')
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"傅里叶频域分析结果 - 样本 {i} (Safe版本)\n")
                f.write("=" * 80 + "\n\n")
                
                # 数据形状信息
                f.write("1. 数据形状信息:\n")
                f.write("-" * 40 + "\n")
                f.write(f"输入数据形状: {input_data.shape}\n")
                f.write(f"目标数据形状: {target_data.shape}\n")
                f.write(f"预测数据形状: {pred_data.shape}\n\n")
                
                # 空间域统计信息
                f.write("2. 空间域统计信息:\n")
                f.write("-" * 40 + "\n")
                f.write(f"输入数据 - 均值: {input_mean:.6f}, 标准差: {input_std:.6f}\n")
                f.write(f"目标数据 - 均值: {target_mean:.6f}, 标准差: {target_std:.6f}\n")
                f.write(f"预测数据 - 均值: {pred_mean:.6f}, 标准差: {pred_std:.6f}\n\n")
                
                # 频谱域统计信息
                f.write("3. 频谱域统计信息:\n")
                f.write("-" * 40 + "\n")
                f.write(f"输入功率谱 - 均值: {input_psd_mean:.6e}, 最大值: {input_psd_max:.6e}\n")
                f.write(f"目标功率谱 - 均值: {target_psd_mean:.6e}, 最大值: {target_psd_max:.6e}\n")
                f.write(f"预测功率谱 - 均值: {pred_psd_mean:.6e}, 最大值: {pred_psd_max:.6e}\n\n")
                
                # 频率特征
                f.write("4. 频率特征:\n")
                f.write("-" * 40 + "\n")
                f.write(f"主频率 - 输入: {input_dominant_freq:.3f}, 目标: {target_dominant_freq:.3f}, 预测: {pred_dominant_freq:.3f}\n")
                f.write(f"高频/低频比 - 输入: {input_hf_ratio:.6f}, 目标: {target_hf_ratio:.6f}, 预测: {pred_hf_ratio:.6f}\n\n")
                
                # 能量分布
                f.write("5. 能量分布:\n")
                f.write("-" * 40 + "\n")
                f.write(f"总能量: {total_energy:.6e}\n")
                f.write(f"能量比例 - 输入: {input_energy_ratio:.6f}, 目标: {target_energy_ratio:.6f}, 预测: {pred_energy_ratio:.6f}\n\n")
                
                # 相似性指标
                f.write("6. 相似性指标:\n")
                f.write("-" * 40 + "\n")
                f.write(f"空间域相关系数: {correlation:.6f}\n")
                f.write(f"频谱域相关系数: {psd_correlation:.6f}\n")
                f.write(f"均方误差 (MSE): {mse:.8f}\n")
                f.write(f"平均绝对误差 (MAE): {mae:.8f}\n\n")
                
                # 详细数值数据
                f.write("7. 详细数值数据 (JSON格式):\n")
                f.write("-" * 40 + "\n")
                import json
                f.write(json.dumps(analysis_results, indent=2, ensure_ascii=False))
                f.write("\n\n")
                
                f.write("=" * 80 + "\n")
                f.write(f"分析完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n")
            
            print(f"Sample {i} Fourier Analysis Completed:")
            print(f"  Data Shapes: Input{input_data.shape}, Target{target_data.shape}, Pred{pred_data.shape}")
            print(f"  Spatial Stats - Input: μ={input_mean:.3f}, σ={input_std:.3f}")
            print(f"  Spatial Stats - Target: μ={target_mean:.3f}, σ={target_std:.3f}")
            print(f"  Spatial Stats - Pred: μ={pred_mean:.3f}, σ={pred_std:.3f}")
            print(f"  Spectral Stats - Input: μ={input_psd_mean:.2e}, max={input_psd_max:.2e}")
            print(f"  Spectral Stats - Target: μ={target_psd_mean:.2e}, max={target_psd_max:.2e}")
            print(f"  Spectral Stats - Pred: μ={pred_psd_mean:.2e}, max={pred_psd_max:.2e}")
            print(f"  Frequency Characteristics:")
            print(f"    Dominant Freq - Input: {input_dominant_freq:.1f}, Target: {target_dominant_freq:.1f}, Pred: {pred_dominant_freq:.1f}")
            print(f"    High-Freq Ratio - Input: {input_hf_ratio:.3f}, Target: {target_hf_ratio:.3f}, Pred: {pred_hf_ratio:.3f}")
            print(f"  Energy Distribution - Input: {input_energy_ratio:.3f}, Target: {target_energy_ratio:.3f}, Pred: {pred_energy_ratio:.3f}")
            print(f"  Similarity Metrics:")
            print(f"    Spatial Correlation: {correlation:.4f}")
            print(f"    Spectral Correlation: {psd_correlation:.4f}")
            print(f"    MSE: {mse:.6f}, MAE: {mae:.6f}")
            print(f"  Results saved to: {txt_file_path}")
            print("-" * 80)
            
        except Exception as e:
            print(f"样本 {i} 傅里叶分析失败: {str(e)}")
            continue

def visualize_encoded(encoded,
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
                      log_prefix='vis/encoded'):
    """
    可视化 Encoder 输出特征并进行空间/频谱统计。
    输入:
        encoded: torch.Tensor [B, C, H, W]
    输出:
        - 每个样本一张 2xK 图 (K=通道数)：第一行空间特征、第二行功率谱(log)
        - 每个样本对应 *.npy 与 *.txt 的统计文件
        - encoded_vis_meta.json 记录通道选择与设置
    """
    import os, time, json, random
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
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
                # 对 HW 求方差，再对 B 求平均
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

    # ---- 频域掩码（HF/LF）准备：与 H,W 无关但按每个样本/通道二维大小重新生成 ----
    def make_freq_masks(h, w):
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        rmin = min(h, w)
        low_mask  = dist < 0.3 * rmin
        high_mask = dist > 0.7 * rmin
        return low_mask, high_mask, (cy, cx)

    # ---- 逐样本可视化与统计 ----
    for i in range(n_samples):
        try:
            # 取该样本的 K 个通道特征：[K,H,W]
            fmap = feat[i, channels].numpy()

            # 计算统一的图像布局：2 行 K 列
            fig, axes = plt.subplots(2, K, figsize=(4.8 * K, 8), constrained_layout=True)
            if K == 1:
                axes = np.array([[axes[0]], [axes[1]]])  # 统一 2xK 索引

            # === 统计容器 ===
            ch_stats = []  # 每通道一个 dict
            # 为了统一颜色范围，可选两种策略：
            #   1) 按通道各自归一化（对比结构更清晰）
            #   2) 也可以先遍历求全通道的全局分位数，再统一 vmin/vmax（需要可比性时可改下面策略）
            # 这里按通道各自归一化（与你现有函数风格一致）

            # 频域掩码（对 H,W 固定，因为本样本内所有通道同尺寸）
            low_mask, high_mask, (cy, cx) = make_freq_masks(H, W)

            # 用于计算通道间相关（空间 & 频谱）
            # 空间：直接用 fmap[k] 拉平
            # 频谱：用 log(PSD+eps) 拉平
            spatial_mat = np.zeros((K, H * W), dtype=np.float64)
            spectral_mat = np.zeros((K, H * W), dtype=np.float64)

            # === 遍历通道 ===
            for j, ch in enumerate(channels):
                arr = fmap[j]  # [H,W]

                # --- 空间统计 ---
                mean_ = float(np.mean(arr))
                std_  = float(np.std(arr))
                min_  = float(np.min(arr))
                max_  = float(np.max(arr))
                l2_   = float(np.sqrt(np.sum(arr**2)) + 1e-12)

                # --- 归一化（用于显示） ---
                if norm_mode == 'percentile':
                    lo = np.percentile(arr, p_low)
                    hi = np.percentile(arr, p_high)
                    if hi <= lo:
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

                # --- 频域 ---
                fft2c = fftshift(fft2(arr))
                psd   = np.abs(fft2c)**2
                log_psd = np.log10(psd + 1e-10)

                # 主频（相对中心的半径）
                max_idx = np.unravel_index(np.argmax(psd), psd.shape)
                dom_freq = float(np.sqrt((max_idx[0] - cy)**2 + (max_idx[1] - cx)**2))

                # 频谱能量比例（本通道 vs 全部通道的总能量需后面再统一）
                # 先记录本通道能量，稍后归一化
                total_energy_ch = float(np.sum(psd))
                low_e  = float(np.sum(psd[low_mask]))
                high_e = float(np.sum(psd[high_mask]))
                hf_ratio = float(high_e / (low_e + 1e-12))

                # --- 可视化：第一行空间，第二行频谱 ---
                im_spatial = axes[0, j].imshow(vis_img, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                axes[0, j].set_title(f'Encoder Feature | sample {i} | ch {ch}\n'
                                     f'{range_note}\nμ={mean_:.3e}, σ={std_:.3e}, L2={l2_:.2e}')
                axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
                plt.colorbar(im_spatial, ax=axes[0, j], fraction=0.046, pad=0.04).set_label('Normalized activation')

                # 为增强对比，将频谱的 vmin/vmax 按该样本所选通道共同范围统一；先暂画，稍后再统一范围（需二次设定）
                im_freq = axes[1, j].imshow(log_psd, cmap='viridis', aspect='auto')
                axes[1, j].set_title(f'Power Spectrum (log10)\nDomFreq={dom_freq:.1f}, HF/LF={hf_ratio:.3f}')
                axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
                plt.colorbar(im_freq, ax=axes[1, j], fraction=0.046, pad=0.04).set_label('log10(PSD)')

                # 收集统计
                ch_stats.append({
                    'channel': int(ch),
                    'spatial': {
                        'mean': mean_, 'std': std_, 'min': min_, 'max': max_, 'l2': l2_
                    },
                    'spectral': {
                        'psd_sum': total_energy_ch,
                        'psd_mean': float(np.mean(psd)),
                        'psd_max': float(np.max(psd)),
                        'dominant_freq': dom_freq,
                        'low_energy': low_e,
                        'high_energy': high_e,
                        'hf_ratio': hf_ratio
                    }
                })

                # 准备相关矩阵数据
                spatial_mat[j, :]  = arr.reshape(-1)
                spectral_mat[j, :] = log_psd.reshape(-1)

            # --- 统一频谱颜色范围（提升通道间可比性） ---
            # 取本样本 K 个通道的 log_psd 联合分位数范围
            # 为简单复用：重新计算一次 log_psd 的全局范围
            #（也可在上面循环时缓存每个通道的 log_psd）
            all_log_psd = []
            for j in range(K):
                arr = fmap[j]
                log_psd = np.log10(np.abs(fftshift(fft2(arr)))**2 + 1e-10)
                all_log_psd.append(log_psd)
            all_log_psd = np.stack(all_log_psd, axis=0)  # [K,H,W]
            vmin_freq = float(np.percentile(all_log_psd, 1.0))
            vmax_freq = float(np.percentile(all_log_psd, 99.0))
            if vmax_freq <= vmin_freq:
                vmax_freq = vmin_freq + 1e-6
            # 重设第二行图像的 clim
            for j in range(K):
                im = axes[1, j].images[0]
                im.set_clim(vmin=vmin_freq, vmax=vmax_freq)

            # --- 计算能量比例（通道在本样本所选通道中的份额） ---
            total_energy_selected = float(sum(cs['spectral']['psd_sum'] for cs in ch_stats)) + 1e-12
            for cs in ch_stats:
                cs['spectral']['energy_ratio_selected'] = float(cs['spectral']['psd_sum'] / total_energy_selected)

            # --- 通道间相关矩阵（空间域 / 频谱域） ---
            # 使用 np.corrcoef，得到 KxK
            def safe_corrcoef(mat):
                try:
                    C = np.corrcoef(mat)
                    if np.isnan(C).any():
                        # 若出现 NaN，用零替换（例如某通道恒常值）
                        C = np.nan_to_num(C, nan=0.0, posinf=0.0, negative_inf=0.0)
                except Exception:
                    C = np.zeros((mat.shape[0], mat.shape[0]), dtype=np.float64)
                return C

            spatial_corr = safe_corrcoef(spatial_mat)
            spectral_corr = safe_corrcoef(spectral_mat)

            # --- 保存/日志 ---
            png_path = os.path.join(save_dir, f'encoded_sample_{i}_ch_{"-".join(map(str,channels))}.png')
            tb_tag   = f"{log_prefix}/sample_{i}/encoded_composed"
            wb_key   = f"{log_prefix}/sample_{i}/encoded_figure"

            if '_log_figure' in globals():
                _log_figure(fig, png_path,
                            tb_writer=tb_writer, tb_tag=tb_tag, step=global_step,
                            wandb_run=wandb_run, wb_key=wb_key, dpi=300)
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

            # --- 写 TXT 摘要（与现有风格一致） ---
            txt_path = os.path.join(save_dir, f'encoded_analysis_sample_{i}.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 90 + "\n")
                f.write(f"Encoder 特征可视化与频域分析 - 样本 {i}\n")
                f.write("=" * 90 + "\n\n")
                f.write(f"通道选择: {channels}  (mode={selection}, topk={K})\n")
                f.write(f"可视化归一化: {norm_mode} (p_low={p_low}, p_high={p_high})\n\n")
                # 每通道
                for cs in ch_stats:
                    ch = cs['channel']
                    s  = cs['spatial']
                    sp = cs['spectral']
                    f.write(f"[Channel {ch}]\n")
                    f.write(f"  空间: mean={s['mean']:.6e}, std={s['std']:.6e}, "
                            f"min={s['min']:.6e}, max={s['max']:.6e}, L2={s['l2']:.6e}\n")
                    f.write(f"  频谱: psd_mean={sp['psd_mean']:.6e}, psd_max={sp['psd_max']:.6e}, "
                            f"dom_freq={sp['dominant_freq']:.3f}, HF/LF={sp['hf_ratio']:.6f}, "
                            f"energy_ratio_selected={sp['energy_ratio_selected']:.6f}\n\n")

                # 通道间相关
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

            # --- 终端打印摘要（与你之前风格相同） ---
            print(f"[visualize_encoded] Sample {i} done.")
            print(f"  Selected channels: {channels}")
            for cs in ch_stats:
                ch = cs['channel']; sp = cs['spectral']; s = cs['spatial']
                print(f"    ch{ch:>3} | μ={s['mean']:.3e}, σ={s['std']:.3e}, "
                      f"L2={s['l2']:.2e} | domF={sp['dominant_freq']:.1f}, HF/LF={sp['hf_ratio']:.3f}, "
                      f"E%={sp['energy_ratio_selected']:.3f}")
            print(f"  Spatial Corr (KxK): min={spatial_corr.min():.3f}, max={spatial_corr.max():.3f}")
            print(f"  Spectral Corr (KxK): min={spectral_corr.min():.3f}, max={spectral_corr.max():.3f}")
            print(f"  Saved: {png_path} | {txt_path}")
            print("-" * 90)

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
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(save_dir, 'encoded_vis_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[visualize_encoded] Done. samples={n_samples}, channels={channels}, savedir='{save_dir}'")
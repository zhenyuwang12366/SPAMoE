import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import re
import pandas as pd
import torch
import os
import time

def plot_loss_curve(log_file, save_path=None):
    """
    从日志文件中解析并绘制 Train Loss 和 Val Loss 曲线
    """
    with open(log_file, "r") as f:
        text = f.read()
    
    pattern = re.compile(
        r"""^                                   # 行首
            \s*(\d+)\s*\|\s*                    # Epoch（整数）
            (\d+(?:\.\d+)?)\s*\|\s*             # Train Loss（浮点）
            (\d+(?:\.\d+)?)\s*\|\s*             # Val Loss（浮点）
            (\d+(?:\.\d+)?)\s*\|\s*             # MAE（浮点）
            (\d+(?:\.\d+)?)\s*\|\s*             # MSE（浮点）
            ([+-]?\d+(?:\.\d+)?)\s*\|           # PSNR（可为负）
            \s*$                                # 行尾
        """,
        re.MULTILINE | re.VERBOSE
    )

    rows = [m.groups() for m in pattern.finditer(text)]
    
    if not rows:
        raise ValueError("日志格式不匹配，请检查 log_file 格式")
    
    df = pd.DataFrame(rows, columns=["Epoch", "Train Loss", "Val Loss", "MAE", "MSE", "PSNR"]).astype(float)
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"保存曲线到 {save_path}")
    else:
        plt.show()
        
def visualize_results(inputs, targets, predictions, save_dir='./results', max_samples=4):
    """可视化地震数据和预测结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    n_samples = min(inputs.shape[0], max_samples)
    
    for i in range(n_samples):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)  
        
        if len(inputs[i].shape) > 2:
            input_data = inputs[i, 0].cpu().numpy()
        else:
            input_data = inputs[i].cpu().numpy()
        im0 = axes[0].imshow(input_data, cmap='viridis', aspect='auto')
        axes[0].set_title('inputs data')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        target_map = targets[i, 0].cpu().numpy()
        pred_map   = predictions[i, 0].cpu().numpy()
        vmin, vmax = float(target_map.min()), float(target_map.max())
        shared_norm = Normalize(vmin=vmin, vmax=vmax)

        im1 = axes[1].imshow(target_map, cmap='jet', norm=shared_norm, aspect='auto')
        axes[1].set_title('targets model')

        im2 = axes[2].imshow(pred_map, cmap='jet', norm=shared_norm, aspect='auto')
        axes[2].set_title('predictions model')

        cbar = fig.colorbar(im1, ax=axes[1:3], fraction=0.046, pad=0.04)
        cbar.set_label('Model amplitude')

        save_path = os.path.join(save_dir, f'sample_{i}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def analyze_fourier_domain(inputs, targets, predictions, save_dir='./results', max_samples=4):
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
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft2, fftshift
    
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

"""
使用MOE（Mixture of Experts）架构训练地震数据的神经算子模型
支持分布式训练
"""

import os
import sys
import math
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import re
import copy
from contextlib import nullcontext
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Union, Optional, Callable
import argparse
import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader, random_split, Subset, DistributedSampler
from torchvision.transforms import Compose
import transforms as T
import time
import datetime
from collections import defaultdict, OrderedDict
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralop.models import MOEOperator, ExpertFactory
from neuralop.training import Trainer, setup
from neuralop.training.torch_setup import setup
from neuralop.data.datasets import SeismicDataset, create_seismic_dataloader
from neuralop.utils import get_wandb_api_key, count_model_params
from config.seismic_moe_config import SeismicMOEConfig
import neuralop.mpu.comm as comm
from scripts.scheduler import WarmupMultiStepLR
import neuralop.mpu.comm as comm
print("-----------------------------------------------------------")

class SeismicMetrics:
    """
    地震数据评估指标
    """
    @staticmethod
    def calculate_mse(pred, target):
        """计算均方误差"""
        return F.mse_loss(pred, target).item()
    
    @staticmethod
    def calculate_mae(pred, target):
        """计算平均绝对误差"""
        return F.l1_loss(pred, target).item()
    
    @staticmethod
    def calculate_psnr(pred, target, data_range=None):
        """计算峰值信噪比"""
        # 确保张量在CPU上
        if pred.is_cuda:
            pred = pred.detach().cpu()
        if target.is_cuda:
            target = target.detach().cpu()
            
        if data_range is None:
            data_range = target.max() - target.min()
        
        # 确保data_range也在CPU上
        if isinstance(data_range, torch.Tensor) and data_range.is_cuda:
            data_range = data_range.detach().cpu()
        
        mse = F.mse_loss(pred, target).item()
        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
        return psnr
    
class EarlyStopping:
    """
    监控 val 指标，当连续 patience 次没有超过 min_delta 的改善时，触发早停。
    mode='min'：指标越小越好（如 val_loss）
    """
    def __init__(self, patience=20, min_delta=0.0, warmup_epochs=0, mode='min'):
        assert mode in ('min', 'max')
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup_epochs = int(warmup_epochs)
        self.mode = mode

        self.best = math.inf if mode == 'min' else -math.inf
        self.num_bad = 0
        self.should_stop = False
        self.best_epoch = -1

    def _is_improved(self, value):
        if self.mode == 'min':
            return value < (self.best - self.min_delta)
        else:
            return value > (self.best + self.min_delta)

    def step(self, value, epoch):
        """
        返回是否应停止（仅供主进程用于判定）。内部更新最佳值和坏轮计数。
        """
        # warmup 阶段不计入早停
        if epoch < self.warmup_epochs:
            return False

        if self._is_improved(value):
            self.best = value
            self.best_epoch = epoch
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
        return self.should_stop

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
    
    # 限制样本数
    n_samples = min(inputs.shape[0], max_samples)
    
    for i in range(n_samples):
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 绘制输入地震数据（选择第一个通道）
        # 如果输入数据是多维的，我们只显示第一个通道
        if len(inputs[i].shape) > 2:
            input_data = inputs[i, 0].cpu().numpy()
        else:
            input_data = inputs[i].cpu().numpy()
        im0 = axes[0].imshow(input_data, cmap='viridis')
        axes[0].set_title('inputs data')
        plt.colorbar(im0, ax=axes[0])
        
        # 绘制目标（速度图或模型）
        im1 = axes[1].imshow(targets[i, 0].cpu().numpy(), cmap='jet')
        axes[1].set_title('targets model')
        plt.colorbar(im1, ax=axes[1])
        
        # 绘制预测（速度图或模型）
        im2 = axes[2].imshow(predictions[i, 0].cpu().numpy(), cmap='jet')
        axes[2].set_title('predictions model')
        plt.colorbar(im2, ax=axes[2])
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i}.png'), dpi=300)
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

        
def safe_random_split(dataset_size, ratios : list):
        assert abs(sum(ratios) - 1.0) < 1e-6, "ratios必须加起来为1"

        total = dataset_size
        raw_sizes = [r * total for r in ratios]
        sizes = [int(x) for x in raw_sizes]
        deficit = total - sum(sizes)

        # 第一步：确保每个 size 至少为 1
        for i in range(len(sizes)):
            if sizes[i] == 0 and deficit > 0:
                sizes[i] += 1
                deficit -= 1

        # 第二步：按最大小数部分补充剩余样本
        frac_with_index = sorted(
            [(raw - int(raw), i) for i, raw in enumerate(raw_sizes)],
            reverse=True
        )

        i = 0
        while deficit > 0:
            sizes[frac_with_index[i % len(sizes)][1]] += 1
            deficit -= 1
            i += 1

        assert sum(sizes) == total, "最终样本数量不一致"
        train_size = sizes[0]
        val_size = sizes[1]
        return train_size, val_size

@torch.no_grad()
def _evaluate_one_epoch(
    model,
    val_loader,
    device,
    criterion,
    metrics_module,  # 需有 calculate_psnr(pred, tgt)
):
    model.eval()
    val_loss = 0.0
    mse_sum, mae_sum, psnr_sum = 0.0, 0.0, 0.0

    for batch in val_loader:
        inputs  = batch['input'].to(device, non_blocking=True)
        targets = batch['output'].to(device, non_blocking=True)

        preds, aux_loss = model(inputs)

        # 兼容 criterion 返回 (loss, loss_g1v, loss_g2v)
        loss_tuple = criterion(preds, targets)
        if isinstance(loss_tuple, (tuple, list)) and len(loss_tuple) >= 3:
            loss, loss_g1v, loss_g2v = loss_tuple[0], loss_tuple[1], loss_tuple[2]
        else:
            loss, loss_g1v, loss_g2v = loss_tuple, torch.tensor(0.), torch.tensor(0.)

        val_loss += loss.item()
        mse_sum  += loss_g2v.item()
        mae_sum  += loss_g1v.item()
        psnr_sum += metrics_module.calculate_psnr(preds, targets)

    n = max(1, len(val_loader))
    return {
        "val_loss": val_loss / n,
        "mse": mse_sum / n,
        "mae": mae_sum / n,
        "psnr": psnr_sum / n
    }


def train_one_epoch(
    *args,
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    epoch: int,
    config,
    is_logger: bool,
    log_file: Optional[str],
    results_dir,
    # MoE负载均衡因子
    coef: float = 0.01,
    # 学习率调度器
    lr_scheduler=None,
    scheduler_step_mode: str = "per_step",   # "per_step" 或 "per_epoch"
    # 梯度累计
    accum_steps: int = 1,
    # 可视化 & 反归一化
    vis_now: bool = False,                   # 是否在本 epoch 可视化
    visualize_results: Optional[Callable] = None,
    input_inverse_transform: Optional[Callable] = None,
    output_inverse_transform: Optional[Callable] = None,
    # WandB
    use_wandb: bool = False,
    wandb_module=None,
    # 早停
    early_stopper=None,                      # 需有 step(val_loss, epoch)->bool
    # 最佳模型保存
    best_val_loss: float = float("inf"),
    best_model_path: Optional[str] = None,
    best_expert_path: Optional[str] = None,
    experts_name: Optional[list] = None,
    experts_name_str: Optional[str] = None,
    data_dict: Optional[dict] = None,
    # 其他工具
    metrics_module=None,
    tqdm_module=None,   # 传入 tqdm（避免在函数内硬依赖）
    profile_timing: bool = False,            # 是否记录耗时
    **kwargs,
):
    """
    进行一个 epoch 的完整训练与验证，返回 (stats_dict, best_val_loss, stop_flag)
    - 等效全局 batch = per_gpu_batch * world_size * accum_steps
    - 若使用 DDP，前 accum_steps-1 次 micro step 用 no_sync() 以减少通信
    - 调度器：per_step 在“优化步”后 step；per_epoch 在 epoch 末 step
    """
    assert metrics_module is not None, "metrics_module 需提供 calculate_psnr(pred, tgt)"
    tqdm = tqdm_module.tqdm if tqdm_module is not None else None

    start_time = time.time()
    model.train()
    running_train_loss = 0.0
    micro_count = 0
    optim_count = 0
    num_steps = len(train_loader)
    
    # router type判断
    router_type = model.module.router_type if hasattr(model, "module") else model.router_type
    if "adamv" == router_type:
        router = model.module.router if hasattr(model, "module") else model.router
        assert hasattr(router, "step_validation"), "adamv router must impl. function step_validation"
    
    # DDP 判断
    is_ddp = hasattr(model, "no_sync")

    # 分布式 sampler 设 epoch
    if getattr(config, "distributed", None) and getattr(config.distributed, "use_distributed", False):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
 
    optimizer.zero_grad(set_to_none=True)

    val_loss = float("inf")
    
    pbar_iter = train_loader
    if tqdm is not None:
        pbar_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False, disable=not is_logger)

    for step, batch in enumerate(pbar_iter):
        inputs  = batch['input'].to(device, non_blocking=True)
        targets = batch['output'].to(device, non_blocking=True)

        last_micro = ((step + 1) % accum_steps == 0) or ((step + 1) == num_steps)
        sync_ctx = (model.no_sync() if (is_ddp and not last_micro) else nullcontext())

        with sync_ctx:
            preds, aux_loss = model(inputs)
            loss_tuple = criterion(preds, targets)
            loss = loss_tuple[0] if isinstance(loss_tuple, (tuple, list)) else loss_tuple
            loss += coef * aux_loss
            # —— 核心：为梯度累计缩放 loss，保证等效大 batch ——
            loss = loss / accum_steps
            loss.backward()
            running_train_loss += loss.item()
            micro_count += 1

        if last_micro:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optim_count += 1

            # 学习率调度（按步）
            if lr_scheduler is not None and scheduler_step_mode == "per_step":
                lr_scheduler.step()

        # 进度条显示
        if is_logger and tqdm is not None:
            pbar_iter.set_postfix({"train_loss": f"{loss.item():.6f}"})

    # —— 训练集 loss（micro-step 平均）——
    avg_train_loss = running_train_loss / max(1, micro_count)

    # —— 验证 —— #
    val_stats = _evaluate_one_epoch(model, val_loader, device, criterion, metrics_module)
    val_loss = val_stats["val_loss"]
    if router_type == 'adamv':
        signal = router.step_validation(val_loss)
        # 多进程之间同步信号
        if is_ddp:
            signal = torch.tensor([1 if signal == "should_break" else 0]
                                  , device=device, dtype=torch.int64) # bool信号常用int64
            dist.all_reduce(signal, op=dist.ReduceOp.MAX)
            should_break = bool(signal.item())
        else:
            should_break = (signal == "should_break")
        
        if should_break:
            router.k = max(1, router.k - 1)
            router.fixed = True
            
            if is_ddp:
                k_tensor = torch.tensor([router.k], device=device)
                dist.broadcast(k_tensor, src=0) # broadcast包含同步原语
                router.k = int(k_tensor.item())
        
            if is_logger:
                print(f'epoch: {epoch} AES probe failed -> fix top_k = {router.k}')
                    
    # —— 日志输出 & WandB —— #
    if is_logger and log_file is not None:
        with open(log_file, "a") as f:
            f.write(
                f"    {epoch+1}    |    {avg_train_loss:.6f}    |    {val_loss:.6f}    |    "
                f"{val_stats['mae']:.6f}    |    {val_stats['mse']:.6f}    |    {val_stats['psnr']:.6f}    |\n"
            )

    if use_wandb and wandb_module is not None:
        wandb_log = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val/psnr": val_stats["psnr"],
            "val/mse": val_stats["mse"],
            "val/mae": val_stats["mae"],
            "optim_steps_in_epoch": optim_count,
        }
        wandb_module.log(wandb_log)

    # —— 保存最佳模型（仅主进程）—— #
    if is_logger and (val_loss < best_val_loss):
        best_val_loss = val_loss
        model_to_save = model.module if (getattr(config, "distributed", None) and getattr(config.distributed, "use_distributed", False) and hasattr(model, "module")) else model

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': {"psnr": val_stats["psnr"], "mse": val_stats["mse"], "mae": val_stats["mae"]},
            'data_dict': data_dict
        }, best_model_path)

        if experts_name is not None and len(experts_name) == 1 and best_expert_path is not None:
            # 仅示例：若你的模型结构中存在 experts[0]
            if hasattr(model_to_save, "experts") and len(model_to_save.experts) > 0:
                torch.save({
                    'expert_state_dict': model_to_save.experts[0].state_dict()
                }, best_expert_path)

    # —— 打印本 epoch 概要（仅主进程）—— #
    if is_logger:
        print(f"Epoch {epoch+1}/{config.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val   Loss: {val_loss:.6f}")
        print(f"  PSNR: {val_stats['psnr']:.2f} dB")
        print(f"  MSE : {val_stats['mse']:.6f}")
        print(f"  MAE : {val_stats['mae']:.6f}")
        print(f"  AuxLoss : {aux_loss:.2f}")

    # —— 可视化（仅主进程 & 触发时）—— #
    if is_logger and vis_now and visualize_results is not None:
        vis_batch = next(iter(val_loader))
        inputs = vis_batch['input'].to(device, non_blocking=True)
        targets = vis_batch['output'].to(device, non_blocking=True)
        with torch.no_grad():
            preds, _ = model(inputs)

        if input_inverse_transform is not None:
            inputs_v = input_inverse_transform(inputs)
        else:
            inputs_v = inputs

        if output_inverse_transform is not None:
            preds_v = output_inverse_transform(preds)
            targets_v = output_inverse_transform(targets)
        else:
            preds_v, targets_v = preds, targets

        visualize_results(inputs_v, targets_v, preds_v, save_dir=results_dir / f"vis_epoch_{epoch+1}")
        
        # 进行傅里叶域分析
        analyze_fourier_domain(inputs_v, targets_v, preds_v, save_dir=results_dir / f"fourier_analysis_epoch_{epoch+1}")

        if use_wandb and wandb_module is not None:
            # 只示例记录前三个
            for i in range(min(3, inputs_v.shape[0])):
                in_img  = inputs_v[i, 0].detach().float().cpu().numpy()
                tgt_img = (targets_v[i, 0] if targets_v.dim() > 3 else targets_v[i]).detach().float().cpu().numpy()
                prd_img = (preds_v[i, 0]   if preds_v.dim()   > 3 else preds_v[i]).detach().float().cpu().numpy()
                wandb_module.log({
                    f"sample_{i}/input_velocity": wandb_module.Image(in_img),
                    f"sample_{i}/target_seismic": wandb_module.Image(tgt_img),
                    f"sample_{i}/prediction_seismic": wandb_module.Image(prd_img),
                })

    # —— 早停（仅主进程判定，后广播）—— #
    stop_flag = 0
    if getattr(config, "early_stop", False):
        if is_logger and early_stopper is not None:
            if early_stopper.step(val_loss, epoch):
                stop_flag = 1

        if getattr(config, "distributed", None) and getattr(config.distributed, "use_distributed", False):
            device_for_flag = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            flag_tensor = torch.tensor([stop_flag], device=device_for_flag, dtype=torch.int32)
            torch.distributed.broadcast(flag_tensor, src=0)
            stop_flag = int(flag_tensor.item())

        if stop_flag == 1 and is_logger:
            print(f"[EARLY STOP] stop at epoch={epoch+1}, best_val_loss={best_val_loss:.6f}")

    # —— 调度器（按 epoch）—— #
    if lr_scheduler is not None and scheduler_step_mode == "per_epoch":
        lr_scheduler.step()

    # —— 分布式 barrier（可选，与日志输出顺序相关）—— #
    if getattr(config, "distributed", None) and getattr(config.distributed, "use_distributed", False):
        torch.distributed.barrier()

    # —— 耗时 —— #
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if is_logger:
        print('Training time', total_time_str)

    # 返回统计与状态
    stats = {
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": val_stats["val_loss"],
        "psnr": val_stats["psnr"],
        "mse": val_stats["mse"],
        "mae": val_stats["mae"],
        "optim_steps": optim_count,
        "micro_steps": micro_count,
        "time_sec": total_time
    }
    return stats, best_val_loss, stop_flag

def run_training(args):
    """
    训练地震数据的MOE模型
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    """
    # 加载配置
    config = SeismicMOEConfig()
    
    # 更新配置
    # 代码解释：如果用户在命令行中传入了参数 --data_dir，那就用用户的这个路径；否则，就使用默认路径 "/data1/wuruoyu/waveform-inversion"。
    
    # 设置随机种子
    config.distributed.seed = args.seed
    
    # 启用分布式训练
    if args.distributed:
        config.distributed.use_distributed = True
        device, is_logger = setup(config)
    else:
        device, is_logger = setup(config)
    
    local_rank = comm.get_local_rank()
    global_rank = comm.get_global_rank()
    world_size = comm.get_world_size()
    
    if args.data_dir:
        config.data_dir = args.data_dir
    else:
        # 设置默认数据目录为新路径
        config.data_dir = r"/root/autodl-tmp/FWINO/FWINO_data"
    config.output_dir = args.output_dir

        #解释见onenote1
    if args.family:
        config.family = args.family
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.hidden_channels:
        config.hidden_channels = args.hidden_channels
    if args.lr_warmup_epochs:
        config.lr_warmup_epochs = args.lr_warmup_epochs
    if args.weight_decay:
        config.weight_decay = args.weight_decay
    if args.scheduler_gamma:
        config.scheduler_gamma = args.scheduler_gamma
    if args.accum_steps is not None:
        config.accum_steps = args.accum_steps
  
    accum_steps = config.accum_steps
    use_amp = config.use_amp
    
    config.lr_warmup_epochs = int(config.epochs * 0.05)
    
    if is_logger:
        print(f'batch_size:{config.batch_size}')
        print(f'effective_batch_size:{world_size * config.batch_size * config.accum_steps}')
        print(f'epochs:{config.epochs}')
        print(f'learning_rate:{config.learning_rate}')
        print(f'hidden_channels:{config.hidden_channels}')

    # 设置验证集比例
    val_ratio = args.val_ratio if args.val_ratio is not None else 0.2
    
    # 设置WandB日志记录
    if args.use_wandb and is_logger:
        wandb.login(key=get_wandb_api_key())
        wandb_name = f"seismic_moe_{config.family}"
        wandb_init_args = dict(
            config=config,
            name=wandb_name,
            project="seismic_moe",
        )
        wandb.init(**wandb_init_args)
    # FNO config setting
    config.expert_configs[0]['n_modes_height'] = args.FNO_n_modes_height
    config.expert_configs[0]['n_modes_width'] = args.FNO_n_modes_width
    config.expert_configs[0]['n_layers'] = args.FNO_n_layers
    # WNO config setting
    config.expert_configs[1]['n_levels_height'] = args.WNO_n_levels_height
    config.expert_configs[1]['n_levels_width'] = args.WNO_n_levels_width
    config.expert_configs[1]['n_layers'] = args.WNO_n_layers
    config.expert_configs[1]['block_n_layers'] = args.WNO_block_n_layers
    config.expert_configs[1]['dropout_rate'] = args.WNO_dropout_rate
    config.expert_configs[1]['wavelet_type'] = args.wavelet_type
    # MNO config setting
    config.expert_configs[2]['n_scales'] = args.MNO_n_scales
    config.expert_configs[2]['scale_factors'] = args.MNO_scale_factors
    config.expert_configs[2]['n_layers'] = args.MNO_n_layers
    # LNO config setting
    config.expert_configs[3]['n_modes'] = tuple(args.LNO_n_modes)
    config.expert_configs[3]['n_layers'] = args.LNO_n_layers
    
    print(f'FNO:n_modes_height:{config.expert_configs[0]["n_modes_height"]}')
    print(f'FNO:n_modes_width:{config.expert_configs[0]["n_modes_width"]}')
    print(f'FNO:n_layers:{config.expert_configs[0]["n_layers"]}')
    
    # 设置专家数
    config.top_k = args.top_k
    # 选择专家，这里后面的config.expert_configs就是config文件中所创建的字典列表，
    # 当你从命令行输入choose——experts之后，这里的for循环会根据你给定的序号找到对应的专家的字典，并将这个字典放入
    #config.expert_configs列表中
    config.expert_configs = [config.expert_configs[i] for i in args.choose_experts]
    #这里的config.expert_configs就是seismic_moe_config中的“字典列表”，关于“字典列表”结构的解释详见OneNote3
    
    # 训练moe架构
         #下面两段代码的解释见OneNote2,意义是：
         # 在使用 Mixture of Experts 模型时，如果用户已经训练并保存了若干个“专家模型”（模型文件保存在某个文件夹中），那么这两段代码就是要：
         #读取那些 .pt 文件（即每个专家的模型参数）；
         #按照文件名中的编号提取出专家编号；
         #做一致性校验（文件数量、编号是否跟配置匹配）；
    if len(config.expert_configs) > 1 and config.top_k > 1 and args.use_moe and args.use_experts_path:
        # 模型文件夹中的专家 best_expert_{experts_name}_{i}.pt
        save_experts = [
            int(f.split('_')[-1].split('.')[0]) for f in os.listdir(args.use_experts_path)
            if f.split('_')[1] == 'expert' and f.endswith('.pt')
        ]
        #注意，这里输出的save_experts是一个代表专家模型序号的整数列表。详见OneNote2

        # 检测正确性
        if len(config.expert_configs) != len(save_experts):
            raise ValueError(f"模型文件夹中专家个数: {args.use_experts_path} 与选择专家个数不匹配: {len(config.expert_configs)}")

        for i in config.expert_configs:
            if(i not in save_experts):
                raise ValueError(f"选择的专家: {i} 无法与模型存储文件夹中的专家匹配")

        config.use_moe = True
        config.use_experts_path = args.use_experts_path

    # 这两行代码解释见OneNote3，不过我想知道为什么这两段代码在我们单个模型训练过程中没有起作用
    # 修复：使用enumerate来获取正确的索引，因为config.expert_configs已经被重新排序
    experts_name = []
    for idx, expert_config in enumerate(config.expert_configs):
        if 'domain_type' in expert_config:
            experts_name.append(f"{expert_config['domain_type']}_{args.choose_experts[idx]}")
        else:
            experts_name.append(f"{expert_config['type']}_{args.choose_experts[idx]}")
    experts_name_str = '_'.join(experts_name)
    config.output_dir = os.path.join(config.output_dir, experts_name_str)   
    
    # 设置损失函数加权系数
    config.lambda_g1v = args.lambda_g1v
    config.lambda_g2v = args.lambda_g2v
    
    # 设置路由形式
    if args.router_type:
        config.router_type = args.router_type
    
    # 设置专家组间融合方式
    if args.fusion_type:
        config.fusion_type = args.fusion_type
    
    # 设置强弱专家组内融合方式
    if args.s_processor_type:
        config.s_processor_type = args.s_processor_type
    if args.w_processor_type:
        config.w_processor_type = args.w_processor_type
        
    # 设置强弱激活参数
    if args.beta:
        config.beta = args.beta
    
    # 设置细化种类
    if args.is_specific:
        config.is_specific = args.is_specific
    
    # 设置是否使用分组专家网络
    if args.is_classier:
        config.is_classier = args.is_classier
    
    # 判断is_specific与选择family是否匹配
    if config.is_specific and config.family not in ['curve_vel', 'flat_vel', 'curve_fault', 'flat_fault', 'style_style']:
        raise ValueError(f"{config.family} 与 {config.is_specific} 不匹配")
    
    #-------------- 设置完毕 -----------#
    # 创建完整数据集
    full_dataset = SeismicDataset(
        data_dir=config.data_dir,
        family=config.family,
        is_specific=config.is_specific,
        split='train',
    )
    
    # 分割数据集为训练集和验证集
    # 获取数据集大小
    dataset_size = len(full_dataset)
    train_size, val_size = safe_random_split(dataset_size, [1-val_ratio, val_ratio])
    # val_size = int(dataset_size * val_ratio)
    # train_size = dataset_size - val_size
    
    if is_logger:
        print(f"数据集总大小: {dataset_size}")
        print(f"训练集大小: {train_size}")
        print(f"验证集大小: {val_size}")
    
    # 使用random_split分割数据集
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    data_dict = full_dataset.getStats()
    
    # 验证训练集和验证集没有重叠
    if is_logger:
        train_indices_set = set(train_dataset.indices)
        val_indices_set = set(val_dataset.indices)
        overlap = train_indices_set.intersection(val_indices_set)
        if overlap:
            print(f"警告：训练集和验证集有{len(overlap)}个重叠样本！")
        else:
            print("验证成功：训练集和验证集没有重叠样本")
        
        # 确认数据集大小
        assert len(train_dataset) == train_size, f"训练集大小不匹配：{len(train_dataset)} vs {train_size}"
        assert len(val_dataset) == val_size, f"验证集大小不匹配：{len(val_dataset)} vs {val_size}"
    
    # 创建数据处理器
    from neuralop.data.datasets.seismic_dataset import SeismicDataProcessor
    input_transform = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(data_dict['input_min'], k=args.k), T.log_transform(data_dict['input_max'], k=args.k))
    ]) # data
    output_transform = Compose([
        T.MinMaxNormalize(data_dict['output_min'], data_dict['output_max'])
    ]) # model
    input_inverse_transform = Compose([
        T.InverseMinMaxNormalize(T.log_transform(data_dict['input_min'], k=args.k), T.log_transform(data_dict['input_max'], k=args.k)),
        T.InverseLogTransform(k=args.k)
    ])
    output_inverse_transform = Compose([
        T.InverseMinMaxNormalize(data_dict['output_min'], data_dict['output_max'])
    ])
    data_processor = SeismicDataProcessor(
        input_transform=input_transform,
        output_transform=output_transform,
        channel_dim=config.channel_dim
    )
    
    # 应用变换到训练集和验证集
    train_dataset_with_transform = TransformedSubset(train_dataset, data_processor)
    val_dataset_with_transform = TransformedSubset(val_dataset, data_processor)
    
    # 创建数据加载器
    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset_with_transform, 
            num_replicas=world_size, 
            rank=local_rank,
            drop_last = True,
            shuffle = True
        )
        train_loader = DataLoader(
            train_dataset_with_transform,
            sampler=train_sampler,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=int(args.num_workers/2),
            pin_memory=True,
            persistent_workers=True
        )

        print(f'prefetch_factor={train_loader.prefetch_factor}')
        
        val_sampler = DistributedSampler(
            val_dataset_with_transform, 
            num_replicas=world_size, 
            rank=local_rank,
            drop_last = True
        )
        val_loader = DataLoader(
            val_dataset_with_transform,
            sampler = val_sampler,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=int(args.num_workers/2),
            pin_memory=True,
            persistent_workers=True
        )
    else:
        train_loader = DataLoader(
            train_dataset_with_transform,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        print(f'prefetch_factor={train_loader.prefetch_factor}')
        
        val_loader = DataLoader(
            val_dataset_with_transform,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    # 检查数据形状
    if is_logger:
        sample_batch = next(iter(train_loader))
        input_shape = sample_batch['input'].shape
        output_shape = sample_batch['output'].shape
        print(f"输入张量形状: {input_shape}")
        print(f"输出张量形状: {output_shape}")
        
        # 检查输入是否为速度图/模型（应该是2D或3D数据）
        if len(input_shape) < 3 or len(input_shape) > 4:
            print(f"警告：输入形状不符合预期，应为3D或4D张量，实际为{len(input_shape)}D")
        
        # 检查输出是否为地震数据（应该是多维数据）
        if len(output_shape) < 3:
            print(f"警告：输出形状不符合预期，应为3D或更高维张量，实际为{len(output_shape)}D")
    
    # 获取实际的输入通道数
    sample_batch = next(iter(train_loader))
    in_channels = sample_batch['input'].shape[1]  # 获取通道维度大小
    config.in_channels = in_channels  # 更新配置
    
    # 检查专家配置
    if is_logger:
        print(f"更新后的输入通道数: {config.in_channels}")
        print(f"输出通道数: {config.out_channels}")
        print(f"隐藏通道数: {config.hidden_channels}")
        print(f"专家数量: {len(config.expert_configs)}")
        
        # 检查每个专家配置
        for i, expert_config in enumerate(config.expert_configs):
            print(f"专家 {i+1} 类型: {expert_config.get('type', 'unknown')}")
            if 'in_channels' in expert_config and expert_config['in_channels'] != in_channels:
                print(f"警告：专家 {i+1} 的输入通道数 {expert_config['in_channels']} 与实际输入通道数 {in_channels} 不匹配")
    
    if config.use_moe and config.use_experts_path:
        experts = load_moe_experts(
            expert_configs=config.expert_configs,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels,
            model_path=config.use_experts_path,
            is_specific=False,
            map_location=device,
            type_dict=config.type_id
        )
    else:
        # 创建专家模型
        experts = ExpertFactory.create_expert_ensemble(
            expert_configs=config.expert_configs,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels
        )
    
    # 创建MOE模型
    model = MOEOperator(
        experts=experts,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        hidden_channels=config.hidden_channels,
        top_k=config.top_k,
        noisy_gating=config.noisy_gating,
        fusion_type=config.fusion_type,
        router_hidden_dim=config.router_hidden_dim,
        is_logger=is_logger,
        router_type=config.router_type,
        s_processor_type = config.s_processor_type,
        w_processor_type = config.w_processor_type,
        beta = config.beta,
        is_specific = config.is_specific,
        is_classier = config.is_classier,
    )
    
    # 移动模型到设备
    model = model.to(device)
    
    # 使用分布式数据并行
    if config.distributed.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = DDP(
            model, device_ids=[device.index], 
            output_device=device.index, 
            static_graph=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
 
    # Scale lr according to effective batch size
    if config.distributed.use_distributed and world_size > 2:
        lr = config.learning_rate * math.sqrt(world_size)
    else:
        lr = config.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    
    # Convert scheduler to be per iteration instead of per epoch
    warmup_iters = config.lr_warmup_epochs * len(train_loader)
    lr_milestones = [len(train_loader) * m for m in config.milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=config.scheduler_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)
     
    # Define loss function
    l1loss = nn.L1Loss() # MAE
    l2loss = nn.MSELoss()
    def criterion(pred, gt):
        loss_g1v = l1loss(pred, gt)
        loss_g2v = l2loss(pred, gt)
        loss = config.lambda_g1v * loss_g1v + config.lambda_g2v * loss_g2v
        return loss, loss_g1v, loss_g2v
    
    # 损失函数
    # criterion = F.mse_loss  # 使用均方误差损失
    
    # 创建结果目录
    results_dir = Path(config.output_dir) / f"seismic_moe_{config.family}"
    if is_logger:
        results_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练日志
    log_file = results_dir / "training_log.txt"
    if is_logger:
        with open(log_file, "w") as f:
            f.write(f"    Epoch    |    Train Loss    |    Val Loss    |    MAE    |    MSE    |    PSNR    |\n")
    
    # 最佳模型保存
    best_val_loss = float("inf")
    best_model_path = results_dir / f"best_model_{experts_name_str}.pt"
    if len(experts_name) == 1:
        best_expert_path = results_dir / f"best_expert_{experts_name_str}.pt"
    
    # 指标计算器
    metrics = SeismicMetrics()
    
    if config.early_stop:
        # ---- 早停参数（可放到 config / args）----
        early_patience = getattr(config, "early_stop_patience", 20)
        early_min_delta = getattr(config, "early_stop_min_delta", 0.0)  # 例如 0.001
        early_warmup   = getattr(config, "early_stop_warmup_epochs", 10)

        early_stopper = EarlyStopping(
            patience=early_patience,
            min_delta=early_min_delta,
            warmup_epochs=early_warmup,
            mode="min"  # 监控 val_loss
        )
    
    # 记录参数数量
    if is_logger:
        if config.distributed.use_distributed:
            # 对于DDP模型，需要获取原始模型
            n_params = count_model_params(model.module)
        else:
            n_params = count_model_params(model)
        
        print(f"模型参数数量: {n_params}")
        
        if args.use_wandb:
            wandb.log({"n_params": n_params})

    # ========= Resume checkpoint if provided =========
    start_epoch = 0
    if hasattr(args, "resume_path") and args.resume_path is not None and os.path.exists(args.resume_path):
        checkpoint = torch.load(args.resume_path, map_location=device, weights_only=False)

        # 加载模型参数
        if config.distributed.use_distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载归一化参数
        data_dict = checkpoint['data_dict']

        # 记录当前开始的epoch（从上一次的下一个开始）
        start_epoch = checkpoint['epoch'] + 1

        if is_logger:
            print(f"==> 成功从 {args.resume_path} 恢复模型")
            print(f"==> 从第 {start_epoch} 个 epoch 继续训练")

    else:
        if is_logger:
            print("未提供 resume 路径，或路径无效，将从头开始训练。")
    
    scaler = torch.amp.GradScaler(device=device,enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

#以上全是准备工作，下面是核心循环
    # 训练循环
    for epoch in range(start_epoch, config.epochs):
        vis_now = (is_logger and ((epoch + 1) % args.vis_freq == 0))
        stats, best_val_loss, stop_flag = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            config=config,
            is_logger=is_logger,
            log_file=log_file,
            results_dir=results_dir,
            lr_scheduler=lr_scheduler,
            scheduler_step_mode=("per_step" if config.use_onecycle else "per_epoch"),
            accum_steps=config.accum_steps,
            vis_now=vis_now,
            visualize_results=visualize_results,
            input_inverse_transform=input_inverse_transform,
            output_inverse_transform=output_inverse_transform,
            use_wandb=args.use_wandb,
            wandb_module=wandb if args.use_wandb else None,
            early_stopper=early_stopper if config.early_stop else None,
            best_val_loss=best_val_loss,
            best_model_path=best_model_path,
            best_expert_path=best_expert_path,
            experts_name=experts_name,
            experts_name_str=experts_name_str,
            data_dict=data_dict,
            metrics_module=metrics,
            tqdm_module=tqdm,
            profile_timing=args.profile_timing,
        )
        if stop_flag == 1:
            break
    
    if is_logger:
        plot_loss_curve(log_file, save_path=results_dir)
        
    return model, best_val_loss

def run_overfit_one_sample(args):
    """
    单样本过拟合：用训练集中的 1 个样本训练到极低误差。
    - 损失：L1 + Sobel梯度一致性(逐步加权) + 极小MSE(仅稳定/提升PSNR)
    - 评估：PSNR(反归一化)、corr(pred,input)、grad_norm、LR 变化等
    """
    import copy
    from itertools import cycle
    import torch.nn.functional as F
    from neuralop.training.torch_setup import setup as setup_env
    from neuralop.data.datasets.seismic_dataset import SeismicDataProcessor

    # ===== 0) 配置与设备 =====
    config = SeismicMOEConfig()
    if args.data_dir:
        config.data_dir = args.data_dir
    else:
        config.data_dir = r"/root/autodl-tmp/FWINO/FWINO_data"
    if args.family:
        config.family = args.family
    if args.hidden_channels:
        config.hidden_channels = args.hidden_channels
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.top_k:
        config.top_k = args.top_k

    # 选用的专家
    config.expert_configs[0]['n_modes_height'] = args.FNO_n_modes_height
    config.expert_configs[0]['n_modes_width']  = args.FNO_n_modes_width
    config.expert_configs[0]['n_layers']       = args.FNO_n_layers
    config.expert_configs[1]['n_levels_height']= args.WNO_n_levels_height
    config.expert_configs[1]['n_levels_width'] = args.WNO_n_levels_width
    config.expert_configs[2]['n_scales']       = args.MNO_n_scales
    config.expert_configs[2]['scale_factors']  = args.MNO_scale_factors
    config.expert_configs[2]['n_layers']       = args.MNO_n_layers
    config.expert_configs[3]['n_modes']        = tuple(args.LNO_n_modes)
    config.expert_configs[3]['n_layers']       = args.LNO_n_layers
    config.expert_configs = [config.expert_configs[i] for i in args.choose_experts]

    experts_name = '_'.join([f"{config.expert_configs[i]['domain_type']}_{i}" if i in (0,1)
                             else f"{config.expert_configs[i]['type']}_{i}" for i in args.choose_experts])

    # 设备/日志开关（与主训练一致）
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device, is_logger = setup_env(config)

    # 结果与日志
    results_dir = Path(args.output_dir) / f"overfit1_{experts_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "training_log.txt"
    with open(log_file, "w") as f:
        f.write("Epoch,Total,L1,Grad,MSE_mon,PSNR_inv,Corr,GradNorm,LR,w_grad\n")

    # ===== 1) 数据与相同变换 =====
    full_dataset = SeismicDataset(data_dir=config.data_dir, family=config.family, split='train')
    data_dict = full_dataset.getStats()

    input_transform = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(data_dict['input_min'], k=args.k),
                          T.log_transform(data_dict['input_max'], k=args.k))
    ])
    output_transform = Compose([
        T.MinMaxNormalize(data_dict['output_min'], data_dict['output_max'])
    ])
    input_inverse_transform = Compose([
        T.InverseMinMaxNormalize(T.log_transform(data_dict['input_min'], k=args.k),
                                 T.log_transform(data_dict['input_max'], k=args.k)),
        T.InverseLogTransform(k=args.k)
    ])
    output_inverse_transform = Compose([
        T.InverseMinMaxNormalize(data_dict['output_min'], data_dict['output_max'])
    ])
    data_processor = SeismicDataProcessor(
        input_transform=input_transform,
        output_transform=output_transform,
        channel_dim=config.channel_dim
    )

    # 取一个样本（你可改 one_idx）
    one_idx = 0
    sample = data_processor(full_dataset[one_idx])

    class OneSampleDataset(torch.utils.data.Dataset):
        def __init__(self, s): self.s = s
        def __len__(self): return 1
        def __getitem__(self, i): return self.s

    train_loader = DataLoader(OneSampleDataset(sample), batch_size=1, shuffle=False, num_workers=0)
    loader_iter = cycle(train_loader)

    # ===== 2) 模型构建 =====
    config.in_channels = sample['input'].shape[0]
    if getattr(args, "use_moe", False) and args.use_experts_path:
        experts = load_moe_experts(
            expert_configs=config.expert_configs,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels,
            model_path=args.use_experts_path
        )
        # 如需解冻专家训练，取消下面注释
        # for e in experts:
        #     for p in e.parameters(): p.requires_grad = True
    else:
        experts = ExpertFactory.create_expert_ensemble(
            expert_configs=config.expert_configs,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels
        )

    model = MOEOperator(
        experts=experts,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        hidden_channels=config.hidden_channels,
        top_k=config.top_k,
        noisy_gating=config.noisy_gating,
        fusion_type=config.fusion_type,
        router_hidden_dim=config.router_hidden_dim
    ).to(device)

    # ===== 3) 优化器/调度/损失 =====
    # 单样本建议 LR=1e-3 起步；如想严格沿用传参，保留 config.learning_rate
    lr = max(1e-3, float(config.learning_rate))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=50)

    def sobel_grad(x):
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
        gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
        return gx, gy

    def grad_loss_sobel(pred, gt):
        gx1, gy1 = sobel_grad(pred); gx2, gy2 = sobel_grad(gt)
        return (gx1-gx2).abs().mean() + (gy1-gy2).abs().mean()

    def criterion(pred, gt, epoch):
        # 梯度项权重：每50个epoch增加0.05，上限0.25；MSE只给很小权重稳态
        w_g = min(0.25, 0.05 * (epoch // 50))
        loss_l1 = F.l1_loss(pred, gt)
        loss_g  = grad_loss_sobel(pred, gt)
        loss_m  = F.mse_loss(pred, gt) * 0.05
        total   = loss_l1 + w_g*loss_g + loss_m
        return total, loss_l1, loss_g, w_g

    def psnr_after_inv(pred, tgt):
        p = output_inverse_transform(pred.detach().cpu())
        t = output_inverse_transform(tgt.detach().cpu())
        mse_val = F.mse_loss(p, t).item()
        if mse_val <= 1e-12: return 99.0
        tmax, tmin = t.max().item(), t.min().item()
        MAX = max(abs(tmax), abs(tmin), 1.0)
        return 10.0 * np.log10((MAX*MAX) / mse_val)

    def corr_pred_input(pred, inp):
        # 输入降维至输出分辨率后计算相关
        x = inp.mean(dim=1, keepdim=True) if inp.dim()==4 else inp
        x = F.interpolate(x, size=pred.shape[-2:], mode='bilinear', align_corners=False)
        p = pred.detach().flatten(); q = x.detach().flatten()
        p = p - p.mean(); q = q - q.mean()
        return (p*q).sum().item() / (p.norm().item()*q.norm().item() + 1e-8)

    # ===== 4) 训练循环 =====
    MAX_EPOCHS = max(2000, args.epochs or 500)
    EARLY_PATIENCE = 100
    best_total = float('inf')
    best_state = None
    bad = 0

    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        batch = next(loader_iter)
        inp = batch['input'].to(device)   # [1,C,T,R]
        tgt = batch['output'].to(device)  # [1,1,H,W]
        assert inp.dim()==4 and tgt.dim()==4, f"Bad shapes: inp={tuple(inp.shape)}, tgt={tuple(tgt.shape)}"
        if epoch == 1:
            print(f"[overfit1] inp: {tuple(inp.shape)} (expect [1,C,T,R]) | tgt: {tuple(tgt.shape)} (expect [1,1,H,W])")

        opt.zero_grad(set_to_none=True)
        pred = model(inp)
        total, loss_l1, loss_g, w_g = criterion(pred, tgt, epoch)

        # 非有限守护
        if not torch.isfinite(total):
            print(f"[overfit1] non-finite loss at epoch {epoch}: {total.item()}")
            break

        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # 指标统计
        with torch.no_grad():
            mae_val  = float(loss_l1.item())
            grad_val = float(loss_g.item())
            mse_mon  = float(F.mse_loss(pred, tgt).item())  # 仅记录
            psnr_val = float(psnr_after_inv(pred, tgt))
            corr_val = float(corr_pred_input(pred, inp))

        # 调度器（用 L1 更稳定）
        old_lr = opt.param_groups[0]['lr']
        sch.step(mae_val)
        new_lr = opt.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"[LR] {old_lr:.2e} → {new_lr:.2e}")

        # 保存最优（以 total 为准）
        improved = total.item() < best_total - 1e-10
        if improved:
            best_total = total.item()
            best_state = {
                k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"[overfit1 {epoch:04d}] Total={total.item():.6f} | L1={mae_val:.6f} | Grad={grad_val:.6f} "
                  f"| MSE(mon)={mse_mon:.6f} | PSNR(inv)={psnr_val:.3f} | corr={corr_val:.3f} "
                  f"| grad_norm={grad_norm:.3e} | lr={new_lr:.2e} | w_g={w_g:.3f}")

        with open(log_file, "a") as f:
            f.write(f"{epoch},{total.item():.6f},{mae_val:.6f},{grad_val:.6f},{mse_mon:.6f},"
                    f"{psnr_val:.3f},{corr_val:.3f},{grad_norm:.3e},{new_lr:.2e},{w_g:.3f}\n")

        if bad >= EARLY_PATIENCE:
            print(f"Early stop at epoch {epoch}, best Total={best_total:.6e}")
            break

    # ===== 5) 保存最佳与可视化 =====
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, results_dir / f"best_overfit1_{experts_name}.pt")

    model.eval()
    with torch.no_grad():
        pred = model(inp)
        pred_inv = output_inverse_transform(pred.detach().cpu())
        tgt_inv  = output_inverse_transform(tgt.detach().cpu())
        inp_inv  = input_inverse_transform(inp.detach().cpu())

        np.save(results_dir/'pred_overfit_inv.npy', pred_inv.numpy())
        np.save(results_dir/'tgt_overfit_inv.npy',  tgt_inv.numpy())

        visualize_results(
            inputs=inp_inv,
            targets=tgt_inv,
            predictions=pred_inv,
            save_dir=str(results_dir / "vis")
        )

    print(f"[overfit1] Done. Artifacts saved at: {results_dir.resolve()}")


# 定义一个TransformedSubset类，用于对Subset应用变换
class TransformedSubset(Subset):
    """
    支持变换的数据集子集
    
    Parameters
    ----------
    dataset : Dataset
        原始数据集
    indices : list
        子集索引列表
    transform : callable, optional
        应用于样本的转换函数，默认为None
    """
    def __init__(self, dataset, transform=None):
        # 不要重新创建索引列表，而是使用dataset.indices
        # 如果dataset已经是Subset类型
        if hasattr(dataset, 'indices'):
            super().__init__(dataset.dataset, dataset.indices)
        else:
            # 如果不是Subset类型，则使用传入的dataset和其默认索引
            super().__init__(dataset, list(range(len(dataset))))
        self.transform = transform
        # self.logger = None
        
    def __getitems__(self, idx):
        # 这个idx是一个batch中各个数据在总数据集中的索引，是一个列表
        # if self.logger is None:
        #     self.logger = logging.getLogger(f"Worker-{os.getpid()}")
        #     if not self.logger.hasHandlers():
        #         handler = logging.FileHandler(f"/root/autodl-tmp/FWINO/workers_logs/worker_{os.getpid()}.log")
        #         handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        #         self.logger.addHandler(handler)
        #         self.logger.setLevel(logging.DEBUG)
        # self.logger.info(f"Loading indices: {idx[0]}-{idx[-1]}")
        batch_sample = []
        for index in idx:
            sample = self.dataset[self.indices[index]]
            sample['idx'] = index
            if self.transform:
                batch_sample.append(self.transform(sample))
        return batch_sample


def run_inference(args):
    """
    使用训练好的模型进行推理
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    """
    # 验证模型路径
    if not os.path.exists(args.model_path):
        raise ValueError(f"模型文件不存在: {args.model_path}")
    
    # 加载模型
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # 获取归一化参数
    data_dict = checkpoint['data_dict']
    
    # 加载配置
    config = SeismicMOEConfig()
    if args.data_dir:
        config.data_dir = args.data_dir
    else:
        # 设置默认数据目录为新路径
        config.data_dir = "/data1/wuruoyu/waveform-inversion"
    
    # 检查测试目录是否存在
    test_dir = os.path.join(config.data_dir, 'test')
    if not os.path.exists(test_dir):
        raise ValueError(f"测试目录不存在: {test_dir}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    input_transform = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(data_dict['input_min'], k=args.k), T.log_transform(data_dict['input_max'], k=args.k))
    ]) # data
    
    output_inverse_transform = Compose([
        T.InverseMinMaxNormalize(data_dict['output_min'], data_dict['output_max'])
    ])
    
    # 创建测试数据集
    test_loader, test_dataset = create_seismic_dataloader(
        data_dir=config.data_dir,
        family='all',  # 测试时使用所有数据
        split='test',
        batch_size=args.batch_size or 4,
        shuffle=False,
        num_workers=args.num_workers,
        input_transform=input_transform
    )
    
    # 获取实际的输入通道数
    sample_batch = next(iter(test_loader))
    in_channels = sample_batch['input'].shape[1]
    config.in_channels = in_channels
    
    # 创建专家模型
    experts = ExpertFactory.create_expert_ensemble(
        expert_configs=config.expert_configs,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        hidden_channels=config.hidden_channels
    )
    
    # 创建MOE模型
    model = MOEOperator(
        experts=experts,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        hidden_channels=config.hidden_channels,
        top_k=config.top_k,
        noisy_gating=config.noisy_gating,
        fusion_type=config.fusion_type,
        router_hidden_dim=config.router_hidden_dim
    )
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 移动模型到设备
    model = model.to(device)
    model.eval()
    
    # 创建结果目录
    results_dir = Path(args.output_dir) / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 进行推理
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="推理中"):
            # 获取数据
            inputs = batch['input'].to(device) # 1*70*70 
            # 假设batch_size是100 dataloader自动增加batch维度 变成B*1*70*70
            # 对于dataset返回的是sample字典，dataloader按字段拼接成列表
            file_names = batch['input_file'] # len=100的列表
            
            # 前向传播
            predictions = model(inputs)
            
            # 反归一化
            predictions = output_inverse_transform(predictions)
            
            # 保存预测结果
            for i, file_name in enumerate(file_names):
                # 将预测结果转换为NumPy数组
                prediction = predictions[i].cpu().numpy()
                
                # 保存为.npy文件
                output_path = results_dir / f"{file_name}.npy"
                np.save(output_path, prediction)
    
    print(f"推理完成！结果保存在: {results_dir}")

def get_expert_dict(full_sd, i: int=0, ddp_prefix='module.'):
    def strip_dp(k):
        return k[len(ddp_prefix):] if k.startswith(ddp_prefix) else k
    
    sub: Dict[str, torch.Tensor] = {}
    prefix = f'experts.{i}'
    for k, v in full_sd.items():
        k2 = strip_dp(k)
        if k2.startswith(prefix):
            sub[k2[len(prefix):]] = v
    
    if not sub:
        # 回退
        sub = {strip_dp(k): v for k, v in full_sd.items()}
    return sub

def load_factory(
    experts_config: List[Dict[str, Any]],
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    model_dict: OrderedDict,
) -> List[nn.Module]: 
    """专家融合工厂

    Args:
        experts_config (List[Dict[str, Any]]): 专家配置字典
        model_dict (Dict): 专家模型参数字典, experts_type == 'math' 
        --> Dict[expert_id, List[Dict[v_type_id, sd]]]

    Returns:
        List[nn.Module]: 返回专家模型列表
    """
    
    experts: List[nn.Module] = []
    
    for k, v in model_dict.items():
        # k: expert_id
        # v: List[Dict[v_type_id, sd]]
        try:
            expert_id = int(k)
        except Exception:
            raise ValueError(f"expert_id 非整数: {k}")
        
        if not (0 <= expert_id < len(experts_config)):
            raise IndexError(f"experts_config 下标越界: {expert_id}")
        
        expert_config = experts_config[expert_id]
        
        # 按v_type升序排列
        try:
            sorted_dict_list = sorted(
            v,
            key = lambda d: next(iter(d.keys())),
        )
        except Exception as e:
            raise RuntimeError(f"对 v_type 列表排序失败 (可能有 None 键): {e}")
        
        for type_expert_sd in sorted_dict_list:
            v_type_id, expert_sd = next(iter(type_expert_sd.items()))
            
            # 创建专家骨架
            expert_raw_model = ExpertFactory.create_expert_ensemble(
                [expert_config],
                in_channels,
                out_channels,
                hidden_channels,
            )[0]
            
            # 加载权重
            missing, unexpected = expert_raw_model.load_state_dict(expert_sd, strict=False)
            if missing or unexpected:
                print(f"[expert {expert_id}] missing: {missing}, unexpected: {unexpected}")

            # 冻结
            for p in expert_raw_model.parameters():
                p.requires_grad = False
            expert_raw_model.eval()
            
            experts.append(expert_raw_model)
            
    return experts # [FNO0, FNO1, FNO2, FNO3, FNO4, WNO0,...., MNO4,..., LNO4]

_SPECIFIC_PAT = re.compile(
    r'best_expert_(?P<name>\w+)_(?P<i>\d+)_(?P<shape>\w+)_(?P<label>\w+)\.pt$'
)
_NORMAL_PAT = re.compile(
    r'best_expert_(?P<name>\w+)_(?P<i>\d+)_(?P<label>\w+)\.pt$'
)
def load_moe_experts(
    experts_config: List[Dict[str, Any]],
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    model_path: str,
    is_specific: bool,
    map_location,
    type_dict: Dict[str, Dict[str, int]],
) -> List[nn.Module]:
    """读取融合专家参数

    Args:
        model_path (str): 专家保存的文件路径,
            保存的文件名：不细化版本: best_expert_{experts_name}_{i}_{vel/fault/style}.pt\
                        细化版本: best_expert_{experts_name}_{i}_{curve/flat/style}_{vel/fault/style}.pt
            
            按math分成FNO, WNO, MNO, LNO四类，每类有多种速度图类型, 直接读取, 每类以\
            
        is_specific (bool): 速度图是否细分
        
    Returns:
        experts (List[nn.Module]): 输出专家列表
    """
    if not os.path.isdir(model_path):
        raise ValueError(f"{model_path}不是有效路径")
    
    # 只取 .pt
    experts_file = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    
    # 组装: expert_id -> List[{v_type_id: sd}]
    grouped: Dict[str, List[Dict[int, Dict[str, torch.Tensor]]]] = defaultdict(list) #Dict[str(type), list]
    
    if(is_specific):
        id_map = type_dict.get('specific', {})
        # 获取所有.pt文件, best_expert_{experts_name}_{i}_{curve/flat/style}_{vel/fault/style}.pt
        for f in experts_file:
            m = _SPECIFIC_PAT.match(f)
            if not m:
                # 兼容 split 解析
                parts = f.split('_')
                if len(parts) >= 6 and parts[0] == 'best' and parts[1] == 'expert':
                    expert_id = parts[3]
                    shape = parts[4]
                    label = parts[5].split('.')[0]
                else:
                    print(f"[WARN] 文件名不匹配 specific 模式, 跳过: {f}")
                    continue
            else:
                expert_id = m.group('i')
                shape = m.group('shape')
                label = m.group('label')
            
            key = f"{shape}_{label}"
            if key not in id_map:
                print(f"[WARN] specific 类型映射缺失 {key}, 跳过: {f}")
                continue
            v_type = id_map[key]
            
            ckpt = torch.load(os.path.join(model_path, f), map_location=map_location)
            full_sd = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
            expert_sd = get_expert_dict(full_sd, i=0)
            grouped[expert_id].append({v_type: expert_sd})            
    else:
        id_map = type_dict.get('normal', {})
        for f in experts_file:
            m = _NORMAL_PAT.match(f)
            if not m:
                # 兼容 split 解析（宽松）
                parts = f.split('_')
                if len(parts) >= 5 and parts[0] == 'best' and parts[1] == 'expert':
                    expert_id = parts[3]
                    label = parts[4].split('.')[0]
                else:
                    print(f"[WARN] 文件名不匹配 normal 模式，跳过：{f}")
                    continue
            else:
                expert_id = m.group('i')
                label = m.group('label')

            if label not in id_map:
                print(f"[WARN] normal 类型映射缺失 {label}，跳过：{f}")
                continue
            v_type = id_map[label]

            ckpt = torch.load(os.path.join(model_path, f), map_location=map_location)
            full_sd = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
            expert_sd = get_expert_dict(full_sd, i=0)
            grouped[expert_id].append({v_type: expert_sd}) # [FNO(3), WNO(3), MNO(3), LNO(3)]
    
    # 对 expert_id 做数字序排序, 保证顺序稳定  
    try:
        ordered = OrderedDict(sorted(grouped.items(), key=lambda kv: int(kv[0])))
    except Exception:
        ordered = OrderedDict(sorted(grouped.items(), key=lambda kv: kv[0]))
    
    loaded_experts = load_factory(
        experts_config,
        in_channels,
        out_channels,
        hidden_channels,
        ordered, 
    )
    
    return loaded_experts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="地震数据MOE训练和推理")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference','overfit1'],
                        help='运行模式: 训练或推理')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录路径')
    parser.add_argument('--family', type=str, default=None, choices=['vel', 'style', 'fault', 'all', 'curve_vel', 'flat_vel', 'curve_fault', 'flat_fault', 'style_style'],
                        help='数据集系列: vel, style, fault 或 all')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    
    parser.add_argument('--lr_warmup_epochs', type=int, default=5,
                        help='学习率预热轮数')
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, 60, 90],
                        help='学习率衰减里程碑')
    parser.add_argument('--scheduler_gamma', type=float, default=0.3,
                        help='学习率衰减因子')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='L2正则化')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='梯度累计步数')
    
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='结果保存目录')
    parser.add_argument('--model_path', type=str, default=None,
                        help='推理模式下使用的模型路径')
    parser.add_argument('--vis_freq', type=int, default=5,
                        help='可视化频率（每隔多少个epoch可视化一次）')
    parser.add_argument('--distributed', action='store_true',
                        help='是否使用分布式训练')
    parser.add_argument('--use_wandb', action='store_true',
                        help='是否使用WandB记录训练过程')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例，默认为0.2（20%）')
    
    parser.add_argument('--k', type=int, default=1,
                        help='预处理缩放比例')
    parser.add_argument('--top_k', type=int, default=1,
                        help='选择前k个专家')
    parser.add_argument('--choose_experts',nargs='+', type=int, default=[0],
                        help='专家选择, FNO:0, WNO:1, MNO:2, LNO:3')
    
    parser.add_argument('--FNO_n_modes_height', type=int, default=16,
                        help='高度傅里叶变换后保留的模态数量')
    parser.add_argument('--FNO_n_modes_width', type=int, default=16,
                        help='宽度傅里叶变换后保留的模态数量')
    parser.add_argument('--FNO_n_layers', type=int, default=4,
                        help='傅里叶layers堆叠数量')
    parser.add_argument('--WNO_n_levels_height', type=int, default=2,
                        help='高度减少级别')
    parser.add_argument('--WNO_n_levels_width', type=int, default=2,
                        help='宽度减少级别')
    parser.add_argument('--WNO_n_layers', type=int, default=4,
                        help='WNO块的数量，控制模型深度')
    parser.add_argument('--WNO_block_n_layers', type=int, default=2,
                        help='每个WNO块内部的层数')
    parser.add_argument('--WNO_dropout_rate', type=float, default=0.1,
                        help='WNO块中的dropout比例，提高泛化能力')
    parser.add_argument('--wavelet_type', type=str, default='haar', choices=['haar', 'db4'],
                        help='小波类型，支持haar和db4小波')
    parser.add_argument('--MNO_n_scales', type=int, default=3,
                        help='总共使用的尺度')
    parser.add_argument('--MNO_scale_factors', nargs='+', type=float, default=[1.0, 0.5, 0.25],
                        help='每个尺度的缩放因子')
    parser.add_argument('--MNO_n_layers', type=int, default=3,
                        help='每个尺度使用的神经网络层数')
    parser.add_argument('--LNO_n_modes', nargs=2, type=int, default=[16, 16],
                        help='局部变换后保留的模态数量')
    parser.add_argument('--LNO_n_layers', type=int, default=3,
                        help='每个尺度使用的神经网络层数')
    
    # MoE融合参数配置
    parser.add_argument('--use_experts_path', type=str, default=None,
                        help='moe使用的专家模型存放路径')
    parser.add_argument('--use_moe', action='store_true',
                        help='是否使用moe, 使用会冻结专家模型')
    parser.add_argument('--router_type', type=str, default='basic',
                        help='路由器类型: \'basic\'/\'adamv\'')
    parser.add_argument('--fusion_type', type=str, default='linear',
                        help='专家组间融合方式: \'linear\'/\'attention\'/\'swa\'')
    parser.add_argument('--s_processor_type', type=str, default='linear',
                        help='强专家组内融合方式: \'linear\'/\'atten\'/\'mean\'/\'sum\'')
    parser.add_argument('--w_processor_type', type=str, default='linear',
                        help='弱专家组内融合方式: \'linear\'/\'atten\'/\'mean\'/\'sum\'')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='强弱激活参数，beta越大，弱激活影响越大')
    parser.add_argument('--is_specific', action='store_true',
                        help='是否细化种类')
    parser.add_argument('--is_classier', action='store_true',
                        help='是否使用分组专家网络')
    
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='隐藏通道数（默认值由配置文件决定，可通过此参数覆盖）')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率（默认值由配置文件决定，可通过此参数覆盖）')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='恢复训练的checkpoint路径，如 best_model_xxx.pt')
    
    # Loss related
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=1.0)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=1.0)
    
    # Performance related
    parser.add_argument('--profile_timing', action='store_true',
                        help='是否记录训练过程中的耗时信息')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'inference':
        if not args.model_path:
            raise ValueError("推理模式需要指定模型路径 --model_path")
        run_inference(args)
    elif args.mode == 'overfit1':
        run_overfit_one_sample(args)
    else:
        raise ValueError(f"不支持的运行模式: {args.mode}") 
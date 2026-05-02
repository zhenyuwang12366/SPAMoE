"""
评估MOE (Mixture of Experts) 神经算子模型的性能
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralop.models import MOEOperator, ExpertFactory
from neuralop.data.datasets import create_seismic_dataloader
from neuralop.data.datasets import TensorDataset
from config.seismic_moe_config import SeismicMOEConfig
from config.moe_config import MOEConfig


class ModelEvaluator:
    """
    MOE模型评估器
    
    用于评估MOE模型在不同数据集上的性能
    """
    def __init__(
        self,
        model_path: str,
        device: torch.device = None,
        output_dir: str = './evaluation_results'
    ):
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # 提取归一化参数
        self.input_mean = self.checkpoint.get('input_mean', 0.0)
        self.input_std = self.checkpoint.get('input_std', 1.0)
        self.output_mean = self.checkpoint.get('output_mean', 0.0)
        self.output_std = self.checkpoint.get('output_std', 1.0)
        
        # 模型尚未加载，将在evaluate方法中根据数据集类型加载
        self.model = None
        
        print(f"模型加载自: {model_path}")
        print(f"使用设备: {device}")
    
    def calculate_metrics(self, predictions, targets):
        """
        计算评估指标
        
        Parameters
        ----------
        predictions : torch.Tensor
            模型预测结果
        targets : torch.Tensor
            真实目标值
            
        Returns
        -------
        dict
            包含各种评估指标的字典
        """
        # 确保输入是PyTorch张量
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
        
        # 移动到CPU计算指标
        predictions = predictions.cpu()
        targets = targets.cpu()
        
        # 计算MSE
        mse = F.mse_loss(predictions, targets).item()
        
        # 计算MAE
        mae = F.l1_loss(predictions, targets).item()
        
        # 计算相对L2误差
        rel_l2_error = torch.norm(predictions - targets) / torch.norm(targets)
        rel_l2_error = rel_l2_error.item()
        
        # 计算PSNR
        data_range = targets.max() - targets.min()
        psnr = 20 * np.log10(data_range.item()) - 10 * np.log10(mse)
        
        # 计算SSIM (如果可用)
        ssim = 0.0
        try:
            from skimage.metrics import structural_similarity as ssim_fn
            # 对每个样本和通道分别计算SSIM
            ssim_values = []
            for i in range(predictions.shape[0]):
                for c in range(predictions.shape[1]):
                    pred = predictions[i, c].numpy()
                    targ = targets[i, c].numpy()
                    ssim_val = ssim_fn(pred, targ, data_range=data_range.item())
                    ssim_values.append(ssim_val)
            ssim = np.mean(ssim_values)
        except ImportError:
            print("警告: 未安装scikit-image，无法计算SSIM")
        
        return {
            'mse': mse,
            'mae': mae,
            'rel_l2_error': rel_l2_error,
            'psnr': psnr,
            'ssim': ssim
        }
    
    def visualize_predictions(self, inputs, targets, predictions, save_dir, max_samples=4):
        """
        可视化模型预测结果
        
        Parameters
        ----------
        inputs : torch.Tensor
            输入数据
        targets : torch.Tensor
            目标数据
        predictions : torch.Tensor
            模型预测结果
        save_dir : str or Path
            保存目录
        max_samples : int, optional
            最大可视化样本数，默认为4
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 限制样本数
        n_samples = min(inputs.shape[0], max_samples)
        
        for i in range(n_samples):
            # 创建图形
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 绘制输入（选择第一个通道）
            im0 = axes[0].imshow(inputs[i, 0].cpu().numpy(), cmap='viridis')
            axes[0].set_title('输入数据 (第一个通道)')
            plt.colorbar(im0, ax=axes[0])
            
            # 绘制目标
            im1 = axes[1].imshow(targets[i, 0].cpu().numpy(), cmap='jet')
            axes[1].set_title('目标')
            plt.colorbar(im1, ax=axes[1])
            
            # 绘制预测
            im2 = axes[2].imshow(predictions[i, 0].cpu().numpy(), cmap='jet')
            axes[2].set_title('预测')
            plt.colorbar(im2, ax=axes[2])
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(save_dir / f'sample_{i}.png', dpi=300)
            plt.close(fig)
    
    def analyze_expert_usage(self, inputs, batch_size=8):
        """
        分析专家使用情况
        
        Parameters
        ----------
        inputs : torch.Tensor
            输入数据
        batch_size : int, optional
            批次大小，默认为8
            
        Returns
        -------
        dict
            专家使用统计信息
        """
        if not hasattr(self.model, 'router'):
            print("警告: 模型没有路由器，无法分析专家使用情况")
            return {}
        
        self.model.eval()
        expert_counts = torch.zeros(self.model.num_experts, device=self.device)
        expert_weights = torch.zeros(self.model.num_experts, device=self.device)
        total_samples = 0
        
        with torch.no_grad():
            # 分批处理
            for i in range(0, inputs.shape[0], batch_size):
                batch_inputs = inputs[i:i+batch_size]
                batch_size_actual = batch_inputs.shape[0]
                total_samples += batch_size_actual
                
                # 提取输入特征进行路由
                x_flat = batch_inputs.view(batch_size_actual, -1, self.model.in_channels).mean(dim=1)
                
                # 获取路由权重和专家索引
                routing_weights, expert_indices = self.model.router(x_flat)
                
                # 统计专家使用情况
                for k in range(self.model.top_k):
                    indices = expert_indices[:, k]
                    weights = routing_weights[:, k]
                    
                    for b in range(batch_size_actual):
                        expert_idx = indices[b].item()
                        expert_counts[expert_idx] += 1
                        expert_weights[expert_idx] += weights[b].item()
        
        # 计算专家使用百分比
        expert_usage_percent = expert_counts / total_samples * 100
        expert_weights_avg = expert_weights / expert_counts
        
        # 转换为Python字典
        expert_stats = {
            'expert_counts': expert_counts.cpu().numpy().tolist(),
            'expert_usage_percent': expert_usage_percent.cpu().numpy().tolist(),
            'expert_weights_avg': expert_weights_avg.cpu().numpy().tolist(),
        }
        
        return expert_stats
    
    def evaluate_seismic(self, data_dir, family='all', batch_size=8, num_workers=4):
        """
        评估地震数据MOE模型
        
        Parameters
        ----------
        data_dir : str
            数据目录
        family : str, optional
            数据集系列，默认为'all'
        batch_size : int, optional
            批次大小，默认为8
        num_workers : int, optional
            数据加载工作进程数，默认为4
            
        Returns
        -------
        dict
            评估结果
        """
        # 加载配置
        config = SeismicMOEConfig()
        config.data_dir = data_dir
        config.family = family
        
        # 创建数据加载器
        val_loader, val_dataset = create_seismic_dataloader(
            data_dir=data_dir,
            family=family,
            split='train',  # 使用训练集的一部分作为验证集
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            normalize_inputs=True,
            normalize_outputs=True
        )
        
        # 获取实际的输入通道数
        sample_batch = next(iter(val_loader))
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
        self.model = MOEOperator(
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
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 评估
        all_metrics = []
        all_inputs = []
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="评估中"):
                # 获取数据
                inputs = batch['input'].to(self.device)
                targets = batch['output'].to(self.device)
                
                # 前向传播
                predictions = self.model(inputs)
                
                # 反归一化
                if 'output_mean' in self.checkpoint and 'output_std' in self.checkpoint:
                    predictions = predictions * self.output_std + self.output_mean
                    targets = targets * self.output_std + self.output_mean
                
                # 计算指标
                batch_metrics = self.calculate_metrics(predictions, targets)
                all_metrics.append(batch_metrics)
                
                # 收集数据用于可视化
                all_inputs.append(inputs.cpu())
                all_targets.append(targets.cpu())
                all_predictions.append(predictions.cpu())
        
        # 计算平均指标
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in all_metrics])
        
        # 可视化
        vis_inputs = torch.cat(all_inputs[:4], dim=0)
        vis_targets = torch.cat(all_targets[:4], dim=0)
        vis_predictions = torch.cat(all_predictions[:4], dim=0)
        
        self.visualize_predictions(
            vis_inputs, vis_targets, vis_predictions,
            save_dir=self.output_dir / "seismic_vis"
        )
        
        # 分析专家使用情况
        expert_stats = self.analyze_expert_usage(torch.cat(all_inputs, dim=0))
        
        # 保存结果
        results = {
            'metrics': avg_metrics,
            'expert_stats': expert_stats
        }
        
        with open(self.output_dir / "seismic_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("评估结果:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        return results
    
    def evaluate_generic(self, data_path, batch_size=8, num_workers=4):
        """
        评估通用数据集的MOE模型
        
        Parameters
        ----------
        data_path : str
            数据文件路径
        batch_size : int, optional
            批次大小，默认为8
        num_workers : int, optional
            数据加载工作进程数，默认为4
            
        Returns
        -------
        dict
            评估结果
        """
        # 加载配置
        config = MOEConfig()
        
        # 加载数据
        data = torch.load(data_path)
        if isinstance(data, dict):
            x_test = data.get('x_test')
            y_test = data.get('y_test')
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")
        
        if x_test is None or y_test is None:
            raise ValueError("数据文件中缺少x_test或y_test")
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(x_test, y_test)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # 获取实际的输入通道数
        in_channels = x_test.shape[1]
        config.in_channels = in_channels
        
        # 创建专家模型
        experts = ExpertFactory.create_expert_ensemble(
            expert_configs=config.expert_configs,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels
        )
        
        # 创建MOE模型
        self.model = MOEOperator(
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
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 评估
        all_metrics = []
        all_inputs = []
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for x_batch, y_batch in tqdm(dataloader, desc="评估中"):
                # 移动数据到设备
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 前向传播
                predictions = self.model(x_batch)
                
                # 反归一化
                if 'output_mean' in self.checkpoint and 'output_std' in self.checkpoint:
                    predictions = predictions * self.output_std + self.output_mean
                    y_batch = y_batch * self.output_std + self.output_mean
                
                # 计算指标
                batch_metrics = self.calculate_metrics(predictions, y_batch)
                all_metrics.append(batch_metrics)
                
                # 收集数据用于可视化
                all_inputs.append(x_batch.cpu())
                all_targets.append(y_batch.cpu())
                all_predictions.append(predictions.cpu())
        
        # 计算平均指标
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in all_metrics])
        
        # 可视化
        vis_inputs = torch.cat(all_inputs[:4], dim=0)
        vis_targets = torch.cat(all_targets[:4], dim=0)
        vis_predictions = torch.cat(all_predictions[:4], dim=0)
        
        self.visualize_predictions(
            vis_inputs, vis_targets, vis_predictions,
            save_dir=self.output_dir / "generic_vis"
        )
        
        # 分析专家使用情况
        expert_stats = self.analyze_expert_usage(torch.cat(all_inputs, dim=0))
        
        # 保存结果
        results = {
            'metrics': avg_metrics,
            'expert_stats': expert_stats
        }
        
        with open(self.output_dir / "generic_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("评估结果:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="评估MOE模型")
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--dataset_type', type=str, default='seismic',
                        choices=['seismic', 'generic'],
                        help='数据集类型: seismic或generic')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据路径，对于seismic是数据目录，对于generic是数据文件')
    parser.add_argument('--family', type=str, default='all',
                        choices=['vel', 'style', 'fault', 'all'],
                        help='地震数据集系列，仅对seismic有效')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='评估结果保存目录')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID，-1表示使用CPU')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    # 创建评估器
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        device=device,
        output_dir=args.output_dir
    )
    
    # 根据数据集类型评估
    if args.dataset_type == 'seismic':
        evaluator.evaluate_seismic(
            data_dir=args.data_path,
            family=args.family,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    elif args.dataset_type == 'generic':
        evaluator.evaluate_generic(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        raise ValueError(f"不支持的数据集类型: {args.dataset_type}")


if __name__ == '__main__':
    main() 
"""
使用任务感知路由器的MOE（Mixture of Experts）架构训练多任务神经算子模型
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralop.models import MOEOperator, ExpertFactory
from neuralop.training import Trainer
from neuralop.data.datasets import create_seismic_dataloader
from neuralop.data.datasets import TensorDataset
from config.task_aware_moe_config import TaskAwareMOEConfig


class MultitaskDataLoader:
    """
    多任务数据加载器
    
    管理多个任务的数据加载器
    
    Parameters
    ----------
    config : TaskAwareMOEConfig
        配置对象
    data_dir : str
        数据目录
    """
    def __init__(self, config, data_dir):
        self.config = config
        self.data_dir = data_dir
        self.task_loaders = {}
        self.task_datasets = {}
        
        # 加载每个任务的数据
        for task in config.tasks:
            task_name = task['name']
            print(f"加载任务 {task_name} 的数据...")
            
            if task_name == 'darcy':
                train_loader, train_dataset = self._load_darcy_data(split='train')
                val_loader, val_dataset = self._load_darcy_data(split='test')
            elif task_name == 'burgers':
                train_loader, train_dataset = self._load_burgers_data(split='train')
                val_loader, val_dataset = self._load_burgers_data(split='test')
            elif task_name == 'navier_stokes':
                train_loader, train_dataset = self._load_navier_stokes_data(split='train')
                val_loader, val_dataset = self._load_navier_stokes_data(split='test')
            elif task_name == 'seismic':
                train_loader, train_dataset = self._load_seismic_data(split='train')
                val_loader, val_dataset = self._load_seismic_data(split='test')
            else:
                raise ValueError(f"不支持的任务类型: {task_name}")
            
            self.task_loaders[task_name] = {
                'train': train_loader,
                'val': val_loader
            }
            
            self.task_datasets[task_name] = {
                'train': train_dataset,
                'val': val_dataset
            }
    
    def _load_darcy_data(self, split='train'):
        """加载Darcy流数据"""
        from neuralop.data.datasets import PT_PATH, load_pt_data
        
        # 加载数据
        data_path = os.path.join(self.data_dir, 'darcy', f'darcy_{split}_16.pt')
        if not os.path.exists(data_path):
            # 尝试从默认路径加载
            data_path = os.path.join(PT_PATH, f'darcy_{split}_16.pt')
        
        x, y = load_pt_data(data_path)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size if split == 'train' else self.config.test_batch_size,
            shuffle=(split == 'train'),
            num_workers=4
        )
        
        return dataloader, dataset
    
    def _load_burgers_data(self, split='train'):
        """加载Burgers方程数据"""
        from neuralop.data.datasets import PT_PATH, load_pt_data
        
        # 加载数据
        data_path = os.path.join(self.data_dir, 'burgers', f'burgers_{split}_16.pt')
        if not os.path.exists(data_path):
            # 尝试从默认路径加载
            data_path = os.path.join(PT_PATH, f'burgers_{split}_16.pt')
        
        x, y = load_pt_data(data_path)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size if split == 'train' else self.config.test_batch_size,
            shuffle=(split == 'train'),
            num_workers=4
        )
        
        return dataloader, dataset
    
    def _load_navier_stokes_data(self, split='train'):
        """加载Navier-Stokes方程数据"""
        from neuralop.data.datasets import load_navier_stokes_data
        
        # 加载数据
        data_path = os.path.join(self.data_dir, 'navier_stokes')
        
        # 使用默认加载函数
        dataset = load_navier_stokes_data(
            data_path,
            split=split,
            n_samples=None,
            normalize_inputs=self.config.normalize_inputs,
            normalize_outputs=self.config.normalize_outputs
        )
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size if split == 'train' else self.config.test_batch_size,
            shuffle=(split == 'train'),
            num_workers=4
        )
        
        return dataloader, dataset
    
    def _load_seismic_data(self, split='train'):
        """加载地震数据"""
        # 使用现有的地震数据加载器
        dataloader, dataset = create_seismic_dataloader(
            data_dir=os.path.join(self.data_dir, 'seismic'),
            family='all',
            split=split,
            batch_size=self.config.batch_size if split == 'train' else self.config.test_batch_size,
            shuffle=(split == 'train'),
            num_workers=4,
            normalize_inputs=self.config.normalize_inputs,
            normalize_outputs=self.config.normalize_outputs
        )
        
        return dataloader, dataset
    
    def get_task_embedding(self, task_name):
        """获取任务嵌入"""
        for task in self.config.tasks:
            if task['name'] == task_name:
                return torch.tensor(task['embedding'], dtype=torch.float32)
        
        raise ValueError(f"未找到任务 {task_name} 的嵌入")


class MultitaskTrainer:
    """
    多任务训练器
    
    用于训练多任务MOE模型
    
    Parameters
    ----------
    config : TaskAwareMOEConfig
        配置对象
    model : nn.Module
        MOE模型
    data_loaders : MultitaskDataLoader
        多任务数据加载器
    device : torch.device
        训练设备
    output_dir : str
        输出目录
    """
    def __init__(self, config, model, data_loaders, device, output_dir):
        self.config = config
        self.model = model
        self.data_loaders = data_loaders
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.milestones,
            gamma=config.scheduler_gamma
        )
        
        # 损失函数
        self.criterion = F.mse_loss
        
        # 训练日志
        self.log_file = self.output_dir / "training_log.txt"
        with open(self.log_file, "w") as f:
            f.write(f"Epoch,Task,Train Loss,Val Loss\n")
        
        # 最佳模型保存
        self.best_val_loss = float("inf")
        self.best_model_path = self.output_dir / "best_model.pt"
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        task_losses = {task['name']: 0.0 for task in self.config.tasks}
        task_samples = {task['name']: 0 for task in self.config.tasks}
        
        # 为每个任务创建迭代器
        task_iterators = {}
        for task in self.config.tasks:
            task_name = task['name']
            task_iterators[task_name] = iter(self.data_loaders.task_loaders[task_name]['train'])
        
        # 确定每个epoch的总批次数
        total_batches = max([len(loader) for loader in 
                            [self.data_loaders.task_loaders[task['name']]['train'] 
                             for task in self.config.tasks]])
        
        # 训练循环
        with tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{self.config.epochs}", leave=False) as pbar:
            for batch_idx in range(total_batches):
                # 随机选择一个任务
                task_idx = np.random.randint(0, len(self.config.tasks))
                task = self.config.tasks[task_idx]
                task_name = task['name']
                task_weight = task['weight']
                
                # 获取任务嵌入
                task_embedding = self.data_loaders.get_task_embedding(task_name).to(self.device)
                
                # 获取数据批次
                try:
                    batch = next(task_iterators[task_name])
                except StopIteration:
                    # 如果迭代器用完，重新创建
                    task_iterators[task_name] = iter(self.data_loaders.task_loaders[task_name]['train'])
                    batch = next(task_iterators[task_name])
                
                # 处理不同格式的数据批次
                if isinstance(batch, dict):  # 地震数据
                    inputs = batch['input'].to(self.device)
                    targets = batch['output'].to(self.device)
                else:  # 其他数据
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                
                # 扩展任务嵌入到批次大小
                batch_size = inputs.shape[0]
                task_embeddings = task_embedding.unsqueeze(0).expand(batch_size, -1)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(inputs, task_features=task_embeddings)
                
                # 计算损失
                loss = self.criterion(outputs, targets) * task_weight
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 更新统计信息
                epoch_loss += loss.item()
                task_losses[task_name] += loss.item()
                task_samples[task_name] += 1
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.6f}", "task": task_name})
        
        # 计算平均损失
        avg_loss = epoch_loss / total_batches
        task_avg_losses = {name: losses / max(1, samples) 
                          for name, losses, samples in 
                          zip(task_losses.keys(), task_losses.values(), task_samples.values())}
        
        return avg_loss, task_avg_losses
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        task_val_losses = {task['name']: 0.0 for task in self.config.tasks}
        task_samples = {task['name']: 0 for task in self.config.tasks}
        
        with torch.no_grad():
            for task in self.config.tasks:
                task_name = task['name']
                task_weight = task['weight']
                
                # 获取任务嵌入
                task_embedding = self.data_loaders.get_task_embedding(task_name).to(self.device)
                
                # 验证每个任务
                for batch in self.data_loaders.task_loaders[task_name]['val']:
                    # 处理不同格式的数据批次
                    if isinstance(batch, dict):  # 地震数据
                        inputs = batch['input'].to(self.device)
                        targets = batch['output'].to(self.device)
                    else:  # 其他数据
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                    
                    # 扩展任务嵌入到批次大小
                    batch_size = inputs.shape[0]
                    task_embeddings = task_embedding.unsqueeze(0).expand(batch_size, -1)
                    
                    # 前向传播
                    outputs = self.model(inputs, task_features=task_embeddings)
                    
                    # 计算损失
                    loss = self.criterion(outputs, targets) * task_weight
                    
                    # 更新统计信息
                    val_loss += loss.item()
                    task_val_losses[task_name] += loss.item()
                    task_samples[task_name] += 1
        
        # 计算平均验证损失
        total_samples = sum(task_samples.values())
        avg_val_loss = val_loss / max(1, total_samples)
        task_avg_val_losses = {name: losses / max(1, samples) 
                              for name, losses, samples in 
                              zip(task_val_losses.keys(), task_val_losses.values(), task_samples.values())}
        
        return avg_val_loss, task_avg_val_losses
    
    def train(self):
        """训练模型"""
        print(f"开始训练多任务MOE模型，共 {self.config.epochs} 个epochs...")
        
        # 训练循环
        for epoch in range(self.config.epochs):
            # 训练一个epoch
            start_time = time.time()
            train_loss, task_train_losses = self.train_epoch(epoch)
            
            # 验证
            val_loss, task_val_losses = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存日志
            with open(self.log_file, "a") as f:
                for task_name in task_train_losses.keys():
                    f.write(f"{epoch+1},{task_name},{task_train_losses[task_name]:.6f},{task_val_losses[task_name]:.6f}\n")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'task_val_losses': task_val_losses
                }, self.best_model_path)
                
                # 保存专家模型
                self.model.save_experts(self.output_dir / "experts")
            
            # 打印进度
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.config.epochs} - {epoch_time:.1f}s:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print("  任务训练损失:")
            for task_name, loss in task_train_losses.items():
                print(f"    {task_name}: {loss:.6f}")
            print("  任务验证损失:")
            for task_name, loss in task_val_losses.items():
                print(f"    {task_name}: {loss:.6f}")
            
            # 可视化专家分布
            if (epoch + 1) % 10 == 0:
                self.visualize_expert_distribution(epoch + 1)
        
        # 保存最终模型
        final_model_path = self.output_dir / "final_model.pt"
        torch.save({
            'epoch': self.config.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'task_val_losses': task_val_losses
        }, final_model_path)
        
        print(f"训练完成！最佳模型保存在: {self.best_model_path}")
        print(f"最终模型保存在: {final_model_path}")
    
    def visualize_expert_distribution(self, epoch):
        """可视化专家分布"""
        self.model.eval()
        
        # 为每个任务收集专家分布
        task_distributions = {}
        
        with torch.no_grad():
            for task in self.config.tasks:
                task_name = task['name']
                
                # 获取任务嵌入
                task_embedding = self.data_loaders.get_task_embedding(task_name).to(self.device)
                
                # 获取一个批次的数据
                try:
                    batch = next(iter(self.data_loaders.task_loaders[task_name]['val']))
                except:
                    continue
                
                # 处理不同格式的数据批次
                if isinstance(batch, dict):  # 地震数据
                    inputs = batch['input'].to(self.device)
                else:  # 其他数据
                    inputs, _ = batch
                    inputs = inputs.to(self.device)
                
                # 扩展任务嵌入到批次大小
                batch_size = inputs.shape[0]
                task_embeddings = task_embedding.unsqueeze(0).expand(batch_size, -1)
                
                # 获取专家分布
                expert_distribution = self.model.get_expert_distribution(inputs, task_embeddings)
                
                # 计算平均分布
                avg_distribution = expert_distribution.mean(dim=0).cpu().numpy()
                task_distributions[task_name] = avg_distribution
        
        # 创建图形
        plt.figure(figsize=(10, 6))
        
        # 绘制每个任务的专家分布
        x = np.arange(self.model.num_experts)
        width = 0.8 / len(task_distributions)
        
        for i, (task_name, distribution) in enumerate(task_distributions.items()):
            plt.bar(x + i * width - 0.4 + width/2, distribution, width, label=task_name)
        
        plt.xlabel('专家索引')
        plt.ylabel('平均路由权重')
        plt.title(f'Epoch {epoch} - 各任务的专家分布')
        plt.xticks(x, [f'专家{i}' for i in range(self.model.num_experts)])
        plt.legend()
        plt.tight_layout()
        
        # 保存图形
        plt.savefig(self.output_dir / f"expert_distribution_epoch_{epoch}.png", dpi=300)
        plt.close()


def run_training(args):
    """运行训练"""
    # 加载配置
    config = TaskAwareMOEConfig()
    
    # 更新配置
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_dir / "config.json", "w") as f:
        config_dict = {k: v for k, v in vars(config).items() 
                      if not k.startswith('__') and not callable(v)}
        json.dump(config_dict, f, indent=2, default=lambda x: str(x))
    
    # 创建多任务数据加载器
    data_loaders = MultitaskDataLoader(config, config.data_dir)
    
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
        router_type=config.router_type,
        task_dim=config.task_dim,
        routing_mode=config.routing_mode,
        v_type_num=getattr(config, "v_type_num", None)
    )
    
    # 移动模型到设备
    model = model.to(device)
    
    # 创建训练器
    trainer = MultitaskTrainer(
        config=config,
        model=model,
        data_loaders=data_loaders,
        device=device,
        output_dir=output_dir
    )
    
    # 开始训练
    trainer.train()
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="多任务MOE训练")
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录路径')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--output_dir', type=str, default='./results/multitask_moe',
                        help='结果保存目录')
    
    args = parser.parse_args()
    
    run_training(args) 

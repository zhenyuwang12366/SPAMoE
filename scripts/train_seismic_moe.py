"""
使用MOE（Mixture of Experts）架构训练地震数据的神经算子模型
支持分布式训练
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Union, Optional
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from torch.utils.data import DataLoader, random_split, Subset, DistributedSampler
import torchvision
from torchvision.transforms import Compose
import transforms as T
import time
import datetime
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralop.models import MOEOperator, ExpertFactory
from neuralop.training import Trainer, setup
from neuralop.training.torch_setup import setup
from neuralop.data.datasets import SeismicDataset, create_seismic_dataloader
from neuralop.utils import get_wandb_api_key, count_model_params
from config.seismic_moe_config import SeismicMOEConfig
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
        
        # # 绘制目标地震数据（选择第一个通道）
        # # 如果目标数据是多维的，我们只显示第一个通道
        # if len(targets[i].shape) > 2:
        #     target_data = targets[i, 0].cpu().numpy()
        # else:
        #     target_data = targets[i].cpu().numpy()
        # im1 = axes[1].imshow(target_data, cmap='viridis')
        # axes[1].set_title('目标地震数据')
        # plt.colorbar(im1, ax=axes[1])
        
        # # 绘制预测地震数据
        # # 如果预测数据是多维的，我们只显示第一个通道
        # if len(predictions[i].shape) > 2:
        #     pred_data = predictions[i, 0].cpu().numpy()
        # else:
        #     pred_data = predictions[i].cpu().numpy()
        # im2 = axes[2].imshow(pred_data, cmap='viridis')
        # axes[2].set_title('预测地震数据')
        # plt.colorbar(im2, ax=axes[2])
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i}.png'), dpi=300)
        plt.close(fig)
        
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

    print(f'batch_size:{config.batch_size}')
    print(f'epochs:{config.epochs}')
    print(f'learning_rate:{config.learning_rate}')
    print(f'hidden_channels:{config.hidden_channels}')

    # 设置验证集比例
    val_ratio = args.val_ratio if args.val_ratio is not None else 0.2
    
    # 设置设备并初始化分布式环境。详细解释见OneNote4
    device, is_logger = setup(config)
    
    # 启用分布式训练
    if args.distributed:
        config.distributed.use_distributed = True
        
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")

        device = torch.device(f"cuda:{local_rank}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
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
    # MNO config setting
    config.expert_configs[2]['n_scales'] = args.MNO_n_scales
    config.expert_configs[2]['scale_factors'] = args.MNO_scale_factors
    config.expert_configs[2]['n_layers'] = args.MNO_n_layers
    # LNO config setting
    config.expert_configs[3]['n_modes'] = tuple(args.LNO_n_modes)
    config.expert_configs[3]['n_layers'] = args.LNO_n_layers
    # 设置专家数
    config.top_k = args.top_k
    # 选择专家，这里后面的config.expert_configs就是config文件中所创建的字典列表，
    # 当你从命令行输入choose——experts之后，这里的for循环会根据你给定的序号找到对应的专家的字典，并将这个字典放入
    #config.expert_configs列表中
    config.expert_configs = [config.expert_configs[i] for i in args.choose_experts]
    #这里的config.expert_configs就是seismic_moe_config中的“字典列表”，关于“字典列表”结构的解释详见OneNote3
    print(f'FNO:n_modes_height:{config.expert_configs[0]["n_modes_height"]}')
    print(f'FNO:n_modes_width:{config.expert_configs[0]["n_modes_width"]}')
    print(f'FNO:n_layers:{config.expert_configs[0]["n_layers"]}')
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
    experts_name = '_'.join([f"{config.expert_configs[i]['domain_type']}_{i}" if i == 0 or i == 1 else  f"{config.expert_configs[i]['type']}_{i}" for i in args.choose_experts ])
    config.output_dir = os.path.join(config.output_dir, experts_name)   
    
    # 设置损失函数加权系数
    config.lambda_g1v = args.lambda_g1v
    config.lambda_g2v = args.lambda_g2v
    
    #-------------- 设置完毕 -----------#
    # 创建完整数据集
    full_dataset = SeismicDataset(
        data_dir=config.data_dir,
        family=config.family,
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
    # input_inverse_transform = Compose([
    #     T.InverseMinMaxNormalize(T.log_transform(data_dict['input_min'], k=args.k), T.log_transform(data_dict['input_max'], k=args.k)),
    #     T.InverseLogTransform(k=args.k)
    # ])
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
        train_sampler = DistributedSampler(train_dataset_with_transform, num_replicas=world_size, rank=local_rank)
        train_loader = DataLoader(
            train_dataset_with_transform,
            sampler=train_sampler,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=int(args.num_workers/2),
            pin_memory=True
        )

        print(f'prefetch_factor={train_loader.prefetch_factor}')
        
        val_sampler = DistributedSampler(val_dataset_with_transform, num_replicas=world_size, rank=local_rank)
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
    
    if config.use_moe:
        experts = load_moe_experts(
            expert_configs=config.expert_configs,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels
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
            router_hidden_dim=config.router_hidden_dim
        )
    
    # 移动模型到设备
    model = model.to(device)
    
    # 使用分布式数据并行
    if config.distributed.use_distributed:
        model = DDP(
            model, device_ids=[device.index], output_device=device.index, static_graph=True
        )
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.milestones,
        gamma=config.scheduler_gamma
    )
    
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
    best_model_path = results_dir / f"best_model_{experts_name}.pt"
    if len(experts_name) == 1:
        best_expert_path = results_dir / f"best_expert_{experts_name}.pt"
    
    # 指标计算器
    metrics = SeismicMetrics()
    
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

    
#以上全是准备工作，下面是核心循环

    # 训练循环
    for epoch in range(start_epoch, config.epochs):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        # 训练
        model.train()
        train_loss = 0.0
        
        # 在分布式环境中设置sampler的epoch
        if config.distributed.use_distributed and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False, disable=not is_logger) as pbar:
            for batch in pbar:
                # 每个epoch要训练n个batch，每个batch的batch_size为m，则总样本数约为n*m
                # 获取数据
                inputs = batch['input'].to(device)
                targets = batch['output'].to(device)
                
                # 前向传播
                predictions = model(inputs)
                
                # 计算损失
                loss, loss_g1v, loss_g2v = criterion(predictions, targets)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 更新统计信息
                train_loss += loss.item()
                if is_logger:
                    pbar.set_postfix({"train_loss": f"{loss.item():.6f}"})
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        all_metrics = {
            'mse': 0.0,
            'mae': 0.0,
            'psnr': 0.0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                # 获取数据
                inputs = batch['input'].to(device)
                targets = batch['output'].to(device)
                
                # 前向传播
                predictions = model(inputs)
                
                # 计算损失
                loss, loss_g1v, loss_g2v = criterion(predictions, targets)
                val_loss += loss.item()
                
                # 计算其他指标
                all_metrics['mse'] += loss_g1v.item()
                all_metrics['mae'] += loss_g2v.item()
                all_metrics['psnr'] += metrics.calculate_psnr(predictions, targets)
        
        # 计算平均验证损失和指标
        val_loss /= len(val_loader)
        for metric in all_metrics:
            all_metrics[metric] /= len(val_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 保存日志
        if is_logger:
            with open(log_file, "a") as f:
                f.write(f"    {epoch+1}    |    {train_loss:.6f}    |    {val_loss:.6f}    |    {all_metrics['mae']:.6f}    |    {all_metrics['mse']:.6f}    |    {all_metrics['psnr']:.6f}    |\n")
            
            # 记录到WandB
            if args.use_wandb:
                wandb_log = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                wandb_log.update(all_metrics)
                wandb.log(wandb_log)
        
        # 保存最佳模型 (只在主进程上保存)
        if is_logger and val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # 保存模型
            if config.distributed.use_distributed:
                # 保存DDP模型的module部分
                model_to_save = model.module
            else:
                model_to_save = model
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': all_metrics,
                # 保存归一化参数，用于推理
                'data_dict' : data_dict
            }, best_model_path)
            if len(experts_name) == 1:
                torch.save({
                    'expert_state_dict': model_to_save.experts[0].state_dict()
                }, best_expert_path)
        
        # 打印进度 (只在主进程上打印)
        if is_logger:
            print(f"Epoch {epoch+1}/{config.epochs}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  PSNR: {all_metrics['psnr']:.2f} dB")
            print(f"  MSE: {all_metrics['mse']:.6f}")
            print(f"  MAE: {all_metrics['mae']:.6f}")
        
        # 可视化验证结果 (只在主进程上可视化)
        if is_logger and (epoch + 1) % args.vis_freq == 0:
            # 选择一个批次进行可视化
            vis_batch = next(iter(val_loader))
            inputs = vis_batch['input'].to(device)
            targets = vis_batch['output'].to(device)
            
            with torch.no_grad():
                predictions = model(inputs)
            
            # 反归一化
            predictions = output_inverse_transform(predictions)
            targets = output_inverse_transform(targets)
            
            # 可视化
            visualize_results(
                inputs, targets, predictions,
                save_dir=results_dir / f"vis_epoch_{epoch+1}"
            )
            
            # 将可视化结果记录到WandB
            if args.use_wandb:
                # 选择前3个样本进行可视化
                for i in range(min(3, inputs.shape[0])):
                    # 处理输入（速度图/模型）
                    input_img = inputs[i, 0].cpu().numpy()
                    
                    # 处理目标（地震数据）
                    if len(targets[i].shape) > 2:
                        target_img = targets[i, 0].cpu().numpy()
                    else:
                        target_img = targets[i].cpu().numpy()
                        
                    # 处理预测（地震数据）
                    if len(predictions[i].shape) > 2:
                        pred_img = predictions[i, 0].cpu().numpy()
                    else:
                        pred_img = predictions[i].cpu().numpy()
                    
                    wandb.log({
                        f"sample_{i}/input_velocity": wandb.Image(input_img),
                        f"sample_{i}/target_seismic": wandb.Image(target_img),
                        f"sample_{i}/prediction_seismic": wandb.Image(pred_img)
                    })
        
        # 在分布式环境中同步进程
        if config.distributed.use_distributed:
            torch.distributed.barrier()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
    
    # 保存最终模型 (只在主进程上保存)
    if is_logger:
        final_model_path = results_dir / f"final_model_{experts_name}.pt"
        
        # 保存模型
        if config.distributed.use_distributed:
            # 保存DDP模型的module部分
            model_to_save = model.module
        else:
            model_to_save = model
        
        if len(experts_name) == 1:
            final_expert_path = results_dir / f"final_expert_{experts_name}.pt"
            torch.save({
                'expert_state_dict': model_to_save.experts[0].state_dict()
            }, final_expert_path)
            print(f"训练完成！单一专家{experts_name}保存在: {best_expert_path}")
            print(f"最终单一专家{experts_name}保存在: {final_expert_path}")
            
        torch.save({
            'epoch': config.epochs,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': all_metrics,
            'data_dict' : data_dict
        }, final_model_path)
        
        print(f"训练完成！最佳模型保存在: {best_model_path}")
        print(f"最终模型保存在: {final_model_path}")
        
        # 关闭WandB
        if args.use_wandb:
            wandb.finish()
    
    return model, best_val_loss


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
        self.logger = None
        
    def __getitems__(self, idx):
        # 这个idx是一个batch中各个数据在总数据集中的索引，是一个列表
        if self.logger is None:
            self.logger = logging.getLogger(f"Worker-{os.getpid()}")
            if not self.logger.hasHandlers():
                handler = logging.FileHandler(f"/root/autodl-tmp/FWINO/workers_logs/worker_{os.getpid()}.log")
                handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"Loading indices: {idx[0]}-{idx[-1]}")
        batch_sample = []
        for index in idx:
            sample = self.dataset[self.indices[index]]
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

def load_moe_experts(
    expert_configs: List[Dict[str, Any]],
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    model_path: str
    ) -> List[nn.Module]:
    """
    从 model_path 文件夹中加载所有专家模型，每个模型一个 .pt 文件。
    
    参数：
    - expert_configs: List[Dict[str, Any]],
    - in_channels: int,
    - out_channels: int,
    - hidden_channels: int,
    - model_path: 包含多个专家模型 .pth 文件的目录路径
    
    返回：
    - experts: List[nn.Module]
    """
    if not os.path.isdir(model_path):
        raise ValueError(f"{model_path} 不是有效的目录路径")

    # 获取所有 .pt 文件，best_expert_{experts_name}_{i}.pt
    expert_files = sorted([
        f for f in os.listdir(model_path)
        if f.split('_')[1] == 'expert' and f.endswith('.pt')
    ], key=lambda x : x.split('_')[-1].split('.')[0])

    experts = []
    new_experts = ExpertFactory.create_expert_ensemble(
            expert_configs=expert_configs,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels
        )
    for i, file_name in enumerate(expert_files):
        full_path = os.path.join(model_path, file_name)
        state_dict = torch.load(full_path, map_location='cpu')
        expert = new_experts[i]
        expert.load_state_dict(state_dict)
        for param in expert.parameters():
            param.requires_grad = False
        experts.append(expert)

    return experts
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="地震数据MOE训练和推理")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                        help='运行模式: 训练或推理')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录路径')
    parser.add_argument('--family', type=str, default=None, choices=['vel', 'style', 'fault', 'all'],
                        help='数据集系列: vel, style, fault 或 all')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
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
    parser.add_argument('--use_experts_path', type=str, default=None,
                        help='moe使用的专家模型存放路径')
    parser.add_argument('--use_moe', action='store_true',
                        help='是否使用moe, 使用会冻结专家模型')
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='隐藏通道数（默认值由配置文件决定，可通过此参数覆盖）')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率（默认值由配置文件决定，可通过此参数覆盖）')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='恢复训练的checkpoint路径，如 best_model_xxx.pt')
    # Loss related
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=1.0)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=1.0)
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'inference':
        if not args.model_path:
            raise ValueError("推理模式需要指定模型路径 --model_path")
        run_inference(args)
    else:
        raise ValueError(f"不支持的运行模式: {args.mode}") 
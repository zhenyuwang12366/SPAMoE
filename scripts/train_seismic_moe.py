"""
使用MOE（Mixture of Experts）架构训练地震数据的神经算子模型
支持分布式训练
"""
import optuna
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable
import tqdm
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from torch.utils.data import DataLoader, random_split, Subset, DistributedSampler
from torchvision.transforms import Compose
import transforms as T
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralop.models import MOEOperator, ExpertFactory
from neuralop.training import setup
from neuralop.data.datasets import SeismicDataset, create_seismic_dataloader
from neuralop.utils import get_wandb_api_key, count_model_params
from config.seismic_moe_config import SeismicMOEConfig
import neuralop.mpu.comm as comm
from scripts.scheduler import WarmupMultiStepLR
from neuralop.losses import L1L2Loss, CombinedLoss
from utils import *
 
print("-----------------------------------------------------------")

def run_training(args, trial: Optional["optuna.trial.Trial"] = None):
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
     
    criterion: Callable = L1L2Loss(config.lambda_g1v, config.lambda_g2v)
    
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
    REPORT_EVERY = max(1, getattr(args, "report_every", 5))
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

        # ====== 中间上报（仅 rank0 打印，供外层 Optuna 解析）======
        # 注意：这里假设 stats["val_loss"] 已经做过 all_reduce 得到“全卡平均值”
        if is_logger:
            cur_val = float(stats.get("val_loss", float("nan")))
            # 满足：到达REPORT_EVERY、或最后一个epoch、或提前stop时，打印一次
            if ((epoch + 1) % REPORT_EVERY == 0) or ((epoch + 1) == config.epochs) or (stop_flag == 1):
                print(f"REPORT:{cur_val}:{epoch+1}", flush=True)

        # 可选：若内部早停信号触发，则跳出
        if stop_flag == 1:
            break

    # ====== 训练结束：打印最终指标（仅 rank0）======
    if is_logger:
        # best_val_loss 为整个 trial 的最好验证损失（应已由 rank0 维护）
        print(f"VAL_LOSS:{best_val_loss}", flush=True)
        plot_loss_curve(log_file, save_path=results_dir)

    return model, best_val_loss

def run_overfit_one_sample(args):
    """
    单样本过拟合：
      - 仅取训练集中的一个样本（可用 --overfit_index 指定，默认0）
      - 每个 epoch 重复该样本 steps_per_epoch 次
      - 评价/可视化均在同一样本上进行（便于确认是否可完全拟合）
    """
    # ---------- 额外命令行参数的默认值 ----------
    overfit_index     = getattr(args, "overfit_index", 0)
    steps_per_epoch   = getattr(args, "steps_per_epoch", 64)  # 每个epoch重复次数
    vis_freq          = getattr(args, "vis_freq", 5)
    report_every      = max(1, getattr(args, "report_every", 5))

    # ---------- 配置 & 设备 ----------
    config = SeismicMOEConfig()
    # 单样本过拟合默认禁用分布式，避免 sampler/drop_last 等边界问题
    config.distributed.use_distributed = False
    device, is_logger = setup(config)

    if args.data_dir:
        config.data_dir = args.data_dir
    else:
        config.data_dir = r"/root/autodl-tmp/FWINO/FWINO_data"

    # 覆盖关键超参
    if args.family:          config.family          = args.family
    if args.batch_size:      config.batch_size      = args.batch_size
    if args.epochs:          config.epochs          = args.epochs
    if args.learning_rate:   config.learning_rate   = args.learning_rate
    if args.hidden_channels: config.hidden_channels = args.hidden_channels
    if args.weight_decay:    config.weight_decay    = args.weight_decay
    if args.scheduler_gamma: config.scheduler_gamma = args.scheduler_gamma
    if args.accum_steps is not None: config.accum_steps = args.accum_steps
    if args.output_dir:      config.output_dir      = args.output_dir
    if args.lambda_ssim:     config.lambda_ssim = args.lambda_ssim
    if args.lambda_grad:     config.lambda_grad = args.lambda_grad
    
    # FNO / WNO / MNO / LNO 配置（与 run_training 一致）
    config.expert_configs[0]['n_modes_height'] = args.FNO_n_modes_height
    config.expert_configs[0]['n_modes_width']  = args.FNO_n_modes_width
    config.expert_configs[0]['n_layers']       = args.FNO_n_layers

    config.expert_configs[1]['n_levels_height'] = args.WNO_n_levels_height
    config.expert_configs[1]['n_levels_width']  = args.WNO_n_levels_width
    config.expert_configs[1]['n_layers']        = args.WNO_n_layers
    config.expert_configs[1]['block_n_layers']  = args.WNO_block_n_layers
    config.expert_configs[1]['dropout_rate']    = args.WNO_dropout_rate
    config.expert_configs[1]['wavelet_type']    = args.wavelet_type

    config.expert_configs[2]['n_scales']       = args.MNO_n_scales
    config.expert_configs[2]['scale_factors']  = args.MNO_scale_factors
    config.expert_configs[2]['n_layers']       = args.MNO_n_layers

    config.expert_configs[3]['n_modes']        = tuple(args.LNO_n_modes)
    config.expert_configs[3]['n_layers']       = args.LNO_n_layers

    # 选择专家
    config.top_k = args.top_k
    config.expert_configs = [config.expert_configs[i] for i in args.choose_experts]
    experts_name = []
    for idx, ec in enumerate(config.expert_configs):
        experts_name.append(f"{ec.get('domain_type', ec.get('type','exp'))}_{args.choose_experts[idx]}")
    experts_name_str = "_".join(experts_name)

    # Loss 权重
    config.lambda_g1v = args.lambda_g1v
    config.lambda_g2v = args.lambda_g2v

    # MoE 路由/融合
    if args.router_type:     config.router_type = args.router_type
    if args.fusion_type:     config.fusion_type = args.fusion_type
    if args.s_processor_type: config.s_processor_type = args.s_processor_type
    if args.w_processor_type: config.w_processor_type = args.w_processor_type
    if args.beta:            config.beta = args.beta
    if args.is_specific:     config.is_specific = True
    if args.is_classier:     config.is_classier = True

    # ---------- 数据 ----------
    full_dataset = SeismicDataset(
        data_dir=config.data_dir,
        family=config.family,
        is_specific=config.is_specific,
        split='train',
    )
    if is_logger:
        print(f"[Overfit] full train size = {len(full_dataset)}; use index = {overfit_index}")

    # 取单一样本
    if not (0 <= overfit_index < len(full_dataset)):
        raise IndexError(f"--overfit_index 越界：{overfit_index} / {len(full_dataset)-1}")
    single_raw = Subset(full_dataset, [overfit_index])

    # 统计（用于归一化/反归一化）
    data_dict = full_dataset.getStats()

    # 变换（与 run_training 完全一致）
    from neuralop.data.datasets.seismic_dataset import SeismicDataProcessor
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
    single_ds = TransformedSubset(single_raw, data_processor)  # len==1

    # —— 定义“重复同一条样本”的数据集 —— #
    class RepeatSingleDataset(torch.utils.data.Dataset):
        def __init__(self, base_subset: TransformedSubset, times_per_epoch):
            assert len(base_subset) == 1
            self.base = base_subset
            self.times = int(times_per_epoch)

        def __len__(self):
            return self.times

        def __getitem__(self, idx):
            # 返回同一个样本的深拷贝，避免 in-place 污染
            sample = self.base.__getitems__([0])[0]
            return {
                'input':  sample['input'].clone(),
                'output': sample['output'].clone(),
                'idx':    torch.tensor([int(sample.get('idx', 0))])
            }

    train_dataset = RepeatSingleDataset(single_ds, steps_per_epoch)
    val_dataset   = single_ds  # 评估也用同一条

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )

    # 输入通道
    sample_batch = next(iter(train_loader))
    config.in_channels = sample_batch['input'].shape[1]
    if is_logger:
        print(f"[Overfit] in_channels={config.in_channels}, out_channels={config.out_channels}, hidden={config.hidden_channels}")

    # ---------- 模型 ----------
    if getattr(args, "use_moe", False) and getattr(args, "use_experts_path", None):
        experts = load_moe_experts(
            experts_config=config.expert_configs,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_channels=config.hidden_channels,
            model_path=args.use_experts_path,
            is_specific=False,
            map_location=device,
            type_dict=config.type_id
        )
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
        router_hidden_dim=config.router_hidden_dim,
        is_logger=is_logger,
        router_type=config.router_type,
        s_processor_type=config.s_processor_type,
        w_processor_type=config.w_processor_type,
        beta=config.beta,
        is_specific=config.is_specific,
        is_classier=config.is_classier,
    ).to(device)

    # ---------- 优化器 & 调度 & 损失 ----------
    lr = config.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)

    # 仍复用 WarmupMultiStepLR（以 step 计）
    warmup_iters = int(max(1, config.lr_warmup_epochs) * len(train_loader))
    lr_milestones = [len(train_loader) * m for m in getattr(config, "milestones", [30, 60, 90])]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=config.scheduler_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5
    )

    criterion: Callable = CombinedLoss(
        config.lambda_g1v, 
        config.lambda_g2v, 
        config.lambda_ssim, 
        config.lambda_grad,
    )

    # ---------- 目录 / 日志 ----------
    results_dir = Path(config.output_dir) / f"overfit1_{config.family}"
    if is_logger:
        results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "training_log.txt"
    if is_logger:
        with open(log_file, "w") as f:
            f.write(f"    Epoch    |    Train Loss    |    Val Loss    |    MAE    |    MSE    |    PSNR    |\n")

    best_val_loss = float("inf")
    best_model_path = results_dir / f"best_model_{experts_name_str}.pt"
    best_expert_path = (results_dir / f"best_expert_{experts_name_str}.pt") if len(experts_name)==1 else None

    metrics = SeismicMetrics()

    # ---------- 训练循环（沿用你封装的 train_one_epoch） ----------
    for epoch in range(config.epochs):
        vis_now = (is_logger and ((epoch + 1) % vis_freq == 0))
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
            scheduler_step_mode="per_step",   # 以 step 计
            accum_steps=config.accum_steps,
            vis_now=vis_now,
            visualize_results=visualize_results,
            input_inverse_transform=input_inverse_transform,
            output_inverse_transform=output_inverse_transform,
            use_wandb=False,
            wandb_module=None,
            early_stopper=None,               # overfit 不早停
            best_val_loss=best_val_loss,
            best_model_path=best_model_path,
            best_expert_path=best_expert_path,
            experts_name=experts_name,
            experts_name_str=experts_name_str,
            data_dict=data_dict,
            metrics_module=metrics,
            tqdm_module=tqdm,
            profile_timing=False,
        )

        if is_logger:
            cur_val = float(stats.get("val_loss", float("nan")))
            if ((epoch + 1) % report_every == 0) or ((epoch + 1) == config.epochs):
                print(f"REPORT:{cur_val}:{epoch+1}", flush=True)

    if is_logger:
        print(f"VAL_LOSS:{best_val_loss}", flush=True)
        plot_loss_curve(log_file, save_path=results_dir)

    print(f"[Overfit] 完成。最佳模型保存在：{best_model_path}")

def run_overfit_one_sample_test(args):
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

    # ---------- 配置 & 设备 ----------
    config = SeismicMOEConfig()
    # 单样本过拟合默认禁用分布式，避免 sampler/drop_last 等边界问题
    config.distributed.use_distributed = False
    device, is_logger = setup(config)

    if args.data_dir:
        config.data_dir = args.data_dir
    else:
        config.data_dir = r"/root/autodl-tmp/FWINO/FWINO_data"

    # 覆盖关键超参
    if args.family:          config.family          = args.family
    if args.batch_size:      config.batch_size      = args.batch_size
    if args.epochs:          config.epochs          = args.epochs
    if args.learning_rate:   config.learning_rate   = args.learning_rate
    if args.hidden_channels: config.hidden_channels = args.hidden_channels
    if args.weight_decay:    config.weight_decay    = args.weight_decay
    if args.scheduler_gamma: config.scheduler_gamma = args.scheduler_gamma
    if args.accum_steps is not None: config.accum_steps = args.accum_steps
    if args.output_dir:      config.output_dir      = args.output_dir
    if args.lambda_ssim:     config.lambda_ssim = args.lambda_ssim
    if args.lambda_grad:     config.lambda_grad = args.lambda_grad
    
    # FNO / WNO / MNO / LNO 配置（与 run_training 一致）
    config.expert_configs[0]['n_modes_height'] = args.FNO_n_modes_height
    config.expert_configs[0]['n_modes_width']  = args.FNO_n_modes_width
    config.expert_configs[0]['n_layers']       = args.FNO_n_layers

    config.expert_configs[1]['n_levels_height'] = args.WNO_n_levels_height
    config.expert_configs[1]['n_levels_width']  = args.WNO_n_levels_width
    config.expert_configs[1]['n_layers']        = args.WNO_n_layers
    config.expert_configs[1]['block_n_layers']  = args.WNO_block_n_layers
    config.expert_configs[1]['dropout_rate']    = args.WNO_dropout_rate
    config.expert_configs[1]['wavelet_type']    = args.wavelet_type

    config.expert_configs[2]['n_scales']       = args.MNO_n_scales
    config.expert_configs[2]['scale_factors']  = args.MNO_scale_factors
    config.expert_configs[2]['n_layers']       = args.MNO_n_layers

    config.expert_configs[3]['n_modes']        = tuple(args.LNO_n_modes)
    config.expert_configs[3]['n_layers']       = args.LNO_n_layers

    # 选择专家
    config.top_k = args.top_k
    config.expert_configs = [config.expert_configs[i] for i in args.choose_experts]
    experts_name = []
    for idx, ec in enumerate(config.expert_configs):
        experts_name.append(f"{ec.get('domain_type', ec.get('type','exp'))}_{args.choose_experts[idx]}")
    experts_name_str = "_".join(experts_name)

    # Loss 权重
    config.lambda_g1v = args.lambda_g1v
    config.lambda_g2v = args.lambda_g2v

    # MoE 路由/融合
    if args.router_type:     config.router_type = args.router_type
    if args.fusion_type:     config.fusion_type = args.fusion_type
    if args.s_processor_type: config.s_processor_type = args.s_processor_type
    if args.w_processor_type: config.w_processor_type = args.w_processor_type
    if args.beta:            config.beta = args.beta
    if args.is_specific:     config.is_specific = True
    if args.is_classier:     config.is_classier = True
    
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
    full_dataset = SeismicDataset(
        data_dir=config.data_dir, 
        family=config.family, 
        split='train',
        is_specific=config.is_specific,
    )
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
        return {
            "total": total, 
            "loss_l1": loss_l1.detach(), 
            "loss_g": loss_g.detach(), 
            "w_g": w_g,
        }

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
        pred, _ = model(inp)
        loss_dict = criterion(pred, tgt, epoch)

        total = loss_dict["total"]
        
        # 非有限守护
        if not torch.isfinite(total):
            print(f"[overfit1] non-finite loss at epoch {epoch}: {total.item()}")
            break

        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # 指标统计
        with torch.no_grad():
            mae_val  = float(loss_dict["loss_l1"].item())
            grad_val = float(loss_dict["loss_g"].item())
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
                  f"| grad_norm={grad_norm:.3e} | lr={new_lr:.2e} | w_g={loss_dict['w_g']:.3f}")

        with open(log_file, "a") as f:
            f.write(f"{epoch},{total.item():.6f},{mae_val:.6f},{grad_val:.6f},{mse_mon:.6f},"
                    f"{psnr_val:.3f},{corr_val:.3f},{grad_norm:.3e},{new_lr:.2e},{loss_dict['w_g']:.3f}\n")

        if bad >= EARLY_PATIENCE:
            print(f"Early stop at epoch {epoch}, best Total={best_total:.6e}")
            break

    # ===== 5) 保存最佳与可视化 =====
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, results_dir / f"best_overfit1_{experts_name}.pt")

    model.eval()
    with torch.no_grad():
        pred, _ = model(inp)
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
        
    def __getitems__(self, idx: list[int]):
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

if __name__ == '__main__':
    args = build_argparser_and_parse()
    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'inference':
        if not args.model_path:
            raise ValueError("推理模式需要指定模型路径 --model_path")
        run_inference(args)
    elif args.mode == 'overfit1':
        run_overfit_one_sample(args)
    elif args.mode == 'overfit1_test':
        run_overfit_one_sample_test(args)
    else:
        raise ValueError(f"不支持的运行模式: {args.mode}") 
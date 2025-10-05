"""
使用MOE（Mixture of Experts）架构训练神经算子模型
"""

import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from timeit import default_timer

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralop.models import MOEOperator, ExpertFactory
from neuralop.training import Trainer
from neuralop.data import DataProcessor
from neuralop.data.datasets import DirectDataset, ShallowWaterDataset

from config.moe_config import MOEConfig


def run_training(config=None):
    """
    使用MOE架构训练神经算子模型
    
    Parameters
    ----------
    config : object, optional
        配置对象，默认为None（使用MOEConfig）
    """
    if config is None:
        config = MOEConfig()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    if hasattr(config, 'dataset_name') and config.dataset_name == 'shallow_water':
        train_data = ShallowWaterDataset(
            path=config.train_data_path,
            n_train=config.n_train_samples,
            n_test=0,
            train=True,
            test=False
        )
        test_data = ShallowWaterDataset(
            path=config.test_data_path,
            n_train=0,
            n_test=config.n_test_samples,
            train=False,
            test=True
        )
    else:
        # 默认使用Darcy流数据集
        train_data = DirectDataset(
            train=True,
            n_samples=config.n_train_samples,
            path=config.train_data_path
        )
        test_data = DirectDataset(
            train=False,
            n_samples=config.n_test_samples,
            path=config.test_data_path
        )
    
    # 数据处理器
    data_processor = DataProcessor(
        normalize_inputs=config.normalize_inputs,
        normalize_outputs=config.normalize_outputs,
        standardize=config.standardize,
        smooth_output=config.smooth_output,
        center=config.center,
        channel_dim=config.channel_dim
    )
    
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
        v_type_num=getattr(config, "v_type_num", None)
    )
    
    # 移动模型到设备
    model = model.to(device)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        data_processor=data_processor,
        train_data=train_data,
        test_data=test_data,
        batch_size=config.batch_size,
        test_batch_size=config.test_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        milestones=config.milestones,
        gamma=config.scheduler_gamma,
        device=device
    )
    
    # 打印模型信息
    print(f"创建MOE模型，包含 {len(experts)} 个专家")
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    if hasattr(config, 'model_save_path'):
        save_path = config.model_save_path
    else:
        save_path = 'moe_model.pt'
    
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")
    
    # 评估模型
    print("评估模型...")
    test_loss = trainer.evaluate()
    print(f"测试损失: {test_loss:.6f}")
    
    return model, trainer


if __name__ == '__main__':
    run_training() 

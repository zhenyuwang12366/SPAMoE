"""
用于地震数据的MOE (Mixture of Experts) 神经算子模型配置
"""

from .default_config import Default
from .distributed import DistributedConfig

class SeismicMOEConfig(Default):
    """地震数据MOE神经算子的配置"""
    
    # 基本配置
    model_name = 'MOE'
    in_channels = 1  # 修正为1，与实际输入通道数一致
    out_channels = 1  # 根据输出张量形状更新，输出通道数为5
    hidden_channels = 128
    
    # 数据集配置
    dataset_name = 'seismic'
    data_dir = '/data1/wuruoyu/waveform-inversion'  # 数据目录路径
    family = 'all'  # 数据集系列，可选 'vel', 'style', 'fault' 或 'all'
    n_train_samples = None  # None表示使用所有可用训练样本
    n_test_samples = None  # None表示使用所有可用测试样本
    channel_dim = 0  # 将num_sources作为通道维度
    
    # MOE配置
    use_moe = False
    use_experts_path = None
    top_k = 1  # 选择前k个专家
    noisy_gating = True  # 是否使用噪声门控
    fusion_type = 'linear'  # 专家输出融合方式
    router_hidden_dim = 256  # 路由器隐藏层维度
    router_type = 'basic' # 路由形式 basic,adamv
    
    # 专家配置
    expert_configs = [
        # 傅里叶域专家 - 适合捕捉频率特征 FNO
        {
            'type': 'domain',
            'domain_type': 'fourier',
            'n_dim': 2,
            'n_modes_height': 16,
            'n_modes_width': 16,
            'lifting_channel_ratio': 2,
            'projection_channel_ratio': 2,
            'n_layers': 8,
        },
        # 小波域专家 - 适合处理局部特征和多尺度结构 WNO
        {
            'type': 'domain',
            'domain_type': 'wavelet',
            'n_dim': 2,
            'n_levels_height': 2,  # 减少级别为2，避免形状不匹配问题
            'n_levels_width': 2,   # 减少级别为2，避免形状不匹配问题
            'wavelet_type': 'haar',
            'ensure_even_shapes': True,  # 确保形状为偶数
            'pad_mode': 'reflect',  # 添加填充模式
            'adaptive_padding': True,  # 启用自适应填充
        },
        # 原生多尺度神经算子专家 - 专门处理多尺度地质结构 MNO
        {
            'type': 'scale',
            'scale_expert_type': 'native',  # 更新为scale_expert_type
            'n_dim': 2,
            'n_scales': 3,
            'scale_factors': [1.0, 0.5, 0.25],
            'fusion_mode': 'hierarchical',
            'n_layers': 3,
        },
        # # 局部处理专家 - 用于局部细节重建 LNO
        {
            'type': 'local',
            'local_type': 'basic',  # 更新为basic类型
            'n_dim': 2,
            'n_modes': (16, 16),
            'disco_layers': True,  # 启用DISCO层
            'diff_layers': True,   # 启用差分层
            'n_layers': 3,         # 设置层数
            'default_in_shape': (256, 256),  # 基于输入张量形状设置
        }
    ]
    
    # 训练配置
    batch_size = 8
    test_batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-4
    epochs = 100
    milestones = [60,90,110]
    scheduler_gamma = 0.3
    lambda_g1v = 1.0
    lambda_g2v = 1.0
    output_dir = './results'
    lr_warmup_epochs = 5
    
    use_onecycle = True
    
    accum_steps = 1
    use_amp = True
    
    early_stop = False
    early_stop_patience = 30
    early_stop_min_delta = 0.001
    early_stop_warmup_epochs = 10
    
    # v_type_id dict
    type_id_specific = {
        'curve_vel': 0,
        'curve_fault': 1,
        'flat_vel': 2,
        'flat_fault': 3,
        'style_style': 4,
    }
    type_id_normal = {
        'vel': 0,
        'fault': 1,
        'style': 2,
    }
    type_id = {
        'specific': type_id_specific,
        'normal': type_id_normal,
    }
    
    
    # 分布式训练配置
    distributed = DistributedConfig(
        use_distributed=False,
        model_parallel_size=1,
        seed=42
    )
    
    # 混合精度训练
    mixed_precision = False
    
    # 评估配置
    eval_interval = 1
    verbose = True
    
    # 损失函数配置
    loss_fn = 'mse and mae'  # 均方误差损失
    
    # 评估指标配置
    metrics = ['mse', 'mae', 'psnr']
    
    # WandB配置
    wandb = {
        'log': False,
        'project': 'seismic_moe',
        'group': None,
        'name': None,
        'entity': None,
        'log_output': False,
        'sweep': False
    } 
"""
使用任务感知路由器的MOE (Mixture of Experts) 神经算子模型配置
"""

from .default_config import DefaultConfig

class TaskAwareMOEConfig(DefaultConfig):
    """使用任务感知路由器的MOE神经算子配置"""
    
    # 基本配置
    model_name = 'MOE'
    in_channels = 3  # 输入通道数
    out_channels = 1  # 输出通道数
    hidden_channels = 64
    
    # 数据集配置
    dataset_name = 'multi_task'
    data_dir = './data'
    n_train_samples = None  # None表示使用所有可用训练样本
    n_test_samples = None  # None表示使用所有可用测试样本
    normalize_inputs = True
    normalize_outputs = True
    
    # MOE配置
    top_k = 2  # 选择前k个专家
    noisy_gating = True  # 是否使用噪声门控
    fusion_type = 'linear'  # 专家输出融合方式
    router_hidden_dim = 256  # 路由器隐藏层维度
    
    # 任务感知路由器配置
    router_type = 'task_aware'  # 使用任务感知路由器
    task_dim = 8  # 任务特征维度
    routing_mode = 'both'  # 同时使用输入和任务特征进行路由
    
    # 任务配置
    tasks = [
        {
            'name': 'darcy',
            'weight': 1.0,
            'embedding': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            'name': 'burgers',
            'weight': 1.0,
            'embedding': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            'name': 'navier_stokes',
            'weight': 1.0,
            'embedding': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            'name': 'seismic',
            'weight': 1.0,
            'embedding': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        }
    ]
    
    # 专家配置
    expert_configs = [
        # 傅里叶域专家 - 适合捕捉频率特征
        {
            'type': 'domain',
            'domain_type': 'fourier',
            'n_dim': 2,
            'n_modes_height': 16,
            'n_modes_width': 16,
            'lifting_channel_ratio': 2,
            'projection_channel_ratio': 2,
        },
        # 小波域专家 - 适合处理局部特征和多尺度结构
        {
            'type': 'domain',
            'domain_type': 'wavelet',
            'n_dim': 2,
            'n_levels_height': 4,
            'n_levels_width': 4,
            'wavelet_type': 'haar',
        },
        # 多尺度专家 - 适合处理多尺度物理现象
        {
            'type': 'scale',
            'expert_type': 'native',
            'n_dim': 2,
            'n_scales': 3,
            'scale_factors': [1.0, 0.5, 0.25],
            'fusion_mode': 'hierarchical',
            'n_layers': 3,
        },
        # 几何专家 - 适合处理非规则网格
        {
            'type': 'geometry',
            'geometry_type': 'irregular',
            'n_neighbors': 8,
            'kernel_size': 3,
            'n_layers': 4,
        }
    ]
    
    # 训练配置
    use_distributed = False
    batch_size = 8
    test_batch_size = 8
    learning_rate = 1e-3
    weight_decay = 1e-4
    epochs = 100
    milestones = [30, 60, 90]
    scheduler_gamma = 0.5
    
    # 损失函数配置
    loss_fn = 'mse'  # 均方误差损失
    
    # 评估指标配置
    metrics = ['mse', 'rel_l2_error'] 
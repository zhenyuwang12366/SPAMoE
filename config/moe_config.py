"""
MOE (Mixture of Experts) 神经算子模型的配置
"""

from .default_config import Default

class MOEConfig(Default):
    """MOE神经算子的配置"""
    
    # 基本配置
    model_name = 'MOE'
    in_channels = 3
    out_channels = 1
    hidden_channels = 64
    
    # MOE配置
    top_k = 2  # 选择前k个专家
    noisy_gating = True  # 是否使用噪声门控
    fusion_type = 'linear'  # 专家输出融合方式
    router_hidden_dim = 256  # 路由器隐藏层维度
    
    # 专家配置
    expert_configs = [
        # # 傅里叶域专家
        # {
        #     'type': 'domain',
        #     'domain_type': 'fourier',
        #     'n_dim': 2,
        #     'n_modes_height': 16,
        #     'n_modes_width': 16,
        # },
        # 小波域专家
        {
            'type': 'domain',
            'domain_type': 'wavelet',
            'n_dim': 2,
            'n_levels_height': 4,
            'n_levels_width': 4,
            'wavelet_type': 'haar',
        },
        # # 多尺度傅里叶专家（包装器类型）
        # {
        #     'type': 'scale',
        #     'expert_type': 'wrapper',
        #     'base_expert': {
        #         'type': 'domain',
        #         'domain_type': 'fourier',
        #         'n_dim': 2,
        #         'n_modes_height': 12,
        #         'n_modes_width': 12,
        #     },
        #     'n_scales': 3,
        #     'scale_factors': [1.0, 0.5, 0.25],
        #     'fusion_type': 'adaptive',
        # },
        # # 原生多尺度神经算子专家
        # {
        #     'type': 'scale',
        #     'expert_type': 'native',
        #     'n_dim': 2,
        #     'n_scales': 3,
        #     'scale_factors': [1.0, 0.5, 0.25],
        #     'fusion_mode': 'hierarchical',
        #     'n_layers': 3,
        # },
        # # 几何专家（非规则网格）
        # {
        #     'type': 'geometry',
        #     'geometry_type': 'irregular',
        #     'gno_coord_dim': 2,
        #     'fno_hidden_channels': 64,
        #     'in_gno_radius': 0.05,
        #     'out_gno_radius': 0.05,
        # },
    ]
    
    # 训练配置
    use_distributed = False
    batch_size = 16
    test_batch_size = 16
    learning_rate = 1e-3
    weight_decay = 1e-4
    epochs = 500
    milestones = [100, 200, 300, 400]
    scheduler_gamma = 0.5 
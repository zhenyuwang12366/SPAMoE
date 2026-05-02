"""
MOE (Mixture of Experts) neural operator configuration for seismic / OpenFWI-style data.
"""

from .default_config import Default
from .distributed import DistributedConfig

# Fine-grained type keys and variants
SPECIFIC_TYPE_VARIANTS = {
    'curve_vel': ('curve_vel_a', 'curve_vel_b'),
    'curve_fault': ('curve_fault_a', 'curve_fault_b'),
    'flat_vel': ('flat_vel_a', 'flat_vel_b'),
    'flat_fault': ('flat_fault_a', 'flat_fault_b'),
    'style_style': ('style_style_a', 'style_style_b'),
    'all': 'all',
}


class SeismicMOEConfig(Default):
    """Configuration for seismic MOE neural operators."""

    # Core model
    model_name = 'MOE'
    in_channels = 1  # Match stacked / single-channel input
    out_channels = 1  # Velocity map channels (often 1 for OpenFWI)
    is_resize = False
    H_size = 256
    W_size = 256
    concat_channels = True
    moe_mode = "standard"    # standard | group | velocity_type
    moe_method = "afmoe"
    use_gpu_proxy = False
    train_encoder = False
    use_encoder = True
    backbone = 'vit'
    enable_freq_metrics = False  # Log per-band metrics during inference
    # v_type_id dict
    type_id_specific = {
        'curve_vel_a': 0,
        'curve_vel_b': 1,
        'curve_fault_a': 2,
        'curve_fault_b': 3,
        'flat_vel_a': 4,
        'flat_vel_b': 5,
        'flat_fault_a': 6,
        'flat_fault_b': 7,
        'style_style_a': 8,
        'style_style_b': 9,
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

    # Dataset
    dataset_name = 'seismic'
    data_dir = '/data1/wuruoyu/waveform-inversion'  # Dataset root
    family = 'all'  # 'vel' | 'style' | 'fault' | 'all' or fine-grained keys
    n_train_samples = None  # None = use all train samples
    n_test_samples = None  # None = use all test samples
    channel_dim = 0  # Treat num_sources as channel dim when stacking

    # MoE routing
    use_moe = False
    use_experts_path = None
    top_k = 2  # Top-k experts
    noisy_gating = True
    fusion_type = 'linear'  # Expert fusion style
    router_hidden_dim = 256
    router_type = 'basic'  # basic | adamv | ...
    # AFreqMoE router / ablation toggles
    band_sharpness = 20.0
    freq_affinity_sharpness = 10.0
    use_soft_bands = True
    enable_freq_attn = True
    enable_band_mixing = True
    enable_band_decomposition = True
    routing_mode = "learned"  # learned | uniform | random
    s_processor_type = 'linear'
    w_processor_type = 'linear'
    beta = 0.5
    is_specific = True
    is_classifier = False
    v_type_num = 0
    router_alpha = 0.1
    
    # Expert templates (shared); per-type overrides live in load_expert_configs
    expert_configs = [
        # Fourier-domain expert (FNO)
        {
            'type': 'domain',
            'domain_type': 'fourier',
            'n_dim': 2,
            'n_modes_height': 16,
            'n_modes_width': 16,
            'lifting_channel_ratio': 2,
            'projection_channel_ratio': 2,
            'n_layers': 4,
        },
        # # Wavelet-domain expert (WNO) — optional
        # {
        #     'type': 'domain',
        #     'domain_type': 'wavelet',
        #     'n_dim': 2,
        #     'n_levels_height': 2,
        #     'n_levels_width': 2,
        #     'conv_kind': 'dwt',
        #     'wavelet': 'db6',
        #     'biort': 'near_sym_b',
        #     'qshift': 'qshift_b',
        #     'n_layers': 4,
        #     'dropout_rate': 0.10,
        # },
        # Native multiscale expert (MNO)
        {
            'type': 'scale',
            'scale_expert_type': 'native',
            'n_dim': 2,
            'n_scales': 3,
            'scale_factors': [1.0, 0.6, 0.3],
            'fusion_mode': 'hierarchical',
            'n_layers': 4,
        },
        # Local spectral expert (LNO)
        {
            'type': 'local',
            'local_type': 'basic',
            'n_dim': 2,
            'n_modes': (16, 16),
            'disco_layers': True,
            'diff_layers': True,
            'n_layers': 3,
            'default_in_shape': (70, 70),
        },
        # # Geometry-aware expert (GeoFNO) — optional
        # {
        #     'type': 'geometry',
        #     'geometry_type': 'geofno',
        #     'modes1': 32,
        #     'modes2': 32,
        #     'n_fourier_layers': 5,
        #     'code_dim': 42,
        #     's1': 1000,
        #     's2': 350,
        #     'is_mesh': True,
        # }
    ]
    
    # Optimization
    batch_size = 8
    test_batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-4
    epochs = 100
    milestones = [60,90,110]
    scheduler_gamma = 0.3
    output_dir = './results'
    log_root = './runs'
    lr_warmup_epochs = 5
    lr_warmup_factor = 1.0 / 3
    lr_warmup_method = 'linear'

    lr_scheduler_type = 'cos_restart'  # cos_restart, cos, multistep
    lr_cosine_eta_min = 1e-6
    lr_cosine_tmax_epochs = 50
    lr_cosine_restart_t0_epochs = 10
    lr_cosine_restart_t_mult = 2
    
    use_onecycle = True
    
    accum_steps = 1
    use_amp = False
    
    early_stop = False
    early_stop_patience = 30
    early_stop_min_delta = 0.001
    early_stop_warmup_epochs = 10
    
    # Distributed training
    distributed = DistributedConfig(
        use_distributed=False,
        model_parallel_size=1,
        seed=42
    )
    
    # Mixed precision
    mixed_precision = False
    
    # Evaluation cadence
    eval_interval = 1
    verbose = True
    
    # Loss weights
    loss_fn = 'mse and mae'
    lambda_g1v = 0.6
    lambda_g2v = 0.4
    lambda_grad_l1 = 0.15
    lambda_fourier_mag_l1 = 0.10
    lambda_ce = 0.2
    
    # Metrics to log
    metrics = ['mse', 'mae', 'psnr']
    
    # Weights & Biases
    wandb = {
        'log': False,
        'project': 'seismic_moe',
        'group': None,
        'name': None,
        'entity': None,
        'log_output': False,
        'sweep': False
    } 
    load_expert_configs = [
        {
            # FNO
            type_id_specific['flat_vel_a']: {
                'type': 'domain',
                'domain_type': 'fourier',
                'n_dim': 2,
                'n_modes_height': 16,
                'n_modes_width': 16,
                'lifting_channel_ratio': 2,
                'projection_channel_ratio': 2,
                'n_layers': 4,
                'hc': 96,
            },
            type_id_specific['flat_vel_b']: {
                'type': 'domain',
                'domain_type': 'fourier',
                'n_dim': 2,
                'n_modes_height': 16,
                'n_modes_width': 16,
                'lifting_channel_ratio': 2,
                'projection_channel_ratio': 2,
                'n_layers': 4,
                'hc': 96,
            },
            type_id_specific['flat_fault_a']: {
                'type': 'domain',
                'domain_type': 'fourier',
                'n_dim': 2,
                'n_modes_height': 16,
                'n_modes_width': 16,
                'lifting_channel_ratio': 2,
                'projection_channel_ratio': 2,
                'n_layers': 4,
                'hc': 96,
            },
            type_id_specific['flat_fault_b']: {
                'type': 'domain',
                'domain_type': 'fourier',
                'n_dim': 2,
                'n_modes_height': 16,
                'n_modes_width': 16,
                'lifting_channel_ratio': 2,
                'projection_channel_ratio': 2,
                'n_layers': 4,
                'hc': 96,
            },
            type_id_specific['curve_vel_a']: {
                'type': 'domain',
                'domain_type': 'fourier',
                'n_dim': 2,
                'n_modes_height': 16,
                'n_modes_width': 16,
                'lifting_channel_ratio': 2,
                'projection_channel_ratio': 2,
                'n_layers': 4,
                'hc': 96,
            },
            type_id_specific['curve_vel_b']: {
                'type': 'domain',
                'domain_type': 'fourier',
                'n_dim': 2,
                'n_modes_height': 16,
                'n_modes_width': 16,
                'lifting_channel_ratio': 2,
                'projection_channel_ratio': 2,
                'n_layers': 4,
                'hc': 96,
            },
            type_id_specific['curve_fault_a']: {
                'type': 'domain',
                'domain_type': 'fourier',
                'n_dim': 2,
                'n_modes_height': 16,
                'n_modes_width': 16,
                'lifting_channel_ratio': 2,
                'projection_channel_ratio': 2,
                'n_layers': 4,
                'hc': 96,
            },
            type_id_specific['curve_fault_b']: {
                'type': 'domain',
                'domain_type': 'fourier',
                'n_dim': 2,
                'n_modes_height': 16,
                'n_modes_width': 16,
                'lifting_channel_ratio': 2,
                'projection_channel_ratio': 2,
                'n_layers': 4,
                'hc': 96,
            },
            type_id_specific['style_style_a']: {
                'type': 'domain',
                'domain_type': 'fourier',
                'n_dim': 2,
                'n_modes_height': 16,
                'n_modes_width': 16,
                'lifting_channel_ratio': 2,
                'projection_channel_ratio': 2,
                'n_layers': 4,
                'hc': 96,
            },
            type_id_specific['style_style_b']: {
                'type': 'domain',
                'domain_type': 'fourier',
                'n_dim': 2,
                'n_modes_height': 16,
                'n_modes_width': 16,
                'lifting_channel_ratio': 2,
                'projection_channel_ratio': 2,
                'n_layers': 4,
                'hc': 96,
            },
        },
        {
            # WNO
            type_id_specific['flat_vel_a']: {
                'type': 'domain',
                'domain_type': 'wavelet',
                'n_dim': 2,
                'n_levels_height': 1,
                'n_levels_width': 3,
                'conv_kind': 'dwt',
                'wavelet': 'coif4',
                'biort': 'near_sym_b',
                'qshift': 'qshift_b',
                'n_layers': 4,
                'dropout_rate': 0.10,
                'hc': 96,
            },
            type_id_specific['flat_vel_b']: {
                'type': 'domain',
                'domain_type': 'wavelet',
                'n_dim': 2,
                'n_levels_height': 1,
                'n_levels_width': 3,
                'conv_kind': 'dwt',
                'wavelet': 'coif4',
                'biort': 'near_sym_b',
                'qshift': 'qshift_b',
                'n_layers': 4,
                'dropout_rate': 0.10,
                'hc': 96,
            },
            type_id_specific['flat_fault_a']: {
                'type': 'domain',
                'domain_type': 'wavelet',
                'n_dim': 2,
                'n_levels_height': 1,
                'n_levels_width': 3,
                'conv_kind': 'dwt',
                'wavelet': 'coif4',
                'biort': 'near_sym_b',
                'qshift': 'qshift_b',
                'n_layers': 4,
                'dropout_rate': 0.10,
                'hc': 96,
            },
            type_id_specific['flat_fault_b']: {
                'type': 'domain',
                'domain_type': 'wavelet',
                'n_dim': 2,
                'n_levels_height': 1,
                'n_levels_width': 3,
                'conv_kind': 'dwt',
                'wavelet': 'coif4',
                'biort': 'near_sym_b',
                'qshift': 'qshift_b',
                'n_layers': 4,
                'dropout_rate': 0.10,
                'hc': 96,
            },
            type_id_specific['curve_vel_a']: {
                'type': 'domain',
                'domain_type': 'wavelet',
                'n_dim': 2,
                'n_levels_height': 1,
                'n_levels_width': 3,
                'conv_kind': 'dwt',
                'wavelet': 'coif4',
                'biort': 'near_sym_b',
                'qshift': 'qshift_b',
                'n_layers': 4,
                'dropout_rate': 0.10,
                'hc': 96,
            },
            type_id_specific['curve_vel_b']: {
                'type': 'domain',
                'domain_type': 'wavelet',
                'n_dim': 2,
                'n_levels_height': 1,
                'n_levels_width': 3,
                'conv_kind': 'dwt',
                'wavelet': 'coif4',
                'biort': 'near_sym_b',
                'qshift': 'qshift_b',
                'n_layers': 4,
                'dropout_rate': 0.10,
                'hc': 96,
            },
            type_id_specific['curve_fault_a']: {
                'type': 'domain',
                'domain_type': 'wavelet',
                'n_dim': 2,
                'n_levels_height': 1,
                'n_levels_width': 3,
                'conv_kind': 'dwt',
                'wavelet': 'coif4',
                'biort': 'near_sym_b',
                'qshift': 'qshift_b',
                'n_layers': 4,
                'dropout_rate': 0.10,
                'hc': 96,
            },
            type_id_specific['curve_fault_b']: {
                'type': 'domain',
                'domain_type': 'wavelet',
                'n_dim': 2,
                'n_levels_height': 1,
                'n_levels_width': 3,
                'conv_kind': 'dwt',
                'wavelet': 'db6',
                'biort': 'near_sym_b',
                'qshift': 'qshift_b',
                'n_layers': 4,
                'dropout_rate': 0.10,
                'hc': 96,
            },
            type_id_specific['style_style_a']: {
                'type': 'domain',
                'domain_type': 'wavelet',
                'n_dim': 2,
                'n_levels_height': 1,
                'n_levels_width': 3,
                'conv_kind': 'dwt',
                'wavelet': 'db6',
                'biort': 'near_sym_b',
                'qshift': 'qshift_b',
                'n_layers': 4,
                'dropout_rate': 0.10,
                'hc': 96,
            },
            type_id_specific['style_style_b']: {
                'type': 'domain',
                'domain_type': 'wavelet',
                'n_dim': 2,
                'n_levels_height': 1,
                'n_levels_width': 3,
                'conv_kind': 'dwt',
                'wavelet': 'db6',
                'biort': 'near_sym_b',
                'qshift': 'qshift_b',
                'n_layers': 4,
                'dropout_rate': 0.10,
                'hc': 96,
            },
        },
        {
            # MNO
            type_id_specific['flat_vel_a']: {
                'type': 'scale',
                'scale_expert_type': 'native',
                'n_dim': 2,
                'n_scales': 3,
                'scale_factors': [1.0, 0.6, 0.3],
                'fusion_mode': 'hierarchical',
                'n_layers': 3,
                'hc': 96,
            },
            type_id_specific['flat_vel_b']: {
                'type': 'scale',
                'scale_expert_type': 'native',
                'n_dim': 2,
                'n_scales': 3,
                'scale_factors': [1.0, 0.6, 0.3],
                'fusion_mode': 'hierarchical',
                'n_layers': 3,
                'hc': 96,
            },
            type_id_specific['flat_fault_a']: {
                'type': 'scale',
                'scale_expert_type': 'native',
                'n_dim': 2,
                'n_scales': 3,
                'scale_factors': [1.0, 0.6, 0.3],
                'fusion_mode': 'hierarchical',
                'n_layers': 3,
                'hc': 96,
            },
            type_id_specific['flat_fault_b']: {
                'type': 'scale',
                'scale_expert_type': 'native',
                'n_dim': 2,
                'n_scales': 3,
                'scale_factors': [1.0, 0.5, 0.25],
                'fusion_mode': 'hierarchical',
                'n_layers': 3,
                'hc': 96,
            },
            type_id_specific['curve_vel_a']: {
                'type': 'scale',
                'scale_expert_type': 'native',
                'n_dim': 2,
                'n_scales': 3,
                'scale_factors': [1.0, 0.6, 0.3],
                'fusion_mode': 'hierarchical',
                'n_layers': 3,
                'hc': 96,
            },
            type_id_specific['curve_vel_b']: {
                'type': 'scale',
                'scale_expert_type': 'native',
                'n_dim': 2,
                'n_scales': 3,
                'scale_factors': [1.0, 0.6, 0.3],
                'fusion_mode': 'hierarchical',
                'n_layers': 3,
                'hc': 96,
            },
            type_id_specific['curve_fault_a']: {
                'type': 'scale',
                'scale_expert_type': 'native',
                'n_dim': 2,
                'n_scales': 3,
                'scale_factors': [1.0, 0.6, 0.3],
                'fusion_mode': 'hierarchical',
                'n_layers': 3,
                'hc': 96,
            },
            type_id_specific['curve_fault_b']: {
                'type': 'scale',
                'scale_expert_type': 'native',
                'n_dim': 2,
                'n_scales': 3,
                'scale_factors': [1.0, 0.6, 0.3],
                'fusion_mode': 'hierarchical',
                'n_layers': 3,
                'hc': 96,
            },
            type_id_specific['style_style_a']: {
                'type': 'scale',
                'scale_expert_type': 'native',
                'n_dim': 2,
                'n_scales': 3,
                'scale_factors': [1.0, 0.6, 0.3],
                'fusion_mode': 'hierarchical',
                'n_layers': 3,
                'hc': 96,
            },
            type_id_specific['style_style_b']: {
                'type': 'scale',
                'scale_expert_type': 'native',
                'n_dim': 2,
                'n_scales': 3,
                'scale_factors': [1.0, 0.6, 0.3],
                'fusion_mode': 'hierarchical',
                'n_layers': 3,
                'hc': 96,
            },
        },
        {
            # LNO
            type_id_specific['flat_vel_a']: {
                'type': 'local',
                'local_type': 'basic',
                'n_dim': 2,
                'n_modes': (16, 16),
                'disco_layers': True,
                'diff_layers': True,
                'n_layers': 4,
                'default_in_shape': (256, 256),
                'hc': 96,
            },
            type_id_specific['flat_vel_b']: {
                'type': 'local',
                'local_type': 'basic',
                'n_dim': 2,
                'n_modes': (16, 16),
                'disco_layers': True,
                'diff_layers': True,
                'n_layers': 4,
                'default_in_shape': (256, 256),
                'hc': 96,
            },
            type_id_specific['flat_fault_a']: {
                'type': 'local',
                'local_type': 'basic',
                'n_dim': 2,
                'n_modes': (16, 16),
                'disco_layers': True,
                'diff_layers': True,
                'n_layers': 4,
                'default_in_shape': (256, 256),
                'hc': 96,
            },
            type_id_specific['flat_fault_b']: {
                'type': 'local',
                'local_type': 'basic',
                'n_dim': 2,
                'n_modes': (16, 16),
                'disco_layers': True,
                'diff_layers': True,
                'n_layers': 4,
                'default_in_shape': (256, 256),
                'hc': 96,
            },
            type_id_specific['curve_vel_a']: {
                'type': 'local',
                'local_type': 'basic',
                'n_dim': 2,
                'n_modes': (16, 16),
                'disco_layers': True,
                'diff_layers': True,
                'n_layers': 4,
                'default_in_shape': (256, 256),
                'hc': 96,
            },
            type_id_specific['curve_vel_b']: {
                'type': 'local',
                'local_type': 'basic',
                'n_dim': 2,
                'n_modes': (16, 16),
                'disco_layers': True,
                'diff_layers': True,
                'n_layers': 4,
                'default_in_shape': (256, 256),
                'hc': 96,
            },
            type_id_specific['curve_fault_a']: {
                'type': 'local',
                'local_type': 'basic',
                'n_dim': 2,
                'n_modes': (16, 16),
                'disco_layers': True,
                'diff_layers': True,
                'n_layers': 4,
                'default_in_shape': (256, 256),
                'hc': 96,
            },
            type_id_specific['curve_fault_b']: {
                'type': 'local',
                'local_type': 'basic',
                'n_dim': 2,
                'n_modes': (16, 16),
                'disco_layers': True,
                'diff_layers': True,
                'n_layers': 4,
                'default_in_shape': (256, 256),
                'hc': 96,
            },
            type_id_specific['style_style_a']: {
                'type': 'local',
                'local_type': 'basic',
                'n_dim': 2,
                'n_modes': (16, 16),
                'disco_layers': True,
                'diff_layers': True,
                'n_layers': 4,
                'default_in_shape': (256, 256),
                'hc': 96,
            },
            type_id_specific['style_style_b']: {
                'type': 'local',
                'local_type': 'basic',
                'n_dim': 2,
                'n_modes': (16, 16),
                'disco_layers': True,
                'diff_layers': True,
                'n_layers': 4,
                'default_in_shape': (256, 256),
                'hc': 96,
            },
        },
    ]

"""
MOE configuration with a task-aware router.
"""

from .default_config import DefaultConfig

class TaskAwareMOEConfig(DefaultConfig):
    """MOE neural operator with task-conditioned routing."""

    # Core model
    model_name = 'MOE'
    in_channels = 3
    out_channels = 1
    hidden_channels = 64

    # Dataset
    dataset_name = 'multi_task'
    data_dir = './data'
    n_train_samples = None  # None = all train samples
    n_test_samples = None   # None = all test samples
    normalize_inputs = True
    normalize_outputs = True

    # MoE routing
    top_k = 2
    noisy_gating = True
    fusion_type = 'linear'
    router_hidden_dim = 256

    # Task-aware router
    router_type = 'task_aware'
    task_dim = 8
    routing_mode = 'both'  # use both input and task embeddings

    # Task metadata
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

    # Expert templates
    expert_configs = [
        # Fourier expert
        {
            'type': 'domain',
            'domain_type': 'fourier',
            'n_dim': 2,
            'n_modes_height': 16,
            'n_modes_width': 16,
            'lifting_channel_ratio': 2,
            'projection_channel_ratio': 2,
        },
        # Wavelet expert
        {
            'type': 'domain',
            'domain_type': 'wavelet',
            'n_dim': 2,
            'n_levels_height': 4,
            'n_levels_width': 4,
            'wavelet_type': 'haar',
        },
        # Multiscale expert
        {
            'type': 'scale',
            'expert_type': 'native',
            'n_dim': 2,
            'n_scales': 3,
            'scale_factors': [1.0, 0.5, 0.25],
            'fusion_mode': 'hierarchical',
            'n_layers': 3,
        },
        # Geometry expert (irregular grids)
        {
            'type': 'geometry',
            'geometry_type': 'irregular',
            'n_neighbors': 8,
            'kernel_size': 3,
            'n_layers': 4,
        }
    ]

    # Optimization
    use_distributed = False
    batch_size = 8
    test_batch_size = 8
    learning_rate = 1e-3
    weight_decay = 1e-4
    epochs = 100
    milestones = [30, 60, 90]
    scheduler_gamma = 0.5

    # Loss / metrics
    loss_fn = 'mse'
    metrics = ['mse', 'rel_l2_error']

"""
MOE (Mixture of Experts) neural operator configuration.
"""

from .default_config import Default

class MOEConfig(Default):
    """Baseline MOE neural operator settings."""

    # Core model
    model_name = 'MOE'
    in_channels = 3
    out_channels = 1
    hidden_channels = 64

    # MoE routing
    top_k = 2
    noisy_gating = True
    fusion_type = 'linear'
    router_hidden_dim = 256

    # Expert templates
    expert_configs = [
        # # Fourier expert
        # {
        #     'type': 'domain',
        #     'domain_type': 'fourier',
        #     'n_dim': 2,
        #     'n_modes_height': 16,
        #     'n_modes_width': 16,
        # },
        # Wavelet expert
        {
            'type': 'domain',
            'domain_type': 'wavelet',
            'n_dim': 2,
            'n_levels_height': 4,
            'n_levels_width': 4,
            'wavelet_type': 'haar',
        },
        # # Multiscale Fourier wrapper expert
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
        # # Native multiscale expert
        # {
        #     'type': 'scale',
        #     'expert_type': 'native',
        #     'n_dim': 2,
        #     'n_scales': 3,
        #     'scale_factors': [1.0, 0.5, 0.25],
        #     'fusion_mode': 'hierarchical',
        #     'n_layers': 3,
        # },
        # # Geometry expert (irregular grids)
        # {
        #     'type': 'geometry',
        #     'geometry_type': 'irregular',
        #     'gno_coord_dim': 2,
        #     'fno_hidden_channels': 64,
        #     'in_gno_radius': 0.05,
        #     'out_gno_radius': 0.05,
        # },
    ]

    # Optimization
    use_distributed = False
    batch_size = 16
    test_batch_size = 16
    learning_rate = 1e-3
    weight_decay = 1e-4
    epochs = 500
    milestones = [100, 200, 300, 400]
    scheduler_gamma = 0.5

import argparse
import json
import sys
from pathlib import Path

from config.seismic_moe_config import SPECIFIC_TYPE_VARIANTS

# ====== Build family (dataset series) choices ======
_SPECIFIC_BASE_FAMILIES = set(SPECIFIC_TYPE_VARIANTS.keys())
_SPECIFIC_VARIANT_FAMILIES = {
    variant for variants in SPECIFIC_TYPE_VARIANTS.values() for variant in variants
}
_FAMILY_CHOICES = ['vel', 'style', 'fault', 'all']
_FAMILY_CHOICES.extend(sorted(_SPECIFIC_BASE_FAMILIES | _SPECIFIC_VARIANT_FAMILIES))


def _normalize_argv(argv=None) -> list[str]:
    if argv is None:
        return list(sys.argv[1:])
    return list(argv)


def _load_json_payload(file_path: str) -> dict:
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve_payload_defaults(
    parser: argparse.ArgumentParser,
    raw_config: dict,
    *,
    ignore_unknown_keys: set[str] | None = None,
) -> tuple[dict, set[str]]:
    if not raw_config:
        return {}, set()
    ignore_unknown_keys = ignore_unknown_keys or set()

    actions_by_dest = {}
    actions_by_option = {}
    for action in parser._actions:
        if not action.dest or action.dest == "help":
            continue
        actions_by_dest[action.dest] = action
        for option_string in action.option_strings:
            if option_string.startswith("--"):
                actions_by_option[option_string.lstrip("-")] = action

    normalized = {}
    specified_dests = set()
    unknown_keys = []

    for raw_key, value in raw_config.items():
        key = str(raw_key).lstrip("-")
        action = actions_by_dest.get(key)
        from_option_alias = False

        if action is None:
            action = actions_by_option.get(key)
            from_option_alias = action is not None

        if action is None:
            if str(raw_key) in ignore_unknown_keys or key in ignore_unknown_keys:
                continue
            unknown_keys.append(str(raw_key))
            continue

        dest = action.dest
        if dest in normalized:
            raise ValueError(f"Duplicate config key maps to same field: {raw_key} -> {dest}")

        if from_option_alias and isinstance(action, argparse._StoreTrueAction):
            normalized[dest] = bool(value)
        elif from_option_alias and isinstance(action, argparse._StoreFalseAction):
            normalized[dest] = not bool(value)
        else:
            normalized[dest] = value

        specified_dests.add(dest)

    if unknown_keys:
        raise ValueError(f"Config contains unknown keys: {sorted(unknown_keys)}")

    return normalized, specified_dests


def _collect_user_specified_args(
    parser: argparse.ArgumentParser,
    argv: list[str],
    config_specified_dests: set[str],
) -> set[str]:
    option_to_dest = {}
    for action in parser._actions:
        if not action.dest or action.dest == "help":
            continue
        for option_string in action.option_strings:
            option_to_dest[option_string] = action.dest

    specified = set(config_specified_dests)
    for token in argv:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            specified.add(dest)

    return specified


def build_argparser_and_parse(argv=None) -> argparse.Namespace:
    """
    Build the CLI argument parser for:
    - Training and inference on seismic data with FNO/WNO/MNO/LNO + MoE
    - Defaults often come from JSON config files; CLI can override when applicable
    """
    argv = _normalize_argv(argv)

    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config; keys load as defaults first, then explicit CLI overrides apply"
    )
    bootstrap_parser.add_argument(
        "--args", dest="args_file", type=str, default=None,
        help="Path to args.json; when set, training args are taken entirely from that file (no other CLI overrides)"
    )
    bootstrap_args, _ = bootstrap_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="Seismic MOE: training and inference")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config; keys load as defaults first, then explicit CLI overrides apply"
    )
    parser.add_argument(
        "--args", dest="args_file", type=str, default=None,
        help="Path to args.json; when set, training args are taken entirely from that file (no other CLI overrides)"
    )

    # ------------------------------------------------------------------
    #  A. Run mode & basics
    # ------------------------------------------------------------------
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'inference', 'train_encoder'],
        help='Mode: train | inference | train_encoder (train encoder only)'
    )
    parser.add_argument(
        '--model_name', type=str, default='MOE',
        help='Model name for logging and output folder naming'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--local_rank', type=int,
        help='Local rank for distributed training (passed by torchrun)'
    )
    parser.add_argument(
        '--distributed', action='store_true',
        help='Use PyTorch DDP distributed training'
    )
    parser.add_argument(
        '--use_deepspeed', action='store_true',
        help='Use DeepSpeed for distributed training'
    )
    parser.add_argument(
        '--ds_config', type=str,
        default='./scripts/ds_zero3_bf16_offload.json',
        help='Path to DeepSpeed JSON config'
    )
    parser.add_argument(
        '--profile_timing', action='store_true',
        help='Profile step timings during training'
    )

    # ------------------------------------------------------------------
    #  B. Data & paths
    # ------------------------------------------------------------------
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Raw or preprocessed data directory (optional)'
    )
    parser.add_argument(
        '--zarr_path', type=str, default=None,
        help='Path to Zarr dataset (primary data entry)'
    )
    parser.add_argument(
        '--setting_path', type=str, default=None,
        help='Inference: directory with saved training args/config for reproducibility'
    )
    parser.add_argument(
        "--resume_path", type=str, default=None,
        help="Training: resume from this checkpoint path"
    )
    parser.add_argument(
        '--family', type=str, default=None, choices=_FAMILY_CHOICES,
        help='Dataset family / subtype (e.g. curve_vel_a)'
    )
    parser.add_argument(
        '--status_json', type=str,
        default='./dataset_status/dataset_status.json',
        help='JSON with dataset statistics (min/max/mean/std, etc.)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Train batch size (None: use config default)'
    )
    parser.add_argument(
        '--test_batch_size', type=int, default=None,
        help='Val/test batch size (None: use config default)'
    )
    parser.add_argument(
        '--n_train_samples', type=int, default=None,
        help='Cap train samples (None: use all)'
    )
    parser.add_argument(
        '--n_test_samples', type=int, default=None,
        help='Cap test samples (None: use all)'
    )
    parser.add_argument(
        '--channel_dim', type=int, default=None,
        help='Channel dimension index, e.g. 1 for [B, C, T, R]'
    )
    parser.add_argument(
        '--concat_channels', dest='concat_channels', action='store_true',
        help='Concat multi-channel waveforms along width into one channel'
    )
    parser.add_argument(
        '--no_concat_channels', dest='concat_channels', action='store_false',
        help='Keep explicit channel dimension (no concat)'
    )
    parser.set_defaults(concat_channels=None)

    parser.add_argument(
        '--val_ratio', type=float, default=0.2,
        help='Fraction of train split used for validation (default 0.2)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='DataLoader worker processes'
    )

    # Preprocess & spatial size
    parser.add_argument(
        '--k', type=int, default=1,
        help='Preprocess scale k (e.g. time/space downsampling factor)'
    )
    parser.add_argument(
        '--is_resize', action='store_true',
        help='Resize inputs/outputs to target grid size'
    )
    parser.add_argument(
        '--H_size', type=int, default=256,
        help='Height after resize (pixels / grid points)'
    )
    parser.add_argument(
        '--W_size', type=int, default=256,
        help='Width after resize (pixels / grid points)'
    )

    # ------------------------------------------------------------------
    #  C. AMP / mixed precision
    # ------------------------------------------------------------------
    parser.add_argument(
        '--use_amp', action='store_true',
        help='Use torch.cuda.amp automatic mixed precision in training'
    )
    parser.add_argument(
        '--mixed_precision', dest='mixed_precision', action='store_true',
        help='Enable higher-level mixed precision (train/infer)'
    )
    parser.add_argument(
        '--disable_mixed_precision', dest='mixed_precision', action='store_false',
        help='Disable mixed precision mode'
    )
    parser.set_defaults(mixed_precision=None)

    # ------------------------------------------------------------------
    #  D. Optimizer & LR schedule
    # ------------------------------------------------------------------
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Epochs to train (None: config default)'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='Base learning rate (overrides config if set)'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4,
        help='L2 weight decay'
    )
    parser.add_argument(
        '--accum_steps', type=int, default=1,
        help='Gradient accumulation steps (>1 emulates larger batch with less memory)'
    )

    # Warmup
    parser.add_argument(
        '--lr_warmup_epochs', type=int, default=5,
        help='LR warmup length in epochs'
    )
    parser.add_argument(
        '--lr_warmup_factor', type=float, default=1.0 / 3,
        help='Warmup start LR = base_lr * lr_warmup_factor'
    )
    parser.add_argument(
        '--lr_warmup_method', type=str,
        default='linear', choices=['linear', 'constant'],
        help='Warmup schedule: linear or constant'
    )

    # LR scheduler type
    parser.add_argument(
        '--lr_scheduler_type', type=str,
        default='cos_restart', choices=['cos_restart', 'cos', 'multistep'],
        help='Learning rate scheduler type'
    )

    # MultiStepLR
    parser.add_argument(
        '--milestones', nargs='+', type=int, default=[30, 60, 90],
        help='MultiStepLR milestone epochs'
    )
    parser.add_argument(
        '--scheduler_gamma', type=float, default=0.2,
        help='MultiStepLR gamma ( multiply LR by this at each milestone)'
    )

    # Cosine / cosine-warm-restarts
    parser.add_argument(
        '--lr_cosine_tmax_epochs', type=float, default=50.0,
        help='Cosine period length after warmup (epochs)'
    )
    parser.add_argument(
        '--lr_cosine_restart_t0_epochs', type=float, default=10.0,
        help='CosineAnnealingWarmRestarts initial period T_0 (epochs)'
    )
    parser.add_argument(
        '--lr_cosine_restart_t_mult', type=int, default=2,
        help='CosineAnnealingWarmRestarts period multiplier T_mult'
    )
    parser.add_argument(
        '--lr_cosine_eta_min', type=float, default=1e-6,
        help='Minimum LR in cosine phase'
    )

    # OneCycle
    parser.add_argument(
        '--use_onecycle', dest='use_onecycle', action='store_true',
        help='Use OneCycle LR policy (mutually exclusive with other schedulers)'
    )
    parser.add_argument(
        '--disable_onecycle', dest='use_onecycle', action='store_false',
        help='Disable OneCycle LR policy'
    )
    parser.set_defaults(use_onecycle=None)

    # ------------------------------------------------------------------
    #  E. Early stopping & eval frequency
    # ------------------------------------------------------------------
    parser.add_argument(
        '--early_stop', action='store_true',
        help='Enable early stopping'
    )
    parser.add_argument(
        '--early_stop_patience', type=int, default=None,
        help='Early-stop patience (epochs without improvement)'
    )
    parser.add_argument(
        '--early_stop_min_delta', type=float, default=None,
        help='Minimum improvement for early stopping'
    )
    parser.add_argument(
        '--early_stop_warmup_epochs', type=int, default=None,
        help='Epochs before early stopping is allowed'
    )
    parser.add_argument(
        '--eval_interval', type=int, default=None,
        help='Validation every N epochs (None: default policy)'
    )

    # ------------------------------------------------------------------
    #  F. Logging, outputs & visualization
    # ------------------------------------------------------------------
    parser.add_argument(
        '--output_dir', type=str, default='./results',
        help='Output root (weights, metrics, plots)'
    )
    parser.add_argument(
        '--log_root', type=str, default=None,
        help='TensorBoard log root (default: under runs/)'
    )
    parser.add_argument(
        '--vis_freq', type=int, default=5,
        help='Visualize every N epochs'
    )
    parser.add_argument(
        '--use_wandb', action='store_true',
        help='Log training to Weights & Biases'
    )
    parser.add_argument(
        '--verbose', dest='verbose', action='store_true',
        help='Verbose console logging'
    )
    parser.add_argument(
        '--quiet', dest='verbose', action='store_false',
        help='Minimal console logging'
    )
    parser.set_defaults(verbose=None)

    # Inference
    parser.add_argument(
        '--model_path', type=str, default=None,
        help='Checkpoint path for inference'
    )
    parser.add_argument(
        '--infer_one', type=int, default=None,
        help='Run inference on a single sample index (eval_split dataset)'
    )

    # ------------------------------------------------------------------
    #  G. Encoder & backbone
    # ------------------------------------------------------------------
    parser.add_argument(
        '--backbone', type=str, default='vit',
        choices=['vit', 'convnext_tiny'],
        help='Encoder backbone: ViT or ConvNeXt-Tiny'
    )
    parser.add_argument(
        '--use_encoder', dest='use_encoder', action='store_true',
        help='Enable encoder: encode input before MoE / operator'
    )
    parser.add_argument(
        '--disable_encoder', dest='use_encoder', action='store_false',
        help='Disable encoder: raw input goes to MoE / operator'
    )
    parser.set_defaults(use_encoder=None)
    parser.add_argument(
        '--hidden_channels', type=int, default=128,
        help='Hidden width (encoder / operator trunk)'
    )
    parser.add_argument(
        '--target_size', type=int, default=70,
        help="Encoder spatial output size"
    )
    parser.add_argument(
        '--enc_channels', type=int, default=128,
        help='Encoder output channels'
    )
    parser.add_argument(
        '--encoder_path', type=str, default=None,
        help='Load encoder weights from checkpoint; may freeze in training'
    )

    # ------------------------------------------------------------------
    #  H. Neural operators (FNO / WNO / MNO / LNO)
    # ------------------------------------------------------------------
    # FNO
    parser.add_argument(
        '--FNO_n_modes_height', type=int, default=16,
        help='FNO: Fourier modes kept along height'
    )
    parser.add_argument(
        '--FNO_n_modes_width', type=int, default=16,
        help='FNO: Fourier modes kept along width'
    )
    parser.add_argument(
        '--FNO_n_layers', type=int, default=4,
        help='FNO: number of layers (depth)'
    )

    # WNO
    parser.add_argument(
        '--WNO_n_levels_height', type=int, default=2,
        help='WNO: wavelet decomposition levels (height)'
    )
    parser.add_argument(
        '--WNO_n_levels_width', type=int, default=2,
        help='WNO: wavelet decomposition levels (width)'
    )
    parser.add_argument(
        '--WNO_n_layers', type=int, default=4,
        help='WNO: number of WNO blocks (depth)'
    )
    parser.add_argument(
        '--WNO_dropout_rate', type=float, default=0.1,
        help='WNO: dropout rate'
    )
    parser.add_argument(
        '--wavelet_type', type=str, default='db6',
        choices=['coif4', 'db4', 'db8', 'sym4', 'coif5', 'sym8', 'db6', 'sym6'],
        help='Real wavelet family'
    )
    parser.add_argument(
        '--dtcwt_type', nargs=2, type=str, default=None,
        help='Dual-tree complex wavelet: two wavelet names, e.g. near_sym_a near_sym_b'
    )

    # MNO
    parser.add_argument(
        '--MNO_n_scales', type=int, default=3,
        help='MNO: number of scales'
    )
    parser.add_argument(
        '--MNO_scale_factors', nargs='+', type=float,
        default=[1.0, 0.6, 0.3],
        help='MNO: spatial/temporal scale factors per level'
    )
    parser.add_argument(
        '--MNO_n_layers', type=int, default=3,
        help='MNO: layers per scale'
    )

    # LNO
    parser.add_argument(
        '--LNO_n_modes', nargs=2, type=int, default=[16, 16],
        help='LNO: retained modes (height, width)'
    )
    parser.add_argument(
        '--LNO_n_layers', type=int, default=3,
        help='LNO: number of layers'
    )

    # ------------------------------------------------------------------
    #  I. MoE / experts / routing / fusion
    # ------------------------------------------------------------------
    # Experts & top-k
    parser.add_argument(
        '--top_k', type=int, default=1,
        help='Top-k experts per sample'
    )
    parser.add_argument(
        '--choose_experts', nargs='*', type=int, default=None,
        help='Expert indices to use, e.g. 0 1 2 3 for FNO WNO MNO LNO'
    )

    # MoE method & mode
    parser.add_argument(
        '--moe_method', type=str, default="afmoe",
        choices=["basic", "afmoe"],
        help='MoE variant: basic or afmoe (adaptive frequency MoE)'
    )
    parser.add_argument(
        '--use_experts_path', type=str, default=None,
        help='Directory of pretrained expert checkpoints for MoE'
    )
    parser.add_argument(
        '--use_moe', action='store_true',
        help='Use MoE (often freeze experts; train router/fusion)'
    )
    parser.add_argument(
        '--moe_mode', type=str, default='standard',
        choices=['standard', 'velocity_type', 'group'],
        help=(
            "MoE mode: "
            "'standard' routing/fusion; "
            "'velocity_type' blend experts with velocity-type weights; "
            "'group' grouped MoE by category"
        )
    )

    # Router
    parser.add_argument(
        '--router_type', type=str, default='basic',
        help="Router type, e.g. 'basic' / 'adamv'"
    )
    parser.add_argument(
        '--band_sharpness', type=float, default=20.0,
        help='AFreqMoE: soft band sharpness (higher -> harder bands)'
    )
    parser.add_argument(
        '--freq_affinity_sharpness', type=float, default=10.0,
        help='Sharpness of expert frequency affinity vs band centers'
    )
    parser.add_argument(
        "--disable_band_decomposition", action='store_true',
        help='Ablation: disable band decomposition; feed full input to experts'
    )
    parser.add_argument(
        '--disable_soft_bands', action='store_true',
        help='Ablation: hard band split instead of soft bands'
    )
    parser.add_argument(
        '--disable_freq_attn', action='store_true',
        help='Ablation: disable freq self-attn; use magnitude spectrum for routing'
    )
    parser.add_argument(
        '--disable_band_mixing', action='store_true',
        help='Ablation: no mixed bands; each expert sees its band only'
    )
    parser.add_argument(
        '--enable_freq_metrics', action='store_true',
        help='Compute low/mid/high band metrics in val/inference'
    )
    parser.add_argument(
        '--router_hidden_dim', type=int, default=None,
        help='Router hidden width (None: config default)'
    )
    parser.add_argument(
        '--routing_mode', type=str, default=None,
        choices=['learned', 'uniform', 'random'],
        help='Routing: learned | uniform | random top-k'
    )

    # Inter-group / intra-group fusion
    parser.add_argument(
        '--fusion_type', type=str, default='linear',
        help="Inter-group fusion: 'linear'/'attention'/'swa'/'basic(sum)'"
    )
    parser.add_argument(
        '--s_processor_type', type=str, default='linear',
        help="Strong-expert intra fusion: 'linear'/'attention'/'mean'/'sum'"
    )
    parser.add_argument(
        '--w_processor_type', type=str, default='linear',
        help="Weak-expert intra fusion: 'linear'/'attention'/'mean'/'sum'"
    )

    # Gating & strong/weak balance
    parser.add_argument(
        '--enable_noisy_gating', dest='noisy_gating', action='store_true',
        help='Noisy gating (noisy top-k)'
    )
    parser.add_argument(
        '--disable_noisy_gating', dest='noisy_gating', action='store_false',
        help='Disable noisy gating'
    )
    parser.set_defaults(noisy_gating=None)
    parser.add_argument(
        '--beta', type=float, default=0.5,
        help='Strong/weak gate; higher beta weights weak experts more'
    )

    # Subtype, classifier, memory proxy
    parser.add_argument(
        '--is_specific', action='store_true',
        help='Fine-grained subtype splits (e.g. CurveVel A/B experts)'
    )
    parser.add_argument(
        '--is_classifier', action='store_true',
        help='Grouped-expert MoE with classifier'
    )
    parser.add_argument(
        '--v_type_num', type=int, default=None,
        help='Number of velocity types (classifier dim / expert groups)'
    )
    parser.add_argument(
        '--use_gpu_proxy', action='store_true',
        help='Expert memory proxy (CPU<->GPU load experts on demand)'
    )

    # ------------------------------------------------------------------
    #  J. Loss weights
    # ------------------------------------------------------------------
    parser.add_argument(
        '-g1v', '--lambda_g1v', type=float, default=0.6,
        help='Weight for global velocity loss term g1v'
    )
    parser.add_argument(
        '-g2v', '--lambda_g2v', type=float, default=0.4,
        help='Weight for global velocity loss term g2v'
    )
    parser.add_argument(
        '--lambda_grad_l1', type=float, default=0.15,
        help='Gradient (edge) L1 weight'
    )
    parser.add_argument(
        '--lambda_fourier_mag_l1', type=float, default=0.10,
        help='Fourier magnitude L1 weight'
    )
    parser.add_argument(
        '--lambda_ce', type=float, default=0.20,
        help='Cross-entropy classification loss weight (if classifier / grouped experts are enabled)'
    )

    # ------------------------------------------------------------------
    #  Parse arguments and return parser
    # ------------------------------------------------------------------
    config_specified_dests = set()
    effective_argv = argv
    if bootstrap_args.args_file is not None:
        args_payload = _load_json_payload(bootstrap_args.args_file)
        args_defaults, config_specified_dests = _resolve_payload_defaults(
            parser,
            args_payload,
            ignore_unknown_keys={"parser", "user_specified_args"},
        )
        parser.set_defaults(**args_defaults)
        args_file_path = str(Path(bootstrap_args.args_file).expanduser())
        parser.set_defaults(args_file=args_file_path, config=None)
        # With args.json, training args come entirely from JSON (precedence over --config).
        effective_argv = ["--args", args_file_path]
    elif bootstrap_args.config is not None:
        config_payload = _load_json_payload(bootstrap_args.config)
        config_defaults, config_specified_dests = _resolve_payload_defaults(parser, config_payload)
        parser.set_defaults(**config_defaults)
        config_path = str(Path(bootstrap_args.config).expanduser())
        parser.set_defaults(config=config_path)
        # With --config JSON, defaults come from file; no further CLI overrides in this path.
        effective_argv = ["--config", config_path]

    args = parser.parse_args(effective_argv)
    if bootstrap_args.config is not None:
        args.user_specified_args = sorted(config_specified_dests)
    else:
        args.user_specified_args = sorted(
            _collect_user_specified_args(parser, argv, config_specified_dests)
        )
    args.parser = parser
    return args

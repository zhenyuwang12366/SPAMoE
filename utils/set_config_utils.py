import sys
import os
import argparse
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.seismic_moe_config import SeismicMOEConfig, SPECIFIC_TYPE_VARIANTS
import neuralop.mpu.comm as comm
from neuralop.training import setup
from neuralop.utils import get_wandb_api_key
from .utils import to_snake_lower

_SPECIFIC_ALLOWED_FAMILIES = (
    set(SPECIFIC_TYPE_VARIANTS.keys()) |
    {variant for variants in SPECIFIC_TYPE_VARIANTS.values() for variant in variants}
)


def _build_runtime_context(
    *,
    device,
    is_logger: bool,
    world_size: int,
    local_rank: int,
    global_rank: int,
    val_ratio: float,
    experts_name,
    experts_name_str: str,
):
    """Collect frequently used runtime info for the training script."""

    return {
        "device": device,
        "is_logger": is_logger,
        "world_size": world_size,
        "local_rank": local_rank,
        "global_rank": global_rank,
        "val_ratio": val_ratio,
        "experts_name": experts_name,
        "experts_name_str": experts_name_str,
    }

def get_seismic_config(args: argparse.Namespace):
    # 加载配置
    config = SeismicMOEConfig()
    if getattr(args, "model_name", None):
        config.model_name = args.model_name
    
    if args.mode == 'train_encoder':
        config.train_encoder = True
    
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
    if getattr(args, "log_root", None):
        config.log_root = args.log_root

        #解释见onenote1
    if args.family:
        config.family = to_snake_lower(args.family)
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.test_batch_size is not None:
        config.test_batch_size = args.test_batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.hidden_channels:
        config.hidden_channels = args.hidden_channels
    if args.n_train_samples is not None:
        config.n_train_samples = args.n_train_samples
    if args.n_test_samples is not None:
        config.n_test_samples = args.n_test_samples
    if args.channel_dim is not None:
        config.channel_dim = args.channel_dim
    if args.concat_channels is not None:
        config.concat_channels = args.concat_channels
    if args.mixed_precision is not None:
        config.mixed_precision = args.mixed_precision
    if args.lr_warmup_epochs is not None:
        config.lr_warmup_epochs = args.lr_warmup_epochs
    if args.use_amp:
        config.use_amp = True
    config.lr_warmup_factor = args.lr_warmup_factor
    config.lr_warmup_method = args.lr_warmup_method
    config.lr_scheduler_type = args.lr_scheduler_type
    config.milestones = list(args.milestones)
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.scheduler_gamma is not None:
        config.scheduler_gamma = args.scheduler_gamma
    config.lr_cosine_tmax_epochs = args.lr_cosine_tmax_epochs
    config.lr_cosine_restart_t0_epochs = args.lr_cosine_restart_t0_epochs
    config.lr_cosine_restart_t_mult = int(args.lr_cosine_restart_t_mult)
    config.lr_cosine_eta_min = args.lr_cosine_eta_min
    if args.accum_steps is not None:
        config.accum_steps = args.accum_steps
    if args.use_onecycle is not None:
        config.use_onecycle = args.use_onecycle
    if args.early_stop:
        config.early_stop = True
    if args.use_encoder is not None:
        config.use_encoder = args.use_encoder
    if getattr(args, "backbone", None) is not None:
        config.backbone = args.backbone
    if args.use_moe:
        config.use_moe = True
    if getattr(args, "routing_mode", None) is not None:
        config.routing_mode = args.routing_mode
    if args.early_stop_patience is not None:
        config.early_stop_patience = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        config.early_stop_min_delta = args.early_stop_min_delta
    if args.early_stop_warmup_epochs is not None:
        config.early_stop_warmup_epochs = args.early_stop_warmup_epochs
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval
    if args.verbose is not None:
        config.verbose = args.verbose
  
    accum_steps = config.accum_steps
    use_amp = config.use_amp
    
    if '--lr_warmup_epochs' not in sys.argv:
        config.lr_warmup_epochs = max(1, int(config.epochs * 0.05))
    
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
    # # WNO config setting
    # config.expert_configs[1]['n_levels_height'] = args.WNO_n_levels_height
    # config.expert_configs[1]['n_levels_width'] = args.WNO_n_levels_width
    # config.expert_configs[1]['n_layers'] = args.WNO_n_layers
    # config.expert_configs[1]['dropout_rate'] = args.WNO_dropout_rate
    # config.expert_configs[1]['wavelet'] = args.wavelet_type
    # if(args.dtcwt_type): 
    #     config.expert_configs[1]['biort'], config.expert_configs[1]['qshift'] = args.dtcwt_type
    #     config.expert_configs[1]['conv_kind'] = "dtcwt" 
    
    # MNO config setting
    config.expert_configs[1]['n_scales'] = args.MNO_n_scales
    config.expert_configs[1]['scale_factors'] = args.MNO_scale_factors
    config.expert_configs[1]['n_layers'] = args.MNO_n_layers
    # LNO config setting
    config.expert_configs[2]['n_modes'] = tuple(args.LNO_n_modes)
    config.expert_configs[2]['n_layers'] = args.LNO_n_layers
    
    print(f'FNO:n_modes_height:{config.expert_configs[0]["n_modes_height"]}')
    print(f'FNO:n_modes_width:{config.expert_configs[0]["n_modes_width"]}')
    print(f'FNO:n_layers:{config.expert_configs[0]["n_layers"]}')
    
    # 设置专家数
    config.top_k = args.top_k
    # 选择专家，这里后面的config.expert_configs就是config文件中所创建的字典列表，
    # 当你从命令行输入choose——experts之后，这里的for循环会根据你给定的序号找到对应的专家的字典，并将这个字典放入
    #config.expert_configs列表中
    if args.choose_experts is not None and len(args.choose_experts) > 0:
        config.expert_configs = [config.expert_configs[i] for i in args.choose_experts]
    else:
        args.choose_experts = list(range(len(config.expert_configs)))
    #这里的config.expert_configs就是seismic_moe_config中的“字典列表”，关于“字典列表”结构的解释详见OneNote3
    
    # 训练moe架构
         #下面两段代码的解释见OneNote2,意义是：
         # 在使用 Mixture of Experts 模型时，如果用户已经训练并保存了若干个“专家模型”（模型文件保存在某个文件夹中），那么这两段代码就是要：
         #读取那些 .pt 文件（即每个专家的模型参数）；
         #按照文件名中的编号提取出专家编号；
         #做一致性校验（文件数量、编号是否跟配置匹配）；
    experts_name = []
    if config.top_k > 1 and args.use_moe and args.use_experts_path:
        # 模型文件夹中的专家 best_expert_{experts_name}_{i}_{curve/flat/style}_{vel/fault/style}.pt
        save_experts = [
            int(f.split('_')[3]) for f in os.listdir(args.use_experts_path)
            if f.split('_')[1] == 'expert' and f.endswith('.pt')
        ]
        save_experts = list(set(save_experts))
        #注意，这里输出的save_experts是一个代表专家组模型序号的整数列表。详见OneNote2

        print(f"选择了 {len(save_experts)} 个专家, 分别为 {save_experts}")

        config.use_moe = True
        config.use_experts_path = args.use_experts_path

        experts_name.append("all")
        experts_name_str = "all"
    # 这两行代码解释见OneNote3，不过我想知道为什么这两段代码在我们单个模型训练过程中没有起作用
    # 修复：使用enumerate来获取正确的索引，因为config.expert_configs已经被重新排序
    else:
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
    if hasattr(args, "lambda_grad") and args.lambda_grad is not None:
        config.lambda_grad = args.lambda_grad
    if hasattr(args, "lambda_ssim") and args.lambda_ssim is not None:
        config.lambda_ssim = args.lambda_ssim
    if hasattr(args, "lambda_grad_l1"):
        config.lambda_grad_l1 = args.lambda_grad_l1
    if hasattr(args, "lambda_fourier_mag_l1"):
        config.lambda_fourier_mag_l1 = args.lambda_fourier_mag_l1
    if hasattr(args, "lambda_ce"):
        config.lambda_ce = args.lambda_ce
    
    # 设置 MOE 模式
    if hasattr(args, "moe_mode") and args.moe_mode:
        config.moe_mode = args.moe_mode
    
    # 设置路由形式
    if args.router_type:
        config.router_type = args.router_type
    if hasattr(args, "router_hidden_dim") and args.router_hidden_dim is not None:
        config.router_hidden_dim = args.router_hidden_dim
    if hasattr(args, "band_sharpness"):
        config.band_sharpness = args.band_sharpness
    if hasattr(args, "freq_affinity_sharpness"):
        config.freq_affinity_sharpness = args.freq_affinity_sharpness
    if hasattr(args, "disable_soft_bands"):
        config.use_soft_bands = not args.disable_soft_bands
    if hasattr(args, "disable_freq_attn"):
        config.enable_freq_attn = not args.disable_freq_attn
    if hasattr(args, "disable_band_mixing"):
        config.enable_band_mixing = not args.disable_band_mixing
    if hasattr(args, "noisy_gating") and args.noisy_gating is not None:
        config.noisy_gating = args.noisy_gating
    
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
    if args.use_gpu_proxy:
        config.use_gpu_proxy = True
    
    # 设置细化种类
    if args.is_specific:
        config.is_specific = args.is_specific
    
    # 设置是否使用分组专家网络
    if args.is_classifier:
        config.is_classifier = args.is_classifier

    # 设置速度类型数量（分类数）
    if hasattr(args, "v_type_num") and args.v_type_num is not None:
        config.v_type_num = args.v_type_num
    else:
        current_v_type_num = getattr(config, "v_type_num", None)
        if not current_v_type_num:
            type_mapping = getattr(config, "type_id", {})
            type_key = "specific" if getattr(config, "is_specific", False) else "normal"
            mapped_types = type_mapping.get(type_key, {})
            if mapped_types:
                config.v_type_num = len(mapped_types)

    # 判断is_specific与选择family是否匹配
    if config.is_specific and config.family not in _SPECIFIC_ALLOWED_FAMILIES:
        raise ValueError(
            f"{config.family} 与细分类配置不匹配，可选细分类包括: {sorted(_SPECIFIC_ALLOWED_FAMILIES)}"
        )
    
    if args.is_resize:
        config.is_resize = args.is_resize

        config.H_size = args.H_size
        config.W_size = args.W_size
    
    config.moe_method = args.moe_method
    
    #-------------- 设置完毕 -----------#
    runtime_ctx = _build_runtime_context(
        device=device,
        is_logger=is_logger,
        world_size=world_size,
        local_rank=local_rank,
        global_rank=global_rank,
        val_ratio=val_ratio,
        experts_name=experts_name,
        experts_name_str=experts_name_str,
    )

    return config, runtime_ctx

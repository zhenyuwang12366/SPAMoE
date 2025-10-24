import argparse
from config.seismic_moe_config import SPECIFIC_TYPE_VARIANTS

_SPECIFIC_BASE_FAMILIES = set(SPECIFIC_TYPE_VARIANTS.keys())
_SPECIFIC_VARIANT_FAMILIES = {variant for variants in SPECIFIC_TYPE_VARIANTS.values() for variant in variants}
_FAMILY_CHOICES = ['vel', 'style', 'fault', 'all']
_FAMILY_CHOICES.extend(sorted(_SPECIFIC_BASE_FAMILIES | _SPECIFIC_VARIANT_FAMILIES))

def build_argparser_and_parse(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="地震数据MOE训练和推理")
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference','train_encoder'],
                        help='运行模式: 训练或推理')
    parser.add_argument('--model_name', type=str, default='MOE')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录路径')
    parser.add_argument('--family', type=str, default=None, choices=_FAMILY_CHOICES,
                        help='数据集系列，可选通用 (vel/style/fault/all) 或细分类别 (如 curve_vel_a)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='训练批次大小')
    parser.add_argument('--test_batch_size', type=int, default=None,
                        help='验证/测试批次大小')
    parser.add_argument('--n_train_samples', type=int, default=None,
                        help='训练子集样本数（None 表示全部）')
    parser.add_argument('--n_test_samples', type=int, default=None,
                        help='测试子集样本数（None 表示全部）')
    parser.add_argument('--channel_dim', type=int, default=None,
                        help='输入张量中通道所在的维度')
    parser.add_argument('--concat_channels', dest='concat_channels', action='store_true',
                        help='将多通道波形按宽度拼接成单通道表示')
    parser.add_argument('--no_concat_channels', dest='concat_channels', action='store_false',
                        help='保留显式通道维度')
    parser.set_defaults(concat_channels=None)
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--use_amp', action='store_true', 
                        help='是否使用混合精度训练')
    parser.add_argument('--mixed_precision', dest='mixed_precision', action='store_true',
                        help='启用混合精度推理/训练模式')
    parser.add_argument('--disable_mixed_precision', dest='mixed_precision', action='store_false',
                        help='禁用混合精度模式')
    parser.set_defaults(mixed_precision=None)
    
    parser.add_argument('--lr_warmup_epochs', type=int, default=5,
                        help='学习率预热轮数（按 epoch 计）')
    parser.add_argument('--lr_warmup_factor', type=float, default=1.0 / 3,
                        help='warmup 初始学习率相对 base_lr 的比例')
    parser.add_argument('--lr_warmup_method', type=str, default='linear', choices=['linear', 'constant'],
                        help='warmup 的方式：线性或常数')
    parser.add_argument('--lr_scheduler_type', type=str, default='cos_restart', choices=['cos_restart', 'cos', 'multistep'],
                        help='学习率调度器类型')
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, 60, 90],
                        help='MultiStepLR 的学习率衰减里程碑（按 epoch 计）')
    parser.add_argument('--scheduler_gamma', type=float, default=0.3,
                        help='MultiStepLR 的学习率衰减因子')
    parser.add_argument('--lr_cosine_tmax_epochs', type=float, default=50.0,
                        help='WarmupCosineLR 进入余弦阶段后的周期长度（按 epoch 计）')
    parser.add_argument('--lr_cosine_restart_t0_epochs', type=float, default=10.0,
                        help='CosineAnnealingWarmRestarts 首个周期长度（按 epoch 计）')
    parser.add_argument('--lr_cosine_restart_t_mult', type=int, default=2,
                        help='CosineAnnealingWarmRestarts 的周期放大倍率')
    parser.add_argument('--lr_cosine_eta_min', type=float, default=1e-6,
                        help='余弦调度阶段的最小学习率')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='L2正则化')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='梯度累计步数')
    parser.add_argument('--use_onecycle', dest='use_onecycle', action='store_true',
                        help='启用 OneCycle 学习率策略')
    parser.add_argument('--disable_onecycle', dest='use_onecycle', action='store_false',
                        help='禁用 OneCycle 学习率策略')
    parser.set_defaults(use_onecycle=None)
    parser.add_argument('--early_stop', action='store_true',
                        help='启用早停')
    parser.add_argument('--early_stop_patience', type=int, default=None,
                        help='早停耐心轮数')
    parser.add_argument('--early_stop_min_delta', type=float, default=None,
                        help='早停最小改进值')
    parser.add_argument('--early_stop_warmup_epochs', type=int, default=None,
                        help='早停开始前的预热轮数')
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='验证间隔（单位：epoch）')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='输出更详细的日志')
    parser.add_argument('--quiet', dest='verbose', action='store_false',
                        help='最小化日志输出')
    parser.set_defaults(verbose=None)
    
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='结果保存目录')
    parser.add_argument('--log_root', type=str, default=None,
                        help='TensorBoard 日志根目录，默认使用 runs 子目录')
    parser.add_argument('--model_path', type=str, default=None,
                        help='推理模式下使用的模型路径')
    parser.add_argument('--vis_freq', type=int, default=5,
                        help='可视化频率（每隔多少个epoch可视化一次）')
    parser.add_argument('--distributed', action='store_true',
                        help='是否使用分布式训练')
    parser.add_argument('--use_wandb', action='store_true',
                        help='是否使用WandB记录训练过程')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例，默认为0.2（20%）')
    
    parser.add_argument('--k', type=int, default=1,
                        help='预处理缩放比例')
    parser.add_argument('--top_k', type=int, default=1,
                        help='选择前k个专家')
    parser.add_argument('--choose_experts',nargs='+', type=int, default=[0],
                        help='专家选择, FNO:0, WNO:1, MNO:2, LNO:3')
    
    parser.add_argument('--FNO_n_modes_height', type=int, default=16,
                        help='高度傅里叶变换后保留的模态数量')
    parser.add_argument('--FNO_n_modes_width', type=int, default=16,
                        help='宽度傅里叶变换后保留的模态数量')
    parser.add_argument('--FNO_n_layers', type=int, default=4,
                        help='傅里叶layers堆叠数量')
    parser.add_argument('--WNO_n_levels_height', type=int, default=2,
                        help='高度减少级别')
    parser.add_argument('--WNO_n_levels_width', type=int, default=2,
                        help='宽度减少级别')
    parser.add_argument('--WNO_n_layers', type=int, default=4,
                        help='WNO块的数量，控制模型深度')
    parser.add_argument('--WNO_block_n_layers', type=int, default=2,
                        help='每个WNO块内部的层数')
    parser.add_argument('--WNO_dropout_rate', type=float, default=0.1,
                        help='WNO块中的dropout比例，提高泛化能力')
    parser.add_argument('--wavelet_type', type=str, default='haar', choices=['coif4','db4','db8','sym4','coif5','sym8','db6'],
                        help='小波类型')
    parser.add_argument('--dtcwt_type', nargs=2, type=str, default=None,
                        help='双树复小波类型')
    parser.add_argument('--WNO_pad_mode', type=str, default=None, choices=['constant', 'reflect', 'replicate', 'circular'],
                        help='WNO填充模式')
    parser.add_argument('--WNO_ensure_even_shapes', dest='WNO_ensure_even_shapes', action='store_true', default=None,
                        help='启用WNO偶数形状约束')
    parser.add_argument('--WNO_disable_ensure_even_shapes', dest='WNO_ensure_even_shapes', action='store_false',
                        help='禁用WNO偶数形状约束')
    parser.add_argument('--WNO_adaptive_padding', dest='WNO_adaptive_padding', action='store_true', default=None,
                        help='启用WNO自适应填充')
    parser.add_argument('--WNO_disable_adaptive_padding', dest='WNO_adaptive_padding', action='store_false',
                        help='禁用WNO自适应填充')
    parser.add_argument('--WNO_use_channel_mlp', dest='WNO_use_channel_mlp', action='store_true', default=None,
                        help='启用WNO通道MLP')
    parser.add_argument('--WNO_disable_channel_mlp', dest='WNO_use_channel_mlp', action='store_false',
                        help='禁用WNO通道MLP')
    parser.add_argument('--WNO_channel_mlp_dropout', type=float, default=0.0,
                        help='WNO通道MLP的dropout比例')
    parser.add_argument('--WNO_channel_mlp_expansion', type=float, default=0.0,
                        help='WNO通道MLP的扩展倍率')
    
    parser.add_argument('--MNO_n_scales', type=int, default=3,
                        help='总共使用的尺度')
    parser.add_argument('--MNO_scale_factors', nargs='+', type=float, default=[1.0, 0.5, 0.25],
                        help='每个尺度的缩放因子')
    parser.add_argument('--MNO_n_layers', type=int, default=3,
                        help='每个尺度使用的神经网络层数')
    parser.add_argument('--LNO_n_modes', nargs=2, type=int, default=[16, 16],
                        help='局部变换后保留的模态数量')
    parser.add_argument('--LNO_n_layers', type=int, default=3,
                        help='每个尺度使用的神经网络层数')
    
    # MoE融合参数配置
    parser.add_argument('--use_experts_path', type=str, default=None,
                        help='moe使用的专家模型存放路径')
    parser.add_argument('--use_moe', action='store_true',
                        help='是否使用moe, 使用会冻结专家模型')
    parser.add_argument('--moe_mode', type=str, default='standard',
                        choices=['standard', 'velocity_type'],
                        help="MOE运行模式：'standard' 使用路由/融合，'velocity_type' 直接按类型权重融合预训练专家")
    parser.add_argument('--router_type', type=str, default='basic',
                        help='路由器类型: \'basic\'/\'adamv\'')
    parser.add_argument('--router_hidden_dim', type=int, default=None,
                        help='路由器隐藏层宽度')
    parser.add_argument('--fusion_type', type=str, default='linear',
                        help='专家组间融合方式: \'linear\'/\'attention\'/\'swa\'/\'basic(sum)\'')
    parser.add_argument('--s_processor_type', type=str, default='linear',
                        help='强专家组内融合方式: \'linear\'/\'attention\'/\'mean\'/\'sum\'')
    parser.add_argument('--w_processor_type', type=str, default='linear',
                        help='弱专家组内融合方式: \'linear\'/\'attention\'/\'mean\'/\'sum\'')
    parser.add_argument('--enable_noisy_gating', dest='noisy_gating', action='store_true',
                        help='启用带噪声门控')
    parser.add_argument('--disable_noisy_gating', dest='noisy_gating', action='store_false',
                        help='禁用带噪声门控')
    parser.set_defaults(noisy_gating=None)
    parser.add_argument('--beta', type=float, default=0.5,
                        help='强弱激活参数，beta越大，弱激活影响越大')
    parser.add_argument('--is_specific', action='store_true',
                        help='是否细化种类')
    parser.add_argument('--is_classifier', action='store_true',
                        help='是否使用分组专家网络')
    parser.add_argument('--v_type_num', type=int, default=None,
                        help='速度类型数量，用于控制分类器输出维度及专家分组')
    parser.add_argument('--use_gpu_proxy', action='store_true',
                        help='启用专家显存代理以缓解显存压力')
    
    parser.add_argument('--backbone', type=str, default='vit', choices=['vit', 'convnext_tiny'],
                        help='encoder backbone')
    parser.add_argument('--use_encoder', dest='use_encoder', action='store_true',
                        help='启用 encoder，将输入先编码再送入 MoE')
    parser.add_argument('--disable_encoder', dest='use_encoder', action='store_false',
                        help='禁用 encoder，直接将原始输入送入 MoE')
    parser.set_defaults(use_encoder=None)


    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='隐藏通道数（默认值由配置文件决定，可通过此参数覆盖）')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率（默认值由配置文件决定，可通过此参数覆盖）')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='恢复训练的checkpoint路径，如 best_model_xxx.pt')
    parser.add_argument('--encoder_path', type=str, default=None,
                        help='单独加载encoder的checkpoint路径，并在训练时冻结其参数')
    
    parser.add_argument('--is_resize', action='store_true')
    parser.add_argument('--H_size', type=int, default=256)
    parser.add_argument('--W_size', type=int, default=256)
    
    # Loss related
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=0.6)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=0.4)
    parser.add_argument('--lambda_grad_l1', type=float, default=0.15)
    parser.add_argument('--lambda_fourier_mag_l1', type=float, default=0.10)
    parser.add_argument('--lambda_ce', type=float, default=0.20)
    
    # Performance related
    parser.add_argument('--profile_timing', action='store_true',
                        help='是否记录训练过程中的耗时信息')
    
    args = parser.parse_args(argv)
    args.parser = parser
    return args

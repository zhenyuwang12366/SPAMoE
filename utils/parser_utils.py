import argparse
from config.seismic_moe_config import SPECIFIC_TYPE_VARIANTS

# ====== 数据系列（family）可选项构建 ======
_SPECIFIC_BASE_FAMILIES = set(SPECIFIC_TYPE_VARIANTS.keys())
_SPECIFIC_VARIANT_FAMILIES = {
    variant for variants in SPECIFIC_TYPE_VARIANTS.values() for variant in variants
}
_FAMILY_CHOICES = ['vel', 'style', 'fault', 'all']
_FAMILY_CHOICES.extend(sorted(_SPECIFIC_BASE_FAMILIES | _SPECIFIC_VARIANT_FAMILIES))


def build_argparser_and_parse(argv=None) -> argparse.Namespace:
    """
    构建命令行参数解析器：
    - 支持地震数据 FNO/WNO/MNO/LNO + MoE 的训练与推理
    - 大量参数通过配置文件默认给出，这里可以覆盖
    """
    parser = argparse.ArgumentParser(description="地震数据 MOE 训练和推理")

    # ------------------------------------------------------------------
    #  A. 运行模式 & 基本设置
    # ------------------------------------------------------------------
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'inference', 'train_encoder'],
        help='运行模式：train=训练，inference=推理，train_encoder=单独训练 encoder'
    )
    parser.add_argument(
        '--model_name', type=str, default='MOE',
        help='模型名称，用于日志记录和保存目录命名'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子，保证实验可复现'
    )
    parser.add_argument(
        '--local_rank', type=int,
        help='分布式训练 local rank（由 torchrun 自动传入）'
    )
    parser.add_argument(
        '--distributed', action='store_true',
        help='是否使用 PyTorch DDP 分布式训练'
    )
    parser.add_argument(
        '--use_deepspeed', action='store_true',
        help='是否使用 DeepSpeed 进行分布式加速'
    )
    parser.add_argument(
        '--ds_config', type=str,
        default='./scripts/ds_zero3_bf16_offload.json',
        help='DeepSpeed 配置文件路径'
    )
    parser.add_argument(
        '--profile_timing', action='store_true',
        help='是否记录训练过程中的耗时信息（profiling）'
    )

    # ------------------------------------------------------------------
    #  B. 数据 & 路径相关
    # ------------------------------------------------------------------
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='原始/预处理数据所在目录（可选）'
    )
    parser.add_argument(
        '--zarr_path', type=str, default=None,
        help='Zarr 格式数据集路径（主数据入口）'
    )
    parser.add_argument(
        '--setting_path', type=str, default=None,
        help='推理模式下：存放训练配置文件的目录（用于复现）'
    )
    parser.add_argument(
        "--resume_path", type=str, default=None,
        help="训练模式下：从指定 checkpoint 恢复训练"
    )
    parser.add_argument(
        '--family', type=str, default=None, choices=_FAMILY_CHOICES,
        help='数据集系列：细分类别 (如 curve_vel_a)'
    )
    parser.add_argument(
        '--status_json', type=str,
        default='./dataset_status/dataset_status.json',
        help='数据集统计信息 JSON 文件（均值/方差/范围等）'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='训练批次大小（None 使用配置文件默认值）'
    )
    parser.add_argument(
        '--test_batch_size', type=int, default=None,
        help='验证/测试批次大小（None 使用配置文件默认值）'
    )
    parser.add_argument(
        '--n_train_samples', type=int, default=None,
        help='训练子集样本数（None 表示使用全部训练样本）'
    )
    parser.add_argument(
        '--n_test_samples', type=int, default=None,
        help='测试子集样本数（None 表示使用全部测试样本）'
    )
    parser.add_argument(
        '--channel_dim', type=int, default=None,
        help='输入张量中通道所在的维度，例如 1 表示 [B, C, T, R]'
    )
    parser.add_argument(
        '--concat_channels', dest='concat_channels', action='store_true',
        help='将多通道波形在宽度维度上拼接成单通道表示'
    )
    parser.add_argument(
        '--no_concat_channels', dest='concat_channels', action='store_false',
        help='保留显式通道维度，不做拼接'
    )
    parser.set_defaults(concat_channels=None)

    parser.add_argument(
        '--val_ratio', type=float, default=0.2,
        help='从训练集中划分为验证集的比例，默认 0.2'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='DataLoader 使用的工作进程数'
    )

    # 预处理和尺寸相关
    parser.add_argument(
        '--k', type=int, default=1,
        help='预处理缩放比例（如时间/空间下采样因子）'
    )
    parser.add_argument(
        '--is_resize', action='store_true',
        help='是否对输入/输出进行 resize 到指定大小'
    )
    parser.add_argument(
        '--H_size', type=int, default=256,
        help='resize 后的高度（像素/网格数）'
    )
    parser.add_argument(
        '--W_size', type=int, default=256,
        help='resize 后的宽度（像素/网格数）'
    )

    # ------------------------------------------------------------------
    #  C. 精度控制（AMP / 混合精度）
    # ------------------------------------------------------------------
    parser.add_argument(
        '--use_amp', action='store_true',
        help='训练时是否使用 torch.cuda.amp 自动混合精度'
    )
    parser.add_argument(
        '--mixed_precision', dest='mixed_precision', action='store_true',
        help='启用更高一层的混合精度模式（训练/推理统一控制）'
    )
    parser.add_argument(
        '--disable_mixed_precision', dest='mixed_precision', action='store_false',
        help='禁用混合精度模式'
    )
    parser.set_defaults(mixed_precision=None)

    # ------------------------------------------------------------------
    #  D. 优化器 & 学习率调度
    # ------------------------------------------------------------------
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='训练总轮数（None 使用配置文件中的默认设置）'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='基础学习率（可覆盖配置文件）'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.05,
        help='L2 正则化系数（weight decay）'
    )
    parser.add_argument(
        '--accum_steps', type=int, default=1,
        help='梯度累积步数（>1 时可用较小显存模拟大 batch）'
    )

    # Warmup 设置
    parser.add_argument(
        '--lr_warmup_epochs', type=int, default=5,
        help='学习率 warmup 持续的 epoch 数'
    )
    parser.add_argument(
        '--lr_warmup_factor', type=float, default=1.0 / 3,
        help='warmup 起始学习率 = base_lr * lr_warmup_factor'
    )
    parser.add_argument(
        '--lr_warmup_method', type=str,
        default='linear', choices=['linear', 'constant'],
        help='warmup 策略：linear 或 constant'
    )

    # 学习率调度器选择
    parser.add_argument(
        '--lr_scheduler_type', type=str,
        default='cos_restart', choices=['cos_restart', 'cos', 'multistep'],
        help='学习率调度器类型'
    )

    # MultiStepLR 参数
    parser.add_argument(
        '--milestones', nargs='+', type=int, default=[30, 60, 90],
        help='MultiStepLR 的学习率衰减里程碑（按 epoch 计）'
    )
    parser.add_argument(
        '--scheduler_gamma', type=float, default=0.3,
        help='MultiStepLR 的学习率衰减因子'
    )

    # Cosine / Cosine 重启参数
    parser.add_argument(
        '--lr_cosine_tmax_epochs', type=float, default=50.0,
        help='Warmup+Cosine 调度中，余弦阶段的周期长度（epoch）'
    )
    parser.add_argument(
        '--lr_cosine_restart_t0_epochs', type=float, default=10.0,
        help='CosineAnnealingWarmRestarts 首个周期长度（epoch）'
    )
    parser.add_argument(
        '--lr_cosine_restart_t_mult', type=int, default=2,
        help='CosineAnnealingWarmRestarts 的周期放大倍率'
    )
    parser.add_argument(
        '--lr_cosine_eta_min', type=float, default=1e-6,
        help='余弦调度阶段的最小学习率'
    )

    # OneCycle 策略
    parser.add_argument(
        '--use_onecycle', dest='use_onecycle', action='store_true',
        help='启用 OneCycle 学习率策略（与其它 scheduler 互斥）'
    )
    parser.add_argument(
        '--disable_onecycle', dest='use_onecycle', action='store_false',
        help='禁用 OneCycle 学习率策略'
    )
    parser.set_defaults(use_onecycle=None)

    # ------------------------------------------------------------------
    #  E. 早停 & 验证频率
    # ------------------------------------------------------------------
    parser.add_argument(
        '--early_stop', action='store_true',
        help='是否启用早停策略（early stopping）'
    )
    parser.add_argument(
        '--early_stop_patience', type=int, default=None,
        help='早停耐心轮数：连续多少个 epoch 无提升就停止'
    )
    parser.add_argument(
        '--early_stop_min_delta', type=float, default=None,
        help='早停判断的最小提升幅度'
    )
    parser.add_argument(
        '--early_stop_warmup_epochs', type=int, default=None,
        help='早停生效前的预热 epoch 数'
    )
    parser.add_argument(
        '--eval_interval', type=int, default=None,
        help='验证间隔（单位：epoch），None 则使用默认策略'
    )

    # ------------------------------------------------------------------
    #  F. 日志、输出 & 可视化
    # ------------------------------------------------------------------
    parser.add_argument(
        '--output_dir', type=str, default='./results',
        help='结果保存目录（模型权重、指标、可视化等）'
    )
    parser.add_argument(
        '--log_root', type=str, default=None,
        help='TensorBoard 日志根目录，默认使用 runs 子目录'
    )
    parser.add_argument(
        '--vis_freq', type=int, default=5,
        help='可视化频率（每隔多少个 epoch 可视化一次）'
    )
    parser.add_argument(
        '--use_wandb', action='store_true',
        help='是否使用 WandB 记录训练过程'
    )
    parser.add_argument(
        '--verbose', dest='verbose', action='store_true',
        help='输出更详细的日志信息'
    )
    parser.add_argument(
        '--quiet', dest='verbose', action='store_false',
        help='最小化日志输出'
    )
    parser.set_defaults(verbose=None)

    # 推理相关
    parser.add_argument(
        '--model_path', type=str, default=None,
        help='推理模式下使用的模型 checkpoint 路径'
    )
    parser.add_argument(
        '--infer_one', type=int, default=None,
        help='推理模式下仅对单一样本进行推理（指定样本索引，基于 eval_split 数据集）'
    )

    # ------------------------------------------------------------------
    #  G. 编码器 & 主干网络
    # ------------------------------------------------------------------
    parser.add_argument(
        '--backbone', type=str, default='vit',
        choices=['vit', 'convnext_tiny'],
        help='encoder 主干网络类型：ViT 或 ConvNeXt-Tiny'
    )
    parser.add_argument(
        '--use_encoder', dest='use_encoder', action='store_true',
        help='启用 encoder，将输入先编码再送入 MoE/算子'
    )
    parser.add_argument(
        '--disable_encoder', dest='use_encoder', action='store_false',
        help='禁用 encoder，直接将原始输入送入 MoE/算子'
    )
    parser.set_defaults(use_encoder=None)
    parser.add_argument(
        '--hidden_channels', type=int, default=128,
        help='模型隐藏通道数（encoder/算子主干宽度）'
    )
    parser.add_argument(
        '--target_size', type=int, default=70,
        help="encoder输出张量形状"
    )
    parser.add_argument(
        '--enc_channels', type=int, default=128,
        help='encoder输出的通道数'
    )
    parser.add_argument(
        '--encoder_path', type=str, default=None,
        help='单独加载 encoder 的 checkpoint，并在训练时可选择冻结其参数'
    )

    # ------------------------------------------------------------------
    #  H. 各类神经算子（FNO / WNO / MNO / LNO）参数
    # ------------------------------------------------------------------
    # FNO 参数
    parser.add_argument(
        '--FNO_n_modes_height', type=int, default=16,
        help='FNO 在高度方向保留的傅里叶模态数量'
    )
    parser.add_argument(
        '--FNO_n_modes_width', type=int, default=16,
        help='FNO 在宽度方向保留的傅里叶模态数量'
    )
    parser.add_argument(
        '--FNO_n_layers', type=int, default=4,
        help='FNO 层数（网络深度）'
    )

    # WNO 参数
    parser.add_argument(
        '--WNO_n_levels_height', type=int, default=2,
        help='WNO 小波分解在高度方向的层数'
    )
    parser.add_argument(
        '--WNO_n_levels_width', type=int, default=2,
        help='WNO 小波分解在宽度方向的层数'
    )
    parser.add_argument(
        '--WNO_n_layers', type=int, default=4,
        help='WNO 块的数量，控制 WNO 模型深度'
    )
    parser.add_argument(
        '--WNO_dropout_rate', type=float, default=0.1,
        help='WNO 块中的 dropout 比例，提高泛化能力'
    )
    parser.add_argument(
        '--wavelet_type', type=str, default='db6',
        choices=['coif4', 'db4', 'db8', 'sym4', 'coif5', 'sym8', 'db6', 'sym6'],
        help='实值小波类型'
    )
    parser.add_argument(
        '--dtcwt_type', nargs=2, type=str, default=None,
        help='双树复小波类型（两个小波名，例如：near_sym_a near_sym_b）'
    )

    # MNO 参数
    parser.add_argument(
        '--MNO_n_scales', type=int, default=3,
        help='MNO 使用的总尺度数'
    )
    parser.add_argument(
        '--MNO_scale_factors', nargs='+', type=float,
        default=[1.0, 0.6, 0.3],
        help='每个尺度的空间/时间缩放因子'
    )
    parser.add_argument(
        '--MNO_n_layers', type=int, default=3,
        help='每个尺度上的网络层数'
    )

    # LNO 参数
    parser.add_argument(
        '--LNO_n_modes', nargs=2, type=int, default=[16, 16],
        help='LNO 局部变换后保留的模态数量（高, 宽）'
    )
    parser.add_argument(
        '--LNO_n_layers', type=int, default=3,
        help='LNO 网络层数'
    )

    # ------------------------------------------------------------------
    #  I. MoE / 专家选择 & 路由 / 融合
    # ------------------------------------------------------------------
    # 专家选择 & Top-k
    parser.add_argument(
        '--top_k', type=int, default=1,
        help='每个样本选择的专家数量（Top-k）'
    )
    parser.add_argument(
        '--choose_experts', nargs='*', type=int, default=None,
        help='专家选择列表，例如 FNO:0, WNO:1, MNO:2, LNO:3'
    )

    # MoE 方法 & 模式
    parser.add_argument(
        '--moe_method', type=str, default="afmoe",
        choices=["basic", "afmoe"],
        help='MoE 融合方法：basic 或 afmoe（自适应融合）'
    )
    parser.add_argument(
        '--use_experts_path', type=str, default=None,
        help='MoE 使用的预训练专家模型存放路径'
    )
    parser.add_argument(
        '--use_moe', action='store_true',
        help='是否使用 MoE（通常会冻结专家，仅训练路由和融合模块）'
    )
    parser.add_argument(
        '--moe_mode', type=str, default='standard',
        choices=['standard', 'velocity_type', 'group'],
        help=(
            "MOE 运行模式："
            "'standard' 使用路由/融合；"
            "'velocity_type' 按速度类型权重融合预训练专家；"
            "'group' 使用分组 MoE（如按类别分专家组）"
        )
    )

    # 路由器设置
    parser.add_argument(
        '--router_type', type=str, default='basic',
        help="路由器类型，例如 'basic' / 'adamv' 等"
    )
    parser.add_argument(
        '--band_sharpness', type=float, default=20.0,
        help='AFreqMoE 软频带的锐度（越大越接近硬分段）'
    )
    parser.add_argument(
        '--freq_affinity_sharpness', type=float, default=10.0,
        help='专家频率偏好与频带中心匹配的锐度（控制频带混合）'
    )
    parser.add_argument(
        "--disable_band_decomposition", action='store_true',
        help='消融：关闭频带分解，直接将输入送入专家'
    )
    parser.add_argument(
        '--disable_soft_bands', action='store_true',
        help='消融：关闭软频带划分，改为硬划分'
    )
    parser.add_argument(
        '--disable_freq_attn', action='store_true',
        help='消融：关闭频率自注意力，直接使用幅度谱作为路由特征'
    )
    parser.add_argument(
        '--disable_band_mixing', action='store_true',
        help='消融：关闭频带混合输入，专家仅接收对应频带'
    )
    parser.add_argument(
        '--enable_freq_metrics', action='store_true',
        help='在验证/推理阶段计算低/中/高频 band 指标'
    )
    parser.add_argument(
        '--router_hidden_dim', type=int, default=None,
        help='路由器隐藏层宽度（若为 None 使用默认配置）'
    )
    parser.add_argument(
        '--routing_mode', type=str, default=None,
        choices=['learned', 'uniform', 'random'],
        help='路由模式：learned=正常学习；uniform=均匀分配；random=随机 top-k'
    )

    # 专家组间/组内融合方式
    parser.add_argument(
        '--fusion_type', type=str, default='linear',
        help="专家组间融合方式：'linear'/'attention'/'swa'/'basic(sum)'"
    )
    parser.add_argument(
        '--s_processor_type', type=str, default='linear',
        help="强专家组内融合方式：'linear'/'attention'/'mean'/'sum'"
    )
    parser.add_argument(
        '--w_processor_type', type=str, default='linear',
        help="弱专家组内融合方式：'linear'/'attention'/'mean'/'sum'"
    )

    # 门控 & 强弱专家权重
    parser.add_argument(
        '--enable_noisy_gating', dest='noisy_gating', action='store_true',
        help='启用带噪声的 gating（noisy top-k）'
    )
    parser.add_argument(
        '--disable_noisy_gating', dest='noisy_gating', action='store_false',
        help='禁用带噪声 gating'
    )
    parser.set_defaults(noisy_gating=None)
    parser.add_argument(
        '--beta', type=float, default=0.5,
        help='强弱激活参数，beta 越大，弱专家的影响越大'
    )

    # 特定类型 & 分类器 & 显存代理
    parser.add_argument(
        '--is_specific', action='store_true',
        help='是否对数据种类做细化（如按 CurveVelA/B 分别建专家/权重）'
    )
    parser.add_argument(
        '--is_classifier', action='store_true',
        help='是否使用分组专家网络（带分类器的 MoE）'
    )
    parser.add_argument(
        '--v_type_num', type=int, default=None,
        help='速度类型数量，用于控制分类器输出维度及专家分组数'
    )
    parser.add_argument(
        '--use_gpu_proxy', action='store_true',
        help='启用专家显存代理（CPU<->GPU 动态加载专家，缓解显存压力）'
    )

    # ------------------------------------------------------------------
    #  J. 损失函数权重
    # ------------------------------------------------------------------
    parser.add_argument(
        '-g1v', '--lambda_g1v', type=float, default=0.6,
        help='全局速度图损失项 g1v 的权重'
    )
    parser.add_argument(
        '-g2v', '--lambda_g2v', type=float, default=0.4,
        help='全局速度图损失项 g2v 的权重'
    )
    parser.add_argument(
        '--lambda_grad_l1', type=float, default=0.15,
        help='梯度（边界）L1 损失权重，用于强化层界面'
    )
    parser.add_argument(
        '--lambda_fourier_mag_l1', type=float, default=0.10,
        help='频谱幅度 L1 损失权重，用于约束频域一致性'
    )
    parser.add_argument(
        '--lambda_ce', type=float, default=0.20,
        help='交叉熵分类损失权重（若启用分类器/分组专家）'
    )

    # ------------------------------------------------------------------
    #  解析参数并回传 parser
    # ------------------------------------------------------------------
    args = parser.parse_args(argv)
    args.parser = parser
    return args

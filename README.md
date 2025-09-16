# 地震数据MOE (Mixture of Experts) 神经算子指南

本指南详细介绍了如何使用、配置和修改seismic_moe代码，以便于处理地震数据的神经算子模型训练和推理。

## 目录

0. [工作分配](#工作分配)
1. [环境配置](#环境配置)
2. [数据集格式](#数据集格式)
3. [代码使用方法](#代码使用方法)
4. [参数说明](#参数说明)
5. [模型架构](#模型架构)
6. [修改指南](#修改指南)
7. [分布式训练](#分布式训练)
8. [结果评估](#结果评估)
9. [常见问题]

## 工作分配

<a id="工作分配"></a>

第一阶段: 单个专家模型性能测试, 特点分析

- syx:WNO专家调试

- lpy:MNO专家调试

- wzy:FNO, LNO专家调试

- 参数设置: 可以一起思考调试

- 目标: 能运行，效果接近benchmark

第二阶段: 模型融合与数学方法(待定)

文件命名协议
normal(未细化):

- 模型分为vel, fault, style三类
- 命名约定：best\_expert\_{experts\_name}\_{i}\_{vel/fault/style}.pt
  
specific(细化)：

- 模型分为curve_vel, curve_fault, flat_vel, flat_fault, style五类
- 命名约定：best_expert_{experts_name}\_{i}\_{curve/flat/style}_{vel/fault/style}.pt

新增功能以及相关参数

In MOEOperator

1. router_type —— adamv, 自适应专家数量选择 'basic'/'adamv'
2. fusion_type —— str, 输出融合模式，'linear'/'attention'/'swa', 新增强弱激活模式
3. s_processor_type —— 强专家输出融合器种类 'linear'/'atten'/'mean'/'sum'
4. w_processor_type —— 弱专家输出融合器种类
5. beta —— 强弱激活参数，beta越大，弱激活影响越大
6. is_specific —— 文件名命名协议选择
7. is_classier —— 是否使用分组专家网络 GMoE

## 环境配置

<a id="环境配置"></a>

### 系统要求

- Python 3.10+
- CUDA 10.2+ (推荐使用GPU进行训练)
- 充足的存储空间用于数据集

### 安装步骤

1. 克隆仓库并进入项目目录

```bash
git clone https://github.com/GrinchWumath/FWINO.git
cd neuraloperator
```

2. 创建虚拟环境(可选)

```bash
conda create -n seismic_moe python=3.8
conda activate seismic_moe
```

3. 安装依赖包

```bash
pip install -r requirements.txt
```

主要依赖包包括：

- torch >= 1.10.0
- numpy
- matplotlib
- tqdm
- scikit-image (用于评估指标)
- wandb (可选，用于实验追踪)

## 数据集格式

<a id="数据集格式"></a>

### 数据集结构

seismic_moe代码支持处理地震数据集，其目录结构应如下：

```
data_dir/
├── train_samples/
│   ├── sample_folder_1/
│   │   ├── model/
│   │   │   └── model{i}.npy  # 速度模型
│   │   └── data/
│   │       └── data{i}.npy   # 对应的地震数据
│   ├── sample_folder_2/
│   │   └── ...
│   └── Fault_A/ or Fault_B/  # 断层数据
│       ├── vel_{n}_1_{i}.npy # 速度模型
│       └── seis_{n}_1_{i}.npy # 地震数据
└── test/
    └── *.npy  # 测试数据
```

### 数据格式说明

1. **训练数据**:
   - 地震数据，波形数据（输入）: 形状为 `[batch_size, num_sources, time_steps, num_receivers]` 的NumPy数组
   - 速度模型，速度图（输出）: 形状为 `[batch_size, height, width]` 的NumPy数组
   - 通常 `num_sources=5`, `time_steps=1000`, `num_receivers=70`

2. **数据集系列**:
   - `vel`: 速度模型数据
   - `style`: 风格化数据
   - `fault`: 断层数据
   - `all`: 使用所有可用数据

### 自定义数据集

如需使用自定义数据集，应确保：

1. 遵循上述目录结构
2. 数据格式为NumPy `.npy`文件
3. 数据维度与上述描述一致

如果数据格式不同，需要修改`neuralop/data/datasets/seismic_dataset.py`文件中的`_load_data`和`__getitem__`方法。

## 代码使用方法

<a id="代码使用方法"></a>

### MoE架构训练

当且仅当命令行中给出了 `--use_moe` `--use_experts_path` 且 `--top_k` > 1 且 `--choose_experts` 选择了多个专家，才会进行专家模型参数的冻结, 否则都是重新训练

当单个专家模型训练好后, 将其放入一个专门的文件夹, 文件夹内, 四个专家四个pt文件, 将文件夹路径作为参数输入命令行

### 单GPU训练

```bash
python scripts/train_seismic_moe.py \
    --data_dir /path/to/your/data \
    --family all \
    --batch_size 8 \
    --epochs 100 \
    --output_dir ./results/seismic_moe
```

```bash
python scripts/train_seismic_moe.py --data_dir .\FWINO_data --family vel --batch_size 2 --epochs 100 --output_dir ./results/seismic_moe
```

### 分布式训练

```bash
bash scripts/run_distributed_seismic_moe.sh \
    --num_gpus 4 \
    --data_dir /path/to/your/data \
    --family all \
    --batch_size 16 \
    --epochs 100 \
    --output_dir ./results/distributed_seismic_moe
```

```bash
sbatch submit_FWINO_XXX.sh
```

现在代码训练的结果存储中增加了使用的专家信息，同时如果是单一专家训练，会在单独的文件夹存储可能会用到的专家模型

### 模型推理

```bash
python scripts/train_seismic_moe.py \
    --mode inference \
    --model_path ./results/seismic_moe/best_model.pt \
    --data_dir /path/to/test/data \
    --output_dir ./results/predictions
```

### 模型评估

```bash
python scripts/evaluate_moe.py \
    --model_path ./results/seismic_moe/best_model.pt \
    --dataset_type seismic \
    --data_path /path/to/your/data \
    --family all \
    --output_dir ./evaluation_results
```

## 参数说明

<a id="参数说明"></a>

<!-- 设置跳转目标锚点 -->
<h3 id="args-table">训练脚本参数</h3>

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--mode` | str | train | 运行模式: 训练或推理 |
| `--data_dir` | str | None | 数据目录路径 |
| `--family` | str | all | 数据集系列: vel, style, fault 或 all |
| `--batch_size` | int | 8 | 批次大小 |
| `--epochs` | int | 100 | 训练轮数 |
| `--num_workers` | int | 4 | 数据加载工作进程数 |
| `--seed` | int | 42 | 随机种子 |
| `--output_dir` | str | ./results | 结果保存目录 |
| `--model_path` | str | None | 推理模式下使用的模型路径 |
| `--vis_freq` | int | 5 | 可视化频率（每隔多少个epoch） |
| `--distributed` | bool | False | 是否使用分布式训练 |
| `--use_wandb` | bool | False | 是否使用WandB记录训练过程 |
| `--val_ratio` | float | 0.2 | 验证集比例 |
| `--top_k` | int | 1 | 选择前k个专家 |
| `--choose_experts` | int nargs | [0] | 选择一个或多个专家，FNO:0，WNO:1，MNO:2，LNO:3 |
| `--FNO_n_modes_height` | int | 16 | 高度傅里叶变换后保留的模态数量 |
| `--FNO_n_modes_width` | int | 16 | 宽度傅里叶变换后保留的模态数量 |
| `--WNO_n_levels_height` | int | 2 | 高度减少级别 |
| `--WNO_n_levels_width` | int | 2 | 宽度减少级别 |
| `--MNO_n_scales` | int | 3 | 总共使用的尺度 |
| `--MNO_scale_factors` | float nargs | [1.0, 0.5, 0.25] | 每个尺度的缩放因子 |
| `--MNO_n_layers` | int | 3 | 每个尺度使用的神经网络层数 |
| `--LNO_n_modes` | int nargs | (16, 16) | 局部变换后保留的模态数量 |
| `--LNO_n_layers` | int | 3 | 每个尺度使用的神经网络层数 |
| `--k` | int | 1 | 数据预处理缩放尺度|
| `--lambda_g1v` | float | 1.0 | L1损失函数的加权系数 |
| `--lambda_g2v` | float | 1.0 | L2损失函数的加权系数 |
| `--use_experts_path` | str | None | moe使用的专家模型存放路径 |
| `--use_moe` | bool | False | 是否使用moe, 使用会冻结专家模型 |

---

### 配置文件参数 (config/seismic_moe_config.py)

| 参数类别 | 参数名 | 默认值 | 说明 |
|----------|--------|--------|------|
| 基本配置 | model_name | MOE | 模型名称 |
|  | in_channels | 1 | 输入通道数 |
|  | out_channels | 5 | 输出通道数 |
|  | hidden_channels | 64 | 隐藏层通道数 |
| 数据集配置 | dataset_name | seismic | 数据集名称 |
|  | data_dir | /data1/... | 数据目录路径 |
|  | family | all | 数据集系列 |
|  | normalize_inputs | True | 是否归一化输入 |
|  | normalize_outputs | True | 是否归一化输出 |
|  | channel_dim | 1 | 通道维度 |
| MOE配置 | top_k | 2 | 选择前k个专家 |
|  | noisy_gating | True | 是否使用噪声门控 |
|  | fusion_type | linear | 专家输出融合方式 |
|  | router_hidden_dim | 256 | 路由器隐藏层维度 |
| 训练配置 | batch_size | 8 | 批次大小 |
|  | learning_rate | 1e-3 | 学习率 |
|  | weight_decay | 1e-4 | 权重衰减 |
|  | epochs | 100 | 训练轮数 |
|  | milestones | [30, 60, 90] | 学习率调整点 |
|  | scheduler_gamma | 0.5 | 学习率衰减系数 |
| 分布式配置 | use_distributed | False | 是否使用分布式训练 |
|  | model_parallel_size | 1 | 模型并行大小 |
|  | seed | 42 | 随机种子 |

## 模型架构

<a id="模型架构"></a>

seismic_moe模型是基于混合专家神经算子（Mixture of Experts Neural Operator）架构，包含以下主要组件：

1. **Router（路由器）**: 负责将输入分配给最合适的专家。
   - 支持基本路由器和任务感知路由器(TaskAwareRouter)

2. **Experts（专家）**: 默认配置包含四种专家:
   - 傅里叶域专家: 捕捉频率特征
   - 小波域专家: 处理局部特征和多尺度结构
   - 多尺度专家: 处理多尺度地质结构
   - 本地处理专家: 用于局部细节重建

3. **Fusion Layer（融合层）**: 整合各专家输出
   - 支持线性融合和注意力融合

4. **Time-to-Space Projection（时间-空间转换）**: 将时间-偏移表示表示转换为空间表示，适用于输入为速度模型数据。

## 修改指南

<a id="修改指南"></a>

### 修改网络架构

1. **调整专家配置**:

- 基本参数调整
    
    直接通过修改脚本中的命令参数来调整，详细参数设置可以参考上方<a href="#args-table">训练脚本参数表格</a>

    实例：

    ```bash
    bash scripts/run_distributed_seismic_moe.sh --num_gpus 2 --data_dir ../FWINO_data --family all --batch_size 1000 --epochs 100 --output_dir ../results/seismic_moe_${SLURM_JOB_NAME}_${SLURM_JOB_ID}  --top_k 2 --choose_experts 0 2 --MNO_n_scales 5
    ```
    解释: 使用FNO和MNO两个专家，调整MNO的n_scales参数为5

- 更多参数修改
  
    不推荐，如果需要修改更多参数，可以提交issue，直接增加命令行参数设置

    修改`config/seismic_moe_config.py`中的`expert_configs`列表：

```python
expert_configs = [
    # 傅里叶域专家
    {
        'type': 'domain',
        'domain_type': 'fourier',
        'n_dim': 2,
        'n_modes_height': 16,  # 调整模式数
        'n_modes_width': 16,   # 调整模式数
        # 添加或修改参数
    },
    # 添加更多专家...
]
```

1. **修改路由器**:
   调整`MOEOperator`初始化参数：

```python
model = MOEOperator(
    experts=experts,
    in_channels=config.in_channels,
    out_channels=config.out_channels,
    hidden_channels=128,  # 增加隐藏层通道数
    top_k=3,  # 增加选择的专家数量
    router_type='task_aware',  # 使用任务感知路由器
    task_dim=8,  # 设置任务特征维度
    routing_mode='both'  # 设置路由模式
)
```

3. **自定义专家**:
   创建自定义专家并添加到`expert_factory.py`中。

### 修改训练过程

1. **调整损失函数**:
   修改`scripts/train_seismic_moe.py`中的`criterion`：

```python
# 使用带权重的MSE损失
criterion = lambda pred, target: F.mse_loss(pred, target) * weight_factor
```

2. **添加正则化**:
   修改优化器配置：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=1e-3,  # 增加权重衰减
)
```

3. **修改学习率调度**:
   调整学习率调度器：

```python
scheduler = torch.optim.CosineAnnealingLR(  # 使用余弦退火调度
    optimizer,
    T_max=config.epochs,
    eta_min=1e-6
)
```

### 修改数据处理

修改`SeismicDataProcessor`类中的数据处理逻辑：

```python
def __call__(self, sample):
    # 获取输入和输出
    if 'input' in sample:
        x = sample['input']
        
        # 添加数据增强
        if self.training:
            # 随机裁剪
            x = random_crop(x, size=(64, 64))
            
            # 随机翻转
            if random.random() > 0.5:
                x = torch.flip(x, dims=(2,))
                
        sample['input'] = x
    # ...其余代码保持不变
```

## 分布式训练
<a id="分布式训练"></a>

### 配置分布式训练

1. 修改配置文件启用分布式训练：

```python
# config/seismic_moe_config.py
distributed = DistributedConfig(
    use_distributed=True,
    model_parallel_size=1,
    seed=42
)
```

2. 使用`run_distributed_seismic_moe.sh`脚本启动分布式训练：

```bash
bash scripts/run_distributed_seismic_moe.sh \
    --num_gpus 4 \
    --data_dir /path/to/data \
    --batch_size 16 \
    --epochs 100
```

### 分布式训练性能优化

1. 增加每个GPU的批大小
2. 启用混合精度训练：`mixed_precision=True`
3. 调整数据加载工作进程数：`--num_workers 8`

## 结果评估
<a id="结果评估"></a>

使用`evaluate_moe.py`脚本评估训练模型的性能：

```bash
python scripts/evaluate_moe.py \
    --model_path ./results/seismic_moe/best_model.pt \
    --dataset_type seismic \
    --data_path /path/to/validation/data \
    --batch_size 16 \
    --output_dir ./evaluation
```

该脚本输出以下指标：
- MSE (均方误差)
- MAE (平均绝对误差)
- PSNR (峰值信噪比)
- 相对L2误差
- SSIM (结构相似性指数，需安装scikit-image)

同时生成可视化结果，包括输入、目标和预测对比图。


import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Union, Optional
import re
from pathlib import Path

class SeismicDataset(Dataset):
    """
    用于地震数据的数据集类
    
    处理OpenFWI数据集中的地震波形数据和对应的速度图
    
    Parameters
    ----------
    data_dir : str
        数据目录路径
    family : str
        数据集系列，可选 'vel', 'style', 'fault' 或 'all'
    is_specific : bool
        是否细化分类, 细化后: 'curve_vel', 'flat_vel', 'curve_fault', 'flat_fault', 'style_style'
    split : str
        数据集分割，可选 'train' 或 'test'
    """
    def __init__(
        self,
        data_dir: str,
        family: str = 'all',
        is_specific: bool = False,
        split: str = 'train',
    ):
        self.data_dir = data_dir
        self.family = family.lower()
        self.split = split
        self.index_map = [] #[(file_idx, sample_idx)]
        self.input_tensors = []
        self.output_tensors = []
        self.is_specific = is_specific
        
        # 验证参数
        if not is_specific:
            if self.family not in ['vel', 'style', 'fault', 'all']:
                raise ValueError(f"不支持的数据集系列: {self.family}")
        else:
            if self.family not in ['curve_vel', 'flat_vel', 'curve_fault', 'flat_fault', 'style_style']:
                raise ValueError(f"不支持的数据集系列: {self.family}")
            
        if self.split not in ['train', 'test']:
            raise ValueError(f"不支持的数据集分割: {self.split}")
        # 加载数据
        self._load_data()
        
        # 扫描所有文件，记录每个样本的索引
        for file_idx, file in enumerate(self.input_files):
            data = np.load(file, mmap_mode='r')  # lazy load
            num_samples = data.shape[0]
            for sample_idx in range(num_samples):
                self.index_map.append((file_idx, sample_idx))
                
        # 加载到内存
        self._preload_into_memory()
            
        # 计算归一化统计量
        self._compute_stats()
    
    def _preload_into_memory(self):
        for i in range(len(self.input_files)):
            try:
                input_array = np.load(self.input_files[i], allow_pickle=True)
                if self.output_files[i] is not None:
                    output_array = np.load(self.output_files[i], allow_pickle=True)
                for j in range(input_array.shape[0]):
                    input_data = np.concatenate([input_array[j][k] for k in range(5)], axis=1)
                    input_tensor = torch.from_numpy(input_data.astype(np.float32)).unsqueeze(0)  # 1×1000×350
                    self.input_tensors.append(input_tensor)

                    if self.output_files[i] is not None:  # b*h*w
                        output_tensor = torch.from_numpy(output_array[j].astype(np.float32))  # 1×70×70
                        self.output_tensors.append(output_tensor)
            except Exception as e:
                print(f"读取第{i}个文件失败: {e}")
                
    def _load_data(self):
        """
        加载数据文件路径  
        """
        self.input_files = []
        self.output_files = []

        def want_family(group: str, variant: str | None) -> bool:
            """根据 family / is_specific 判定是否保留该目录"""
            if not getattr(self, 'is_specific', False):
                if self.family == 'all':
                    return True
                return self.family == group
            else:
                if group == 'style':
                    return self.family == 'style_style'
                if variant is None:
                    return False
                return self.family == f"{variant}_{group}"

        if self.split == 'train':
            train_dir = os.path.join(self.data_dir, 'train_samples')
            if not os.path.isdir(train_dir):
                raise RuntimeError(f"训练目录不存在: {train_dir}")

            subdirs = [d for d in os.listdir(train_dir)
                    if os.path.isdir(os.path.join(train_dir, d))]

            for sub in sorted(subdirs):
                sub_path = os.path.join(train_dir, sub)

                # 目录名到 (group, variant) 的判定
                group = None
                variant = None
                if sub.startswith("CurveFault_"):
                    group, variant = 'fault', 'curve'
                elif sub.startswith("FlatFault_"):
                    group, variant = 'fault', 'flat'
                elif sub.startswith("FlatVel_"):
                    group, variant = 'vel', 'flat'     
                elif sub.startswith("Style_"):
                    group, variant = 'style', None
                elif sub.startswith("CurveVel_"):
                    group, variant = 'vel', 'curve'
                else:
                    # 未知命名，跳过（也可选择 raise）
                    continue

                if not want_family(group, variant):
                    continue

                if group == 'fault':
                    # Fault：同目录下 seis_?{n}_1_{i}.npy ↔ vel_{n}_1_{i}.npy
                    pattern = re.compile(r"seis_?(\d+)_1_(\d+)\.npy")
                    seis_files = sorted(glob.glob(os.path.join(sub_path, 'seis*.npy')))
                    for seis_file in seis_files:
                        stem = Path(seis_file).name
                        m = pattern.fullmatch(stem)
                        if m:
                            n = int(m.group(1))
                            i = int(m.group(2))
                        else:
                            print(f"{stem} 跳过")
                            continue
                        try:
                            vel_file = os.path.join(sub_path, f"vel_{n}_1_{i}.npy")
                            assert os.path.exists(vel_file)
                        except Exception:
                            vel_file = os.path.join(sub_path, f"vel{n}_1_{i}.npy")
                        if os.path.exists(vel_file):
                            self.input_files.append(seis_file)   # 输入：seis
                            self.output_files.append(vel_file)   # 输出：vel

                elif group in ('vel', 'style'):
                    # Vel/Style：data/{data{i}.npy} ↔ model/{model{i}.npy}
                    data_dir = os.path.join(sub_path, 'data')
                    model_dir = os.path.join(sub_path, 'model')
                    if not (os.path.isdir(data_dir) and os.path.isdir(model_dir)):
                        continue

                    data_files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
                    for data_file in data_files:
                        base = os.path.basename(data_file)   # data{i}.npy
                        stem, _ = os.path.splitext(base)
                        if not stem.startswith('data'):
                            continue
                        idx = stem.replace('data', '')
                        model_file = os.path.join(model_dir, f"model{idx}.npy")
                        if os.path.exists(model_file):
                            self.input_files.append(data_file)   # 输入：data
                            self.output_files.append(model_file) # 输出：model

        else:
            # test：只需要输入
            test_dir = os.path.join(self.data_dir, 'test')
            if not os.path.isdir(test_dir):
                raise RuntimeError(f"测试目录不存在: {test_dir}")
            self.input_files = sorted(glob.glob(os.path.join(test_dir, '*.npy')))
            self.output_files = [None] * len(self.input_files)

        # 校验
        if not self.input_files:
            raise RuntimeError(
                f"未找到任何输入文件。请检查路径与过滤条件："
                f"data_dir={self.data_dir}, split={self.split}, "
                f"family={self.family}, is_specific={getattr(self, 'is_specific', None)}"
            )
        if self.split == 'train' and len(self.input_files) != len(self.output_files):
            raise RuntimeError(
                f"输入与输出数量不一致：inputs={len(self.input_files)}, outputs={len(self.output_files)}"
            )
    
    def _compute_stats(self):
        """从已加载的内存数据中计算归一化统计量"""
        n_total = int(len(self.input_tensors))
        n_samples = int(min(n_total * 0.03, 300))  # 3%最多不超过300个
        sample_indices = np.random.choice(n_total, n_samples, replace=False)

        # 计算输入的统计量
        input_values = []
        for idx in sample_indices:
            try:
                input_tensor = self.input_tensors[idx]  # shape: [1, 1000, 350]
                flat = input_tensor.view(-1)  # 展平为 1D
                sample_points = flat[torch.randperm(flat.numel())[:min(1000, flat.numel())]]
                input_values.append(sample_points)
            except Exception as e:
                print(f"警告: 处理输入样本 {idx} 失败: {e}")

        if input_values:
            input_all = torch.cat(input_values)
            self.input_min = float(torch.min(input_all))
            self.input_max = float(torch.max(input_all))
            self.input_mean = float(torch.mean(input_all))
            self.input_std = float(torch.std(input_all))
            if self.input_std == 0:
                self.input_std = 1.0

        # 计算输出的统计量（仅 train 阶段）
        if self.split == 'train' and hasattr(self, 'output_tensors') and self.output_tensors:
            output_values = []
            for idx in sample_indices:
                try:
                    output_tensor = self.output_tensors[idx]  # shape: [1, 70, 70]
                    flat = output_tensor.view(-1)
                    sample_points = flat[torch.randperm(flat.numel())[:min(1000, flat.numel())]]
                    output_values.append(sample_points)
                except Exception as e:
                    print(f"警告: 处理输出样本 {idx} 失败: {e}")

            if output_values:
                output_all = torch.cat(output_values)
                self.output_min = float(torch.min(output_all))
                self.output_max = float(torch.max(output_all))
                self.output_mean = float(torch.mean(output_all))
                self.output_std = float(torch.std(output_all))
                if self.output_std == 0:
                    self.output_std = 1.0
            
    def getStats(self):
        return {
            'input_max' : self.input_max,
            'input_min' : self.input_min,
            'input_mean' : self.input_mean,
            'input_std' : self.input_std,
            'output_max' : self.output_max,
            'output_min' : self.output_min,
            'output_mean' : self.output_mean,
            'output_std' : self.output_std
        }
        
    def __len__(self):
        return len(self.input_tensors)
         
    def __getitem__(self, idx):
        # 加载输入数据
        try:
            file_idx, _  = self.index_map[idx]
        
            # 获取输入文件名（对测试集预测有用）
            input_filename = os.path.basename(self.input_files[file_idx])
            
            # 处理训练集和测试集
            if self.split == 'train': # 训练集和验证集
                # 返回配对的输入和输出
                sample = {
                    'input': self.input_tensors[idx].clone(),
                    'output': self.output_tensors[idx].clone(),
                    'input_file': input_filename
                }
            else:  # 测试集
                # 只返回输入数据
                sample = {
                    'input': self.input_tensors[idx].clone(),
                    'input_file': input_filename.split('.')[0]  # 去除扩展名
                }
            return sample
            
        except Exception as e:
            print(f"加载样本 {idx} 失败: {e}")
            # 返回一个空样本，或者重新尝试另一个索引
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(0)
    
    def get_input_size(self):
        """获取输入数据的形状"""
        # 加载第一个输入文件，获取形状
        try:
            data = self.input_tensors
            if data.shape[0] > 1:
                return data[0].shape
            else:
                return data.shape
        except Exception as e:
            print(f"获取输入大小失败: {e}")
            return None
    
    def get_output_size(self):
        """获取输出数据的形状"""
        if self.split == 'test':
            return None
        
        try:
            data = self.output_tensors
            if data.shape[0] > 1:
                return data[0].shape
            else:
                return data.shape
        except Exception as e:
            print(f"获取输出大小失败: {e}")
            return None

class SeismicDataProcessor:
    """
    地震数据处理器
    
    用于预处理地震数据，包括维度转换、通道调整等
    
    Parameters
    ----------
    channel_dim : int, optional
        通道维度，默认为1
    input_transform : callable, optional
        额外的输入转换函数，默认为None
    output_transform : callable, optional
        额外的输出转换函数，默认为None
    """
    def __init__(
        self,
        channel_dim: int = 0,
        input_transform : callable = None,
        output_transform : callable = None
    ):
        self.channel_dim = channel_dim
        # input_transform可以使用transforms中写好的类
        self.input_transform = input_transform
        self.output_transform = output_transform
          
    def __call__(self, sample):
        """
        处理地震数据或速度图

        Args:
            sample : dict 包含input和output, input为torch.Tensor 形状为(1, 1000, 350) output为torch.Tensor 形状为(1, 70, 70)

        Returns:
            sample : 预处理后的dict 包含input和output, input为torch.Tensor 形状为(1, 1000, 350) output为torch.Tensor 形状为(1, 70, 70)
        """
        # 获取输入和输出
        if 'input' in sample:
            
            x = sample['input']
            # 输入数据现在是地震数据或数据文件
            # 地震数据维度调整
            # 输入数据形状为 (num_sources, time_steps, num_receivers)
            
            # 将震源维度作为通道维度
            if self.channel_dim == 0:
                # 调整为 (num_sources, time_steps, num_receivers)，第0维作为通道维
                pass  # 默认已经是这个形状
            else:
                # 移动通道维度到指定位置
                if self.channel_dim == 1: # 第一维作为通道维
                    # (num_sources, time_steps, num_receivers) -> (time_steps, num_receivers, num_sources)
                    x = torch.permute(x, (1, 2, 0))
                elif self.channel_dim == -1: # 最后一维作为通道维
                    # (num_sources, time_steps, num_receivers) -> (num_receivers, num_sources, time_steps)
                    x = torch.permute(x, (2, 0, 1))
            
            # 应用额外的转换
            # log
            if self.input_transform:
                x = self.input_transform(x)
                
            sample['input'] = x

        # 处理输出
        if 'output' in sample and sample['output'] is not None:
            y = sample['output']
            
            # 输出数据现在是速度图或模型文件
            # 如果输出是2D数据，添加通道维度
            if len(y.shape) == 2:
                # (height, width) -> (1, height, width)
                y = y.unsqueeze(0)
            
            # 应用额外的转换
            if self.output_transform:
                y = self.output_transform(y)
                
            sample['output'] = y
        
        return sample


def create_seismic_dataloader(
    data_dir: str,
    family: str = 'all',
    split: str = 'train',
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
    channel_dim: int = 1,
    input_transform = None,
    output_transform = None
):
    """
    创建地震数据加载器
    
    Parameters
    ----------
    data_dir : str
        数据目录路径
    family : str, optional
        数据集系列，可选 'vel', 'style', 'fault' 或 'all'，默认为'all'
    split : str, optional
        数据集分割，可选 'train' 或 'test'，默认为'train'
    batch_size : int, optional
        批次大小，默认为16
    shuffle : bool, optional
        是否打乱数据，默认为True
    num_workers : int, optional
        数据加载工作进程数，默认为4
    normalize_inputs : bool, optional
        是否归一化输入数据，默认为True
    normalize_outputs : bool, optional
        是否归一化输出数据，默认为True
    channel_dim : int, optional
        通道维度，默认为1
    input_transform : callable, optional
        额外的输入转换函数，默认为None
    output_transform : callable, optional
        额外的输出转换函数，默认为None
    
    Returns
    -------
    dataloader : DataLoader
        PyTorch数据加载器
    dataset : SeismicDataset
        数据集实例，用于访问归一化参数等
    """
    # 创建数据集
    dataset = SeismicDataset(
        data_dir=data_dir,
        family=family,
        split=split,
        transform=SeismicDataProcessor(
            channel_dim=channel_dim,
            input_transform=input_transform,
            output_transform=output_transform
        )
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == 'train' else False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset 
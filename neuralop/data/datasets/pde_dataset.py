# neuralop/data/datasets/pdebench_dataset.py

import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PDEBenchDataset(Dataset):
    """
    PDEBench 通用 Dataset 封装：

    假定每个任务对应一个或多个 HDF5 文件，结构大致为：
      - f[input_key]  : [N, ...] 输入场
      - f[target_key] : [N, ...] 目标场

    为了兼容 Burgers / Navier-Stokes / Darcy 等任务，做了若干约定：
      1) 无论原数据是 1D/2D/时序，最后都会转成 [C, H, W] 的 tensor：
         - 若数据为 [H] 或 [X]           -> [1, 1, X]
         - 若数据为 [H, W]               -> [1, H, W]
         - 若数据为 [T, H]               -> [T, 1, H]      (time_as_channel=False 时可选)
         - 若数据为 [T, H, W] 且 time_as_channel=True -> [T, H, W]
         - 若数据为 [T, H, W] 且 time_as_channel=False -> 默认最后一维为时间，取 target_time_index
      2) target 若是时序数据 [T, ...]，默认取 target_time_index=-1（最后一个时间步）
      3) transform 机制：
         - transform(sample: Dict[str, Tensor]) -> Dict[str, Tensor]   （最高优先级）
         - input_transform(x: Tensor) / output_transform(y: Tensor)   （次优先级）

    参数
    ----
    task: str
        任务名，例如 'burgers1d', 'navier2d', 'darcy2d'。
        主要用于推文件名模式和 meta 信息记录。

    root: str or Path
        数据根目录，例如 "./pdebench_data"。

    split: str
        数据划分：'train' / 'val' / 'test'。如果没有 'val'，
        你可以自己在外面用 'train' + Subset 来切分。

    file_name: Optional[str]
        显式指定 HDF5 文件名；若为 None，则使用常见模式自动猜测：
          - {task}_{split}.h5
          - {task}_{split}.hdf5
          - {task}.h5
          - {task}.hdf5

    input_key: str
        HDF5 中输入数据集名称，默认 'input'。

    target_key: str
        HDF5 中目标数据集名称，默认 'output'。

    target_time_index: Optional[int]
        若 target 是时序数据（形状首维为 time），用哪个时间索引作为监督信号。
        默认 -1 表示最后一帧；若为 None，则保留整个时序，并按 time_as_channel 规则处理。

    time_as_channel: bool
        若输入是 [T, H, W]，是否把 T 当作通道维，输出形状 [C=T, H, W]。
        这对把 time 当 channel 的 operator 比较方便。

    transform: Optional[Callable]
        对 sample 整体的 transform，签名：
          transform({"input": x, "output": y, "meta": meta}) -> 同结构的 dict

    input_transform: Optional[Callable]
        对输入 x 的 transform：input_transform(x: Tensor) -> Tensor

    output_transform: Optional[Callable]
        对输出 y 的 transform：output_transform(y: Tensor) -> Tensor
    """

    def __init__(
        self,
        task: str,
        root: str,
        split: str = "train",
        file_name: Optional[str] = None,
        input_key: str = "input",
        target_key: str = "output",
        target_time_index: Optional[int] = -1,
        time_as_channel: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        output_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        self.task = task.lower()
        self.root = Path(root)
        self.split = split.lower()
        self.input_key = input_key
        self.target_key = target_key
        self.target_time_index = target_time_index
        self.time_as_channel = time_as_channel

        self.transform = transform
        self.input_transform = input_transform
        self.output_transform = output_transform

        # 结合任务名应用合理的默认值（除非用户显式传入）
        self._apply_task_defaults()

        # ---- 解析 HDF5 文件路径 ----
        self.file_path = self._resolve_file_path(file_name)

        if not self.file_path.exists():
            raise FileNotFoundError(f"PDEBench HDF5 文件不存在: {self.file_path}")

        # ---- 读取一次 meta 信息（长度 & 形状），之后关闭文件 ----
        with h5py.File(self.file_path, "r") as f:
            if self.input_key not in f or self.target_key not in f:
                raise KeyError(
                    f"HDF5 文件 {self.file_path} 中未找到数据集 "
                    f"'{self.input_key}' 或 '{self.target_key}'，"
                    f"可通过 PDEBenchDataset(..., input_key=..., target_key=...) 显式指定。"
                )

            input_ds = f[self.input_key]
            target_ds = f[self.target_key]

            self._len = int(input_ds.shape[0])
            if int(target_ds.shape[0]) != self._len:
                raise ValueError(
                    f"输入和输出样本数不一致: input.shape[0]={input_ds.shape[0]}, "
                    f"target.shape[0]={target_ds.shape[0]}"
                )

            # 记录原始形状（除 batch 维外）
            self.input_shape_raw = tuple(input_ds.shape[1:])
            self.target_shape_raw = tuple(target_ds.shape[1:])

            # 读取可能存在的附加数据（例如 plasticity 的 pos/time），小尺寸一次性缓存
            self.extra_fields: Dict[str, torch.Tensor] = {}
            for extra_key in ("pos", "time"):
                if extra_key in f:
                    extra_arr = np.asarray(f[extra_key])
                    self.extra_fields[extra_key] = torch.from_numpy(extra_arr).float()

        # 供外部调试查看
        print(
            f"[PDEBenchDataset] task={self.task}, split={self.split}, file={self.file_path.name}, "
            f"N={self._len}, input_shape_raw={self.input_shape_raw}, target_shape_raw={self.target_shape_raw}"
        )

    # -----------------------------
    #   任务默认参数
    # -----------------------------
    def _apply_task_defaults(self) -> None:
        """
        针对已知任务（LaMO/PDEBench 预处理为 HDF5 后的形状），
        调整合理的默认 time/target 处理方式：
          - navier*: input/output 为 [T, H, W]，保持全序列 -> target_time_index=None
          - pipe/airfoil/darcy/plasticity: 首维为通道而非时间 -> target_time_index=None
          - 其它任务保持原始通道，不做时间裁剪。
        用户若显式传入参数，则尊重传入值。
        """
        name = self.task.lower()
        navier_alias = {"navier", "navier2d", "navier-stokes", "navier_stokes", "ns", "ns2d"}
        channel_only = {"pipe", "airfoil", "darcy", "darcy2d", "plasticity", "plas"}

        if name in navier_alias:
            if self.target_time_index == -1:
                self.target_time_index = None
            # navier 期望时间维作通道
            self.time_as_channel = True
        elif name in channel_only:
            # 首维是通道，不是时间
            if self.target_time_index == -1:
                self.target_time_index = None
            self.time_as_channel = True

    # -----------------------------
    #   文件路径解析
    # -----------------------------
    def _resolve_file_path(self, file_name: Optional[str]) -> Path:
        if file_name is not None:
            return (self.root / file_name).resolve()

        # 自动猜测：优先 task_split.h5 / .hdf5
        candidates = [
            self.root / self.task / f"{self.task}_{self.split}.h5",
            self.root / self.task / f"{self.task}_{self.split}.hdf5",
            self.root / f"{self.task}_{self.split}.h5",
            self.root / f"{self.task}_{self.split}.hdf5",
            self.root / self.task / f"{self.task}.h5",
            self.root / self.task / f"{self.task}.hdf5",
            self.root / f"{self.task}.h5",
            self.root / f"{self.task}.hdf5",
        ]

        for p in candidates:
            if p.exists():
                return p.resolve()

        # 如果都没找到，就返回第一个候选，后面会报 FileNotFoundError
        return candidates[0].resolve()

    # -----------------------------
    #   长度
    # -----------------------------
    def __len__(self) -> int:
        return self._len

    # -----------------------------
    #   __getitem__
    # -----------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 为了 DataLoader 多进程安全，每次重新打开文件
        with h5py.File(self.file_path, "r") as f:
            input_arr = f[self.input_key][idx]   # np.ndarray
            target_arr = f[self.target_key][idx]

        # ---- 转成 [C,H,W] 形式的 torch.Tensor ----
        x = self._to_chw_tensor(input_arr, is_input=True)
        y = self._to_chw_tensor(target_arr, is_input=False)

        # ---- 单独对 input/output 做 transform（若提供）----
        if self.input_transform is not None:
            x = self.input_transform(x)
        if self.output_transform is not None:
            y = self.output_transform(y)

        sample: Dict[str, Any] = {
            "input": x,
            "output": y,
            "meta": {
                "idx": int(idx),
                "task": self.task,
                "split": self.split,
                "input_shape_raw": self.input_shape_raw,
                "target_shape_raw": self.target_shape_raw,
            },
        }
        if getattr(self, "extra_fields", None):
            sample["meta"]["extras"] = self.extra_fields

        # ---- 对整体 sample 做 transform（若提供）----
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # -----------------------------
    #   转为 [C,H,W] 的小工具
    # -----------------------------
    def _to_chw_tensor(self, arr: np.ndarray, is_input: bool) -> torch.Tensor:
        """
        将 numpy 数组转换为 [C,H,W] 形式的 tensor，兼容 1D/2D/时序。
        """
        arr = np.asarray(arr)

        # 0 维（极端情况）
        if arr.ndim == 0:
            # 标量 -> [1, 1, 1]
            arr = arr.reshape(1, 1, 1)

        # 1 维：假设为 1D 空间 [X]
        if arr.ndim == 1:
            # -> [C=1, H=1, W=X]
            c = 1
            h = 1
            w = arr.shape[0]
            arr = arr.reshape(c, h, w)
            return torch.from_numpy(arr).float()

        # 2 维：假设为 [H, W] 或 [T, X]
        if arr.ndim == 2:
            # 一律视作 [H, W]，-> [C=1, H, W]
            h, w = arr.shape
            arr = arr.reshape(1, h, w)
            return torch.from_numpy(arr).float()

        # 3 维：可能是 [T, H, W] 或 [C, H, W]
        if arr.ndim == 3:
            # 若是 target 且指定了 target_time_index，则先在 time 维取一帧
            if (not is_input) and (self.target_time_index is not None) and arr.shape[0] > 1:
                t_idx = self.target_time_index
                # 假定 time 在第 0 维：arr [T, H, W]
                if abs(t_idx) >= arr.shape[0]:
                    raise IndexError(
                        f"target_time_index={t_idx} 越界，"
                        f"target.shape[0]={arr.shape[0]}"
                    )
                arr = arr[t_idx]  # [H, W] -> 回到 2D 情形
                h, w = arr.shape
                arr = arr.reshape(1, h, w)
                return torch.from_numpy(arr).float()

            # 否则保留整个 time 维
            if self.time_as_channel:
                # 将第 0 维视作通道维：arr [T,H,W] -> [C=T,H,W]
                return torch.from_numpy(arr).float()
            else:
                # 如果不把 time 当 channel，可以在这里选择：
                # - 把 T 作为 W 维折叠，例如 [T,H,W] -> [1, H, T*W]
                # 这里给一个简单折叠示例：
                T, H, W = arr.shape
                arr = arr.reshape(1, H, T * W)
                return torch.from_numpy(arr).float()

        # 更高维：一般是 [C, T, H, W] 之类，此处不做复杂假设，交给用户自己 reshape
        # 这里只给一个安全兜底：把所有非 batch 维拉平到 W
        if arr.ndim > 3:
            c = 1
            h = 1
            w = int(np.prod(arr.shape))
            arr = arr.reshape(c, h, w)
            return torch.from_numpy(arr).float()

        # 理论上不会走到这里
        raise ValueError(f"不支持的数组维度: arr.ndim={arr.ndim}, shape={arr.shape}")

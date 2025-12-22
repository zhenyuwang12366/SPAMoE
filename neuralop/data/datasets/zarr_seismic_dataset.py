import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from zarr.storage import DirectoryStore


class ZarrSeismicDataset(Dataset):
    """
    线程安全、进程安全的 Zarr 读取版
    - 每个 worker 进程独立只读打开 zarr
    - 防止句柄在 fork 时共享
    """
    def __init__(
        self,
        zarr_path: str,
        split: str = "train",
        input_transform=None,
        output_transform=None,
        expect_input_shape=(1, 1000, 350),
        to_float32: bool = True,
    ):
        assert split in {"train", "val", "test"}

        # 只存路径，不在 __init__ 打开
        self.zarr_path = zarr_path
        self.split = split
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.expect_input_shape = tuple(expect_input_shape)
        self.to_float32 = bool(to_float32)
        
        # 惰性打开时缓存这些
        self._root = None
        self._inputs = None
        self._outputs = None
        self._labels = None
        self._type_name = None
        self._input_file = None
        self._split_group = None
        self._idx = None

    # ---- 解决 DataLoader 多进程 pickle 问题 ----
    def __getstate__(self):
        state = self.__dict__.copy()
        # 清除已打开的对象引用
        for key in ["_root", "_inputs", "_outputs", "_labels", "_type_name", "_input_file", "_split_group"]:
            state[key] = None
        return state

    # ---- 延迟打开 Zarr ----
    def _ensure_open(self):
        if self._root is not None:
            return

        # 每个 worker 独立只读打开 Zarr
        store = DirectoryStore(self.zarr_path)
        self._root = zarr.open_group(store=store, mode="r")

        self._inputs = self._root["inputs"]
        self._outputs = self._root.get("outputs", None)
        self._labels = self._root.get("labels", None)
        self._type_name = self._root.get("type_name", None)
        self._input_file = self._root.get("input_file", None)
        self._split_group = self._root["splits"]

        split_key = f"{self.split}_idx"
        if split_key not in self._split_group:
            raise KeyError(f"splits/{split_key} not found in {self.zarr_path}")

        self._idx = np.asarray(self._split_group[split_key][:], dtype=np.int64)

        # train/val 有监督
        self.has_output = (self._outputs is not None) and (self._labels is not None) and (self.split != "test")

    def __len__(self):
        self._ensure_open()
        return int(self._idx.shape[0])

    def __getitem__(self, i: int):
        self._ensure_open()
        j = int(self._idx[i])

        # ---- 输入 ----
        x = self._inputs[j]
        if x.shape != self.expect_input_shape:
            raise ValueError(f"inputs[{j}] shape={x.shape}, expect={self.expect_input_shape}")
        if self.to_float32:
            x = x.astype(np.float32, copy=False)
        x_t = torch.from_numpy(x)
        if self.input_transform is not None:
            x_t = self.input_transform(x_t)

        sample = {
            "input": x_t,
            "v_type": torch.tensor(int(self._labels[j])) if self._labels is not None else torch.tensor(-1),
            "type_name": (str(self._type_name[j]) if self._type_name is not None else ""),
            "input_file": (str(self._input_file[j]) if self._input_file is not None else ""),
        }

        # ---- 输出 ----
        if self.has_output:
            y = self._outputs[j]
            if y.ndim == 3 and y.shape[0] == 1:
                y = y[0]
            if y.shape != (70, 70):
                raise ValueError(f"outputs[{j}] shape={y.shape}, expect=(70,70)")
            if self.to_float32:
                y = y.astype(np.float32, copy=False)
            y_t = torch.from_numpy(y)
            if self.output_transform is not None:
                y_t = self.output_transform(y_t)
            sample["output"] = y_t

        return sample
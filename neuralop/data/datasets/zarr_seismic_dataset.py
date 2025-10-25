import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

class ZarrSeismicDataset(Dataset):
    """
    直接从 seismic_moe.zarr 读取：
      - split ∈ {'train','val','test'} -> 使用 splits/{split}_idx 作为采样索引
      - 返回：
          train/val: {'input', 'output', 'v_type', 'type_name', 'input_file'}
          test    : {'input',           'v_type', 'type_name', 'input_file'}  （若 labels 存在则提供）
      - 支持 input_transform / output_transform（均在 torch.Tensor 上执行）
    """
    def __init__(
        self,
        zarr_path: str,
        split: str = "train",
        input_transform=None,
        output_transform=None,
        expect_input_shape=(5, 1000, 70),
        to_float32: bool = True,
    ):
        assert split in {"train", "val", "test"}
        self.root = zarr.open_group(zarr_path, mode="r")
        self.inputs      = self.root["inputs"]                    # [N, 5, 1000, 70]
        self.outputs     = self.root.get("outputs", None)         # [N, 70, 70]（test 可能没有）
        self.labels      = self.root.get("labels", None)          # [N]
        self.type_name   = self.root.get("type_name", None)       # [N] vlen-utf8
        self.input_file  = self.root.get("input_file", None)      # [N] vlen-utf8
        self.split_group = self.root["splits"]

        split_key = f"{split}_idx"
        if split_key not in self.split_group:
            raise KeyError(f"splits/{split_key} not found in {zarr_path}")
        self.idx = np.asarray(self.split_group[split_key][:], dtype=np.int64)

        self.split = split
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.expect_input_shape = tuple(expect_input_shape)
        self.to_float32 = bool(to_float32)

        # 该 split 是否有监督信号（train/val 多为 True，test 可能 False）
        self.has_output = (self.outputs is not None) and (self.labels is not None) and (split != "test")

    def __len__(self):
        return int(self.idx.shape[0])

    def __getitem__(self, i: int):
        j = int(self.idx[i])

        # ---- 输入 ----
        x = self.inputs[j]
        if x.shape != self.expect_input_shape:
            raise ValueError(f"inputs[{j}] shape={x.shape}, expect={self.expect_input_shape}")
        if self.to_float32:
            x = x.astype(np.float32, copy=False)
        x_t = torch.from_numpy(x)
        if self.input_transform is not None:
            x_t = self.input_transform(x_t)

        sample = {
            "input": x_t,
            "v_type": torch.tensor(int(self.labels[j])) if self.labels is not None else torch.tensor(-1),
            "type_name": (str(self.type_name[j]) if self.type_name is not None else ""),
            "input_file": (str(self.input_file[j]) if self.input_file is not None else ""),
        }

        # ---- 输出（train/val 必须）----
        if self.has_output:
            y = self.outputs[j]
            if y.ndim == 3 and y.shape[0] == 1:
                y = y[0]          # [70, 70]
            if y.shape != (70, 70):
                raise ValueError(f"outputs[{j}] shape={y.shape}, expect=(70,70)")
            if self.to_float32:
                y = y.astype(np.float32, copy=False)
            y_t = torch.from_numpy(y)
            if self.output_transform is not None:
                y_t = self.output_transform(y_t)
            sample["output"] = y_t

        return sample
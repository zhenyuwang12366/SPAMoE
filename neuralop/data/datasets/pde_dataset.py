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
    Generic PDEBench Dataset wrapper.

    Each task is expected to map to one or more HDF5 files with roughly:
      - f[input_key]  : [N, ...] input field
      - f[target_key] : [N, ...] target field

    Conventions for Burgers / Navier-Stokes / Darcy, etc.:
      1) Regardless of original 1D/2D/time layout, data end up as [C, H, W] tensors:
         - Data [H] or [X]           -> [1, 1, X]
         - Data [H, W]               -> [1, H, W]
         - Data [T, H]               -> [T, 1, H]      (optional when time_as_channel=False)
         - Data [T, H, W] with time_as_channel=True -> [T, H, W]
         - Data [T, H, W] with time_as_channel=False -> last dim treated as time by default; use target_time_index
      2) If target is time-series [T, ...], default target_time_index=-1 (last time step).
      3) Transform precedence:
         - transform(sample: Dict[str, Tensor]) -> Dict[str, Tensor]   (highest)
         - input_transform(x: Tensor) / output_transform(y: Tensor)   (lower)

    Parameters
    ----------
    task: str
        Task name, e.g. 'burgers1d', 'navier2d', 'darcy2d'.
        Used mainly to infer filenames and metadata.

    root: str or Path
        Data root, e.g. "./pdebench_data".

    split: str
        Split: 'train' / 'val' / 'test'. If there is no 'val',
        split 'train' externally with Subset.

    file_name: Optional[str]
        Explicit HDF5 filename; if None, common patterns are tried:
          - {task}_{split}.h5
          - {task}_{split}.hdf5
          - {task}.h5
          - {task}.hdf5

    input_key: str
        HDF5 dataset name for inputs, default 'input'.

    target_key: str
        HDF5 dataset name for targets, default 'output'.

    target_time_index: Optional[int]
        If target is time-shaped (leading dim is time), which index to supervise.
        Default -1 is the last frame; None keeps the full series and applies time_as_channel rules.

    time_as_channel: bool
        If input is [T, H, W], whether T is treated as channels -> [C=T, H, W].
        Convenient for operators that treat time as channels.

    transform: Optional[Callable]
        Whole-sample transform:
          transform({"input": x, "output": y, "meta": meta}) -> same dict structure

    input_transform: Optional[Callable]
        Per-input transform: input_transform(x: Tensor) -> Tensor

    output_transform: Optional[Callable]
        Per-output transform: output_transform(y: Tensor) -> Tensor
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

        # Sensible defaults per task (unless user set them explicitly)
        self._apply_task_defaults()

        # ---- Resolve HDF5 path ----
        self.file_path = self._resolve_file_path(file_name)

        if not self.file_path.exists():
            raise FileNotFoundError(f"PDEBench HDF5 file not found: {self.file_path}")

        # ---- Read meta (length & shapes) once, then close ----
        with h5py.File(self.file_path, "r") as f:
            if self.input_key not in f or self.target_key not in f:
                raise KeyError(
                    f"HDF5 file {self.file_path} has no datasets "
                    f"'{self.input_key}' or '{self.target_key}'; "
                    f"pass PDEBenchDataset(..., input_key=..., target_key=...) explicitly."
                )

            input_ds = f[self.input_key]
            target_ds = f[self.target_key]

            self._len = int(input_ds.shape[0])
            if int(target_ds.shape[0]) != self._len:
                raise ValueError(
                    f"Input and target sample counts differ: input.shape[0]={input_ds.shape[0]}, "
                    f"target.shape[0]={target_ds.shape[0]}"
                )

            # Raw shapes (excluding batch dim)
            self.input_shape_raw = tuple(input_ds.shape[1:])
            self.target_shape_raw = tuple(target_ds.shape[1:])

            # Optional small aux arrays (e.g. plasticity pos/time), cache once
            self.extra_fields: Dict[str, torch.Tensor] = {}
            for extra_key in ("pos", "time"):
                if extra_key in f:
                    extra_arr = np.asarray(f[extra_key])
                    self.extra_fields[extra_key] = torch.from_numpy(extra_arr).float()

        # Debug print for external inspection
        print(
            f"[PDEBenchDataset] task={self.task}, split={self.split}, file={self.file_path.name}, "
            f"N={self._len}, input_shape_raw={self.input_shape_raw}, target_shape_raw={self.target_shape_raw}"
        )

    # -----------------------------
    #   Task-specific defaults
    # -----------------------------
    def _apply_task_defaults(self) -> None:
        """
        For known tasks (LaMO/PDEBench HDF layouts), adjust time/target defaults:
          - navier*: input/output [T, H, W], full series -> target_time_index=None
          - pipe/airfoil/darcy/plasticity: leading dim is channel, not time -> target_time_index=None
          - Other tasks: keep channels as-is, no time cropping unless user overrides.
        """
        name = self.task.lower()
        navier_alias = {"navier", "navier2d", "navier-stokes", "navier_stokes", "ns", "ns2d"}
        channel_only = {"pipe", "airfoil", "darcy", "darcy2d", "plasticity", "plas"}

        if name in navier_alias:
            if self.target_time_index == -1:
                self.target_time_index = None
            # Navier: treat time as channels
            self.time_as_channel = True
        elif name in channel_only:
            # Leading dim is channel, not time
            if self.target_time_index == -1:
                self.target_time_index = None
            self.time_as_channel = True

    # -----------------------------
    #   Path resolution
    # -----------------------------
    def _resolve_file_path(self, file_name: Optional[str]) -> Path:
        if file_name is not None:
            return (self.root / file_name).resolve()

        # Auto-guess: prefer task_split.h5 / .hdf5
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

        # If none exist, return first candidate (caller gets FileNotFoundError)
        return candidates[0].resolve()

    # -----------------------------
    #   Length
    # -----------------------------
    def __len__(self) -> int:
        return self._len

    # -----------------------------
    #   __getitem__
    # -----------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Re-open each time for multi-process DataLoader safety
        with h5py.File(self.file_path, "r") as f:
            input_arr = f[self.input_key][idx]   # np.ndarray
            target_arr = f[self.target_key][idx]

        # ---- To [C,H,W] torch.Tensor ----
        x = self._to_chw_tensor(input_arr, is_input=True)
        y = self._to_chw_tensor(target_arr, is_input=False)

        # ---- Per-field transforms ----
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

        # ---- Whole-sample transform ----
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # -----------------------------
    #   [C,H,W] conversion helper
    # -----------------------------
    def _to_chw_tensor(self, arr: np.ndarray, is_input: bool) -> torch.Tensor:
        """
        Convert numpy array to [C,H,W] tensor for 1D/2D/time layouts.
        """
        arr = np.asarray(arr)

        # 0-D (edge case)
        if arr.ndim == 0:
            # scalar -> [1, 1, 1]
            arr = arr.reshape(1, 1, 1)

        # 1-D: 1D spatial [X]
        if arr.ndim == 1:
            # -> [C=1, H=1, W=X]
            c = 1
            h = 1
            w = arr.shape[0]
            arr = arr.reshape(c, h, w)
            return torch.from_numpy(arr).float()

        # 2-D: [H, W] or [T, X]
        if arr.ndim == 2:
            # Treat as [H, W] -> [C=1, H, W]
            h, w = arr.shape
            arr = arr.reshape(1, h, w)
            return torch.from_numpy(arr).float()

        # 3-D: [T, H, W] or [C, H, W]
        if arr.ndim == 3:
            # For target with target_time_index, slice time first
            if (not is_input) and (self.target_time_index is not None) and arr.shape[0] > 1:
                t_idx = self.target_time_index
                # Assume time on axis 0: arr [T, H, W]
                if abs(t_idx) >= arr.shape[0]:
                    raise IndexError(
                        f"target_time_index={t_idx} out of range, "
                        f"target.shape[0]={arr.shape[0]}"
                    )
                arr = arr[t_idx]  # [H, W] -> back to 2D
                h, w = arr.shape
                arr = arr.reshape(1, h, w)
                return torch.from_numpy(arr).float()

            # Else keep full time dimension
            if self.time_as_channel:
                # Axis 0 as channels: [T,H,W] -> [C=T,H,W]
                return torch.from_numpy(arr).float()
            else:
                # If time is not channel, fold T into W, e.g. [T,H,W] -> [1, H, T*W]
                # Simple example:
                T, H, W = arr.shape
                arr = arr.reshape(1, H, T * W)
                return torch.from_numpy(arr).float()

        # Higher rank: e.g. [C, T, H, W]; no strong assumption — flatten non-batch to W
        if arr.ndim > 3:
            c = 1
            h = 1
            w = int(np.prod(arr.shape))
            arr = arr.reshape(c, h, w)
            return torch.from_numpy(arr).float()

        raise ValueError(f"Unsupported array rank: arr.ndim={arr.ndim}, shape={arr.shape}")

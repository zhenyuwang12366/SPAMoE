#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 LaMO 的六个 PDE 任务数据转换为 HDF5 文件，方便在 FWINO/EMO 里直接使用。

支持的任务：
  - pipe        : Pipe_X / Pipe_Y / Pipe_Q.npy -> input[2,H,W], output[1,H,W]
  - airfoil     : NACA_Cylinder_X/Y/Q.npy      -> input[2,221,51], output[1,221,51]
  - darcy       : piececonst_r421_*.mat        -> input/output[1,s,s]
  - navier      : NavierStokes_V1e-5_N1200_T20.mat -> input[T_in,h,h], output[T_out,h,h]
  - plasticity  : plas_N987_T20.mat -> input[1,H,W], output[4*T,H,W]（时间展平到通道），额外写 pos、time

注意：
  - plasticity 改为规则网格 BCHW（时间展平成通道）。
  - 本脚本生成的 HDF5 至少包含 input/output 两个 dataset，并额外写出 *_stats.json 供归一化使用。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import scipy.io as scio


def _save_h5_with_stats(
    dst: Path,
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    extra: Dict[str, np.ndarray] | None = None,
    stats: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """
    写入 h5 并保存 min/max 统计到 attrs 和旁边的 json。
    若传入 stats 则复用同一组（用于全量统计），否则按当前数组计算。
    返回实际写入的 stats。
    """
    if stats is None:
        stats = {
            "input_min": float(np.min(input_arr)),
            "input_max": float(np.max(input_arr)),
            "output_min": float(np.min(output_arr)),
            "output_max": float(np.max(output_arr)),
        }

    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dst, "w") as f:
        f.create_dataset("input", data=input_arr.astype(np.float32))
        f.create_dataset("output", data=output_arr.astype(np.float32))
        if extra:
            for k, v in extra.items():
                f.create_dataset(k, data=v.astype(np.float32))
        for k, v in stats.items():
            f.attrs[k] = v

    stats_path = dst.with_name(dst.stem + "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[write] {dst} (input {input_arr.shape}, output {output_arr.shape})")
    print(f"[stats] {stats_path}")
    return stats


def _split_indices(N: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42):
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test 比例之和必须为 1，目前为 {total:.4f}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    n_test = N - n_train - n_val
    if n_train <= 0 or n_test <= 0:
        raise ValueError(f"样本数 N={N} 太小或比例设置不合理: train={n_train}, val={n_val}, test={n_test}")
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]
    return idx_train, idx_val, idx_test


def _write_split_files(
    base_path: Path,
    input_arr: np.ndarray,
    output_arr: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    pos: np.ndarray | None = None,
    time: np.ndarray | None = None,
    seed: int = 42,
    stats_full: Dict[str, float] | None = None,
):
    """根据比例划分并写出 train/val/test 三个文件，附带 pos/time（若提供），stats 复用全量统计。"""
    idx_train, idx_val, idx_test = _split_indices(input_arr.shape[0], train_ratio, val_ratio, test_ratio, seed)
    splits = {
        "train": idx_train,
        "val": idx_val,
        "test": idx_test,
    }
    for split, idx in splits.items():
        dst = base_path.with_name(base_path.stem + f"_{split}").with_suffix(base_path.suffix)
        in_s = input_arr[idx]
        out_s = output_arr[idx]
        extra = {}
        if pos is not None:
            extra["pos"] = pos
        if time is not None:
            extra["time"] = time
        _save_h5_with_stats(dst, in_s, out_s, extra=extra, stats=stats_full)


def _downsample_hw(arr: np.ndarray, r1: int, r2: int) -> np.ndarray:
    """按 (r1,r2) 下采样 H、W 维度。"""
    return arr[..., ::r1, ::r2]


def convert_pipe(data_root: Path, downsamplex: int, downsampley: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.load(data_root / "Pipe_X.npy")  # [N,129,129]
    y = np.load(data_root / "Pipe_Y.npy")  # [N,129,129]
    q = np.load(data_root / "Pipe_Q.npy")[:, 0]  # [N,129,129]

    if downsamplex != 1 or downsampley != 1:
        x = _downsample_hw(x, downsamplex, downsampley)
        y = _downsample_hw(y, downsamplex, downsampley)
        q = _downsample_hw(q, downsamplex, downsampley)

    input_arr = np.stack([x, y], axis=1)          # [N,2,H,W]
    output_arr = q[:, None, :, :]                 # [N,1,H,W]
    return input_arr, output_arr


def convert_airfoil(data_root: Path, downsamplex: int, downsampley: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.load(data_root / "NACA_Cylinder_X.npy")  # [N,221,51]
    y = np.load(data_root / "NACA_Cylinder_Y.npy")
    q = np.load(data_root / "NACA_Cylinder_Q.npy")[:, 4]  # 取第 5 个通道

    if downsamplex != 1 or downsampley != 1:
        x = _downsample_hw(x, downsamplex, downsampley)
        y = _downsample_hw(y, downsamplex, downsampley)
        q = _downsample_hw(q, downsamplex, downsampley)

    input_arr = np.stack([x, y], axis=1)          # [N,2,H,W]
    output_arr = q[:, None, :, :]                 # [N,1,H,W]
    return input_arr, output_arr


def convert_darcy(data_root: Path, downsample: int) -> Tuple[np.ndarray, np.ndarray]:
    train_path = data_root / "piececonst_r421_N1024_smooth1.mat"
    test_path = data_root / "piececonst_r421_N1024_smooth2.mat"
    train = scio.loadmat(train_path)
    test = scio.loadmat(test_path)

    coeff = np.concatenate([train["coeff"], test["coeff"]], axis=0)  # [N,421,421]
    sol = np.concatenate([train["sol"], test["sol"]], axis=0)

    if downsample != 1:
        coeff = _downsample_hw(coeff, downsample, downsample)
        sol = _downsample_hw(sol, downsample, downsample)

    input_arr = coeff[:, None, :, :]             # [N,1,H,W]
    output_arr = sol[:, None, :, :]              # [N,1,H,W]
    return input_arr, output_arr


def convert_navier(data_root: Path, downsample: int, t_in: int, t_out: int) -> Tuple[np.ndarray, np.ndarray]:
    mat_path = data_root / "NavierStokes_V1e-5_N1200_T20.mat"
    data = scio.loadmat(mat_path)
    u = data["u"]  # [N,64,64,20]

    h = int(((u.shape[1] - 1) / downsample) + 1)
    u = u[:, ::downsample, ::downsample, :][:, :h, :h, :]  # 截断到整数网格

    input_arr = u[:, :, :, :t_in].transpose(0, 3, 1, 2)   # [N,T_in,H,W]
    output_arr = u[:, :, :, t_in : t_in + t_out].transpose(0, 3, 1, 2)  # [N,T_out,H,W]
    return input_arr, output_arr


def convert_plasticity(data_root: Path, downsamplex: int, downsampley: int):
    """
    塑性问题：一维输入沿 y 方向复制，输出含 4 个 deformation 通道和 T=20 时间步。
    这里将输出的时间维展平到通道，得到 output [N, 4*T, H, W]，便于直接用 BCHW。
    额外返回 pos（规则网格）和 time（原始时间序列，供可视化/逆归一化）。
    """
    data = scio.loadmat(data_root / "plas_N987_T20.mat")
    inp_raw = data["input"]          # 期望 [N, 101]
    out_raw = data["output"]         # 期望 [N, 101, 31, T, 4] (LaMO 中通过 transpose(-2,-1) 得到 [N,101,31,4,T])

    s1 = int(((101 - 1) / downsamplex) + 1)
    s2 = int(((31 - 1) / downsampley) + 1)
    T = out_raw.shape[-2] if out_raw.ndim >= 5 else 20

    # 输入：沿 y 方向复制，再裁剪
    inp = inp_raw[:, ::downsamplex][:, :s1]          # [N, s1]
    inp = np.repeat(inp[:, :, None], s2, axis=2)     # [N, s1, s2]
    input_arr = inp[:, None, :, :]                   # [N,1,H,W]

    # 输出：reorder 成 [N, s1, s2, deform, T]
    if out_raw.shape[-1] == 4:  # 如果最后一维是 deformation
        out_reordered = np.transpose(out_raw, (0, 1, 2, 4, 3))
    else:
        out_reordered = out_raw
    out_reordered = out_reordered[:, ::downsamplex, ::downsampley, :, :][:, :s1, :s2, :, :]  # [N,H,W,4,T]
    out_chw = np.transpose(out_reordered, (0, 3, 4, 1, 2))  # [N,4,T,H,W]
    n, c, t, h, w = out_chw.shape
    output_arr = out_chw.reshape(n, c * t, h, w)  # [N, 4*T, H, W]

    # 网格坐标（与 LaMO 一致）：均匀网格 0~1
    x = np.linspace(0, 1, s1)
    y = np.linspace(0, 1, s2)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    pos_grid = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # [N_pts,2]
    time = np.linspace(0, 1, T) if T > 0 else np.array([])
    return input_arr, output_arr, pos_grid, time


def main():
    parser = argparse.ArgumentParser("Convert LaMO datasets to BCHW HDF5")
    parser.add_argument("--task", type=str, required=True,
                        choices=["pipe", "airfoil", "darcy", "navier", "plasticity"],
                        help="选择要转换的任务")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="任务数据所在目录，例如 LaMO/data/Pipe")
    parser.add_argument("--output", type=Path, required=True,
                        help="输出 h5 路径，例如 ./pdebench_data/pipe")

    # pipe / airfoil
    parser.add_argument(
        "--downsamplex",
        type=int,
        default=None,
        help="X 方向下采样 (pipe/airfoil/plasticity)，默认按任务设定（pipe/airfoil/plasticity=1）",
    )
    parser.add_argument(
        "--downsampley",
        type=int,
        default=None,
        help="Y 方向下采样 (pipe/airfoil/plasticity)，默认按任务设定（pipe/airfoil/plasticity=1）",
    )

    # darcy
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="下采样因子（darcy 默认 5，navier 默认 1）",
    )

    # navier
    parser.add_argument(
        "--t-in",
        type=int,
        default=None,
        help="Navier-Stokes 历史步数（默认 10）",
    )
    parser.add_argument(
        "--t-out",
        type=int,
        default=None,
        help="Navier-Stokes 预测步数（默认 10）",
    )
    # split
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")

    args = parser.parse_args()

    pos_arr = None
    time_arr = None

    task = args.task.lower()
    if task == "pipe":
        dsx = 1 if args.downsamplex is None else args.downsamplex
        dsy = 1 if args.downsampley is None else args.downsampley
        input_arr, output_arr = convert_pipe(args.data_root, dsx, dsy)
    elif task == "airfoil":
        dsx = 1 if args.downsamplex is None else args.downsamplex
        dsy = 1 if args.downsampley is None else args.downsampley
        input_arr, output_arr = convert_airfoil(args.data_root, dsx, dsy)
    elif task == "darcy":
        ds = 5 if args.downsample is None else args.downsample
        input_arr, output_arr = convert_darcy(args.data_root, ds)
    elif task == "navier":
        ds = 1 if args.downsample is None else args.downsample
        t_in = 10 if args.t_in is None else args.t_in
        t_out = 10 if args.t_out is None else args.t_out
        input_arr, output_arr = convert_navier(args.data_root, ds, t_in, t_out)
    elif task == "plasticity":
        dsx = 1 if args.downsamplex is None else args.downsamplex
        dsy = 1 if args.downsampley is None else args.downsampley
        input_arr, output_arr, pos_arr, time_arr = convert_plasticity(args.data_root, dsx, dsy)
    else:
        raise ValueError(f"Unsupported task: {task}")

    # 仅计算全量统计（不写全量文件），将其复用到各 split
    stats_full = {
        "input_min": float(np.min(input_arr)),
        "input_max": float(np.max(input_arr)),
        "output_min": float(np.min(output_arr)),
        "output_max": float(np.max(output_arr)),
    }

    # 按比例划分并写 train/val/test，plasticity 额外带 pos/time，统计来源于全量
    _write_split_files(
        args.output,
        input_arr,
        output_arr,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pos=pos_arr if task == "plasticity" else None,
        time=time_arr if task == "plasticity" else None,
        seed=args.seed,
        stats_full=stats_full,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 PDEBench 官方 HDF5 数据转换为 PDEBenchDataset 可直接使用的格式：
- 生成的新文件中只包含两个 dataset：'input' 和 'output'
- 每个样本的形状最多 3 维，以便 PDEBenchDataset._to_chw_tensor 正常工作
- 额外计算数据统计量：input_min, input_max, output_min, output_max
  并：
    1) 打印到终端
    2) 写入目标 HDF5 文件的 attrs
    3) 写入旁边的 *_stats.json 文件，供归一化使用

支持三类典型任务：
1) burgers1d :  [N, T, X] 或 [N, T, X, 1]
   - input  = u(t_in,  x)
   - output = u(t_out, x)

2) darcy2d   :  [N, H, W] 或 [N, H, W, C]
   - input  = 某个通道（通常是系数场，如渗透率）
   - output = 某个通道（通常是解场，如压强）

3) ns2d / swe2d : [N, T, H, W, C] 或 [N, T, H, W]
   - input  = 全时序 u(t, h, w)，shape [N, T, H, W]
   - output = 同样 [N, T, H, W]
   - 之后在 PDEBenchDataset 中通过 target_time_index=-1 取末帧监督

新增：
- burgers1d_split / darcy2d_split / ts2d_split 三个子命令
  自动从一个大文件中随机划分 train/val/test 三个文件。
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


# --------------------------
#   通用小工具
# --------------------------
def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _open_and_inspect(src_path: Path):
    """简单打印文件中有哪些 dataset 和 shape，方便你确认 key。"""
    print(f"[inspect] 打开文件: {src_path}")
    with h5py.File(src_path, "r") as f:
        def _print_group(g, prefix=""):
            for k, v in g.items():
                if isinstance(v, h5py.Dataset):
                    print(f"  {prefix}{k} : shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, h5py.Group):
                    print(f"  {prefix}{k}/ (group)")
                    _print_group(v, prefix + "  ")

        _print_group(f)


def _compute_stats(input_arr: np.ndarray, output_arr: np.ndarray):
    """
    计算全局 min/max 统计量，返回一个 dict：
    {
        'input_min': float,
        'input_max': float,
        'output_min': float,
        'output_max': float,
    }
    """
    stats = {
        "input_min": float(np.min(input_arr)),
        "input_max": float(np.max(input_arr)),
        "output_min": float(np.min(output_arr)),
        "output_max": float(np.max(output_arr)),
    }
    print("[stats] 统计信息：")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")
    return stats


def _save_stats_json(dst_path: Path, stats: dict):
    """
    将统计信息写到 JSON 文件：
        <dst_path.stem>_stats.json
    例如:
        burgers1d_train.h5 -> burgers1d_train_stats.json
    """
    json_path = dst_path.with_name(dst_path.stem + "_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[stats] 已写入 JSON: {json_path}")


def _write_h5_with_stats(dst_path: Path, input_arr: np.ndarray, output_arr: np.ndarray):
    """
    创建 HDF5 文件，写入 input/output 两个 dataset，
    并计算+写入统计信息到 attrs 和 JSON。
    """
    # 先计算统计信息
    stats = _compute_stats(input_arr, output_arr)

    # 写 HDF5
    _ensure_dir(dst_path)
    with h5py.File(dst_path, "w") as f_dst:
        d_in = f_dst.create_dataset("input", data=input_arr.astype(np.float32))
        d_out = f_dst.create_dataset("output", data=output_arr.astype(np.float32))

        # 在 root attrs 写统计量
        f_dst.attrs["input_min"] = stats["input_min"]
        f_dst.attrs["input_max"] = stats["input_max"]
        f_dst.attrs["output_min"] = stats["output_min"]
        f_dst.attrs["output_max"] = stats["output_max"]

        # 也可以顺手写到 dataset attrs（可选）
        d_in.attrs["min"] = stats["input_min"]
        d_in.attrs["max"] = stats["input_max"]
        d_out.attrs["min"] = stats["output_min"]
        d_out.attrs["max"] = stats["output_max"]

    print(f"[h5] 已写入 HDF5 文件: {dst_path}")

    # 再写 JSON
    _save_stats_json(dst_path, stats)


# --------------------------
#   Split 工具
# --------------------------
def _split_indices(N: int,
                   train_ratio: float,
                   val_ratio: float,
                   test_ratio: float,
                   seed: int = 42):
    """根据比例划分索引，返回 (idx_train, idx_val, idx_test)。"""
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test 比例之和必须为 1，目前为 {total:.4f}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    # 剩余全部给 test，避免因取整导致样本丢失
    n_test = N - n_train - n_val

    if n_train <= 0 or n_test <= 0:
        raise ValueError(
            f"样本数 N={N} 太小或比例设置不合理，导致 train/test 为空 "
            f"(n_train={n_train}, n_val={n_val}, n_test={n_test})"
        )

    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]
    return idx_train, idx_val, idx_test


def _split_and_write(dst_prefix: Path,
                     input_arr: np.ndarray,
                     output_arr: np.ndarray,
                     train_ratio: float,
                     val_ratio: float,
                     test_ratio: float,
                     seed: int = 42):
    """
    按比例随机划分 (train/val/test) 并分别写入：
        <dst_prefix>_train.h5
        <dst_prefix>_val.h5
        <dst_prefix>_test.h5
    """
    if input_arr.shape[0] != output_arr.shape[0]:
        raise ValueError(
            f"input/output 第一维样本数不一致: {input_arr.shape[0]} vs {output_arr.shape[0]}"
        )

    N = input_arr.shape[0]
    idx_train, idx_val, idx_test = _split_indices(
        N, train_ratio, val_ratio, test_ratio, seed
    )
    print(f"[split] 总样本数 N={N}")
    print(f"[split] train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")

    # 生成三个目标文件路径
    base = dst_prefix
    train_path = base.with_name(base.name + "_train.h5")
    val_path = base.with_name(base.name + "_val.h5")
    test_path = base.with_name(base.name + "_test.h5")

    # 写入
    _write_h5_with_stats(train_path, input_arr[idx_train], output_arr[idx_train])
    if len(idx_val) > 0:
        _write_h5_with_stats(val_path, input_arr[idx_val], output_arr[idx_val])
    _write_h5_with_stats(test_path, input_arr[idx_test], output_arr[idx_test])


# --------------------------
#   1D Burgers 转换
# --------------------------
def convert_burgers1d(
    src_path: Path,
    dst_path: Path,
    src_key: str = "u",
    t_in: int = 0,
    t_out: int = -1,
    squeeze_last_dim: bool = True,
):
    """
    假定原始 dataset 为:
        [N, T, X] 或 [N, T, X, 1]

    转换为:
        input : [N, X]
        output: [N, X]

    之后由 PDEBenchDataset._to_chw_tensor 转为 [1,1,X]
    """
    print(f"[burgers1d] 从 {src_path} 读取 '{src_key}'，写入 {dst_path}")
    _ensure_dir(dst_path)

    with h5py.File(src_path, "r") as f_src:
        data = f_src[src_key][...]  # -> np.ndarray
        print(f"  原始 data.shape = {data.shape}")

    if data.ndim == 4 and squeeze_last_dim:
        # [N, T, X, 1] -> [N, T, X]
        if data.shape[-1] != 1:
            raise ValueError(f"最后一维不是 1，无法 squeeze: shape={data.shape}")
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(f"期望 [N, T, X]，但得到 shape={data.shape}")

    N, T, X = data.shape
    print(f"  N={N}, T={T}, X={X}")

    # 支持负索引
    if t_in < 0:
        t_in = T + t_in
    if t_out < 0:
        t_out = T + t_out

    if not (0 <= t_in < T and 0 <= t_out < T):
        raise IndexError(f"t_in={t_in}, t_out={t_out} 超出时间长度 T={T}")

    # [N, X]
    input_arr = data[:, t_in, :]
    output_arr = data[:, t_out, :]

    print(f"  input_arr.shape = {input_arr.shape}")
    print(f"  output_arr.shape = {output_arr.shape}")

    _write_h5_with_stats(dst_path, input_arr, output_arr)
    print("[burgers1d] 转换完成。")


def convert_burgers1d_split(
    src_path: Path,
    dst_prefix: Path,
    src_key: str = "u",
    t_in: int = 0,
    t_out: int = -1,
    squeeze_last_dim: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """同 burgers1d，但从一个大文件中自动随机划分 train/val/test 三个文件。"""
    print(f"[burgers1d_split] 从 {src_path} 读取 '{src_key}'，写入前缀 {dst_prefix}")
    with h5py.File(src_path, "r") as f_src:
        data = f_src[src_key][...]
        print(f"  原始 data.shape = {data.shape}")

    if data.ndim == 4 and squeeze_last_dim:
        if data.shape[-1] != 1:
            raise ValueError(f"最后一维不是 1，无法 squeeze: shape={data.shape}")
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(f"期望 [N, T, X]，但得到 shape={data.shape}")

    N, T, X = data.shape
    print(f"  N={N}, T={T}, X={X}")

    if t_in < 0:
        t_in = T + t_in
    if t_out < 0:
        t_out = T + t_out

    if not (0 <= t_in < T and 0 <= t_out < T):
        raise IndexError(f"t_in={t_in}, t_out={t_out} 超出时间长度 T={T}")

    input_arr = data[:, t_in, :]
    output_arr = data[:, t_out, :]

    print(f"  input_arr.shape = {input_arr.shape}")
    print(f"  output_arr.shape = {output_arr.shape}")

    _split_and_write(dst_prefix, input_arr, output_arr,
                     train_ratio, val_ratio, test_ratio, seed)
    print("[burgers1d_split] 转换 + 划分 完成。")


# --------------------------
#   2D Darcy 转换
# --------------------------
def convert_darcy2d(
    src_path: Path,
    dst_path: Path,
    src_key_input: str = "coeff",
    src_key_output: str = "sol",
    input_channel: int = 0,
    output_channel: int = 0,
):
    """
    假定原始 Darcy 结构大致为:
        coeff: [N, H, W] 或 [N, H, W, C_in]
        sol  : [N, H, W] 或 [N, H, W, C_out]

    转换为:
        input : [N, H, W]
        output: [N, H, W]

    之后由 PDEBenchDataset._to_chw_tensor 转为 [1,H,W]。
    """
    print(f"[darcy2d] 从 {src_path} 读取 '{src_key_input}' / '{src_key_output}'，写入 {dst_path}")
    _ensure_dir(dst_path)

    with h5py.File(src_path, "r") as f_src:
        arr_in = f_src[src_key_input][...]
        arr_out = f_src[src_key_output][...]
        print(f"  原始 coeff.shape = {arr_in.shape}")
        print(f"  原始 sol.shape   = {arr_out.shape}")

    # --- 处理输入 ---
    if arr_in.ndim == 3:
        # [N, H, W] 直接用
        input_arr = arr_in
    elif arr_in.ndim == 4:
        # [N, H, W, C_in] -> 选一个通道
        if not (0 <= input_channel < arr_in.shape[-1]):
            raise IndexError(f"input_channel={input_channel} 超出范围: C_in={arr_in.shape[-1]}")
        input_arr = arr_in[..., input_channel]
    else:
        raise ValueError(f"不支持的 coeff 形状: {arr_in.shape}")

    # --- 处理输出 ---
    if arr_out.ndim == 3:
        output_arr = arr_out
    elif arr_out.ndim == 4:
        if not (0 <= output_channel < arr_out.shape[-1]):
            raise IndexError(f"output_channel={output_channel} 超出范围: C_out={arr_out.shape[-1]}")
        output_arr = arr_out[..., output_channel]
    else:
        raise ValueError(f"不支持的 sol 形状: {arr_out.shape}")

    if input_arr.shape != output_arr.shape:
        raise ValueError(f"input/output 形状不一致: {input_arr.shape} vs {output_arr.shape}")

    print(f"  最终 input_arr.shape  = {input_arr.shape}")
    print(f"  最终 output_arr.shape = {output_arr.shape}")

    _write_h5_with_stats(dst_path, input_arr, output_arr)
    print("[darcy2d] 转换完成。")


def convert_darcy2d_split(
    src_path: Path,
    dst_prefix: Path,
    src_key_input: str = "coeff",
    src_key_output: str = "sol",
    input_channel: int = 0,
    output_channel: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """Darcy 版本的自动 train/val/test 划分。"""
    print(f"[darcy2d_split] 从 {src_path} 读取 '{src_key_input}' / '{src_key_output}'，写入前缀 {dst_prefix}")

    with h5py.File(src_path, "r") as f_src:
        arr_in = f_src[src_key_input][...]
        arr_out = f_src[src_key_output][...]
        print(f"  原始 coeff.shape = {arr_in.shape}")
        print(f"  原始 sol.shape   = {arr_out.shape}")

    if arr_in.ndim == 3:
        input_arr = arr_in
    elif arr_in.ndim == 4:
        if not (0 <= input_channel < arr_in.shape[-1]):
            raise IndexError(f"input_channel={input_channel} 超出范围: C_in={arr_in.shape[-1]}")
        input_arr = arr_in[..., input_channel]
    else:
        raise ValueError(f"不支持的 coeff 形状: {arr_in.shape}")

    if arr_out.ndim == 3:
        output_arr = arr_out
    elif arr_out.ndim == 4:
        if not (0 <= output_channel < arr_out.shape[-1]):
            raise IndexError(f"output_channel={output_channel} 超出范围: C_out={arr_out.shape[-1]}")
        output_arr = arr_out[..., output_channel]
    else:
        raise ValueError(f"不支持的 sol 形状: {arr_out.shape}")

    if input_arr.shape != output_arr.shape:
        raise ValueError(f"input/output 形状不一致: {input_arr.shape} vs {output_arr.shape}")

    print(f"  最终 input_arr.shape  = {input_arr.shape}")
    print(f"  最终 output_arr.shape = {output_arr.shape}")

    _split_and_write(dst_prefix, input_arr, output_arr,
                     train_ratio, val_ratio, test_ratio, seed)
    print("[darcy2d_split] 转换 + 划分 完成。")


# --------------------------
#   2D 时序任务 (NS / SWE / CFD)
# --------------------------
def convert_2d_timeseries(
    src_path: Path,
    dst_path: Path,
    src_key: str = "u",
    var_channel: int = 0,
    keep_full_time_for_target: bool = True,
):
    """
    假定原始结构为:
        data: [N, T, H, W]       或
        data: [N, T, H, W, Cvar] (最后一维为变量通道)

    默认行为：
        - 先选定某个变量通道 -> 得到 [N, T, H, W]
        - input  = data 全时序 [N, T, H, W]
        - output = data 全时序 [N, T, H, W] 或末帧 [N, H, W]

    之后在 PDEBenchDataset 中：
        - input  会被视为 [C=T, H, W] (time_as_channel=True)
        - output 如果你设置 target_time_index=-1，则只会用最后一帧监督（在 output 保留全 T 的情况下）。
    """
    print(f"[2d-timeseries] 从 {src_path} 读取 '{src_key}'，写入 {dst_path}")
    _ensure_dir(dst_path)

    with h5py.File(src_path, "r") as f_src:
        data = f_src[src_key][...]
        print(f"  原始 data.shape = {data.shape}")

    if data.ndim == 5:
        # [N, T, H, W, Cvar] -> 选一个变量
        N, T, H, W, Cvar = data.shape
        if not (0 <= var_channel < Cvar):
            raise IndexError(f"var_channel={var_channel} 超出范围: Cvar={Cvar}")
        data = data[..., var_channel]  # -> [N, T, H, W]
    elif data.ndim == 4:
        # [N, T, H, W] 直接用
        N, T, H, W = data.shape
    else:
        raise ValueError(f"不支持的 data 形状: {data.shape}")

    print(f"  经变量选择后 data.shape = {data.shape}")

    input_arr = data  # [N,T,H,W]

    if keep_full_time_for_target:
        output_arr = data  # [N,T,H,W]
    else:
        # 只保留最后一帧作为输出: [N,H,W]
        output_arr = data[:, -1, :, :]

    print(f"  最终 input_arr.shape  = {input_arr.shape}")
    print(f"  最终 output_arr.shape = {output_arr.shape}")

    _write_h5_with_stats(dst_path, input_arr, output_arr)
    print("[2d-timeseries] 转换完成。")


def convert_2d_timeseries_split(
    src_path: Path,
    dst_prefix: Path,
    src_key: str = "u",
    var_channel: int = 0,
    keep_full_time_for_target: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    2D 时序 PDE（如 Navier-Stokes / CFD / Shallow Water）的自动 train/val/test 划分版本。
    """
    print(f"[2d-timeseries_split] 从 {src_path} 读取 '{src_key}'，写入前缀 {dst_prefix}")

    with h5py.File(src_path, "r") as f_src:
        data = f_src[src_key][...]
        print(f"  原始 data.shape = {data.shape}")

    if data.ndim == 5:
        N, T, H, W, Cvar = data.shape
        if not (0 <= var_channel < Cvar):
            raise IndexError(f"var_channel={var_channel} 超出范围: Cvar={Cvar}")
        data = data[..., var_channel]
    elif data.ndim == 4:
        N, T, H, W = data.shape
    else:
        raise ValueError(f"不支持的 data 形状: {data.shape}")

    print(f"  经变量选择后 data.shape = {data.shape}")

    input_arr = data  # [N,T,H,W]
    if keep_full_time_for_target:
        output_arr = data
    else:
        output_arr = data[:, -1, :, :]

    print(f"  最终 input_arr.shape  = {input_arr.shape}")
    print(f"  最终 output_arr.shape = {output_arr.shape}")

    _split_and_write(dst_prefix, input_arr, output_arr,
                     train_ratio, val_ratio, test_ratio, seed)
    print("[2d-timeseries_split] 转换 + 划分 完成。")


# --------------------------
#   CLI
# --------------------------
def build_argparser():
    parser = argparse.ArgumentParser(
        description="将 PDEBench HDF5 转换为 EMO/PDEBenchDataset 兼容格式 "
                    "(input/output, <=3D per sample, 带 min/max 统计，支持自动 train/val/test 划分)"
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    # ---- inspect ----
    p_inspect = subparsers.add_parser("inspect", help="查看 HDF5 文件内容和 shape")
    p_inspect.add_argument("--src", type=str, required=True, help="源 HDF5 路径")

    # ---- burgers1d ----
    p_burgers = subparsers.add_parser("burgers1d", help="转换 1D Burgers 数据（不划分）")
    p_burgers.add_argument("--src", type=str, required=True, help="源 HDF5 路径")
    p_burgers.add_argument("--dst", type=str, required=True, help="目标 HDF5 路径")
    p_burgers.add_argument("--src-key", type=str, default="u", help="原始 dataset key，默认 'u'")
    p_burgers.add_argument("--t-in", type=int, default=0, help="输入时间步 (可为负索引)，默认 0")
    p_burgers.add_argument("--t-out", type=int, default=-1, help="输出时间步 (可为负索引)，默认 -1")
    p_burgers.add_argument("--no-squeeze-last-dim", action="store_true",
                           help="若原始维度为 [N,T,X,1]，默认会 squeeze 到 [N,T,X]；加此选项则不 squeeze")

    # ---- burgers1d_split ----
    p_burgers_s = subparsers.add_parser("burgers1d_split", help="转换 1D Burgers 数据并自动划分 train/val/test")
    p_burgers_s.add_argument("--src", type=str, required=True, help="源 HDF5 路径（如 1D_Burgers_Sols_Nu0.001.h5）")
    p_burgers_s.add_argument("--dst-prefix", type=str, required=True,
                             help="输出文件前缀，比如 './processed/burgers1d'")
    p_burgers_s.add_argument("--src-key", type=str, default="u", help="原始 dataset key，默认 'u'")
    p_burgers_s.add_argument("--t-in", type=int, default=0, help="输入时间步 (可为负索引)，默认 0")
    p_burgers_s.add_argument("--t-out", type=int, default=-1, help="输出时间步 (可为负索引)，默认 -1")
    p_burgers_s.add_argument("--no-squeeze-last-dim", action="store_true",
                             help="若原始维度为 [N,T,X,1]，默认会 squeeze；加此选项则不 squeeze")
    p_burgers_s.add_argument("--train-ratio", type=float, default=0.8, help="train 比例，默认 0.8")
    p_burgers_s.add_argument("--val-ratio", type=float, default=0.1, help="val 比例，默认 0.1")
    p_burgers_s.add_argument("--test-ratio", type=float, default=0.1, help="test 比例，默认 0.1")
    p_burgers_s.add_argument("--seed", type=int, default=42, help="划分随机种子，默认 42")

    # ---- darcy2d ----
    p_darcy = subparsers.add_parser("darcy2d", help="转换 2D Darcy 数据（不划分）")
    p_darcy.add_argument("--src", type=str, required=True, help="源 HDF5 路径")
    p_darcy.add_argument("--dst", type=str, required=True, help="目标 HDF5 路径")
    p_darcy.add_argument("--src-key-input", type=str, default="coeff", help="系数场 dataset key，默认 'coeff'")
    p_darcy.add_argument("--src-key-output", type=str, default="sol", help="解场 dataset key，默认 'sol'")
    p_darcy.add_argument("--input-channel", type=int, default=0,
                         help="当 coeff 为 [N,H,W,C] 时，选择第几个通道作为 input，默认 0")
    p_darcy.add_argument("--output-channel", type=int, default=0,
                         help="当 sol 为 [N,H,W,C] 时，选择第几个通道作为 output，默认 0")

    # ---- darcy2d_split ----
    p_darcy_s = subparsers.add_parser("darcy2d_split", help="转换 2D Darcy 数据并自动划分 train/val/test")
    p_darcy_s.add_argument("--src", type=str, required=True,
                           help="源 HDF5 路径（如 2D_DarcyFlow_beta1.0_Train.h5）")
    p_darcy_s.add_argument("--dst-prefix", type=str, required=True,
                           help="输出文件前缀，比如 './processed/darcy2d'")
    p_darcy_s.add_argument("--src-key-input", type=str, default="coeff",
                           help="系数场 dataset key（PDEBench Darcy 通常是 'a'）")
    p_darcy_s.add_argument("--src-key-output", type=str, default="sol",
                           help="解场 dataset key（PDEBench Darcy 通常是 'u'）")
    p_darcy_s.add_argument("--input-channel", type=int, default=0,
                           help="当 coeff 为 [N,H,W,C] 时，选择第几个通道作为 input，默认 0")
    p_darcy_s.add_argument("--output-channel", type=int, default=0,
                           help="当 sol 为 [N,H,W,C] 时，选择第几个通道作为 output，默认 0")
    p_darcy_s.add_argument("--train-ratio", type=float, default=0.8, help="train 比例，默认 0.8")
    p_darcy_s.add_argument("--val-ratio", type=float, default=0.1, help="val 比例，默认 0.1")
    p_darcy_s.add_argument("--test-ratio", type=float, default=0.1, help="test 比例，默认 0.1")
    p_darcy_s.add_argument("--seed", type=int, default=42, help="划分随机种子，默认 42")

    # ---- ns2d / swe2d / CFD 通用 ----
    p_ts2d = subparsers.add_parser("ts2d", help="转换 2D 时序 PDE 数据 (Navier-Stokes / Shallow Water / CFD 等)（不划分）")
    p_ts2d.add_argument("--src", type=str, required=True, help="源 HDF5 路径")
    p_ts2d.add_argument("--dst", type=str, required=True, help="目标 HDF5 路径")
    p_ts2d.add_argument("--src-key", type=str, default="u", help="原始 dataset key，默认 'u'")
    p_ts2d.add_argument("--var-channel", type=int, default=0,
                        help="当 data 为 [N,T,H,W,C] 时，选择哪个变量通道，默认 0")
    p_ts2d.add_argument("--target-full-time", action="store_true",
                        help="若指定，则 output 也保留全时序 [N,T,H,W]；否则只保留末帧 [N,H,W]")

    # ---- ts2d_split ----
    p_ts2d_s = subparsers.add_parser(
        "ts2d_split",
        help="转换 2D 时序 PDE 数据并自动划分 train/val/test (适用于 2D_CFD_Rand_*.h5 等)"
    )
    p_ts2d_s.add_argument("--src", type=str, required=True,
                          help="源 HDF5 路径（如 2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.h5）")
    p_ts2d_s.add_argument("--dst-prefix", type=str, required=True,
                          help="输出文件前缀，比如 './processed/cfd2d'")
    p_ts2d_s.add_argument("--src-key", type=str, default="u",
                          help="原始 dataset key（PDEBench CFD 通常 'u' 是速度场）")
    p_ts2d_s.add_argument("--var-channel", type=int, default=0,
                          help="当 data 为 [N,T,H,W,C] 时，选择哪个变量通道，默认 0")
    p_ts2d_s.add_argument("--target-full-time", action="store_true",
                          help="若指定，则 output 也保留全时序 [N,T,H,W]；否则只保留末帧 [N,H,W]")
    p_ts2d_s.add_argument("--train-ratio", type=float, default=0.8, help="train 比例，默认 0.8")
    p_ts2d_s.add_argument("--val-ratio", type=float, default=0.1, help="val 比例，默认 0.1")
    p_ts2d_s.add_argument("--test-ratio", type=float, default=0.1, help="test 比例，默认 0.1")
    p_ts2d_s.add_argument("--seed", type=int, default=42, help="划分随机种子，默认 42")

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    task = args.task
    if task == "inspect":
        _open_and_inspect(Path(args.src))

    elif task == "burgers1d":
        convert_burgers1d(
            src_path=Path(args.src),
            dst_path=Path(args.dst),
            src_key=args.src_key,
            t_in=args.t_in,
            t_out=args.t_out,
            squeeze_last_dim=not args.no_squeeze_last_dim,
        )

    elif task == "burgers1d_split":
        convert_burgers1d_split(
            src_path=Path(args.src),
            dst_prefix=Path(args.dst_prefix),
            src_key=args.src_key,
            t_in=args.t_in,
            t_out=args.t_out,
            squeeze_last_dim=not args.no_squeeze_last_dim,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    elif task == "darcy2d":
        convert_darcy2d(
            src_path=Path(args.src),
            dst_path=Path(args.dst),
            src_key_input=args.src_key_input,
            src_key_output=args.src_key_output,
            input_channel=args.input_channel,
            output_channel=args.output_channel,
        )

    elif task == "darcy2d_split":
        convert_darcy2d_split(
            src_path=Path(args.src),
            dst_prefix=Path(args.dst_prefix),
            src_key_input=args.src_key_input,
            src_key_output=args.src_key_output,
            input_channel=args.input_channel,
            output_channel=args.output_channel,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    elif task == "ts2d":
        convert_2d_timeseries(
            src_path=Path(args.src),
            dst_path=Path(args.dst),
            src_key=args.src_key,
            var_channel=args.var_channel,
            keep_full_time_for_target=args.target_full_time,
        )

    elif task == "ts2d_split":
        convert_2d_timeseries_split(
            src_path=Path(args.src),
            dst_prefix=Path(args.dst_prefix),
            src_key=args.src_key,
            var_channel=args.var_channel,
            keep_full_time_for_target=args.target_full_time,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    else:
        raise ValueError(f"未知 task: {task}")


if __name__ == "__main__":
    main()
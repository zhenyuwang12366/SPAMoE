#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert official PDEBench HDF5 data to a format PDEBenchDataset can load directly:
- Output files contain only two datasets: 'input' and 'output'
- Each sample has at most 3 dimensions so PDEBenchDataset._to_chw_tensor works
- Computes statistics: input_min, input_max, output_min, output_max
  and:
    1) prints them
    2) writes them to the destination HDF5 attrs
    3) writes a sidecar *_stats.json for normalization

Supports three typical setups:
1) burgers1d :  [N, T, X] or [N, T, X, 1]
   - input  = u(t_in,  x)
   - output = u(t_out, x)

2) darcy2d   :  [N, H, W] or [N, H, W, C]
   - input  = one channel (often coefficient field, e.g. permeability)
   - output = one channel (often solution field, e.g. pressure)

3) ns2d / swe2d : [N, T, H, W, C] or [N, T, H, W]
   - input  = full time series u(t, h, w), shape [N, T, H, W]
   - output = same [N, T, H, W]
   - PDEBenchDataset then uses target_time_index=-1 for last-frame supervision

Also:
- burgers1d_split / darcy2d_split / ts2d_split subcommands
  randomly split one large file into train/val/test files.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


# --------------------------
#   Shared helpers
# --------------------------
def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _open_and_inspect(src_path: Path):
    """Print datasets and shapes to verify keys."""
    print(f"[inspect] opening: {src_path}")
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
    Compute global min/max and return:
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
    print("[stats] statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")
    return stats


def _save_stats_json(dst_path: Path, stats: dict):
    """
    Write statistics to JSON:
        <dst_path.stem>_stats.json
    Example:
        burgers1d_train.h5 -> burgers1d_train_stats.json
    """
    json_path = dst_path.with_name(dst_path.stem + "_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[stats] wrote JSON: {json_path}")


def _write_h5_with_stats(dst_path: Path, input_arr: np.ndarray, output_arr: np.ndarray):
    """
    Create HDF5 with input/output datasets,
    and compute + write stats to attrs and JSON.
    """
    stats = _compute_stats(input_arr, output_arr)

    _ensure_dir(dst_path)
    with h5py.File(dst_path, "w") as f_dst:
        d_in = f_dst.create_dataset("input", data=input_arr.astype(np.float32))
        d_out = f_dst.create_dataset("output", data=output_arr.astype(np.float32))

        f_dst.attrs["input_min"] = stats["input_min"]
        f_dst.attrs["input_max"] = stats["input_max"]
        f_dst.attrs["output_min"] = stats["output_min"]
        f_dst.attrs["output_max"] = stats["output_max"]

        d_in.attrs["min"] = stats["input_min"]
        d_in.attrs["max"] = stats["input_max"]
        d_out.attrs["min"] = stats["output_min"]
        d_out.attrs["max"] = stats["output_max"]

    print(f"[h5] wrote HDF5: {dst_path}")

    _save_stats_json(dst_path, stats)


# --------------------------
#   Split helpers
# --------------------------
def _split_indices(N: int,
                   train_ratio: float,
                   val_ratio: float,
                   test_ratio: float,
                   seed: int = 42):
    """Split indices by ratio; returns (idx_train, idx_val, idx_test)."""
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test ratios must sum to 1, got {total:.4f}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    # Remainder goes to test to avoid losing samples to rounding
    n_test = N - n_train - n_val

    if n_train <= 0 or n_test <= 0:
        raise ValueError(
            f"N={N} is too small or ratios invalid; empty train/test "
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
    Randomly split (train/val/test) and write:
        <dst_prefix>_train.h5
        <dst_prefix>_val.h5
        <dst_prefix>_test.h5
    """
    if input_arr.shape[0] != output_arr.shape[0]:
        raise ValueError(
            f"input/output batch size mismatch: {input_arr.shape[0]} vs {output_arr.shape[0]}"
        )

    N = input_arr.shape[0]
    idx_train, idx_val, idx_test = _split_indices(
        N, train_ratio, val_ratio, test_ratio, seed
    )
    print(f"[split] total samples N={N}")
    print(f"[split] train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")

    base = dst_prefix
    train_path = base.with_name(base.name + "_train.h5")
    val_path = base.with_name(base.name + "_val.h5")
    test_path = base.with_name(base.name + "_test.h5")

    _write_h5_with_stats(train_path, input_arr[idx_train], output_arr[idx_train])
    if len(idx_val) > 0:
        _write_h5_with_stats(val_path, input_arr[idx_val], output_arr[idx_val])
    _write_h5_with_stats(test_path, input_arr[idx_test], output_arr[idx_test])


# --------------------------
#   1D Burgers
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
    Expect source dataset:
        [N, T, X] or [N, T, X, 1]

    Produces:
        input : [N, X]
        output: [N, X]

    PDEBenchDataset._to_chw_tensor then maps to [1,1,X].
    """
    print(f"[burgers1d] read '{src_key}' from {src_path}, write {dst_path}")
    _ensure_dir(dst_path)

    with h5py.File(src_path, "r") as f_src:
        data = f_src[src_key][...]
        print(f"  raw data.shape = {data.shape}")

    if data.ndim == 4 and squeeze_last_dim:
        if data.shape[-1] != 1:
            raise ValueError(f"last dim is not 1; cannot squeeze: shape={data.shape}")
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(f"expected [N, T, X], got shape={data.shape}")

    N, T, X = data.shape
    print(f"  N={N}, T={T}, X={X}")

    if t_in < 0:
        t_in = T + t_in
    if t_out < 0:
        t_out = T + t_out

    if not (0 <= t_in < T and 0 <= t_out < T):
        raise IndexError(f"t_in={t_in}, t_out={t_out} out of range for T={T}")

    input_arr = data[:, t_in, :]
    output_arr = data[:, t_out, :]

    print(f"  input_arr.shape = {input_arr.shape}")
    print(f"  output_arr.shape = {output_arr.shape}")

    _write_h5_with_stats(dst_path, input_arr, output_arr)
    print("[burgers1d] done.")


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
    """Same as burgers1d but randomly splits into train/val/test files."""
    print(f"[burgers1d_split] read '{src_key}' from {src_path}, prefix {dst_prefix}")
    with h5py.File(src_path, "r") as f_src:
        data = f_src[src_key][...]
        print(f"  raw data.shape = {data.shape}")

    if data.ndim == 4 and squeeze_last_dim:
        if data.shape[-1] != 1:
            raise ValueError(f"last dim is not 1; cannot squeeze: shape={data.shape}")
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(f"expected [N, T, X], got shape={data.shape}")

    N, T, X = data.shape
    print(f"  N={N}, T={T}, X={X}")

    if t_in < 0:
        t_in = T + t_in
    if t_out < 0:
        t_out = T + t_out

    if not (0 <= t_in < T and 0 <= t_out < T):
        raise IndexError(f"t_in={t_in}, t_out={t_out} out of range for T={T}")

    input_arr = data[:, t_in, :]
    output_arr = data[:, t_out, :]

    print(f"  input_arr.shape = {input_arr.shape}")
    print(f"  output_arr.shape = {output_arr.shape}")

    _split_and_write(dst_prefix, input_arr, output_arr,
                     train_ratio, val_ratio, test_ratio, seed)
    print("[burgers1d_split] convert + split done.")


# --------------------------
#   2D Darcy
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
    Expect Darcy layout:
        coeff: [N, H, W] or [N, H, W, C_in]
        sol  : [N, H, W] or [N, H, W, C_out]

    Produces:
        input : [N, H, W]
        output: [N, H, W]

    PDEBenchDataset._to_chw_tensor maps to [1,H,W].
    """
    print(f"[darcy2d] read '{src_key_input}' / '{src_key_output}' from {src_path}, write {dst_path}")
    _ensure_dir(dst_path)

    with h5py.File(src_path, "r") as f_src:
        arr_in = f_src[src_key_input][...]
        arr_out = f_src[src_key_output][...]
        print(f"  raw coeff.shape = {arr_in.shape}")
        print(f"  raw sol.shape   = {arr_out.shape}")

    if arr_in.ndim == 3:
        input_arr = arr_in
    elif arr_in.ndim == 4:
        if not (0 <= input_channel < arr_in.shape[-1]):
            raise IndexError(f"input_channel={input_channel} out of range: C_in={arr_in.shape[-1]}")
        input_arr = arr_in[..., input_channel]
    else:
        raise ValueError(f"unsupported coeff shape: {arr_in.shape}")

    if arr_out.ndim == 3:
        output_arr = arr_out
    elif arr_out.ndim == 4:
        if not (0 <= output_channel < arr_out.shape[-1]):
            raise IndexError(f"output_channel={output_channel} out of range: C_out={arr_out.shape[-1]}")
        output_arr = arr_out[..., output_channel]
    else:
        raise ValueError(f"unsupported sol shape: {arr_out.shape}")

    if input_arr.shape != output_arr.shape:
        raise ValueError(f"input/output shape mismatch: {input_arr.shape} vs {output_arr.shape}")

    print(f"  final input_arr.shape  = {input_arr.shape}")
    print(f"  final output_arr.shape = {output_arr.shape}")

    _write_h5_with_stats(dst_path, input_arr, output_arr)
    print("[darcy2d] done.")


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
    """Darcy variant with automatic train/val/test split."""
    print(f"[darcy2d_split] read '{src_key_input}' / '{src_key_output}' from {src_path}, prefix {dst_prefix}")

    with h5py.File(src_path, "r") as f_src:
        arr_in = f_src[src_key_input][...]
        arr_out = f_src[src_key_output][...]
        print(f"  raw coeff.shape = {arr_in.shape}")
        print(f"  raw sol.shape   = {arr_out.shape}")

    if arr_in.ndim == 3:
        input_arr = arr_in
    elif arr_in.ndim == 4:
        if not (0 <= input_channel < arr_in.shape[-1]):
            raise IndexError(f"input_channel={input_channel} out of range: C_in={arr_in.shape[-1]}")
        input_arr = arr_in[..., input_channel]
    else:
        raise ValueError(f"unsupported coeff shape: {arr_in.shape}")

    if arr_out.ndim == 3:
        output_arr = arr_out
    elif arr_out.ndim == 4:
        if not (0 <= output_channel < arr_out.shape[-1]):
            raise IndexError(f"output_channel={output_channel} out of range: C_out={arr_out.shape[-1]}")
        output_arr = arr_out[..., output_channel]
    else:
        raise ValueError(f"unsupported sol shape: {arr_out.shape}")

    if input_arr.shape != output_arr.shape:
        raise ValueError(f"input/output shape mismatch: {input_arr.shape} vs {output_arr.shape}")

    print(f"  final input_arr.shape  = {input_arr.shape}")
    print(f"  final output_arr.shape = {output_arr.shape}")

    _split_and_write(dst_prefix, input_arr, output_arr,
                     train_ratio, val_ratio, test_ratio, seed)
    print("[darcy2d_split] convert + split done.")


# --------------------------
#   2D time series (NS / SWE / CFD)
# --------------------------
def convert_2d_timeseries(
    src_path: Path,
    dst_path: Path,
    src_key: str = "u",
    var_channel: int = 0,
    keep_full_time_for_target: bool = True,
):
    """
    Expect:
        data: [N, T, H, W]       or
        data: [N, T, H, W, Cvar] (last dim = variable channel)

    Default:
        - pick one variable channel -> [N, T, H, W]
        - input  = full time [N, T, H, W]
        - output = full time [N, T, H, W] or last frame [N, H, W]

    In PDEBenchDataset:
        - input is treated as [C=T, H, W] when time_as_channel=True
        - with target_time_index=-1, supervision uses only the last frame when output keeps full T.
    """
    print(f"[2d-timeseries] read '{src_key}' from {src_path}, write {dst_path}")
    _ensure_dir(dst_path)

    with h5py.File(src_path, "r") as f_src:
        data = f_src[src_key][...]
        print(f"  raw data.shape = {data.shape}")

    if data.ndim == 5:
        N, T, H, W, Cvar = data.shape
        if not (0 <= var_channel < Cvar):
            raise IndexError(f"var_channel={var_channel} out of range: Cvar={Cvar}")
        data = data[..., var_channel]
    elif data.ndim == 4:
        N, T, H, W = data.shape
    else:
        raise ValueError(f"unsupported data shape: {data.shape}")

    print(f"  data.shape after var select = {data.shape}")

    input_arr = data

    if keep_full_time_for_target:
        output_arr = data
    else:
        output_arr = data[:, -1, :, :]

    print(f"  final input_arr.shape  = {input_arr.shape}")
    print(f"  final output_arr.shape = {output_arr.shape}")

    _write_h5_with_stats(dst_path, input_arr, output_arr)
    print("[2d-timeseries] done.")


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
    2D time-series PDE (e.g. Navier-Stokes / CFD / shallow water) with train/val/test split.
    """
    print(f"[2d-timeseries_split] read '{src_key}' from {src_path}, prefix {dst_prefix}")

    with h5py.File(src_path, "r") as f_src:
        data = f_src[src_key][...]
        print(f"  raw data.shape = {data.shape}")

    if data.ndim == 5:
        N, T, H, W, Cvar = data.shape
        if not (0 <= var_channel < Cvar):
            raise IndexError(f"var_channel={var_channel} out of range: Cvar={Cvar}")
        data = data[..., var_channel]
    elif data.ndim == 4:
        N, T, H, W = data.shape
    else:
        raise ValueError(f"unsupported data shape: {data.shape}")

    print(f"  data.shape after var select = {data.shape}")

    input_arr = data
    if keep_full_time_for_target:
        output_arr = data
    else:
        output_arr = data[:, -1, :, :]

    print(f"  final input_arr.shape  = {input_arr.shape}")
    print(f"  final output_arr.shape = {output_arr.shape}")

    _split_and_write(dst_prefix, input_arr, output_arr,
                     train_ratio, val_ratio, test_ratio, seed)
    print("[2d-timeseries_split] convert + split done.")


# --------------------------
#   CLI
# --------------------------
def build_argparser():
    parser = argparse.ArgumentParser(
        description="Convert PDEBench HDF5 to EMO/PDEBenchDataset format "
                    "(input/output, <=3D per sample, min/max stats, optional train/val/test split)"
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    p_inspect = subparsers.add_parser("inspect", help="Print HDF5 contents and shapes")
    p_inspect.add_argument("--src", type=str, required=True, help="Source HDF5 path")

    p_burgers = subparsers.add_parser("burgers1d", help="Convert 1D Burgers (no split)")
    p_burgers.add_argument("--src", type=str, required=True, help="Source HDF5 path")
    p_burgers.add_argument("--dst", type=str, required=True, help="Destination HDF5 path")
    p_burgers.add_argument("--src-key", type=str, default="u", help="Source dataset key, default 'u'")
    p_burgers.add_argument("--t-in", type=int, default=0, help="Input time index (may be negative), default 0")
    p_burgers.add_argument("--t-out", type=int, default=-1, help="Output time index (may be negative), default -1")
    p_burgers.add_argument("--no-squeeze-last-dim", action="store_true",
                           help="If raw shape is [N,T,X,1], default squeezes to [N,T,X]; set this to keep 4D")

    p_burgers_s = subparsers.add_parser("burgers1d_split", help="Convert 1D Burgers and split train/val/test")
    p_burgers_s.add_argument("--src", type=str, required=True, help="Source HDF5 (e.g. 1D_Burgers_Sols_Nu0.001.h5)")
    p_burgers_s.add_argument("--dst-prefix", type=str, required=True,
                             help="Output prefix, e.g. './processed/burgers1d'")
    p_burgers_s.add_argument("--src-key", type=str, default="u", help="Source dataset key, default 'u'")
    p_burgers_s.add_argument("--t-in", type=int, default=0, help="Input time index (may be negative), default 0")
    p_burgers_s.add_argument("--t-out", type=int, default=-1, help="Output time index (may be negative), default -1")
    p_burgers_s.add_argument("--no-squeeze-last-dim", action="store_true",
                             help="If raw shape is [N,T,X,1], default squeezes; set this to keep 4D")
    p_burgers_s.add_argument("--train-ratio", type=float, default=0.8, help="Train fraction, default 0.8")
    p_burgers_s.add_argument("--val-ratio", type=float, default=0.1, help="Val fraction, default 0.1")
    p_burgers_s.add_argument("--test-ratio", type=float, default=0.1, help="Test fraction, default 0.1")
    p_burgers_s.add_argument("--seed", type=int, default=42, help="Split RNG seed, default 42")

    p_darcy = subparsers.add_parser("darcy2d", help="Convert 2D Darcy (no split)")
    p_darcy.add_argument("--src", type=str, required=True, help="Source HDF5 path")
    p_darcy.add_argument("--dst", type=str, required=True, help="Destination HDF5 path")
    p_darcy.add_argument("--src-key-input", type=str, default="coeff", help="Coefficient dataset key, default 'coeff'")
    p_darcy.add_argument("--src-key-output", type=str, default="sol", help="Solution dataset key, default 'sol'")
    p_darcy.add_argument("--input-channel", type=int, default=0,
                         help="When coeff is [N,H,W,C], which channel to use as input, default 0")
    p_darcy.add_argument("--output-channel", type=int, default=0,
                         help="When sol is [N,H,W,C], which channel to use as output, default 0")

    p_darcy_s = subparsers.add_parser("darcy2d_split", help="Convert 2D Darcy and split train/val/test")
    p_darcy_s.add_argument("--src", type=str, required=True,
                           help="Source HDF5 (e.g. 2D_DarcyFlow_beta1.0_Train.h5)")
    p_darcy_s.add_argument("--dst-prefix", type=str, required=True,
                           help="Output prefix, e.g. './processed/darcy2d'")
    p_darcy_s.add_argument("--src-key-input", type=str, default="coeff",
                           help="Coefficient key (PDEBench Darcy often 'a')")
    p_darcy_s.add_argument("--src-key-output", type=str, default="sol",
                           help="Solution key (PDEBench Darcy often 'u')")
    p_darcy_s.add_argument("--input-channel", type=int, default=0,
                           help="When coeff is [N,H,W,C], which input channel, default 0")
    p_darcy_s.add_argument("--output-channel", type=int, default=0,
                           help="When sol is [N,H,W,C], which output channel, default 0")
    p_darcy_s.add_argument("--train-ratio", type=float, default=0.8, help="Train fraction, default 0.8")
    p_darcy_s.add_argument("--val-ratio", type=float, default=0.1, help="Val fraction, default 0.1")
    p_darcy_s.add_argument("--test-ratio", type=float, default=0.1, help="Test fraction, default 0.1")
    p_darcy_s.add_argument("--seed", type=int, default=42, help="Split RNG seed, default 42")

    p_ts2d = subparsers.add_parser("ts2d", help="Convert 2D time-series PDE (NS / SWE / CFD, no split)")
    p_ts2d.add_argument("--src", type=str, required=True, help="Source HDF5 path")
    p_ts2d.add_argument("--dst", type=str, required=True, help="Destination HDF5 path")
    p_ts2d.add_argument("--src-key", type=str, default="u", help="Source dataset key, default 'u'")
    p_ts2d.add_argument("--var-channel", type=int, default=0,
                        help="When data is [N,T,H,W,C], which variable channel, default 0")
    p_ts2d.add_argument("--target-full-time", action="store_true",
                        help="If set, output keeps full time [N,T,H,W]; else last frame only [N,H,W]")

    p_ts2d_s = subparsers.add_parser(
        "ts2d_split",
        help="Convert 2D time-series PDE and split train/val/test (e.g. 2D_CFD_Rand_*.h5)"
    )
    p_ts2d_s.add_argument("--src", type=str, required=True,
                          help="Source HDF5 (e.g. 2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.h5)")
    p_ts2d_s.add_argument("--dst-prefix", type=str, required=True,
                          help="Output prefix, e.g. './processed/cfd2d'")
    p_ts2d_s.add_argument("--src-key", type=str, default="u",
                          help="Source key (PDEBench CFD often uses 'u' for velocity)")
    p_ts2d_s.add_argument("--var-channel", type=int, default=0,
                          help="When data is [N,T,H,W,C], which variable channel, default 0")
    p_ts2d_s.add_argument("--target-full-time", action="store_true",
                          help="If set, output keeps full time [N,T,H,W]; else last frame [N,H,W]")
    p_ts2d_s.add_argument("--train-ratio", type=float, default=0.8, help="Train fraction, default 0.8")
    p_ts2d_s.add_argument("--val-ratio", type=float, default=0.1, help="Val fraction, default 0.1")
    p_ts2d_s.add_argument("--test-ratio", type=float, default=0.1, help="Test fraction, default 0.1")
    p_ts2d_s.add_argument("--seed", type=int, default=42, help="Split RNG seed, default 42")

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
        raise ValueError(f"unknown task: {task}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert LaMO's six PDE task datasets to HDF5 for direct use in FWINO/EMO.

Supported tasks:
  - pipe        : Pipe_X / Pipe_Y / Pipe_Q.npy -> input[2,H,W], output[1,H,W]
  - airfoil     : NACA_Cylinder_X/Y/Q.npy      -> input[2,221,51], output[1,221,51]
  - darcy       : piececonst_r421_*.mat        -> input/output[1,s,s]
  - navier      : NavierStokes_V1e-5_N1200_T20.mat -> input[T_in,h,h], output[T_out,h,h]
  - plasticity  : plas_N987_T20.mat -> input[1,H,W], output[4*T,H,W] (time flattened to channels);
                  also writes pos and time

Notes:
  - plasticity uses a regular BCHW grid (time flattened into channels).
  - Output HDF5 files contain at least input/output datasets and a sidecar *_stats.json for normalization.
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
    Write HDF5 and save min/max stats to attrs and a sidecar JSON.
    If ``stats`` is provided, reuse it (for global statistics); otherwise compute from current arrays.
    Returns the stats actually written.
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
        raise ValueError(f"train/val/test ratios must sum to 1, got {total:.4f}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    n_test = N - n_train - n_val
    if n_train <= 0 or n_test <= 0:
        raise ValueError(f"N={N} is too small or ratios are invalid: train={n_train}, val={n_val}, test={n_test}")
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
    """Split by ratio and write train/val/test files; include pos/time if provided; reuse global stats."""
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
    """Downsample H and W by factors (r1, r2)."""
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
    q = np.load(data_root / "NACA_Cylinder_Q.npy")[:, 4]  # channel index 4 (5th channel)

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
    u = u[:, ::downsample, ::downsample, :][:, :h, :h, :]  # truncate to integer grid

    input_arr = u[:, :, :, :t_in].transpose(0, 3, 1, 2)   # [N,T_in,H,W]
    output_arr = u[:, :, :, t_in : t_in + t_out].transpose(0, 3, 1, 2)  # [N,T_out,H,W]
    return input_arr, output_arr


def convert_plasticity(data_root: Path, downsamplex: int, downsampley: int):
    """
    Plasticity: 1D input replicated along y; output has 4 deformation channels and T=20 time steps.
    Time is flattened into channels: output shape [N, 4*T, H, W] for BCHW consumption.
    Also returns pos (regular grid) and time (original time axis, for viz / inverse normalization).
    """
    data = scio.loadmat(data_root / "plas_N987_T20.mat")
    inp_raw = data["input"]          # expected [N, 101]
    out_raw = data["output"]         # expected [N, 101, 31, T, 4] (LaMO may transpose to [N,101,31,4,T])

    s1 = int(((101 - 1) / downsamplex) + 1)
    s2 = int(((31 - 1) / downsampley) + 1)
    T = out_raw.shape[-2] if out_raw.ndim >= 5 else 20

    # Input: replicate along y, then crop
    inp = inp_raw[:, ::downsamplex][:, :s1]          # [N, s1]
    inp = np.repeat(inp[:, :, None], s2, axis=2)     # [N, s1, s2]
    input_arr = inp[:, None, :, :]                   # [N,1,H,W]

    # Output: reorder to [N, s1, s2, deform, T]
    if out_raw.shape[-1] == 4:  # last dim is deformation
        out_reordered = np.transpose(out_raw, (0, 1, 2, 4, 3))
    else:
        out_reordered = out_raw
    out_reordered = out_reordered[:, ::downsamplex, ::downsampley, :, :][:, :s1, :s2, :, :]  # [N,H,W,4,T]
    out_chw = np.transpose(out_reordered, (0, 3, 4, 1, 2))  # [N,4,T,H,W]
    n, c, t, h, w = out_chw.shape
    output_arr = out_chw.reshape(n, c * t, h, w)  # [N, 4*T, H, W]

    # Grid coords (match LaMO): uniform [0, 1]
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
                        help="Task to convert")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory with task data, e.g. LaMO/data/Pipe")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .h5 path, e.g. ./pdebench_data/pipe")

    # pipe / airfoil
    parser.add_argument(
        "--downsamplex",
        type=int,
        default=None,
        help="X downsampling (pipe/airfoil/plasticity); default 1 for those tasks",
    )
    parser.add_argument(
        "--downsampley",
        type=int,
        default=None,
        help="Y downsampling (pipe/airfoil/plasticity); default 1 for those tasks",
    )

    # darcy
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Downsample factor (darcy default 5, navier default 1)",
    )

    # navier
    parser.add_argument(
        "--t-in",
        type=int,
        default=None,
        help="Navier-Stokes history length (default 10)",
    )
    parser.add_argument(
        "--t-out",
        type=int,
        default=None,
        help="Navier-Stokes prediction horizon (default 10)",
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

    # Global stats only (no full-dataset file); reuse across splits
    stats_full = {
        "input_min": float(np.min(input_arr)),
        "input_max": float(np.max(input_arr)),
        "output_min": float(np.min(output_arr)),
        "output_max": float(np.max(output_arr)),
    }

    # Write train/val/test by ratio; plasticity includes pos/time; stats from full data
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

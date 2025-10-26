#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, argparse
from typing import List, Tuple, Optional, Dict
import numpy as np
from tqdm import tqdm

import zarr
from numcodecs import Blosc
from config.seismic_moe_config import SeismicMOEConfig

# ---------------------------
# 命名与类别推断
# ---------------------------

def to_snake_lower(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    # 驼峰 -> 下划线边界
    s = re.sub(r'([A-Z]+)([A-Z][a-z0-9])', r'\1_\2', s)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub(r'[^A-Za-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_').lower()
    parts = [p for p in s.split('_') if p]
    if not parts:
        return s
    if parts[0] == 'style':
        # 与原脚本一致：合并 style 子类型，仅区分 a/b
        suffix = parts[-1] if len(parts) >= 2 else 'a'
        if len(parts) >= 3 and parts[1] == 'style':
            return '_'.join(parts[:3])  # style_style_a/b
        return f"style_style_{suffix}"
    return '_'.join(parts)

def infer_group_and_variant(subdir_name: str):
    # 返回 (group, variant, norm_type_key)
    # group ∈ {'vel','style','fault'} ; variant ∈ {'flat','curve','style'}
    name = subdir_name
    if name.startswith("CurveFault_"):
        return 'fault', 'curve', to_snake_lower(name)
    if name.startswith("FlatFault_"):
        return 'fault', 'flat', to_snake_lower(name)
    if name.startswith("CurveVel_"):
        return 'vel', 'curve', to_snake_lower(name)
    if name.startswith("FlatVel_"):
        return 'vel', 'flat', to_snake_lower(name)
    if name.startswith("Style_"):
        return 'style', 'style', to_snake_lower(name)
    return None, None, None

def family_from_type_key(type_key: str) -> str:
    tk = type_key.lower()
    if '_vel_' in tk:
        return 'vel'
    if '_fault_' in tk:
        return 'fault'
    if tk.startswith('style'):
        return 'style'
    raise ValueError(f"无法从 type_key='{type_key}' 推断 family")

# fault 正则
FAULT_SEIS_RE = re.compile(r"seis_?(\d+)_1_(\d+)\.npy$")


# ---------------------------
# 扫描 train_samples 生成 “样本清单”
# 每条记录： (input_path, output_path or None, type_key, input_file_tag)
# ---------------------------

def collect_supervised_samples(train_dir: str,
                               type_id_map: Dict[str, int],
                               expect_ch: int = 5) -> List[Tuple[str, Optional[str], str, str]]:
    if not os.path.isdir(train_dir):
        raise RuntimeError(f"训练目录不存在: {train_dir}")

    subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    records = []

    for sub in sorted(subdirs):
        group, variant, type_key = infer_group_and_variant(sub)
        if group is None:
            continue
        if not (type_key.endswith('_a') or type_key.endswith('_b')):
            raise ValueError(f"目录 '{sub}' 规范化为 '{type_key}'，但未以 _a/_b 结尾。")
        if type_key not in type_id_map:
            raise KeyError(f"type_key '{type_key}' 不在映射表中。")

        sub_path = os.path.join(train_dir, sub)

        if group in ('vel', 'style'):
            data_dir = os.path.join(sub_path, 'data')
            model_dir = os.path.join(sub_path, 'model')
            if not (os.path.isdir(data_dir) and os.path.isdir(model_dir)):
                continue
            data_files = sorted(glob.glob(os.path.join(data_dir, 'data*.npy')))
            for df in data_files:
                stem = os.path.splitext(os.path.basename(df))[0]   # dataX
                idxs = stem.replace('data', '')
                mf = os.path.join(model_dir, f"model{idxs}.npy")
                if os.path.exists(mf):
                    records.append((df, mf, type_key, f"{sub}/{os.path.basename(df)}"))
        elif group == 'fault':
            all_seis = sorted(glob.glob(os.path.join(sub_path, 'seis*.npy')))
            for sf in all_seis:
                name = os.path.basename(sf)
                m = FAULT_SEIS_RE.fullmatch(name)
                if not m:
                    continue
                n = int(m.group(1)); i = int(m.group(2))
                vel1 = os.path.join(sub_path, f"vel_{n}_1_{i}.npy")
                vel2 = os.path.join(sub_path, f"vel{n}_1_{i}.npy")
                of = vel1 if os.path.exists(vel1) else (vel2 if os.path.exists(vel2) else None)
                if of is not None:
                    records.append((sf, of, type_key, f"{sub}/{name}"))
        else:
            continue

    if not records:
        raise RuntimeError("未在 train_samples/ 下找到可监督样本。")
    return records

def collect_test_inputs(test_dir: str) -> List[str]:
    if not os.path.isdir(test_dir):
        return []
    return sorted(glob.glob(os.path.join(test_dir, '*.npy')))


# ---------------------------
# 统计总样本数（文件内 shape[0] 之和）
# ---------------------------

def count_total_samples_records(records: List[Tuple[str, Optional[str], str, str]]) -> int:
    total = 0
    for in_path, _, _, _ in records:
        arr = np.load(in_path, mmap_mode='r')
        total += int(arr.shape[0])
    return total

def count_total_samples_files(file_list: List[str]) -> int:
    total = 0
    for in_path in file_list:
        arr = np.load(in_path, mmap_mode='r')
        total += int(arr.shape[0])
    return total


# ---------------------------
# 子类型固定比例划分（核心）
# ---------------------------

def get_val_ratio_for_family(family: str) -> float:
    family = family.lower()
    if family == 'vel':
        return 6.0 / 30.0
    if family == 'fault':
        return 6.0 / 54.0
    if family == 'style':
        return 7.0 / 67.0
    raise ValueError(f"未知 family: {family}")

def split_fixed_ratio_indices(
    idxs: np.ndarray, val_ratio: float, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idxs = np.array(idxs, dtype=np.int64)
    if idxs.size == 0:
        return idxs, idxs
    rng.shuffle(idxs)
    n = idxs.size
    n_val = int(round(n * val_ratio))
    if n >= 2:
        n_val = max(1, min(n_val, n - 1))
    else:
        n_val = 0
    val_idx = idxs[:n_val]
    train_idx = idxs[n_val:]
    return train_idx, val_idx

def split_by_subtype_fixed_ratio(
    type_names: np.ndarray,  # 前 sup_end 段的 type_name（字符串数组）
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    subtype_to_indices: Dict[str, list] = {}
    for idx, tk in enumerate(type_names.tolist()):
        subtype_to_indices.setdefault(tk, []).append(idx)

    all_train, all_val = [], []
    for subtype, idx_list in subtype_to_indices.items():
        fam = family_from_type_key(subtype)
        val_ratio = get_val_ratio_for_family(fam)
        tr, va = split_fixed_ratio_indices(np.array(idx_list, dtype=np.int64), val_ratio, seed=seed)
        if tr.size:
            all_train.append(tr)
        if va.size:
            all_val.append(va)

    train_idx = np.concatenate(all_train) if all_train else np.array([], dtype=np.int64)
    val_idx   = np.concatenate(all_val)   if all_val   else np.array([], dtype=np.int64)
    return train_idx, val_idx


# ---------------------------
# 主流程：写 Zarr
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Build seismic_moe.zarr from OpenFWI-style folders")
    ap.add_argument('--data_dir', required=True, help='根数据目录，含 train_samples/ 与(可选) test/')
    ap.add_argument('--zarr_out', required=True, help='输出 Zarr 目录（不存在会创建）')
    ap.add_argument('--chunks', type=int, default=32, help='样本维 chunk 大小')
    ap.add_argument('--dtype', type=str, default='float32', choices=['float32','float16'], help='存盘 dtype')
    ap.add_argument('--seed', type=int, default=42)

    # family/单类型控制
    ap.add_argument('--family', type=str, default='all',
                    help="可选 'all' 或指定单类型（如 curve_vel_a / flat_vel_b / curve_fault_a / style_a）")
    ap.add_argument('--include_test', type=int, default=0,
                    help="当 family != 'all' 时是否并入 test/ 无标签输入（0/1，默认0）。family='all'时忽略。")
    ap.add_argument('--remap_single_label', type=int, default=1,
                    help="当 family != 'all' 时是否把该类的 label 重映射为 0（0/1，默认1）")

    args = ap.parse_args()

    data_dir   = args.data_dir
    zarr_out   = args.zarr_out
    dtype_str  = args.dtype

    config = SeismicMOEConfig()
    type_id_map = config.type_id_specific

    # family 解析
    family_arg = to_snake_lower(args.family) if args.family.lower() != 'all' else 'all'
    if family_arg != 'all':
        if not (family_arg.endswith('_a') or family_arg.endswith('_b')):
            raise ValueError(f"--family '{args.family}' 需要以 _a/_b 结尾（如 curve_vel_a / style_a）")
        if family_arg not in type_id_map:
            raise KeyError(f"--family '{args.family}' 规范化为 '{family_arg}'，但不在 type_id_map 中。")

    train_dir = os.path.join(data_dir, 'train_samples')
    test_dir  = os.path.join(data_dir, 'test')

    print("扫描有监督样本……")
    sup_records_all = collect_supervised_samples(train_dir, type_id_map)

    # family 过滤
    if family_arg == 'all':
        sup_records = sup_records_all
    else:
        sup_records = [rec for rec in sup_records_all if rec[2] == family_arg]
        if not sup_records:
            raise RuntimeError(f"未找到 family='{family_arg}' 的有监督样本。")

    N_sup_files = len(sup_records)
    N_sup = count_total_samples_records(sup_records)
    print(f"family = {family_arg} | 监督文件对 {N_sup_files} 个，总样本数 ≈ {N_sup}")

    # test 行为
    if family_arg == 'all':
        test_files = collect_test_inputs(test_dir)
        N_test = count_total_samples_files(test_files) if test_files else 0
    else:
        if args.include_test:
            test_files = collect_test_inputs(test_dir)
            N_test = count_total_samples_files(test_files) if test_files else 0
            print(f"[include_test=1] 发现无标签测试输入 {len(test_files)} 个文件，总样本数 ≈ {N_test}")
        else:
            test_files = []
            N_test = 0

    # 预创建 Zarr
    compressor = Blosc(cname="zstd", clevel=4, shuffle=Blosc.SHUFFLE)
    root = zarr.open_group(zarr_out, mode='w')
    target_dtype = np.float32 if dtype_str == 'float32' else np.float16

    # 计算总量
    N_total = N_sup + N_test
    if N_total == 0:
        raise RuntimeError("没有可写入的样本。")

    # === 关键改动：创建新形状的数据集 ===
    inputs_ds = root.create_dataset(
        'inputs', shape=(N_total, 1, 1000, 350), chunks=(args.chunks, 1, 1000, 350),
        dtype=target_dtype, compressor=compressor
    )
    outputs_ds = root.create_dataset(
        'outputs', shape=(N_total, 1, 70, 70), chunks=(args.chunks, 1, 70, 70),
        dtype=target_dtype, compressor=compressor
    )
    labels_ds = root.create_dataset('labels', shape=(N_total,), chunks=(max(1,args.chunks),), dtype='int64', compressor=compressor)
    type_name_ds = root.create_dataset('type_name', shape=(N_total,), dtype=object, object_codec=zarr.codecs.VLenUTF8())
    input_file_ds = root.create_dataset('input_file', shape=(N_total,), dtype=object, object_codec=zarr.codecs.VLenUTF8())

    # 写入 supervised
    write_ptr = 0
    print("写入有监督样本到 Zarr ……")
    for in_path, out_path, type_key, tag in tqdm(sup_records):
        x_arr = np.load(in_path, mmap_mode='r')   # 期望 [M, 5, 1000, 70] 或已拼接 [M, 1, 1000, 350]
        y_arr = np.load(out_path, mmap_mode='r')  # [M, 70, 70] 或 [M, 1, 70, 70]

        M = int(x_arr.shape[0])

        # === 输入规范化到 [M, 1, 1000, 350] ===
        if x_arr.ndim != 4:
            raise ValueError(f"{in_path} 维度应为 4，实际 {x_arr.shape}")
        if x_arr.shape[1:] == (5, 1000, 70):
            # [M,5,1000,70] -> [M,1,1000,350]  （沿最后一维拼接 5 个通道）
            # 先把通道维移到宽度前： [M,1000,5,70]，再 reshape 到 [M,1000,350]
            x_cat = x_arr.transpose(0, 2, 1, 3).reshape(M, 1000, 5 * 70)
            x_cat = x_cat[:, :, :350]  # 防御性截断（理论上正好 350）
            x_cat = x_cat.astype(target_dtype, copy=False)
            x_cat = x_cat[:, None, :, :]  # [M,1,1000,350]
        elif x_arr.shape[1:] == (1, 1000, 350):
            # 已经是目标形状
            x_cat = x_arr.astype(target_dtype, copy=False)
        else:
            raise ValueError(f"{in_path} 的 inputs 形状不支持：{x_arr.shape[1:]}, 期望 [5,1000,70] 或 [1,1000,350]")

        # === 输出规范化到 [M, 1, 70, 70] ===
        if y_arr.ndim == 4 and y_arr.shape[1:] == (1, 70, 70):
            y_cat = y_arr.astype(target_dtype, copy=False)
        elif y_arr.ndim == 3 and y_arr.shape[1:] == (70, 70):
            y_cat = y_arr[:, None, :, :].astype(target_dtype, copy=False)  # [M,1,70,70]
        else:
            raise ValueError(f"{out_path} 的 outputs 形状不支持：{y_arr.shape}, 期望 [M,70,70] 或 [M,1,70,70]")

        # label：全体时用全局 id；单类型可选重映射为 0
        if family_arg == 'all':
            lab_val = int(type_id_map[type_key])
        else:
            lab_val = 0 if args.remap_single_label else int(type_id_map[type_key])

        sl = slice(write_ptr, write_ptr + M)
        inputs_ds[sl] = x_cat
        outputs_ds[sl] = y_cat
        labels_ds[sl] = lab_val
        type_name_ds[sl] = [type_key] * M
        input_file_ds[sl] = [tag] * M
        write_ptr += M

    sup_end = write_ptr

    # 写入无标签 test（只写 inputs 与元信息；labels=-1; type_name="test"）
    if N_test > 0:
        print("写入无标签测试样本到 Zarr ……")
        for in_path in tqdm(test_files):
            x_arr = np.load(in_path, mmap_mode='r')  # [M,5,1000,70] 或 [M,1,1000,350]
            M = int(x_arr.shape[0])

            if x_arr.shape[1:] == (5, 1000, 70):
                x_cat = x_arr.transpose(0, 2, 1, 3).reshape(M, 1000, 5 * 70)
                x_cat = x_cat[:, :, :350]
                x_cat = x_cat.astype(target_dtype, copy=False)
                x_cat = x_cat[:, None, :, :]  # [M,1,1000,350]
            elif x_arr.shape[1:] == (1, 1000, 350):
                x_cat = x_arr.astype(target_dtype, copy=False)
            else:
                raise ValueError(f"{in_path} 的 inputs 形状不支持：{x_arr.shape[1:]}, 期望 [5,1000,70] 或 [1,1000,350]")

            sl = slice(write_ptr, write_ptr + M)
            inputs_ds[sl] = x_cat
            labels_ds[sl] = -1
            type_name_ds[sl] = ["test"] * M
            input_file_ds[sl] = [f"test/{os.path.basename(in_path)}"] * M
            write_ptr += M

    assert write_ptr == N_total

    # —— 划分 train / val （每个子类型固定比例），不做 test —— #
    sup_types = np.asarray(type_name_ds[:sup_end], dtype=object)

    if family_arg == 'all':
        # 每个子类型各自按其 family 的固定比例划分，再合并
        train_idx, val_idx = split_by_subtype_fixed_ratio(
            type_names=sup_types, seed=args.seed
        )
    else:
        # 单一子类型：直接按其 family 的固定比例划分
        fam = family_from_type_key(family_arg)
        val_ratio = get_val_ratio_for_family(fam)
        all_sup_indices = np.arange(sup_end, dtype=np.int64)
        train_idx, val_idx = split_fixed_ratio_indices(
            all_sup_indices, val_ratio=val_ratio, seed=args.seed
        )

    test_idx = np.array([], dtype=np.int64)  # 不划分 test

    splits = root.create_group('splits')
    splits.create_dataset('train_idx', data=train_idx.astype(np.int64), compressor=compressor)
    splits.create_dataset('val_idx',   data=val_idx.astype(np.int64),   compressor=compressor)
    splits.create_dataset('test_idx',  data=test_idx.astype(np.int64),  compressor=compressor)

    # 若存在“无标签 test 区间”，另外保存它们的索引段（可选）
    if N_test > 0:
        unsup_idx = np.arange(sup_end, N_total, dtype=np.int64)
        splits.create_dataset('unsup_test_idx', data=unsup_idx, compressor=compressor)

    # 元信息写到 .zattrs
    type_id_map_attr = {str(k): int(v) for k, v in type_id_map.items()}
    id_type_map_attr = {int(v): str(k) for k, v in type_id_map.items()}

    root.attrs.update({
        "schema": {
            "inputs":  {"shape": [N_total, 1, 1000, 350], "dtype": dtype_str},
            "outputs": {"shape": [N_total, 1, 70, 70],   "dtype": dtype_str},
            "labels":  {"shape": [N_total], "dtype": "int64"},
            "type_name": "vlen-utf8",
            "input_file": "vlen-utf8",
        },
        "counts": {
            "N_total": int(N_total),
            "N_supervised": int(sup_end),
            "N_test_inputs": int(N_test)
        },
        "chunks": int(args.chunks),
        "compressor": "blosc-zstd",
        "seed": int(args.seed),
        "family": family_arg,
        "label_remap": bool(args.remap_single_label if family_arg != 'all' else False),
        "type_id_map": type_id_map_attr,
        "id_type_map": id_type_map_attr,
        "split_policy": "per-subtype-fixed-family-ratio",
        "family_val_ratios": {
            "vel":   6.0/30.0,
            "fault": 6.0/54.0,
            "style": 7.0/67.0
        },
        "note": "Inputs reshaped to [1,1000,350]; outputs expanded to [1,70,70]."
    })

    print(f"  完成：Zarr 写入 {zarr_out}")
    print(f"  family: {family_arg} | 监督样本: {sup_end}  | 无标签测试样本: {N_test}")
    print(f"  splits/train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    if N_test > 0:
        print(f"  splits/unsup_test_idx: {len(unsup_idx)}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, json, math, argparse, sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from tqdm import tqdm

import zarr
from numcodecs import Blosc

# ---------------------------
# 命名与类别推断
# ---------------------------

def to_snake_lower(s: str) -> str:
    s = re.sub(r'[^A-Za-z0-9]+', '_', s.strip()).strip('_').lower()
    parts = [p for p in s.split('_') if p]  # 去重空片段
    if not parts:
        return s
    if parts[0] == 'style':
        # 情况1: style_a -> style_style_a
        # 情况2: style_style_a -> 保持不变
        # 其他(如 style_xxx_a) -> 取最后一段作为 a/b 后缀
        suffix = parts[-1] if len(parts) >= 2 else 'a'
        if len(parts) >= 3 and parts[1] == 'style':
            return '_'.join(parts[:3])  # 已经是 style_style_a/b
        return f"style_style_{suffix}"
    return '_'.join(parts)
    
def infer_group_and_variant(subdir_name: str):
    # 返回 (group, variant, norm_type_key)
    # group ∈ {'vel','style','fault'}
    # variant ∈ {'flat','curve', None}
    # type_key 统一为小写蛇形，必须以 _a/_b 结尾，否则后续会报错
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

# fault 正则
FAULT_SEIS_RE = re.compile(r"seis_?(\d+)_1_(\d+)\.npy$")

# ---------------------------
# 扫描 train_samples 生成 “样本清单”
# 每个样本一条记录： (input_path, output_path or None, type_key, input_file_tag)
# ---------------------------

def collect_supervised_samples(train_dir: str,
                               type_id_map: Dict[str, int],
                               expect_ch: int = 5) -> List[Tuple[str, Optional[str], str, str]]:
    """
    返回有监督样本清单：[(in_path, out_path, type_key, input_file_tag), ...]
    Vel/Style: data{i}.npy ↔ model{i}.npy（文件内通常有多条样本，shape[0]=num_samples）
    Fault:     同目录 pair: seis_* ↔ vel_*（通常每个文件内一条或多条样本）
    """
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
                    # 不展开到样本维，这里只记录“文件对”；写入时再逐样本写入
                    records.append((df, mf, type_key, f"{sub}/{os.path.basename(df)}"))
        elif group == 'fault':
            # fault: 目录下成对文件
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
# 训练/验证/测试 划分
# ---------------------------

def stratified_split_indices(labels: np.ndarray, ratios=(0.8, 0.1, 0.1), seed=42):
    rng = np.random.RandomState(seed)
    N = len(labels)
    idx = np.arange(N)
    # 简单随机 + 比例；如需严格分层，可按 label 分组后再按比例取
    rng.shuffle(idx)
    n_train = int(round(N * ratios[0]))
    n_val   = int(round(N * ratios[1]))
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]
    return train_idx, val_idx, test_idx

# ---------------------------
# 主流程：写 Zarr
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Build seismic_moe.zarr from OpenFWI-style folders")
    ap.add_argument('--data_dir', required=True, help='根数据目录，含 train_samples/ 与(可选) test/')
    ap.add_argument('--zarr_out', required=True, help='输出 Zarr 目录（不存在会创建）')
    ap.add_argument('--mapping_json', required=True, help='type_id_specific 映射 JSON 文件路径')
    ap.add_argument('--chunks', type=int, default=64, help='样本维 chunk 大小，如 64')
    ap.add_argument('--dtype', type=str, default='float32', choices=['float32','float16'], help='存盘 dtype')
    ap.add_argument('--split_ratio', nargs=3, type=float, default=[0.8,0.2,0.0], help='train/val/test 比例（针对有监督样本）')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    data_dir   = args.data_dir
    zarr_out   = args.zarr_out
    dtype_str  = args.dtype
    ratios     = tuple(args.split_ratio)
    assert abs(sum(ratios)-1.0) < 1e-6, "split_ratio 之和必须为 1.0"

    with open(args.mapping_json, 'r', encoding='utf-8') as f:
        type_id_map = json.load(f)
    if not isinstance(type_id_map, dict) or not type_id_map:
        raise ValueError("mapping_json 必须是非空字典")

    train_dir = os.path.join(data_dir, 'train_samples')
    test_dir  = os.path.join(data_dir, 'test')

    print("扫描有监督样本……")
    sup_records = collect_supervised_samples(train_dir, type_id_map)

    N_sup_files = len(sup_records)
    N_sup = count_total_samples_records(sup_records)
    print(f"找到监督文件对 {N_sup_files} 个，总样本数 ≈ {N_sup}")

    test_files = collect_test_inputs(test_dir)
    N_test = count_total_samples_files(test_files) if test_files else 0
    if N_test > 0:
        print(f"发现无标签测试输入 {len(test_files)} 个文件，总样本数 ≈ {N_test}")

    # 预创建 Zarr
    compressor = Blosc(cname="zstd", clevel=4, shuffle=Blosc.SHUFFLE)
    root = zarr.open_group(zarr_out, mode='w')
    target_dtype = np.float32 if dtype_str == 'float32' else np.float16

    # 先统计 supervised 的 labels/type_name 数组（边写边填也可，这里一次创建）
    # 我们需要先知道 N_total = N_sup + N_test
    N_total = N_sup + N_test
    if N_total == 0:
        raise RuntimeError("没有可写入的样本。")

    # 创建数据集（inputs/outputs/labels/type_name/input_file）
    # 注意：outputs 仅对有监督区间写入；test 区间不写（保持未定义）
    inputs_ds = root.create_dataset(
        'inputs', shape=(N_total, 5, 1000, 70), chunks=(args.chunks, 5, 1000, 70),
        dtype=target_dtype, compressor=compressor
    )
    # outputs 可选：如果你 **确定** 所有 sup 都有标签，直接创建；
    outputs_ds = root.create_dataset(
        'outputs', shape=(N_total, 70, 70), chunks=(args.chunks, 70, 70),
        dtype=target_dtype, compressor=compressor
    )
    labels_ds = root.create_dataset('labels', shape=(N_total,), chunks=(max(1,args.chunks),), dtype='i8', compressor=compressor)
    # 可变长 utf-8
    type_name_ds = root.create_dataset('type_name', shape=(N_total,), dtype=object, object_codec=zarr.codecs.VLenUTF8())
    input_file_ds = root.create_dataset('input_file', shape=(N_total,), dtype=object, object_codec=zarr.codecs.VLenUTF8())

    # 写入 supervised 区间
    write_ptr = 0
    print("写入有监督样本到 Zarr ……")
    for in_path, out_path, type_key, tag in tqdm(sup_records):
        x_arr = np.load(in_path, mmap_mode='r')   # [M, 5, 1000, 70] 或 [M, C, H, W]
        y_arr = np.load(out_path, mmap_mode='r')  # [M, 70, 70] 或 [M, 1, 70, 70]
        M = int(x_arr.shape[0])

        # 规范形状
        if x_arr.shape[1:] != (5, 1000, 70):
            raise ValueError(f"{in_path} 的 shape 后 3 维应为 [5,1000,70]，实际 {x_arr.shape[1:]}")
        if y_arr.ndim == 4 and y_arr.shape[1] == 1:
            y_arr = y_arr[:,0]  # [M,70,70]
        if y_arr.shape[1:] != (70,70):
            raise ValueError(f"{out_path} 的 shape 后 2 维应为 [70,70]，实际 {y_arr.shape[1:]}")

        lab = int(type_id_map[type_key])

        # 批量写入
        sl = slice(write_ptr, write_ptr+M)
        inputs_ds[sl] = x_arr.astype(target_dtype, copy=False)
        outputs_ds[sl] = y_arr.astype(target_dtype, copy=False)
        labels_ds[sl] = lab
        type_name_ds[sl] = [type_key]*M
        # 给每条样本一个可回溯的 tag：文件名#样本索引 的感觉；这里仅写文件名，必要可再细化
        input_file_ds[sl] = [tag]*M

        write_ptr += M

    sup_end = write_ptr

    # 写入无标签 test（只写 inputs 与元信息；labels/outputs 留空）
    if N_test > 0:
        print("写入无标签测试样本到 Zarr ……")
        for in_path in tqdm(test_files):
            x_arr = np.load(in_path, mmap_mode='r')  # [M,5,1000,70]
            M = int(x_arr.shape[0])
            if x_arr.shape[1:] != (5, 1000, 70):
                raise ValueError(f"{in_path} 的 shape 后 3 维应为 [5,1000,70]，实际 {x_arr.shape[1:]}")
            sl = slice(write_ptr, write_ptr+M)
            inputs_ds[sl] = x_arr.astype(target_dtype, copy=False)
            # 对于 test：labels 写 -1，占位；type_name 写 "test"; outputs 不写
            labels_ds[sl] = -1
            type_name_ds[sl] = ["test"]*M
            input_file_ds[sl] = [f"test/{os.path.basename(in_path)}"]*M
            write_ptr += M

    assert write_ptr == N_total

    # ---- splits：仅对有监督样本做 train/val/test 划分；无标签 test 另行放在 splits/test_idx 的“无监督区”
    sup_labels = np.asarray(labels_ds[:sup_end], dtype=np.int64)
    train_idx, val_idx, test_idx = stratified_split_indices(sup_labels, ratios=ratios, seed=args.seed)

    splits = root.create_group('splits')
    splits.create_dataset('train_idx', data=train_idx.astype(np.int64), compressor=compressor)
    splits.create_dataset('val_idx',   data=val_idx.astype(np.int64),   compressor=compressor)
    splits.create_dataset('test_idx',  data=test_idx.astype(np.int64),  compressor=compressor)

    # 若存在“无标签 test 区间”，另外保存它们的索引段（可选）
    if N_test > 0:
        unsup_idx = np.arange(sup_end, N_total, dtype=np.int64)
        splits.create_dataset('unsup_test_idx', data=unsup_idx, compressor=compressor)

    # 元信息写到 .zattrs
    root.attrs.update({
        "schema": {
            "inputs":  {"shape": [N_total, 5, 1000, 70], "dtype": dtype_str},
            "outputs": {"shape": [N_total, 70, 70], "dtype": dtype_str},
            "labels":  {"shape": [N_total], "dtype": "int64"},
            "type_name": "vlen-utf8",
            "input_file": "vlen-utf8",
        },
        "counts": {
            "N_total": int(N_total),
            "N_supervised": int(N_sup),
            "N_test_inputs": int(N_test)
        },
        "chunks": int(args.chunks),
        "compressor": "blosc-zstd",
        "split_ratio": ratios,
        "seed": int(args.seed),
        "note": "Built from OpenFWI-style folders."
    })

    print(f"  完成：Zarr 写入 {zarr_out}")
    print(f"  监督样本: {N_sup}  | 无标签测试样本: {N_test}")
    print(f"  splits/train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    if N_test > 0:
        print(f"  splits/unsup_test_idx: {len(unsup_idx)}")

if __name__ == "__main__":
    main()
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
# Naming and type inference
# ---------------------------

def to_snake_lower(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    # CamelCase -> snake_case boundaries
    s = re.sub(r'([A-Z]+)([A-Z][a-z0-9])', r'\1_\2', s)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub(r'[^A-Za-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_').lower()
    parts = [p for p in s.split('_') if p]
    if not parts:
        return s
    if parts[0] == 'style':
        # Match legacy naming: collapse style subtypes; only a/b differ
        suffix = parts[-1] if len(parts) >= 2 else 'a'
        if len(parts) >= 3 and parts[1] == 'style':
            return '_'.join(parts[:3])  # style_style_a/b
        return f"style_style_{suffix}"
    return '_'.join(parts)

def infer_group_and_variant(subdir_name: str):
    # Returns (group, variant, norm_type_key)
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
    raise ValueError(f"Cannot infer family from type_key='{type_key}'")

# Fault seismogram filename regex
FAULT_SEIS_RE = re.compile(r"seis_?(\d+)_1_(\d+)\.npy$")


# ---------------------------
# Scan train_samples/ into a manifest of tuples:
#   (input_path, output_path or None, type_key, input_file_tag)
# ---------------------------

def collect_supervised_samples(train_dir: str,
                               type_id_map: Dict[str, int],
                               expect_ch: int = 5) -> List[Tuple[str, Optional[str], str, str]]:
    if not os.path.isdir(train_dir):
        raise RuntimeError(f"Training directory does not exist: {train_dir}")

    subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    records = []

    for sub in sorted(subdirs):
        group, variant, type_key = infer_group_and_variant(sub)
        if group is None:
            continue
        if not (type_key.endswith('_a') or type_key.endswith('_b')):
            raise ValueError(f"Directory '{sub}' normalizes to '{type_key}' but must end with _a/_b.")
        if type_key not in type_id_map:
            raise KeyError(f"type_key '{type_key}' is missing from type_id_map.")

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
        raise RuntimeError("No supervised samples found under train_samples/.")
    return records

def collect_test_inputs(test_dir: str) -> List[str]:
    if not os.path.isdir(test_dir):
        return []
    return sorted(glob.glob(os.path.join(test_dir, '*.npy')))


# ---------------------------
# Total sample count (sum of leading dimension per file)
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
# Per-subtype fixed train/val split ratios
# ---------------------------

def get_val_ratio_for_family(family: str) -> float:
    family = family.lower()
    if family == 'vel':
        return 6.0 / 30.0
    if family == 'fault':
        return 6.0 / 54.0
    if family == 'style':
        return 7.0 / 67.0
    raise ValueError(f"Unknown family: {family}")

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
    type_names: np.ndarray,  # type_name strings for supervised rows only
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
# Main entry: materialize Zarr
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Build seismic_moe.zarr from OpenFWI-style folders")
    ap.add_argument('--data_dir', required=True, help='Dataset root with train_samples/ and optional test/')
    ap.add_argument('--zarr_out', required=True, help='Output Zarr directory (created if missing)')
    ap.add_argument('--chunks', type=int, default=32, help='Chunk size along sample dimension')
    ap.add_argument('--dtype', type=str, default='float32', choices=['float32','float16'], help='Storage dtype')
    ap.add_argument('--seed', type=int, default=42)

    # Single-family controls
    ap.add_argument('--family', type=str, default='all',
                    help="'all' or a fine-grained key (e.g. curve_vel_a / flat_vel_b / style_a)")
    ap.add_argument('--include_test', type=int, default=0,
                    help="If family != 'all', include label-free test/*.npy rows (0/1). Ignored when family='all'.")
    ap.add_argument('--remap_single_label', type=int, default=1,
                    help="If family != 'all', remap that class label to 0 (0/1, default 1).")

    ap.add_argument('--concat_channels', type=int, default=1,
                    help="Concatenate [5,1000,70] -> [1,1000,350] along last dim (1=yes, 0=no)")

    args = ap.parse_args()

    data_dir   = args.data_dir
    zarr_out   = args.zarr_out
    dtype_str  = args.dtype
    do_concat  = bool(args.concat_channels)

    config = SeismicMOEConfig()
    type_id_map = config.type_id_specific

    # Resolve family string
    family_arg = to_snake_lower(args.family) if args.family.lower() != 'all' else 'all'
    if family_arg != 'all':
        if not (family_arg.endswith('_a') or family_arg.endswith('_b')):
            raise ValueError(f"--family '{args.family}' must end with _a/_b (e.g. curve_vel_a / style_a).")
        if family_arg not in type_id_map:
            raise KeyError(f"--family '{args.family}' normalizes to '{family_arg}' but is absent from type_id_map.")

    train_dir = os.path.join(data_dir, 'train_samples')
    test_dir  = os.path.join(data_dir, 'test')

    print("Scanning supervised samples...")
    sup_records_all = collect_supervised_samples(train_dir, type_id_map)

    # Filter by family
    if family_arg == 'all':
        sup_records = sup_records_all
    else:
        sup_records = [rec for rec in sup_records_all if rec[2] == family_arg]
        if not sup_records:
            raise RuntimeError(f"No supervised samples for family='{family_arg}'.")

    N_sup_files = len(sup_records)
    N_sup = count_total_samples_records(sup_records)
    print(f"family = {family_arg} | supervised file pairs: {N_sup_files}, total samples ≈ {N_sup}")

    # Optional unlabeled test split
    if family_arg == 'all':
        test_files = collect_test_inputs(test_dir)
        N_test = count_total_samples_files(test_files) if test_files else 0
    else:
        if args.include_test:
            test_files = collect_test_inputs(test_dir)
            N_test = count_total_samples_files(test_files) if test_files else 0
            print(f"[include_test=1] unlabeled test inputs: {len(test_files)} files, total samples ≈ {N_test}")
        else:
            test_files = []
            N_test = 0

    # Allocate Zarr datasets
    compressor = Blosc(cname="zstd", clevel=4, shuffle=Blosc.SHUFFLE)
    root = zarr.open_group(zarr_out, mode='w')
    target_dtype = np.float32 if dtype_str == 'float32' else np.float16

    # Total rows
    N_total = N_sup + N_test
    if N_total == 0:
        raise RuntimeError("No samples to write.")

    # Input tensor layout
    if do_concat:
        inputs_shape = (N_total, 1, 1000, 350)
        inputs_chunks = (args.chunks, 1, 1000, 350)
        inputs_note = "Inputs concatenated to [1,1000,350]"
    else:
        inputs_shape = (N_total, 5, 1000, 70)
        inputs_chunks = (args.chunks, 5, 1000, 70)
        inputs_note = "Inputs kept as [5,1000,70] (no concatenation)"

    # Create datasets
    inputs_ds = root.create_dataset(
        'inputs', shape=inputs_shape, chunks=inputs_chunks,
        dtype=target_dtype, compressor=compressor
    )
    outputs_ds = root.create_dataset(
        'outputs', shape=(N_total, 1, 70, 70), chunks=(args.chunks, 1, 70, 70),
        dtype=target_dtype, compressor=compressor
    )
    labels_ds = root.create_dataset('labels', shape=(N_total,), chunks=(max(1,args.chunks),), dtype='int64', compressor=compressor)

    # Variable-length UTF-8 metadata
    type_name_ds = root.create_dataset('type_name', shape=(N_total,), dtype=object, object_codec=zarr.codecs.VLenUTF8())
    input_file_ds = root.create_dataset('input_file', shape=(N_total,), dtype=object, object_codec=zarr.codecs.VLenUTF8())

    # Write supervised rows
    write_ptr = 0
    print("Writing supervised rows to Zarr...")
    for in_path, out_path, type_key, tag in tqdm(sup_records):
        x_arr = np.load(in_path, mmap_mode='r')   # expected [M,5,1000,70] or stacked [M,1,1000,350]
        y_arr = np.load(out_path, mmap_mode='r')  # [M,70,70] or [M,1,70,70]

        M = int(x_arr.shape[0])

        # Normalize input layout
        if x_arr.ndim != 4:
            raise ValueError(f"{in_path} expected 4D tensor, got shape {x_arr.shape}")

        if do_concat:
            # target [M,1,1000,350]
            if x_arr.shape[1:] == (5, 1000, 70):
                # per-sample concat along receiver axis
                concatenated = np.empty((M, 1000, 350), dtype=target_dtype)
                for j in range(M):
                    input_data = np.concatenate([x_arr[j][k] for k in range(5)], axis=1)  # [1000,350]
                    concatenated[j] = input_data.astype(target_dtype, copy=False)
                x_cat = concatenated[:, None, :, :]  # [M,1,1000,350]
            elif x_arr.shape[1:] == (1, 1000, 350):
                x_cat = x_arr.astype(target_dtype, copy=False)
            else:
                raise ValueError(f"{in_path} unsupported input shape {x_arr.shape[1:]}; expect [5,1000,70] or [1,1000,350]")
        else:
            # target [M,5,1000,70]
            if x_arr.shape[1:] == (5, 1000, 70):
                x_cat = x_arr.astype(target_dtype, copy=False)
            elif x_arr.shape[1:] == (1, 1000, 350):
                # split stacked tensor back to 5 sources
                if x_arr.shape[3] != 350:
                    raise ValueError(f"{in_path} width {x_arr.shape[3]} cannot reshape to 5×70")
                # [M,1,1000,350] -> [M,5,1000,70]
                reshaped = x_arr.reshape(M, 1, 1000, 5, 70)   # [M,1,1000,5,70]
                x_cat = reshaped[:, 0].transpose(0, 1, 2, 3).astype(target_dtype, copy=False)  # [M,5,1000,70]
            else:
                raise ValueError(f"{in_path} unsupported input shape {x_arr.shape[1:]}; expect [5,1000,70] or [1,1000,350]")

        # Normalize velocity maps to [M,1,70,70]
        if y_arr.ndim == 4 and y_arr.shape[1:] == (1, 70, 70):
            y_cat = y_arr.astype(target_dtype, copy=False)
        elif y_arr.ndim == 3 and y_arr.shape[1:] == (70, 70):
            y_cat = y_arr[:, None, :, :].astype(target_dtype, copy=False)  # [M,1,70,70]
        else:
            raise ValueError(f"{out_path} unsupported label shape {y_arr.shape}; expect [M,70,70] or [M,1,70,70]")

        # Integer labels: global id for multi-family; optional remap for single-family runs
        if family_arg == 'all':
            lab_val = int(type_id_map[type_key])
        else:
            lab_val = 0 if args.remap_single_label else int(type_id_map[type_key])

        sl = slice(write_ptr, write_ptr + M)
        inputs_ds[sl] = x_cat
        outputs_ds[sl] = y_cat
        labels_ds[sl] = lab_val

        # Broadcast scalars so Zarr chunk writes stay aligned
        type_name_ds[sl] = type_key
        input_file_ds[sl] = tag
        # =================================================================

        write_ptr += M

    sup_end = write_ptr

    # Append unlabeled test rows (labels = -1, type_name = "test")
    if N_test > 0:
        print("Writing unlabeled test rows to Zarr...")
        for in_path in tqdm(test_files):
            x_arr = np.load(in_path, mmap_mode='r')  # [M,5,1000,70] or [M,1,1000,350]
            M = int(x_arr.shape[0])

            if do_concat:
                if x_arr.shape[1:] == (5, 1000, 70):
                    concatenated = np.empty((M, 1000, 350), dtype=target_dtype)
                    for j in range(M):
                        input_data = np.concatenate([x_arr[j][k] for k in range(5)], axis=1)
                        concatenated[j] = input_data.astype(target_dtype, copy=False)
                    x_cat = concatenated[:, None, :, :]  # [M,1,1000,350]
                elif x_arr.shape[1:] == (1, 1000, 350):
                    x_cat = x_arr.astype(target_dtype, copy=False)
                else:
                    raise ValueError(f"{in_path} unsupported input shape {x_arr.shape[1:]}; expect [5,1000,70] or [1,1000,350]")
            else:
                if x_arr.shape[1:] == (5, 1000, 70):
                    x_cat = x_arr.astype(target_dtype, copy=False)
                elif x_arr.shape[1:] == (1, 1000, 350):
                    if x_arr.shape[3] != 350:
                        raise ValueError(f"{in_path} width {x_arr.shape[3]} cannot reshape to 5×70")
                    reshaped = x_arr.reshape(M, 1, 1000, 5, 70)
                    x_cat = reshaped[:, 0].transpose(0, 1, 2, 3).astype(target_dtype, copy=False)
                else:
                    raise ValueError(f"{in_path} unsupported input shape {x_arr.shape[1:]}; expect [5,1000,70] or [1,1000,350]")

            sl = slice(write_ptr, write_ptr + M)
            inputs_ds[sl] = x_cat
            labels_ds[sl] = -1

            # Scalar broadcast for metadata
            type_name_ds[sl] = "test"
            input_file_ds[sl] = f"test/{os.path.basename(in_path)}"
            # =================================

            write_ptr += M

    assert write_ptr == N_total

    # Train/val indices (fixed ratio per subtype); no held-out test split here
    sup_types = np.asarray(type_name_ds[:sup_end], dtype=object)

    if family_arg == 'all':
        # Each subtype uses its family-specific val ratio, then indices are merged
        train_idx, val_idx = split_by_subtype_fixed_ratio(
            type_names=sup_types, seed=args.seed
        )
    else:
        # Single subtype: apply that family's val ratio directly
        fam = family_from_type_key(family_arg)
        val_ratio = get_val_ratio_for_family(fam)
        all_sup_indices = np.arange(sup_end, dtype=np.int64)
        train_idx, val_idx = split_fixed_ratio_indices(
            all_sup_indices, val_ratio=val_ratio, seed=args.seed
        )

    test_idx = np.array([], dtype=np.int64)  # supervised test split unused here

    splits = root.create_group('splits')
    splits.create_dataset('train_idx', data=train_idx.astype(np.int64), compressor=compressor)
    splits.create_dataset('val_idx',   data=val_idx.astype(np.int64),   compressor=compressor)
    splits.create_dataset('test_idx',  data=test_idx.astype(np.int64),  compressor=compressor)

    # Optional index range for appended unlabeled test rows
    if N_test > 0:
        unsup_idx = np.arange(sup_end, N_total, dtype=np.int64)
        splits.create_dataset('unsup_test_idx', data=unsup_idx, compressor=compressor)

    # Serialize metadata to root attrs
    type_id_map_attr = {str(k): int(v) for k, v in type_id_map.items()}
    id_type_map_attr = {int(v): str(k) for k, v in type_id_map.items()}

    root.attrs.update({
        "schema": {
            "inputs":  {"shape": list(inputs_shape), "dtype": dtype_str},
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
        "note": f"{inputs_note}; outputs expanded to [1,70,70].",
        "concat_channels": do_concat
    })

    print(f"  Done writing Zarr to {zarr_out}")
    print(f"  family: {family_arg} | supervised rows: {sup_end} | unlabeled test rows: {N_test}")
    print(f"  splits/train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    if N_test > 0:
        print(f"  splits/unsup_test_idx: {len(unsup_idx)}")

if __name__ == "__main__":
    main()
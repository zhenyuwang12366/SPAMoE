import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import re
from pathlib import Path
import torch.nn.functional as F
from config.seismic_moe_config import SeismicMOEConfig, SPECIFIC_TYPE_VARIANTS

# ---------------------------
# SPECIFIC family mapping (preserves original logic)
# ---------------------------
_SPECIFIC_VARIANT_TO_BASE = {
    variant: base
    for base, variants in SPECIFIC_TYPE_VARIANTS.items()
    for variant in variants
}
_SPECIFIC_BASE_FAMILIES = set(SPECIFIC_TYPE_VARIANTS.keys())
_SPECIFIC_VARIANT_FAMILIES = set(_SPECIFIC_VARIANT_TO_BASE.keys())
_ALLOWED_SPECIFIC_FAMILIES = _SPECIFIC_BASE_FAMILIES | _SPECIFIC_VARIANT_FAMILIES

# ---------------------------
# Directory name normalization / extraction
# ---------------------------
def _to_snake_lower(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    # CamelCase -> underscore boundaries first
    s = re.sub(r'([A-Z]+)([A-Z][a-z0-9])', r'\1_\2', s)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub(r'[^A-Za-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_').lower()
    parts = [p for p in s.split('_') if p]  # drop empty segments
    if not parts:
        return s
    if parts[0] == 'style':
        # Case 1: style_a -> style_style_a
        # Case 2: style_style_a -> unchanged
        # Other (e.g. style_xxx_a) -> use last segment as a/b suffix
        suffix = parts[-1] if len(parts) >= 2 else 'a'
        if len(parts) >= 3 and parts[1] == 'style':
            return '_'.join(parts[:3])  # already style_style_a/b
        return f"style_style_{suffix}"
    return '_'.join(parts)

def _infer_type_dir_name_from_input_file(p: str) -> str:
    """
    Infer the type directory name from an input file path:
      vel/style: .../<TypeDir>/data/dataXX.npy -> <TypeDir>
      fault:     .../<TypeDir>/seis_...npy    -> <TypeDir>
    """
    path = Path(p)
    parent = path.parent.name
    if parent in ("data", "model"):
        return path.parent.parent.name
    return path.parent.name


class SeismicDataset(Dataset):
    """
    Seismic dataset for OpenFWI-style data.
    - Reads vel/style (data/model pairs) and fault (seis/vel pairs).
    - Infers labels from directory names (strict *_a / *_b suffix), mapped via config.type_id_specific.
    - __getitem__ returns input/output plus v_type and type_name.
    """
    def __init__(
        self,
        data_dir: str,
        family: str = 'all',
        is_specific: bool = False,
        split: str = 'train',
        concat_channels: bool = False,
        config: Optional[SeismicMOEConfig] = None,   # for type_id_specific mapping
        processor: Optional[object] = None,          # optional data processor
    ):
        self.data_dir = data_dir
        self.family = family.lower()
        self.split = split
        self.index_map: List[Tuple[int, int]] = []
        self.input_tensors: List[torch.Tensor] = []
        self.output_tensors: List[torch.Tensor] = []
        self.is_specific = is_specific
        self.concat_channels = bool(concat_channels)
        self._specific_target_family: Optional[str] = None

        # Persist config / processor / label containers
        if config is None:
            raise ValueError("SeismicDataset requires config (for lower-case type_id_specific mapping)")
        self.config: SeismicMOEConfig = config
        self.processor = processor
        self.labels: List[int] = []
        self.type_names: List[str] = []

        # Validate parameters (original behavior)
        if not is_specific:
            if self.family not in ['vel', 'style', 'fault', 'all']:
                raise ValueError(f"Unsupported dataset family: {self.family}")
        else:
            if self.family not in _ALLOWED_SPECIFIC_FAMILIES:
                raise ValueError(
                    f"Unsupported specific family: {self.family}. Allowed: {sorted(_ALLOWED_SPECIFIC_FAMILIES)}"
                )
            self._specific_target_family = _SPECIFIC_VARIANT_TO_BASE.get(self.family, self.family)

        if self.split not in ['train', 'test']:
            raise ValueError(f"Unsupported split: {self.split}")

        # Discover file paths
        self._load_data()

        # Build (file_idx, sample_idx) index
        for file_idx, file in enumerate(self.input_files):
            data = np.load(file, mmap_mode='r')  # lazy load
            num_samples = data.shape[0]
            for sample_idx in range(num_samples):
                self.index_map.append((file_idx, sample_idx))

        # Preload into memory (labels aligned)
        self._preload_into_memory()

        # Summary statistics
        self._compute_stats()

    def _preload_into_memory(self):
        # Lower-case key mapping
        type_id_map: Dict[str, int] = getattr(self.config, "type_id_specific", None)
        if not isinstance(type_id_map, dict) or not type_id_map:
            raise ValueError("config.type_id_specific must be a non-empty dict (keys lower-case).")

        for i, input_path in enumerate(self.input_files):
            try:
                input_array = np.load(input_path, allow_pickle=True)
                output_array = None
                if self.output_files[i] is not None:
                    output_array = np.load(self.output_files[i], allow_pickle=True)

                # Directory name -> normalize -> check a/b suffix -> map to id
                type_dir_raw = _infer_type_dir_name_from_input_file(input_path)   # e.g., FlatFault_a
                type_key = _to_snake_lower(type_dir_raw)                          # e.g., flat_fault_a

                if not (type_key.endswith('_a') or type_key.endswith('_b')):
                    raise ValueError(
                        f"Directory name '{type_dir_raw}' normalizes to '{type_key}' but does not end with '_a' or '_b'. "
                        "Only a/b suffixes are allowed (no numeric suffixes)."
                    )
                if type_key not in type_id_map:
                    raise KeyError(
                        f"Directory name '{type_dir_raw}' -> '{type_key}' is missing from config.type_id_specific. "
                        f"(example keys: {list(type_id_map.keys())[:6]} ... total {len(type_id_map)} entries)"
                    )
                label_id = int(type_id_map[type_key])

                num_samples = input_array.shape[0]
                for j in range(num_samples):
                    if self.concat_channels:
                        input_data = np.concatenate([input_array[j][k] for k in range(5)], axis=1)
                        input_tensor = torch.from_numpy(input_data.astype(np.float32)).unsqueeze(0)
                    else:    
                        input_tensor = torch.from_numpy(input_array[j])
                    self.input_tensors.append(input_tensor)

                    if output_array is not None:
                        output_tensor = torch.from_numpy(output_array[j].astype(np.float32))
                        self.output_tensors.append(output_tensor)

                    # Per-sample label / class name
                    self.labels.append(label_id)
                    self.type_names.append(type_key)

            except Exception as e:
                print(f"Failed to read file {i}: {e}")

    def _load_data(self):
        """
        Load input/output file paths.
        """
        self.input_files = []
        self.output_files = []

        def want_family(group: str, variant: Optional[str]) -> bool:
            """Return whether this subdirectory should be kept given family / is_specific."""
            target_family = self._specific_target_family or self.family
            if not getattr(self, 'is_specific', False):
                if self.family == 'all':
                    return True
                return self.family == group
            else:
                if group == 'style':
                    return target_family == 'style_style'
                if variant is None:
                    return False
                return target_family == f"{variant}_{group}"

        if self.split == 'train':
            train_dir = os.path.join(self.data_dir, 'train_samples')
            if not os.path.isdir(train_dir):
                raise RuntimeError(f"Training directory not found: {train_dir}")

            subdirs = [d for d in os.listdir(train_dir)
                    if os.path.isdir(os.path.join(train_dir, d))]

            for sub in sorted(subdirs):
                sub_path = os.path.join(train_dir, sub)

                # Map subdirectory name to (group, variant)
                group = None
                variant = None
                if sub.startswith("CurveFault_"):
                    group, variant = 'fault', 'curve'
                elif sub.startswith("FlatFault_"):
                    group, variant = 'fault', 'flat'
                elif sub.startswith("FlatVel_"):
                    group, variant = 'vel', 'flat'     
                elif sub.startswith("Style_"):
                    group, variant = 'style', None
                elif sub.startswith("CurveVel_"):
                    group, variant = 'vel', 'curve'
                else:
                    # Unknown naming; skip (could raise instead)
                    continue

                if not want_family(group, variant):
                    continue

                if group == 'fault':
                    # Fault: seis_?{n}_1_{i}.npy <-> vel_{n}_1_{i}.npy
                    pattern = re.compile(r"seis_?(\d+)_1_(\d+)\.npy")
                    seis_files = sorted(glob.glob(os.path.join(sub_path, 'seis*.npy')))
                    for seis_file in seis_files:
                        stem = Path(seis_file).name
                        m = pattern.fullmatch(stem)
                        if m:
                            n = int(m.group(1))
                            i = int(m.group(2))
                        else:
                            print(f"{stem} skipped")
                            continue
                        try:
                            vel_file = os.path.join(sub_path, f"vel_{n}_1_{i}.npy")
                            assert os.path.exists(vel_file)
                        except Exception:
                            vel_file = os.path.join(sub_path, f"vel{n}_1_{i}.npy")
                        if os.path.exists(vel_file):
                            self.input_files.append(seis_file)   # input: seis
                            self.output_files.append(vel_file)   # output: vel

                elif group in ('vel', 'style'):
                    # Vel/Style: data/{data{i}.npy} <-> model/{model{i}.npy}
                    data_dir = os.path.join(sub_path, 'data')
                    model_dir = os.path.join(sub_path, 'model')
                    if not (os.path.isdir(data_dir) and os.path.isdir(model_dir)):
                        continue

                    data_files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
                    for data_file in data_files:
                        base = os.path.basename(data_file)   # data{i}.npy
                        stem, _ = os.path.splitext(base)
                        if not stem.startswith('data'):
                            continue
                        idx = stem.replace('data', '')
                        model_file = os.path.join(model_dir, f"model{idx}.npy")
                        if os.path.exists(model_file):
                            self.input_files.append(data_file)   # input: data
                            self.output_files.append(model_file) # output: model

        else:
            # test split: inputs only
            test_dir = os.path.join(self.data_dir, 'test')
            if not os.path.isdir(test_dir):
                raise RuntimeError(f"Test directory not found: {test_dir}")
            self.input_files = sorted(glob.glob(os.path.join(test_dir, '*.npy')))
            self.output_files = [None] * len(self.input_files)

        # Consistency checks
        if not self.input_files:
            raise RuntimeError(
                f"No input files found. Check path and filters: "
                f"data_dir={self.data_dir}, split={self.split}, "
                f"family={self.family}, is_specific={getattr(self, 'is_specific', None)}"
            )
        if self.split == 'train' and len(self.input_files) != len(self.output_files):
            raise RuntimeError(
                f"Input/output file count mismatch: inputs={len(self.input_files)}, outputs={len(self.output_files)}"
            )
    
    def _compute_stats(self):
        """Compute normalization stats from in-memory tensors."""
        n_total = int(len(self.input_tensors))
        if n_total == 0:
            return
        n_samples = int(min(max(1, n_total * 0.03), 300))  # 3% of data, cap at 300
        sample_indices = np.random.choice(n_total, n_samples, replace=False)

        # Input stats
        input_values = []
        for idx in sample_indices:
            try:
                input_tensor = self.input_tensors[idx]  # shape: [C, H, W]
                flat = input_tensor.view(-1)  # flatten to 1D
                sample_points = flat[torch.randperm(flat.numel())[:min(1000, flat.numel())]]
                input_values.append(sample_points)
            except Exception as e:
                print(f"Warning: failed to process input sample {idx}: {e}")

        if input_values:
            input_all = torch.cat(input_values)
            self.input_min = float(torch.min(input_all))
            self.input_max = float(torch.max(input_all))
            self.input_mean = float(torch.mean(input_all))
            self.input_std = float(torch.std(input_all)) or 1.0

        # Output stats (train only)
        if self.split == 'train' and hasattr(self, 'output_tensors') and self.output_tensors:
            output_values = []
            for idx in sample_indices:
                try:
                    output_tensor = self.output_tensors[idx]  # shape: [C_out, H_out, W_out]
                    flat = output_tensor.view(-1)
                    sample_points = flat[torch.randperm(flat.numel())[:min(1000, flat.numel())]]
                    output_values.append(sample_points)
                except Exception as e:
                    print(f"Warning: failed to process output sample {idx}: {e}")

            if output_values:
                output_all = torch.cat(output_values)
                self.output_min = float(torch.min(output_all))
                self.output_max = float(torch.max(output_all))
                self.output_mean = float(torch.mean(output_all))
                self.output_std = float(torch.std(output_all)) or 1.0
            
    def getStats(self):
        return {
            'input_max' : getattr(self, 'input_max', None),
            'input_min' : getattr(self, 'input_min', None),
            'input_mean' : getattr(self, 'input_mean', None),
            'input_std' : getattr(self, 'input_std', None),
            'output_max' : getattr(self, 'output_max', None),
            'output_min' : getattr(self, 'output_min', None),
            'output_mean' : getattr(self, 'output_mean', None),
            'output_std' : getattr(self, 'output_std', None),
        }
        
    def __len__(self):
        return len(self.input_tensors)
         
    def __getitem__(self, idx):
        try:
            file_idx, _  = self.index_map[idx]
            input_filename = os.path.basename(self.input_files[file_idx])
            
            if self.split == 'train':  # train / val with labels
                sample = {
                    'input': self.input_tensors[idx].clone(),
                    'output': self.output_tensors[idx].clone(),
                    'v_type': torch.tensor(self.labels[idx], dtype=torch.long),  # class id
                    'type_name': self.type_names[idx],                            # lower_snake class name
                    'input_file': input_filename
                }
            else:  # test
                sample = {
                    'input': self.input_tensors[idx].clone(),
                    'v_type': torch.tensor(self.labels[idx], dtype=torch.long),
                    'type_name': self.type_names[idx],
                    'input_file': input_filename.split('.')[0]
                }

            # Optional processor (resize / normalize, etc.)
            if self.processor is not None:
                sample = self.processor(sample)
            return sample
            
        except Exception as e:
            print(f"Failed to load sample {idx}: {e}")
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(0)
    
    def get_input_size(self):
        if len(self.input_tensors) == 0:
            return None
        return tuple(self.input_tensors[0].shape)
    
    def get_output_size(self):
        if self.split == 'test' or len(self.output_tensors) == 0:
            return None
        return tuple(self.output_tensors[0].shape)


class SeismicDataProcessor:
    """
    Seismic data processor: channel reordering, optional resize/pad, external transforms.
    """
    def __init__(
        self,
        channel_dim: int = 0,
        input_transform : Optional[callable] = None,
        output_transform : Optional[callable] = None,
        config: SeismicMOEConfig = None,
    ):
        assert config is not None, "please input config in data_processor"
        self.config = config
        self.channel_dim = channel_dim
        self.input_transform = input_transform
        self.output_transform = output_transform

    def _flexible_resize(
        self,
        x: torch.Tensor,
        keep: bool = False,
        size: Optional[Tuple[int, int]] = None,
        mode: str = "bilinear",
        align_corners: Optional[bool] = False,
        antialias: bool = True,
        auto_pad: bool = True,
        pad_mode: str = "reflect",
    ) -> torch.Tensor:
        """
        Optional auto-pad + resize helper.
        Input x: (C, H, W)  -> temporarily expanded to (1, C, H, W) internally.
        """
        x = x.unsqueeze(0)
        if keep:
            return x.squeeze(0)
        _, C, H, W = x.shape
        if auto_pad and H != W:
            M = max(H, W)
            pad_h = M - H
            pad_w = M - W
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad = (pad_left, pad_right, pad_top, pad_bottom)
            x = F.pad(x, pad, mode=pad_mode)

        if size is None:
            return x.squeeze(0)

        if x.shape[-2:] == tuple(size):
            return x.squeeze(0)

        _align = align_corners if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None
        x = F.interpolate(x, size=size, mode=mode, align_corners=_align, antialias=antialias)
        return x.squeeze(0)
    
    def __call__(self, sample: Dict):
        # Input branch
        if 'input' in sample and sample['input'] is not None:
            x = sample['input']  # (C,H,W)
            if x.ndim == 2:
                x = x.unsqueeze(0)

            # Channel reorder (extend here if needed)
            if self.channel_dim == 1:
                # (C,H,W) unchanged here
                pass
            elif self.channel_dim == -1:
                pass

            if self.input_transform:
                x = self.input_transform(x)

            # Optional resize
            resize_enabled = bool(getattr(self.config, 'is_resize', 0))
            target_h = getattr(self.config, 'H_size', None)
            target_w = getattr(self.config, 'W_size', None)
            if resize_enabled:
                if target_h is None or target_w is None:
                    target_size = tuple(x.shape[2:])
                else:
                    target_size = (int(target_h), int(target_w))
                x = self._flexible_resize(
                    x, keep=False, size=target_size,
                    mode="bilinear", align_corners=True,
                    antialias=True, auto_pad=True, pad_mode="reflect",
                )
            sample['input'] = x

        # Output branch
        if 'output' in sample and sample['output'] is not None:
            y = sample['output']
            if y.ndim == 2:
                y = y.unsqueeze(0)
            if self.output_transform:
                y = self.output_transform(y)
            sample['output'] = y
        
        return sample

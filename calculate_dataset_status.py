#!/usr/bin/env python3
"""Compute per-type and overall normalization statistics for the seismic dataset.

The script expects the dataset layout illustrated in the project README. It scans
`train_samples/` under the provided dataset root, gathers all `.npy` files for inputs
and outputs, and writes summary statistics (min, max, mean, std) to a JSON file.

Enhancements:
- Normalize type names to lower_snake_case via `to_snake_lower`.
- Show progress bars using `tqdm` if available (fallback to simple prints otherwise).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
from utils.utils import to_snake_lower

# --------------------------
# Progress-bar utilities
# --------------------------
def _get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


def _simple_progress(total: int) -> Tuple[Callable[[int], None], Callable[[], None]]:
    """Fallback progress printer when tqdm is not available."""
    # Print every ~5% progress or at least every 50 files
    step = max(1, total // 20, 50)

    state = {"n": 0}

    def update(n: int = 1):
        state["n"] += n
        if state["n"] % step == 0 or state["n"] >= total:
            pct = (state["n"] / total * 100) if total > 0 else 100.0
            print(f"[Progress] {state['n']}/{total} files ({pct:.1f}%)", file=sys.stderr)

    def close():
        pass

    return update, close


# --------------------------
# Stats computation classes
# --------------------------
class StatsAccumulator:
    """Streaming statistics tracker to avoid storing all samples in memory."""

    __slots__ = ("count", "sum", "sumsq", "min", "max")

    def __init__(self) -> None:
        self.count: int = 0
        self.sum: float = 0.0
        self.sumsq: float = 0.0
        self.min: float | None = None
        self.max: float | None = None

    def update(self, array: np.ndarray) -> None:
        if array.size == 0:
            return
        data = array.astype(np.float64, copy=False)
        arr_min = float(data.min())
        arr_max = float(data.max())

        self.count += data.size
        self.sum += float(data.sum())
        self.sumsq += float(np.square(data).sum())
        self.min = arr_min if self.min is None else min(self.min, arr_min)
        self.max = arr_max if self.max is None else max(self.max, arr_max)

    def to_dict(self, prefix: str) -> Dict[str, float | None]:
        if self.count == 0:
            return {
                f"{prefix}_min": None,
                f"{prefix}_max": None,
                f"{prefix}_mean": None,
                f"{prefix}_std": None,
            }

        mean = self.sum / self.count
        variance = max(self.sumsq / self.count - mean * mean, 0.0)
        std = math.sqrt(variance)

        return {
            f"{prefix}_min": self.min,
            f"{prefix}_max": self.max,
            f"{prefix}_mean": mean,
            f"{prefix}_std": std,
        }


@dataclass
class TypeStatistics:
    input_stats: StatsAccumulator = field(default_factory=StatsAccumulator)
    output_stats: StatsAccumulator = field(default_factory=StatsAccumulator)

    def as_dict(self) -> Dict[str, float | None]:
        result = {}
        result.update(self.input_stats.to_dict("input"))
        result.update(self.output_stats.to_dict("output"))
        return result


# --------------------------
# File discovery
# --------------------------
def gather_sample_files(sample_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Return the input and output files for a given domain directory."""
    input_files: List[Path] = []
    output_files: List[Path] = []

    data_dir = sample_dir / "data"
    if data_dir.is_dir():
        input_files.extend(sorted(data_dir.rglob("*.npy")))

    model_dir = sample_dir / "model"
    if model_dir.is_dir():
        output_files.extend(sorted(model_dir.rglob("*.npy")))

    # Handle flat directories where files are named `seis*.npy` and `vel*.npy`.
    if not input_files or not output_files:
        for file_path in sorted(sample_dir.glob("*.npy")):
            lower_name = file_path.name.lower()
            if lower_name.startswith("seis"):
                input_files.append(file_path)
            elif lower_name.startswith("vel"):
                output_files.append(file_path)

    return input_files, output_files


def load_array(file_path: Path) -> np.ndarray:
    try:
        return np.load(file_path, allow_pickle=False)
    except ValueError as exc:
        raise RuntimeError(f"Failed to load {file_path}") from exc


# --------------------------
# Core computation with progress
# --------------------------
def compute_statistics(train_dir: Path) -> Tuple[Dict[str, TypeStatistics], TypeStatistics]:
    per_type_stats: Dict[str, TypeStatistics] = defaultdict(TypeStatistics)
    overall_stats = TypeStatistics()

    # Collect subdirs
    raw_subdirs = [d for d in sorted(train_dir.iterdir()) if d.is_dir()]
    if not raw_subdirs:
        raise FileNotFoundError(f"No data directories found in {train_dir}")

    # Pre-scan to normalize names and count files for progress bars
    normalized_domains: List[Tuple[str, Path, List[Path], List[Path]]] = []
    total_files = 0
    for domain_dir in raw_subdirs:
        type_name_snake = to_snake_lower(domain_dir.name)
        in_files, out_files = gather_sample_files(domain_dir)
        if not in_files:
            raise FileNotFoundError(f"No input `.npy` files found under {domain_dir}")
        if not out_files:
            raise FileNotFoundError(f"No output `.npy` files found under {domain_dir}")
        total_files += len(in_files) + len(out_files)
        normalized_domains.append((type_name_snake, domain_dir, in_files, out_files))

    # Setup progress bars
    tqdm = _get_tqdm()
    if tqdm:
        pbar_files = tqdm(total=total_files, desc="Accumulating stats", unit="file")
        pbar_types = tqdm(total=len(normalized_domains), desc="Types", unit="type")
        def _update_files(n=1):
            pbar_files.update(n)
        def _close():
            pbar_files.close()
            pbar_types.close()
    else:
        _update_files, _close = _simple_progress(total_files)
        pbar_types = None  # not used in fallback

    # Accumulate
    for type_name_snake, _domain_dir, input_files, output_files in normalized_domains:
        type_stats = per_type_stats[type_name_snake]

        for npy_file in input_files:
            array = load_array(npy_file)
            type_stats.input_stats.update(array)
            overall_stats.input_stats.update(array)
            _update_files(1)

        for npy_file in output_files:
            array = load_array(npy_file)
            type_stats.output_stats.update(array)
            overall_stats.output_stats.update(array)
            _update_files(1)

        if pbar_types is not None:
            pbar_types.update(1)

    _close()
    return per_type_stats, overall_stats


# --------------------------
# I/O
# --------------------------
def write_json(output_path: Path, per_type_stats: Dict[str, TypeStatistics], overall_stats: TypeStatistics) -> None:
    payload = {
        "per_type": {type_name: stats.as_dict() for type_name, stats in sorted(per_type_stats.items())},
        "overall": overall_stats.as_dict(),
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


# --------------------------
# CLI
# --------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the dataset root directory containing `train_samples/`.",
    )
    parser.add_argument(
        "--train-subdir",
        type=str,
        default="train_samples",
        help="Subdirectory name under data-dir that holds the training samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the JSON file to write. Defaults to `<data-dir>/dataset_stats.json`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = args.data_dir / args.train_subdir
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    per_type_stats, overall_stats = compute_statistics(train_dir)

    output_path = args.output or (args.data_dir / "dataset_stats.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, per_type_stats, overall_stats)

    print(f"Dataset statistics written to {output_path}")


if __name__ == "__main__":
    main()
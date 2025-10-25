#!/usr/bin/env python3
"""Compute per-type and overall normalization statistics for the seismic dataset.

The script expects the dataset layout illustrated in the project README.  It scans
`train_samples/` under the provided dataset root, gathers all `.npy` files for inputs
and outputs, and writes summary statistics (min, max, mean, std) to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


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


def compute_statistics(train_dir: Path) -> Tuple[Dict[str, TypeStatistics], TypeStatistics]:
    per_type_stats: Dict[str, TypeStatistics] = defaultdict(TypeStatistics)
    overall_stats = TypeStatistics()

    subdirs = [d for d in sorted(train_dir.iterdir()) if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No data directories found in {train_dir}")

    for domain_dir in subdirs:
        type_name = domain_dir.name
        type_stats = per_type_stats[type_name]

        input_files, output_files = gather_sample_files(domain_dir)
        if not input_files:
            raise FileNotFoundError(f"No input `.npy` files found under {domain_dir}")
        if not output_files:
            raise FileNotFoundError(f"No output `.npy` files found under {domain_dir}")

        for npy_file in input_files:
            array = load_array(npy_file)
            type_stats.input_stats.update(array)
            overall_stats.input_stats.update(array)

        for npy_file in output_files:
            array = load_array(npy_file)
            type_stats.output_stats.update(array)
            overall_stats.output_stats.update(array)

    return per_type_stats, overall_stats


def write_json(output_path: Path, per_type_stats: Dict[str, TypeStatistics], overall_stats: TypeStatistics) -> None:
    payload = {
        "per_type": {type_name: stats.as_dict() for type_name, stats in sorted(per_type_stats.items())},
        "overall": overall_stats.as_dict(),
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


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

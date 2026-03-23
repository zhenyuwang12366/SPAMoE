#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple experiment runner for encoder + AFreqMoE EMO.

覆盖当前可转成 BCHW 的 PDE/Mesh 任务与 Seismic 波形反演：
  - Regular grid: navier2d, darcy2d
  - Structured mesh: pipe, airfoil, plasticity
  - Seismic: curve_vel_a / flat_fault_a / all

Features:
  - Predefined experiments sweeping router type/top-k/backbone per task.
  - Writes each run into its own output directory under --save-root.
  - Logs stdout/stderr to file and keeps a summary.json for quick review.
  - Dry-run mode to preview commands.

Usage examples:
  python exp/run_afreqmoe_pipeline.py --dry-run
  python exp/run_afreqmoe_pipeline.py --only navier_sar_top2 pipe_basic_top2
  python exp/run_afreqmoe_pipeline.py --data-root ./pdebench_data --save-root ./results/pde_exp
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import subprocess
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PDE_TRAIN_SCRIPT = REPO_ROOT / "pde" / "train_pde.py"
SEISMIC_TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_seismic_moe.py"
PDE_DISTRIBUTED_SCRIPT = REPO_ROOT / "scripts" / "run_distributed_train_pde.sh"
SEISMIC_DISTRIBUTED_SCRIPT = REPO_ROOT / "scripts" / "run_distributed_seismic_moe.sh"


@dataclass
class Experiment:
    name: str
    domain: str = "pde"            # "pde" | "seismic"
    task: str = ""                 # PDE task name
    family: str = "vel"            # Seismic family (vel/fault/style/all/curve_vel_a...)
    moe_method: str = "afmoe"      # seismic: afmoe/basic; pde ignores
    router_type: str = "sar"      # "sar" -> AdaptiveFreqMoE, "basic" -> baseline router
    top_k: int = 2
    backbone: str = "vit"         # "vit" | "convnext_tiny"
    hidden_channels: int = 128
    batch_size: int = 32
    test_batch_size: int = 32
    epochs: int = 160
    lr: float = 1e-4
    weight_decay: float = 0.0
    aux_loss_weight: float = 0.1
    section: str = "misc"         # 实验大类标签（e1/e3/abl 等）
    seed: int | None = None
    notes: str = ""
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    extra_args: Sequence[str] = ()

    def build_cmd(
        self,
        *,
        pde_data_root: Path,
        pde_status_json: Path,
        seismic_data_root: Path,
        seismic_zarr: Path | None,
        seismic_status_json: Path,
        save_dir: Path,
        num_gpus: int,
        distributed: bool,
    ) -> List[str]:
        """Construct the training command for this experiment."""
        use_distributed = distributed and num_gpus > 1
        if self.domain == "seismic":
            if use_distributed:
                script = SEISMIC_DISTRIBUTED_SCRIPT
                cmd = [
                    "bash",
                    str(script),
                    "--mode", "train",
                    "--num_gpus", str(num_gpus),
                    "--family", self.family,
                    "--moe_method", self.moe_method,
                    "--router_type", self.router_type,
                    "--top_k", str(self.top_k),
                    "--learning_rate", str(self.lr),
                    "--weight_decay", str(self.weight_decay),
                    "--epochs", str(self.epochs),
                    "--batch_size", str(self.batch_size),
                    "--test_batch_size", str(self.test_batch_size),
                    "--vis_freq", str(1_000_000_000),
                    "--backbone", self.backbone,
                    "--output_dir", str(save_dir),
                    "--status_json", str(seismic_status_json),
                ]
            else:
                script = SEISMIC_TRAIN_SCRIPT
                cmd = [
                    sys.executable,
                    str(script),
                    "--mode", "train",
                    "--family", self.family,
                    "--moe_method", self.moe_method,
                    "--top_k", str(self.top_k),
                    "--learning_rate", str(self.lr),
                    "--weight_decay", str(self.weight_decay),
                    "--epochs", str(self.epochs),
                    "--batch_size", str(self.batch_size),
                    "--test_batch_size", str(self.test_batch_size),
                    "--vis_freq", str(1_000_000_000),
                    "--backbone", self.backbone,
                    "--output_dir", str(save_dir),
                    "--status_json", str(seismic_status_json),
                ]

            if seismic_zarr is not None:
                seismic_zarr = os.path.join(seismic_zarr, self.family) + ".zarr"
                cmd.extend(["--zarr_path", str(seismic_zarr)])
            else:
                cmd.extend(["--data_dir", str(seismic_data_root)])

            if self.use_amp:
                cmd.append("--use_amp")

            if self.seed is not None:
                cmd.extend(["--seed", str(self.seed)])

            cmd.extend(self.extra_args)
            return cmd

        # PDE branch
        if use_distributed:
            script = PDE_DISTRIBUTED_SCRIPT
            cmd = [
                "bash",
                str(script),
                "--num_gpus", str(num_gpus),
                "--task", self.task,
                "--data_root", str(pde_data_root),
                "--save_dir", str(save_dir),
                "--status_json", str(pde_status_json),
                "--router_type", self.router_type,
                "--top_k", str(self.top_k),
                "--hidden_channels", str(self.hidden_channels),
                "--backbone", self.backbone,
                "--batch_size", str(self.batch_size),
                "--test_batch_size", str(self.test_batch_size),
                "--epochs", str(self.epochs),
                "--lr", str(self.lr),
                "--weight_decay", str(self.weight_decay),
                "--aux_loss_weight", str(self.aux_loss_weight),
                "--amp_dtype", self.amp_dtype,
            ]
        else:
            script = PDE_TRAIN_SCRIPT
            cmd = [
                sys.executable,
                str(script),
                "--task", self.task,
                "--data_root", str(pde_data_root),
                "--save_dir", str(save_dir),
                "--status_json", str(pde_status_json),
                "--router_type", self.router_type,
                "--top_k", str(self.top_k),
                "--hidden_channels", str(self.hidden_channels),
                "--backbone", self.backbone,
                "--batch_size", str(self.batch_size),
                "--test_batch_size", str(self.test_batch_size),
                "--epochs", str(self.epochs),
                "--lr", str(self.lr),
                "--weight_decay", str(self.weight_decay),
                "--aux_loss_weight", str(self.aux_loss_weight),
                "--amp_dtype", self.amp_dtype,
            ]

        if self.use_amp:
            cmd.append("--use_amp")
        else:
            cmd.append("--no_amp")

        cmd.extend(self.extra_args)
        return cmd


@dataclass
class ResumeCandidate:
    checkpoint: Path
    run_dir: Path
    last_epoch: int | None
    target_epochs: int | None
    is_complete: bool = False

DEFAULT_EXPERIMENTS: List[Experiment] = []
DEFAULT_SEISMIC_FAMILIES: List[str] = [
    # 按需求默认只跑这三类；如需更多请用 --families 覆盖
    "flat_vel_a",
    "curve_vel_a",
    "curve_fault_a",
]
DEFAULT_SEEDS: List[int] = [0, 1, 2]


def _mk_seismic_exp(
    *,
    section: str,
    tag: str,
    family: str,
    seed: int,
    top_k: int = 2,
    moe_method: str = "afmoe",
    router_type: str = "sar",
    notes: str = "",
    extra: Sequence[str] = (),
) -> Experiment:
    """Helper to standardize naming & defaults for seismic experiments."""
    name = f"{section}_{tag}_{family}_s{seed}"
    return Experiment(
        name=name,
        domain="seismic",
        family=family,
        moe_method=moe_method,
        router_type=router_type,
        top_k=top_k,
        backbone="vit",
        hidden_channels=128,
        batch_size=32,
        test_batch_size=32,
        epochs=160,
        lr=1e-4,
        weight_decay=1e-4,
        aux_loss_weight=0.1,
        section=section,
        seed=seed,
        use_amp=True,
        amp_dtype="bfloat16",
        notes=notes,
        extra_args=list(extra),
    )


def build_freq_specialization_suite(
    families: Sequence[str],
    seeds: Sequence[int],
) -> List[Experiment]:
    """低/中/高频专家对照 + 融合的实验组合。"""
    experiments: List[Experiment] = []
    for fam in families:
        for s in seeds:
            experiments.append(
                _mk_seismic_exp(
                    section="freq",
                    tag="fno",
                    family=fam,
                    seed=s,
                    top_k=1,
                    moe_method="basic",
                    router_type="basic",
                    notes="单专家 FNO，偏低频",
                    extra=["--choose_experts", "0", "--top_k", "1", "--enable_freq_metrics"],
                )
            )
            experiments.append(
                _mk_seismic_exp(
                    section="freq",
                    tag="mno",
                    family=fam,
                    seed=s,
                    top_k=1,
                    moe_method="basic",
                    router_type="basic",
                    notes="单专家 MNO，偏中频",
                    extra=["--choose_experts", "1", "--top_k", "1", "--enable_freq_metrics"],
                )
            )
            experiments.append(
                _mk_seismic_exp(
                    section="freq",
                    tag="lno",
                    family=fam,
                    seed=s,
                    top_k=1,
                    moe_method="basic",
                    router_type="basic",
                    notes="单专家 LNO，偏高频",
                    extra=["--choose_experts", "2", "--top_k", "1", "--enable_freq_metrics"],
                )
            )
            experiments.append(
                _mk_seismic_exp(
                    section="freq",
                    tag="fusion",
                    family=fam,
                    seed=s,
                    top_k=2,
                    moe_method="afmoe",
                    router_type="sar",
                    notes="FNO+MNO+LNO 互补",
                    extra=[
                        "--choose_experts", "0", "1", "2",
                        "--top_k", "2",
                        "--enable_freq_metrics",
                        "--band_sharpness", "20",
                        "--freq_affinity_sharpness", "10",
                    ],
                )
            )
    return experiments


def build_seismic_suite(
    families: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> List[Experiment]:
    """构造覆盖消融/超参的 seismic 实验列表（不含 E1 主结果对比）。"""
    fams = list(families) if families else list(DEFAULT_SEISMIC_FAMILIES)
    seed_list = list(seeds) if seeds else list(DEFAULT_SEEDS)
    experiments: List[Experiment] = []

    # 频段对照/互补实验
    experiments.extend(build_freq_specialization_suite(fams, seed_list))

    for fam in fams:
        for s in seed_list:
            # E3：硬分频
            experiments.append(
                _mk_seismic_exp(
                    section="e3",
                    tag="hard_bands",
                    family=fam,
                    seed=s,
                    top_k=2,
                    moe_method="afmoe",
                    router_type="sar",
                    notes="E3 hard bands",
                    extra=["--disable_soft_bands"],
                )
            )
            # E3：全频（不分频，关闭 band mixing + 极低 band_sharpness）
            experiments.append(
                _mk_seismic_exp(
                    section="e3",
                    tag="full_band",
                    family=fam,
                    seed=s,
                    top_k=2,
                    moe_method="afmoe",
                    router_type="sar",
                    notes="E3 no frequency split",
                    extra=["--disable_band_decomposition", "--band_sharpness", "0.1"],
                )
            )
            # E4：关闭频域注意力
            experiments.append(
                _mk_seismic_exp(
                    section="e4",
                    tag="no_freq_attn",
                    family=fam,
                    seed=s,
                    top_k=2,
                    moe_method="afmoe",
                    router_type="sar",
                    notes="E4 disable freq attn",
                    extra=["--disable_freq_attn"],
                )
            )
            # E4：均匀路由
            experiments.append(
                _mk_seismic_exp(
                    section="e4",
                    tag="uniform_router",
                    family=fam,
                    seed=s,
                    top_k=2,
                    moe_method="afmoe",
                    router_type="sar",
                    notes="E4 uniform routing",
                    extra=["--routing_mode", "uniform"],
                )
            )
            # E4：随机路由
            experiments.append(
                _mk_seismic_exp(
                    section="e4",
                    tag="random_router",
                    family=fam,
                    seed=s,
                    top_k=2,
                    moe_method="afmoe",
                    router_type="sar",
                    notes="E4 random routing",
                    extra=["--routing_mode", "random"],
                )
            )
            # E5：关闭 band 混合（专家直接对应单 band）
            experiments.append(
                _mk_seismic_exp(
                    section="e5",
                    tag="no_band_mix",
                    family=fam,
                    seed=s,
                    top_k=2,
                    moe_method="afmoe",
                    router_type="sar",
                    notes="E5 disable band mixing",
                    extra=["--disable_band_mixing"],
                )
            )
            # E7：软分频对照
            experiments.append(
                _mk_seismic_exp(
                    section="e7",
                    tag="soft_bands",
                    family=fam,
                    seed=s,
                    top_k=2,
                    moe_method="afmoe",
                    router_type="sar",
                    notes="E7 soft bands control",
                )
            )
            # E9：top-k 扫描（1 与 3）
            experiments.append(
                _mk_seismic_exp(
                    section="e9",
                    tag="top1",
                    family=fam,
                    seed=s,
                    top_k=1,
                    moe_method="afmoe",
                    router_type="sar",
                    notes="E9 top-k=1",
                )
            )
            experiments.append(
                _mk_seismic_exp(
                    section="e9",
                    tag="top3",
                    family=fam,
                    seed=s,
                    top_k=3,
                    moe_method="afmoe",
                    router_type="sar",
                    notes="E9 top-k=3",
                )
            )
            # E10：band_sharpness 扫描
            for bs in (10, 40):
                experiments.append(
                    _mk_seismic_exp(
                        section="e10",
                        tag=f"bs{bs}",
                        family=fam,
                        seed=s,
                        top_k=2,
                        moe_method="afmoe",
                        router_type="sar",
                        notes=f"E10 band_sharpness={bs}",
                        extra=["--band_sharpness", str(bs)],
                    )
                )
            # E10：freq_affinity_sharpness 扫描
            for fa in (5, 20):
                experiments.append(
                    _mk_seismic_exp(
                        section="e10",
                        tag=f"fa{fa}",
                        family=fam,
                        seed=s,
                        top_k=2,
                        moe_method="afmoe",
                        router_type="sar",
                        notes=f"E10 freq_affinity_sharpness={fa}",
                        extra=["--freq_affinity_sharpness", str(fa)],
                    )
                )
            experiments.append(
                _mk_seismic_exp(
                    section="e11",
                    tag="only_fno",
                    family=fam,
                    seed=s,
                    top_k=1,
                    moe_method="basic",
                    router_type="basic",
                    notes="E11 only FNO expert",
                    extra=["--choose_experts", "0", "--disable_encoder", "--is_resize", "--H_size", "70", "--W_size", "70"],
                )
            )

    return experiments


def parse_args():
    parser = argparse.ArgumentParser("AFreqMoE experiment pipeline")
    parser.add_argument("--data-root", type=Path, default=Path("./pdebench_data"),
                        help="Path to PDEBench-style data root.")
    parser.add_argument("--save-root", type=Path, default=Path("./exp/runs"),
                        help="Folder to place all experiment outputs.")
    parser.add_argument("--status-json", type=Path, default=Path("./pde_status.json"),
                        help="Status json for data availability; same as train_pde.py expects.")
    parser.add_argument("--seis-data-root", type=Path, default=Path("./FWINO_data"),
                        help="Seismic raw/preprocessed data dir (train_seismic_moe.py --data_dir).")
    parser.add_argument("--seis-zarr", type=Path, default=None,
                        help="Optional seismic Zarr dataset path; if set, overrides --seis-data-root.")
    parser.add_argument("--seis-status-json", type=Path, default=Path("./dataset_status/dataset_status.json"),
                        help="Seismic status JSON for normalization stats.")
    parser.add_argument("--families", nargs="+", default=None,
                        help="Limit seismic families (default: all supported).")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds for seismic sweeps (default: 0 1 2).")
    parser.add_argument("--list-experiments", action="store_true",
                        help="List experiments and exit.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only the named experiments (by name).")
    parser.add_argument("--skip", nargs="+", default=None,
                        help="Skip the named experiments (by name).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--inference", action="store_true",
                        help="Only run inference using existing checkpoints under --save-root.")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="If set, keep running remaining experiments even if one fails.")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="GPU count. >1 will trigger distributed shell scripts.")
    parser.add_argument("--infer-one", type=int, default=None,
                        help="If set, only run inference on the specified experiment index (0-based) among the selected experiments.")
    return parser.parse_args()


def filter_experiments(
    experiments: Sequence[Experiment],
    only: Iterable[str] | None,
    skip: Iterable[str] | None,
) -> List[Experiment]:
    exp_map = {exp.name: exp for exp in experiments}

    if only:
        unknown = [name for name in only if name not in exp_map]
        if unknown:
            raise ValueError(f"--only contains unknown experiments: {', '.join(unknown)}")
        selected = [exp_map[name] for name in only]
    else:
        selected = list(experiments)

    if skip:
        skip_set = set(skip)
        selected = [exp for exp in selected if exp.name not in skip_set]

    return selected


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in str(text))


def _parse_timestamp_from_dir(path: Path) -> float | None:
    """Extract training timestamp suffix like 20240101-120000 to a POSIX float."""
    match = re.search(r"(\d{8}-\d{6})", path.name)
    if not match:
        return None
    try:
        dt = datetime.strptime(match.group(1), "%Y%m%d-%H%M%S")
        return dt.timestamp()
    except Exception:
        return None


def find_latest_run_dir(exp_dir: Path, family: str) -> Path | None:
    """Locate the latest training run directory for a given family."""
    run_group = _slugify(family or "all")
    root = exp_dir / f"seismic_moe_{run_group}"
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_best_checkpoint(run_dir: Path) -> Path | None:
    """Prefer best_model*.pt, otherwise last_model*.pt."""
    best = sorted(run_dir.glob("best_model*.pt"))
    if best:
        return best[0]
    last = sorted(run_dir.glob("last_model*.pt"))
    if last:
        return last[0]
    return None


def find_deepest_best_checkpoint(exp_dir: Path) -> Path | None:
    """Locate the best_model*.pt with the greatest directory depth under exp_dir."""
    candidates = list(exp_dir.rglob("best_model*.pt"))
    if not candidates:
        return None

    def _depth_key(path: Path) -> tuple[int, float, float]:
        rel_parts = path.relative_to(exp_dir).parts
        ts = _parse_timestamp_from_dir(path.parent)
        # Prefer deeper paths; tie-breaker: latest timestamp suffix, then mtime.
        return (len(rel_parts), ts if ts is not None else -1.0, path.stat().st_mtime)

    return max(candidates, key=_depth_key)


def find_deepest_last_checkpoint(exp_dir: Path) -> Path | None:
    """Locate the last_model*.pt with the greatest directory depth under exp_dir."""
    candidates = list(exp_dir.rglob("last_model*.pt"))
    if not candidates:
        return None

    def _depth_key(path: Path) -> tuple[int, float, float]:
        rel_parts = path.relative_to(exp_dir).parts
        ts = _parse_timestamp_from_dir(path.parent)
        return (len(rel_parts), ts if ts is not None else -1.0, path.stat().st_mtime)

    return max(candidates, key=_depth_key)


def _safe_read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_epochs(payload, *, fallback: int | None = None) -> int | None:
    if isinstance(payload, dict):
        for key in ("epochs", "num_epochs", "max_epochs"):
            if key in payload and payload[key] is not None:
                try:
                    return int(payload[key])
                except Exception:
                    pass
        for value in payload.values():
            if isinstance(value, dict):
                nested = _extract_epochs(value)
                if nested is not None:
                    return nested
    return fallback


def _load_checkpoint_epoch(ckpt: Path) -> int | None:
    try:
        import torch  # type: ignore
    except Exception as e:
        print(f"[resume] Unable to import torch to inspect {ckpt}: {e}")
        return None
    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[resume] Failed to load checkpoint {ckpt}: {e}")
        return None
    if isinstance(payload, dict):
        epoch_val = payload.get("epoch")
        if epoch_val is not None:
            try:
                return int(epoch_val)
            except Exception:
                return None
    return None


def detect_resume_candidate(exp: Experiment, exp_dir: Path) -> ResumeCandidate | None:
    """Search for the deepest last_model*.pt and report resume info."""
    checkpoint = find_deepest_last_checkpoint(exp_dir)
    if checkpoint is None:
        return None

    run_dir = checkpoint.parent

    target_epochs = exp.epochs
    for cfg_path in (
        run_dir / "config.json",
        run_dir / "pdebench_config.json",
        run_dir / "args.json",
    ):
        payload = _safe_read_json(cfg_path)
        maybe_epochs = _extract_epochs(payload) if payload else None
        if maybe_epochs is not None:
            target_epochs = maybe_epochs
            break

    last_epoch = _load_checkpoint_epoch(checkpoint)
    is_complete = False
    if target_epochs is not None and last_epoch is not None:
        is_complete = (last_epoch + 1) >= int(target_epochs)

    return ResumeCandidate(
        checkpoint=checkpoint,
        run_dir=run_dir,
        last_epoch=last_epoch,
        target_epochs=target_epochs,
        is_complete=is_complete,
    )


def run_command(cmd: Sequence[str], log_file: Path, *, append: bool = False) -> int:
    mode = "a" if append else "w"
    with log_file.open(mode) as f:
        if append:
            f.write("\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        proc.wait()
        return int(proc.returncode)


def main():
    args = parse_args()
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1.")

    all_experiments = build_seismic_suite(
        families=args.families,
        seeds=args.seeds,
    )

    if args.list_experiments:
        print("Available experiments:")
        for exp in all_experiments:
            print(f"- {exp.name:35s} | {exp.section:3s} | {exp.family:12s} | top_k={exp.top_k} | router={exp.router_type} | {exp.notes}")
        return

    experiments = filter_experiments(all_experiments, args.only, args.skip)
    ensure_dir(args.save_root)

    summary_path = args.save_root / "summary.json"
    summary = []

    print(f"[info] Repository root  : {REPO_ROOT}")
    print(f"[info] PDE train script : {PDE_TRAIN_SCRIPT}")
    print(f"[info] Seis train script: {SEISMIC_TRAIN_SCRIPT}")
    print(f"[info] PDE data root    : {args.data_root}")
    print(f"[info] Seis data root   : {args.seis_data_root}")
    print(f"[info] Seis zarr        : {args.seis_zarr if args.seis_zarr else '(none)'}")
    print(f"[info] Save root        : {args.save_root}")
    print(f"[info] Num GPUs         : {args.num_gpus} ({'distributed' if args.num_gpus > 1 else 'single GPU'})")
    print(f"[info] Experiments      : {[exp.name for exp in experiments]}")
    mode_str = "INFERENCE-ONLY" if args.inference else "RUN"
    print(f"[info] Mode             : {mode_str}{' DRY-RUN' if args.dry_run else ''}")

    for exp in experiments:
        exp_dir = args.save_root / exp.name
        return_code = None
        inference_code = None

        if args.inference:
            start_time = datetime.now().isoformat(timespec="seconds")
            if not exp_dir.exists():
                print(f"[warn] Exp dir {exp_dir} not found, skip inference for {exp.name}.")
            elif exp.domain != "seismic":
                print(f"[info] Skip inference for non-seismic experiment {exp.name}.")
            else:
                ckpt = find_deepest_best_checkpoint(exp_dir)
                if ckpt is None:
                    print(f"[warn] No best_model*.pt under {exp_dir}, skip inference.")
                else:
                    setting_path = ckpt.parent
                    infer_log = exp_dir / "inference.log"
                    infer_dir = exp_dir / "inference_results"
                    ensure_dir(infer_dir)
                    infer_cmd = [
                        sys.executable,
                        str(SEISMIC_TRAIN_SCRIPT),
                        "--mode", "inference",
                        "--setting_path", str(setting_path),
                        "--model_path", str(ckpt),
                        "--status_json", str(args.seis_status_json),
                        "--output_dir", str(infer_dir),
                    ]
                    if args.seis_zarr is not None:
                        infer_cmd.extend(["--zarr_path", str(os.path.join(args.seis_zarr, exp.family) + ".zarr")])
                    else:
                        infer_cmd.extend(["--data_dir", str(args.seis_data_root)])
                    if exp.seed is not None:
                        infer_cmd.extend(["--seed", str(exp.seed)])
                    if args.num_gpus > 1:
                        infer_cmd.extend(["--distributed"])
                    if args.infer_one is not None:
                        infer_cmd.extend(["--infer_one", str(args.infer_one)])
                    print(f"[run][inference] {exp.name}")
                    print(f"      setting_path: {setting_path}")
                    print(f"      model_path  : {ckpt}")
                    print("      cmd:", " ".join(infer_cmd))
                    if not args.dry_run:
                        inference_code = run_command(infer_cmd, infer_log)
                    else:
                        inference_code = 0
            end_time = datetime.now().isoformat(timespec="seconds")
            resume_info = None
        else:
            ensure_dir(exp_dir)
            log_file = exp_dir / "train.log"
            resume_info = detect_resume_candidate(exp, exp_dir)
            should_resume = bool(resume_info and not resume_info.is_complete)

            if resume_info and resume_info.is_complete:
                start_time = datetime.now().isoformat(timespec="seconds")
                last_ep = f"{resume_info.last_epoch + 1}" if resume_info.last_epoch is not None else "?"
                tgt_ep = resume_info.target_epochs or exp.epochs
                print(f"\n[skip] {exp.name}")
                print(f"      latest run in {resume_info.run_dir} looks complete (epoch {last_ep}/{tgt_ep}), skip training.")
                end_time = datetime.now().isoformat(timespec="seconds")
                summary.append(
                    {
                        "name": exp.name,
                        "return_code": 0,
                        "inference_return_code": None,
                        "start": start_time,
                        "end": end_time,
                        "notes": exp.notes,
                        "params": asdict(exp),
                        "resume_from": str(resume_info.checkpoint),
                        "resume_last_epoch": resume_info.last_epoch,
                        "resume_target_epochs": resume_info.target_epochs,
                        "resume_completed": resume_info.is_complete,
                        "skipped": True,
                    }
                )
                with summary_path.open("w") as f:
                    json.dump(summary, f, indent=2)
                continue

            cmd = exp.build_cmd(
                pde_data_root=args.data_root,
                pde_status_json=args.status_json,
                seismic_data_root=args.seis_data_root,
                seismic_zarr=args.seis_zarr,
                seismic_status_json=args.seis_status_json,
                save_dir=exp_dir,
                num_gpus=args.num_gpus,
                distributed=args.num_gpus > 1,
            )
            if should_resume and resume_info is not None:
                cmd.extend(["--resume_path", str(resume_info.checkpoint)])

            start_time = datetime.now().isoformat(timespec="seconds")
            print(f"\n[run] {exp.name}")
            print("      cmd:", " ".join(cmd))
            if resume_info:
                last_ep = f"{resume_info.last_epoch + 1}" if resume_info.last_epoch is not None else "?"
                tgt_ep = resume_info.target_epochs or exp.epochs
                if resume_info.is_complete:
                    print(f"      resume: latest run in {resume_info.run_dir} looks complete (epoch {last_ep}/{tgt_ep}), start fresh.")
                else:
                    print(f"      resume: {resume_info.checkpoint} (epoch {last_ep}/{tgt_ep})")

            if args.dry_run:
                return_code = 0
            else:
                return_code = run_command(cmd, log_file, append=should_resume)

            end_time = datetime.now().isoformat(timespec="seconds")

            # 自动推理：仅针对 seismic 任务且训练成功，使用最深层 best_model*.pt
            if return_code == 0 and exp.domain == "seismic":
                ckpt = find_deepest_best_checkpoint(exp_dir)
                if ckpt is None:
                    print(f"[warn] No best_model*.pt under {exp_dir}, skip inference.")
                else:
                    setting_path = ckpt.parent
                    infer_log = exp_dir / "inference.log"
                    infer_dir = exp_dir / "inference_results"
                    ensure_dir(infer_dir)
                    infer_cmd = [
                        sys.executable,
                        str(SEISMIC_TRAIN_SCRIPT),
                        "--mode", "inference",
                        "--setting_path", str(setting_path),
                        "--model_path", str(ckpt),
                        "--status_json", str(args.seis_status_json),
                        "--output_dir", str(infer_dir),
                    ]
                    if args.seis_zarr is not None:
                        infer_cmd.extend(["--zarr_path", str(os.path.join(args.seis_zarr, exp.family) + ".zarr")])
                    else:
                        infer_cmd.extend(["--data_dir", str(args.seis_data_root)])
                    if exp.seed is not None:
                        infer_cmd.extend(["--seed", str(exp.seed)])
                    if args.num_gpus > 1:
                        infer_cmd.extend(["--distributed"])
                    print(f"[run][inference] {exp.name}")
                    print(f"      setting_path: {setting_path}")
                    print(f"      model_path  : {ckpt}")
                    print("      cmd:", " ".join(infer_cmd))
                    if not args.dry_run:
                        inference_code = run_command(infer_cmd, infer_log)
                    else:
                        inference_code = 0

            if return_code != 0:
                print(f"[warn] Experiment {exp.name} failed with code {return_code}. Check {log_file}.")
                if not args.continue_on_failure:
                    print("[info] Stopping subsequent runs (sequential mode).")
                    # Persist summary before breaking
                    summary.append(
                        {
                            "name": exp.name,
                            "return_code": return_code,
                            "inference_return_code": inference_code,
                            "start": start_time,
                            "end": end_time,
                            "notes": exp.notes,
                            "params": asdict(exp),
                            "resume_from": str(resume_info.checkpoint) if resume_info else None,
                            "resume_last_epoch": resume_info.last_epoch if resume_info else None,
                            "resume_target_epochs": resume_info.target_epochs if resume_info else None,
                            "resume_completed": resume_info.is_complete if resume_info else None,
                        }
                    )
                    with summary_path.open("w") as f:
                        json.dump(summary, f, indent=2)
                    break

        summary.append(
            {
                "name": exp.name,
                "return_code": return_code,
                "inference_return_code": inference_code,
                "start": start_time,
                "end": end_time,
                "notes": exp.notes,
                "params": asdict(exp),
                "resume_from": str(resume_info.checkpoint) if resume_info else None,
                "resume_last_epoch": resume_info.last_epoch if resume_info else None,
                "resume_target_epochs": resume_info.target_epochs if resume_info else None,
                "resume_completed": resume_info.is_complete if resume_info else None,
            }
        )

        # Persist summary after each run to avoid loss on interruption
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        if not args.inference and return_code != 0:
            break


if __name__ == "__main__":
    main()

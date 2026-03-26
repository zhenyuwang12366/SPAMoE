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
"""

from __future__ import annotations

import os
import argparse
import json
import sys
import subprocess
import re
import time       # ===== 修改 1：用于轮询监控 TRAIN_DONE =====
import signal     # ===== 修改 2：用于结束整个进程组，避免 Slurm 下卡住 =====
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
    router_type: str = "sar"       # "sar" -> AdaptiveFreqMoE, "basic" -> baseline router
    top_k: int = 2
    backbone: str = "vit"          # "vit" | "convnext_tiny"
    hidden_channels: int = 128
    batch_size: int = 32
    test_batch_size: int = 32
    epochs: int = 160
    lr: float = 1e-4
    weight_decay: float = 0.0
    aux_loss_weight: float = 0.1
    section: str = "misc"          # 实验大类标签（e1/e3/abl 等）
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
                    "-u",  # ===== 修改 3：Python 无缓冲输出 =====
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
                # ===== 修改 4：改为 Path 风格拼接，避免 Path 和 os.path.join 混用 =====
                zarr_path = seismic_zarr / f"{self.family}.zarr"
                cmd.extend(["--zarr_path", str(zarr_path)])
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
                "-u",  # ===== 修改 5：Python 无缓冲输出 =====
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
    "flat_vel_a",
    "curve_vel_a",
    "curve_fault_a",
    "style_style_a",
    "style_style_b",
    "curve_vel_b",
    "curve_fault_b",
    "flat_vel_b",
    "flat_fault_a",
    "flat_fault_b",
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
    fams = list(families) if families else list(DEFAULT_SEISMIC_FAMILIES)
    seed_list = list(seeds) if seeds else list(DEFAULT_SEEDS)
    experiments: List[Experiment] = []

    experiments.extend(build_freq_specialization_suite(fams, seed_list))

    for fam in fams:
        for s in seed_list:
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
    parser.add_argument("--data-root", type=Path, default=Path("./pdebench_data"))
    parser.add_argument("--save-root", type=Path, default=Path("./exp/runs"))
    parser.add_argument("--status-json", type=Path, default=Path("./pde_status.json"))
    parser.add_argument("--seis-data-root", type=Path, default=Path("./FWINO_data"))
    parser.add_argument("--seis-zarr", type=Path, default=None)
    parser.add_argument("--seis-status-json", type=Path, default=Path("./dataset_status/dataset_status.json"))
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--skip", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--infer-one", type=int, default=None)

    # ===== 修改 6：新增手动指定 pt 文件路径 =====
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Manual checkpoint path for inference, e.g. /path/to/best_model.pt",
    )
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
    match = re.search(r"(\d{8}-\d{6})", path.name)
    if not match:
        return None
    try:
        dt = datetime.strptime(match.group(1), "%Y%m%d-%H%M%S")
        return dt.timestamp()
    except Exception:
        return None


def find_latest_run_dir(exp_dir: Path, family: str) -> Path | None:
    run_group = _slugify(family or "all")
    root = exp_dir / f"seismic_moe_{run_group}"
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_best_checkpoint(run_dir: Path) -> Path | None:
    best = sorted(run_dir.glob("best_model*.pt"))
    if best:
        return best[0]
    last = sorted(run_dir.glob("last_model*.pt"))
    if last:
        return last[0]
    return None


def find_deepest_best_checkpoint(exp_dir: Path) -> Path | None:
    candidates = list(exp_dir.rglob("best_model*.pt"))
    if not candidates:
        return None

    def _depth_key(path: Path) -> tuple[int, float, float]:
        rel_parts = path.relative_to(exp_dir).parts
        ts = _parse_timestamp_from_dir(path.parent)
        return (len(rel_parts), ts if ts is not None else -1.0, path.stat().st_mtime)

    return max(candidates, key=_depth_key)


def find_deepest_last_checkpoint(exp_dir: Path) -> Path | None:
    candidates = list(exp_dir.rglob("last_model*.pt"))
    if not candidates:
        return None

    def _depth_key(path: Path) -> tuple[int, float, float]:
        rel_parts = path.relative_to(exp_dir).parts
        ts = _parse_timestamp_from_dir(path.parent)
        return (len(rel_parts), ts if ts is not None else -1.0, path.stat().st_mtime)

    return max(candidates, key=_depth_key)


# ===== 修改 7：新增 TRAIN_DONE 检索函数，替代 loss_curve.png 作为完成标志 =====
def find_train_done(exp_dir: Path) -> Path | None:
    """
    在 exp_dir 下递归查找 TRAIN_DONE。

    目录形态示例：
      exp_dir=/.../exp/runs/freq_mno_curve_fault_a_s0
      result_dir=/.../freq_mno_curve_fault_a_s0/scale_1/seismic_moe_curve_fault_a/MOE_router-basic_lr..._20260325-080804
      TRAIN_DONE 位于 result_dir 中

    一个 exp_dir 下可能存在多个 result_dir，因此这里取“更深且更新”的那个。
    """
    candidates = list(exp_dir.rglob("TRAIN_DONE"))
    if not candidates:
        return None

    def _rank_key(path: Path) -> tuple[int, float, float]:
        rel_parts = path.relative_to(exp_dir).parts
        ts = _parse_timestamp_from_dir(path.parent)
        return (
            len(rel_parts),                         # 优先更深层
            ts if ts is not None else -1.0,        # 再优先时间戳目录
            path.stat().st_mtime,                  # 最后按文件修改时间
        )

    return max(candidates, key=_rank_key)


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


# ===== 修改 8：将 run_command 改为“运行中轮询监控 TRAIN_DONE”的版本 =====
def run_command(
    cmd: Sequence[str],
    log_file: Path,
    *,
    append: bool = False,
    watch_dir: Path | None = None,
    watch_train_done: bool = False,
    poll_interval: float = 5.0,
    graceful_wait_after_done: float = 20.0,
    kill_wait_timeout: float = 15.0,
) -> int:
    """
    运行子进程。

    若 watch_train_done=True，则递归监控 watch_dir 下是否出现 TRAIN_DONE：
      1. 一旦出现 TRAIN_DONE，先等待 graceful_wait_after_done 秒，给训练脚本自然退出机会
      2. 若仍未退出，则对整个进程组发送 SIGTERM
      3. 若仍未退出，再发送 SIGKILL

    这样可解决：
      - 训练逻辑已完成
      - TRAIN_DONE 已写好
      - 但 bash/torchrun/DDP worker 仍未完全退出，导致外层卡住
    """
    mode = "a" if append else "w"
    with log_file.open(mode, encoding="utf-8") as f:
        if append:
            f.write("\n\n")
            f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,   # ===== 修改 9：创建新的进程组，便于 killpg 整组结束 =====
        )

        detected_done: Path | None = None
        done_detect_time: float | None = None

        while True:
            # 先看子进程是否自然退出
            return_code = proc.poll()
            if return_code is not None:
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
                return int(return_code)

            # 运行中递归监控 TRAIN_DONE
            if watch_train_done and watch_dir is not None:
                done_file = find_train_done(watch_dir)
                if done_file is not None:
                    if detected_done is None:
                        detected_done = done_file
                        done_detect_time = time.time()
                        print(f"[info] Detected TRAIN_DONE: {detected_done}")
                        print(f"[info] Waiting up to {graceful_wait_after_done:.1f}s for natural process exit...")

                    assert done_detect_time is not None
                    if time.time() - done_detect_time >= graceful_wait_after_done:
                        print("[info] TRAIN_DONE exists but process still alive, terminating process group...")
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass

                        deadline = time.time() + kill_wait_timeout
                        while time.time() < deadline:
                            return_code = proc.poll()
                            if return_code is not None:
                                f.flush()
                                try:
                                    os.fsync(f.fileno())
                                except OSError:
                                    pass
                                # ===== 修改 10：有 TRAIN_DONE 时，即使是外层强制收尾，也按成功处理 =====
                                return 0
                            time.sleep(1.0)

                        print("[warn] Process group still alive after SIGTERM, sending SIGKILL...")
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass

                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except OSError:
                            pass

                        return 0

            time.sleep(poll_interval)


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

            if args.model_path is None:
                raise ValueError("When using --inference, you must provide --model-path /path/to/model.pt")

            ckpt = args.model_path
            if not ckpt.exists():
                raise FileNotFoundError(f"--model-path not found: {ckpt}")

            setting_path = ckpt.parent
            infer_log = exp_dir / "inference.log"
            infer_dir = exp_dir / "inference_results"
            ensure_dir(exp_dir)
            ensure_dir(infer_dir)

            infer_cmd = [
                sys.executable,
                "-u",  # ===== 修改 11：推理阶段也使用无缓冲输出 =====
                str(SEISMIC_TRAIN_SCRIPT),
                "--mode", "inference",
                "--setting_path", str(setting_path),
                "--model_path", str(ckpt),
                "--status_json", str(args.seis_status_json),
                "--output_dir", str(infer_dir),
            ]
            if args.seis_zarr is not None:
                # ===== 修改 12：改为 Path 风格拼接 =====
                zarr_path = args.seis_zarr / f"{exp.family}.zarr"
                infer_cmd.extend(["--zarr_path", str(zarr_path)])
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

            # ===== 修改 13：启动前先检查 exp_dir 下是否已有 TRAIN_DONE，若有则直接跳过 =====
            train_done = find_train_done(exp_dir)
            if train_done is not None:
                start_time = datetime.now().isoformat(timespec="seconds")
                print(f"\n[skip] {exp.name}")
                print(f"      found completed marker: {train_done}")
                end_time = datetime.now().isoformat(timespec="seconds")
                resume_info = None
                summary.append(
                    {
                        "name": exp.name,
                        "return_code": 0,
                        "inference_return_code": None,
                        "start": start_time,
                        "end": end_time,
                        "notes": exp.notes,
                        "params": asdict(exp),
                        "resume_from": None,
                        "resume_last_epoch": None,
                        "resume_target_epochs": None,
                        "resume_completed": True,
                        "skipped": True,
                        "completed_by": "TRAIN_DONE",
                        "train_done_path": str(train_done),
                    }
                )
                with summary_path.open("w") as f:
                    json.dump(summary, f, indent=2)
                continue

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
                        "completed_by": "resume_checkpoint_epoch",
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
                # ===== 修改 14：训练时开启对 exp_dir 下 TRAIN_DONE 的运行中监控 =====
                return_code = run_command(
                    cmd,
                    log_file,
                    append=should_resume,
                    watch_dir=exp_dir,
                    watch_train_done=True,
                    poll_interval=5.0,
                    graceful_wait_after_done=20.0,
                    kill_wait_timeout=15.0,
                )

            end_time = datetime.now().isoformat(timespec="seconds")

            # ===== 修改 15：删除训练成功后的自动推理功能 =====
            inference_code = None

            if return_code != 0:
                print(f"[warn] Experiment {exp.name} failed with code {return_code}. Check {log_file}.")
                if not args.continue_on_failure:
                    print("[info] Stopping subsequent runs (sequential mode).")
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

        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        if not args.inference and return_code != 0:
            break


if __name__ == "__main__":
    main()
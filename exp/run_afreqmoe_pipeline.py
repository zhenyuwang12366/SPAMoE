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
    batch_size: int = 16
    test_batch_size: int = 16
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.0
    aux_loss_weight: float = 0.1
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
                    "--backbone", self.backbone,
                    "--output_dir", str(save_dir),
                    "--status_json", str(seismic_status_json),
                    "--amp_dtype", self.amp_dtype,
                ]

            if seismic_zarr is not None:
                seismic_zarr = os.path.join(seismic_zarr, self.family) + ".zarr"
                cmd.extend(["--zarr_path", str(seismic_zarr)])
            else:
                cmd.extend(["--data_dir", str(seismic_data_root)])

            if self.use_amp:
                cmd.append("--use_amp")

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

DEFAULT_EXPERIMENTS: List[Experiment] = [
    # ================= Navier-Stokes 2D =================
    # 主结果：Navier 2D AFMoE
    Experiment(
        name="navier_sar_vit_top2",
        task="navier2d",
        router_type="sar",
        top_k=2,
        batch_size=32,
        test_batch_size=32,
        epochs=120,
        lr=5e-4,
        notes="Navier 2D AFreqMoE，ViT 编码器，主实验。",
    ),
    # Navier 消融：basic router 对照
    Experiment(
        name="navier_basic_vit_top2",
        task="navier2d",
        router_type="basic",
        top_k=2,
        batch_size=32,
        test_batch_size=32,
        epochs=120,
        lr=5e-4,
        notes="Navier 2D basic router 基线（唯一 PDE 任务上的 router 对照）。",
    ),
    # Navier 消融：单专家 vs 多专家
    Experiment(
        name="navier_sar_vit_top1",
        task="navier2d",
        router_type="sar",
        top_k=1,
        batch_size=32,
        test_batch_size=32,
        epochs=120,
        lr=5e-4,
        aux_loss_weight=0.05,
        notes="Navier 2D AFreqMoE，单专家激活消融。",
    ),

    # ================= Darcy 2D =================
    # 主结果：Darcy 2D AFMoE
    Experiment(
        name="darcy_sar_vit_top2",
        task="darcy2d",
        router_type="sar",
        top_k=2,
        batch_size=32,
        test_batch_size=32,
        epochs=160,
        lr=1e-4,
        notes="Darcy 2D AFreqMoE，ViT 编码器，主实验。",
    ),
    # Darcy 消融：单专家 vs 多专家（稳态 PDE 上多专家的必要性）
    Experiment(
        name="darcy_sar_vit_top1",
        task="darcy2d",
        router_type="sar",
        top_k=1,
        batch_size=32,
        test_batch_size=32,
        epochs=160,
        lr=1e-4,
        aux_loss_weight=0.05,
        notes="Darcy 2D AFreqMoE，单专家激活消融。",
    ),

    # ================= Pipe =================
    # 仅保留 AFMoE 主结果
    Experiment(
        name="pipe_sar_vit_top2",
        task="pipe",
        router_type="sar",
        top_k=2,
        batch_size=32,
        test_batch_size=32,
        epochs=150,
        lr=2e-4,
        notes="Pipe AFreqMoE，ViT 编码器，主实验。",
    ),

    # ================= Airfoil =================
    Experiment(
        name="airfoil_sar_vit_top2",
        task="airfoil",
        router_type="sar",
        top_k=2,
        batch_size=32,
        test_batch_size=32,
        epochs=150,
        lr=2e-4,
        notes="Airfoil AFreqMoE，ViT 编码器，主实验。",
    ),

    # ================= Plasticity =================
    Experiment(
        name="plasticity_sar_vit_top2",
        task="plasticity",
        router_type="sar",
        top_k=2,
        batch_size=32,
        test_batch_size=32,
        epochs=200,
        lr=1e-4,
        notes="Plasticity AFreqMoE，4*T 展平通道，主实验。",
    ),

    # ================= Seismic: Curve_Vel_A =================
    # Seismic：局部曲线速度 AFMoE vs basic
    Experiment(
        name="seis_afmoe_curve_vel_a_top2_vit",
        domain="seismic",
        family="curve_vel_a",
        moe_method="afmoe",
        router_type="sar",
        top_k=2,
        backbone="vit",
        batch_size=32,
        test_batch_size=32,
        epochs=120,
        lr=2e-4,
        weight_decay=5e-4,
        notes="Seismic AFreqMoE，curve_vel_a，ViT。",
    ),
    Experiment(
        name="seis_basic_curve_vel_a_top2_vit",
        domain="seismic",
        family="curve_vel_a",
        moe_method="basic",
        router_type="basic",
        top_k=2,
        backbone="vit",
        batch_size=32,
        test_batch_size=32,
        epochs=120,
        lr=2e-4,
        weight_decay=5e-4,
        notes="Seismic basic router 基线，curve_vel_a。",
    ),

    # ================= Seismic: Flat_Fault_A =================
    # Seismic：平坦断层 AFMoE vs basic
    Experiment(
        name="seis_afmoe_flat_fault_a_top2_vit",
        domain="seismic",
        family="flat_fault_a",
        moe_method="afmoe",
        router_type="sar",
        top_k=2,
        backbone="vit",
        batch_size=32,
        test_batch_size=32,
        epochs=120,
        lr=2e-4,
        weight_decay=5e-4,
        notes="Seismic AFreqMoE，flat_fault_a，ViT。",
    ),
    Experiment(
        name="seis_basic_flat_fault_a_top2_vit",
        domain="seismic",
        family="flat_fault_a",
        moe_method="basic",
        router_type="basic",
        top_k=2,
        backbone="vit",
        batch_size=32,
        test_batch_size=32,
        epochs=120,
        lr=2e-4,
        weight_decay=5e-4,
        notes="Seismic basic router 基线，flat_fault_a。",
    ),

    # ================= Seismic: All families =================
    # 全量 all：AFMoE vs basic
    Experiment(
        name="seis_afmoe_all_top2_vit",
        domain="seismic",
        family="all",
        moe_method="afmoe",
        router_type="sar",
        top_k=2,
        backbone="vit",
        batch_size=32,
        test_batch_size=32,
        epochs=140,
        lr=1.5e-4,
        weight_decay=5e-4,
        notes="Seismic AFreqMoE，全量 all，ViT。",
    ),
    Experiment(
        name="seis_basic_all_top2_vit",
        domain="seismic",
        family="all",
        moe_method="basic",
        router_type="basic",
        top_k=2,
        backbone="vit",
        batch_size=32,
        test_batch_size=32,
        epochs=140,
        lr=1.5e-4,
        weight_decay=5e-4,
        notes="Seismic basic router 全量 all 基线。",
    ),
    # 全量 all 消融：单专家
    Experiment(
        name="seis_afmoe_all_top1_vit",
        domain="seismic",
        family="all",
        moe_method="afmoe",
        router_type="sar",
        top_k=1,
        backbone="vit",
        batch_size=32,
        test_batch_size=32,
        epochs=140,
        lr=1.5e-4,
        weight_decay=5e-4,
        aux_loss_weight=0.05,
        notes="Seismic AFreqMoE，全量 all，单专家激活消融。",
    ),
    # 全量 all 消融：ConvNeXt 骨干
    Experiment(
        name="seis_afmoe_all_top2_convnext",
        domain="seismic",
        family="all",
        moe_method="afmoe",
        router_type="sar",
        top_k=2,
        backbone="convnext_tiny",
        batch_size=32,
        test_batch_size=32,
        epochs=140,
        lr=1.5e-4,
        weight_decay=5e-4,
        notes="Seismic AFreqMoE，全量 all，ConvNeXt 骨干消融。",
    ),
]


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
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only the named experiments (by name).")
    parser.add_argument("--skip", nargs="+", default=None,
                        help="Skip the named experiments (by name).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="If set, keep running remaining experiments even if one fails.")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="GPU count. >1 will trigger distributed shell scripts.")
    return parser.parse_args()


def filter_experiments(only: Iterable[str] | None, skip: Iterable[str] | None) -> List[Experiment]:
    exp_map = {exp.name: exp for exp in DEFAULT_EXPERIMENTS}

    if only:
        unknown = [name for name in only if name not in exp_map]
        if unknown:
            raise ValueError(f"--only contains unknown experiments: {', '.join(unknown)}")
        selected = [exp_map[name] for name in only]
    else:
        selected = list(DEFAULT_EXPERIMENTS)

    if skip:
        skip_set = set(skip)
        selected = [exp for exp in selected if exp.name not in skip_set]

    return selected


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(cmd: Sequence[str], log_file: Path) -> int:
    with log_file.open("w") as f:
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

    experiments = filter_experiments(args.only, args.skip)
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
    print(f"[info] Mode             : {'DRY-RUN' if args.dry_run else 'RUN'}")

    for exp in experiments:
        exp_dir = args.save_root / exp.name
        ensure_dir(exp_dir)
        log_file = exp_dir / "train.log"

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

        start_time = datetime.now().isoformat(timespec="seconds")
        print(f"\n[run] {exp.name}")
        print("      cmd:", " ".join(cmd))

        if args.dry_run:
            return_code = 0
        else:
            return_code = run_command(cmd, log_file)

        end_time = datetime.now().isoformat(timespec="seconds")
        summary.append(
            {
                "name": exp.name,
                "return_code": return_code,
                "start": start_time,
                "end": end_time,
                "notes": exp.notes,
                "params": asdict(exp),
            }
        )

        # Persist summary after each run to avoid loss on interruption
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        if return_code != 0:
            print(f"[warn] Experiment {exp.name} failed with code {return_code}. Check {log_file}.")
            if not args.continue_on_failure:
                print("[info] Stopping subsequent runs (sequential mode).")
                break


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna 调参外层调度器（单进程）
- 每个 trial 调用你的多卡 bash 启动器（内部再 torchrun 吃满 2 卡）
- 通过流式解析 stdout 的 REPORT/VAL_LOSS 实现实时上报与剪枝
"""
import os
import json
import shlex
import argparse
import subprocess
import signal
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import threading
import re
import math
import time

def suggest_params(trial: optuna.Trial):
    """在这里集中定义要搜索的超参空间"""
    return {
        # -------- 优化/训练 --------
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 0.0, 0.1),
        "batch_size":    trial.suggest_categorical("batch_size", [2, 4, 6, 8]),
        "epochs":        trial.suggest_categorical("epochs", [100, 150, 200]),
        "accum_steps":   trial.suggest_categorical("accum_steps", [1, 2]),
        "scheduler_gamma": trial.suggest_float("scheduler_gamma", 0.2, 0.5),

        # -------- 模型容量/结构 --------
        "hidden_channels": trial.suggest_categorical("hidden_channels", [32, 48, 64, 96, 128, 160]),
        "FNO_n_layers":  trial.suggest_categorical("FNO_n_layers", [4, 6, 8]),
        "WNO_n_layers":  trial.suggest_int("WNO_n_layers", 2, 7),
        "MNO_n_layers":  trial.suggest_categorical("MNO_n_layers", [2, 3, 4]),
        "LNO_n_layers":  trial.suggest_categorical("LNO_n_layers", [3, 4, 5]),
        "WNO_block_n_layers": trial.suggest_categorical("WNO_block_n_layers", [2, 4]),
        "WNO_dropout_rate":   trial.suggest_float("WNO_dropout_rate", 0.1, 0.2),
        "WNO_n_levels_height": trial.suggest_int("WNO_n_levels_height", 2, 4),
        "WNO_n_levels_width":  trial.suggest_int("WNO_n_levels_width", 2, 4),

        # -------- Loss 权重 --------
        "lambda_g1v": trial.suggest_float("lambda_g1v", 0.3, 1.5, log=True),
        "lambda_g2v": trial.suggest_float("lambda_g2v", 0.3, 1.5, log=True),
    }

def build_bash_cmd(args, trial_number: int, hp: dict) -> list:
    """
    返回 list 形式的 argv（不再用 shell=True 拼接大字符串）。
    通过 stdbuf 强制行缓冲，确保 stdout/stderr 实时刷新。
    """
    out_dir = os.path.join(args.output_dir, f"trial_{trial_number}")
    os.makedirs(out_dir, exist_ok=True)

    choose_exp = list(map(str, args.choose_experts))

    argv = [
        "stdbuf", "-oL", "-eL",                 # 关键：行缓冲
        "bash", args.bash_launcher,             # 显式用 bash 调用
        "--mode", "train",
        "--num_gpus", str(args.num_gpus),
        "--data_dir", args.data_dir,
        "--family", args.family,
        "--output_dir", out_dir,
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--top_k", str(args.top_k),
        "--choose_experts", *choose_exp,
        "--wavelet_type", str(args.wavelet_type),
    ]
    if args.is_specific:
        argv.append("--is_specific")

    # 超参部分
    argv += [
        "--batch_size", str(hp["batch_size"]),
        "--epochs", str(hp["epochs"]),
        "--learning_rate", str(hp["learning_rate"]),
        "--weight_decay", str(hp["weight_decay"]),
        "--accum_steps", str(hp["accum_steps"]),
        "--scheduler_gamma", str(hp["scheduler_gamma"]),
        "--hidden_channels", str(hp["hidden_channels"]),
        "--FNO_n_layers", str(hp["FNO_n_layers"]),
        "--WNO_n_layers", str(hp["WNO_n_layers"]),
        "--MNO_n_layers", str(hp["MNO_n_layers"]),
        "--LNO_n_layers", str(hp["LNO_n_layers"]),
        "--WNO_block_n_layers", str(hp["WNO_block_n_layers"]),
        "--WNO_dropout_rate", str(hp["WNO_dropout_rate"]),
        "--WNO_n_levels_height", str(hp["WNO_n_levels_height"]),
        "--WNO_n_levels_width", str(hp["WNO_n_levels_width"]),
        "--lambda_g1v", str(hp["lambda_g1v"]),
        "--lambda_g2v", str(hp["lambda_g2v"]),
    ]
    return argv

def main():
    ap = argparse.ArgumentParser(description="Optuna 调参（外层单进程）")
    # ---- Optuna 基本参数 ----
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--study_name", type=str, default="seismic_moe_tune")
    ap.add_argument("--storage", type=str, default="sqlite:///../results/optuna/moe_flatvel_tpe.db")
    ap.add_argument("--seed", type=int, default=42)

    # ---- 你的多卡 bash 启动器路径/资源 ----
    ap.add_argument("--bash_launcher", type=str, default="scripts/run_distributed_seismic_moe.sh")
    ap.add_argument("--num_gpus", type=int, default=2)
    ap.add_argument("--cuda_visible_devices", type=str, default="0,1")

    # ---- 训练固定参数（非搜索） ----
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--family", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--choose_experts", nargs="+", type=int, default=[0])
    ap.add_argument("--is_specific", action="store_true")
    ap.add_argument('--wavelet_type', type=str, default='haar', choices=['coif4','db4','db8','sym4','coif5','sym8'],
                        help='小波类型')
    
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sampler = TPESampler(
        multivariate=True,   # 利用参数相关性
        group=True,          # 共同采样一组参数（维度更高→需要更多试验）
        n_startup_trials=40, # 30–50 之间更稳
        seed=args.seed
    )

    pruner = MedianPruner(
        n_startup_trials=8,                     # 前几个 trial 完全不剪枝
        n_warmup_steps=15,# 跑到 10% 再比较
        interval_steps=5                        # 每 5 个 epoch 比较一次
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    REPORT_RE = re.compile(r".*REPORT:\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)\s*:\s*(\d+)")
    VAL_RE    = re.compile(r".*VAL_LOSS:\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)")

    def objective(trial: optuna.Trial):
        hp = suggest_params(trial)
        argv = build_bash_cmd(args, trial.number, hp)

        trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        stdout_path = os.path.join(trial_dir, "child_stdout.txt")
        stderr_path = os.path.join(trial_dir, "child_stderr.txt")

        # 环境变量：把 CUDA 设备放到 env，而不是命令行前缀
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        # （可选）进一步确保 Python 无缓冲
        env.setdefault("PYTHONUNBUFFERED", "1")

        final_val = None
        start_ts = time.time()

        with open(stdout_path, "w", encoding="utf-8") as fout, open(stderr_path, "w", encoding="utf-8") as ferr:
            # 建立新进程组，便于整组杀掉（torchrun 的子进程一并终止）
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                preexec_fn=os.setsid,   # Linux 下新建进程组
            )

            # 后台线程：持续转储 stderr，防止阻塞
            def _drain_stderr():
                try:
                    for eline in proc.stderr:
                        ferr.write(eline)
                except Exception:
                    pass

            t_err = threading.Thread(target=_drain_stderr, daemon=True)
            t_err.start()

            try:
                for line in proc.stdout:
                    fout.write(line)
                    s = line.strip()

                    # 解析 REPORT:<loss>:<step>
                    m = REPORT_RE.match(s)
                    if m:
                        loss_val = float(m.group(1))
                        step_val = int(m.group(4))
                        if math.isfinite(loss_val):
                            trial.report(loss_val, step=step_val)
                            if trial.should_prune():
                                # 剪枝：整组 kill
                                try:
                                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                                except Exception:
                                    pass
                                raise optuna.TrialPruned("Pruned by median rule")

                    # 解析 VAL_LOSS:<best_val_loss>
                    m2 = VAL_RE.match(s)
                    if m2:
                        try:
                            final_val = float(m2.group(1))
                        except Exception:
                            final_val = None

                retcode = proc.wait(timeout=5)
            except optuna.TrialPruned:
                # 写个标记文件，便于回溯
                with open(os.path.join(trial_dir, "PRUNED"), "w") as f:
                    f.write("pruned\n")
                raise
            except Exception as e:
                # 异常：整组强杀
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
                # 标记异常原因
                with open(os.path.join(trial_dir, "ERROR"), "w") as f:
                    f.write(str(e))
                return float("inf")

        # 子进程非零退出码，直接判 inf（把原因留在 child_stderr）
        if retcode != 0:
            with open(os.path.join(trial_dir, "NONZERO_EXIT"), "w") as f:
                f.write(f"retcode={retcode}\n")
            return float("inf")

        # 没有拿到最终指标：通常是缓冲/0 batch/日志没打印到
        if final_val is None or not math.isfinite(final_val):
            with open(os.path.join(trial_dir, "NO_FINAL_VAL"), "w") as f:
                f.write("No VAL_LOSS parsed. Check buffering / dataset / val loop.\n")
            return float("inf")

        return float(final_val)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print("\n===== Optuna 调参完成 =====")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best val_loss: {study.best_value:.6f}")

    best_path = os.path.join(args.output_dir, f"optuna_best_params_{args.study_name}.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(study.best_trial.params, f, ensure_ascii=False, indent=2)
    print(f"Best params saved to: {best_path}")


if __name__ == "__main__":
    main()
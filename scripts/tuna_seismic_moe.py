#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna 调参独立脚本
依赖:
    pip install optuna
"""
import os
import json
import copy
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from train_seismic_moe import build_argparser_and_parse, run_training  # 直接使用你的辅助函数

def apply_trial_suggestions(args, trial):
    """
    仅针对关键超参进行搜索
    """
    import copy
    new_args = copy.deepcopy(args)

    # ---------- 优化器/训练相关 ----------
    new_args.learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-3, log=True)      # 学习率
    new_args.weight_decay  = trial.suggest_float(
        "weight_decay", 0.0, 0.1)                   # 权重衰减
    new_args.batch_size    = trial.suggest_categorical(
        "batch_size", [2, 4, 6, 8])                 # 批大小
    new_args.accum_steps   = trial.suggest_categorical(
        "accum_steps", [1, 2])                      # 梯度累计
    new_args.scheduler_gamma = trial.suggest_float(
        "scheduler_gamma", 0.2, 0.5)                # 学习率衰减因子

    # ---------- 模型容量 ----------
    new_args.hidden_channels = trial.suggest_categorical(
        "hidden_channels", [64, 96, 128, 160])      # 隐藏通道数

    # # ---------- MoE 核心 ----------
    # new_args.top_k = trial.suggest_categorical(
    #     "top_k", [1, 2])                             # 每次激活专家数量
    # # 关键专家组合
    # new_args.choose_experts = trial.suggest_categorical(
    #     "choose_experts", [[0], [1], [2], [0,1], [1,2], [0,1,2]])

    # ---------- 各专家主要深度 ----------
    new_args.FNO_n_layers = trial.suggest_categorical(
        "FNO_n_layers", [4, 6, 8])                   # FNO 层数
    new_args.WNO_n_layers = trial.suggest_categorical(
        "WNO_n_layers", [4, 6, 8])                   # WNO 层数
    new_args.MNO_n_layers = trial.suggest_categorical(
        "MNO_n_layers", [2, 3, 4])                   # MNO 层数
    new_args.LNO_n_layers = trial.suggest_categorical(
        "LNO_n_layers", [3, 4, 5])                   # LNO 层数

    # ---------- Loss 权重 ----------
    new_args.lambda_g1v = trial.suggest_float(
        "lambda_g1v", 0.3, 1.5, log=True)
    new_args.lambda_g2v = trial.suggest_float(
        "lambda_g2v", 0.3, 1.5, log=True)

    # 若用户未在命令行指定 epoch，可在此限定范围
    if getattr(new_args, "epochs", None) is None:
        new_args.epochs = trial.suggest_categorical("epochs", [100, 150, 200])

    return new_args


def main():
    # 继承原脚本所有参数，再加上调参专用
    parser = build_argparser_and_parse([]).parser  # 如果你把 parser 暴露出来，也可直接用 parser
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--study_name', type=str, default='seismic_moe_tune')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage, e.g. sqlite:///moe.db')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sampler = TPESampler(multivariate=True, group=True, n_startup_trials=10, seed=args.seed)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage)
    )

    def objective(trial):
        trial_args = apply_trial_suggestions(args, trial)
        trial_args.use_wandb = False
        trial_args.vis_freq  = max(getattr(trial_args, "vis_freq", 9999), trial_args.epochs + 1)
        trial_args.seed      = args.seed
        try:
            _, best_val = run_training(trial_args, trial=trial)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"[Optuna] Trial failed: {e}")
            return float("inf")
        return float(best_val)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print("\n===== Optuna 调参完成 =====")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best val_loss: {study.best_value:.6f}")
    with open(os.path.join(args.output_dir,
              f"optuna_best_params_{args.study_name}.json"), "w", encoding="utf-8") as f:
        json.dump(study.best_trial.params, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
# train_process.py
import os
import datetime
import time
from contextlib import nullcontext, contextmanager
from typing import Optional, Callable, Any

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

try:
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    from deepspeed.runtime.zero import GatheredParameters as DS_GatheredParameters
except Exception:
    deepspeed = None
    DS_GatheredParameters = None

from .plot_fig import analyze_fourier_domain, visualize_results, save_type_predictions_txt


@contextmanager
def maybe_autocast(enabled: bool, device: torch.device, dtype=torch.bfloat16):
    if enabled:
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            yield
    else:
        yield


def _is_main_process(is_logger: bool, engine: Optional[Any] = None) -> bool:
    """
    保持你原有 is_logger 逻辑；当传入 DeepSpeed engine 时，用 engine 判断主进程，二者“与”以更安全。
    """
    if engine is not None and hasattr(engine, "global_rank"):
        return is_logger and (engine.global_rank == 0)
    return is_logger


def _get_current_lr(optimizer, engine=None, lr_scheduler=None):
    """
    统一获取当前学习率：
    - DeepSpeed 优先从 engine.get_lr() 拿；
    - 否则从外部 optimizer.param_groups[0]["lr"] 读取。
    """
    if engine is not None and hasattr(engine, "get_lr"):
        lr_list = engine.get_lr()
        if isinstance(lr_list, (list, tuple)) and len(lr_list) > 0:
            return float(lr_list[0])
    if optimizer is not None and len(optimizer.param_groups) > 0:
        return float(optimizer.param_groups[0].get("lr", 0.0))
    return 0.0


def _maybe_step_scheduler(scheduler_step_mode: str, when: str, lr_scheduler, engine=None):
    """
    在 per_step / per_epoch 两种模式下调用 step。
    对 DeepSpeed：若 ds_config 里已经配置了 LR scheduler，一般会被 engine 管理，这里仅在 lr_scheduler 非 None 时调用。
    """
    if lr_scheduler is None:
        return
    if when == "per_step" and scheduler_step_mode == "per_step":
        lr_scheduler.step()
    elif when == "per_epoch" and scheduler_step_mode == "per_epoch":
        lr_scheduler.step()


@torch.no_grad()
def _evaluate_one_epoch(
    epoch,
    total_epoch,
    model,
    encoder,
    is_logger,
    val_loader,
    device,
    criterion,
    metrics_module,  # 需有 calculate_psnr(pred, tgt)
    tqdm,
    amp_enabled: bool = False,
    train_encoder: bool = False,
    # 新增：DeepSpeed engine（评估阶段仅用于 is_main 判断，不参与计算）
    engine: Optional[Any] = None,
):
    model.eval()
    if encoder is not None:
        encoder.eval()

    val_loss = 0.0
    mse_sum = mae_sum = psnr_sum = ce_sum = rmse_sum = ssim_sum = 0.0

    if tqdm is not None:
        pbar_iter = tqdm(val_loader, desc=f"Epoch(eval) {epoch+1}/{total_epoch}", leave=False,
                         disable=not _is_main_process(is_logger, engine))

    for batch in pbar_iter:
        inputs = batch['input'].to(device, non_blocking=True)
        targets = batch['output'].to(device, non_blocking=True)
        if train_encoder:
            labels = batch['v_type'].to(device, non_blocking=True)

        if encoder is not None:
            with maybe_autocast(amp_enabled, device):
                encoded, weights, _ = encoder(inputs)
        else:
            encoded, weights = inputs, None

        with maybe_autocast(amp_enabled, device):
            preds, aux_loss = model(encoded, weights)
        if aux_loss is None:
            aux_loss = preds.new_zeros(())

        if train_encoder:
            loss_dict = criterion(preds, targets, weights, labels)
        else:
            loss_dict = criterion(preds, targets)

        val_loss += loss_dict["loss"].item()
        mse_sum  += loss_dict["l2"].item()
        mae_sum  += loss_dict["l1"].item()
        psnr_sum += metrics_module.calculate_psnr(preds, targets)
        rmse_sum += metrics_module.calculate_rmse(preds, targets)
        ssim_sum += metrics_module.calculate_ssim(preds, targets)
        if train_encoder:
            ce_sum += loss_dict["ce"].item()

    n = max(1, len(val_loader))
    out = {
        "val_loss": val_loss / n,
        "mse": mse_sum / n,
        "mae": mae_sum / n,
        "psnr": psnr_sum / n,
        "rmse": rmse_sum / n,
        "ssim": ssim_sum / n,
    }
    if train_encoder:
        out["ce"] = ce_sum / n
    return out


def _ds_gather_state_dict(module: torch.nn.Module) -> dict:
    """
    DeepSpeed ZeRO 下安全获取 module 的 state_dict（仅在 rank0 调用）。
    非 DeepSpeed 或未使用 ZeRO 时，直接返回 module.state_dict()。
    """
    if (deepspeed is None) or (DS_GatheredParameters is None):
        return module.state_dict()

    # 若不是 ZeRO 优化器，直接返回
    params = [p for p in module.parameters() if hasattr(p, "ds_status")]
    if len(params) == 0:
        return module.state_dict()

    # ZeRO：需要聚合参数后再取 state_dict
    state_dict = {}
    for name, param in module.named_parameters():
        if hasattr(param, "ds_id") and hasattr(param, "ds_status"):
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                with DS_GatheredParameters([param], modifier_rank=0):
                    state_dict[name] = param.detach().cpu().clone()
            else:
                state_dict[name] = param.detach().cpu().clone()
        else:
            state_dict[name] = param.detach().cpu().clone()

    # buffer 同样取一下
    for name, buf in module.named_buffers():
        state_dict[f"_buffer_{name}"] = buf.detach().cpu().clone()
    return state_dict


def train_one_epoch(
    *args,
    model,
    encoder=None,
    optimizer=None,
    criterion=None,
    train_loader=None,
    val_loader=None,
    device=None,
    epoch: int = 0,
    config=None,
    is_logger: bool = True,
    log_file: Optional[str] = None,
    results_dir=None,
    coef: float = 0.01,                   # MoE负载均衡因子
    lr_scheduler=None,                    # 学习率调度器
    scheduler_step_mode: str = "per_step",
    accum_steps: int = 1,                 # 梯度累计
    vis_now: bool = False,
    input_inverse_transform: Optional[Callable] = None,
    output_inverse_transform: Optional[Callable] = None,
    use_wandb: bool = False,
    wandb_module=None,
    early_stopper=None,
    best_val_loss: float = float("inf"),
    best_model_path: Optional[str] = None,
    best_expert_path: Optional[str] = None,
    best_encoder_path: Optional[str] = None,
    best_router_path: Optional[str] = None,
    last_model_path: Optional[str] = None,
    last_expert_path: Optional[str] = None,
    last_encoder_path: Optional[str] = None,
    last_router_path: Optional[str] = None,
    experts_name: Optional[list] = None,
    data_dict: Optional[dict] = None,
    metrics_module=None,
    tqdm_module=None,
    profile_timing: bool = False,
    amp_enabled: bool = False,
    encoder_frozen: bool = False,
    train_encoder: bool = False,
    tb_writer: Optional[SummaryWriter] = None,
    # ===== 新增参数（可不传） =====
    engine: Optional[Any] = None,         # DeepSpeed engine（传入则启用 DS 路径）
    use_deepspeed: bool = False,          # 切换标志，默认 False；若 engine 非 None 将被自动置 True
    **kwargs,
):
    """
    深度保持你原本行为一致，只在“优化步、学习率、保存、is_main”处自动切换到 DeepSpeed。
    """
    use_deepspeed = use_deepspeed or (engine is not None)
    tqdm = tqdm_module.tqdm if tqdm_module is not None else None

    # 记录器启用条件（再与 DeepSpeed 主进程判定结合）
    tb_active = bool(tb_writer) and _is_main_process(is_logger, engine)
    type_weight_hist_sample = None
    max_type_weight_hist_samples = 65536
    image_log_limit = 4

    start_time = time.time()
    model.train()
    if encoder is not None:
        encoder.train()

    running_train_loss = 0.0
    running_aux_loss = 0.0
    micro_count = 0
    nan_detected = False
    use_amp = bool(amp_enabled)

    # 读取 router / DDP 信息（保持你原逻辑）
    router_type = model.module.moe.router_type if hasattr(model, "module") else model.moe.router_type
    if "adamv" == router_type:
        router = model.module.moe.router if hasattr(model, "module") else model.moe.router
        assert hasattr(router, "step_validation"), "adamv router must impl. function step_validation"

    is_ddp_like = hasattr(model, "no_sync")  # DDP 情况
    num_steps = len(train_loader)

    # 分布式 sampler 设 epoch（原逻辑保留）
    if getattr(config, "distributed", None) and getattr(config.distributed, "use_distributed", False):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

    # 优化器归零：DeepSpeed 用 engine.zero_grad()
    if use_deepspeed:
        engine.zero_grad(set_to_none=True)
    else:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

    pbar_iter = train_loader
    if tqdm is not None:
        pbar_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{getattr(config, 'epochs', '?')}",
            leave=False,
            disable=not _is_main_process(is_logger, engine),
        )

    for step, batch in enumerate(pbar_iter):
        global_step = epoch * num_steps + step
        inputs = batch['input'].to(device, non_blocking=True)
        targets = batch['output'].to(device, non_blocking=True)
        if train_encoder:
            labels = batch['v_type'].to(device, non_blocking=True)

        # 编码器（可冻结）
        grad_ctx = torch.no_grad() if (encoder is not None and encoder_frozen) else nullcontext()
        with grad_ctx:
            with maybe_autocast(use_amp, device):
                if encoder is not None:
                    encoded, weights, _ = encoder(inputs)
                else:
                    encoded, weights = inputs, None

        # 释放原始 inputs
        del inputs

        if tb_active and type_weight_hist_sample is None and weights is not None:
            flat_weights = weights.detach().float().cpu().reshape(-1)
            if flat_weights.numel() > max_type_weight_hist_samples:
                flat_weights = flat_weights[:max_type_weight_hist_samples]
            type_weight_hist_sample = flat_weights

        # 是否为该累计组的“最后一个 micro”
        last_micro = ((step + 1) % accum_steps == 0) or ((step + 1) == num_steps)

        # DDP 同步控制：DeepSpeed 自管同步；仅在非 DeepSpeed 的 DDP 下使用 no_sync()
        sync_ctx = (
            (model.no_sync() if (is_ddp_like and not last_micro and not use_deepspeed) else nullcontext())
        )

        step_has_nan = False
        with sync_ctx:
            with maybe_autocast(use_amp, device):
                preds, aux_loss = model(encoded, weights)
            if aux_loss is None:
                aux_loss = preds.new_zeros(())
            if train_encoder:
                loss_dict = criterion(preds, targets, weights, labels)
            else:
                loss_dict = criterion(preds, targets)

            # 未缩放真实训练损失
            loss_raw = loss_dict["loss"] + coef * aux_loss

            if not torch.isfinite(loss_raw).item():
                step_has_nan = True
                nan_detected = True
            else:
                # 按组缩放
                current_group_size = accum_steps if not last_micro else ((step % accum_steps) + 1)
                loss_for_backward = loss_raw / current_group_size

                if use_deepspeed:
                    # DeepSpeed 反传
                    engine.backward(loss_for_backward)
                else:
                    loss_for_backward.backward()

                # 统计（用未缩放口径）
                running_train_loss += loss_raw.item()
                running_aux_loss += aux_loss.item()
                micro_count += 1

        if step_has_nan:
            if use_deepspeed:
                engine.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)
            if _is_main_process(is_logger, engine):
                print(f"[NaN Detected] loss became NaN at step {step + 1}, aborting epoch early.")
            break

        if last_micro:
            # 优化步
            if use_deepspeed:
                engine.step()
                engine.zero_grad(set_to_none=True)
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # 学习率调度（per_step）
            _maybe_step_scheduler(scheduler_step_mode, "per_step", lr_scheduler, engine)

            # 记录 LR
            if tb_active:
                current_lr = _get_current_lr(optimizer, engine, lr_scheduler)
                tb_writer.add_scalar("train/learning_rate", current_lr, global_step)

        # tqdm 显示
        if _is_main_process(is_logger, engine) and tqdm is not None:
            pbar_iter.set_postfix({"train_loss": f"{loss_raw.item():.6f}"})

        encoded = None
        weights = None

    # —— 训练集平均 —— #
    if nan_detected and micro_count == 0:
        avg_train_loss = float("nan")
        avg_aux_loss = float("nan")
    else:
        avg_train_loss = running_train_loss / max(1, micro_count)
        avg_aux_loss = running_aux_loss / max(1, micro_count)

    epoch_step = epoch + 1
    if tb_active:
        tb_writer.add_scalar("train/epoch_loss", avg_train_loss, epoch_step)
        tb_writer.add_scalar("train/epoch_aux_loss", avg_aux_loss, epoch_step)

    # —— 验证 —— #
    if nan_detected:
        val_stats = {
            "val_loss": float("inf"),
            "mse": float("nan"),
            "mae": float("nan"),
            "psnr": float("nan"),
            "rmse": float("nan"),
            "ssim": float("nan"),
        }
        if train_encoder:
            val_stats["ce"] = float("nan")
        val_loss = float("inf")
    else:
        val_stats = _evaluate_one_epoch(
            epoch,
            getattr(config, "epochs", 0),
            model,
            encoder,
            _is_main_process(is_logger, engine),
            val_loader,
            device,
            criterion,
            metrics_module,
            tqdm,
            amp_enabled=use_amp,
            train_encoder=train_encoder,
            engine=engine,
        )
        val_loss = val_stats["val_loss"]

    # —— 记录到 TensorBoard —— #
    if tb_active:
        tb_writer.add_scalar("val/loss", val_stats.get("val_loss", float("nan")), epoch_step)
        tb_writer.add_scalar("val/psnr", val_stats.get("psnr", float("nan")), epoch_step)
        tb_writer.add_scalar("val/mse", val_stats.get("mse", float("nan")), epoch_step)
        tb_writer.add_scalar("val/mae", val_stats.get("mae", float("nan")), epoch_step)
        tb_writer.add_scalar("val/rmse", val_stats.get("rmse", float("nan")), epoch_step)
        tb_writer.add_scalar("val/ssim", val_stats.get("ssim", float("nan")), epoch_step)
        tb_writer.add_scalars(
            "loss/epoch",
            {"train": avg_train_loss, "val": val_stats.get("val_loss", float("nan"))},
            epoch_step,
        )
        if train_encoder:
            tb_writer.add_scalar("val/ce", val_stats.get("ce", float("nan")), epoch_step)

    # —— Router 自适应控制（原逻辑保留）—— #
    if router_type == 'adamv':
        signal = router.step_validation(val_loss)
        if is_ddp_like and not use_deepspeed:
            signal_tensor = torch.tensor([1 if signal == "should_break" else 0], device=device, dtype=torch.int64)
            dist.all_reduce(signal_tensor, op=dist.ReduceOp.MAX)
            should_break = bool(signal_tensor.item())
        else:
            # DeepSpeed 或非 DDP：rank0 直接判定
            should_break = (signal == "should_break")

        if should_break:
            router.k = max(1, router.k - 1)
            router.fixed = True

            # 广播给其余 rank
            if (is_ddp_like and not use_deepspeed) or (use_deepspeed and dist.is_initialized()):
                k_tensor = torch.tensor([router.k], device=device)
                dist.broadcast(k_tensor, src=0)
                router.k = int(k_tensor.item())

            if _is_main_process(is_logger, engine):
                print(f'epoch: {epoch} AES probe failed -> fix top_k = {router.k}')

    # —— 文本日志 / W&B —— #
    if _is_main_process(is_logger, engine) and (log_file is not None):
        cols = [
            f"    {epoch+1}",
            f"{avg_train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{val_stats['mae']:.6f}",
            f"{val_stats['mse']:.6f}",
            f"{val_stats['psnr']:.6f}",
            f"{val_stats['rmse']:.6f}",
            f"{val_stats['ssim']:.6f}",
        ]
        if train_encoder:
            cols.append(f"{val_stats['ce']:.6f}")
        line = "    |    ".join(cols) + "    |\n"
        with open(log_file, "a") as f:
            f.write(line)

    if use_wandb and wandb_module is not None:
        wandb_log = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "learning_rate": _get_current_lr(optimizer, engine, lr_scheduler),
            "val/psnr": val_stats["psnr"],
            "val/mse": val_stats["mse"],
            "val/mae": val_stats["mae"],
            "val/rmse": val_stats["rmse"],
            "val/ssim": val_stats["ssim"],
        }
        if train_encoder:
            wandb_log["val/ce"] = val_stats["ce"]
        wandb_module.log(wandb_log)

    if tb_active and type_weight_hist_sample is not None:
        tb_writer.add_histogram("encoder/type_weights", type_weight_hist_sample, epoch_step)

    # —— 保存（最佳 / 最新）—— #
    def _save_checkpoints(tag_path: Optional[str], best: bool):
        if not _is_main_process(is_logger, engine):
            return
        if tag_path is None:
            return

        # 统一取得“可保存”的 model / encoder / router
        model_for_save = model.module if hasattr(model, "module") else model
        encoder_for_save = encoder.module if (encoder is not None and hasattr(encoder, "module")) else encoder
        router_for_save = (model_for_save.moe.router if hasattr(model_for_save, "moe") else None)

        base_ckpt = {
            "epoch": epoch,
            "optimizer_state_dict": None if use_deepspeed else (optimizer.state_dict() if optimizer is not None else None),
            "val_loss": val_loss,
            "metrics": {
                "psnr": val_stats["psnr"],
                "mse": val_stats["mse"],
                "mae": val_stats["mae"],
                "rmse": val_stats["rmse"],
                "ssim": val_stats["ssim"],
            },
            "data_dict": data_dict,
        }

        # 1) DeepSpeed：保存引擎 checkpoint（会把分布式分片都保存好）
        if use_deepspeed:
            # tag 用 “best” 或 “last”+epoch
            tag = "best" if best else f"last-ep{epoch+1}"
            engine.save_checkpoint(os.path.dirname(tag_path), tag=tag)

            # 额外在 rank0 保存“聚合后（state_dict）”的模块：model/encoder/router
            attach = {}
            attach["model_state_dict"] = _ds_gather_state_dict(model_for_save)
            if encoder_for_save is not None:
                attach["encoder_state_dict"] = _ds_gather_state_dict(encoder_for_save)
            if router_for_save is not None:
                attach["router_state_dict"] = _ds_gather_state_dict(router_for_save)

            # 你原逻辑中 “只保存 expert” 的分支也保留
            if experts_name is not None and len(experts_name) == 1 and experts_name[0] != "all":
                if hasattr(model_for_save, "experts") and len(model_for_save.experts) > 0:
                    attach["expert_state_dict"] = _ds_gather_state_dict(model_for_save.experts[0])

            torch.save({**base_ckpt, **attach}, tag_path)

        else:
            # 非 DeepSpeed：与原逻辑一致
            checkpoint = {
                **base_ckpt,
                "model_state_dict": model_for_save.state_dict(),
            }
            if train_encoder or (experts_name and len(experts_name) == 1 and experts_name[0] == "all"):
                checkpoint.pop("model_state_dict", None)  # 走 MOE 格式
                if router_for_save is not None:
                    checkpoint["router_state_dict"] = router_for_save.state_dict()
            if encoder_for_save is not None:
                checkpoint["encoder_state_dict"] = encoder_for_save.state_dict()

            torch.save(checkpoint, tag_path)

            if experts_name is not None and len(experts_name) == 1 and experts_name[0] != "all":
                if hasattr(model_for_save, "experts") and len(model_for_save.experts) > 0:
                    torch.save(
                        {"expert_state_dict": model_for_save.experts[0].state_dict()},
                        (best_expert_path if best else last_expert_path),
                    )

    # 保存最佳
    if _is_main_process(is_logger, engine) and (val_loss < best_val_loss):
        best_val_loss = val_loss
        _save_checkpoints(best_model_path, best=True)

    # 保存最新
    if _is_main_process(is_logger, engine) and (last_model_path is not None):
        _save_checkpoints(last_model_path, best=False)

    # —— 打印概要 —— #
    if _is_main_process(is_logger, engine):
        print(f"Epoch {epoch+1}/{getattr(config, 'epochs', '?')}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val   Loss: {val_loss:.6f}")

    # —— 可视化（仅主进程 & 触发时）—— #
    if _is_main_process(is_logger, engine) and vis_now and (visualize_results is not None):
        vis_batch = next(iter(val_loader))
        inputs = vis_batch['input'].to(device, non_blocking=True)
        targets = vis_batch['output'].to(device, non_blocking=True)
        with torch.no_grad():
            with maybe_autocast(use_amp, device):
                if encoder is not None:
                    vis_encoded, vis_weights, _ = encoder(inputs)
                else:
                    vis_encoded, vis_weights = inputs, None
                preds, _ = model(vis_encoded, vis_weights)

        inputs_v = input_inverse_transform(inputs) if input_inverse_transform else inputs
        if output_inverse_transform:
            preds_v = output_inverse_transform(preds)
            targets_v = output_inverse_transform(targets)
        else:
            preds_v, targets_v = preds, targets

        num_samples = min(image_log_limit, inputs_v.shape[0])
        wandb_run = wandb_module if (use_wandb and wandb_module is not None) else None
        tb = tb_writer if tb_active else None

        visualize_results(
            inputs_v, targets_v, preds_v,
            save_dir=results_dir / f"vis_epoch_{epoch+1}",
            max_samples=num_samples,
            tb_writer=tb,
            wandb_run=wandb_run,
            global_step=epoch,
        )

        save_type_predictions_txt(
            logits=vis_weights,
            batch=vis_batch,
            save_dir=results_dir / f"vis_epoch_{epoch+1}",
            epoch=epoch,
            config=config,
            filename="type_predictions.txt",
            append=True,
            is_logger=_is_main_process(is_logger, engine),
        )

        analyze_fourier_domain(
            inputs_v, targets_v, preds_v,
            save_dir=results_dir / f"fourier_analysis_epoch_{epoch+1}",
            max_samples=num_samples,
            tb_writer=tb,
            wandb_run=wandb_run,
            global_step=epoch,
        )

    # —— 早停（主进程判定，后广播）—— #
    stop_flag = 0
    if getattr(config, "early_stop", False):
        if _is_main_process(is_logger, engine) and (early_stopper is not None) and (not nan_detected):
            if early_stopper.step(val_loss, epoch):
                stop_flag = 1

        # 广播（DeepSpeed/原生均可用 torch.distributed）
        if dist.is_initialized():
            flag_tensor = torch.tensor([stop_flag], device=device, dtype=torch.int32)
            torch.distributed.broadcast(flag_tensor, src=0)
            stop_flag = int(flag_tensor.item())

        if (stop_flag == 1) and _is_main_process(is_logger, engine):
            print(f"[EARLY STOP] stop at epoch={epoch+1}, best_val_loss={best_val_loss:.6f}")

    if nan_detected:
        stop_flag = 1
        if dist.is_initialized():
            flag_tensor = torch.tensor([stop_flag], device=device, dtype=torch.int32)
            torch.distributed.broadcast(flag_tensor, src=0)
            stop_flag = int(flag_tensor.item())

    # —— 学习率调度（per_epoch）—— #
    _maybe_step_scheduler(scheduler_step_mode, "per_epoch", lr_scheduler, engine)

    # —— 耗时 —— #
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if _is_main_process(is_logger, engine):
        print('Training time', total_time_str)

    stats = {
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": val_stats["val_loss"],
        "psnr": val_stats["psnr"],
        "mse": val_stats["mse"],
        "mae": val_stats["mae"],
        "rmse": val_stats["rmse"],
        "ssim": val_stats["ssim"],
    }
    return stats, best_val_loss, stop_flag
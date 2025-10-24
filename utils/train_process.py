import torch
from typing import Optional, Callable
import time
from contextlib import nullcontext
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from .plot_fig import analyze_fourier_domain, visualize_results, save_type_predictions_txt
import datetime

@torch.no_grad()
def _evaluate_one_epoch(
    model,
    encoder,
    # classifier,
    val_loader,
    device,
    criterion,
    metrics_module,  # 需有 calculate_psnr(pred, tgt)
    amp_enabled: bool = False,
    train_encoder: bool = False,
):
    model.eval()
    if encoder is not None:
        encoder.eval()
    # if classifier is not None:
    #     classifier.eval()
    val_loss = 0.0
    mse_sum, mae_sum, psnr_sum, ce_sum = 0.0, 0.0, 0.0, 0.0

    for batch in val_loader:
        inputs  = batch['input'].to(device, non_blocking=True)
        targets = batch['output'].to(device, non_blocking=True)
        if train_encoder:
            labels = batch['v_type'].to(device, non_blocking=True)
        targets = targets.to(dtype=torch.float32)

        if encoder is not None:
            if amp_enabled:
                with torch.amp.autocast(device_type=device.type, enabled=True, dtype=torch.bfloat16):
                    encoded, weights, _ = encoder(inputs)
            else:
                encoded, weights, _ = encoder(inputs)
            encoded = encoded.to(dtype=torch.float32).detach()
            weights = weights.to(dtype=torch.float32).detach() if weights is not None else None
        else:
            encoded = inputs.to(dtype=torch.float32)
            weights = None
        with torch.amp.autocast(device_type=device.type, enabled=False):
            preds, aux_loss = model(encoded, weights)
        if aux_loss is None:
            aux_loss = preds.new_zeros(())

        # 兼容 criterion 返回 (loss, loss_g1v, loss_g2v)
        if train_encoder:
            loss_dict = criterion(preds, targets, weights, labels)
        else:
            loss_dict = criterion(preds, targets)
        
        val_loss += loss_dict["loss"].item()
        mse_sum  += loss_dict["l2"].item()
        mae_sum  += loss_dict["l1"].item()
        psnr_sum += metrics_module.calculate_psnr(preds, targets)
        if train_encoder:
            ce_sum += loss_dict["ce"].item()
            
    n = max(1, len(val_loader))
    if train_encoder:
        return {
            "val_loss": val_loss / n,
            "mse": mse_sum / n,
            "mae": mae_sum / n,
            "psnr": psnr_sum / n,
            "ce": ce_sum / n,
        }
    return {
        "val_loss": val_loss / n,
        "mse": mse_sum / n,
        "mae": mae_sum / n,
        "psnr": psnr_sum / n
    }


def train_one_epoch(
    *args,
    model,
    encoder=None,
    # classifier=None,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    epoch: int,
    config,
    is_logger: bool,
    log_file: Optional[str],
    results_dir,
    # MoE负载均衡因子
    coef: float = 0.01,
    # 学习率调度器
    lr_scheduler=None,
    scheduler_step_mode: str = "per_step",   # "per_step" 或 "per_epoch"
    # 梯度累计
    accum_steps: int = 1,
    # 可视化 & 反归一化
    vis_now: bool = False,                   # 是否在本 epoch 可视化
    input_inverse_transform: Optional[Callable] = None,
    output_inverse_transform: Optional[Callable] = None,
    # WandB
    use_wandb: bool = False,
    wandb_module=None,
    # 早停
    early_stopper=None,                      # 需有 step(val_loss, epoch)->bool
    # 最佳模型保存
    best_val_loss: float = float("inf"),
    best_model_path: Optional[str] = None,
    best_expert_path: Optional[str] = None,
    best_encoder_path: Optional[str] = None,
    last_model_path: Optional[str] = None,
    last_expert_path: Optional[str] = None,
    last_encoder_path: Optional[str] = None,
    experts_name: Optional[list] = None,
    data_dict: Optional[dict] = None,
    # 其他工具
    metrics_module=None,
    tqdm_module=None,   # 传入 tqdm（避免在函数内硬依赖）
    profile_timing: bool = False,            # 是否记录耗时
    amp_enabled: bool = False,
    amp_scaler: Optional[torch.amp.GradScaler] = None,
    encoder_frozen: bool = False,
    # 是否训练encoder
    train_encoder: bool = False,
    tb_writer: Optional[SummaryWriter] = None,
    **kwargs,
):
    """
    进行一个 epoch 的完整训练与验证，返回 (stats_dict, best_val_loss, stop_flag)
    - 等效全局 batch = per_gpu_batch * world_size * accum_steps
    - 若使用 DDP，前 accum_steps-1 次 micro step 用 no_sync() 以减少通信
    - 调度器：per_step 在“优化步”后 step；per_epoch 在 epoch 末 step
    """
    def mem():
        """打印当前 GPU 显存状态"""
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(device) / 1e9  # 已分配显存
        reserved  = torch.cuda.memory_reserved(device)  / 1e9  # 已缓存（已申请但未使用）显存
        max_alloc = torch.cuda.max_memory_allocated(device) / 1e9  # 运行以来的峰值显存
        print(f"[{device}] allocated={allocated:.2f} GB | reserved={reserved:.2f} GB | max={max_alloc:.2f} GB")
    
    assert metrics_module is not None, "metrics_module 需提供 calculate_psnr(pred, tgt)"
    tqdm = tqdm_module.tqdm if tqdm_module is not None else None
    tb_active = bool(tb_writer) and bool(is_logger)
    router_hist_sample = None
    max_router_hist_samples = 65536
    image_log_limit = 3

    start_time = time.time()
    model.train()
    if encoder is not None:
        encoder.train()
    running_train_loss = 0.0
    running_aux_loss = 0.0
    micro_count = 0
    optim_count = 0
    num_steps = len(train_loader)
    nan_detected = False
    use_amp = bool(amp_enabled) and (amp_scaler is not None)

    # router type判断
    router_type = model.module.router_type if hasattr(model, "module") else model.router_type
    if "adamv" == router_type:
        router = model.module.router if hasattr(model, "module") else model.router
        assert hasattr(router, "step_validation"), "adamv router must impl. function step_validation"
    
    # DDP 判断
    is_ddp = hasattr(model, "no_sync")

    # 分布式 sampler 设 epoch
    if getattr(config, "distributed", None) and getattr(config.distributed, "use_distributed", False):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
 
    optimizer.zero_grad(set_to_none=True)

    val_loss = float("inf")
    
    pbar_iter = train_loader
    if tqdm is not None:
        pbar_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False, disable=not is_logger)

    num_steps = len(train_loader)
    micro_in_group = 0  # 追踪当前累计组内的 micro 数
    
    for step, batch in enumerate(pbar_iter):
        global_step = epoch * num_steps + step
        inputs  = batch['input'].to(device, non_blocking=True)
        targets = batch['output'].to(device, non_blocking=True)
        
        if encoder is not None:
            grad_ctx = torch.no_grad() if encoder_frozen else nullcontext()
            with grad_ctx:
                if use_amp:
                    with torch.amp.autocast(device_type=device.type, enabled=True, dtype=torch.bfloat16):
                        encoded, weights, _ = encoder(inputs)
                else:
                    encoded, weights, _ = encoder(inputs)
        else:
            encoded = inputs
            weights = None
        
        encoded = encoded.to(dtype=torch.float32)
        weights = weights.to(dtype=torch.float32) if weights is not None else None
        targets = targets.to(dtype=torch.float32)
        
        del inputs
        
        # mem()
        
        if encoder_frozen:
            encoded = encoded.detach()
        
        if encoder_frozen and weights is not None:
            weights = weights.detach()
        
        if tb_active and router_hist_sample is None and weights is not None:
            flat_weights = weights.detach().float().cpu().reshape(-1)
            if flat_weights.numel() > max_router_hist_samples:
                flat_weights = flat_weights[:max_router_hist_samples]
            router_hist_sample = flat_weights

        if train_encoder:
            labels = batch['v_type'].to(device, non_blocking=True)
        
        # —— 计算该 step 是否是这一累计组的“最后一个” —— #
        # 规则：自然分组 + 末尾不足一组也强制结算
        last_micro = ((step + 1) % accum_steps == 0) or ((step + 1) == num_steps)

        # DDP 同步控制
        sync_ctx = (model.no_sync() if (is_ddp and not last_micro) else nullcontext())

        step_has_nan = False
        with sync_ctx:
            with torch.amp.autocast(device_type=device.type, enabled=False):
                preds, aux_loss = model(encoded, weights)
            if aux_loss is None:
                aux_loss = preds.new_zeros(())
            if train_encoder:
                loss_dict = criterion(preds, targets, weights, labels)
            else:
                loss_dict = criterion(preds, targets)
            # —— 未缩放的“真实训练损失”（用于日志统计） —— #
            loss_raw = loss_dict["loss"] + coef * aux_loss

            if not torch.isfinite(loss_raw).item():
                step_has_nan = True
                nan_detected = True
            else:
                # —— 用于反传的缩放 —— #
                current_group_size = accum_steps if not last_micro else ( (step % accum_steps) + 1 )
                loss = loss_raw / current_group_size
                if use_amp:
                    amp_scaler.scale(loss).backward()
                else:
                    loss.backward()

                # —— 统计（用未缩放的口径，便于和 val 对齐） —— #
                running_train_loss += loss_raw.item()
                running_aux_loss   += aux_loss.item()
                micro_in_group     += 1
                micro_count        += 1
                if tb_active:
                    tb_writer.add_scalar("train/micro_loss", loss_raw.item(), global_step)
                    tb_writer.add_scalar("train/aux_loss", aux_loss.item(), global_step)

        if step_has_nan:
            optimizer.zero_grad(set_to_none=True)
            if is_logger:
                print(f"[NaN Detected] loss became NaN at step {step + 1}, aborting epoch early.")
            break

        if last_micro:
            if use_amp:
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optim_count += 1
            micro_in_group = 0  # 结算完当前组，清零

            # 学习率调度（按步）
            if lr_scheduler is not None and scheduler_step_mode == "per_step":
                lr_scheduler.step()
            if tb_active:
                current_lr = optimizer.param_groups[0]["lr"]
                tb_writer.add_scalar("train/learning_rate", current_lr, global_step)
                tb_writer.add_scalar("train/optim_steps_in_epoch", optim_count, global_step)

        if is_logger and tqdm is not None:
            pbar_iter.set_postfix({"train_loss": f"{loss_raw.item():.6f}"})  # 展示未缩放损失
        encoded = None
        weights = None
        
    # —— 训练集 loss（micro-step 平均）——
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
        tb_writer.add_scalar("epoch/micro_steps", micro_count, epoch_step)
        tb_writer.add_scalar("epoch/optim_steps", optim_count, epoch_step)

    # —— 验证 —— #
    if nan_detected:
        if train_encoder:
            val_stats = {
                "val_loss": float("inf"),
                "mse": float("nan"),
                "mae": float("nan"),
                "psnr": float("nan"),
                "ce": float("nan"),
            }
        else:
            val_stats = {
                "val_loss": float("inf"),
                "mse": float("nan"),
                "mae": float("nan"),
                "psnr": float("nan"),
            }
        val_loss = float("inf")
    else:
        val_stats = _evaluate_one_epoch(
            model,
            encoder,
            val_loader,
            device,
            criterion,
            metrics_module,
            amp_enabled=use_amp,
        )
        val_loss = val_stats["val_loss"]
    if tb_active:
        tb_writer.add_scalar("val/loss", val_stats.get("val_loss", float("nan")), epoch_step)
        tb_writer.add_scalar("val/psnr", val_stats.get("psnr", float("nan")), epoch_step)
        tb_writer.add_scalar("val/mse", val_stats.get("mse", float("nan")), epoch_step)
        tb_writer.add_scalar("val/mae", val_stats.get("mae", float("nan")), epoch_step)
        if train_encoder:
            tb_writer.add_scalar("val/ce", val_stats.get("ce", float("nan")), epoch_step)
    if router_type == 'adamv':
        signal = router.step_validation(val_loss)
        # 多进程之间同步信号
        if is_ddp:
            signal = torch.tensor([1 if signal == "should_break" else 0]
                                  , device=device, dtype=torch.int64) # bool信号常用int64
            dist.all_reduce(signal, op=dist.ReduceOp.MAX)
            should_break = bool(signal.item())
        else:
            should_break = (signal == "should_break")
        
        if should_break:
            router.k = max(1, router.k - 1)
            router.fixed = True
            
            if is_ddp:
                k_tensor = torch.tensor([router.k], device=device)
                dist.broadcast(k_tensor, src=0) # broadcast包含同步原语
                router.k = int(k_tensor.item())
        
            if is_logger:
                print(f'epoch: {epoch} AES probe failed -> fix top_k = {router.k}')
                    
    # —— 日志输出 & WandB —— #
    if is_logger and log_file is not None:
        if train_encoder:
            with open(log_file, "a") as f:
                f.write(
                    f"    {epoch+1}    |    {avg_train_loss:.6f}    |    {val_loss:.6f}    |    "
                    f"{val_stats['mae']:.6f}    |    {val_stats['mse']:.6f}    |    {val_stats['psnr']:.6f}    |    {val_stats['ce']:.6f}    |\n"
                )
        else:
            with open(log_file, "a") as f:
                f.write(
                    f"    {epoch+1}    |    {avg_train_loss:.6f}    |    {val_loss:.6f}    |    "
                    f"{val_stats['mae']:.6f}    |    {val_stats['mse']:.6f}    |    {val_stats['psnr']:.6f}    |\n"
                )
    if use_wandb and wandb_module is not None:
        if train_encoder:
            wandb_log = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "val/psnr": val_stats["psnr"],
                "val/mse": val_stats["mse"],
                "val/mae": val_stats["mae"],
                "val/ce": val_stats["ce"],
                "optim_steps_in_epoch": optim_count,
            }
        else:
            wandb_log = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "val/psnr": val_stats["psnr"],
                "val/mse": val_stats["mse"],
                "val/mae": val_stats["mae"],
                "optim_steps_in_epoch": optim_count,
            }
        wandb_module.log(wandb_log)

    if tb_active and router_hist_sample is not None:
        tb_writer.add_histogram("router/gates", router_hist_sample, epoch_step)

    # —— 保存最佳模型（仅主进程）—— #
    if is_logger and (val_loss < best_val_loss):
        best_val_loss = val_loss
        model_to_save = model.module if (getattr(config, "distributed", None) and getattr(config.distributed, "use_distributed", False) and hasattr(model, "module")) else model
        encoder_to_save = encoder.module if (encoder is not None and hasattr(encoder, "module")) else encoder
        # classifier_to_save = classifier.module if (classifier is not None and hasattr(classifier, "module")) else classifier

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': {"psnr": val_stats["psnr"], "mse": val_stats["mse"], "mae": val_stats["mae"]},
            'data_dict': data_dict
        }
        if encoder_to_save is not None:
            checkpoint['encoder_state_dict'] = encoder_to_save.state_dict()
        # if classifier_to_save is not None:
        #     checkpoint['classifier_state_dict'] = classifier_to_save.state_dict()
        if train_encoder:
            torch.save(encoder_to_save.state_dict(), best_encoder_path)
        else:
            torch.save(checkpoint, best_model_path)
            
            if experts_name is not None and len(experts_name) == 1 and best_expert_path is not None:
                # 仅示例：若你的模型结构中存在 experts[0]
                if hasattr(model_to_save, "experts") and len(model_to_save.experts) > 0:
                    torch.save({
                        'expert_state_dict': model_to_save.experts[0].state_dict()
                    }, best_expert_path)

    if is_logger and last_model_path is not None:
        model_to_save = model.module if (getattr(config, "distributed", None) and getattr(config.distributed, "use_distributed", False) and hasattr(model, "module")) else model
        encoder_to_save = encoder.module if (encoder is not None and hasattr(encoder, "module")) else encoder
        # classifier_to_save = classifier.module if (classifier is not None and hasattr(classifier, "module")) else classifier

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': {"psnr": val_stats["psnr"], "mse": val_stats["mse"], "mae": val_stats["mae"]},
            'data_dict': data_dict
        }
        if encoder_to_save is not None:
            checkpoint['encoder_state_dict'] = encoder_to_save.state_dict()
        # if classifier_to_save is not None:
        #     checkpoint['classifier_state_dict'] = classifier_to_save.state_dict()
        if train_encoder:
            torch.save(encoder_to_save.state_dict(), last_encoder_path)
        else:
            torch.save(checkpoint, last_model_path)

            if experts_name is not None and len(experts_name) == 1 and last_expert_path is not None:
                if hasattr(model_to_save, "experts") and len(model_to_save.experts) > 0:
                    torch.save({
                        'expert_state_dict': model_to_save.experts[0].state_dict()
                    }, last_expert_path)

    # —— 打印本 epoch 概要（仅主进程）—— #
    if is_logger:
        if train_encoder:
            print(f"Epoch {epoch+1}/{config.epochs}:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val   Loss: {val_loss:.6f}")
            print(f"  PSNR: {val_stats['psnr']:.2f} dB")
            print(f"  MSE : {val_stats['mse']:.6f}")
            print(f"  MAE : {val_stats['mae']:.6f}")
            print(f"  CE : {val_stats['ce']:.6f}")
            print(f"  AuxLoss : {avg_aux_loss:.2f}")
        else:
            print(f"Epoch {epoch+1}/{config.epochs}:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val   Loss: {val_loss:.6f}")
            print(f"  PSNR: {val_stats['psnr']:.2f} dB")
            print(f"  MSE : {val_stats['mse']:.6f}")
            print(f"  MAE : {val_stats['mae']:.6f}")
            print(f"  AuxLoss : {avg_aux_loss:.2f}")

    # —— 可视化（仅主进程 & 触发时）—— #
    if is_logger and vis_now and visualize_results is not None:
        vis_batch = next(iter(val_loader))
        inputs = vis_batch['input'].to(device, non_blocking=True)
        targets = vis_batch['output'].to(device, non_blocking=True)
        with torch.no_grad():
            if encoder is not None:
                if use_amp:
                    with torch.amp.autocast(device_type=device.type, enabled=True, dtype=torch.bfloat16):
                        vis_encoded, vis_weights, _ = encoder(inputs)
                else:
                    vis_encoded, vis_weights, _ = encoder(inputs)
                vis_encoded = vis_encoded.to(dtype=torch.float32).detach()
                vis_weights = vis_weights.to(dtype=torch.float32).detach() if vis_weights is not None else None
            else:
                vis_encoded = inputs.to(dtype=torch.float32)
                vis_weights = None
                
            with torch.amp.autocast(device_type=device.type, enabled=False):
                preds, _ = model(vis_encoded, vis_weights)

        if input_inverse_transform is not None:
            inputs_v = input_inverse_transform(inputs)
        else:
            inputs_v = inputs

        if output_inverse_transform is not None:
            preds_v = output_inverse_transform(preds)
            targets_v = output_inverse_transform(targets)
        else:
            preds_v, targets_v = preds, targets

        visualize_results(inputs_v, targets_v, preds_v, save_dir=results_dir / f"vis_epoch_{epoch+1}")
        
        save_type_predictions_txt(
            logits=vis_weights,     # encoder输出的未softmax logits
            batch=vis_batch,
            save_dir=results_dir / f"vis_epoch_{epoch+1}",
            epoch=epoch,
            config=config,          # 直接传 config，内部自动反转 type_id_specific
            filename="type_predictions.txt",
            append=True,
            is_logger=is_logger,
        )
        
        # 进行傅里叶域分析
        analyze_fourier_domain(inputs_v, targets_v, preds_v, save_dir=results_dir / f"fourier_analysis_epoch_{epoch+1}")

        if use_wandb and wandb_module is not None:
            # 只示例记录前三个
            for i in range(min(4, inputs_v.shape[0])):
                in_img  = inputs_v[i, 0].detach().float().cpu().numpy()
                tgt_img = (targets_v[i, 0] if targets_v.dim() > 3 else targets_v[i]).detach().float().cpu().numpy()
                prd_img = (preds_v[i, 0]   if preds_v.dim()   > 3 else preds_v[i]).detach().float().cpu().numpy()
                wandb_module.log({
                    f"sample_{i}/input_velocity": wandb_module.Image(in_img),
                    f"sample_{i}/target_seismic": wandb_module.Image(tgt_img),
                    f"sample_{i}/prediction_seismic": wandb_module.Image(prd_img),
                })
        if tb_active:
            num_samples = min(image_log_limit, inputs_v.shape[0])
            for idx in range(num_samples):
                tb_writer.add_image(f"samples/{idx}/input", inputs_v[idx], epoch_step)
                tb_writer.add_image(f"samples/{idx}/target", targets_v[idx], epoch_step)
                tb_writer.add_image(f"samples/{idx}/prediction", preds_v[idx], epoch_step)

    # —— 早停（仅主进程判定，后广播）—— #
    stop_flag = 0
    if getattr(config, "early_stop", False):
        if is_logger and early_stopper is not None and not nan_detected:
            if early_stopper.step(val_loss, epoch):
                stop_flag = 1

        if getattr(config, "distributed", None) and getattr(config.distributed, "use_distributed", False):
            device_for_flag = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            flag_tensor = torch.tensor([stop_flag], device=device_for_flag, dtype=torch.int32)
            torch.distributed.broadcast(flag_tensor, src=0)
            stop_flag = int(flag_tensor.item())

        if stop_flag == 1 and is_logger:
            print(f"[EARLY STOP] stop at epoch={epoch+1}, best_val_loss={best_val_loss:.6f}")

    if nan_detected:
        stop_flag = 1
        if getattr(config, "distributed", None) and getattr(config.distributed, "use_distributed", False):
            device_for_flag = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            flag_tensor = torch.tensor([stop_flag], device=device_for_flag, dtype=torch.int32)
            torch.distributed.broadcast(flag_tensor, src=0)
            stop_flag = int(flag_tensor.item())

    # —— 调度器（按 epoch）—— #
    if lr_scheduler is not None and scheduler_step_mode == "per_epoch":
        lr_scheduler.step()

    # —— 分布式 barrier（可选，与日志输出顺序相关）—— #
    if getattr(config, "distributed", None) and getattr(config.distributed, "use_distributed", False):
        torch.distributed.barrier()

    # —— 耗时 —— #
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if is_logger:
        print('Training time', total_time_str)
    if tb_active:
        tb_writer.add_scalar("epoch/duration_sec", total_time, epoch_step)

    # 返回统计与状态
    stats = {
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": val_stats["val_loss"],
        "psnr": val_stats["psnr"],
        "mse": val_stats["mse"],
        "mae": val_stats["mae"],
        "optim_steps": optim_count,
        "micro_steps": micro_count,
        "time_sec": total_time
    }
    return stats, best_val_loss, stop_flag

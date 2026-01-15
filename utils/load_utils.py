from typing import Dict, Iterable, Any, List, Optional, Tuple, Union
import torch
from collections import OrderedDict, defaultdict
import torch.nn as nn
from neuralop.models import ExpertFactory
import re
import os
from pathlib import Path

from config.seismic_moe_config import SPECIFIC_TYPE_VARIANTS

# =========================
# Label/Variant utilities
# =========================

_SPECIFIC_VARIANT_TO_BASE = {
    variant: base
    for base, variants in SPECIFIC_TYPE_VARIANTS.items()
    for variant in variants
}

_SPECIFIC_BASE_DEFAULT_VARIANT = {
    base: variants[0] for base, variants in SPECIFIC_TYPE_VARIANTS.items()
}

_EXPERT_FILE_PATTERN = re.compile(
    r'^best_expert_(?P<name>[A-Za-z][A-Za-z0-9_]*)_(?P<i>\d+)_(?P<label>[A-Za-z0-9_]+)\.pt$',
    re.IGNORECASE
)


def _strip_prefixes(key: str, prefixes: Iterable[str]) -> str:
    """剥离类似 'module.' 的前缀。"""
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key


# =========================
# Encoder loader
# =========================

def _extract_encoder_state_dict(
    source: Any,
    *,
    strip_prefixes: Iterable[str] = ("module.",),
) -> Dict[str, torch.Tensor]:
    """
    从任意形式的 checkpoint 中解析 encoder 的 state_dict。
    支持多种常见存储格式：
      - 直接是 state_dict
      - {'encoder_state_dict': state_dict}
      - {'state_dict': state_dict}
    并自动剥离 DDP 保存时的 ``module.`` 等前缀。
    """
    if source is None:
        raise ValueError("encoder checkpoint 为空，无法加载。")

    if not isinstance(source, dict):
        raise TypeError(f"无法解析类型为 {type(source)} 的 encoder checkpoint。期望字典或 state_dict。")

    if "encoder_state_dict" in source:
        state_dict = source["encoder_state_dict"]
    elif "state_dict" in source:
        state_dict = source["state_dict"]
    else:
        state_dict = source

    if not isinstance(state_dict, dict):
        raise ValueError("encoder checkpoint 中缺少有效的 state_dict。")

    cleaned: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if not isinstance(key, str):
            continue
        new_key = _strip_prefixes(key, strip_prefixes)
        cleaned[new_key] = tensor

    if not cleaned:
        raise ValueError("encoder state_dict 解析为空，请检查保存格式。")

    return cleaned


def load_encoder_weights(
    encoder: nn.Module,
    checkpoint: Optional[Union[str, Dict[str, Any]]],
    *,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = False,
    strip_prefixes: Iterable[str] = ("module.",),
    drop_prefixes: Iterable[str] = ("type_head.",),   # <- 新增：要跳过的 head 前缀
    drop_if_contains: Iterable[str] = (),              # <- 可选：按关键字丢弃
) -> Tuple[List[str], List[str]]:
    """
    将 encoder checkpoint 加载到给定模型中。
    返回 (missing_keys, unexpected_keys)。
    """
    if checkpoint is None:
        return [], []

    if isinstance(checkpoint, str):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Encoder checkpoint 不存在: {checkpoint}")
        loaded = torch.load(checkpoint, map_location=map_location)
    else:
        loaded = checkpoint

    state_dict = _extract_encoder_state_dict(loaded, strip_prefixes=strip_prefixes)

    # --------- 1) 显式丢弃 head / 指定前缀 ----------
    if drop_prefixes or drop_if_contains:
        filtered = {}
        dropped = []
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in drop_prefixes) or any(s in k for s in drop_if_contains):
                dropped.append(k)
                continue
            filtered[k] = v
        state_dict = filtered
        if dropped:
            print(f"[load_encoder_weights] dropped keys ({len(dropped)}): {dropped[:20]}"
                  + (" ..." if len(dropped) > 20 else ""))

    missing, unexpected = encoder.load_state_dict(state_dict, strict=strict)
    return list(missing), list(unexpected)


# =========================
# Expert checkpoint parsers
# =========================

def get_expert_dict(
    ckpt: Dict[str, Any],
    ddp_prefixes: Iterable[str] = ("module.",),
    expert_prefix: str = "experts.0.",
) -> Dict[str, torch.Tensor]:
    """从 state_dict 或保存的专家字典中提取第一个专家权重。
    - 支持单卡/多卡保存（自动剥离 ``module.`` 前缀）。
    - 若存在 ``experts.0.`` 前缀，仅保留其下的键并剥离该前缀。
    """
    raw_sd: Optional[Dict[str, Any]] = None
    if isinstance(ckpt, dict):
        for key in ("expert_state_dict", "state_dict", "model_state_dict"):
            maybe_sd = ckpt.get(key)
            if isinstance(maybe_sd, dict):
                raw_sd = maybe_sd
                break
        if raw_sd is None and all(isinstance(k, str) for k in ckpt.keys()):
            raw_sd = ckpt  # 直接就是 state_dict

    if raw_sd is None:
        raise TypeError("无法从 checkpoint 中提取 state_dict，检查保存格式是否符合保存逻辑。")

    if not isinstance(raw_sd, dict):
        raise TypeError("state_dict 应为字典，无法解析专家权重。")

    cleaned = OrderedDict()
    has_expert_prefix = False
    for k, v in raw_sd.items():
        if not isinstance(k, str):
            continue
        k2 = _strip_prefixes(k, ddp_prefixes)
        if k2.startswith(expert_prefix):
            if not has_expert_prefix:
                cleaned.clear()
                has_expert_prefix = True
            cleaned[k2[len(expert_prefix):]] = v
        elif not has_expert_prefix:
            cleaned[k2] = v

    if has_expert_prefix:
        if not cleaned:
            raise ValueError(f"未能解析出 '{expert_prefix}' 下的专家参数，请确认文件来源。")
        return cleaned

    if not cleaned:
        raise ValueError("state_dict 中没有可用的专家参数键。")

    return cleaned


def _extract_expert_module(ckpt: Any) -> Optional[nn.Module]:
    """尝试从已反序列化对象中直接拿到单个专家模块。"""
    def _maybe_from_container(module_obj: nn.Module) -> Optional[nn.Module]:
        experts = getattr(module_obj, "experts", None)
        if isinstance(experts, (nn.ModuleList, list, tuple)) and len(experts) > 0:
            maybe_expert = experts[0]
            if isinstance(maybe_expert, nn.Module):
                return maybe_expert
        return None

    if isinstance(ckpt, nn.Module):
        expert = _maybe_from_container(ckpt)
        return expert if expert is not None else ckpt

    if isinstance(ckpt, dict):
        for key in ("module.experts.0", "experts.0", "expert_module", "expert"):
            maybe_module = ckpt.get(key)
            if isinstance(maybe_module, nn.Module):
                return maybe_module

        for container_key in ("module", "model"):
            maybe_container = ckpt.get(container_key)
            if isinstance(maybe_container, nn.Module):
                expert = _maybe_from_container(maybe_container)
                if expert is not None:
                    return expert

    return None


# =========================
# Expert instantiation (A: expert-first order)
# =========================

def load_factory(
    experts_config: List[Any],
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    model_dict: OrderedDict,
) -> List[nn.Module]:
    """专家工厂（方案A：按 expert_id → v_type 升序装载）。"""
    experts: List[nn.Module] = []

    for k, v in model_dict.items():
        # k: expert_id (str)
        # v: List[Dict[v_type_id, payload]]
        try:
            expert_id = int(k)
        except Exception:
            raise ValueError(f"expert_id 非整数: {k}")

        if not (0 <= expert_id < len(experts_config)):
            raise IndexError(f"experts_config 下标越界: {expert_id}")

        expert_config_group = experts_config[expert_id]

        # 按 v_type 升序
        try:
            sorted_dict_list = sorted(v, key=lambda d: next(iter(d.keys())))
        except Exception as e:
            raise RuntimeError(f"对 v_type 列表排序失败 (可能有 None 键): {e}")

        for type_expert_sd in sorted_dict_list:
            v_type_id, expert_sd = next(iter(type_expert_sd.items()))

            # 解析 config_for_type → config_list
            if isinstance(expert_config_group, dict) and all(isinstance(key, int) for key in expert_config_group.keys()):
                config_for_type = expert_config_group.get(v_type_id)
                if config_for_type is None:
                    raise KeyError(f"expert {expert_id} 缺少 v_type={v_type_id} 的配置")
            else:
                config_for_type = expert_config_group

            if isinstance(config_for_type, list):
                config_list = config_for_type
            else:
                if not isinstance(config_for_type, dict):
                    raise TypeError(
                        f"expert_config[{expert_id}] 无法解析到有效的字典配置，收到类型 {type(config_for_type)}"
                    )
                config_list = [config_for_type]

            # 两条分支：已有模块 vs 仅权重
            if isinstance(expert_sd, nn.Module):
                expert_raw_model = expert_sd
            else:
                expert_raw_model = ExpertFactory.create_expert_ensemble(
                    expert_configs=config_list,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=hidden_channels,
                    v_type_id=v_type_id,
                )[0]

                missing, unexpected = expert_raw_model.load_state_dict(expert_sd, strict=False)
                if missing or unexpected:
                    print(f"[expert {expert_id}] missing: {missing}, unexpected: {unexpected}")

            for p in expert_raw_model.parameters():
                p.requires_grad = False
            expert_raw_model.eval()

            experts.append(expert_raw_model)

    return experts  # 形如 [FNO0.., WNO.., MNO.., LNO..]


# =========================
# Expert instantiation (B: v_type-first interleaved order)
# =========================

def load_factory_interleaved(
    experts_config: List[Any],
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    interleaved: List[Tuple[str, int, Any]],
) -> List[nn.Module]:
    """
    方案B：按 v_type 优先的交错顺序实例化/装载专家。
    interleaved: List[(expert_id(str), v_type_id(int), payload)]
      - payload 要么是 nn.Module，要么是 state_dict
    返回值即为“按 v_type 交错顺序”排好的一维专家列表。
    """
    experts: List[nn.Module] = []

    for eid_str, v_type_id, expert_sd in interleaved:
        try:
            expert_id = int(eid_str)
        except Exception:
            raise ValueError(f"expert_id 非整数: {eid_str}")

        if not (0 <= expert_id < len(experts_config)):
            raise IndexError(f"experts_config 下标越界: {expert_id}")

        expert_config_group = experts_config[expert_id]

        # 解析 config_for_type → config_list
        if isinstance(expert_config_group, dict) and all(isinstance(key, int) for key in expert_config_group.keys()):
            config_for_type = expert_config_group.get(v_type_id)
            if config_for_type is None:
                raise KeyError(f"expert {expert_id} 缺少 v_type={v_type_id} 的配置")
        else:
            config_for_type = expert_config_group

        if isinstance(config_for_type, list):
            config_list = config_for_type
        else:
            if not isinstance(config_for_type, dict):
                raise TypeError(
                    f"expert_config[{expert_id}] 无法解析到有效的字典配置，收到类型 {type(config_for_type)}"
                )
            config_list = [config_for_type]

        # 两条分支：已有模块 vs 仅权重
        if isinstance(expert_sd, nn.Module):
            expert_raw_model = expert_sd
        else:
            expert_raw_model = ExpertFactory.create_expert_ensemble(
                expert_configs=config_list,
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                v_type_id=v_type_id,
            )[0]

            missing, unexpected = expert_raw_model.load_state_dict(expert_sd, strict=False)
            if missing or unexpected:
                print(f"[expert {expert_id} | v_type {v_type_id}] missing: {missing}, unexpected: {unexpected}")

        for p in expert_raw_model.parameters():
            p.requires_grad = False
        expert_raw_model.eval()

        experts.append(expert_raw_model)

    return experts


# =========================
# Label resolver for specific types
# =========================

def _resolve_specific_label(raw_label: str, id_map: Dict[str, int]) -> Optional[str]:
    """
    将 checkpoint 文件名中的标签映射到 `type_dict['specific']` 的键。
    支持以下格式：
      - 精确的细分类名（curve_vel_a）
      - 基础类别名（curve_vel），默认映射到 variants[0]
      - 旧格式组合（curve_vel 等），或前缀组合（curve_vel_xx）
    """
    if raw_label in id_map:
        return raw_label

    if raw_label in SPECIFIC_TYPE_VARIANTS:
        for variant in SPECIFIC_TYPE_VARIANTS[raw_label]:
            if variant in id_map:
                return variant

    if raw_label in _SPECIFIC_VARIANT_TO_BASE:
        # 原始标签可能是已知的细分类，但 type_dict 尚未覆盖；尝试返回同名或基础默认
        if raw_label in id_map:
            return raw_label
        base = _SPECIFIC_VARIANT_TO_BASE[raw_label]
        for variant in SPECIFIC_TYPE_VARIANTS.get(base, ()):
            if variant in id_map:
                return variant

    tokens = raw_label.split('_')
    if len(tokens) >= 2:
        base_candidate = '_'.join(tokens[:2])
        if base_candidate in SPECIFIC_TYPE_VARIANTS:
            for variant in SPECIFIC_TYPE_VARIANTS[base_candidate]:
                if variant in id_map:
                    return variant

    return None


# =========================
# Main loader with two modes (A/B)
# =========================

def load_moe_experts(
    experts_config: List[Any],
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    model_path: str,
    is_specific: bool,
    map_location,
    type_dict: Dict[str, Dict[str, int]],
    moe_mode: str,
) -> List[nn.Module]:
    """
    读取融合专家参数。
    - 当 moe_mode == "velocity_type" 时，使用方案B：按 v_type 交错的顺序返回专家列表；
    - 否则使用方案A：按 expert_id 排序，每个 expert 内按 v_type 升序返回。
    """
    if not os.path.isdir(model_path):
        raise ValueError(f"{model_path}不是有效路径")

    # 只取 .pt
    experts_file = sorted(f for f in os.listdir(model_path) if f.endswith('.pt'))

    # 组装: expert_id -> List[{v_type_id: payload}]
    grouped: Dict[str, List[Dict[int, Any]]] = defaultdict(list)

    # -------- 扫描与解析 checkpoint --------
    if is_specific:
        id_map = type_dict.get('specific', {})
        if not id_map:
            raise ValueError("type_dict['specific'] 为空，无法映射细分类别。")

        for f in experts_file:
            m = _EXPERT_FILE_PATTERN.match(f)
            if not m:
                print(f"[WARN] 文件名不匹配细化专家模式, 跳过: {f}")
                continue

            expert_id = m.group('i')
            raw_label = m.group('label')
            mapped_label = _resolve_specific_label(raw_label, id_map)
            if mapped_label is None:
                print(f"[WARN] specific 类型映射缺失 {raw_label}, 跳过: {f}")
                continue
            if mapped_label != raw_label and raw_label not in id_map:
                print(f"[INFO] 将细化标签 {raw_label} 映射为 {mapped_label}（使用默认匹配）。")

            v_type = id_map[mapped_label]

            ckpt = torch.load(
                os.path.join(model_path, f),
                map_location=map_location,
                weights_only=False,
            )
            expert_module = _extract_expert_module(ckpt)
            if expert_module is not None:
                grouped[expert_id].append({v_type: expert_module})
            else:
                expert_sd = get_expert_dict(ckpt)
                grouped[expert_id].append({v_type: expert_sd})

    else:
        id_map = type_dict.get('normal', {})
        if not id_map:
            raise ValueError("type_dict['normal'] 为空，无法映射普通类别。")

        for f in experts_file:
            stem = Path(f).stem
            m = _EXPERT_FILE_PATTERN.match(stem)
            if not m:
                print(f"[WARN] 文件名不匹配普通专家模式，跳过：{f}")
                continue

            expert_id = m.group('i')
            label = m.group('label')
            if label not in id_map:
                print(f"[WARN] normal 类型映射缺失 {label}，跳过：{f}")
                continue

            v_type = id_map[label]

            ckpt = torch.load(
                os.path.join(model_path, f),
                map_location=map_location,
                weights_only=False,
            )
            expert_module = _extract_expert_module(ckpt)
            if expert_module is not None:
                grouped[expert_id].append({v_type: expert_module})
            else:
                expert_sd = get_expert_dict(ckpt)
                grouped[expert_id].append({v_type: expert_sd})

    # -------- 两种输出顺序 --------
    if moe_mode == "velocity_type":
        # 方案B：按 v_type 交错顺序输出
        # 1) 将每个 expert 的列表转 dict，便于快速访问
        grouped_map: Dict[str, Dict[int, Any]] = {}
        for eid, vlist in grouped.items():
            vt_map: Dict[int, Any] = {}
            for d in vlist:
                if isinstance(d, dict) and d:
                    vt, payload = next(iter(d.items()))
                    vt_map[int(vt)] = payload
            grouped_map[eid] = vt_map

        # 2) 收集所有 v_type（升序）
        all_vtypes = sorted({vt for vt_map in grouped_map.values() for vt in vt_map.keys()})

        # 3) expert_id 按数字序（同一 v_type 内部的专家遍历顺序）
        try:
            expert_ids_sorted = sorted(grouped_map.keys(), key=lambda k: int(k))
        except (ValueError, TypeError):
            expert_ids_sorted = sorted(grouped_map.keys())

        # 4) 交错展开：(v_type 外层) × (expert_id 内层)
        interleaved: List[Tuple[str, int, Any]] = []
        for vt in all_vtypes:
            for eid in expert_ids_sorted:
                vt_map = grouped_map[eid]
                if vt in vt_map:
                    interleaved.append((eid, vt, vt_map[vt]))

        # 5) 按交错顺序实例化/装载
        loaded_experts = load_factory_interleaved(
            experts_config,
            in_channels,
            out_channels,
            hidden_channels,
            interleaved,
        )

    else:
        # 方案A：按 expert_id 排序，随后在 load_factory 内部按 v_type 升序装载
        try:
            ordered = OrderedDict(sorted(grouped.items(), key=lambda kv: int(kv[0])))
        except (ValueError, TypeError):
            ordered = OrderedDict(sorted(grouped.items(), key=lambda kv: kv[0]))

        loaded_experts = load_factory(
            experts_config,
            in_channels,
            out_channels,
            hidden_channels,
            ordered,
        )

    print(f"成功读取专家，专家数: {len(loaded_experts)}")

    # ====== 打印每个专家的详细信息 ======
    print("\n==== 已加载专家信息 ====")
    for idx, expert in enumerate(loaded_experts):
        num_params = sum(p.numel() for p in expert.parameters())
        trainable_params = sum(p.numel() for p in expert.parameters() if p.requires_grad)
        expert_class = expert.__class__.__name__
        expert_shape_info = ""
        try:
            # 尝试提取模型的核心层结构
            first_layer = next(expert.modules())
            expert_shape_info = f"({type(first_layer).__name__})"
        except Exception:
            pass
        print(f"[{idx:02d}] {expert_class:<25} "
              f"参数总数: {num_params:<10} 可训练: {trainable_params:<10} {expert_shape_info}")

    print("==== 专家加载完毕 ====\n")

    return loaded_experts
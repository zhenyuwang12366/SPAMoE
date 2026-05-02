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
    """Strip prefixes like 'module.'."""
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
    Parse encoder state_dict from any checkpoint layout.
    Supported formats:
      - bare state_dict
      - {'encoder_state_dict': state_dict}
      - {'state_dict': state_dict}
    Also strips DDP prefixes such as ``module.``.
    """
    if source is None:
        raise ValueError("encoder checkpoint is empty; cannot load.")

    if not isinstance(source, dict):
        raise TypeError(
            f"Cannot parse encoder checkpoint of type {type(source)}. Expected dict or state_dict."
        )

    if "encoder_state_dict" in source:
        state_dict = source["encoder_state_dict"]
    elif "state_dict" in source:
        state_dict = source["state_dict"]
    else:
        state_dict = source

    if not isinstance(state_dict, dict):
        raise ValueError("encoder checkpoint has no valid state_dict.")

    cleaned: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if not isinstance(key, str):
            continue
        new_key = _strip_prefixes(key, strip_prefixes)
        cleaned[new_key] = tensor

    if not cleaned:
        raise ValueError("encoder state_dict parsed empty; check save format.")

    return cleaned


def load_encoder_weights(
    encoder: nn.Module,
    checkpoint: Optional[Union[str, Dict[str, Any]]],
    *,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = False,
    strip_prefixes: Iterable[str] = ("module.",),
    drop_prefixes: Iterable[str] = ("type_head.",),   # head keys to skip
    drop_if_contains: Iterable[str] = (),              # optional substring drop
) -> Tuple[List[str], List[str]]:
    """
    Load encoder weights into the given module.
    Returns (missing_keys, unexpected_keys).
    """
    if checkpoint is None:
        return [], []

    if isinstance(checkpoint, str):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint}")
        loaded = torch.load(checkpoint, map_location=map_location)
    else:
        loaded = checkpoint

    state_dict = _extract_encoder_state_dict(loaded, strip_prefixes=strip_prefixes)

    # --------- 1) Drop head / prefix keys explicitly ----------
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
    """Extract first-expert weights from a state_dict or saved expert dict.
    - Single-/multi-GPU saves (``module.`` prefix stripped automatically).
    - If ``experts.0.`` prefix exists, keep only those keys and strip it.
    """
    raw_sd: Optional[Dict[str, Any]] = None
    if isinstance(ckpt, dict):
        for key in ("expert_state_dict", "state_dict", "model_state_dict"):
            maybe_sd = ckpt.get(key)
            if isinstance(maybe_sd, dict):
                raw_sd = maybe_sd
                break
        if raw_sd is None and all(isinstance(k, str) for k in ckpt.keys()):
            raw_sd = ckpt  # already a state_dict

    if raw_sd is None:
        raise TypeError("Cannot extract state_dict from checkpoint; check save format.")

    if not isinstance(raw_sd, dict):
        raise TypeError("state_dict must be a dict; cannot parse expert weights.")

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
            raise ValueError(f"No expert parameters under '{expert_prefix}'; verify checkpoint source.")
        return cleaned

    if not cleaned:
        raise ValueError("state_dict has no usable expert parameter keys.")

    return cleaned


def _extract_expert_module(ckpt: Any) -> Optional[nn.Module]:
    """Try to obtain a single expert module from a deserialized object."""
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
    """Expert factory (plan A: load by expert_id, v_type ascending)."""
    experts: List[nn.Module] = []

    for k, v in model_dict.items():
        # k: expert_id (str)
        # v: List[Dict[v_type_id, payload]]
        try:
            expert_id = int(k)
        except Exception:
            raise ValueError(f"expert_id is not integer: {k}")

        if not (0 <= expert_id < len(experts_config)):
            raise IndexError(f"experts_config index out of range: {expert_id}")

        expert_config_group = experts_config[expert_id]

        # Sort by v_type ascending
        try:
            sorted_dict_list = sorted(v, key=lambda d: next(iter(d.keys())))
        except Exception as e:
            raise RuntimeError(f"Failed to sort v_type list (possible None key): {e}")

        for type_expert_sd in sorted_dict_list:
            v_type_id, expert_sd = next(iter(type_expert_sd.items()))

            # Resolve config_for_type -> config_list
            if isinstance(expert_config_group, dict) and all(isinstance(key, int) for key in expert_config_group.keys()):
                config_for_type = expert_config_group.get(v_type_id)
                if config_for_type is None:
                    raise KeyError(f"expert {expert_id} missing config for v_type={v_type_id}")
            else:
                config_for_type = expert_config_group

            if isinstance(config_for_type, list):
                config_list = config_for_type
            else:
                if not isinstance(config_for_type, dict):
                    raise TypeError(
                        f"expert_config[{expert_id}] is not a valid dict config; got type {type(config_for_type)}"
                    )
                config_list = [config_for_type]

            # Branch: full module vs weights only
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

    return experts  # e.g. [FNO0.., WNO.., MNO.., LNO..]


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
    Plan B: instantiate/load experts in v_type-first interleaved order.
    interleaved: List[(expert_id(str), v_type_id(int), payload)]
      - payload is either nn.Module or state_dict
    Returns a flat expert list in v_type-interleaved order.
    """
    experts: List[nn.Module] = []

    for eid_str, v_type_id, expert_sd in interleaved:
        try:
            expert_id = int(eid_str)
        except Exception:
            raise ValueError(f"expert_id is not integer: {eid_str}")

        if not (0 <= expert_id < len(experts_config)):
            raise IndexError(f"experts_config index out of range: {expert_id}")

        expert_config_group = experts_config[expert_id]

        # Resolve config_for_type -> config_list
        if isinstance(expert_config_group, dict) and all(isinstance(key, int) for key in expert_config_group.keys()):
            config_for_type = expert_config_group.get(v_type_id)
            if config_for_type is None:
                raise KeyError(f"expert {expert_id} missing config for v_type={v_type_id}")
        else:
            config_for_type = expert_config_group

        if isinstance(config_for_type, list):
            config_list = config_for_type
        else:
            if not isinstance(config_for_type, dict):
                raise TypeError(
                    f"expert_config[{expert_id}] is not a valid dict config; got type {type(config_for_type)}"
                )
            config_list = [config_for_type]

        # Branch: full module vs weights only
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
    Map a checkpoint filename label to a key in ``type_dict['specific']``.
    Supported forms:
      - exact subtype name (curve_vel_a)
      - base category (curve_vel), default maps to variants[0]
      - legacy combos (curve_vel etc.) or prefix combos (curve_vel_xx)
    """
    if raw_label in id_map:
        return raw_label

    if raw_label in SPECIFIC_TYPE_VARIANTS:
        for variant in SPECIFIC_TYPE_VARIANTS[raw_label]:
            if variant in id_map:
                return variant

    if raw_label in _SPECIFIC_VARIANT_TO_BASE:
        # Label may be a known subtype while type_dict is incomplete; try same name or base default
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
    Load fused expert weights.
    - If moe_mode == "velocity_type", use plan B: return experts in v_type-interleaved order.
    - Else plan A: sort by expert_id, within each expert sort by ascending v_type.
    """
    if not os.path.isdir(model_path):
        raise ValueError(f"{model_path} is not a valid directory path")

    # Keep .pt files only
    experts_file = sorted(f for f in os.listdir(model_path) if f.endswith('.pt'))

    # Build: expert_id -> List[{v_type_id: payload}]
    grouped: Dict[str, List[Dict[int, Any]]] = defaultdict(list)

    # -------- Scan and parse checkpoints --------
    if is_specific:
        id_map = type_dict.get('specific', {})
        if not id_map:
            raise ValueError("type_dict['specific'] is empty; cannot map specific subtypes.")

        for f in experts_file:
            m = _EXPERT_FILE_PATTERN.match(f)
            if not m:
                print(f"[WARN] Filename does not match specific-expert pattern, skip: {f}")
                continue

            expert_id = m.group('i')
            raw_label = m.group('label')
            mapped_label = _resolve_specific_label(raw_label, id_map)
            if mapped_label is None:
                print(f"[WARN] No specific mapping for {raw_label}, skip: {f}")
                continue
            if mapped_label != raw_label and raw_label not in id_map:
                print(f"[INFO] Mapped subtype label {raw_label} -> {mapped_label} (default match).")

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
            raise ValueError("type_dict['normal'] is empty; cannot map normal categories.")

        for f in experts_file:
            stem = Path(f).stem
            m = _EXPERT_FILE_PATTERN.match(stem)
            if not m:
                print(f"[WARN] Filename does not match normal-expert pattern, skip: {f}")
                continue

            expert_id = m.group('i')
            label = m.group('label')
            if label not in id_map:
                print(f"[WARN] No normal mapping for label {label}, skip: {f}")
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

    # -------- Two output orderings --------
    if moe_mode == "velocity_type":
        # Plan B: interleave by v_type
        # 1) Turn each expert's list into a dict for fast lookup
        grouped_map: Dict[str, Dict[int, Any]] = {}
        for eid, vlist in grouped.items():
            vt_map: Dict[int, Any] = {}
            for d in vlist:
                if isinstance(d, dict) and d:
                    vt, payload = next(iter(d.items()))
                    vt_map[int(vt)] = payload
            grouped_map[eid] = vt_map

        # 2) Collect all v_types (sorted)
        all_vtypes = sorted({vt for vt_map in grouped_map.values() for vt in vt_map.keys()})

        # 3) Numeric sort of expert_id (iteration order within one v_type)
        try:
            expert_ids_sorted = sorted(grouped_map.keys(), key=lambda k: int(k))
        except (ValueError, TypeError):
            expert_ids_sorted = sorted(grouped_map.keys())

        # 4) Interleave: outer v_type, inner expert_id
        interleaved: List[Tuple[str, int, Any]] = []
        for vt in all_vtypes:
            for eid in expert_ids_sorted:
                vt_map = grouped_map[eid]
                if vt in vt_map:
                    interleaved.append((eid, vt, vt_map[vt]))

        # 5) Instantiate/load in interleaved order
        loaded_experts = load_factory_interleaved(
            experts_config,
            in_channels,
            out_channels,
            hidden_channels,
            interleaved,
        )

    else:
        # Plan A: sort by expert_id, then load_factory sorts v_type ascending
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

    print(f"Loaded experts successfully, count: {len(loaded_experts)}")

    # ====== Print per-expert details ======
    print("\n==== Loaded expert summary ====")
    for idx, expert in enumerate(loaded_experts):
        num_params = sum(p.numel() for p in expert.parameters())
        trainable_params = sum(p.numel() for p in expert.parameters() if p.requires_grad)
        expert_class = expert.__class__.__name__
        expert_shape_info = ""
        try:
            # Try to show first module type
            first_layer = next(expert.modules())
            expert_shape_info = f"({type(first_layer).__name__})"
        except Exception:
            pass
        print(f"[{idx:02d}] {expert_class:<25} "
              f"params: {num_params:<10} trainable: {trainable_params:<10} {expert_shape_info}")

    print("==== Expert load complete ====\n")

    return loaded_experts
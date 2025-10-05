from typing import Dict,Iterable,Any,List,Optional
import torch
from collections import OrderedDict,defaultdict
import torch.nn as nn
from neuralop.models import ExpertFactory
import re
import os

def _strip_prefixes(key: str, prefixes: Iterable[str]) -> str:
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key

def get_expert_dict(
    ckpt: Dict[str, Any],
    ddp_prefixes: Iterable[str] = ("module.",),
    expert_prefix: str = "experts.0.",
) -> Dict[str, torch.Tensor]:
    """从 state_dict 或保存的专家字典中提取第一个专家权重。

    - 支持单卡/多卡保存（自动剥离常见的 ``module.`` 前缀）。
    - 仅保留以 ``experts.0.`` 开头的权重键，并去掉该前缀，方便后续直接 ``load_state_dict``。
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
        raise TypeError("无法从 checkpoint 中提取 state_dict，检查保存格式是否符合 train_process.py 逻辑。")

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
            raise ValueError(
                f"未能解析出 '{expert_prefix}' 下的专家参数，请确认文件来源。"
            )
        return cleaned

    if not cleaned:
        raise ValueError("state_dict 中没有可用的专家参数键。")

    return cleaned
def _extract_expert_module(ckpt: Any) -> Optional[nn.Module]:
    """尝试从已反序列化对象中直接拿到专家模块。"""

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

def load_factory(
    experts_config: List[Any],
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    model_dict: OrderedDict,
) -> List[nn.Module]: 
    """专家工厂

    Args:
        experts_config (List[Any]): 专家配置字典（可为基础配置或 v_type -> 配置映射）
        model_dict (Dict): 专家模型参数字典, experts_type == 'math' 
        --> Dict[expert_id, List[Dict[v_type_id, sd]]]

    Returns:
        List[nn.Module]: 返回专家模型列表
    """
    
    experts: List[nn.Module] = []
    
    for k, v in model_dict.items():
        # k: expert_id
        # v: List[Dict[v_type_id, sd]]
        try:
            expert_id = int(k)
        except Exception:
            raise ValueError(f"expert_id 非整数: {k}")
        
        if not (0 <= expert_id < len(experts_config)):
            raise IndexError(f"experts_config 下标越界: {expert_id}")
        
        expert_config_group = experts_config[expert_id]
        
        # 按v_type升序排列
        try:
            sorted_dict_list = sorted(
            v,
            key = lambda d: next(iter(d.keys())),
        )
        except Exception as e:
            raise RuntimeError(f"对 v_type 列表排序失败 (可能有 None 键): {e}")
        
        for type_expert_sd in sorted_dict_list:
            v_type_id, expert_sd = next(iter(type_expert_sd.items()))

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
                        f"expert_config[{expert_id}] 无法解析到有效的字典配置，"
                        f"收到类型 {type(config_for_type)}"
                    )
                config_list = [config_for_type]

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
            
    return experts # [FNO0, FNO1, FNO2, FNO3, FNO4, WNO0,...., MNO4,..., LNO4]

_SPECIFIC_PAT = re.compile(
    r'best_expert_(?P<name>\w+)_(?P<i>\d+)_(?P<shape>\w+)_(?P<label>\w+)\.pt$'
)
_NORMAL_PAT = re.compile(
    r'best_expert_(?P<name>\w+)_(?P<i>\d+)_(?P<label>\w+)\.pt$'
)
def load_moe_experts(
    experts_config: List[Any],
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    model_path: str,
    is_specific: bool,
    map_location,
    type_dict: Dict[str, Dict[str, int]],
) -> List[nn.Module]:
    """读取融合专家参数

    Args:
        model_path (str): 专家保存的文件路径,
            保存的文件名：不细化版本: best_expert_{experts_name}_{i}_{vel/fault/style}.pt\
                        细化版本: best_expert_{experts_name}_{i}_{curve/flat/style}_{vel/fault/style}.pt
            
            按math分成FNO, WNO, MNO, LNO四类，每类有多种速度图类型, 直接读取, 每类以\
            
        is_specific (bool): 速度图是否细分
        
    Returns:
        experts (List[nn.Module]): 输出专家列表
    """
    if not os.path.isdir(model_path):
        raise ValueError(f"{model_path}不是有效路径")
    
    # 只取 .pt
    experts_file = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    
    # 组装: expert_id -> List[{v_type_id: sd}]
    grouped: Dict[str, List[Dict[int, Any]]] = defaultdict(list) #Dict[str(type), list]
    
    if(is_specific):
        id_map = type_dict.get('specific', {})
        # 获取所有.pt文件, best_expert_{experts_name}_{i}_{curve/flat/style}_{vel/fault/style}.pt
        for f in experts_file:
            m = _SPECIFIC_PAT.match(f)
            if not m:
                # 兼容 split 解析
                parts = f.split('_')
                if len(parts) >= 6 and parts[0] == 'best' and parts[1] == 'expert':
                    expert_id = parts[3]
                    shape = parts[4]
                    label = parts[5].split('.')[0]
                else:
                    print(f"[WARN] 文件名不匹配 specific 模式, 跳过: {f}")
                    continue
            else:
                expert_id = m.group('i')
                shape = m.group('shape')
                label = m.group('label')
            
            key = f"{shape}_{label}"
            if key not in id_map:
                print(f"[WARN] specific 类型映射缺失 {key}, 跳过: {f}")
                continue
            v_type = id_map[key]
            
            ckpt = torch.load(
                os.path.join(model_path, f),
                map_location=map_location,
                weights_only=False,
            )
            expert_module = _extract_expert_module(ckpt)
            if expert_module is not None:
                grouped[expert_id].append({v_type: expert_module})
                continue

            expert_sd = get_expert_dict(ckpt)
            grouped[expert_id].append({v_type: expert_sd})            
    else:
        id_map = type_dict.get('normal', {})
        for f in experts_file:
            m = _NORMAL_PAT.match(f)
            if not m:
                # 兼容 split 解析（宽松）
                parts = f.split('_')
                if len(parts) >= 5 and parts[0] == 'best' and parts[1] == 'expert':
                    expert_id = parts[3]
                    label = parts[4].split('.')[0]
                else:
                    print(f"[WARN] 文件名不匹配 normal 模式，跳过：{f}")
                    continue
            else:
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
                continue

            expert_sd = get_expert_dict(ckpt)
            grouped[expert_id].append({v_type: expert_sd}) # [FNO(3), WNO(3), MNO(3), LNO(3)]
    
    # 对 expert_id 做数字序排序, 保证顺序稳定  
    try:
        ordered = OrderedDict(sorted(grouped.items(), key=lambda kv: int(kv[0])))
    except Exception:
        ordered = OrderedDict(sorted(grouped.items(), key=lambda kv: kv[0]))
    
    loaded_experts = load_factory(
        experts_config,
        in_channels,
        out_channels,
        hidden_channels,
        ordered, 
    )
    
    return loaded_experts

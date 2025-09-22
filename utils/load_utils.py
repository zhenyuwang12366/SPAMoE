from typing import Dict,Iterable,Any,List
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

def get_expert_dict(ckpt: Dict[str, Any],
                    ddp_prefixes: Iterable[str] = ("module.",),
                    allow_experts_prefix_fallback: bool = True
                   ) -> Dict[str, torch.Tensor]:
    """
    从 checkpoint 中提取专家权重（你的保存方式：{'expert_state_dict': ...}）。
    - 默认去掉常见并行前缀（如 'module.'）。
    - 若误存成 'experts.0.xxx'，可选地剥掉 'experts.0.'（fallback）。

    Args:
        ckpt: 通过 torch.load(...) 得到的 checkpoint 字典，必须包含 'expert_state_dict'
        ddp_prefixes: 需要剥除的并行前缀集合（如 DDP 的 'module.'）
        allow_experts_prefix_fallback: 若键以 'experts.0.' 开头，是否自动剥掉

    Returns:
        清洗后的专家 state_dict（键名干净，可直接 expert.load_state_dict(...)）
    """
    if not isinstance(ckpt, dict) or "expert_state_dict" not in ckpt:
        raise ValueError("checkpoint 中未找到 'expert_state_dict'，请确认保存与加载路径一致。")

    raw_sd = ckpt["expert_state_dict"]
    if not isinstance(raw_sd, dict):
        raise TypeError("'expert_state_dict' 应是一个 state_dict (dict)。")

    cleaned = OrderedDict()
    for k, v in raw_sd.items():
        k2 = _strip_prefixes(k, ddp_prefixes)
        if allow_experts_prefix_fallback and k2.startswith("experts.0."):
            k2 = k2[len("experts.0."):]
        cleaned[k2] = v

    return cleaned

def load_factory(
    experts_config: List[Dict[str, Any]],
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    model_dict: OrderedDict,
) -> List[nn.Module]: 
    """专家融合工厂

    Args:
        experts_config (List[Dict[str, Any]]): 专家配置字典
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
        
        expert_config = experts_config[expert_id]
        
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
            
            # 创建专家骨架
            expert_raw_model = ExpertFactory.create_expert_ensemble(
                [expert_config],
                in_channels,
                out_channels,
                hidden_channels,
            )[0]
            
            # 加载权重
            missing, unexpected = expert_raw_model.load_state_dict(expert_sd, strict=False)
            if missing or unexpected:
                print(f"[expert {expert_id}] missing: {missing}, unexpected: {unexpected}")

            # 冻结
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
    experts_config: List[Dict[str, Any]],
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
    grouped: Dict[str, List[Dict[int, Dict[str, torch.Tensor]]]] = defaultdict(list) #Dict[str(type), list]
    
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
            
            ckpt = torch.load(os.path.join(model_path, f), map_location=map_location)
            full_sd = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
            expert_sd = get_expert_dict(full_sd, i=0)
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

            ckpt = torch.load(os.path.join(model_path, f), map_location=map_location)
            full_sd = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
            expert_sd = get_expert_dict(full_sd, i=0)
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
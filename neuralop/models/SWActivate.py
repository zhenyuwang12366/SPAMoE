import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Callable
import math

class GroupActMerge(nn.Module):
    """强弱激活集合融合器

    Args:
        processor (Callable[[Tensor[B,K,1,H,W]], Tensor[B,1,H,W]]): 集合内部融合方式
    """
    def __init__(
        self,
        processor: Optional[nn.Module] = None,   # 对强激活专家的融合方法
    ):
        super().__init__()
        self.processor = processor
    
    def forward(self, group_outputs: torch.Tensor) -> torch.Tensor:
        """

        Args:
            group_outputs (torch.Tensor[B, K, 1, H, W]):输入 

        Returns:
            torch.Tensor (torch.Tensor[B, 1, H, W]): 某激活集合的融合输出
        """
        return self.processor(group_outputs)
    
class SWActMerge(nn.Module):
    def __init__(
        self,
        beta: float = 0.5
    ):
        super().__init__()
        self.beta = beta
        
    def forward(self, merged_strong: torch.Tensor, merged_weak: torch.Tensor,):
        """

        Args:
            merged_strong (torch.Tensor[B,1,H,W]):强专家输出
            merged_weak (torch.Tensor[B,1,H,W]): 弱专家输出

        Returns:
            torch.Tensor[B,1,H,W]: 融合输出
        """
        assert merged_strong.shape == merged_weak.shape, "strong shape no equip weak shape, can not merge"
        result = (1 + self.beta) * merged_strong - self.beta * merged_weak
        
        return result
        
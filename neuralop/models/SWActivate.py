import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Callable
import math

class GroupActMerge(nn.Module):
    """Fuser for strong/weak activation expert groups.

    Args:
        processor (Callable[[Tensor[B,K,1,H,W]], Tensor[B,1,H,W]]): intra-group fusion
    """
    def __init__(
        self,
        processor: Optional[nn.Module] = None,   # fusion module for strongly activated experts
    ):
        super().__init__()
        self.processor = processor
    
    def forward(self, group_outputs: torch.Tensor) -> torch.Tensor:
        """

        Args:
            group_outputs (torch.Tensor[B, K, 1, H, W]): inputs 

        Returns:
            torch.Tensor (torch.Tensor[B, 1, H, W]): fused output for one activation set
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
            merged_strong (torch.Tensor[B,1,H,W]): strong-expert branch output
            merged_weak (torch.Tensor[B,1,H,W]): weak-expert branch output

        Returns:
            torch.Tensor[B,1,H,W]: fused output
        """
        assert merged_strong.shape == merged_weak.shape, "strong shape no equip weak shape, can not merge"
        result = (1 + self.beta) * merged_strong - self.beta * merged_weak
        
        return result
        

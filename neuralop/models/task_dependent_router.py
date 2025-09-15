import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
import math

class TaskDependentRouter(nn.Module):
    def __init__(
        self,
        input_dim:int,
        num_experts:int,
        hidden_dim: int,
        bsz: int,
        init_num: int = 1,
        patience: int = 3,
        noisy_gating: bool = True,
        alpha: float = 0.5,
    ):
        # 子类调用父类初始化方法，assert判断
        super().__init__()
        assert num_experts >= 1, "num_experts must be >= 1"
        assert 1 <= init_num <= num_experts, "init_num must be in [1, num_experts]"
        assert patience >= 1, "patience must be >= 1"
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.init_num = init_num
        self.patience = patience
        self.top_k = init_num
        self.noisy_gating = noisy_gating
        self.alpha = alpha
        
        # 验证表现跟踪，使用buffer，可以保证可以保存进state_dict
        # 状态变量
        self.register_buffer("best_val_loss", torch.tensor(float("inf")))
        self.no_improved_cnt = 0 # 连续无改进验证轮次数
        self.improved = True # 当前k下产生了改进
        self.fixed = False # 外层回退与固定
        
        # x_flat b*1(input_dim)
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    @torch.no_grad()
    def step_validation(self, val_loss: float):
        if self.fixed: # 如果外层已回退固定，就不更改了
            return 
        
        if math.isnan(val_loss) or math.isinf(val_loss):
            return
        
        if val_loss + 1e-12 < float(self.best_val_loss.item()): # 确认真的很小，而不是因为浮点数的抖动
            # 有改进
            self.best_val_loss.fill_(val_loss)
            self.improved = True
            self.no_improved_cnt = 0
        else:
            self.no_improved_cnt += 1
            if self.no_improved_cnt >= self.patience:
                if not self.improved:
                    # 当前k没有产生改进，向外层发出信号
                    return "should_break"
                else:
                    # 试探阶段，增加k
                    if self.top_k < self.num_experts:
                        self.top_k += 1
                    self.improved = False
                    self.no_improved_cnt = 0  
        return None
        
    def forward(self, feats):
        assert feats.shape[-1] == self.input_dim, f"feats last dim {feats.shape[-1]} != input_dim {self.input_dim}"
        # b * num 
        logits = self.router(feats)
        if self.noisy_gating and self.training:
            # 训练时添加噪声以增加探索性
            noise = torch.randn_like(logits) * 1.0
            logits = logits + noise
        
        routing_weights = F.softmax(logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim = -1)
        
        # 归一化top-k权重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        
        # 弱激活专家
        w_top_k = self.num_experts - self.top_k
        if w_top_k > 0:
            mask = torch.zeros_like(routing_weights, dtype=torch.bool)
            src = torch.ones_like(top_k_indices, dtype=torch.bool)
            mask.scatter_(1, top_k_indices, src)
            masked_weights = routing_weights.masked_fill_(mask, -1e9)
            
            w_weights, w_indices = torch.topk(masked_weights, k=w_top_k, dim=-1)
            # 归一化top-k权重
            w_weights = w_weights / w_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            w_weights = None
            w_indices = None
        
        # aux
        if self.training and self.alpha > 0.0:
            ce = F.one_hot(top_k_indices.reshape(-1), num_classes=self.num_experts).float().mean(dim=0)            
            Pi = routing_weights.mean(dim=0)
            fi = ce * self.num_experts
            aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        
        return top_k_weights, top_k_indices, w_weights, w_indices, aux_loss, self.top_k
    
                
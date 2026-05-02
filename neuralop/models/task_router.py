import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple, Optional

class TaskAwareRouter(nn.Module):
    """
    Task-aware router: routes using input features and optional task features.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    task_dim : int
        Task feature dimension; if 0, task features are unused
    num_experts : int
        Number of experts
    hidden_dim : int, optional
        Router MLP hidden size
    top_k : int, optional
        Top-k experts to select
    noisy_gating : bool, optional
        Whether to use noisy gating during training
    routing_mode : str, optional
        One of 'input' (input only), 'task' (task only), 'both' (concatenate both)
    """
    def __init__(
        self,
        input_dim: int,
        task_dim: int = 0,
        num_experts: int = 4,
        hidden_dim: int = 256,
        top_k: int = 2,
        noisy_gating: bool = True,
        routing_mode: str = 'both',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.task_dim = task_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noisy_gating = noisy_gating
        self.routing_mode = routing_mode
        
        # Validate routing mode
        if routing_mode not in ['input', 'task', 'both']:
            raise ValueError(f"Unsupported routing mode: {routing_mode}")
        
        if routing_mode in ['task', 'both'] and task_dim <= 0:
            raise ValueError(f"When routing_mode is '{routing_mode}', task_dim must be > 0")
        
        # Router MLP input size
        router_input_dim = 0
        if routing_mode in ['input', 'both']:
            router_input_dim += input_dim
        if routing_mode in ['task', 'both']:
            router_input_dim += task_dim
        
        # Router MLP
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize MLP weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, task_features=None):
        """
        Compute per-expert routing weights.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape [batch_size, input_dim]
        task_features : torch.Tensor, optional
            Task features, shape [batch_size, task_dim]; required if routing_mode is 'task' or 'both'
            
        Returns
        -------
        routing_weights : torch.Tensor
            Selected gate weights, shape [batch_size, top_k]
        expert_indices : torch.Tensor
            Expert indices, shape [batch_size, top_k]
        """
        batch_size = x.shape[0]
        
        if self.routing_mode in ['task', 'both'] and task_features is None:
            raise ValueError(f"When routing_mode is '{self.routing_mode}', task_features must be provided")
        
        router_input = []
        if self.routing_mode in ['input', 'both']:
            if x.dim() > 2:
                x_flat = x.view(batch_size, -1)
                if x_flat.shape[1] > self.input_dim:
                    # simple mean pooling
                    x_flat = x_flat.view(batch_size, -1, self.input_dim).mean(dim=1)
            else:
                x_flat = x
            router_input.append(x_flat)
            
        if self.routing_mode in ['task', 'both'] and task_features is not None:
            router_input.append(task_features)
        
        router_input = torch.cat(router_input, dim=1)
        
        logits = self.router(router_input)
        
        if self.noisy_gating and self.training:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise
            
        routing_weights = F.softmax(logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_weights, top_k_indices
    
    def get_expert_distribution(self, x, task_features=None):
        """
        Full softmax distribution over experts.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features
        task_features : torch.Tensor, optional
            Task features
            
        Returns
        -------
        expert_distribution : torch.Tensor
            Shape [batch_size, num_experts]
        """
        batch_size = x.shape[0]
        
        router_input = []
        if self.routing_mode in ['input', 'both']:
            if x.dim() > 2:
                x_flat = x.view(batch_size, -1)
                if x_flat.shape[1] > self.input_dim:
                    # simple mean pooling
                    x_flat = x_flat.view(batch_size, -1, self.input_dim).mean(dim=1)
            else:
                x_flat = x
            router_input.append(x_flat)
            
        if self.routing_mode in ['task', 'both'] and task_features is not None:
            router_input.append(task_features)
        
        router_input = torch.cat(router_input, dim=1)
        
        logits = self.router(router_input)
        
        routing_weights = F.softmax(logits, dim=-1)
        
        return routing_weights

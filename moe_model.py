import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch.nn.utils import stateless
import random
import numpy as np

class ModelArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Gate(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.input_window_size = args.input_window_size
        self.topk = args.n_activated_experts  # Number of experts to activate per input
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, self.input_window_size))  
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)

        scores = F.linear(x, self.weight)  
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)  
        else:
            scores = scores.sigmoid()  
        original_scores = scores  
        indices = torch.topk(scores, self.topk, dim=-1)[1]  
        weights = original_scores.gather(1, indices)  
        weights *= self.route_scale  
        return weights.type_as(x), indices  

class Expert(nn.Module):

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)  # First linear layer
        #self.bn1 = nn.BatchNorm1d(inter_dim) # Add BatchNorm after the first linear layer
        self.w2 = nn.Linear(inter_dim, dim)  # Second linear layer
        self.w3 = nn.Linear(dim, inter_dim)  # Third linear layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.leaky_relu(self.w1(x)) * self.w3(x))  

    
class Shared_Expert(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.input_window_size = args.input_window_size
        self.inter_dim = args.inter_dim
        self.n_shared_experts = args.n_shared_experts
        self.w1 = nn.Linear(self.input_window_size, self.n_shared_experts * self.inter_dim)
        #self.bn1 = nn.BatchNorm1d(self.n_shared_experts * self.inter_dim) 
        self.w2 = nn.Linear(self.n_shared_experts * self.inter_dim, self.input_window_size)
        self.w3 = nn.Linear(self.input_window_size, self.n_shared_experts * self.inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.leaky_relu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs, task_names: list):
        super().__init__()
        self.dim = args.dim  
        self.batch_size = args.batch_size
        self.input_window_size = args.input_window_size
        self.n_routed_experts = args.n_routed_experts  
        self.n_activated_experts = args.n_activated_experts  
        self.output_window_size = args.output_window_size
        self.gate = Gate(args)  
        self.experts = nn.ModuleList([Expert(args.input_window_size, args.moe_inter_dim) for _ in range(self.n_routed_experts)])  
        self.shared_experts = Shared_Expert(args)
        self.output_layer = nn.Linear(args.input_window_size, args.output_window_size)  
        self.register_buffer('global_counts', torch.zeros(args.n_routed_experts))  
        self.register_buffer('total_selections', torch.tensor(0))                 
    def forward(self, x: torch.Tensor, task_id: int, shared_params=None) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # [batch, input_dim]
        batch_size = x.size(0)
        weights, indices = self.gate(x)  
        y = torch.zeros_like(x)
    
        expanded_x = x.repeat_interleave(self.n_activated_experts, dim=0)  
        expanded_weights = weights.reshape(-1)  
        expanded_indices = indices.reshape(-1) 
    
        for expert_id in range(self.n_routed_experts):
            mask = (expanded_indices == expert_id)
            if not mask.any():
                continue
        
            expert = self.experts[expert_id]
            expert_input = expanded_x[mask]  # [num_samples, D]
            expert_output = expert(expert_input)
        
            weighted_output = expert_output * expanded_weights[mask].unsqueeze(1)  # [num_samples, D]
        
            original_idx = torch.div(
                torch.nonzero(mask, as_tuple=False).squeeze(),
                self.n_activated_experts,
                rounding_mode='floor'
            )
        
            y.index_add_(0, original_idx, weighted_output)

        if shared_params is None:
            z = self.shared_experts(x)
        else:
            param_names = [name for name, _ in self.shared_experts.named_parameters()]
            param_dict = dict(zip(param_names, shared_params))
            z = torch.func.functional_call(self.shared_experts, param_dict, (x,))
            
        output = self.output_layer(y + z)
        return output.view(self.batch_size, self.output_window_size)

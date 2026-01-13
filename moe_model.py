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
    """
    Gating mechanism for routing inputs in a MoE model.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.input_window_size = args.input_window_size
        self.topk = args.n_activated_experts  # Number of experts to activate per input
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        #self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))  # Learnable gating weight
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, self.input_window_size))  # (input_window_size, num_experts)
        #self.bn = nn.BatchNorm1d(args.dim)  # BatchNorm after the weight matrix
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts))
        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Xavier 初始化 weight 和 zeros 初始化 bias
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gate layer. It decides which experts to activate.
        """
        x = x.view(x.size(0), -1)

        scores = F.linear(x, self.weight)  # Compute scores for each expert using a linear transformation
        #scores = self.bn(scores)  # Apply BN after the scores calculation
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)  # Use softmax to get the probabilities of selecting experts
        else:
            scores = scores.sigmoid()  # Sigmoid activation for gating

        #scores = scores + self.bias
        original_scores = scores  # Preserve original scores for scaling
        
        indices = torch.topk(scores, self.topk, dim=-1)[1]  # Select the top-k experts based on scores
        weights = original_scores.gather(1, indices)  # Gather the top-k weights from the original scores
        weights *= self.route_scale  # Scale the weights based on the route_scale
        return weights.type_as(x), indices  # Return the selected expert weights and indices


class Expert(nn.Module):
    """
    Expert layer in the MoE model.
    """
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)  # First linear layer
        #self.bn1 = nn.BatchNorm1d(inter_dim) # Add BatchNorm after the first linear layer
        self.w2 = nn.Linear(inter_dim, dim)  # Second linear layer
        self.w3 = nn.Linear(dim, inter_dim)  # Third linear layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #return self.w2(F.silu(self.w1(x)) * self.w3(x)) 
        #print(f"[Debug] Expert 输入均值: {x.mean().item():.4f}, 输出均值: {(self.w2(F.relu(self.w1(x)) * self.w3(x))).mean().item():.4f}")
        #return self.w2(F.relu(self.w1(x)) * self.w3(x))  
        return self.w2(F.leaky_relu(self.w1(x)) * self.w3(x))  

    
class Shared_Expert(nn.Module):
    
    def __init__(self, args: ModelArgs):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.dim = args.dim
        self.input_window_size = args.input_window_size
        self.inter_dim = args.inter_dim
        self.n_shared_experts = args.n_shared_experts
        self.w1 = nn.Linear(self.input_window_size, self.n_shared_experts * self.inter_dim)
        #self.bn1 = nn.BatchNorm1d(self.n_shared_experts * self.inter_dim)  # Add BatchNorm after the first linear layer
        self.w2 = nn.Linear(self.n_shared_experts * self.inter_dim, self.input_window_size)
        self.w3 = nn.Linear(self.input_window_size, self.n_shared_experts * self.inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        #x = F.silu(self.bn1(self.w1(x)))  # Apply BN after the first layer
        #return self.w2(x) * self.w3(x)
        #return self.w2(F.silu(self.w1(x)) * self.w3(x))
        #print(f"[Debug] Shared_Expert 输入均值: {x.mean().item():.4f}, 输出均值: {(self.w2(F.relu(self.w1(x)) * self.w3(x))).mean().item():.4f}")
        #return self.w2(F.relu(self.w1(x)) * self.w3(x))
        return self.w2(F.leaky_relu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture of Experts (MoE) model for time-series forecasting.
    """
    def __init__(self, args: ModelArgs, task_names: list):
        super().__init__()
        self.dim = args.dim  # The input dimension
        self.batch_size = args.batch_size
        self.input_window_size = args.input_window_size
        self.n_routed_experts = args.n_routed_experts  # Number of experts in the model
        self.n_activated_experts = args.n_activated_experts  # Number of activated experts per input
        self.output_window_size = args.output_window_size
        self.gate = Gate(args)  # Initialize the gating mechanism
        self.experts = nn.ModuleList([Expert(args.input_window_size, args.moe_inter_dim) for _ in range(self.n_routed_experts)])  # Initialize the experts
        self.shared_experts = Shared_Expert(args)
        self.output_layer = nn.Linear(args.input_window_size, args.output_window_size)  # Output layer for regression
        self.register_buffer('global_counts', torch.zeros(args.n_routed_experts))  # 全局专家激活次数
        self.register_buffer('total_selections', torch.tensor(0))                 # 全局总激活次数
    def forward(self, x: torch.Tensor, task_id: int, shared_params=None) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # [batch, input_dim]
        batch_size = x.size(0)
    
        # 门控网络输出 (确保gate返回的indices是[batch, top_k])
        weights, indices = self.gate(x)  # weights形状应为[batch, top_k], indices形状[batch, top_k]
    
        # 初始化输出张量
        y = torch.zeros_like(x)
    
        # 将每个样本的top_k选择展开为独立条目
        expanded_x = x.repeat_interleave(self.n_activated_experts, dim=0)  # [batch*top_k, D]
        expanded_weights = weights.reshape(-1)  # [batch*top_k]
        expanded_indices = indices.reshape(-1)  # [batch*top_k]
    
        # 按专家处理所有分配到的样本
        for expert_id in range(self.n_routed_experts):
            # 找出当前专家需要处理的所有条目
            mask = (expanded_indices == expert_id)
            if not mask.any():
                continue
        
            # 处理该专家的输入
            expert = self.experts[expert_id]
            expert_input = expanded_x[mask]  # [num_samples, D]
            expert_output = expert(expert_input)
        
            # 应用权重 (关键维度修正)
            weighted_output = expert_output * expanded_weights[mask].unsqueeze(1)  # [num_samples, D]
        
            # 计算原始样本索引 (通过整除还原)
            original_idx = torch.div(
                torch.nonzero(mask, as_tuple=False).squeeze(),
                self.n_activated_experts,
                rounding_mode='floor'
            )
        
            # 累加到输出 (避免重复索引问题)
            y.index_add_(0, original_idx, weighted_output)

        # 共享专家处理
        if shared_params is None:
            z = self.shared_experts(x)
        else:
            param_names = [name for name, _ in self.shared_experts.named_parameters()]
            param_dict = dict(zip(param_names, shared_params))
            z = torch.func.functional_call(self.shared_experts, param_dict, (x,))
            
        # 最终输出
        output = self.output_layer(y + z)
        return output.view(self.batch_size, self.output_window_size)
    """
    def forward(self, x: torch.Tensor, task_id: int, shared_params=None) -> torch.Tensor:
        """
    """
        Forward pass for the MoE model.
        """
    """
        #shape = x.size()  # Get the input shape
        #x = x.view(-1, self.dim)  # Flatten the input tensor
        x = x.view(x.size(0), -1)
        weights, indices = self.gate(x)  # Get the expert weights and selected indices from the gate
        y = torch.zeros_like(x)  # Initialize an empty tensor for expert outputs
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()  # Count how many times each expert is activated
        print(counts)
        for i in range(self.n_routed_experts):  # Loop through each expert
            if counts[i] == 0:
                continue  # Skip if no input is routed to this expert
            print(indices)
            expert = self.experts[i]  # Get the corresponding expert
            idx, top = torch.where(indices == i)  # Get the indices of inputs routed to this expert
            print(idx, top)
            y[idx] += expert(x[idx]) * weights[idx, top, None]  # Apply the expert and scale by the expert weight
 
        if shared_params is None:
            z = self.shared_experts(x)  # 直接使用共享专家
        else:
            param_dict = {name: p for (name, _), p in zip(self.shared_experts.named_parameters(), shared_params)}
            z = torch.func.functional_call(self.shared_experts, param_dict, (x,)) 
            #z = stateless.functional_call(self.shared_experts, param_dict, (x,)) 
        #z = self.shared_experts(x)
        output = self.output_layer(y+z)  # Apply the final output layer
        return output.view(self.batch_size, self.output_window_size)
        #return output.view(-1, self.output_window_size)  # Reshape the output to match the required shape for regression tasks
        """
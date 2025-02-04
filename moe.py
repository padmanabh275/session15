import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.hidden_size = config.hidden_size
        
        # Create experts
        self.experts = nn.ModuleList([ExpertLayer(config) for _ in range(self.num_experts)])
        
        # Router
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        dtype = x.dtype
        device = x.device
        
        # Reshape input for routing
        x_reshaped = x.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        
        # Router logits
        router_logits = self.router(x_reshaped)  # [batch_size * seq_len, num_experts]
        
        # Calculate routing probabilities
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, k=self.expert_capacity, dim=-1
        )
        
        # Normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output with same dtype as input
        combined_output = torch.zeros_like(x_reshaped, dtype=dtype, device=device)
        
        # Process inputs through experts
        for i, expert in enumerate(self.experts):
            # Get indices for this expert
            expert_mask = (top_k_indices == i).any(dim=-1)
            if expert_mask.any():
                # Process tokens assigned to this expert
                expert_input = x_reshaped[expert_mask]
                expert_output = expert(expert_input).to(dtype)  # Ensure output dtype matches
                combined_output[expert_mask] = expert_output
        
        # Reshape output back to original dimensions
        return combined_output.view(batch_size, seq_len, hidden_size) 
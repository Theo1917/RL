"""Neural network architectures for DQN query optimization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class QNetwork(nn.Module):
    """Deep Q-Network for query optimization.
    
    Input state includes:
    - Joined tables (binary vector)
    - Cardinality features (log scale)
    - Selectivity estimates
    - Cost accumulator
    - Available actions mask
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (256, 128, 64)):
        """Initialize Q-network.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Number of possible actions (tables)
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build fully connected layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer: Q-values for each action
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier initialization for weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass to compute Q-values.
        
        Args:
            state: Batch of state observations (batch_size, state_dim)
            action_mask: Boolean mask for valid actions (batch_size, action_dim)
            
        Returns:
            Q-values for each action (batch_size, action_dim)
        """
        q_values = self.network(state)
        
        # Apply action mask: set invalid actions to -inf
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask, float('-inf'))
        
        return q_values


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture separating value and advantage streams.
    
    This architecture learns both:
    - V(s): State value function
    - A(s, a): Action advantage function
    
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """Initialize dueling network.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """Compute Q-values using dueling architecture.
        
        Args:
            state: Batch of states (batch_size, state_dim)
            action_mask: Valid action mask (batch_size, action_dim)
            
        Returns:
            Q-values (batch_size, action_dim)
        """
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Apply mask before computing mean advantage
        if action_mask is not None:
            masked_advantages = advantages.masked_fill(~action_mask, 0)
            advantage_mean = masked_advantages.sum(dim=1, keepdim=True) / action_mask.sum(dim=1, keepdim=True)
        else:
            advantage_mean = advantages.mean(dim=1, keepdim=True)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))
        q_values = value + (advantages - advantage_mean)
        
        # Mask invalid actions
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask, float('-inf'))
        
        return q_values


class AttentionQNetwork(nn.Module):
    """Q-Network with attention mechanism over table features.
    
    Learns which table statistics are most important for join decisions.
    """
    
    def __init__(self, state_dim: int, action_dim: int, num_heads: int = 4):
        """Initialize attention-based Q-network.
        
        Args:
            state_dim: State dimension
            action_dim: Number of actions
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Embedding layer
        self.embedding = nn.Linear(state_dim, 256)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Output layer
        self.output_layer = nn.Linear(256, action_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with attention.
        
        Args:
            state: State tensor (batch_size, state_dim)
            action_mask: Valid actions mask (batch_size, action_dim)
            
        Returns:
            Q-values (batch_size, action_dim)
        """
        # Embed state
        x = self.embedding(state)
        
        # Add batch dimension for attention
        x = x.unsqueeze(1)  # (batch_size, 1, 256)
        
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        
        # Feed-forward
        x = self.ffn(attn_output)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # (batch_size, 256)
        
        # Output Q-values
        q_values = self.output_layer(x)
        
        # Mask invalid actions
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask, float('-inf'))
        
        return q_values


def create_network(network_type: str, state_dim: int, action_dim: int, **kwargs) -> nn.Module:
    """Factory function to create Q-networks.
    
    Args:
        network_type: One of 'standard', 'dueling', 'attention'
        state_dim: State dimension
        action_dim: Action dimension
        **kwargs: Additional network-specific parameters
        
    Returns:
        Initialized Q-network
    """
    if network_type == 'standard':
        return QNetwork(state_dim, action_dim, **kwargs)
    elif network_type == 'dueling':
        return DuelingQNetwork(state_dim, action_dim, **kwargs)
    elif network_type == 'attention':
        return AttentionQNetwork(state_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

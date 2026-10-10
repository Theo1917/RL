"""Deep Q-Network agent for query optimization."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

from .network import create_network
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """Deep Q-Network agent with experience replay and target network."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        network_type: str = 'standard',
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = 'auto',
        seed: int = 42
    ):
        """Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            network_type: 'standard', 'dueling', or 'attention'
            learning_rate: Optimizer learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exponential decay rate for epsilon
            buffer_capacity: Replay buffer size
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            device: 'cpu', 'cuda', or 'auto'
            seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.seed = seed
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self.q_network = create_network(network_type, state_dim, action_dim).to(self.device)
        self.target_network = create_network(network_type, state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)
        
        # Training stats
        self.steps = 0
        self.episodes = 0
        self.losses = []
        
        # Random number generator
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
    
    def select_action(self, state: np.ndarray, valid_actions: np.ndarray, 
                     training: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            valid_actions: Boolean mask of valid actions
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if training and self.rng.random() < self.epsilon:
            # Random valid action
            valid_indices = np.where(valid_actions)[0]
            return self.rng.choice(valid_indices)
        
        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_tensor = torch.BoolTensor(valid_actions).unsqueeze(0).to(self.device)
            
            q_values = self.q_network(state_tensor, mask_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool, valid_actions: np.ndarray):
        """Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Episode termination flag
            valid_actions: Valid actions in next state
        """
        valid_action_indices = np.where(valid_actions)[0].tolist()
        self.replay_buffer.push(state, action, reward, next_state, done, valid_action_indices)
    
    def train_step(self) -> Optional[float]:
        """Perform one training step (if enough samples available).
        
        Returns:
            Loss value, or None if not enough samples
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones, masks = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        masks = torch.BoolTensor(masks).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states, masks).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss (Huber loss for stability)
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def update_epsilon(self):
        """Decay epsilon after episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes += 1
    
    def save(self, path: str):
        """Save agent state to disk.
        
        Args:
            path: Directory path to save checkpoint
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon
        }, path / 'checkpoint.pt')
        
        # Save hyperparameters
        config = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'seed': self.seed
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: str):
        """Load agent state from disk.
        
        Args:
            path: Directory path containing checkpoint
        """
        path = Path(path)
        
        checkpoint = torch.load(path / 'checkpoint.pt', map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.epsilon = checkpoint['epsilon']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics.
        
        Returns:
            Dictionary of current stats
        """
        return {
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss_last_100': np.mean(self.losses[-100:]) if self.losses else 0.0
        }


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent to reduce overestimation bias."""
    
    def train_step(self) -> Optional[float]:
        """Training step using Double DQN target computation."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones, masks = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        masks = torch.BoolTensor(masks).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            # Select best action using online network
            next_actions = self.q_network(next_states, masks).argmax(1)
            # Evaluate using target network
            next_q = self.target_network(next_states, masks).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value

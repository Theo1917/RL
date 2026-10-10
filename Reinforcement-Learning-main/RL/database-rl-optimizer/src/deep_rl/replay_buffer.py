"""Experience replay buffer for DQN training."""

import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple


Transition = namedtuple('Transition', 
                       ['state', 'action', 'reward', 'next_state', 'done', 'valid_actions'])


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, capacity: int = 100000, seed: int = 42):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for sampling
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.rng = random.Random(seed)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, valid_actions: List[int]):
        """Add a transition to the buffer.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode terminated
            valid_actions: Valid actions in next state (for masking)
        """
        self.buffer.append(Transition(state, action, reward, next_state, done, valid_actions))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, valid_actions_masks)
        """
        transitions = self.rng.sample(self.buffer, batch_size)
        
        # Transpose batch for easier processing
        batch = Transition(*zip(*transitions))
        
        states = np.array(batch.state)
        actions = np.array(batch.action)
        rewards = np.array(batch.reward)
        next_states = np.array(batch.next_state)
        dones = np.array(batch.done)
        
        # Create action masks from valid actions lists
        max_actions = max(len(va) for va in batch.valid_actions)
        masks = []
        for valid_actions in batch.valid_actions:
            mask = np.zeros(max_actions, dtype=bool)
            mask[valid_actions] = True
            masks.append(mask)
        
        return states, actions, rewards, next_states, dones, np.array(masks)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all stored transitions."""
        self.buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """Experience replay with prioritized sampling (optional enhancement)."""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, seed: int = 42):
        """Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            seed: Random seed
        """
        super().__init__(capacity, seed)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, valid_actions: List[int],
             priority: float = None):
        """Add transition with priority.
        
        Args:
            priority: Initial priority (defaults to max existing priority)
        """
        max_priority = max(self.priorities) if self.priorities else 1.0
        priority = priority if priority is not None else max_priority
        
        super().push(state, action, reward, next_state, done, valid_actions)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """Sample batch with importance sampling weights.
        
        Args:
            batch_size: Number of samples
            beta: Importance sampling exponent (annealed from 0.4 to 1.0)
            
        Returns:
            Same as ReplayBuffer.sample() plus importance sampling weights
        """
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = self.rng.choices(range(len(self.buffer)), weights=probs, k=batch_size)
        
        transitions = [self.buffer[idx] for idx in indices]
        batch = Transition(*zip(*transitions))
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        states = np.array(batch.state)
        actions = np.array(batch.action)
        rewards = np.array(batch.reward)
        next_states = np.array(batch.next_state)
        dones = np.array(batch.done)
        
        max_actions = max(len(va) for va in batch.valid_actions)
        masks = []
        for valid_actions in batch.valid_actions:
            mask = np.zeros(max_actions, dtype=bool)
            mask[valid_actions] = True
            masks.append(mask)
        
        return states, actions, rewards, next_states, dones, np.array(masks), weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values (e.g., TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

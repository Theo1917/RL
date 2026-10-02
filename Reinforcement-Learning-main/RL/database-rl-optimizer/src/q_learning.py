"""Small, masked tabular Q-learning implementation for join ordering."""

from collections import defaultdict
import math
import random
from typing import Any, Dict, List, Optional


class QLearningAgent:
	def __init__(self, alpha: float = 0.1, gamma: float = 0.95,
				 epsilon: float = 1.0, epsilon_schedule: str = "constant",
				 epsilon_min: float = 0.05, epsilon_decay: float = 0.995,
				 episodes: int = 500, seed: int = 0,
				 optimistic_init: float = 0.0) -> None:
		if not 0 <= alpha <= 1 or not 0 <= gamma <= 1:
			raise ValueError("alpha and gamma must be in [0, 1]")
		if epsilon_schedule not in ("constant", "linear", "exponential"):
			raise ValueError("unknown epsilon schedule")
		self.alpha, self.gamma = alpha, gamma
		self.epsilon = epsilon
		self.initial_epsilon = epsilon
		self.epsilon_schedule = epsilon_schedule
		self.epsilon_min, self.epsilon_decay = epsilon_min, epsilon_decay
		self.episodes = episodes
		self.rng = random.Random(seed)
		self.optimistic_init = optimistic_init
		self.q_table: Dict[Any, Dict[int, float]] = defaultdict(dict)
		self.visited_states, self.visited_actions = set(), set()
		self.history: List[Dict[str, float]] = []

	@property
	def Q(self):
		return self.q_table

	def _value(self, state, action) -> float:
		return self.q_table[state].get(action, self.optimistic_init)

	def choose_action(self, env, greedy: bool = False) -> int:
		actions = env.valid_actions()
		if not actions:
			raise RuntimeError("No valid actions remain")
		state = env.state_key()
		explore = not greedy and self.rng.random() < self.epsilon
		if explore:
			action = self.rng.choice(actions)
		else:
			best = max(self._value(state, action) for action in actions)
			action = min(action for action in actions
						 if self._value(state, action) == best)
		self.visited_states.add(state)
		self.visited_actions.add((state, action))
		return action

	def update(self, state, action: int, reward: float, next_state,
			   next_actions: List[int], terminal: bool = False) -> float:
		old = self._value(state, action)
		target = reward if terminal else reward + self.gamma * max(
			(self._value(next_state, candidate) for candidate in next_actions),
			default=0.0)
		error = target - old
		self.q_table[state][action] = old + self.alpha * error
		return abs(error)

	def train(self, env, episodes: Optional[int] = None) -> List[Dict[str, float]]:
		count = self.episodes if episodes is None else episodes
		self.history = []
		for episode in range(count):
			env.reset()
			total_return, total_cost, td_error = 0.0, 0.0, 0.0
			actions_taken = 0
			while True:
				state = env.state_key()
				action = self.choose_action(env)
				_, reward, terminated, truncated, info = env.step(action)
				next_state = env.state_key()
				td_error += self.update(state, action, reward, next_state,
										env.valid_actions(), terminated or truncated)
				total_return += reward
				total_cost -= reward
				actions_taken += 1
				if terminated or truncated:
					break
			self.history.append({
				"episode": episode + 1, "return": total_return,
				"plan_cost": total_cost, "epsilon": self.epsilon,
				"unique_states": len(self.visited_states),
				"unique_state_actions": len(self.visited_actions),
				"td_error": td_error,
				"exploration_ratio": 0.0 if not actions_taken else
					float(sum(1 for _ in range(0)) / actions_taken),
			})
			self._decay_epsilon(episode, count)
		return self.history

	def _decay_epsilon(self, episode: int, total: int) -> None:
		if self.epsilon_schedule == "constant":
			return
		if self.epsilon_schedule == "linear":
			self.epsilon = max(self.epsilon_min,
							   self.initial_epsilon - (self.initial_epsilon - self.epsilon_min) /
							   max(1, total - 1))
		else:
			self.epsilon = max(self.epsilon_min,
							   self.initial_epsilon * self.epsilon_decay ** (episode + 1))

	def greedy_plan(self, env):
		env.reset()
		while not env.is_complete:
			env.step(self.choose_action(env, greedy=True))
		return env.plan, env.accumulated_cost

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
					 optimistic_init: float = 0.0, shared: bool = False,
					 state_mode: Optional[str] = None) -> None:
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
		self.shared = bool(shared if state_mode is None else state_mode == "shared")
		if state_mode not in (None, "per_query", "shared"):
			raise ValueError("state_mode must be 'per_query' or 'shared'")
		self.q_table: Dict[Any, Dict[int, float]] = defaultdict(dict)
		self.visited_states, self.visited_actions = set(), set()
		self.visit_counts: Dict[Any, int] = defaultdict(int)
		self.exploration_count = 0
		self.last_exploratory = False
		self.history: List[Dict[str, float]] = []

	@property
	def Q(self):
		return self.q_table

	def _value(self, state, action) -> float:
		return self.q_table[state].get(action, self.optimistic_init)

	def _state_key(self, env):
		if not self.shared:
			return env.state_key()
		obs = env._observation()
		joined = tuple(int(value) for value in obs["joined"])
		cardinality = int(obs["log_cardinality"][0])
		selectivities = tuple(
			-1 if value <= 0 else min(
				12, int(-math.log10(max(float(value), 1e-12))))
			for value in obs["candidate_selectivity"]
		)
		indexes = tuple(int(value) for value in obs.get(
			"has_index", ()))
		return (sum(joined), joined, cardinality, selectivities, indexes)

	def state_key(self, env):
		"""Return the configured per-query or bucketed shared state key."""
		return self._state_key(env)

	def choose_action(self, env, greedy: bool = False) -> int:
		actions = env.valid_actions()
		if not actions:
			raise RuntimeError("No valid actions remain")
		state = self._state_key(env)
		explore = not greedy and self.rng.random() < self.epsilon
		self.last_exploratory = explore
		if explore:
			self.exploration_count += 1
		if explore:
			action = self.rng.choice(actions)
		else:
			best = max(self._value(state, action) for action in actions)
			action = min(action for action in actions
						 if self._value(state, action) == best)
		self.visited_states.add(state)
		self.visited_actions.add((state, action))
		self.visit_counts[(state, action)] += 1
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
			exploratory_actions = 0
			while True:
				state = self._state_key(env)
				action = self.choose_action(env)
				exploratory_actions += int(self.last_exploratory)
				_, reward, terminated, truncated, info = env.step(action)
				next_state = self._state_key(env)
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
				"exploration_ratio": (exploratory_actions / actions_taken
					if actions_taken else 0.0),
				"exploratory_actions": exploratory_actions,
			})
			self._decay_epsilon(episode, count)
		return self.history

	def _decay_epsilon(self, episode: int, total: int) -> None:
		if self.epsilon_schedule == "constant":
			return
		if self.epsilon_schedule == "linear":
			self.epsilon = max(self.epsilon_min,
							   self.initial_epsilon - (self.initial_epsilon - self.epsilon_min) *
							   (episode + 1) / max(1, total - 1))
		else:
			self.epsilon = max(self.epsilon_min,
							   self.initial_epsilon * self.epsilon_decay ** (episode + 1))

	def greedy_plan(self, env):
		env.reset()
		while not env.is_complete:
			env.step(self.choose_action(env, greedy=True))
		return env.plan, env.accumulated_cost

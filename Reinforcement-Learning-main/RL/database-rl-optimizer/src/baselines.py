"""Reference policies for join-order experiments."""

import random
from typing import Any, Dict, List, Tuple


class RandomAgent:
	def __init__(self, seed: int = 0) -> None:
		self.rng = random.Random(seed)

	def choose_action(self, env) -> int:
		actions = env.valid_actions()
		if not actions:
			raise RuntimeError("No valid actions remain")
		return self.rng.choice(actions)


class GreedyMinIntermediate:
	def choose_action(self, env) -> int:
		actions = env.valid_actions()
		if not actions:
			raise RuntimeError("No valid actions remain")
		if env.plan is None:
			return actions[0]
		return min(actions, key=lambda action: (
			env.cost_model.estimate_plan_join(
				env.plan, env.action_to_table(action)).output_cardinality,
			action,
		))


class GreedyMinImmediateCost:
	def choose_action(self, env) -> int:
		actions = env.valid_actions()
		if not actions:
			raise RuntimeError("No valid actions remain")
		if env.plan is None:
			return actions[0]
		return min(actions, key=lambda action: (
			env.cost_model.estimate_plan_join(
				env.plan, env.action_to_table(action)).cost,
			action,
		))


class ExhaustiveOptimal:
	"""Enumerate every connected left-deep plan and retain the best one."""

	def __init__(self, env=None) -> None:
		self.best_plan: List[int] = []
		self.best_cost = float("inf")
		if env is not None:
			self._solve(env)

	def _solve(self, env) -> None:
		candidates = []

		def search(sequence: List[int]) -> None:
			probe = type(env)(env.query, cost_model=env.cost_model,
							  start_mode=env.start_mode,
							  start_table=env._default_start)
			probe.reset()
			total = 0.0
			for action in sequence:
				_, reward, terminated, _, _ = probe.step(action)
				total -= reward
				if terminated:
					candidates.append((total, sequence.copy()))
					return
			if probe.is_complete:
				candidates.append((total, sequence.copy()))
				return
			for action in probe.valid_actions():
				search(sequence + [action])

		search([])
		self.best_cost, self.best_plan = min(candidates, key=lambda item: (item[0], item[1]))

	def choose_action(self, env) -> int:
		if not self.best_plan:
			self._solve(env)
		index = len(env.plan.join_sequence) if env.plan is not None else 0
		if env.start_mode == "agent" and env.plan is not None:
			index += 1
		return self.best_plan[index]


def run_episode(env, agent) -> Tuple[Any, float, Dict[str, Any]]:
	"""Run a policy until termination and return plan, cost and episode info."""
	env.reset()
	total_cost = 0.0
	info: Dict[str, Any] = {}
	while True:
		action = agent.choose_action(env)
		_, reward, terminated, truncated, info = env.step(action)
		total_cost -= reward
		if terminated or truncated:
			break
	return env.plan, total_cost, info

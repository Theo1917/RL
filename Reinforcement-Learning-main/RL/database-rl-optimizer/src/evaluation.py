"""Reproducible metrics and experiment helpers for simulated join planning."""

import csv
import itertools
import json
import math
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence

from src.baselines import ExhaustiveOptimal, run_episode
from src.cost_model import CostModel
from src.database import generate_database
from src.query import generate_query
from src.environment import JoinOrderEnv


def evaluate_agent(env: JoinOrderEnv, agent, optimal_cost: float = None) -> Dict[str, Any]:
	plan, cost, info = run_episode(env, agent)
	peak_rows = 0.0
	joined = set(plan.joined_tables) - {table for _, table in plan.join_sequence}
	for _, table in plan.join_sequence:
		joined.add(table)
		peak_rows = max(peak_rows, env.cost_model.set_cardinality(joined))
	if not plan.join_sequence:
		peak_rows = env.cost_model.current_cardinality(plan)
	result = {
		"plan_cost": cost,
		"optimal_cost": optimal_cost,
		"suboptimality_ratio": cost / optimal_cost if optimal_cost else None,
		"suboptimality_gap": (cost - optimal_cost) / optimal_cost
			if optimal_cost else None,
		"peak_intermediate_rows": peak_rows,
		"memory_proxy": peak_rows,
		"latency_proxy": math.log10(1.0 + cost),
		"optimal": optimal_cost is not None and math.isclose(cost, optimal_cost),
		"plan": list(plan.join_sequence),
		"start_table": info.get("start_table"),
	}
	return result


def evaluate_agents(env: JoinOrderEnv, agents: Dict[str, Any]) -> List[Dict[str, Any]]:
	optimal = ExhaustiveOptimal(env).best_cost
	return [{"agent": name, **evaluate_agent(env, agent, optimal)}
			for name, agent in agents.items()]


def train_agent_on_queries(agent, queries: Sequence[Any],
						 episodes_per_query: int = 100):
	"""Train one shared agent sequentially across query instances."""
	for query in queries:
		env = JoinOrderEnv(query, start_mode="agent")
		agent.train(env, episodes=episodes_per_query)
	return agent


def evaluate_shared_agent(agent, queries: Sequence[Any]) -> List[Dict[str, Any]]:
	"""Evaluate a shared agent on unseen queries using only valid actions."""
	results = []
	for query in queries:
		env = JoinOrderEnv(query, start_mode="agent")
		optimal = ExhaustiveOptimal(env).best_cost
		results.append({"query_id": query.query_id,
						**evaluate_agent(env, agent, optimal)})
	return results


def confidence_interval(values: Sequence[float], confidence: float = 0.95):
	if not values:
		return (None, None)
	if len(values) == 1:
		return values[0], values[0]
	half_width = 1.96 * stdev(values) / math.sqrt(len(values))
	return mean(values) - half_width, mean(values) + half_width


def plan_costs(query, start_mode="agent") -> List[float]:
	"""Enumerate valid plan costs for a query using the simulator."""
	env = JoinOrderEnv(query, start_mode=start_mode)
	costs = []

	def search(sequence):
		probe = JoinOrderEnv(query, start_mode=start_mode)
		probe.reset()
		total = 0.0
		for action in sequence:
			_, reward, terminated, _, _ = probe.step(action)
			total -= reward
			if terminated:
				costs.append(total)
				return
		for action in probe.valid_actions():
			search(sequence + [action])

	search([])
	return costs


def compute_plan_cost_spread(query, start_mode="agent") -> float:
	"""Return worst/best valid-plan cost ratio for difficulty analysis."""
	costs = plan_costs(query, start_mode=start_mode)
	if not costs or min(costs) <= 0:
		return 1.0
	return max(costs) / min(costs)


def spread_distribution(queries: Sequence[Any]) -> List[Dict[str, Any]]:
	return [{"query_id": query.query_id, "tables": len(query.tables),
				"spread": compute_plan_cost_spread(query)} for query in queries]


def make_query_set(table_counts=(3, 4, 5, 6), seeds=range(10),
				   min_spread=None, return_metadata=False):
	queries = [generate_query(generate_database(count, seed), seed)
				for count, seed in itertools.product(table_counts, seeds)]
	if min_spread is not None:
		queries = [query for query in queries
				   if compute_plan_cost_spread(query) >= min_spread]
	if return_metadata:
		return queries, spread_distribution(queries)
	return queries


def train_test_split(queries: Sequence[Any], fraction: float = 0.6,
					 seed: int = 0):
	"""Split reproducibly while preserving each table-count stratum."""
	rng = random.Random(seed)
	groups = {}
	for query in queries:
		groups.setdefault(len(query.tables), []).append(query)
	train, test = [], []
	for group in groups.values():
		items = list(group)
		rng.shuffle(items)
		cut = min(len(items), max(1, round(len(items) * fraction)))
		train.extend(items[:cut])
		test.extend(items[cut:])
	return train, test


def save_results(records: Iterable[Dict[str, Any]], directory="results",
				 config=None, seed=None, git_commit=None) -> Path:
	directory = Path(directory)
	directory.mkdir(parents=True, exist_ok=True)
	records = list(records)
	timestamp = datetime.now(timezone.utc).isoformat()
	if git_commit is None:
		try:
			git_commit = subprocess.check_output(
				["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1],
				text=True, stderr=subprocess.DEVNULL).strip()
		except (OSError, subprocess.CalledProcessError):
			git_commit = None
	metadata = {"timestamp": timestamp, "config": config or {},
					"seed": seed, "git_commit": git_commit}
	for record in records:
		record.setdefault("run_metadata", metadata)
	runs_path = directory / "runs.json"
	previous = []
	if runs_path.exists():
		try:
			previous = json.loads(runs_path.read_text())
		except (OSError, json.JSONDecodeError):
			previous = []
	if not isinstance(previous, list):
		previous = []
	all_records = previous + records
	runs_path.write_text(json.dumps(all_records, indent=2, default=str))
	if records:
		fields = sorted({key for record in records for key in record})
		csv_path = directory / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
		with csv_path.open("w", newline="") as handle:
			writer = csv.DictWriter(handle, fieldnames=fields)
			writer.writeheader()
			writer.writerows(records)
	return directory

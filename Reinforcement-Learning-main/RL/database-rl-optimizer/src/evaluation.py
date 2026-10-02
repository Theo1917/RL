"""Reproducible metrics and experiment helpers for simulated join planning."""

import csv
import itertools
import json
import math
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


def confidence_interval(values: Sequence[float], confidence: float = 0.95):
	if not values:
		return (None, None)
	if len(values) == 1:
		return values[0], values[0]
	half_width = 1.96 * stdev(values) / math.sqrt(len(values))
	return mean(values) - half_width, mean(values) + half_width


def make_query_set(table_counts=(3, 4, 5, 6), seeds=range(10)):
	return [generate_query(generate_database(count, seed), seed)
			for count, seed in itertools.product(table_counts, seeds)]


def train_test_split(queries: Sequence[Any], fraction: float = 0.6):
	split = max(1, int(len(queries) * fraction))
	return list(queries[:split]), list(queries[split:])


def save_results(records: Iterable[Dict[str, Any]], directory="results") -> Path:
	directory = Path(directory)
	directory.mkdir(parents=True, exist_ok=True)
	records = list(records)
	(directory / "runs.json").write_text(json.dumps(records, indent=2, default=str))
	if records:
		fields = sorted({key for record in records for key in record})
		with (directory / "runs.csv").open("w", newline="") as handle:
			writer = csv.DictWriter(handle, fieldnames=fields)
			writer.writeheader()
			writer.writerows(records)
	return directory

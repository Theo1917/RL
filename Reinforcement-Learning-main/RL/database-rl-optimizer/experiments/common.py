"""Shared, seeded helpers for the reproducible experiment scripts."""

import csv
from pathlib import Path
import statistics
import time

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"

from src.baselines import (ExhaustiveOptimal, GreedyMinImmediateCost,
                           GreedyMinIntermediate, RandomAgent)
from src.database import generate_database
from src.environment import JoinOrderEnv
from src.evaluation import evaluate_agent
from src.q_learning import QLearningAgent
from src.query import generate_query


def make_queries(table_counts=(3, 4, 5, 6), seeds=range(10)):
    return [(count, seed,
             generate_query(generate_database(count, seed=seed), seed=seed))
            for count in table_counts for seed in seeds]


def train_query(query, episodes=250, seed=0, shared=False,
                reward_weights=None, **kwargs):
    env = JoinOrderEnv(query, start_mode="agent",
                       reward_weights=reward_weights)
    agent = QLearningAgent(episodes=episodes, seed=seed, shared=shared, **kwargs)
    history = agent.train(env)
    agent.epsilon = 0.0
    return agent, history


def query_row(query, count, seed, episodes=250, shared=False,
              reward_weights=None):
    env = JoinOrderEnv(query, start_mode="agent",
                       reward_weights=reward_weights)
    optimal = ExhaustiveOptimal(env).best_cost
    started = time.perf_counter()
    learner, history = train_query(query, episodes, seed, shared,
                                   reward_weights=reward_weights)
    elapsed = time.perf_counter() - started
    rows = []
    agents = {
        "random": RandomAgent(seed),
        "greedy_intermediate": GreedyMinIntermediate(),
        "greedy_cost": GreedyMinImmediateCost(),
        "q_learning_shared" if shared else "q_learning": learner,
        "optimal": ExhaustiveOptimal(env),
    }
    for name, agent in agents.items():
        metrics = evaluate_agent(env, agent, optimal)
        rows.append({"tables": count, "seed": seed, "agent": name,
                     "training_episodes": episodes,
                     "runtime_seconds": elapsed if name.startswith("q_") else 0.0,
                     **{key: value for key, value in metrics.items()
                        if key not in ("plan",)}})
    return rows, history


def write_csv(rows, filename):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / filename
    if not rows:
        return path
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def mean_ci(values):
    values = list(values)
    if len(values) < 2:
        return (statistics.mean(values) if values else None, 0.0)
    return statistics.mean(values), 1.96 * statistics.stdev(values) / len(values) ** 0.5


def save_plot(figure, filename):
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / filename
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_plan_tree(plan, filename="plan_tree.png"):
    figure, axis = plt.subplots(figsize=(7, 3))
    axis.axis("off")
    axis.text(0.02, 0.5, "start", fontsize=11, weight="bold")
    for index, (left, right) in enumerate(plan.join_sequence):
        x = 0.18 + index * 0.2
        axis.annotate("", (x, 0.5), (x - 0.13, 0.5),
                     arrowprops={"arrowstyle": "->", "color": "#155d56"})
        axis.text(x, 0.5, f"{left} + {right}", ha="center", va="center",
                  bbox={"boxstyle": "round,pad=.4", "fc": "#eff8d7",
                        "ec": "#c9ee68"}, fontsize=8)
    return save_plot(figure, filename)
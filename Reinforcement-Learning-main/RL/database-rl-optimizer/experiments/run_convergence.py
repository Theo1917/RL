"""Convergence thresholds and smoothed learning curves."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import make_queries, mean_ci, save_plot, train_query, write_csv
import matplotlib.pyplot as plt
from src.baselines import ExhaustiveOptimal
from src.environment import JoinOrderEnv


def episodes_to_threshold(history, optimal, threshold):
    for row in history:
        if row["plan_cost"] <= optimal * (1 + threshold):
            return row["episode"]
    return None


def main(episodes=500):
    rows, curves = [], []
    for count, seed, query in make_queries(seeds=range(10)):
        env = JoinOrderEnv(query, start_mode="agent")
        optimal = ExhaustiveOptimal(env).best_cost
        agent, history = train_query(query, episodes, seed)
        rows.append({"tables": count, "seed": seed,
                     "within_5_percent": episodes_to_threshold(history, optimal, .05),
                     "within_1_percent": episodes_to_threshold(history, optimal, .01)})
        curves.append(history)
    write_csv(rows, "convergence.csv")
    figure, axis = plt.subplots(figsize=(8, 4))
    width = max(len(curves[0]), 1)
    values = [sum(curve[i]["plan_cost"] for curve in curves) / len(curves)
              for i in range(width)]
    axis.plot(values)
    axis.set(xlabel="Episode", ylabel="Mean estimated cost")
    save_plot(figure, "learning_curve_ci.png")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    main(parser.parse_args().episodes)
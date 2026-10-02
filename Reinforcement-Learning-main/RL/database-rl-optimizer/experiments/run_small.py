"""Fast seeded end-to-end comparison for one four-table query."""
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines import (ExhaustiveOptimal, GreedyMinImmediateCost,
                           GreedyMinIntermediate, RandomAgent, run_episode)
from src.database import generate_database
from src.environment import JoinOrderEnv
from src.evaluation import save_results
from src.q_learning import QLearningAgent
from src.query import generate_query


def main(seed=42):
    query = generate_query(generate_database(4, seed), seed)
    env = JoinOrderEnv(query, start_mode="agent")
    optimal = ExhaustiveOptimal(env)
    learner = QLearningAgent(episodes=500, epsilon_decay=0.98, seed=seed)
    history = learner.train(env)
    agents = {"random": RandomAgent(seed),
              "greedy_intermediate": GreedyMinIntermediate(),
              "greedy_cost": GreedyMinImmediateCost(),
              "q_learning": learner, "optimal": optimal}
    rows = []
    for name, agent in agents.items():
        if name == "q_learning":
            agent.epsilon = 0.0
        plan, cost, _ = run_episode(env, agent)
        rows.append({"agent": name, "plan_cost": cost,
                     "ratio_to_optimal": cost / optimal.best_cost,
                     "plan": str(plan.join_sequence)})
    for row in rows:
        print(f"{row['agent']:<22} {row['plan_cost']:>12.2f} "
              f"{row['ratio_to_optimal']:>8.3f}")
    save_results(rows)
    plot_dir = Path("results/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.plot([row["plan_cost"] for row in history])
    plt.xlabel("Episode")
    plt.ylabel("Estimated plan cost")
    plt.tight_layout()
    plt.savefig(plot_dir / "small_learning_curve.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args().seed)
"""Terminal live demo for the adaptive join-order optimizer."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import (ExhaustiveOptimal, GreedyMinImmediateCost,
                           GreedyMinIntermediate, RandomAgent, run_episode)
from src.database import generate_database
from src.environment import JoinOrderEnv
from src.q_learning import QLearningAgent
from src.query import generate_query


def print_schema(query):
    print("\n=== SCHEMA AND QUERY ===")
    for table in query.database.tables:
        indexes = ", ".join(table.indexed_columns) or "none"
        print(f"{table.name}: {table.row_count:,} rows; indexes: {indexes}")
    print("Join graph:")
    for join in query.joins:
        print(f"  {join}  selectivity={join.selectivity:.5g}")
    print("\nConceptual SQL:\n" + query.to_sql())


def trace_episode(env, agent):
    env.reset()
    print("\n=== MDP WALKTHROUGH ===")
    while True:
        state = sorted(env.state_key())
        valid = env.valid_actions()
        action = agent.choose_action(env, greedy=True)
        _, reward, terminated, truncated, info = env.step(action)
        print(f"state={state} valid={valid} action={action}:{info['selected_table']} "
              f"reward={reward:,.2f}")
        if terminated or truncated:
            return env.accumulated_cost


def train_live(agent, env):
    print("\n=== Q-LEARNING TRAINING ===")
    history = []
    for episode in range(agent.episodes):
        agent.train(env, episodes=1)
        agent.epsilon = max(
            agent.epsilon_min,
            agent.initial_epsilon * agent.epsilon_decay ** (episode + 1))
        history.append(agent.history[-1])
        if episode == 0 or (episode + 1) % max(1, agent.episodes // 10) == 0:
            row = history[-1]
            print(f"episode {episode + 1:>4}/{agent.episodes} "
                  f"cost={row['plan_cost']:>12.2f} epsilon={agent.epsilon:.3f}")
    return history


def load_postgres_fallback(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main(seed=42, episodes=500, no_postgres=False, replay=False):
    query = generate_query(generate_database(4, seed=seed, indexed=True), seed=seed)
    print_schema(query)
    env = JoinOrderEnv(query, start_mode="agent")
    learner = QLearningAgent(episodes=episodes, epsilon=1.0,
                             epsilon_schedule="exponential",
                             epsilon_decay=0.98, seed=seed)
    trace_episode(env, learner)
    train_live(learner, env)
    learner.epsilon = 0.0
    optimal = ExhaustiveOptimal(env)
    agents = {"random": RandomAgent(seed),
              "greedy-intermediate": GreedyMinIntermediate(),
              "greedy-cost": GreedyMinImmediateCost(),
              "q-learning": learner, "optimal": optimal}
    print("\n=== RESULTS ===")
    results = []
    for name, agent in agents.items():
        plan, cost, _ = run_episode(env, agent)
        ratio = cost / optimal.best_cost
        results.append((name, cost, ratio))
        print(f"{name:<22} cost={cost:>12.2f} ratio={ratio:>6.3f} "
              f"plan={plan.join_sequence}")
    learned_ratio = next(ratio for name, _, ratio in results
                         if name == "q-learning")
    status = "converged" if learned_ratio <= 1.000001 else "not converged"
    print(f"\nQ-learning gap to exhaustive optimum: "
          f"{(learned_ratio - 1.0) * 100:.2f}% ({status})")

    if no_postgres:
        print("\nPostgreSQL: skipped (--no-postgres).")
    else:
        fallback = (load_postgres_fallback(ROOT / "results" / "pg" / "summary.json")
                    if replay else None)
        if fallback is None:
            print("\nPostgreSQL: unavailable; simulator results shown above. "
                  "Use --replay after saving results/pg/summary.json.")
        else:
            print(f"\nPostgreSQL fallback: {fallback}")

    output_dir = ROOT / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar([name for name, _, _ in results], [cost for _, cost, _ in results])
    axis.set_ylabel("Estimated simulator cost")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output_dir / "demo_comparison.png", dpi=140)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--no-postgres", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    main(args.seed, args.episodes, args.no_postgres, args.replay)

"""Pure cost versus normalized multi-objective reward ablation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import make_queries, train_query, write_csv
from src.environment import JoinOrderEnv
from src.baselines import ExhaustiveOptimal


def main(episodes=250):
    rows = []
    weights = {"cost": 1.0, "inter": .25, "mem": .25, "lat": .1}
    for count, seed, query in make_queries(table_counts=(3, 4, 5), seeds=range(10)):
        for name, reward_weights in (("pure_cost", None), ("multi_objective", weights)):
            env = JoinOrderEnv(query, start_mode="agent", reward_weights=reward_weights)
            optimal = ExhaustiveOptimal(env).best_cost
            agent, _ = train_query(query, episodes, seed,
                                   reward_weights=reward_weights)
            plan, cost = agent.greedy_plan(env)
            rows.append({"agent": name, "tables": count, "seed": seed,
                         "plan_cost": cost, "ratio": cost / optimal,
                         "peak_rows": env.cost_model.current_cardinality(plan),
                         "weights": reward_weights or {"cost": 1.0}})
    write_csv(rows, "reward_ablation.csv")
    return rows


if __name__ == "__main__":
    main()
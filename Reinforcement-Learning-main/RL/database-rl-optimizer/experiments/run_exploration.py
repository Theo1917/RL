"""Exploration schedule and alpha/gamma sensitivity study."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import make_queries, save_plot, train_query, write_csv
from src.baselines import ExhaustiveOptimal
from src.environment import JoinOrderEnv
import matplotlib.pyplot as plt


def main(episodes=250):
    configs = [{"name": "constant_0.05", "epsilon": .05, "epsilon_schedule": "constant"},
               {"name": "constant_0.1", "epsilon": .1, "epsilon_schedule": "constant"},
               {"name": "constant_0.3", "epsilon": .3, "epsilon_schedule": "constant"},
               {"name": "linear", "epsilon_schedule": "linear"},
               {"name": "exponential", "epsilon_schedule": "exponential"},
               {"name": "optimistic", "optimistic_init": 1.0},
               {"name": "alpha_0.3", "alpha": .3},
               {"name": "gamma_0.8", "gamma": .8}]
    rows = []
    query = make_queries(table_counts=(4,), seeds=(42,))[0][2]
    optimal = ExhaustiveOptimal(JoinOrderEnv(query, start_mode="agent")).best_cost
    for config in configs:
        agent, history = train_query(query, episodes, 42, **{k: v for k, v in config.items() if k != "name"})
        for record in history:
            rows.append({"config": config["name"], **record,
                         "visit_count": sum(agent.visit_counts.values()),
                         "cumulative_regret": sum(
                             item["plan_cost"] - optimal for item in history[:record["episode"]])})
    write_csv(rows, "exploration.csv")
    figure, axis = plt.subplots(figsize=(8, 4))
    for config in configs:
        values = [r["exploration_ratio"] for r in rows if r["config"] == config["name"]]
        axis.plot(values, label=config["name"])
    axis.set_ylabel("Exploration ratio"); axis.set_xlabel("Episode"); axis.legend()
    save_plot(figure, "exploration_ratio.png")
    return rows


if __name__ == "__main__":
    main()
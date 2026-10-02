"""Final gap and simulator cost evaluations versus training budget."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import make_queries, save_plot, write_csv, train_query
from src.baselines import ExhaustiveOptimal
from itertools import permutations
from src.environment import JoinOrderEnv
import matplotlib.pyplot as plt


def main(budgets=(10, 25, 50, 100, 250, 500, 1000, 2000)):
    rows = []
    query_set = make_queries(table_counts=(3, 4), seeds=range(10))
    for budget in budgets:
        for count, seed, query in query_set:
            env = JoinOrderEnv(query, start_mode="agent")
            optimal = ExhaustiveOptimal(env).best_cost
            agent, _ = train_query(query, budget, seed)
            cost = agent.greedy_plan(env)[1]
            exhaustive_evaluations = sum(
                1 for start in query.tables
                for _ in permutations([table for table in query.tables if table != start]))
            rows.append({"episodes": budget, "tables": count, "seed": seed,
                         "gap": cost / optimal - 1.0,
                         "rl_cost_evaluations": budget * count,
                         "exhaustive_cost_evaluations": exhaustive_evaluations})
    write_csv(rows, "sample_efficiency.csv")
    figure, axis = plt.subplots(figsize=(8, 4))
    for count in (3, 4):
        values = [sum(r["gap"] for r in rows if r["tables"] == count and r["episodes"] == budget) /
                  sum(1 for r in rows if r["tables"] == count and r["episodes"] == budget)
                  for budget in budgets]
        axis.plot(budgets, values, marker="o", label=f"{count} tables")
    axis.set_xscale("log"); axis.set_ylabel("Final suboptimality gap"); axis.set_xlabel("Episodes")
    axis.legend(); save_plot(figure, "sample_efficiency.png")
    return rows


if __name__ == "__main__":
    main()
"""Multi-seed baseline and Q-learning comparison."""
import argparse
import json
import sys
sys.path.insert(0, str(__file__).rsplit("experiments", 1)[0])
from experiments.common import (PLOTS, make_queries, mean_ci, plot_plan_tree,
                                query_row, save_plot, write_csv)
from src.evaluation import evaluate_shared_agent, train_agent_on_queries
from src.q_learning import QLearningAgent
import matplotlib.pyplot as plt


def main(seeds=range(10), episodes=250):
    rows = []
    histories = []
    for count, seed, query in make_queries(seeds=seeds):
        new_rows, history = query_row(query, count, seed, episodes)
        rows.extend(new_rows)
        histories.append((count, seed, history))
    for count in (3, 4, 5, 6):
        group = [(seed, query) for item_count, seed, query in make_queries(
            table_counts=(count,), seeds=seeds) if item_count == count]
        cut = max(1, int(len(group) * .6))
        shared = QLearningAgent(episodes=episodes, shared=True, seed=count)
        train_agent_on_queries(shared, [query for _, query in group[:cut]], episodes)
        for seed, query in group[cut:]:
            result = evaluate_shared_agent(shared, [query])[0]
            rows.append({"tables": count, "seed": seed,
                         "agent": "q_learning_shared",
                         "training_episodes": episodes,
                         **{key: value for key, value in result.items()
                            if key not in ("query_id", "plan")}})
    write_csv(rows, "comparison.csv")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["agent"], []).append(row["suboptimality_ratio"])
    summary = [{"agent": name, "mean_ratio": mean_ci(values)[0],
                "ci95": mean_ci(values)[1], "queries": len(values)}
               for name, values in grouped.items()]
    (PLOTS.parent / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar([item["agent"] for item in summary], [item["mean_ratio"] for item in summary])
    axis.set_ylabel("Mean ratio to exhaustive optimum")
    axis.tick_params(axis="x", rotation=25)
    save_plot(figure, "cost_ratio_comparison.png")
    figure, axis = plt.subplots(figsize=(8, 4))
    for agent, values in grouped.items():
        axis.boxplot(values, positions=[len(axis.lines) + 1], labels=[agent])
    axis.set_ylabel("Plan cost ratio")
    save_plot(figure, "cost_ratio_boxplot.png")
    if histories:
        query = make_queries(table_counts=(3,), seeds=(0,))[0][2]
        rows_for_plan, _ = query_row(query, 3, 0, episodes)
        from src.baselines import ExhaustiveOptimal
        from src.environment import JoinOrderEnv
        plan = ExhaustiveOptimal(JoinOrderEnv(query, start_mode="agent"))
        probe = JoinOrderEnv(query, start_mode="agent")
        probe.reset()
        for action in plan.best_plan:
            probe.step(action)
        plot_plan_tree(probe.plan)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=250)
    main(episodes=parser.parse_args().episodes)
"""Scaling study for 3, 4, 5, and 6 table queries."""
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import make_queries, train_query, write_csv
from src.baselines import ExhaustiveOptimal
from src.environment import JoinOrderEnv


def main(episodes=250):
    rows = []
    for count, seed, query in make_queries(seeds=range(10)):
        env = JoinOrderEnv(query, start_mode="agent")
        started = time.perf_counter()
        optimal = ExhaustiveOptimal(env).best_cost
        agent, _ = train_query(query, episodes, seed)
        cost = agent.greedy_plan(env)[1]
        rows.append({"tables": count, "seed": seed, "states_visited": len(agent.visited_states),
                     "state_actions_visited": len(agent.visited_actions),
                     "episodes": episodes, "final_gap": cost / optimal - 1,
                     "runtime_seconds": time.perf_counter() - started})
    write_csv(rows, "scaling.csv")
    return rows


if __name__ == "__main__":
    main()
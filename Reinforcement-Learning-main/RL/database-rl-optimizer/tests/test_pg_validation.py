from src.database import generate_database
from src.pg_validation import (join_order_sql, parse_explain_plan,
                               spearman_rank_correlation)
from src.environment import JoinOrderEnv
from src.baselines import ExhaustiveOptimal
from src.query import generate_query


def test_agent_plan_sql_starts_with_selected_start_table():
    query = generate_query(generate_database(3, 2), 2)
    env = JoinOrderEnv(query, start_mode="agent")
    optimal = ExhaustiveOptimal(env)
    env.reset()
    for action in optimal.best_plan:
        env.step(action)
    sql = join_order_sql(query, env.plan)
    start = next(iter(env.plan.joined_tables -
                      {right for _, right in env.plan.join_sequence}))
    assert f'FROM {start}' in sql
    assert "join_collapse_limit = 1" in sql
    assert sql.count("JOIN ") == len(query.tables) - 1


def test_explain_json_metrics_are_parsed():
    fixture = [{"Plan": {"Node Type": "Hash Join", "Actual Rows": 12,
                          "Total Cost": 8.5, "Memory Usage": 4,
                          "Shared Hit Blocks": 3,
                          "Plans": [{"Node Type": "Seq Scan", "Actual Rows": 20,
                                     "Shared Read Blocks": 2}]},
                "Planning Time": 0.4, "Execution Time": 1.2}]
    parsed = parse_explain_plan(fixture)
    assert parsed["total_cost"] == 8.5
    assert parsed["peak_actual_rows"] == 20
    assert parsed["memory_usage"] == 4
    assert parsed["shared_buffers"] == 5


def test_spearman_known_orderings():
    assert spearman_rank_correlation([1, 2, 3], [10, 20, 30]) == 1.0
    assert spearman_rank_correlation([1, 2, 3], [30, 20, 10]) == -1.0
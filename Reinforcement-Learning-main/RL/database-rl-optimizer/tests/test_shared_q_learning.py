from src.database import generate_database
from src.environment import JoinOrderEnv
from src.evaluation import evaluate_shared_agent, train_agent_on_queries
from src.q_learning import QLearningAgent
from src.query import generate_query


def make_query(table_count, seed):
    return generate_query(generate_database(table_count, seed), seed)


def test_shared_agent_runs_valid_unseen_query_to_completion():
    train_queries = [make_query(3, seed) for seed in range(3)]
    test_queries = [make_query(3, seed) for seed in (10, 11)]
    agent = QLearningAgent(episodes=40, shared=True, seed=12)
    train_agent_on_queries(agent, train_queries, episodes_per_query=40)
    results = evaluate_shared_agent(agent, test_queries)
    assert len(results) == 2
    assert all(result["plan_cost"] > 0 for result in results)
    assert all(result["start_table"] in query.tables
               for result, query in zip(results, test_queries))
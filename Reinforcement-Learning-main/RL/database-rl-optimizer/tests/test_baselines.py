from src.baselines import (ExhaustiveOptimal, GreedyMinImmediateCost,
                            GreedyMinIntermediate, RandomAgent, run_episode)
from src.database import generate_database
from src.environment import JoinOrderEnv
from src.query import generate_query


def make_env(seed, tables=3, start_mode="fixed"):
    query = generate_query(generate_database(tables, seed), seed)
    return JoinOrderEnv(query, start_mode=start_mode)


def test_baselines_only_choose_valid_actions():
    for seed in range(10):
        env = make_env(seed)
        for agent in (RandomAgent(seed), GreedyMinIntermediate(),
                      GreedyMinImmediateCost(), ExhaustiveOptimal(env)):
            env.reset()
            while not env.is_complete:
                action = agent.choose_action(env)
                assert action in env.valid_actions()
                env.step(action)


def test_exhaustive_is_no_worse_than_other_agents():
    for seed in range(30):
        env = make_env(seed)
        optimal = ExhaustiveOptimal(env).best_cost
        for agent in (RandomAgent(seed), GreedyMinIntermediate(),
                      GreedyMinImmediateCost()):
            assert run_episode(env, agent)[1] + 1e-9 >= optimal


def test_random_is_deterministic_with_seed():
    first = run_episode(make_env(7), RandomAgent(99))[1]
    second = run_episode(make_env(7), RandomAgent(99))[1]
    assert first == second


def test_exhaustive_supports_both_start_modes():
    for mode in ("fixed", "agent"):
        env = make_env(4, start_mode=mode)
        optimal = ExhaustiveOptimal(env)
        assert run_episode(env, optimal)[1] == optimal.best_cost
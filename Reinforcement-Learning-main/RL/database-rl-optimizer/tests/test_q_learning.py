import pytest

from src.database import generate_database
from src.environment import JoinOrderEnv
from src.query import generate_query
from src.baselines import ExhaustiveOptimal
from src.q_learning import QLearningAgent


def env(seed=1, tables=3):
    return JoinOrderEnv(generate_query(generate_database(tables, seed), seed),
                        start_mode="agent")


def test_hand_computed_update_and_terminal_handling():
    agent = QLearningAgent(alpha=0.5, gamma=0.9)
    error = agent.update("s", 1, 2.0, "next", [0, 2], terminal=False)
    assert agent.Q["s"][1] == pytest.approx(1.0)
    assert error == pytest.approx(2.0)
    agent.update("s", 1, 4.0, "next", [0, 2], terminal=True)
    assert agent.Q["s"][1] == pytest.approx(2.5)


def test_mask_is_respected_in_selection_and_update():
    environment = env()
    environment.reset()
    agent = QLearningAgent(epsilon=0.0)
    state = agent.state_key(environment)
    agent.Q[state][0] = 10.0
    agent.Q[state][1] = 1.0
    assert agent.choose_action(environment) in environment.valid_actions()
    agent.update("s", 0, 1.0, "next", [2], terminal=False)
    assert agent.Q["s"][0] == pytest.approx(agent.alpha * 1.0)


def test_seed_reproduces_history_and_visit_counts():
    first = QLearningAgent(episodes=30, seed=4).train(env())
    second = QLearningAgent(episodes=30, seed=4).train(env())
    assert first == second


def test_schedule_values_are_deterministic():
    linear = QLearningAgent(episodes=4, epsilon=1.0, epsilon_min=0.0,
                            epsilon_schedule="linear", seed=1)
    history = linear.train(env())
    assert [row["epsilon"] for row in history] == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0])
    exponential = QLearningAgent(episodes=3, epsilon=1.0, epsilon_decay=0.5,
                                 epsilon_schedule="exponential", seed=1)
    assert [row["epsilon"] for row in exponential.train(env())] == pytest.approx([1.0, .5, .25])


def test_constant_exploration_ratio_tracks_epsilon():
    agent = QLearningAgent(episodes=300, epsilon=0.2,
                           epsilon_schedule="constant", seed=8)
    history = agent.train(env())
    average = sum(row["exploration_ratio"] for row in history) / len(history)
    assert average == pytest.approx(0.2, abs=0.08)
    assert agent.visit_counts


def test_small_query_reaches_exhaustive_optimum():
    environment = env(seed=2)
    optimal = ExhaustiveOptimal(environment).best_cost
    agent = QLearningAgent(episodes=1000, epsilon_decay=0.98, seed=2)
    agent.train(environment)
    assert agent.greedy_plan(environment)[1] == pytest.approx(optimal)


def test_shared_state_key_is_hashable_and_deterministic():
    environment = env(seed=3)
    environment.reset()
    agent = QLearningAgent(shared=True)
    key = agent.state_key(environment)
    assert key == agent.state_key(environment)
    assert hash(key)
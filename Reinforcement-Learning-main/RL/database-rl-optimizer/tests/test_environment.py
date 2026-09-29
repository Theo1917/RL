"""
Unit tests for Milestone 5: the join-order Gymnasium environment.

Hand-built schemas make expected costs/cardinalities computable by hand.
Action ids follow sorted(query.tables), so for the fixtures below:

  chain  : A=0, B=1, C=2        (A - B - C)
  star   : H=0, X=1, Y=2, Z=3   (hub H, spokes X, Y, Z)
  cyclic : A=0, B=1, C=2, D=3   (triangle A-B-C, tail D on C)
"""

import copy
import math
from typing import Dict, List, Tuple

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.cost_model import CostModel, JoinEstimate
from src.database import Column, Database, ForeignKey, Table, generate_database
from src.environment import (
    JoinOrderEnv, STATUS_COMPLETED, STATUS_INVALID_ACTION,
    STATUS_ONGOING, STATUS_TRUNCATED,
)
from src.query import Query, generate_query


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------

def make_query(rows: Dict[str, int],
               edges: List[Tuple[str, str, float]]) -> Query:
    """Build a Query from row counts and (child, parent, selectivity) edges."""
    tables, sels = [], {}
    for name, count in rows.items():
        pk = f"{name.lower()}_id"
        cols, fks = [Column(pk)], []
        for child, parent, _ in edges:
            if child == name:
                fk_col = f"{parent.lower()}_id"
                fks.append(ForeignKey(fk_col, parent, fk_col))
                cols.append(Column(fk_col))
        tables.append(Table(name, cols, pk, count, fks))
    for child, parent, sel in edges:
        sels[tuple(sorted((child, parent)))] = sel
    return generate_query(Database(tables, sels, seed=0), seed=0)


@pytest.fixture
def chain_query() -> Query:
    return make_query({"A": 1000, "B": 2000, "C": 500},
                      [("B", "A", 0.01), ("C", "B", 0.002)])


@pytest.fixture
def star_query() -> Query:
    return make_query({"H": 10000, "X": 1000, "Y": 5000, "Z": 2000},
                      [("X", "H", 0.001), ("Y", "H", 0.0001), ("Z", "H", 0.01)])


@pytest.fixture
def cyclic_query() -> Query:
    return make_query({"A": 1000, "B": 1000, "C": 1000, "D": 1000},
                      [("B", "A", 0.01), ("C", "A", 0.02),
                       ("C", "B", 0.05), ("D", "C", 0.1)])


@pytest.fixture
def chain_env(chain_query) -> JoinOrderEnv:
    env = JoinOrderEnv(chain_query, start_table="A")
    env.reset()
    return env


@pytest.fixture
def star_env(star_query) -> JoinOrderEnv:
    env = JoinOrderEnv(star_query, start_table="H")
    env.reset()
    return env


def run_actions(env: JoinOrderEnv, actions: List[int]):
    """Reset and apply actions; return list of (obs, reward, term, trunc, info)."""
    env.reset()
    return [env.step(a) for a in actions]


def obs_equal(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


# ----------------------------------------------------------------------
# 1. Construction
# ----------------------------------------------------------------------

def test_construction_spaces_and_mapping(chain_query):
    env = JoinOrderEnv(chain_query)
    assert env.action_space.n == 3
    assert env.action_table_names == ("A", "B", "C")
    assert env.table_to_action("B") == 1 and env.action_to_table(2) == "C"
    assert env.cost_model.query is chain_query


def test_action_ids_independent_of_query_table_order():
    """Action ids come from sorted names, not from query.tables order."""
    q = make_query({"C": 500, "A": 1000, "B": 2000},
                   [("B", "A", 0.01), ("C", "B", 0.002)])
    assert q.tables[0] == "C"
    env = JoinOrderEnv(q)
    assert env.action_table_names == ("A", "B", "C")
    assert env.reset()[1]["start_table"] == "C"      # default start = query.tables[0]


def test_construction_rejects_bad_arguments(chain_query, star_query):
    with pytest.raises(ValueError):
        JoinOrderEnv(chain_query, start_table="Nope")
    with pytest.raises(ValueError):
        JoinOrderEnv(chain_query, cost_model=CostModel(star_query))
    with pytest.raises(ValueError):
        JoinOrderEnv(chain_query, invalid_action_penalty=-1)


def test_use_before_reset_raises(chain_query):
    env = JoinOrderEnv(chain_query)
    with pytest.raises(RuntimeError):
        env.step(1)
    with pytest.raises(RuntimeError):
        env.valid_action_mask()


def test_passes_gymnasium_env_checker(chain_query):
    check_env(JoinOrderEnv(chain_query, start_table="A"), skip_render_check=True)


# ----------------------------------------------------------------------
# 2-4. reset(): observation, required relations, initial mask
# ----------------------------------------------------------------------

def test_reset_returns_valid_observation(chain_env):
    obs, info = chain_env.reset()
    assert chain_env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_initial_observation_values(chain_env):
    obs, _ = chain_env.reset()
    assert obs["joined"].tolist() == [1, 0, 0]
    # Only B is a valid next join from {A}: selectivity 0.01, others 0.
    assert obs["candidate_selectivity"].tolist() == pytest.approx([0.0, 0.01, 0.0])
    assert obs["log_cardinality"][0] == pytest.approx(math.log10(1000))
    assert obs["log_accumulated_cost"][0] == 0.0


def test_initial_state_contains_all_relations(star_env):
    _, info = star_env.reset()
    assert info["joined_tables"] == ["H"]
    assert info["remaining_tables"] == ["X", "Y", "Z"]
    assert set(info["joined_tables"]) | set(info["remaining_tables"]) == set(star_env.query.tables)
    assert info["partial_plan"] == []
    assert info["accumulated_cost"] == 0.0 and info["estimated_cardinality"] == 10000
    assert info["status"] == STATUS_ONGOING and not info["is_complete"]


def test_initial_mask_chain(chain_query):
    env = JoinOrderEnv(chain_query)
    for start, expected in (("A", [0, 1, 0]), ("B", [1, 0, 1]), ("C", [0, 1, 0])):
        _, info = env.reset(options={"start_table": start})
        assert env.valid_action_mask().tolist() == expected
        assert info["action_mask"].tolist() == expected
        assert env.valid_actions() == [i for i, v in enumerate(expected) if v]


def test_initial_mask_star(star_env):
    assert star_env.valid_action_mask().tolist() == [0, 1, 1, 1]
    assert star_env.valid_action_mask().dtype == np.int8


def test_reset_start_table_option_validated(chain_env):
    with pytest.raises(ValueError):
        chain_env.reset(options={"start_table": "Nope"})


# ----------------------------------------------------------------------
# 5-8. Valid transitions, cost model used, cardinality and cost updates
# ----------------------------------------------------------------------

def test_valid_join_expected_next_state(chain_env):
    obs, reward, term, trunc, info = chain_env.step(1)        # join B
    assert term is False and trunc is False
    assert obs["joined"].tolist() == [1, 1, 0]
    assert info["joined_tables"] == ["A", "B"]
    assert info["remaining_tables"] == ["C"]
    assert info["selected_action"] == 1 and info["selected_table"] == "B"
    assert info["status"] == STATUS_ONGOING
    assert chain_env.state_key() == frozenset({"A", "B"})
    assert chain_env.observation_space.contains(obs)


def test_reward_is_negative_incremental_cost_by_hand(chain_env):
    # A(1000) JOIN B(2000), sel 0.01: out 20,000; cost 1000+2000+20000.
    _, reward, *_ = chain_env.step(1)
    assert reward == pytest.approx(-23000.0)
    # {A,B}(20,000) JOIN C(500), sel 0.002: out 20,000; cost 20000+500+20000.
    _, reward2, *_ = chain_env.step(2)
    assert reward2 == pytest.approx(-40500.0)


def test_cost_model_is_actually_used(chain_query):
    """Rewards come from CostModel.estimate_plan_join: a scaled cost model
    scales the rewards, and the spy sees the exact calls."""
    calls = []

    class DoubledCostModel(CostModel):
        def estimate_plan_join(self, plan, candidate):
            calls.append((frozenset(plan.joined_tables), candidate))
            est = super().estimate_plan_join(plan, candidate)
            return JoinEstimate(**{**est.__dict__, "cost": est.cost * 2})

    env = JoinOrderEnv(chain_query, cost_model=DoubledCostModel(chain_query),
                       start_table="A")
    env.reset()
    calls.clear()
    _, reward, *_ = env.step(1)
    assert reward == pytest.approx(-2 * 23000.0)
    assert (frozenset({"A"}), "B") in calls


def test_cardinality_updated_correctly(chain_env):
    _, _, _, _, info = chain_env.step(1)
    assert info["estimated_cardinality"] == pytest.approx(20000)
    assert info["left_cardinality"] == pytest.approx(1000)
    obs = chain_env.step(2)[0]
    assert obs["log_cardinality"][0] == pytest.approx(math.log10(20000))
    assert chain_env.cost_model.current_cardinality(chain_env.plan) == pytest.approx(20000)


def test_accumulated_cost_updated_correctly(chain_env):
    _, r1, _, _, i1 = chain_env.step(1)
    assert i1["incremental_cost"] == pytest.approx(23000)
    assert i1["accumulated_cost"] == pytest.approx(23000)
    obs, r2, _, _, i2 = chain_env.step(2)
    assert i2["incremental_cost"] == pytest.approx(40500)
    assert i2["accumulated_cost"] == pytest.approx(63500)
    assert chain_env.accumulated_cost == pytest.approx(63500)
    assert obs["log_accumulated_cost"][0] == pytest.approx(math.log10(1 + 63500), rel=1e-6)
    assert -(r1 + r2) == pytest.approx(chain_env.cost_model.plan_cost(chain_env.plan))


def test_cyclic_join_uses_all_predicates(cyclic_query):
    """C joins {A,B} through two predicates; the env reports both."""
    env = JoinOrderEnv(cyclic_query, start_table="A")
    env.reset()
    env.step(1)                                       # B
    _, reward, _, _, info = env.step(2)               # C via C-A and C-B
    assert info["applied_joins"] == (("A", "C"), ("B", "C"))
    out = 1000 * 1000 * 0.01 * 1000 * 0.02 * 0.05
    assert info["estimated_cardinality"] == pytest.approx(max(out, 1.0))
    assert reward == pytest.approx(-(10000 + 1000 + max(out, 1.0)))


# ----------------------------------------------------------------------
# 9-10. Invalid actions
# ----------------------------------------------------------------------

@pytest.mark.parametrize("bad, fragment", [
    (0, "already joined"),          # start table
    (2, "no join predicate"),       # C not adjacent to {A}
    (3, "out of range"),
    (-1, "out of range"),
    (1.0, "integer"),
    ("B", "integer"),
    (None, "integer"),
    (True, "integer"),
])
def test_invalid_actions_detected(chain_env, bad, fragment):
    obs, reward, term, trunc, info = chain_env.step(bad)
    assert info["invalid_action"] is True
    assert info["status"] == STATUS_INVALID_ACTION
    assert fragment in info["invalid_reason"]
    assert term is False and trunc is False
    assert reward == 0.0


def test_invalid_action_does_not_corrupt_state(chain_env):
    chain_env.step(1)                                              # valid: join B
    before_key = chain_env.state_key()
    before_plan = copy.deepcopy(chain_env.plan.join_sequence)
    before_cost = chain_env.accumulated_cost
    before_mask = chain_env.valid_action_mask().copy()
    before_obs = chain_env._observation()

    for bad in (0, 1, 7, -3, 0.5, "x"):                            # all illegal now
        chain_env.step(bad)

    assert chain_env.state_key() == before_key
    assert chain_env.plan.join_sequence == before_plan
    assert chain_env.accumulated_cost == before_cost
    assert np.array_equal(chain_env.valid_action_mask(), before_mask)
    assert obs_equal(chain_env._observation(), before_obs)
    # ...and the episode still completes normally with the right cost.
    _, _, term, _, _ = chain_env.step(2)
    assert term and chain_env.accumulated_cost == pytest.approx(63500)


def test_invalid_action_penalty_and_counter(chain_query):
    env = JoinOrderEnv(chain_query, start_table="A", invalid_action_penalty=5.0)
    env.reset()
    _, reward, _, _, info = env.step(2)
    assert reward == -5.0 and info["invalid_actions_taken"] == 1
    assert info["accumulated_cost"] == 0.0            # penalty is not plan cost


def test_too_many_invalid_actions_truncates(chain_query):
    env = JoinOrderEnv(chain_query, start_table="A", max_invalid_actions=2)
    env.reset()
    assert env.step(2)[3] is False and env.step(2)[3] is False
    _, _, term, trunc, info = env.step(2)
    assert trunc is True and term is False and info["status"] == STATUS_TRUNCATED
    with pytest.raises(RuntimeError):
        env.step(1)
    env.reset()                                       # reset recovers cleanly
    assert env.step(1)[4]["status"] == STATUS_ONGOING


# ----------------------------------------------------------------------
# 11-12. Termination and number of joins
# ----------------------------------------------------------------------

def test_reaches_terminal_state_after_complete_plan(chain_env):
    _, _, t1, _, i1 = chain_env.step(1)
    assert not t1 and not i1["is_complete"]
    obs, _, t2, tr2, i2 = chain_env.step(2)
    assert t2 is True and tr2 is False
    assert i2["status"] == STATUS_COMPLETED and i2["is_complete"]
    assert i2["remaining_tables"] == [] and i2["valid_actions"] == []
    assert obs["joined"].tolist() == [1, 1, 1]
    assert chain_env.is_complete


def test_step_after_termination_raises_and_state_kept(chain_env):
    chain_env.step(1)
    chain_env.step(2)
    cost = chain_env.accumulated_cost
    with pytest.raises(RuntimeError):
        chain_env.step(1)
    assert chain_env.accumulated_cost == cost


@pytest.mark.parametrize("num_tables", [3, 4, 5, 6])
@pytest.mark.parametrize("seed", [0, 1, 42, 123])
def test_number_of_joins_is_tables_minus_one(num_tables, seed):
    q = generate_query(generate_database(num_tables, seed), seed)
    env = JoinOrderEnv(q)
    _, info = env.reset()
    steps, terminated = 0, False
    while not terminated:
        assert info["valid_actions"], "connected query never dead-ends"
        _, _, terminated, truncated, info = env.step(info["valid_actions"][0])
        steps += 1
        assert not truncated and steps <= num_tables
    assert steps == num_tables - 1 == len(env.plan.join_sequence)


# ----------------------------------------------------------------------
# 13. reset() restores the initial state
# ----------------------------------------------------------------------

def test_reset_restores_initial_state(star_env):
    obs0, info0 = star_env.reset()
    star_env.step(2)
    star_env.step(1)
    star_env.step(9)                                   # invalid too
    obs1, info1 = star_env.reset()

    assert obs_equal(obs0, obs1)
    for key in ("status", "joined_tables", "remaining_tables", "partial_plan",
                "accumulated_cost", "valid_actions", "steps_taken",
                "invalid_actions_taken", "estimated_cardinality", "start_table"):
        assert info0[key] == info1[key], key
    assert star_env.accumulated_cost == 0.0
    assert star_env.state_key() == frozenset({"H"})


def test_reset_after_terminal_allows_new_episode(chain_env):
    chain_env.step(1)
    chain_env.step(2)
    chain_env.reset()
    assert chain_env.step(1)[1] == pytest.approx(-23000.0)


# ----------------------------------------------------------------------
# 14. Determinism
# ----------------------------------------------------------------------

def test_same_action_sequence_same_results(star_query):
    a = JoinOrderEnv(star_query, start_table="H")
    b = JoinOrderEnv(star_query, start_table="H")
    ra = run_actions(a, [3, 1, 2])
    rb = run_actions(b, [3, 1, 2])
    for (oa, rwa, ta, tra, ia), (ob, rwb, tb, trb, ib) in zip(ra, rb):
        assert obs_equal(oa, ob) and rwa == rwb and ta == tb and tra == trb
        assert ia["accumulated_cost"] == ib["accumulated_cost"]
        assert ia["partial_plan"] == ib["partial_plan"]
    # Repeating on the SAME env instance is identical as well.
    rc = run_actions(a, [3, 1, 2])
    assert [r[1] for r in rc] == [r[1] for r in ra]


def test_determinism_with_regenerated_database():
    def episode():
        q = generate_query(generate_database(5, seed=7), seed=7)
        env = JoinOrderEnv(q)
        _, info = env.reset(seed=123)
        out = []
        while True:
            obs, r, term, _, info = env.step(info["valid_actions"][-1])
            out.append((r, term, obs["joined"].tolist(), info["estimated_cardinality"]))
            if term:
                return out
    assert episode() == episode()


def test_seed_argument_does_not_change_behaviour(star_query):
    env = JoinOrderEnv(star_query, start_table="H")
    env.reset(seed=1)
    r1 = env.step(1)[1]
    env.reset(seed=999)
    assert env.step(1)[1] == r1


# ----------------------------------------------------------------------
# 15. Different join orders -> different total cost
# ----------------------------------------------------------------------

def test_different_orders_give_different_total_cost(star_env):
    """Star: joining the most selective spoke (Y) first is cheaper."""
    def total(order):
        star_env.reset()
        return sum(-star_env.step(a)[1] for a in order)

    cheap = total([2, 1, 3])       # Y, X, Z
    dear = total([3, 1, 2])        # Z, X, Y
    assert cheap < dear
    # Hand-check both totals.
    # Y first: 10000+5000+5000 = 20000 (card 5000)
    # then X : 5000+1000+5000  = 11000 (card 5000*1000*0.001)
    # then Z : 5000+2000+card(all)
    final = 10000 * 1000 * 5000 * 2000 * 0.001 * 0.0001 * 0.01
    assert cheap == pytest.approx(20000 + 11000 + (5000 + 2000 + final))
    # Z first: 10000+2000+200000 = 212000 (card 200000)
    # then X : 200000+1000+card(H,Z,X) ; card = 10000*2000*1000*0.01*0.001 = 200000
    # then Y : 200000+5000+final
    assert dear == pytest.approx(212000 + (200000 + 1000 + 200000) + (200000 + 5000 + final))


def test_all_orders_end_in_same_final_cardinality(star_env):
    finals = set()
    for order in ([1, 2, 3], [3, 2, 1], [2, 3, 1]):
        star_env.reset()
        for a in order:
            info = star_env.step(a)[4]
        finals.add(round(info["estimated_cardinality"], 6))
    assert len(finals) == 1


# ----------------------------------------------------------------------
# 16. Mask correctness after each transition
# ----------------------------------------------------------------------

def test_mask_after_each_transition_chain(chain_query):
    env = JoinOrderEnv(chain_query, start_table="B")
    env.reset()
    assert env.valid_action_mask().tolist() == [1, 0, 1]
    _, _, _, _, info = env.step(0)                       # join A
    assert info["action_mask"].tolist() == [0, 0, 1]
    assert env.valid_action_mask().tolist() == [0, 0, 1]
    _, _, _, _, info = env.step(2)                       # join C -> done
    assert info["action_mask"].tolist() == [0, 0, 0]


def test_mask_after_each_transition_star(star_env):
    assert star_env.valid_action_mask().tolist() == [0, 1, 1, 1]
    star_env.step(2)
    assert star_env.valid_action_mask().tolist() == [0, 1, 0, 1]
    star_env.step(3)
    assert star_env.valid_action_mask().tolist() == [0, 1, 0, 0]
    star_env.step(1)
    assert star_env.valid_action_mask().tolist() == [0, 0, 0, 0]


def test_mask_matches_query_graph_on_cyclic_schema(cyclic_query):
    """Mask == tables adjacent to the joined set and not yet joined."""
    env = JoinOrderEnv(cyclic_query, start_table="A")
    env.reset()
    assert env.valid_action_mask().tolist() == [0, 1, 1, 0]        # B, C (D not adjacent)
    env.step(2)                                                    # join C
    assert env.valid_action_mask().tolist() == [0, 1, 0, 1]        # B via A/C, D via C


def test_mask_is_exactly_the_set_of_accepted_actions(star_query):
    """At every non-terminal state along a path, each masked-in action is
    accepted and each masked-out action is rejected; at the terminal state
    the mask is empty."""
    for path in ([1, 2, 3], [3, 2, 1], [2, 1, 3]):
        for depth in range(len(path) + 1):
            probe_mask = None
            for a in range(4):
                env = JoinOrderEnv(star_query, start_table="H")
                env.reset()
                for prev in path[:depth]:
                    env.step(prev)
                mask = env.valid_action_mask()
                if depth == len(path):                  # terminal: nothing legal
                    assert mask.tolist() == [0, 0, 0, 0]
                    assert env.is_complete
                    break
                info = env.step(a)[4]
                assert info["invalid_action"] == (mask[a] == 0), (path, depth, a)


def test_mask_never_contains_joined_tables(star_env):
    star_env.step(1)
    joined = star_env._observation()["joined"]
    assert not np.any(joined & star_env.valid_action_mask())


# ----------------------------------------------------------------------
# 17. info dictionary
# ----------------------------------------------------------------------

def test_info_contents_after_valid_step(star_env):
    _, reward, _, _, info = star_env.step(2)                       # join Y
    assert info["selected_action"] == 2 and info["selected_table"] == "Y"
    assert info["selected_join"] == ("H", "Y")
    assert info["applied_joins"] == (("H", "Y"),)
    assert info["incremental_cost"] == pytest.approx(-reward) == pytest.approx(20000)
    assert info["estimated_cardinality"] == pytest.approx(5000)
    assert info["left_cardinality"] == pytest.approx(10000)
    assert info["accumulated_cost"] == pytest.approx(20000)
    assert info["partial_plan"] == [("H", "Y")]
    assert info["joined_tables"] == ["H", "Y"]
    assert info["remaining_tables"] == ["X", "Z"]
    assert info["valid_actions"] == [1, 3]
    assert info["is_complete"] is False and info["invalid_action"] is False
    assert info["steps_taken"] == 1
    assert "explanation" in info and "Y" in info["explanation"]


def test_info_partial_plan_is_a_copy(star_env):
    _, _, _, _, info = star_env.step(2)
    info["partial_plan"].append(("junk", "junk"))
    info["remaining_tables"].clear()
    assert star_env.plan.join_sequence == [("H", "Y")]
    assert sorted(star_env.plan.remaining_tables) == ["X", "Z"]


def test_info_on_invalid_action_reports_reason_and_unchanged_totals(star_env):
    star_env.step(1)
    _, _, _, _, info = star_env.step(0)
    assert info["invalid_action"] and info["selected_action"] == 0
    assert info["selected_table"] is None and info["selected_join"] is None
    assert info["incremental_cost"] == 0.0
    assert info["accumulated_cost"] == pytest.approx(star_env.accumulated_cost)
    assert info["joined_tables"] == ["H", "X"]


def test_info_does_not_leak_future_information(star_env):
    """Debug info only describes the present state and the transition just
    taken -- no lookahead costs for unchosen actions or the optimal plan."""
    _, _, _, _, info = star_env.step(2)
    forbidden = {"optimal_cost", "best_action", "candidate_costs", "oracle"}
    assert not (forbidden & info.keys())


# ----------------------------------------------------------------------
# Observation / state-key properties
# ----------------------------------------------------------------------

def test_observation_always_in_space_over_random_episodes():
    for seed in range(6):
        q = generate_query(generate_database(3 + seed % 4, seed), seed)
        env = JoinOrderEnv(q)
        rng = np.random.default_rng(seed)
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        term = False
        while not term:
            a = int(rng.choice(info["valid_actions"]))
            obs, r, term, trunc, info = env.step(a)
            assert env.observation_space.contains(obs)
            assert r < 0 and math.isfinite(r)


def test_state_key_depends_only_on_joined_set(star_query):
    env = JoinOrderEnv(star_query, start_table="H")
    env.reset()
    env.step(1); env.step(2)
    k1, o1 = env.state_key(), env._observation()
    env.reset()
    env.step(2); env.step(1)
    k2, o2 = env.state_key(), env._observation()
    assert k1 == k2 == frozenset({"H", "X", "Y"})
    assert o1["joined"].tolist() == o2["joined"].tolist()
    assert o1["log_cardinality"][0] == o2["log_cardinality"][0]


def test_candidate_selectivity_matches_cost_model(cyclic_query):
    env = JoinOrderEnv(cyclic_query, start_table="A")
    env.reset()
    env.step(1)                                                   # {A,B}
    obs = env._observation()
    # C is reachable via two predicates: 0.02 * 0.05
    assert obs["candidate_selectivity"].tolist() == pytest.approx([0, 0, 0.001, 0])


def test_env_does_not_modify_database_or_query(star_query):
    db_before, joins_before = repr(star_query.database), repr(star_query.joins)
    env = JoinOrderEnv(star_query, start_table="H")
    env.reset()
    for a in (1, 9, 2, 3):
        env.step(a)
    assert repr(star_query.database) == db_before
    assert repr(star_query.joins) == joins_before


# ----------------------------------------------------------------------
# End-to-end consistency with the cost model
# ----------------------------------------------------------------------

@pytest.mark.parametrize("num_tables", [3, 4, 5, 6])
@pytest.mark.parametrize("seed", [0, 42])
def test_episode_return_equals_negative_plan_cost(num_tables, seed):
    q = generate_query(generate_database(num_tables, seed), seed)
    env = JoinOrderEnv(q)
    _, info = env.reset()
    total, rng = 0.0, np.random.default_rng(seed)
    term = False
    while not term:
        _, r, term, _, info = env.step(int(rng.choice(info["valid_actions"])))
        total += r
    assert total == pytest.approx(-env.cost_model.plan_cost(env.plan))
    assert total == pytest.approx(-info["accumulated_cost"])

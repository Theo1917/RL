"""
Unit tests for Milestone 4: cardinality estimation and cost model.

Most tests use small hand-built schemas so expected values can be computed
by hand; a few run across generated databases to check invariants.
"""

import copy
import math
from typing import Dict, List, Tuple

import pytest

from src.cost_model import (
    CostModel, InvalidJoinError, JoinEstimate,
    compute_join_cost, estimate_join_cardinality,
)
from src.database import (
    Column, Database, ForeignKey, Table, generate_database,
)
from src.query import Query, create_initial_plan, generate_query


# ----------------------------------------------------------------------
# Helpers: hand-built databases/queries with exactly known statistics
# ----------------------------------------------------------------------

def make_query(rows: Dict[str, int],
               edges: List[Tuple[str, str, float]]) -> Query:
    """
    Build a Query from row counts and (child, parent, selectivity) edges.

    Mirrors what generate_database/generate_query produce: the child holds a
    foreign key column referencing the parent's primary key.
    """
    tables = []
    selectivities = {}
    for name, count in rows.items():
        pk = f"{name.lower()}_id"
        columns = [Column(pk)]
        fks = []
        for child, parent, _ in edges:
            if child == name:
                fk_col = f"{parent.lower()}_id"
                fks.append(ForeignKey(fk_col, parent, fk_col))
                if fk_col != pk:
                    columns.append(Column(fk_col))
        tables.append(Table(name, columns, pk, count, fks))
    for child, parent, sel in edges:
        selectivities[tuple(sorted((child, parent)))] = sel
    db = Database(tables=tables, join_selectivities=selectivities, seed=0)
    return generate_query(db, seed=0)


@pytest.fixture
def chain_query() -> Query:
    """A - B - C chain: B references A, C references B."""
    return make_query(
        {"A": 1000, "B": 2000, "C": 500},
        [("B", "A", 0.01), ("C", "B", 0.002)],
    )


@pytest.fixture
def star_query() -> Query:
    """Hub H with three spokes X, Y, Z (each references H)."""
    return make_query(
        {"H": 10000, "X": 1000, "Y": 5000, "Z": 2000},
        [("X", "H", 0.001), ("Y", "H", 0.0001), ("Z", "H", 0.01)],
    )


@pytest.fixture
def cyclic_query() -> Query:
    """Triangle A-B-C plus tail D: C joins both A and B."""
    return make_query(
        {"A": 1000, "B": 1000, "C": 1000, "D": 1000},
        [("B", "A", 0.01), ("C", "A", 0.02), ("C", "B", 0.05), ("D", "C", 0.1)],
    )


# ----------------------------------------------------------------------
# 1. Basic cardinality calculation
# ----------------------------------------------------------------------

def test_pure_cardinality_formula():
    """|R JOIN S| = |R| * |S| * selectivity."""
    assert estimate_join_cardinality(1000, 2000, 0.01) == pytest.approx(20000.0)


def test_readme_example_cardinality():
    """README example: 10,000 x 100,000 x 0.01 = 10,000,000 rows."""
    assert estimate_join_cardinality(10_000, 100_000, 0.01) == pytest.approx(10_000_000)


def test_model_two_table_cardinality(chain_query):
    """CostModel uses the query's stored selectivity for a direct join."""
    est = CostModel(chain_query).estimate_table_join("A", "B")
    assert est.output_cardinality == pytest.approx(1000 * 2000 * 0.01)
    assert est.selectivity == pytest.approx(0.01)
    assert est.left_cardinality == 1000
    assert est.right_cardinality == 2000


def test_join_selectivity_is_symmetric(chain_query):
    """Selectivity lookup ignores argument order."""
    model = CostModel(chain_query)
    assert model.join_selectivity("A", "B") == model.join_selectivity("B", "A") == 0.01


def test_cardinality_symmetric_in_inputs():
    """Swapping the two inputs does not change the output size."""
    assert estimate_join_cardinality(300, 7000, 0.003) == \
        estimate_join_cardinality(7000, 300, 0.003)


# ----------------------------------------------------------------------
# 2. Zero / very small selectivity
# ----------------------------------------------------------------------

def test_zero_selectivity_is_floored_to_one_row():
    """Selectivity 0 gives 0 rows raw, floored to 1 by default."""
    assert estimate_join_cardinality(1000, 1000, 0.0) == 1.0


def test_zero_selectivity_floor_can_be_disabled():
    """min_cardinality=0 exposes the raw 0-row estimate."""
    assert estimate_join_cardinality(1000, 1000, 0.0, min_cardinality=0.0) == 0.0


def test_very_small_selectivity_is_floored():
    """1000 x 1000 x 1e-9 = 0.001 rows -> floored to 1."""
    assert estimate_join_cardinality(1000, 1000, 1e-9) == 1.0


def test_very_small_selectivity_above_floor_is_exact():
    """The minimum generated selectivity (1e-4) on 1e6 x 1e6 stays exact."""
    assert estimate_join_cardinality(10**6, 10**6, 1e-4) == pytest.approx(1e8)


def test_custom_floor_applied():
    """A custom floor is honoured."""
    assert estimate_join_cardinality(10, 10, 0.001, min_cardinality=5.0) == 5.0


# ----------------------------------------------------------------------
# 3. Larger selectivity
# ----------------------------------------------------------------------

def test_selectivity_one_is_cartesian_product():
    """Selectivity 1.0 yields the full Cartesian product."""
    assert estimate_join_cardinality(1000, 2000, 1.0) == pytest.approx(2_000_000)


def test_larger_selectivity_gives_larger_output():
    """Output grows monotonically with selectivity."""
    sels = [0.0001, 0.001, 0.01, 0.1, 1.0]
    outs = [estimate_join_cardinality(5000, 8000, s) for s in sels]
    assert outs == sorted(outs) and len(set(outs)) == len(outs)


def test_model_large_selectivity_join():
    """A 10% selectivity join through the model."""
    q = make_query({"P": 4000, "Q": 5000, "R": 1000},
                   [("Q", "P", 0.1), ("R", "Q", 0.5)])
    est = CostModel(q).estimate_table_join("P", "Q")
    assert est.output_cardinality == pytest.approx(4000 * 5000 * 0.1)


def test_pure_formula_rejects_bad_inputs():
    """Negative sizes / out-of-range selectivity / negative floor raise."""
    with pytest.raises(ValueError):
        estimate_join_cardinality(-1, 10, 0.5)
    with pytest.raises(ValueError):
        estimate_join_cardinality(10, -1, 0.5)
    with pytest.raises(ValueError):
        estimate_join_cardinality(10, 10, -0.1)
    with pytest.raises(ValueError):
        estimate_join_cardinality(10, 10, 1.5)
    with pytest.raises(ValueError):
        estimate_join_cardinality(10, 10, 0.5, min_cardinality=-1)


# ----------------------------------------------------------------------
# 4. Determinism
# ----------------------------------------------------------------------

def test_repeated_calculation_identical(chain_query):
    """Same call repeated gives bit-identical results (incl. cache hits)."""
    model = CostModel(chain_query)
    first = model.estimate_join(["A", "B"], "C")
    for _ in range(5):
        assert model.estimate_join(["A", "B"], "C") == first


def test_fresh_models_agree(chain_query):
    """Independent CostModel instances give identical estimates."""
    a = CostModel(chain_query).estimate_join(["A"], "B")
    b = CostModel(chain_query).estimate_join(["A"], "B")
    assert a == b


def test_same_seed_same_estimates():
    """Databases regenerated from the same seed cost identically."""
    def run():
        q = generate_query(generate_database(5, seed=7), seed=7)
        m = CostModel(q)
        plan = create_initial_plan(q)
        out = []
        while not plan.is_complete():
            ests = m.estimate_available_joins(plan)
            out.append([(t, e.cost, e.output_cardinality) for t, e in ests.items()])
            plan = plan.apply_join(next(iter(ests)))
        return out
    assert run() == run()


def test_set_cardinality_order_independent(star_query):
    """Cardinality depends on the table set, not the order it is given."""
    m = CostModel(star_query)
    assert m.set_cardinality(["H", "X", "Y"]) == m.set_cardinality(["Y", "X", "H"])
    assert m.set_cardinality(["H", "X", "Y"]) == m.set_cardinality(frozenset("HXY"))


def test_join_order_does_not_change_final_cardinality(star_query):
    """Any join order over the same tables ends at the same result size."""
    m = CostModel(star_query)
    finals = set()
    for order in (["X", "Y", "Z"], ["Z", "Y", "X"], ["Y", "X", "Z"]):
        joined = {"H"}
        for t in order:
            est = m.estimate_join(joined, t)
            joined.add(t)
        finals.add(est.output_cardinality)
    assert len(finals) == 1


# ----------------------------------------------------------------------
# 5. Valid join cost
# ----------------------------------------------------------------------

def test_cost_formula():
    """join_cost = left + right + output."""
    assert compute_join_cost(100, 200, 300) == 600.0


def test_valid_join_cost_value(chain_query):
    """Hand-computed: A(1000) JOIN B(2000) sel 0.01 -> out 20,000; cost 23,000."""
    est = CostModel(chain_query).estimate_join(["A"], "B")
    assert est.output_cardinality == pytest.approx(20000)
    assert est.cost == pytest.approx(1000 + 2000 + 20000)


def test_cost_positive_and_consistent(chain_query):
    """Cost always equals the sum of its three components."""
    est = CostModel(chain_query).estimate_join(["A", "B"], "C")
    assert est.cost > 0
    assert est.cost == pytest.approx(
        est.left_cardinality + est.right_cardinality + est.output_cardinality)


def test_second_join_uses_intermediate_size(chain_query):
    """A JOIN B (20,000 rows) JOIN C(500) sel 0.002 -> out 20,000; cost 40,500."""
    est = CostModel(chain_query).estimate_join(["A", "B"], "C")
    assert est.left_cardinality == pytest.approx(20000)
    assert est.output_cardinality == pytest.approx(20000 * 500 * 0.002)
    assert est.cost == pytest.approx(20000 + 500 + 20000)


def test_cost_pure_function_rejects_negative():
    """Negative sizes are rejected."""
    with pytest.raises(ValueError):
        compute_join_cost(-1, 1, 1)


def test_join_estimate_explain_and_result_tables(chain_query):
    """explain() is human readable; result_tables includes the new table."""
    est = CostModel(chain_query).estimate_join(["A"], "B")
    assert est.result_tables == frozenset({"A", "B"})
    text = est.explain()
    assert "B" in text and "cost" in text


# ----------------------------------------------------------------------
# 6. Invalid join handling
# ----------------------------------------------------------------------

def test_non_adjacent_tables_rejected(chain_query):
    """A and C have no direct foreign key."""
    m = CostModel(chain_query)
    with pytest.raises(InvalidJoinError):
        m.estimate_join(["A"], "C")
    with pytest.raises(InvalidJoinError):
        m.join_selectivity("A", "C")


def test_unknown_table_rejected(chain_query):
    """Unknown names raise InvalidJoinError."""
    m = CostModel(chain_query)
    with pytest.raises(InvalidJoinError):
        m.estimate_join(["A"], "Nope")
    with pytest.raises(InvalidJoinError):
        m.estimate_join(["Nope"], "A")
    with pytest.raises(InvalidJoinError):
        m.table_cardinality("Nope")


def test_self_join_rejected(chain_query):
    """A table cannot be joined with itself."""
    m = CostModel(chain_query)
    with pytest.raises(InvalidJoinError):
        m.join_selectivity("A", "A")
    with pytest.raises(InvalidJoinError):
        m.estimate_join(["A"], "A")


def test_already_joined_table_rejected(chain_query):
    """Re-joining a table already in the intermediate is invalid."""
    with pytest.raises(InvalidJoinError):
        CostModel(chain_query).estimate_join(["A", "B"], "B")


def test_empty_left_rejected(chain_query):
    """Empty left side is invalid."""
    with pytest.raises(InvalidJoinError):
        CostModel(chain_query).estimate_join([], "A")
    with pytest.raises(InvalidJoinError):
        CostModel(chain_query).set_cardinality([])


def test_disconnected_left_set_rejected(chain_query):
    """{A, C} is not a connected intermediate relation."""
    with pytest.raises(InvalidJoinError):
        CostModel(chain_query).set_cardinality(["A", "C"])


def test_invalid_join_error_is_value_error():
    """InvalidJoinError can be caught as ValueError (matches query.py style)."""
    assert issubclass(InvalidJoinError, ValueError)


def test_negative_floor_rejected(chain_query):
    """CostModel validates its floor."""
    with pytest.raises(ValueError):
        CostModel(chain_query, min_cardinality=-1)


# ----------------------------------------------------------------------
# 7. Partial-plan cost calculation
# ----------------------------------------------------------------------

def test_initial_plan_state(chain_query):
    """Start-only plan: cardinality = table size, cost 0, rest remaining."""
    m = CostModel(chain_query)
    plan = create_initial_plan(chain_query, "A")
    assert m.current_cardinality(plan) == 1000
    assert m.remaining_tables(plan) == ["B", "C"]
    assert m.plan_cost(plan) == 0.0


def test_estimate_plan_join_matches_estimate_join(chain_query):
    """Plan-level estimate equals the set-level estimate."""
    m = CostModel(chain_query)
    plan = create_initial_plan(chain_query, "A")
    assert m.estimate_plan_join(plan, "B") == m.estimate_join(["A"], "B")


def test_partial_plan_progression(chain_query):
    """Cardinality/remaining/cost evolve correctly as joins are applied."""
    m = CostModel(chain_query)
    plan = create_initial_plan(chain_query, "A")

    step1 = m.estimate_plan_join(plan, "B")
    plan = plan.apply_join("B")
    assert m.current_cardinality(plan) == pytest.approx(step1.output_cardinality)
    assert m.remaining_tables(plan) == ["C"]
    assert m.plan_cost(plan) == pytest.approx(step1.cost)

    step2 = m.estimate_plan_join(plan, "C")
    plan = plan.apply_join("C")
    assert plan.is_complete() and m.remaining_tables(plan) == []
    assert m.plan_cost(plan) == pytest.approx(step1.cost + step2.cost)
    assert m.plan_cost(plan) == pytest.approx(23000 + 40500)


def test_plan_join_rejects_unavailable_candidate(chain_query):
    """C is remaining but not yet joinable from {A}."""
    m = CostModel(chain_query)
    plan = create_initial_plan(chain_query, "A")
    with pytest.raises(InvalidJoinError):
        m.estimate_plan_join(plan, "C")


def test_plan_join_rejects_joined_or_unknown_candidate(chain_query):
    """Already-joined and unknown candidates are invalid."""
    m = CostModel(chain_query)
    plan = create_initial_plan(chain_query, "A")
    with pytest.raises(InvalidJoinError):
        m.estimate_plan_join(plan, "A")
    with pytest.raises(InvalidJoinError):
        m.estimate_plan_join(plan, "Nope")


def test_plan_from_other_query_rejected(chain_query, star_query):
    """A plan built for another query cannot be costed by this model."""
    m = CostModel(chain_query)
    foreign = create_initial_plan(star_query, "H")
    with pytest.raises(ValueError):
        m.current_cardinality(foreign)
    with pytest.raises(ValueError):
        m.estimate_plan_join(foreign, "X")


def test_plan_cost_independent_of_start_choice_on_final_size(chain_query):
    """Different start tables end at the same final cardinality."""
    m = CostModel(chain_query)
    finals = set()
    for start in ("A", "B", "C"):
        plan = create_initial_plan(chain_query, start)
        while not plan.is_complete():
            plan = plan.apply_join(plan.get_available_joins()[0])
        finals.add(m.current_cardinality(plan))
    assert len(finals) == 1


# ----------------------------------------------------------------------
# 8. Multiple possible joins
# ----------------------------------------------------------------------

def test_multiple_candidates_from_star_hub(star_query):
    """Hub start exposes three candidates with hand-computed costs."""
    m = CostModel(star_query)
    plan = create_initial_plan(star_query, "H")
    ests = m.estimate_available_joins(plan)
    assert list(ests) == ["X", "Y", "Z"]                    # sorted, = get_available_joins
    assert list(ests) == plan.get_available_joins()
    # X: 10000*1000*0.001 = 10,000 ; Y: 10000*5000*0.0001 = 5,000 ; Z: 10000*2000*0.01 = 200,000
    assert ests["X"].output_cardinality == pytest.approx(10_000)
    assert ests["Y"].output_cardinality == pytest.approx(5_000)
    assert ests["Z"].output_cardinality == pytest.approx(200_000)
    assert ests["X"].cost == pytest.approx(10000 + 1000 + 10000)
    assert ests["Y"].cost == pytest.approx(10000 + 5000 + 5000)
    assert ests["Z"].cost == pytest.approx(10000 + 2000 + 200000)


def test_candidate_costs_differ_and_are_comparable(star_query):
    """Different candidates give different costs the agent can rank."""
    ests = CostModel(star_query).estimate_available_joins(
        create_initial_plan(star_query, "H"))
    costs = [e.cost for e in ests.values()]
    assert len(set(costs)) == 3


def test_join_order_affects_total_cost(star_query):
    """Order matters: joining the highly selective spoke first is cheaper."""
    m = CostModel(star_query)

    def total(order):
        plan = create_initial_plan(star_query, "H")
        for t in order:
            plan = plan.apply_join(t)
        return m.plan_cost(plan)

    assert total(["Y", "X", "Z"]) < total(["Z", "X", "Y"])


def test_estimating_candidates_does_not_advance_plan(star_query):
    """Estimation is side-effect free with respect to the plan."""
    m = CostModel(star_query)
    plan = create_initial_plan(star_query, "H")
    before = (plan.joined_tables, plan.remaining_tables, list(plan.join_sequence))
    m.estimate_available_joins(plan)
    assert before == (plan.joined_tables, plan.remaining_tables, list(plan.join_sequence))


# ----------------------------------------------------------------------
# Cyclic schema (two predicates connect the candidate to the intermediate)
# ----------------------------------------------------------------------

def test_cyclic_join_applies_all_predicates(cyclic_query):
    """C joins {A,B} via both C-A (0.02) and C-B (0.05)."""
    m = CostModel(cyclic_query)
    est = m.estimate_join(["A", "B"], "C")
    assert est.applied_joins == (("A", "C"), ("B", "C"))
    assert est.selectivity == pytest.approx(0.02 * 0.05)
    left = 1000 * 1000 * 0.01                     # |A JOIN B| = 10,000
    assert est.left_cardinality == pytest.approx(left)
    assert est.output_cardinality == pytest.approx(left * 1000 * 0.02 * 0.05)


def test_cyclic_final_cardinality_order_independent(cyclic_query):
    """Every valid join order over the cyclic schema agrees on the final size."""
    m = CostModel(cyclic_query)
    finals = set()
    for start in ("A", "B", "C", "D"):
        plan = create_initial_plan(cyclic_query, start)
        while not plan.is_complete():
            plan = plan.apply_join(plan.get_available_joins()[-1])
        finals.add(round(m.current_cardinality(plan), 6))
    assert len(finals) == 1


# ----------------------------------------------------------------------
# 9. Small / large tables and extreme values
# ----------------------------------------------------------------------

def test_small_tables():
    """Smallest generated table sizes (1,000 rows)."""
    q = make_query({"S1": 1000, "S2": 1000, "S3": 1000},
                   [("S2", "S1", 0.01), ("S3", "S2", 0.01)])
    est = CostModel(q).estimate_table_join("S1", "S2")
    assert est.output_cardinality == pytest.approx(10_000)
    assert est.cost == pytest.approx(12_000)


def test_tiny_tables_hit_floor():
    """1-row tables with a small selectivity are floored to 1 row."""
    q = make_query({"T1": 1, "T2": 1, "T3": 1},
                   [("T2", "T1", 0.0001), ("T3", "T2", 0.0001)])
    m = CostModel(q)
    est = m.estimate_table_join("T1", "T2")
    assert est.output_cardinality == 1.0
    assert est.cost == pytest.approx(3.0)


def test_large_tables_no_overflow():
    """1M-row tables: exact, finite, and correctly ordered."""
    q = make_query({"L1": 10**6, "L2": 10**6, "L3": 10**6},
                   [("L2", "L1", 0.1), ("L3", "L2", 0.1)])
    m = CostModel(q)
    e1 = m.estimate_table_join("L1", "L2")
    assert e1.output_cardinality == pytest.approx(1e11)
    e2 = m.estimate_join(["L1", "L2"], "L3")
    assert e2.output_cardinality == pytest.approx(1e11 * 1e6 * 0.1)
    assert math.isfinite(e2.cost) and e2.cost > e1.cost


def test_mixed_small_and_large_tables():
    """A tiny and a huge table: cost dominated by the large input/output."""
    q = make_query({"Tiny": 1000, "Huge": 1_000_000, "Mid": 10_000},
                   [("Huge", "Tiny", 0.0001), ("Mid", "Huge", 0.0001)])
    est = CostModel(q).estimate_table_join("Tiny", "Huge")
    assert est.output_cardinality == pytest.approx(1000 * 1_000_000 * 0.0001)
    assert est.cost == pytest.approx(1000 + 1_000_000 + 100_000)


def test_floor_applies_to_long_selective_chain():
    """Six 1,000-row tables at selectivity 1e-4: raw estimate < 1, floored."""
    names = [f"N{i}" for i in range(6)]
    rows = {n: 1000 for n in names}
    edges = [(names[i + 1], names[i], 1e-4) for i in range(5)]
    q = make_query(rows, edges)
    assert CostModel(q).set_cardinality(names) == 1.0
    assert CostModel(q, min_cardinality=0.0).set_cardinality(names) == \
        pytest.approx(1000.0 ** 6 * 1e-4 ** 5)


# ----------------------------------------------------------------------
# Read-only guarantee and generated-database invariants
# ----------------------------------------------------------------------

def test_cost_model_does_not_modify_database(star_query):
    """Database and query are unchanged after heavy use of the model."""
    db_before = repr(star_query.database)
    joins_before = repr(star_query.joins)
    sel_before = copy.deepcopy(star_query.database.join_selectivities)

    m = CostModel(star_query)
    plan = create_initial_plan(star_query, "H")
    m.estimate_available_joins(plan)
    plan = plan.apply_join("X")
    m.plan_cost(plan)
    m.set_cardinality(["H", "X", "Y"])

    assert repr(star_query.database) == db_before
    assert repr(star_query.joins) == joins_before
    assert star_query.database.join_selectivities == sel_before


@pytest.mark.parametrize("num_tables", [3, 4, 5, 6])
@pytest.mark.parametrize("seed", [0, 1, 42, 123])
def test_generated_databases_full_walk(num_tables, seed):
    """Walk a full plan on generated data: every step is valid and finite,
    and plan_cost equals the sum of the per-step costs."""
    q = generate_query(generate_database(num_tables, seed), seed)
    m = CostModel(q)
    plan = create_initial_plan(q)
    running = 0.0
    while not plan.is_complete():
        ests = m.estimate_available_joins(plan)
        assert ests, "connected query must always have an available join"
        for table, est in ests.items():
            assert isinstance(est, JoinEstimate)
            assert est.right_table == table
            assert est.output_cardinality >= 1.0
            assert est.cost > 0 and math.isfinite(est.cost)
            assert 0 < est.selectivity <= 1
        chosen = next(iter(ests))
        running += ests[chosen].cost
        plan = plan.apply_join(chosen)
    assert m.plan_cost(plan) == pytest.approx(running)
    # Final size is independent of the path taken.
    assert m.current_cardinality(plan) == pytest.approx(
        m.set_cardinality(q.tables))


def test_selectivity_matches_database_source():
    """Model selectivities equal the database module's stored values."""
    db = generate_database(5, seed=42)
    q = generate_query(db, seed=42)
    m = CostModel(q)
    for (t1, t2), sel in db.join_selectivities.items():
        assert m.join_selectivity(t1, t2) == sel
        assert m.join_selectivity(t2, t1) == sel

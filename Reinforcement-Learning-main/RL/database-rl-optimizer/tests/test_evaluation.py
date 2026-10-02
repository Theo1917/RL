import json

from src.database import generate_database
from src.evaluation import (compute_plan_cost_spread, confidence_interval,
                             plan_costs, save_results, train_test_split)
from src.query import generate_query


def queries():
    return [generate_query(generate_database(size, seed), seed)
            for size in (3, 4, 5, 6) for seed in range(4)]


def test_split_is_stratified_and_reproducible():
    train, test = train_test_split(queries(), seed=11)
    train_again, test_again = train_test_split(queries(), seed=11)
    assert [q.query_id for q in train] == [q.query_id for q in train_again]
    assert [q.query_id for q in test] == [q.query_id for q in test_again]
    assert {len(q.tables) for q in train} == {3, 4, 5, 6}
    assert {len(q.tables) for q in test} == {3, 4, 5, 6}


def test_spread_matches_plan_cost_extrema():
    query = queries()[3]
    costs = plan_costs(query)
    assert compute_plan_cost_spread(query) == max(costs) / min(costs)


def test_confidence_interval_known_values():
    low, high = confidence_interval([1.0, 2.0, 3.0])
    assert low < 2.0 < high
    assert confidence_interval([]) == (None, None)


def test_save_results_records_metadata_and_appends(tmp_path):
    save_results([{"agent": "x", "cost": 1}], tmp_path,
                 config={"alpha": 0.1}, seed=7, git_commit="abc")
    save_results([{"agent": "y", "cost": 2}], tmp_path,
                 config={"alpha": 0.2}, seed=8, git_commit="def")
    records = json.loads((tmp_path / "runs.json").read_text())
    assert len(records) == 2
    assert records[0]["run_metadata"]["seed"] == 7
    assert list(tmp_path.glob("run_*.csv"))
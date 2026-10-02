"""
Cardinality estimation and cost model for join-order optimization (Milestone 4).

This module answers two questions for a (partial) left-deep join plan:

1. How many rows will the result of a candidate join contain?  (cardinality)
2. How expensive is that join step?                            (cost)

It is deliberately simple, deterministic and explainable. It does NOT try to
reproduce a real optimizer (no indexes, no physical operators, no histograms).

------------------------------------------------------------------------
Formulas
------------------------------------------------------------------------
Two-relation join (the textbook formula)::

    |R JOIN S| = |R| * |S| * selectivity(R, S)

Joining a base table T onto an intermediate result covering the table set S::

    |S JOIN T| = |S| * |T| * prod(selectivity(a, T) for every join edge (a, T)
                                  with a in S)

i.e. every foreign-key predicate that connects T to the tables already joined
is applied (independence assumption). In tree-shaped schemas exactly one edge
connects T to S, so this reduces to the two-relation formula. In schemas with
a cycle (the database generator may add one extra edge) two or more edges
connect T to S and all of them filter the result.

Equivalent closed form for any set of tables X (this is what the code uses,
so the estimate depends only on WHICH tables are joined, never on the order)::

    card(X) = prod(row_count(t) for t in X)
              * prod(selectivity(e) for every join edge e inside X)

Cardinalities are floored at ``min_cardinality`` (default 1 row, as real
optimizers do) so extremely selective joins never produce 0 or fractional
rows. Set ``min_cardinality=0.0`` to disable the floor.

Cost of one join step (documented, transparent)::

    join_cost = |left input| + |right input| + |estimated output|

Cost of a whole (partial) plan is the sum of its join-step costs. A plan that
contains only its start table has cost 0 (no join has happened yet).

Design notes
------------
* The cost model only READS the Query / Database; it never mutates them.
* The existing APIs (``Query.can_join``, ``Query.get_join``,
  ``PartialJoinPlan``) are used unchanged. ``Join.selectivity`` (copied from
  ``Database.join_selectivities`` by ``generate_query``) is the selectivity
  source.
* ``PartialJoinPlan.apply_join`` records only ONE join partner per step. The
  cost model looks at ALL edges between the candidate and the joined set, so
  it stays correct on cyclic schemas without touching ``query.py``.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

from src.query import PartialJoinPlan, Query


class InvalidJoinError(ValueError):
    """Raised when a requested join is not valid for the query/plan."""


@dataclass(frozen=True)
class JoinEstimate:
    """
    Explainable result of estimating one join step.

    Attributes:
        left_tables: Tables already joined (the left, intermediate input).
        right_table: Base table being joined in.
        left_cardinality: Estimated rows of the left (intermediate) input.
        right_cardinality: Row count of the right base table.
        selectivity: Combined selectivity of all predicates applied.
        applied_joins: Canonical table pairs of every predicate applied.
        output_cardinality: Estimated rows of the join result (after floor).
        cost: ``left_cardinality + right_cardinality + output_cardinality``.
    """
    left_tables: FrozenSet[str]
    right_table: str
    left_cardinality: float
    right_cardinality: float
    selectivity: float
    applied_joins: Tuple[Tuple[str, str], ...]
    output_cardinality: float
    cost: float
    used_index: bool = False

    @property
    def result_tables(self) -> FrozenSet[str]:
        """Tables covered by the intermediate relation after this join."""
        return self.left_tables | {self.right_table}

    def explain(self) -> str:
        """Return a one-line human-readable derivation of this estimate."""
        left = "{" + ", ".join(sorted(self.left_tables)) + "}"
        return (
            f"join {left} + {self.right_table}: "
            f"output = {self.left_cardinality:,.0f} x "
            f"{self.right_cardinality:,.0f} x {self.selectivity:.6g} "
            f"= {self.output_cardinality:,.2f}; "
            f"cost = {self.left_cardinality:,.0f} + "
            f"{self.right_cardinality:,.0f} + {self.output_cardinality:,.2f} "
            f"= {self.cost:,.2f}"
        )


# ----------------------------------------------------------------------
# Pure formulas (no Query needed) -- easy to unit test in isolation
# ----------------------------------------------------------------------

def estimate_join_cardinality(left_rows: float,
                              right_rows: float,
                              selectivity: float,
                              min_cardinality: float = 1.0) -> float:
    """
    Estimate ``|R JOIN S| = |R| * |S| * selectivity``.

    Args:
        left_rows: Cardinality of the left relation (>= 0).
        right_rows: Cardinality of the right relation (>= 0).
        selectivity: Fraction of the Cartesian product that survives, in [0, 1].
        min_cardinality: Floor applied to the result (>= 0).

    Returns:
        Estimated output rows, never below ``min_cardinality``.

    Raises:
        ValueError: On negative sizes/floor or selectivity outside [0, 1].
    """
    if left_rows < 0 or right_rows < 0:
        raise ValueError(
            f"Relation sizes must be non-negative, got {left_rows}, {right_rows}")
    if not 0.0 <= selectivity <= 1.0:
        raise ValueError(f"Selectivity must be in [0, 1], got {selectivity}")
    if min_cardinality < 0:
        raise ValueError(f"min_cardinality must be >= 0, got {min_cardinality}")
    return max(float(left_rows) * float(right_rows) * float(selectivity),
               float(min_cardinality))


def compute_join_cost(left_rows: float, right_rows: float,
                      output_rows: float) -> float:
    """
    Cost of a single join step: ``left + right + output``.

    Args:
        left_rows: Rows in the left (intermediate) input.
        right_rows: Rows in the right input.
        output_rows: Estimated rows produced.

    Raises:
        ValueError: If any size is negative.
    """
    if left_rows < 0 or right_rows < 0 or output_rows < 0:
        raise ValueError("Sizes must be non-negative")
    return float(left_rows) + float(right_rows) + float(output_rows)


# ----------------------------------------------------------------------
# Query-aware cost model
# ----------------------------------------------------------------------

class CostModel:
    """
    Deterministic cardinality/cost estimator bound to one ``Query``.

    Typical use by the future RL environment::

        model = CostModel(query)
        for table in plan.get_available_joins():
            est = model.estimate_plan_join(plan, table)   # cost + new size
        plan = plan.apply_join(chosen)

    The model holds only a private cache of derived numbers; the query and
    database are never modified.
    """

    def __init__(self, query: Query, min_cardinality: float = 1.0,
                 use_indexes: bool = False,
                 index_nested_loop_factor: float = 0.1) -> None:
        """
        Args:
            query: The query (join graph + database statistics).
            min_cardinality: Floor for every cardinality estimate (>= 0).
        """
        if min_cardinality < 0:
            raise ValueError(f"min_cardinality must be >= 0, got {min_cardinality}")
        self.query = query
        self.min_cardinality = float(min_cardinality)
        if index_nested_loop_factor <= 0:
            raise ValueError("index_nested_loop_factor must be > 0")
        self.use_indexes = bool(use_indexes)
        self.index_nested_loop_factor = float(index_nested_loop_factor)

        # Read-only snapshots of the statistics we need.
        self._rows: Dict[str, int] = {
            name: query.database.get_table(name).row_count
            for name in query.tables
        }
        self._selectivity: Dict[Tuple[str, str], float] = {
            join.get_canonical_pair(): join.selectivity
            for join in query.joins
        }
        self._card_cache: Dict[FrozenSet[str], float] = {}

    # -- base statistics -------------------------------------------------

    def table_cardinality(self, table: str) -> int:
        """Row count of a base table in the query."""
        if table not in self._rows:
            raise InvalidJoinError(f"Table '{table}' is not part of the query")
        return self._rows[table]

    def join_selectivity(self, table1: str, table2: str) -> float:
        """
        Selectivity of the direct join between two tables.

        Raises:
            InvalidJoinError: If the tables are unknown, identical, or not
                connected by a foreign-key join in the query.
        """
        for t in (table1, table2):
            if t not in self._rows:
                raise InvalidJoinError(f"Table '{t}' is not part of the query")
        if table1 == table2:
            raise InvalidJoinError(f"Cannot join table '{table1}' with itself")
        pair = tuple(sorted((table1, table2)))
        if pair not in self._selectivity:
            raise InvalidJoinError(
                f"No valid join between '{table1}' and '{table2}'")
        return self._selectivity[pair]

    # -- cardinality -----------------------------------------------------

    def set_cardinality(self, tables: Iterable[str]) -> float:
        """
        Estimated size of the join result covering ``tables``.

        Order-independent closed form (see module docstring). A single table
        returns its row count. The set must be non-empty and connected.

        Raises:
            InvalidJoinError: On empty, unknown or disconnected table sets.
        """
        key = frozenset(tables)
        if not key:
            raise InvalidJoinError("Cannot estimate cardinality of an empty table set")
        if key in self._card_cache:
            return self._card_cache[key]

        for t in key:
            if t not in self._rows:
                raise InvalidJoinError(f"Table '{t}' is not part of the query")
        if not self._is_connected(key):
            raise InvalidJoinError(
                f"Tables {sorted(key)} are not connected by join predicates")

        card = 1.0
        for t in sorted(key):                       # sorted => deterministic
            card *= float(self._rows[t])
        for pair in sorted(self._selectivity):      # sorted => deterministic
            if pair[0] in key and pair[1] in key:
                card *= self._selectivity[pair]
        card = max(card, self.min_cardinality)

        self._card_cache[key] = card
        return card

    def estimate_join_cardinality(self, left_rows: float, right_rows: float,
                                  selectivity: float) -> float:
        """Apply ``|R|*|S|*sel`` with this model's cardinality floor."""
        return estimate_join_cardinality(left_rows, right_rows, selectivity,
                                         self.min_cardinality)

    # -- join / cost estimation -----------------------------------------

    def applicable_joins(self, left_tables: Iterable[str],
                         right_table: str) -> List[Tuple[str, str]]:
        """Canonical pairs of all join edges between ``right_table`` and
        ``left_tables`` (sorted)."""
        left = frozenset(left_tables)
        pairs = [tuple(sorted((t, right_table)))
                 for t in left if tuple(sorted((t, right_table))) in self._selectivity]
        return sorted(pairs)

    def estimate_join(self, left_tables: Iterable[str],
                      right_table: str) -> JoinEstimate:
        """
        Estimate joining ``right_table`` onto the intermediate ``left_tables``.

        Raises:
            InvalidJoinError: If ``left_tables`` is empty/unknown/disconnected,
                ``right_table`` is unknown or already joined, or no join edge
                connects ``right_table`` to ``left_tables``.
        """
        left = frozenset(left_tables)
        if not left:
            raise InvalidJoinError("Left side of a join cannot be empty")
        if right_table not in self._rows:
            raise InvalidJoinError(f"Table '{right_table}' is not part of the query")
        if right_table in left:
            raise InvalidJoinError(f"Table '{right_table}' is already joined")

        applied = self.applicable_joins(left, right_table)
        if not applied:
            raise InvalidJoinError(
                f"'{right_table}' has no join predicate with {sorted(left)}")

        left_rows = self.set_cardinality(left)      # validates left set
        right_rows = float(self._rows[right_table])

        selectivity = 1.0
        for pair in applied:
            selectivity *= self._selectivity[pair]

        # Same value as set_cardinality(left | {right}); computed via the
        # closed form so the result depends only on the table SET.
        output_rows = self.set_cardinality(left | {right_table})
        join = next((candidate for candidate in self.query.joins
                     if candidate.get_canonical_pair() == applied[0]), None)
        indexed_column = None
        if join is not None:
            indexed_column = (join.left_column if join.left_table == right_table
                              else join.right_column)
        indexed = bool(
            self.use_indexes and indexed_column and
            indexed_column in self.query.database.get_table(right_table).indexed_columns
        )
        if indexed:
            cost = left_rows * self.index_nested_loop_factor + output_rows
        else:
            cost = compute_join_cost(left_rows, right_rows, output_rows)

        return JoinEstimate(
            left_tables=left,
            right_table=right_table,
            left_cardinality=left_rows,
            right_cardinality=right_rows,
            selectivity=selectivity,
            applied_joins=tuple(applied),
            output_cardinality=output_rows,
            cost=cost,
            used_index=indexed,
        )

    def estimate_table_join(self, table1: str, table2: str) -> JoinEstimate:
        """Estimate the two-relation join ``table1 JOIN table2``
        (table1 is treated as the left input)."""
        return self.estimate_join([table1], table2)

    # -- partial plan support -------------------------------------------

    def _check_plan(self, plan: PartialJoinPlan) -> None:
        if plan.query is not self.query:
            raise ValueError("Plan belongs to a different query than this CostModel")

    def current_cardinality(self, plan: PartialJoinPlan) -> float:
        """Estimated size of the plan's current intermediate relation."""
        self._check_plan(plan)
        return self.set_cardinality(plan.joined_tables)

    def remaining_tables(self, plan: PartialJoinPlan) -> List[str]:
        """Tables not yet joined, sorted."""
        self._check_plan(plan)
        return sorted(plan.remaining_tables)

    def estimate_plan_join(self, plan: PartialJoinPlan,
                           candidate: str) -> JoinEstimate:
        """
        Cost and resulting cardinality of adding ``candidate`` to ``plan``.

        Raises:
            InvalidJoinError: If ``candidate`` is not a remaining table or is
                not joinable to the current intermediate relation.
        """
        self._check_plan(plan)
        if candidate not in plan.remaining_tables:
            raise InvalidJoinError(
                f"Table '{candidate}' is not among the remaining tables")
        return self.estimate_join(plan.joined_tables, candidate)

    def estimate_available_joins(
            self, plan: PartialJoinPlan) -> Dict[str, JoinEstimate]:
        """
        Estimate every valid next join for ``plan``.

        Returns:
            Dict keyed by candidate table name, in sorted key order (same
            order as ``PartialJoinPlan.get_available_joins``).
        """
        self._check_plan(plan)
        return {table: self.estimate_plan_join(plan, table)
                for table in plan.get_available_joins()}

    def plan_cost(self, plan: PartialJoinPlan) -> float:
        """
        Total cost of the joins already made in ``plan`` (sum of step costs).

        The plan's start table is the joined table that never appears as the
        added table in ``join_sequence``. A start-only plan costs 0.
        """
        self._check_plan(plan)
        added = [added for _, added in plan.join_sequence]
        starts = set(plan.joined_tables) - set(added)
        if len(starts) != 1:
            raise ValueError("Plan's join_sequence is inconsistent with joined_tables")

        joined: Set[str] = set(starts)
        total = 0.0
        for table in added:
            total += self.estimate_join(joined, table).cost
            joined.add(table)
        return total

    # -- internals -------------------------------------------------------

    def _is_connected(self, tables: FrozenSet[str]) -> bool:
        """True if ``tables`` form one connected component via join edges."""
        if len(tables) <= 1:
            return True
        start = min(tables)
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for other in tables:
                if other not in seen and \
                        tuple(sorted((cur, other))) in self._selectivity:
                    seen.add(other)
                    stack.append(other)
        return len(seen) == len(tables)


def main() -> None:
    """Demonstration: cost each candidate join while walking one plan."""
    from src.database import generate_database
    from src.query import create_initial_plan, generate_query

    print("=" * 70)
    print("MILESTONE 4 DEMONSTRATION")
    print("Cardinality Estimation and Cost Model")
    print("=" * 70)
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)
    model = CostModel(query)
    print(db)
    print()

    plan = create_initial_plan(query)
    while not plan.is_complete():
        print(f"Intermediate: {sorted(plan.joined_tables)} "
              f"(~{model.current_cardinality(plan):,.0f} rows)")
        estimates = model.estimate_available_joins(plan)
        for est in estimates.values():
            print("  candidate ->", est.explain())
        choice = next(iter(estimates))  # first available (demo only, no policy)
        print(f"  apply {choice}\n")
        plan = plan.apply_join(choice)
    print(f"Total plan cost: {model.plan_cost(plan):,.2f}")


if __name__ == "__main__":
    main()

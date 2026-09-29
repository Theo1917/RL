"""
Query and join graph representation for RL query optimization.

This module represents queries as graphs where:
- Nodes = tables
- Edges = valid join relationships

It provides functionality to:
- Generate queries from database schemas
- Represent join relationships
- Determine valid joins
- Track partial join plans
- Identify available next joins
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, FrozenSet, Optional
import random

from src.database import Database, Table, ForeignKey


@dataclass
class Join:
    """
    Represents a join operation between two tables.

    Attributes:
        left_table: Name of the left table
        right_table: Name of the right table
        left_column: Column name in left table
        right_column: Column name in right table
        selectivity: Join selectivity estimate
    """
    left_table: str
    right_table: str
    left_column: str
    right_column: str
    selectivity: float

    def get_canonical_pair(self) -> Tuple[str, str]:
        """Return canonical (alphabetically sorted) table pair."""
        return tuple(sorted([self.left_table, self.right_table]))

    def involves_table(self, table_name: str) -> bool:
        """Check if this join involves the given table."""
        return table_name in (self.left_table, self.right_table)

    def involves_tables(self, table1: str, table2: str) -> bool:
        """Check if this join connects the two given tables."""
        tables = {self.left_table, self.right_table}
        return tables == {table1, table2}

    def __str__(self) -> str:
        return f"{self.left_table}.{self.left_column} = {self.right_table}.{self.right_column}"


@dataclass
class Query:
    """
    Represents a relational query as a join graph.

    Attributes:
        query_id: Unique identifier for the query
        tables: List of table names participating in the query
        joins: List of valid join operations
        database: Reference to the source database
    """
    query_id: str
    tables: List[str]
    joins: List[Join]
    database: Database

    def __post_init__(self):
        """Validate query after initialization."""
        _validate_query(self)

    def get_table_set(self) -> FrozenSet[str]:
        """Return canonical frozenset of table names."""
        return frozenset(self.tables)

    def can_join(self, table1: str, table2: str) -> bool:
        """
        Check if two tables can be joined.

        Args:
            table1: First table name
            table2: Second table name

        Returns:
            True if a valid join exists between the tables
        """
        for join in self.joins:
            if join.involves_tables(table1, table2):
                return True
        return False

    def get_join(self, table1: str, table2: str) -> Optional[Join]:
        """
        Get the join operation between two tables.

        Args:
            table1: First table name
            table2: Second table name

        Returns:
            Join object if exists, None otherwise
        """
        for join in self.joins:
            if join.involves_tables(table1, table2):
                return join
        return None

    def get_neighbors(self, table_name: str) -> List[str]:
        """
        Get all tables that can be joined with the given table.

        Args:
            table_name: Name of the table

        Returns:
            List of table names that can be joined
        """
        neighbors = []
        for join in self.joins:
            if join.left_table == table_name:
                neighbors.append(join.right_table)
            elif join.right_table == table_name:
                neighbors.append(join.left_table)
        return neighbors

    def is_connected(self) -> bool:
        """
        Check if the query graph is connected.

        Returns:
            True if all tables are reachable from any starting table
        """
        if len(self.tables) == 0:
            return True

        if len(self.tables) == 1:
            return True

        # BFS from first table
        visited = set()
        queue = [self.tables[0]]
        visited.add(self.tables[0])

        while queue:
            current = queue.pop(0)
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor in self.tables:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(self.tables)

    def get_available_joins(self, current_tables: Set[str]) -> List[str]:
        """
        Get tables that can be joined next given current joined tables.

        Args:
            current_tables: Set of table names already joined

        Returns:
            List of table names that can be joined next
        """
        if not current_tables:
            return []

        available = set()
        remaining = set(self.tables) - current_tables

        # Find tables in remaining that connect to current_tables
        for remaining_table in remaining:
            for current_table in current_tables:
                if self.can_join(current_table, remaining_table):
                    available.add(remaining_table)
                    break

        return sorted(list(available))

    def __str__(self) -> str:
        """Return human-readable representation of the query."""
        lines = [f"Query: {self.query_id}", "=" * 50, ""]

        lines.append("Tables:")
        for table in sorted(self.tables):
            lines.append(f"  - {table}")
        lines.append("")

        lines.append("Joins:")
        for i, join in enumerate(self.joins, 1):
            lines.append(f"  {i}. {join} (selectivity: {join.selectivity:.4f})")
        lines.append("")

        return "\n".join(lines)

    def to_sql(self) -> str:
        """
        Generate a conceptual SQL representation (for readability only).

        Note: This is NOT an execution layer, just for debugging/visualization.
        """
        if not self.tables:
            return "-- Empty query"

        if len(self.tables) == 1:
            return f"SELECT * FROM {self.tables[0]};"

        # Start with first table
        sql_lines = ["SELECT *"]
        sql_lines.append(f"FROM {self.tables[0]}")

        # Build join tree (simple left-deep plan for display)
        joined = {self.tables[0]}

        while len(joined) < len(self.tables):
            # Find next table to join
            available = self.get_available_joins(joined)
            if not available:
                break

            next_table = available[0]

            # Find join condition
            for current_table in joined:
                join = self.get_join(current_table, next_table)
                if join:
                    sql_lines.append(f"JOIN {next_table}")
                    sql_lines.append(f"    ON {join}")
                    joined.add(next_table)
                    break

        sql_lines.append(";")
        return "\n".join(sql_lines)


@dataclass
class PartialJoinPlan:
    """
    Represents a partial join plan during construction.

    This tracks the sequence of join decisions made so far.

    Attributes:
        joined_tables: Set of tables already joined (canonical frozenset)
        remaining_tables: Set of tables not yet joined
        join_sequence: Ordered list of joins performed
        query: Reference to the original query
    """
    joined_tables: FrozenSet[str]
    remaining_tables: FrozenSet[str]
    join_sequence: List[Tuple[str, str]]
    query: Query

    def is_complete(self) -> bool:
        """Check if all tables have been joined."""
        return len(self.remaining_tables) == 0

    def get_available_joins(self) -> List[str]:
        """Get tables that can be joined next."""
        return self.query.get_available_joins(set(self.joined_tables))

    def apply_join(self, table_name: str) -> 'PartialJoinPlan':
        """
        Create a new partial plan by joining the given table.

        Args:
            table_name: Name of table to join next

        Returns:
            New PartialJoinPlan with the table joined

        Raises:
            ValueError: If the join is invalid
        """
        if table_name not in self.remaining_tables:
            raise ValueError(f"Table {table_name} is not in remaining tables")

        available = self.get_available_joins()
        if table_name not in available:
            raise ValueError(f"Table {table_name} cannot be joined at this point")

        # Find which current table it joins with (pick first match)
        join_with = None
        for current_table in self.joined_tables:
            if self.query.can_join(current_table, table_name):
                join_with = current_table
                break

        if join_with is None:
            raise ValueError(f"Could not find valid join for {table_name}")

        # Create new plan
        new_joined = frozenset(set(self.joined_tables) | {table_name})
        new_remaining = frozenset(set(self.remaining_tables) - {table_name})
        new_sequence = self.join_sequence + [(join_with, table_name)]

        return PartialJoinPlan(
            joined_tables=new_joined,
            remaining_tables=new_remaining,
            join_sequence=new_sequence,
            query=self.query
        )

    def __str__(self) -> str:
        """Return readable representation of partial plan."""
        lines = []
        lines.append("Partial Join Plan:")
        lines.append(f"  Joined: {sorted(self.joined_tables)}")
        lines.append(f"  Remaining: {sorted(self.remaining_tables)}")
        lines.append(f"  Sequence: {self.join_sequence}")
        return "\n".join(lines)


def generate_query(database: Database, seed: int, query_id: Optional[str] = None) -> Query:
    """
    Generate a query from a database schema.

    For the initial implementation, uses all tables in the database.

    Args:
        database: Source database
        seed: Random seed for reproducibility
        query_id: Optional query identifier (auto-generated if None)

    Returns:
        Query object representing the join graph
    """
    rng = random.Random(seed)

    # Generate query ID if not provided
    if query_id is None:
        query_id = f"query_{seed:06d}"

    # Use all tables
    table_names = [table.name for table in database.tables]

    # Build joins from database relationships
    joins = []

    for table in database.tables:
        for fk in table.foreign_keys:
            # Create join from foreign key
            left_table = table.name
            right_table = fk.references_table
            left_column = fk.column
            right_column = fk.references_column

            # Get selectivity
            canonical_pair = tuple(sorted([left_table, right_table]))
            selectivity = database.join_selectivities.get(canonical_pair, 0.01)

            join = Join(
                left_table=left_table,
                right_table=right_table,
                left_column=left_column,
                right_column=right_column,
                selectivity=selectivity
            )

            joins.append(join)

    # Create query
    query = Query(
        query_id=query_id,
        tables=table_names,
        joins=joins,
        database=database
    )

    return query


def create_initial_plan(query: Query, start_table: Optional[str] = None) -> PartialJoinPlan:
    """
    Create an initial partial join plan.

    Args:
        query: Query to create plan for
        start_table: Optional starting table (uses first table if None)

    Returns:
        PartialJoinPlan with only the start table joined
    """
    if start_table is None:
        start_table = query.tables[0]

    if start_table not in query.tables:
        raise ValueError(f"Start table {start_table} not in query")

    joined = frozenset([start_table])
    remaining = frozenset(set(query.tables) - {start_table})

    return PartialJoinPlan(
        joined_tables=joined,
        remaining_tables=remaining,
        join_sequence=[],
        query=query
    )


def _validate_query(query: Query) -> None:
    """
    Validate that a query satisfies all invariants.

    Raises:
        AssertionError: If any invariant is violated
    """
    # 1. At least 3 tables (consistent with database generation)
    assert len(query.tables) >= 3, f"Query must have at least 3 tables, got {len(query.tables)}"

    # 2. All tables exist in database
    db_table_names = {table.name for table in query.database.tables}
    for table_name in query.tables:
        assert table_name in db_table_names, f"Table {table_name} not in database"

    # 3. All joins reference existing tables
    for join in query.joins:
        assert join.left_table in query.tables, f"Join references non-query table {join.left_table}"
        assert join.right_table in query.tables, f"Join references non-query table {join.right_table}"

    # 4. All join columns exist
    for join in query.joins:
        left_table = query.database.get_table(join.left_table)
        right_table = query.database.get_table(join.right_table)

        left_col_names = [col.name for col in left_table.columns]
        right_col_names = [col.name for col in right_table.columns]

        assert join.left_column in left_col_names, \
            f"Column {join.left_column} not in table {join.left_table}"
        assert join.right_column in right_col_names, \
            f"Column {join.right_column} not in table {join.right_table}"

    # 5. All joins correspond to database relationships
    db_relationships = query.database.get_relationships()
    for join in query.joins:
        canonical_pair = join.get_canonical_pair()
        assert canonical_pair in db_relationships, \
            f"Join {canonical_pair} does not correspond to database relationship"

    # 6. Query graph is connected
    assert query.is_connected(), "Query graph is not connected"

    # 7. No duplicate relationships
    seen_pairs = set()
    for join in query.joins:
        pair = join.get_canonical_pair()
        assert pair not in seen_pairs, f"Duplicate join for {pair}"
        seen_pairs.add(pair)

    # 8. Join selectivities are valid
    for join in query.joins:
        assert 0 < join.selectivity <= 1, \
            f"Invalid selectivity {join.selectivity} for join {join.get_canonical_pair()}"


def main():
    """Demonstration: Generate and display a query."""
    from src.database import generate_database

    print("=" * 70)
    print("MILESTONE 3 DEMONSTRATION")
    print("Query and Join Graph Representation")
    print("=" * 70)
    print()

    # Generate database
    print("Generating 4-table database with seed 42...")
    db = generate_database(num_tables=4, seed=42)
    print(f"Database contains: {[t.name for t in db.tables]}")
    print()

    # Generate query
    print("Generating query from database...")
    query = generate_query(db, seed=42)
    print(query)

    # Demonstrate valid/invalid joins
    print("=" * 70)
    print("JOIN VALIDATION")
    print("=" * 70)
    print()

    if len(query.tables) >= 2:
        t1, t2 = query.tables[0], query.tables[1]
        can_join = query.can_join(t1, t2)
        print(f"Can join {t1} with {t2}? {can_join}")

        if len(query.tables) >= 3:
            t3 = query.tables[2]
            # Check if t1 and t3 can join directly
            can_join_13 = query.can_join(t1, t3)
            print(f"Can join {t1} with {t3}? {can_join_13}")
    print()

    # Demonstrate partial join plan
    print("=" * 70)
    print("PARTIAL JOIN PLAN DEMONSTRATION")
    print("=" * 70)
    print()

    plan = create_initial_plan(query)
    print(f"Initial plan starts with: {list(plan.joined_tables)}")
    print(f"Available next joins: {plan.get_available_joins()}")
    print()

    # Apply first join
    if plan.get_available_joins():
        next_table = plan.get_available_joins()[0]
        print(f"Joining {next_table}...")
        plan = plan.apply_join(next_table)
        print(f"Joined tables: {sorted(plan.joined_tables)}")
        print(f"Available next joins: {plan.get_available_joins()}")
        print()

    # Apply second join
    if plan.get_available_joins():
        next_table = plan.get_available_joins()[0]
        print(f"Joining {next_table}...")
        plan = plan.apply_join(next_table)
        print(f"Joined tables: {sorted(plan.joined_tables)}")
        print(f"Available next joins: {plan.get_available_joins()}")
        print()

    # SQL representation
    print("=" * 70)
    print("SQL REPRESENTATION (for readability only)")
    print("=" * 70)
    print()
    print(query.to_sql())
    print()

    # Reproducibility test
    print("=" * 70)
    print("REPRODUCIBILITY TEST")
    print("=" * 70)
    print()

    query2 = generate_query(db, seed=42)
    print(f"Same seed produces same query ID: {query.query_id == query2.query_id}")
    print(f"Same tables: {query.tables == query2.tables}")
    print(f"Same number of joins: {len(query.joins) == len(query2.joins)}")
    print()


if __name__ == "__main__":
    main()

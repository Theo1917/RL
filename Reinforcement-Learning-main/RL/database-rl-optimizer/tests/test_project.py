"""
Unit tests for database RL optimizer.

Tests will cover:
- Database generation
- Query generation
- Valid join detection
- State transitions
- Cardinality calculation
- Cost calculation
- Reward calculation
- Episode termination
- Q-learning update
- Epsilon-greedy behavior
- Reproducibility with fixed seed
"""

import pytest
from src.database import (
    generate_database, Database, Table, Column, ForeignKey,
    _validate_connectivity
)
from src.query import (
    generate_query, Query, Join, PartialJoinPlan, create_initial_plan
)


def test_project_structure():
    """Test that basic project structure is set up correctly."""
    import os

    # Check that source modules exist
    assert os.path.exists('src/database.py')
    assert os.path.exists('src/query.py')
    assert os.path.exists('src/environment.py')
    assert os.path.exists('src/q_learning.py')
    assert os.path.exists('src/baselines.py')
    assert os.path.exists('src/evaluation.py')

    # Check that results directory exists
    assert os.path.exists('results/plots')


# ============================================================
# MILESTONE 2: Database Generation Tests
# ============================================================

def test_generate_3_tables():
    """TEST 1: Generate a 3-table database successfully."""
    db = generate_database(num_tables=3, seed=42)
    assert len(db.tables) == 3
    assert db.seed == 42
    assert isinstance(db, Database)


def test_generate_4_tables():
    """TEST 2: Generate a 4-table database successfully."""
    db = generate_database(num_tables=4, seed=42)
    assert len(db.tables) == 4
    assert db.seed == 42


def test_generate_6_tables():
    """TEST 3: Generate a 6-table database successfully."""
    db = generate_database(num_tables=6, seed=42)
    assert len(db.tables) == 6
    assert db.seed == 42


def test_reject_too_few_tables():
    """TEST 4: Reject fewer than 3 tables."""
    with pytest.raises(ValueError, match="must be between 3 and 6"):
        generate_database(num_tables=2, seed=42)

    with pytest.raises(ValueError, match="must be between 3 and 6"):
        generate_database(num_tables=0, seed=42)


def test_reject_too_many_tables():
    """TEST 5: Reject more than 6 tables."""
    with pytest.raises(ValueError, match="must be between 3 and 6"):
        generate_database(num_tables=7, seed=42)

    with pytest.raises(ValueError, match="must be between 3 and 6"):
        generate_database(num_tables=10, seed=42)


def test_every_table_has_primary_key():
    """TEST 6: Every table has a primary key."""
    db = generate_database(num_tables=4, seed=42)

    for table in db.tables:
        # Table must have a primary_key attribute
        assert table.primary_key is not None
        assert isinstance(table.primary_key, str)
        assert len(table.primary_key) > 0

        # Primary key must exist in columns
        pk_columns = [col for col in table.columns if col.name == table.primary_key]
        assert len(pk_columns) == 1, f"Table {table.name} primary key not found or duplicated in columns"


def test_foreign_keys_valid():
    """TEST 7: Every foreign key references a valid table and column."""
    db = generate_database(num_tables=5, seed=123)

    table_names = {t.name for t in db.tables}

    for table in db.tables:
        for fk in table.foreign_keys:
            # Referenced table must exist
            assert fk.references_table in table_names, \
                f"Foreign key references non-existent table {fk.references_table}"

            # Referenced column must exist in referenced table
            ref_table = db.get_table(fk.references_table)
            ref_col_names = [col.name for col in ref_table.columns]
            assert fk.references_column in ref_col_names, \
                f"Foreign key references non-existent column {fk.references_column}"


def test_positive_row_counts():
    """TEST 8: Every row count is positive."""
    db = generate_database(num_tables=4, seed=456)

    for table in db.tables:
        assert table.row_count > 0, f"Table {table.name} has non-positive row count"


def test_valid_selectivities():
    """TEST 9: Every selectivity is between 0 and 1."""
    db = generate_database(num_tables=5, seed=789)

    for (t1, t2), sel in db.join_selectivities.items():
        assert 0 < sel <= 1, f"Selectivity for {t1}-{t2} is {sel}, must be in (0, 1]"


def test_schema_connectivity():
    """TEST 10: Generated schema is connected."""
    # Test multiple different databases
    for seed in [42, 123, 456, 789]:
        for num_tables in [3, 4, 5, 6]:
            db = generate_database(num_tables=num_tables, seed=seed)

            # Build adjacency list
            adjacency = {table.name: set() for table in db.tables}

            for table in db.tables:
                for fk in table.foreign_keys:
                    adjacency[table.name].add(fk.references_table)
                    adjacency[fk.references_table].add(table.name)

            # BFS to check connectivity
            visited = set()
            queue = [db.tables[0].name]
            visited.add(db.tables[0].name)

            while queue:
                current = queue.pop(0)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            assert len(visited) == len(db.tables), \
                f"Database with {num_tables} tables and seed {seed} is not connected"


def test_reproducibility_same_seed():
    """TEST 11: Same seed produces identical output."""
    db1 = generate_database(num_tables=4, seed=42)
    db2 = generate_database(num_tables=4, seed=42)

    # Check table names match
    assert [t.name for t in db1.tables] == [t.name for t in db2.tables]

    # Check row counts match
    assert [t.row_count for t in db1.tables] == [t.row_count for t in db2.tables]

    # Check primary keys match
    assert [t.primary_key for t in db1.tables] == [t.primary_key for t in db2.tables]

    # Check number of foreign keys match
    assert [len(t.foreign_keys) for t in db1.tables] == [len(t.foreign_keys) for t in db2.tables]

    # Check selectivities match
    assert db1.join_selectivities == db2.join_selectivities


def test_different_seeds_produce_different_databases():
    """TEST 12: Different seeds can produce different output."""
    db1 = generate_database(num_tables=4, seed=42)
    db2 = generate_database(num_tables=4, seed=123)

    # At least one property should differ
    # (table names, row counts, or selectivities)

    names_different = [t.name for t in db1.tables] != [t.name for t in db2.tables]
    counts_different = [t.row_count for t in db1.tables] != [t.row_count for t in db2.tables]
    selectivities_different = db1.join_selectivities != db2.join_selectivities

    assert names_different or counts_different or selectivities_different, \
        "Different seeds should normally produce different databases"


def test_varied_table_sizes():
    """Test that generated tables have varied row counts."""
    db = generate_database(num_tables=5, seed=2026)

    row_counts = [t.row_count for t in db.tables]

    # Check that we have variation (not all the same)
    assert len(set(row_counts)) > 1, "All tables have the same row count"

    # Check that there's meaningful variation (at least 2x difference)
    min_count = min(row_counts)
    max_count = max(row_counts)
    assert max_count >= 2 * min_count, "Table sizes should have meaningful variation"


def test_database_get_table():
    """Test the get_table method."""
    db = generate_database(num_tables=4, seed=42)

    # Should be able to retrieve each table
    for table in db.tables:
        retrieved = db.get_table(table.name)
        assert retrieved.name == table.name
        assert retrieved.row_count == table.row_count

    # Non-existent table should raise error
    with pytest.raises(ValueError, match="not found"):
        db.get_table("NonExistentTable")


def test_database_get_relationships():
    """Test the get_relationships method."""
    db = generate_database(num_tables=4, seed=42)

    relationships = db.get_relationships()

    # Should return list of tuples
    assert isinstance(relationships, list)
    for rel in relationships:
        assert isinstance(rel, tuple)
        assert len(rel) == 2

    # Each relationship should be canonically ordered (alphabetically)
    for t1, t2 in relationships:
        assert t1 < t2, f"Relationship ({t1}, {t2}) not canonically ordered"


def test_database_str_representation():
    """Test that database has readable string representation."""
    db = generate_database(num_tables=3, seed=42)

    str_repr = str(db)

    # Should contain key information
    assert "Synthetic Database" in str_repr
    assert "Tables: 3" in str_repr
    assert "Seed: 42" in str_repr

    # Should contain table names and row counts
    for table in db.tables:
        assert table.name in str_repr
        assert str(table.row_count) in str_repr


# ============================================================
# MILESTONE 3: Query and Join Graph Representation Tests
# ============================================================

def test_generate_query_from_database():
    """TEST 1: Generate a database and construct a query."""
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)

    assert isinstance(query, Query)
    assert query.query_id is not None
    assert len(query.tables) > 0
    assert len(query.joins) > 0


def test_query_contains_expected_tables():
    """TEST 2: Query contains the expected number of tables."""
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)

    # Query should contain all database tables
    assert len(query.tables) == len(db.tables)

    # All query tables should be from the database
    db_table_names = {t.name for t in db.tables}
    for table in query.tables:
        assert table in db_table_names


def test_all_query_tables_exist_in_database():
    """TEST 3: All query tables exist in the database."""
    db = generate_database(num_tables=5, seed=123)
    query = generate_query(db, seed=123)

    db_table_names = {t.name for t in db.tables}
    for table_name in query.tables:
        assert table_name in db_table_names


def test_valid_join_relationships_detected():
    """TEST 4: Valid join relationships are detected."""
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)

    # All joins in the query should be valid
    for join in query.joins:
        assert join.left_table in query.tables
        assert join.right_table in query.tables

        # Verify the join corresponds to a database relationship
        canonical_pair = join.get_canonical_pair()
        db_relationships = db.get_relationships()
        assert canonical_pair in db_relationships


def test_invalid_table_pairs_rejected():
    """TEST 5: Invalid table pairs are rejected."""
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)

    # Create an invalid table name
    invalid_table = "NonExistentTable123"

    # Should not be able to join with non-existent table
    if query.tables:
        result = query.can_join(query.tables[0], invalid_table)
        assert result == False


def test_join_columns_exist():
    """TEST 6: Join columns exist in their respective tables."""
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)

    for join in query.joins:
        # Get the actual tables
        left_table = db.get_table(join.left_table)
        right_table = db.get_table(join.right_table)

        # Get column names
        left_col_names = [col.name for col in left_table.columns]
        right_col_names = [col.name for col in right_table.columns]

        # Verify columns exist
        assert join.left_column in left_col_names, \
            f"Column {join.left_column} not in table {join.left_table}"
        assert join.right_column in right_col_names, \
            f"Column {join.right_column} not in table {join.right_table}"


def test_query_graph_connected():
    """TEST 7: Query graph is connected."""
    # Test multiple queries
    for seed in [42, 123, 456]:
        for num_tables in [3, 4, 5, 6]:
            db = generate_database(num_tables=num_tables, seed=seed)
            query = generate_query(db, seed=seed)

            # Query should be connected
            assert query.is_connected(), \
                f"Query with {num_tables} tables and seed {seed} is not connected"


def test_available_joins_correctly_identified():
    """TEST 8: Available joins are correctly identified."""
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)

    # Start with first table
    current_tables = {query.tables[0]}
    available = query.get_available_joins(current_tables)

    # Should have at least one available join (since graph is connected)
    assert len(available) > 0

    # Each available table should be joinable with at least one current table
    for avail_table in available:
        can_join = False
        for current_table in current_tables:
            if query.can_join(current_table, avail_table):
                can_join = True
                break
        assert can_join, f"Available table {avail_table} cannot actually be joined"


def test_same_seed_same_query():
    """TEST 9: The same seed produces the same query."""
    db = generate_database(num_tables=4, seed=42)

    query1 = generate_query(db, seed=42)
    query2 = generate_query(db, seed=42)

    # Should have same query ID
    assert query1.query_id == query2.query_id

    # Should have same tables
    assert query1.tables == query2.tables

    # Should have same number of joins
    assert len(query1.joins) == len(query2.joins)


def test_canonical_representations_deterministic():
    """TEST 10: Canonical representations are deterministic."""
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)

    # Get table set multiple times
    set1 = query.get_table_set()
    set2 = query.get_table_set()

    # Should be identical
    assert set1 == set2

    # Should be frozenset (immutable and hashable)
    assert isinstance(set1, frozenset)

    # Test canonical pair for joins
    if query.joins:
        join = query.joins[0]
        pair1 = join.get_canonical_pair()
        pair2 = join.get_canonical_pair()

        assert pair1 == pair2
        # Should be sorted
        assert pair1[0] <= pair1[1]


def test_no_duplicate_joins():
    """TEST 11: No duplicate joins are created."""
    db = generate_database(num_tables=5, seed=456)
    query = generate_query(db, seed=456)

    # Check for duplicate joins
    seen_pairs = set()
    for join in query.joins:
        canonical_pair = join.get_canonical_pair()
        assert canonical_pair not in seen_pairs, \
            f"Duplicate join found: {canonical_pair}"
        seen_pairs.add(canonical_pair)


def test_partial_join_connectivity():
    """TEST 12: Partial join connectivity works correctly."""
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)

    # Create initial plan
    plan = create_initial_plan(query)

    assert len(plan.joined_tables) == 1
    assert len(plan.remaining_tables) == len(query.tables) - 1

    # Get available joins
    available = plan.get_available_joins()
    assert len(available) > 0

    # Apply a join
    next_table = available[0]
    new_plan = plan.apply_join(next_table)

    # Verify new plan state
    assert len(new_plan.joined_tables) == 2
    assert len(new_plan.remaining_tables) == len(query.tables) - 2
    assert next_table in new_plan.joined_tables
    assert next_table not in new_plan.remaining_tables


def test_partial_join_plan_sequence():
    """Test that partial join plans track the sequence correctly."""
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)

    plan = create_initial_plan(query)
    assert len(plan.join_sequence) == 0

    # Apply joins until complete
    steps = 0
    while not plan.is_complete() and steps < 10:
        available = plan.get_available_joins()
        if not available:
            break
        plan = plan.apply_join(available[0])
        steps += 1

    # Should have joined all tables
    assert plan.is_complete()
    assert len(plan.joined_tables) == len(query.tables)
    assert len(plan.remaining_tables) == 0

    # Sequence length should be number of joins
    assert len(plan.join_sequence) == len(query.tables) - 1


def test_query_get_join():
    """Test getting specific join between two tables."""
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)

    # Test with tables that have a join
    if len(query.joins) > 0:
        join = query.joins[0]
        retrieved = query.get_join(join.left_table, join.right_table)

        assert retrieved is not None
        assert retrieved.involves_tables(join.left_table, join.right_table)


def test_query_get_neighbors():
    """Test getting neighbor tables."""
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)

    if query.tables:
        neighbors = query.get_neighbors(query.tables[0])

        # Should have at least one neighbor (connected graph)
        assert len(neighbors) > 0

        # Each neighbor should have a valid join
        for neighbor in neighbors:
            assert query.can_join(query.tables[0], neighbor)


def test_query_to_sql():
    """Test SQL generation (for readability only)."""
    db = generate_database(num_tables=3, seed=42)
    query = generate_query(db, seed=42)

    sql = query.to_sql()

    # Should contain SQL keywords
    assert "SELECT" in sql
    assert "FROM" in sql

    # Should contain table names
    for table in query.tables:
        assert table in sql


def test_join_selectivity_valid():
    """Test that all joins have valid selectivities."""
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)

    for join in query.joins:
        assert 0 < join.selectivity <= 1, \
            f"Invalid selectivity {join.selectivity} for join"


def test_invalid_partial_join_rejected():
    """Test that invalid joins in partial plans are rejected."""
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)

    plan = create_initial_plan(query)

    # Try to join a table that's not available
    available = plan.get_available_joins()

    # Find a table that's not in available list (if any)
    unavailable = None
    for table in plan.remaining_tables:
        if table not in available:
            unavailable = table
            break

    if unavailable:
        with pytest.raises(ValueError):
            plan.apply_join(unavailable)


# Additional tests will be added in later milestones:
# - Milestone 4: Cardinality and cost model tests
# - Milestone 5: Environment tests
# - Milestone 6: Additional environment and cost tests
# - Milestone 8: Q-learning tests

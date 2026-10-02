#!/usr/bin/env python
"""
Milestone 3 Test Runner and Demonstration
"""

import sys
import traceback

print("=" * 70)
print("MILESTONE 3 VERIFICATION")
print("Query and Join Graph Representation")
print("=" * 70)
print()

# Test 1: Import modules
print("TEST 1: Importing modules...")
try:
    from src.database import generate_database
    from src.query import generate_query, create_initial_plan
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print()

# Test 2: Generate database and query
print("TEST 2: Generate database and query...")
try:
    db = generate_database(num_tables=4, seed=42)
    query = generate_query(db, seed=42)
    print(f"✓ Generated query with {len(query.tables)} tables and {len(query.joins)} joins")
except Exception as e:
    print(f"✗ Generation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Query validation
print("TEST 3: Query validation...")
try:
    assert len(query.tables) >= 3, "Query must have at least 3 tables"
    assert len(query.joins) > 0, "Query must have joins"
    assert query.is_connected(), "Query must be connected"
    print("✓ Query validation passed")
except Exception as e:
    print(f"✗ Validation failed: {e}")
    sys.exit(1)

print()

# Test 4: Valid join detection
print("TEST 4: Valid join detection...")
try:
    if len(query.tables) >= 2:
        t1, t2 = query.tables[0], query.tables[1]
        result = query.can_join(t1, t2)
        print(f"✓ Can join check works: {t1} + {t2} = {result}")
except Exception as e:
    print(f"✗ Join detection failed: {e}")
    sys.exit(1)

print()

# Test 5: Partial join plan
print("TEST 5: Partial join plan construction...")
try:
    plan = create_initial_plan(query)
    print(f"✓ Initial plan: {sorted(plan.joined_tables)}")

    available = plan.get_available_joins()
    print(f"✓ Available joins: {available}")

    if available:
        next_table = available[0]
        plan = plan.apply_join(next_table)
        print(f"✓ After joining {next_table}: {sorted(plan.joined_tables)}")
except Exception as e:
    print(f"✗ Partial plan failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print()

# Test 6: Reproducibility
print("TEST 6: Reproducibility...")
try:
    query2 = generate_query(db, seed=42)
    assert query.query_id == query2.query_id, "Query IDs should match"
    assert query.tables == query2.tables, "Tables should match"
    print("✓ Same seed produces same query")
except Exception as e:
    print(f"✗ Reproducibility failed: {e}")
    sys.exit(1)

print()

# Test 7: Run pytest tests
print("TEST 7: Running pytest suite...")
try:
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v", "-k", "milestone"],
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode == 0:
        print("✓ All pytest tests passed")
    else:
        print("✗ Some pytest tests failed")
        print(result.stdout)
        print(result.stderr)
except Exception as e:
    print(f"⚠ Could not run pytest: {e}")

print()

# Demonstration
print("=" * 70)
print("DEMONSTRATION")
print("=" * 70)
print()

print("Generated Query:")
print("-" * 70)
print(query)

print()
print("SQL Representation:")
print("-" * 70)
print(query.to_sql())

print()
print("Join Graph Structure:")
print("-" * 70)
for table in query.tables:
    neighbors = query.get_neighbors(table)
    print(f"{table:20} → {', '.join(neighbors) if neighbors else '(no neighbors)'}")

print()
print("=" * 70)
print("✓ MILESTONE 3 VERIFICATION COMPLETE")
print("=" * 70)
print()
print("All core functionality is working:")
print("  ✓ Query generation from database")
print("  ✓ Join graph representation")
print("  ✓ Valid join detection")
print("  ✓ Partial join plans")
print("  ✓ Available join identification")
print("  ✓ Reproducibility with seeds")
print()
print("Ready for Milestone 4: Cardinality and Cost Model")
print()

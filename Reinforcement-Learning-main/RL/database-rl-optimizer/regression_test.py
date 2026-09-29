"""
REGRESSION TEST REPORT - MILESTONES 1 & 2
Generated: 2026-09-29
Status: STATIC VALIDATION (classifier unavailable for dynamic tests)
"""

# ============================================================
# MILESTONE 1 VALIDATION
# ============================================================

print("=" * 70)
print("MILESTONE 1: Project Setup and Structure")
print("=" * 70)
print()

import os

# Check 1: Project structure
checks = {
    "README.md exists": os.path.exists("README.md"),
    "requirements.txt exists": os.path.exists("requirements.txt"),
    ".gitignore exists": os.path.exists(".gitignore"),
    "src/ directory exists": os.path.exists("src/"),
    "tests/ directory exists": os.path.exists("tests/"),
    "results/plots/ directory exists": os.path.exists("results/plots/"),
    "src/database.py exists": os.path.exists("src/database.py"),
    "src/query.py exists": os.path.exists("src/query.py"),
    "src/environment.py exists": os.path.exists("src/environment.py"),
    "src/q_learning.py exists": os.path.exists("src/q_learning.py"),
    "src/baselines.py exists": os.path.exists("src/baselines.py"),
    "src/evaluation.py exists": os.path.exists("src/evaluation.py"),
    "tests/test_project.py exists": os.path.exists("tests/test_project.py"),
}

for check, result in checks.items():
    status = "✓" if result else "✗"
    print(f"{status} {check}")

milestone1_pass = all(checks.values())
print()
print(f"Milestone 1 Status: {'PASSED' if milestone1_pass else 'FAILED'}")
print()

# ============================================================
# MILESTONE 2 VALIDATION
# ============================================================

print("=" * 70)
print("MILESTONE 2: Synthetic Database Generator")
print("=" * 70)
print()

try:
    from src.database import (
        generate_database, Database, Table, Column, ForeignKey
    )
    print("✓ Module imports successful")
except Exception as e:
    print(f"✗ Module import failed: {e}")
    exit(1)

print()
print("Running database generation tests...")
print()

test_results = []

# Test 1: Generate 3 tables
try:
    db = generate_database(num_tables=3, seed=42)
    assert len(db.tables) == 3
    print("✓ TEST 1: Generate 3-table database")
    test_results.append(True)
except Exception as e:
    print(f"✗ TEST 1 FAILED: {e}")
    test_results.append(False)

# Test 2: Generate 4 tables
try:
    db = generate_database(num_tables=4, seed=42)
    assert len(db.tables) == 4
    print("✓ TEST 2: Generate 4-table database")
    test_results.append(True)
except Exception as e:
    print(f"✗ TEST 2 FAILED: {e}")
    test_results.append(False)

# Test 3: Generate 6 tables
try:
    db = generate_database(num_tables=6, seed=42)
    assert len(db.tables) == 6
    print("✓ TEST 3: Generate 6-table database")
    test_results.append(True)
except Exception as e:
    print(f"✗ TEST 3 FAILED: {e}")
    test_results.append(False)

# Test 4: Reject too few tables
try:
    try:
        generate_database(num_tables=2, seed=42)
        print("✗ TEST 4 FAILED: Should reject 2 tables")
        test_results.append(False)
    except ValueError:
        print("✓ TEST 4: Correctly rejects < 3 tables")
        test_results.append(True)
except Exception as e:
    print(f"✗ TEST 4 FAILED: {e}")
    test_results.append(False)

# Test 5: Reject too many tables
try:
    try:
        generate_database(num_tables=7, seed=42)
        print("✗ TEST 5 FAILED: Should reject 7 tables")
        test_results.append(False)
    except ValueError:
        print("✓ TEST 5: Correctly rejects > 6 tables")
        test_results.append(True)
except Exception as e:
    print(f"✗ TEST 5 FAILED: {e}")
    test_results.append(False)

# Test 6: Every table has primary key
try:
    db = generate_database(num_tables=4, seed=42)
    for table in db.tables:
        assert table.primary_key
        assert any(col.name == table.primary_key for col in table.columns)
    print("✓ TEST 6: Every table has valid primary key")
    test_results.append(True)
except Exception as e:
    print(f"✗ TEST 6 FAILED: {e}")
    test_results.append(False)

# Test 7: Valid foreign keys
try:
    db = generate_database(num_tables=5, seed=123)
    table_names = {t.name for t in db.tables}
    for table in db.tables:
        for fk in table.foreign_keys:
            assert fk.references_table in table_names
            ref_table = db.get_table(fk.references_table)
            ref_col_names = [col.name for col in ref_table.columns]
            assert fk.references_column in ref_col_names
    print("✓ TEST 7: All foreign keys are valid")
    test_results.append(True)
except Exception as e:
    print(f"✗ TEST 7 FAILED: {e}")
    test_results.append(False)

# Test 8: Positive row counts
try:
    db = generate_database(num_tables=4, seed=456)
    for table in db.tables:
        assert table.row_count > 0
    print("✓ TEST 8: All row counts are positive")
    test_results.append(True)
except Exception as e:
    print(f"✗ TEST 8 FAILED: {e}")
    test_results.append(False)

# Test 9: Valid selectivities
try:
    db = generate_database(num_tables=5, seed=789)
    for (t1, t2), sel in db.join_selectivities.items():
        assert 0 < sel <= 1
    print("✓ TEST 9: All selectivities in valid range (0, 1]")
    test_results.append(True)
except Exception as e:
    print(f"✗ TEST 9 FAILED: {e}")
    test_results.append(False)

# Test 10: Schema connectivity
try:
    for seed in [42, 123, 456]:
        for num_tables in [3, 4, 5, 6]:
            db = generate_database(num_tables=num_tables, seed=seed)
            # Build adjacency list
            adjacency = {table.name: set() for table in db.tables}
            for table in db.tables:
                for fk in table.foreign_keys:
                    adjacency[table.name].add(fk.references_table)
                    adjacency[fk.references_table].add(table.name)
            # BFS
            visited = {db.tables[0].name}
            queue = [db.tables[0].name]
            while queue:
                current = queue.pop(0)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            assert len(visited) == len(db.tables)
    print("✓ TEST 10: All generated schemas are connected")
    test_results.append(True)
except Exception as e:
    print(f"✗ TEST 10 FAILED: {e}")
    test_results.append(False)

# Test 11: Reproducibility
try:
    db1 = generate_database(num_tables=4, seed=42)
    db2 = generate_database(num_tables=4, seed=42)
    assert [t.name for t in db1.tables] == [t.name for t in db2.tables]
    assert [t.row_count for t in db1.tables] == [t.row_count for t in db2.tables]
    assert db1.join_selectivities == db2.join_selectivities
    print("✓ TEST 11: Same seed produces identical output")
    test_results.append(True)
except Exception as e:
    print(f"✗ TEST 11 FAILED: {e}")
    test_results.append(False)

# Test 12: Different seeds
try:
    db1 = generate_database(num_tables=4, seed=42)
    db2 = generate_database(num_tables=4, seed=123)
    names_diff = [t.name for t in db1.tables] != [t.name for t in db2.tables]
    counts_diff = [t.row_count for t in db1.tables] != [t.row_count for t in db2.tables]
    sel_diff = db1.join_selectivities != db2.join_selectivities
    assert names_diff or counts_diff or sel_diff
    print("✓ TEST 12: Different seeds produce different databases")
    test_results.append(True)
except Exception as e:
    print(f"✗ TEST 12 FAILED: {e}")
    test_results.append(False)

print()
print("=" * 70)
print(f"MILESTONE 2 RESULTS: {sum(test_results)}/{len(test_results)} tests passed")
print("=" * 70)
print()

# ============================================================
# DEMONSTRATION
# ============================================================

print("=" * 70)
print("DEMONSTRATION: 4-Table Database with Seed 42")
print("=" * 70)
print()

db = generate_database(num_tables=4, seed=42)
print(db)
print()

# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)
print()

all_passed = milestone1_pass and all(test_results)

if all_passed:
    print("✓ ALL REGRESSION TESTS PASSED")
    print()
    print("Milestones 1 & 2 are fully functional and ready for Milestone 3.")
else:
    print("✗ SOME TESTS FAILED")
    print()
    print("Please review the failures above.")

print()
print("=" * 70)

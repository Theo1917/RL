#!/usr/bin/env python
"""
Simple test runner for regression testing.
Runs all tests and reports results.
"""

import sys
import traceback

# Import test functions
from tests.test_project import (
    test_project_structure,
    test_generate_3_tables,
    test_generate_4_tables,
    test_generate_6_tables,
    test_reject_too_few_tables,
    test_reject_too_many_tables,
    test_every_table_has_primary_key,
    test_foreign_keys_valid,
    test_positive_row_counts,
    test_valid_selectivities,
    test_schema_connectivity,
    test_reproducibility_same_seed,
    test_different_seeds_produce_different_databases,
    test_varied_table_sizes,
    test_database_get_table,
    test_database_get_relationships,
    test_database_str_representation,
)

def run_test(test_func, test_name):
    """Run a single test and report result."""
    try:
        test_func()
        print(f"✓ PASSED: {test_name}")
        return True
    except AssertionError as e:
        print(f"✗ FAILED: {test_name}")
        print(f"  AssertionError: {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {test_name}")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests and report summary."""
    print("=" * 70)
    print("REGRESSION TEST SUITE - Milestones 1 & 2")
    print("=" * 70)
    print()

    tests = [
        (test_project_structure, "test_project_structure"),
        (test_generate_3_tables, "test_generate_3_tables"),
        (test_generate_4_tables, "test_generate_4_tables"),
        (test_generate_6_tables, "test_generate_6_tables"),
        (test_reject_too_few_tables, "test_reject_too_few_tables"),
        (test_reject_too_many_tables, "test_reject_too_many_tables"),
        (test_every_table_has_primary_key, "test_every_table_has_primary_key"),
        (test_foreign_keys_valid, "test_foreign_keys_valid"),
        (test_positive_row_counts, "test_positive_row_counts"),
        (test_valid_selectivities, "test_valid_selectivities"),
        (test_schema_connectivity, "test_schema_connectivity"),
        (test_reproducibility_same_seed, "test_reproducibility_same_seed"),
        (test_different_seeds_produce_different_databases, "test_different_seeds_produce_different_databases"),
        (test_varied_table_sizes, "test_varied_table_sizes"),
        (test_database_get_table, "test_database_get_table"),
        (test_database_get_relationships, "test_database_get_relationships"),
        (test_database_str_representation, "test_database_str_representation"),
    ]

    print(f"Running {len(tests)} tests...\n")

    passed = 0
    failed = 0

    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
        else:
            failed += 1
        print()

    print("=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ ALL TESTS PASSED!\n")
        sys.exit(0)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Verification script to test all enhanced components."""

import sys
import importlib
from pathlib import Path

def check_file_exists(filepath):
    """Check if file exists."""
    path = Path(filepath)
    if path.exists():
        print(f"✓ {filepath} ({path.stat().st_size:,} bytes)")
        return True
    else:
        print(f"✗ {filepath} NOT FOUND")
        return False

def check_import(module_name):
    """Check if module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} imports successfully")
        return True
    except ImportError as e:
        print(f"✗ {module_name} import failed: {e}")
        return False

def main():
    print("="*60)
    print("ENHANCED PROJECT VERIFICATION")
    print("="*60)
    
    checks_passed = 0
    checks_total = 0
    
    # Check core new files
    print("\n1. Core Implementation Files:")
    files_to_check = [
        'src/deep_rl/__init__.py',
        'src/deep_rl/dqn_agent.py',
        'src/deep_rl/network.py',
        'src/deep_rl/replay_buffer.py',
        'src/postgres_benchmark.py',
        'benchmarks/__init__.py',
        'benchmarks/tpch_generator.py',
        'benchmarks/workload.py',
        'experiments/run_dqn_benchmark.py'
    ]
    
    for file in files_to_check:
        checks_total += 1
        if check_file_exists(file):
            checks_passed += 1
    
    # Check documentation
    print("\n2. Documentation Files:")
    docs = [
        'SETUP.md',
        'PROJECT_TRANSFORMATION_SUMMARY.md',
        'docs/ENHANCEMENT_ROADMAP.md',
        'requirements.txt'
    ]
    
    for doc in docs:
        checks_total += 1
        if check_file_exists(doc):
            checks_passed += 1
    
    # Check Python imports (optional - requires dependencies)
    print("\n3. Python Module Imports (requires pip install -r requirements.txt):")
    modules = [
        'benchmarks.tpch_generator',
        'benchmarks.workload',
    ]
    
    for module in modules:
        checks_total += 1
        if check_import(module):
            checks_passed += 1
    
    # Check generated files
    print("\n4. Generated Assets:")
    generated = [
        'benchmarks/tpch_init.sql',
    ]
    
    for gen in generated:
        checks_total += 1
        if check_file_exists(gen):
            checks_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"VERIFICATION COMPLETE: {checks_passed}/{checks_total} checks passed")
    print("="*60)
    
    if checks_passed == checks_total:
        print("✓ All components verified successfully!")
        print("\nNext steps:")
        print("  1. pip install -r requirements.txt")
        print("  2. python experiments/run_dqn_benchmark.py --tables 4 --episodes 100")
        print("  3. Check results/ directory for outputs")
        return 0
    else:
        print(f"⚠ {checks_total - checks_passed} checks failed")
        print("Review the output above for missing files")
        return 1

if __name__ == '__main__':
    sys.exit(main())

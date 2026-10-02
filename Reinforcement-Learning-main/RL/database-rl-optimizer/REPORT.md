# Reinforcement Learning for Join Ordering

This mini-project models left-deep join ordering as a deterministic MDP. A
state is the canonical set of joined tables and an action selects a valid
remaining neighbor. The simulator estimates cardinality under independent
selectivities and charges intermediate input plus output rows.

The implementation includes seeded random and greedy baselines, exhaustive
optimal search for up to six tables, and masked tabular Q-learning. Run
`python experiments/run_small.py --seed 42` to regenerate the CSV/JSON result
logs and learning curve. Those outputs are estimates from the simulator, not
PostgreSQL measurements. PostgreSQL validation is an optional adapter in
`src/pg_validation.py` and skips clearly when no server or driver is present.

Known limitations are the synthetic independence assumptions, left-deep-only
plans, proxy memory/latency terms, and tabular scaling. The next experimental
steps are multi-seed train/test evaluation, exploration sensitivity, and live
PostgreSQL EXPLAIN comparisons.
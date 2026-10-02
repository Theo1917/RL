# Five-Minute Live Demo

The demo is deterministic and reports simulator estimates separately from any
PostgreSQL measurements.

## 0:00 Setup

From `database-rl-optimizer/`:

```powershell
pip install -r requirements.txt
python demo.py --seed 42 --no-postgres
```

Expected: the schema, join graph, conceptual SQL, MDP walkthrough, training
progress, five-policy table, and an explicit `Q-learning gap to exhaustive
optimum` line. The verified 500-episode preset converges for the demo seed.
If a different seed reports `not converged`, show that result; do not replace
it with an oracle plan.

## 1:00 Explain the MDP

Point to the empty initial state, the valid-action list, the selected table,
and the negative incremental reward. Explain that the first action selects the
start table and later actions add adjacent tables. The exhaustive policy is an
oracle reference, not a training signal.

## 2:00 Show the browser showcase

```powershell
python -m webapp.server --port 8000
```

Open `http://127.0.0.1:8000`. Change the seed or table count, click
`Generate & train`, then use `Replay trace`. Point out the schema, learning
curve, benchmark ratios, and the footer warning that estimated cost is not
measured database time.

## 3:30 Show reproducible studies

```powershell
python experiments/run_comparison.py --episodes 250
python experiments/run_exploration.py
python experiments/run_scaling.py
```

Open the generated `results/comparison.csv` and timestamped `results/run_*.csv`
files, plus plots in `results/plots/`.
Mention that the checked-in comparison uses ten seeds per table count and that
the held-out shared-Q result is reported even when it is worse than greedy.

## 4:15 Optional PostgreSQL path

```powershell
docker compose up -d
python -m src.pg_validation
python demo.py --seed 42 --replay
```

The live path needs `psycopg2` and the `PGHOST`, `PGPORT`, `PGUSER`,
`PGPASSWORD`, and `PGDATABASE` variables. If Docker, the driver, or the server
is unavailable, use `--no-postgres`; if saved `results/pg/summary.json` exists,
use `--replay`. Never describe simulator values as execution times.

## 4:50 Close

State the result and limitation plainly: the per-query tabular agent can learn
small join orders, but generalization and six-table scaling remain difficult.
The optimizer model is synthetic and left-deep, so PostgreSQL validation is
needed before making database-performance claims.

## Viva Questions

1. **Why is this an RL problem?** A sequential join-order policy makes actions whose future cost depends on the current partial plan.
2. **What is the state?** Joined-table structure plus cardinality, candidate selectivity, accumulated-cost, and index features.
3. **What is the action?** Select one valid remaining table; the action mask removes disconnected or already joined tables.
4. **What is the reward?** Negative incremental simulator cost by default, with an optional normalized cost/intermediate/memory/latency objective.
5. **Why epsilon-greedy?** It guarantees occasional discovery while increasingly exploiting learned Q-values.
6. **Why tabular Q-learning?** Three-to-six table left-deep plans are small enough to remain transparent for a college project.
7. **What is the update rule?** The current value moves toward reward plus discounted maximum valid next-action value.
8. **Why mask the next maximum?** Invalid actions must not influence the learned continuation value.
9. **What is the exhaustive baseline?** Enumeration of every valid left-deep order; it is feasible through six tables and supplies the simulator optimum.
10. **Why can greedy tie or beat RL?** A heuristic can exploit an informative immediate signal while a short or poorly explored Q run has not learned delayed effects.
11. **Why is shared Q useful?** It reuses experience across queries; its bucketed representation trades table-specific precision for transfer.
12. **Why does six-table learning struggle?** The number of partial states and possible plans grows rapidly while a tabular agent has no function approximation.
13. **What does the cost model assume?** Independent join predicates, closed-form cardinality, and a transparent intermediate-materialization cost.
14. **How is PostgreSQL validated?** Synthetic tables are materialized at a documented scale, analyzed, and compared with forced-order and default-planner EXPLAIN ANALYZE runs.
15. **Why not claim the simulator is reality?** Its costs are estimates; only PostgreSQL execution-time measurements can support real database-performance conclusions.

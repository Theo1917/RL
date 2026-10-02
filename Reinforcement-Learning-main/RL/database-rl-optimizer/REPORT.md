# Reinforcement Learning for Adaptive Database Query Plan Optimization

## Problem

Join ordering is a sequential decision problem: the next table changes the
size and cost of every later intermediate relation. This project studies
whether a transparent tabular Q-learning agent can learn useful left-deep join
orders for synthetic three-to-six-table queries.

## MDP

The environment is `JoinOrderEnv`. A state contains the joined-table set,
remaining valid actions, candidate selectivities, log intermediate cardinality,
accumulated cost, and optional index flags. An action selects a valid adjacent
table. The transition is deterministic and applies all predicates connecting
the new table to the current component. The default reward is negative
incremental cost, `|left| + |right| + |output|`; an optional normalized
multi-objective reward adds intermediate rows, a memory proxy, and a latency
proxy. The horizon is `n - 1` for a fixed start and `n` when the agent selects
the start. Exhaustive valid left-deep enumeration supplies the simulator
reference optimum.

## Method

`QLearningAgent` uses masked epsilon-greedy tabular updates. Per-query mode
uses the exact joined-table key. Shared mode uses joined-position structure and
bucketed cardinality, selectivity, and index features so one table can act on
unseen query instances. Random, greedy-intermediate, greedy-immediate-cost,
and exhaustive baselines use the same environment interface. All scripts use
explicit seeds and write raw logs before plotting.

## Experiment Setup

The comparison study was generated with:

```powershell
python experiments/run_comparison.py --episodes 250
```

It used ten seeds for each of 3, 4, 5, and 6 tables, 40 per-query test
instances, and 16 held-out shared-Q instances. Ratios are simulator plan cost
divided by exhaustive simulator cost. They are not PostgreSQL execution times.
The raw data is in `results/comparison.csv` and the aggregate is in
`results/comparison_summary.json`.

## Generated Results

| Agent | Queries | Mean ratio | 95% CI half-width |
|---|---:|---:|---:|
| Random | 40 | 1.657 | 0.618 |
| Greedy intermediate | 40 | 1.131 | 0.158 |
| Greedy immediate cost | 40 | 1.131 | 0.158 |
| Q-learning, per query | 40 | 1.022 | 0.015 |
| Q-learning, shared held-out | 16 | 1.735 | 0.866 |
| Exhaustive optimum | 40 | 1.000 | 0.000 |

These values are generated from the checked-in script and are included to
make the current run auditable. The result is favorable to per-query
Q-learning but unfavorable to the simple shared representation; that is an
important limitation rather than a result to hide. Greedy policies are close
to Q-learning and can tie it on flat or simple instances.

The companion scripts generate convergence, sample-efficiency, exploration,
scaling, and reward-ablation logs. Their plots include learning curves, ratio
comparisons, boxplots, exploration ratios, and plan-tree views under
`results/plots/`.

## PostgreSQL Validation

`src/pg_validation.py` is an optional validation layer. It can create scaled
synthetic tables with primary keys, foreign keys, indexes, and `ANALYZE`; build
exact explicit JOIN order SQL; force order with `join_collapse_limit = 1` and
`from_collapse_limit = 1`; and compare forced plans with PostgreSQL's default
planner using five-run warm-up/median EXPLAIN ANALYZE measurements. It parses
planning time, execution time, total cost, peak actual rows, memory, and shared
buffers, saving raw JSON under `results/pg/`.

No live PostgreSQL execution-time numbers are claimed in this report because
the current validation environment does not have `psycopg2` installed. The
module skips clearly, and the simulator results above must not be read as
measured database timings.

## Limitations

The cardinality model assumes independent predicates and synthetic row-count
statistics. Plans are left-deep, exhaustive enumeration is only realistic for
this small table range, and tabular Q-learning does not scale gracefully. The
shared bucketed state loses table-specific information, which is visible in
its held-out result. The memory and latency terms are documented proxies, not
resource measurements. PostgreSQL validation is necessary before claiming
real workload performance.

## Future Work

Future work is to run the PostgreSQL container with the driver installed,
collect cross-plan Spearman correlations, calibrate FK generation against
actual cardinalities, and test a richer shared representation or function
approximator while preserving the transparent tabular baseline.

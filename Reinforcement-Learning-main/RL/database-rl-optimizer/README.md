# Reinforcement Learning for Database Query Plan Optimization

A RL mini-project that uses tabular Q-learning to learn efficient JOIN ORDER selection for relational database queries.

## Project Overview

This project builds an RL agent that learns to select the order in which tables should be joined in multi-table relational queries. The goal is to minimize estimated query execution cost through sequential decision-making.

### Why This Is an RL Problem

- **Agent**: Q-learning agent
- **Environment**: Relational query optimization environment
- **State**: Current partial join plan and relevant database/query statistics
- **Action**: Select the next valid join
- **Transition**: Selected join produces new partial query plan with updated statistics
- **Reward**: Negative estimated execution cost
- **Episode**: Constructing one complete join order
- **Objective**: Maximize cumulative reward (minimize total query cost)

The system demonstrates:
- Sequential decision making
- Trial and error learning
- Exploration vs exploitation
- Delayed consequences
- Cumulative reward optimization

## Project Scope

- Synthetic relational databases with 3-6 tables
- Tabular Q-learning (no deep learning)
- JOIN ORDER optimization (not full query optimization)
- Transparent cost models for educational purposes
- Comparison with baseline heuristic strategies

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
database-rl-optimizer/
│
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore patterns
│
├── src/
│   ├── database.py         # Synthetic database generation
│   ├── query.py            # Query representation
│   ├── cost_model.py       # Cardinality estimation and cost model
│   ├── environment.py      # RL environment (Gymnasium)
│   ├── q_learning.py       # Tabular Q-learning implementation
│   ├── baselines.py        # Baseline strategies
│   ├── evaluation.py       # Training and evaluation
│   └── pg_validation.py    # Optional PostgreSQL EXPLAIN validation
│
├── tests/
│   ├── test_project.py, test_cost_model.py, test_environment.py
│   ├── test_baselines.py, test_q_learning.py, test_shared_q_learning.py
│   ├── test_evaluation.py, test_pg_validation.py
│
├── demo.py                 # Terminal live demo
├── webapp/                 # Browser showcase and API server
├── experiments/            # Re-runnable multi-seed studies
│
└── results/
    └── plots/              # Experiment visualizations
```

(`gymnasium` is required for the environment tests; it is in `requirements.txt`.)

## Research Question

**Can a reinforcement learning agent learn effective join-ordering policies for relational query optimization, and how do those policies compare with heuristic strategies across query complexity, exploration settings, and training budgets?**
#### Schema Generation

Databases are generated with 3-6 tables representing common relational entities (Customer, Order, Product, etc.). Each generation creates:

1. **Table names**: Selected from a pool of common database entity names
2. **Connected relationships**: A tree structure ensures all tables are reachable via foreign keys
3. **Primary keys**: Every table has a unique identifier (e.g., `customer_id`)
4. **Foreign keys**: Child tables reference parent tables, defining valid join operations

#### Join Selectivity

Each foreign key relationship has an associated selectivity estimate representing the fraction of the Cartesian product that appears in the join result. Selectivities are generated with log-uniform distribution between 0.01% and 10%.

For example, if `Customers` has 10,000 rows and `Orders` has 100,000 rows with selectivity 0.01:
- Cartesian product: 10,000 × 100,000 = 1,000,000,000
- Estimated join result: 1,000,000,000 × 0.01 = 10,000,000 rows

### Query and Join Graph Representation (Milestone 3)

The database schema is converted into a structured query representation that will later be consumed by the RL environment. This conversion is necessary because the RL agent needs to make sequential decisions about join order, which requires a graph-based view of valid operations.

#### Why Graph Representation?

Join-order optimization is inherently a graph problem:
- **Nodes** represent tables
- **Edges** represent valid join relationships (foreign keys)
- **Paths** represent sequences of join operations
- **Connected components** ensure meaningful queries

This graph structure allows the RL agent to determine which actions (joins) are valid at each decision point.

#### Query Structure

A `Query` object contains:
- **Tables**: List of tables participating in the query
- **Joins**: Explicit join operations with columns and selectivities
- **Database reference**: Link to source database for validation

Each `Join` specifies:
- Left and right table names
- Participating columns
- Join selectivity (from database generation)

#### Valid Join Detection

The system determines whether two tables can be joined based on the database schema's foreign key relationships. Only joins corresponding to actual foreign keys are permitted. This prevents the agent from attempting impossible operations.
#### Partial Join Plans

A `PartialJoinPlan` tracks the state during join-order construction:
- **Joined tables**: Set of tables already joined (canonical frozenset)
- **Remaining tables**: Tables not yet joined
- **Join sequence**: Ordered list of join decisions made so far

This representation will later form the basis of the RL state.

#### Available Joins

Given a partial plan, the system identifies which tables can be joined next. A table is available if:
1. It's not already joined
2. It has a foreign key relationship with at least one joined table
3. It connects to the current joined component (maintains connectivity)

This functionality ensures the RL agent only sees valid actions at each step.


#### How This Supports RL

The query representation provides everything the RL environment will need:
- **State space**: Partial join plans with canonical representation
- **Action space**: Available next joins at each state
- **Transition logic**: Applying a join produces a new partial plan
- **Validation**: Only valid, connected plans are allowed

The RL agent will navigate this structured representation to learn effective join orders.

### RL Environment (Milestone 5)

Implemented in `src/environment.py` as `JoinOrderEnv`, a
[Gymnasium](https://gymnasium.farama.org/) environment (passes
`gymnasium.utils.env_checker.check_env`). It builds directly on `Query`,
`PartialJoinPlan` and `CostModel`; nothing in Milestones 1-4 was changed.


#### Action representation and mapping

The action space is fixed: `Discrete(n_tables)`. Action `i` means "join the
table at position `i` of `sorted(query.tables)`" (alphabetical, independent of
`query.tables` order, identical at every step). Example for seed 42:
`{0: Assignment, 1: Department, 2: Invoice, 3: Location}`. Use
`env.action_table_names`, `env.action_to_table(i)` and `env.table_to_action(name)`.
The join itself is not part of the action: it is determined by the current
plan (every predicate between the chosen table and the joined set is applied
by the cost model).

#### Action mask

`env.valid_action_mask()` returns an `int8` vector of length `n_tables`
(`1` = legal). It is also in `info["action_mask"]` (ids in
`info["valid_actions"]`). Action `i` is legal iff its table is not yet joined
and has a join predicate with the current plan
(`PartialJoinPlan.get_available_joins`). The mask is all zeros once the plan is
complete. The action-space size never changes during an episode.


### Q-Learning Algorithm
`src/q_learning.py` provides `QLearningAgent` with the standard masked update
`Q(s,a) += alpha * (r + gamma * max(Q(s', a')) - Q(s,a))`. Terminal transitions
omit the future term, and the maximum is restricted to valid actions. Constant,
linear, and exponential epsilon schedules plus optimistic initialization are
supported. The initial key is the empty set when `start_mode="agent"` selects
the start table; otherwise it is the fixed start-table set.

Two state-table modes are available. The default per-query table uses the
exact joined-table `frozenset` and is strongest when training and evaluation
use the same query. `shared=True` uses a generalized key containing joined
position structure, bucketed log-cardinality, log-selectivity buckets, and
index flags. It can act on unseen queries with the same action schema, but
loses table-specific detail and can be substantially less accurate.

The generated ten-seed comparison currently reports mean simulator cost ratios
of 1.657 for random, 1.131 for greedy intermediate, 1.131 for greedy cost,
1.022 for per-query Q-learning, and 1.000 for exhaustive optimal over 40
queries. The held-out shared-Q result is 1.735 over 16 test queries. These are
reproducible simulator estimates from `results/comparison_summary.json`, not
PostgreSQL execution times; the weaker shared result is retained as an honest
generalization limitation.



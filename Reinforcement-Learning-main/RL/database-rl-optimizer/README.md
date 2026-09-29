# Reinforcement Learning for Database Query Plan Optimization

A college-level RL mini-project that uses tabular Q-learning to learn efficient JOIN ORDER selection for relational database queries.

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
│   └── evaluation.py       # Training and evaluation
│
├── tests/
│   ├── test_project.py     # Unit tests (Milestones 1-3)
│   ├── test_cost_model.py  # Unit tests (Milestone 4)
│   └── test_environment.py # Unit tests (Milestone 5)
│
└── results/
    └── plots/              # Experiment visualizations
```

## Running Tests

```bash
pytest tests/
```

(`gymnasium` is required for the environment tests; it is in `requirements.txt`.)

## Research Question

**Can a reinforcement learning agent learn effective join-ordering policies for relational query optimization, and how do those policies compare with heuristic strategies across query complexity, exploration settings, and training budgets?**

## Development Status

- [x] Milestone 1: Project setup and minimal structure
- [x] Milestone 2: Synthetic database generator
- [x] Milestone 3: Query and join graph representation
- [x] Milestone 4: Cardinality and cost model
- [x] Milestone 5: RL environment
- [ ] Milestone 6: Tests for environment and cost model
- [ ] Milestone 7: Random and greedy baselines
- [ ] Milestone 8: Manual tabular Q-learning
- [ ] Milestone 9: Small end-to-end training experiment
- [ ] Milestone 10: Full evaluation and metrics
- [ ] Milestone 11: Visualization and result logging
- [ ] Milestone 12: Sample-efficiency and exploration experiments
- [ ] Milestone 13: Scaling experiments
- [ ] Milestone 14: Optional SQLite validation

## Detailed Documentation

### Synthetic Database Generation (Milestone 2)

The project uses synthetic relational databases rather than real databases for several reasons:
- **Controlled experiments**: Known statistics enable reproducible results
- **Scalability**: Easy to generate databases of varying complexity
- **Educational focus**: Transparent cost models without real database complexity
- **Reproducibility**: Deterministic generation via random seeds

#### Schema Generation

Databases are generated with 3-6 tables representing common relational entities (Customer, Order, Product, etc.). Each generation creates:

1. **Table names**: Selected from a pool of common database entity names
2. **Connected relationships**: A tree structure ensures all tables are reachable via foreign keys
3. **Primary keys**: Every table has a unique identifier (e.g., `customer_id`)
4. **Foreign keys**: Child tables reference parent tables, defining valid join operations

#### Row Counts

Table sizes follow a logarithmic distribution to create realistic variation:
- Small tables: ~1,000 rows
- Medium tables: ~10,000-100,000 rows  
- Large tables: ~100,000-1,000,000 rows

This variation is critical because different join orders produce different intermediate result sizes, making the optimization problem meaningful.

#### Join Selectivity

Each foreign key relationship has an associated selectivity estimate representing the fraction of the Cartesian product that appears in the join result. Selectivities are generated with log-uniform distribution between 0.01% and 10%.

For example, if `Customers` has 10,000 rows and `Orders` has 100,000 rows with selectivity 0.01:
- Cartesian product: 10,000 × 100,000 = 1,000,000,000
- Estimated join result: 1,000,000,000 × 0.01 = 10,000,000 rows

#### Reproducibility

All generation is deterministic using explicit random seeds:
```python
db = generate_database(num_tables=4, seed=42)
```

The same seed always produces identical schemas, row counts, and selectivities. Different seeds produce different databases, enabling varied training and test sets.

#### Running the Generator

Demonstrate the generator:
```bash
python -m src.database
```

This generates a 4-table database with seed 42 and displays its structure.

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

Example:
```python
query.can_join("Customer", "Order")  # True (foreign key exists)
query.can_join("Customer", "Product")  # False (no direct relationship)
```

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

#### Canonical Representations

All table sets use `frozenset` for deterministic representation:
```python
{"Order", "Customer"} == {"Customer", "Order"}  # Same canonical form
```

This is critical for Q-learning because states must have stable hashable keys for the Q-table.

#### SQL Generation

The system can generate readable SQL (for debugging/visualization only):
```sql
SELECT *
FROM Customer
JOIN Order
    ON Customer.customer_id = Order.customer_id
JOIN OrderItem
    ON Order.order_id = OrderItem.order_id
;
```

**Important**: This is NOT an execution layer. The structured representation is the source of truth.

#### Running the Query Generator

Demonstrate query generation:
```bash
python -m src.query
```

This generates a database, creates a query, and demonstrates valid joins and partial plan construction.

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

```python
from src.database import generate_database
from src.query import generate_query
from src.environment import JoinOrderEnv

env = JoinOrderEnv(generate_query(generate_database(4, seed=42), seed=42))
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(info["valid_actions"][0])
```

Demonstration (no learning, just the interface): `python -m src.environment`.

#### MDP definition

| Element | Definition |
|---|---|
| Episode start | `create_initial_plan(query, start_table)`: one table joined. Default start is `query.tables[0]`; override with `JoinOrderEnv(..., start_table=)` or `reset(options={"start_table": name})`. |
| Action | "Join table *t* next" (left-deep plan). |
| Transition | Deterministic (below). |
| Reward | `-incremental_cost` from the cost model. |
| Termination | All tables joined, after exactly `n_tables - 1` valid steps. |

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

#### Observation representation

`spaces.Dict` with these features (`n` = number of tables, positions follow the
action ordering):

| Key | Type / range | Meaning |
|---|---|---|
| `joined` | `MultiBinary(n)`, int8 | 1 if the table is already in the plan (remaining tables = `1 - joined`) |
| `candidate_selectivity` | `Box(n,)` float32 in [0, 1] | For each legal next join: combined selectivity of all predicates linking it to the plan; 0.0 for illegal actions (real selectivities are > 0) |
| `log_cardinality` | `Box(1,)` float32 >= 0 | log10 of the estimated size of the current intermediate relation |
| `log_accumulated_cost` | `Box(1,)` float32 >= 0 | log10(1 + cost paid so far) |

Variable query size is handled by fixed-length vectors indexed by the action
ordering (queries have 3-6 tables). Upper bounds of the two `Box` features are
derived from the query's row counts (see module docstring).

**Tabular Q-learning:** the expected future cost of a state depends only on
which tables are joined, so `env.state_key()` returns the canonical
`frozenset` of joined table names (equivalent to `tuple(obs["joined"])`).
Accumulated cost is path-dependent and continuous, so it is deliberately *not*
part of the tabular key; the continuous features are for function
approximation and debugging.

#### Transition logic

`step(action)`:

1. Validate the action (integer id, in range, not joined, adjacent to the plan).
2. `CostModel.estimate_plan_join(plan, table)` -> output cardinality and cost.
3. `PartialJoinPlan.apply_join(table)` -> new partial plan.
4. Commit the new plan and add the step cost to the accumulated cost.
5. Compute the next mask and check completion.
6. Return `(observation, reward, terminated, truncated, info)`.

#### Reward

```
reward = -join_cost = -(|left intermediate| + |right table| + |output|)
```

No shaping. The undiscounted return of an episode equals `-plan_cost` from the
cost model, so maximizing return minimizes total estimated cost.

#### Terminal condition and invalid actions

| Situation | `terminated` | `truncated` | `info["status"]` |
|---|---|---|---|
| Valid step, tables remain | False | False | `ongoing` |
| Valid step completes the plan | True | False | `completed` |
| Invalid action | False | False | `invalid_action` (`info["invalid_action"]=True`, `info["invalid_reason"]`) |
| More than `max_invalid_actions` (default 100) invalid actions | False | True | `truncated` |

An invalid action leaves the plan, costs and mask untouched and returns reward
`-invalid_action_penalty` (default `0.0`, i.e. no shaping; use the mask instead).
Calling `step()` before `reset()` or after the episode ended raises `RuntimeError`.

#### Info dictionary

`status`, `invalid_action`, `invalid_reason`, `is_complete`, `selected_action`,
`selected_table`, `selected_join` (the `(joined_table, new_table)` pair recorded
in the plan), `applied_joins` (all predicates applied), `incremental_cost`,
`estimated_cardinality` (current intermediate), `left_cardinality`,
`accumulated_cost`, `start_table`, `partial_plan`, `joined_tables`,
`remaining_tables`, `valid_actions`, `action_mask`, `steps_taken`,
`invalid_actions_taken`, `explanation`. It describes only the present state and
the transition just taken (no lookahead costs or oracle information).

#### Determinism

The environment uses no randomness: the same query, start table and action
sequence always give identical observations, rewards and termination. The
`seed` argument of `reset()` is accepted for API compatibility only.

### Q-Learning Algorithm
*To be documented in Milestone 8*

### Cardinality Estimation

Implemented in `src/cost_model.py` (Milestone 4). The module only *reads* the
`Query`/`Database`; it never modifies them, and all results are deterministic.

**Two-relation join** (uses `Join.selectivity`, which comes from
`Database.join_selectivities`):

```
|R JOIN S| = |R| x |S| x selectivity(R, S)
```

**Joining a table T onto an intermediate result covering the table set S:**

```
|S JOIN T| = |S| x |T| x prod(selectivity(a, T) for every join edge (a, T), a in S)
```

Every foreign-key predicate connecting T to the already-joined tables is
applied (independence assumption). In tree-shaped schemas exactly one edge
connects T to S, so this is the two-relation formula. The database generator
can add one extra edge (a cycle); in that case both predicates filter the
result. (`PartialJoinPlan.apply_join` records only one partner per step, so the
cost model looks at all edges itself and `query.py` is left unchanged.)

**Closed form used by the code** (result depends only on *which* tables are
joined, never on the order, which keeps the future RL state well defined):

```
card(X) = prod(row_count(t) for t in X) x prod(selectivity(e) for every edge e inside X)
```

Estimates are floored at 1 row (`min_cardinality`, configurable, `0.0`
disables) so very selective joins never give 0 or fractional rows.

Example (README figures): `Customers` 10,000 rows, `Orders` 100,000 rows,
selectivity 0.01 -> 10,000 x 100,000 x 0.01 = 10,000,000 rows.

### Cost Model

A transparent, educational cost model (not PostgreSQL's optimizer):

```
join_cost = |left input| + |right input| + |estimated output|
plan_cost = sum of join_cost over all join steps made so far
```

The left input is the current intermediate relation, the right input is the
base table being added. A plan containing only its start table costs 0. The
output term charges for materializing intermediate results, which is what makes
join order matter.

API summary (`CostModel(query)`):

- `estimate_join(left_tables, right_table)` -> `JoinEstimate` (sizes, selectivity, output rows, cost)
- `estimate_plan_join(plan, candidate)` -> cost and resulting cardinality of adding `candidate` to a `PartialJoinPlan`
- `estimate_available_joins(plan)` -> estimate for every valid next join
- `current_cardinality(plan)`, `remaining_tables(plan)`, `plan_cost(plan)`
- Invalid joins (unknown table, already joined, no foreign key path, empty/disconnected input) raise `InvalidJoinError` (a `ValueError`)

Demonstration: `python -m src.cost_model`. Tests: `python -m pytest tests/test_cost_model.py`.

### Baselines
*To be documented in Milestone 7*

### Experimental Methodology
*To be documented in Milestone 10*

### Results
*To be documented after experiments*

### Limitations
*To be documented after implementation*

### Future Work

Possible extensions (not in initial scope):
- Deep Q-Networks (DQN)
- Policy gradient methods (PPO)
- Physical operator optimization
- Index selection
- Distributed databases
- Real PostgreSQL validation

## Scientific Integrity

This project adheres to strict scientific honesty:
- No fabricated results
- Clear distinction between estimated cost and actual execution time
- Transparent synthetic cost models (not claimed as real database internals)
- Honest reporting of unexpected results
- Reproducible experiments with fixed random seeds

## License

Academic project for educational purposes.

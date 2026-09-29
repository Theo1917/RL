"""
Gymnasium environment for sequential join-order optimization (Milestone 5).

An episode builds one complete left-deep join plan, one join at a time.

MDP
---
* Start of episode : a ``PartialJoinPlan`` holding a single start table
                     (``create_initial_plan``).
* Action           : "join table ``t`` next". Fixed action space
                     ``Discrete(n_tables)``; action ``i`` means the table at
                     position ``i`` of ``sorted(query.tables)``. A validity
                     mask says which actions are legal in the current state.
* Transition       : validate -> ``CostModel.estimate_plan_join`` ->
                     ``PartialJoinPlan.apply_join`` -> update totals.
* Reward           : ``-JoinEstimate.cost`` of the chosen join (negative
                     incremental cost from the Milestone 4 cost model). No
                     shaping. The undiscounted return of a full episode is
                     therefore ``-plan_cost``.
* Termination      : ``terminated=True`` when every table has been joined
                     (exactly ``n_tables - 1`` valid steps).
* Truncation       : only if an agent submits more than
                     ``max_invalid_actions`` invalid actions in one episode
                     (a safety net against endless invalid-action loops).

Observation (``gymnasium.spaces.Dict``; ``n`` = number of query tables,
positions follow the action ordering above)
-------------------------------------------
``joined``                MultiBinary(n), int8. 1 if the table is already part
                          of the plan.
``candidate_selectivity`` Box(n,), float32 in [0, 1]. For every table that is a
                          valid next join: the combined selectivity of all
                          predicates connecting it to the current plan. 0.0
                          for tables that are not valid next joins (real
                          selectivities are always > 0, so 0 is unambiguous).
``log_cardinality``       Box(1,), float32 in [0, U]. log10 of the estimated size
                          of the current intermediate relation.
``log_accumulated_cost``  Box(1,), float32 in [0, V]. log10(1 + cost paid so far).

Upper bounds are derived from the query: with P = product of all table row
counts, every cardinality is <= P (selectivities are <= 1), so
U = log10(P) + 1 and, since a step costs at most 3P, V = log10(3 (n-1) P) + 1.
(V assumes the default Milestone 4 cost formula.)

The remaining tables are ``1 - joined``. The action mask is derivable from
the observation and is also provided in ``info["action_mask"]``.

Tabular Q-learning: the estimated future cost of a state depends only on the
set of joined tables, so ``state_key()`` returns that canonical
``frozenset`` of table names (equivalently ``tuple(obs["joined"])``).
Continuous features are for function approximation and debugging and should
not be part of a tabular key.

Determinism: the environment contains no randomness. The same query, start
table and action sequence always give identical observations, rewards and
termination.
"""

from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.cost_model import CostModel, InvalidJoinError, JoinEstimate
from src.query import PartialJoinPlan, Query, create_initial_plan

# Values reported in info["status"].
STATUS_ONGOING = "ongoing"
STATUS_COMPLETED = "completed"
STATUS_INVALID_ACTION = "invalid_action"
STATUS_TRUNCATED = "truncated"


class JoinOrderEnv(gym.Env):
    """
    Gymnasium environment: choose the next table to join, one step at a time.

    Args:
        query: The query (join graph + database statistics) to plan.
        cost_model: Optional pre-built ``CostModel`` for ``query``. A default
            one is created when omitted.
        start_table: Table the plan starts from. Defaults to
            ``query.tables[0]`` (same default as ``create_initial_plan``).
            Can be overridden per episode via ``reset(options=...)``.
        invalid_action_penalty: Reward returned for an invalid action
            (subtracted, so pass a non-negative number). Default 0.0: an
            invalid action changes nothing and is not rewarded or punished.
            Prefer the action mask over penalties.
        max_invalid_actions: Truncate the episode after more than this many
            invalid actions.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 query: Query,
                 cost_model: Optional[CostModel] = None,
                 start_table: Optional[str] = None,
                 invalid_action_penalty: float = 0.0,
                 max_invalid_actions: int = 100) -> None:
        super().__init__()
        if cost_model is not None and cost_model.query is not query:
            raise ValueError("cost_model must be built for the same query")
        if invalid_action_penalty < 0:
            raise ValueError("invalid_action_penalty must be >= 0")
        if max_invalid_actions < 0:
            raise ValueError("max_invalid_actions must be >= 0")

        self.query = query
        self.cost_model = cost_model if cost_model is not None else CostModel(query)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.max_invalid_actions = int(max_invalid_actions)

        # Fixed, order-independent action ordering.
        self._tables: List[str] = sorted(query.tables)
        self._action_of: Dict[str, int] = {t: i for i, t in enumerate(self._tables)}
        self.n_tables = len(self._tables)

        self._default_start = start_table if start_table is not None else query.tables[0]
        if self._default_start not in self._action_of:
            raise ValueError(f"Start table {self._default_start} not in query")

        # Finite, query-derived bounds (see module docstring).
        log_p = float(sum(np.log10(query.database.get_table(t).row_count)
                          for t in self._tables))
        card_high = log_p + 1.0
        cost_high = float(np.log10(3.0 * max(self.n_tables - 1, 1))) + log_p + 1.0

        self.action_space = spaces.Discrete(self.n_tables)
        self.observation_space = spaces.Dict({
            "joined": spaces.MultiBinary(self.n_tables),
            "candidate_selectivity": spaces.Box(
                0.0, 1.0, shape=(self.n_tables,), dtype=np.float32),
            "log_cardinality": spaces.Box(0.0, card_high, shape=(1,), dtype=np.float32),
            "log_accumulated_cost": spaces.Box(0.0, cost_high, shape=(1,), dtype=np.float32),
        })

        # Episode state (populated by reset()).
        self._plan: Optional[PartialJoinPlan] = None
        self._start_table: Optional[str] = None
        self._accumulated_cost = 0.0
        self._steps = 0
        self._invalid_count = 0
        self._done = False

    # ------------------------------------------------------------------
    # Action encoding
    # ------------------------------------------------------------------

    @property
    def action_table_names(self) -> Tuple[str, ...]:
        """Table for each action id: ``action_table_names[i]`` is action i."""
        return tuple(self._tables)

    def action_to_table(self, action: int) -> str:
        """Map an action id to the table it joins next."""
        if not self._is_action_id(action) or not 0 <= int(action) < self.n_tables:
            raise ValueError(f"Action {action!r} is outside [0, {self.n_tables})")
        return self._tables[int(action)]

    def table_to_action(self, table: str) -> int:
        """Map a table name to its action id."""
        if table not in self._action_of:
            raise ValueError(f"Table {table} not in query")
        return self._action_of[table]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def plan(self) -> PartialJoinPlan:
        """Current partial plan (treat as read-only)."""
        self._require_reset()
        return self._plan

    @property
    def accumulated_cost(self) -> float:
        """Total cost paid so far in this episode."""
        return self._accumulated_cost

    @property
    def is_complete(self) -> bool:
        """True when a complete join plan has been built."""
        self._require_reset()
        return self._plan.is_complete()

    def state_key(self) -> FrozenSet[str]:
        """Canonical hashable state for tabular methods: the joined-table set."""
        self._require_reset()
        return frozenset(self._plan.joined_tables)

    def valid_action_mask(self) -> np.ndarray:
        """int8 vector of length ``n_tables``; 1 = legal action, 0 = illegal.

        Legal means: the table is not joined yet and shares a join predicate
        with the current plan (``PartialJoinPlan.get_available_joins``).
        All zeros once the plan is complete.
        """
        self._require_reset()
        mask = np.zeros(self.n_tables, dtype=np.int8)
        for table in self._plan.get_available_joins():
            mask[self._action_of[table]] = 1
        return mask

    def valid_actions(self) -> List[int]:
        """Ids of all legal actions, ascending."""
        return [int(i) for i in np.flatnonzero(self.valid_action_mask())]

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Start a new episode.

        Args:
            seed: Accepted for API compatibility (the environment is fully
                deterministic and uses no randomness).
            options: Optional dict; ``{"start_table": name}`` overrides the
                default start table for this episode only.

        Returns:
            ``(observation, info)``.
        """
        super().reset(seed=seed)
        start = self._default_start
        if options and options.get("start_table") is not None:
            start = options["start_table"]
            if start not in self._action_of:
                raise ValueError(f"Start table {start} not in query")

        self._plan = create_initial_plan(self.query, start)
        self._start_table = start
        self._accumulated_cost = 0.0
        self._steps = 0
        self._invalid_count = 0
        self._done = False
        return self._observation(), self._info(status=STATUS_ONGOING)

    def step(self, action: Any
             ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Join the table selected by ``action``.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
            An invalid action leaves the state untouched, returns reward
            ``-invalid_action_penalty``, and sets ``info["invalid_action"]``.

        Raises:
            RuntimeError: If called before ``reset()`` or after the episode
                has ended (terminated or truncated).
        """
        self._require_reset()
        if self._done:
            raise RuntimeError("Episode has ended; call reset() before step()")

        reason = self._invalid_reason(action)
        if reason is not None:
            self._invalid_count += 1
            truncated = self._invalid_count > self.max_invalid_actions
            if truncated:
                self._done = True
            status = STATUS_TRUNCATED if truncated else STATUS_INVALID_ACTION
            info = self._info(status=status, invalid_action=True,
                              invalid_reason=reason, selected_action=self._echo(action))
            return self._observation(), -self.invalid_action_penalty, False, truncated, info

        table = self._tables[int(action)]
        estimate: JoinEstimate = self.cost_model.estimate_plan_join(self._plan, table)
        new_plan = self._plan.apply_join(table)

        # Commit only after everything above succeeded.
        self._plan = new_plan
        self._accumulated_cost += estimate.cost
        self._steps += 1
        terminated = new_plan.is_complete()
        if terminated:
            self._done = True

        info = self._info(
            status=STATUS_COMPLETED if terminated else STATUS_ONGOING,
            selected_action=int(action),
            selected_table=table,
            estimate=estimate,
        )
        return self._observation(), -estimate.cost, terminated, False, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _is_action_id(action: Any) -> bool:
        return isinstance(action, (int, np.integer)) and not isinstance(action, (bool, np.bool_))

    @staticmethod
    def _echo(action: Any) -> Any:
        return int(action) if JoinOrderEnv._is_action_id(action) else repr(action)

    def _require_reset(self) -> None:
        if self._plan is None:
            raise RuntimeError("Call reset() before using the environment")

    def _invalid_reason(self, action: Any) -> Optional[str]:
        """Return why ``action`` is illegal, or None if it is legal."""
        if not self._is_action_id(action):
            return f"action must be an integer id, got {action!r}"
        a = int(action)
        if not 0 <= a < self.n_tables:
            return f"action {a} out of range [0, {self.n_tables})"
        table = self._tables[a]
        if table in self._plan.joined_tables:
            return f"table {table} is already joined"
        if table not in self._plan.get_available_joins():
            return f"table {table} has no join predicate with the current plan"
        return None

    def _observation(self) -> Dict[str, np.ndarray]:
        plan = self._plan
        joined = np.zeros(self.n_tables, dtype=np.int8)
        for t in plan.joined_tables:
            joined[self._action_of[t]] = 1

        cand = np.zeros(self.n_tables, dtype=np.float32)
        for t in plan.get_available_joins():
            try:
                cand[self._action_of[t]] = self.cost_model.estimate_plan_join(
                    plan, t).selectivity
            except InvalidJoinError:      # cannot happen for available joins
                pass

        card = self.cost_model.current_cardinality(plan)
        return {
            "joined": joined,
            "candidate_selectivity": cand,
            "log_cardinality": np.array([np.log10(card)], dtype=np.float32),
            "log_accumulated_cost": np.array(
                [np.log10(1.0 + self._accumulated_cost)], dtype=np.float32),
        }

    def _info(self, status: str, invalid_action: bool = False,
              invalid_reason: Optional[str] = None,
              selected_action: Optional[Any] = None,
              selected_table: Optional[str] = None,
              estimate: Optional[JoinEstimate] = None) -> Dict[str, Any]:
        plan = self._plan
        mask = self.valid_action_mask()
        info: Dict[str, Any] = {
            "status": status,
            "invalid_action": invalid_action,
            "invalid_reason": invalid_reason,
            "is_complete": plan.is_complete(),
            "selected_action": selected_action,
            "selected_table": selected_table,
            "selected_join": None,
            "applied_joins": (),
            "incremental_cost": 0.0,
            "estimated_cardinality": self.cost_model.current_cardinality(plan),
            "left_cardinality": None,
            "accumulated_cost": self._accumulated_cost,
            "start_table": self._start_table,
            "partial_plan": list(plan.join_sequence),
            "joined_tables": sorted(plan.joined_tables),
            "remaining_tables": sorted(plan.remaining_tables),
            "valid_actions": [int(i) for i in np.flatnonzero(mask)],
            "action_mask": mask,
            "steps_taken": self._steps,
            "invalid_actions_taken": self._invalid_count,
        }
        if estimate is not None:
            info["selected_join"] = plan.join_sequence[-1]
            info["applied_joins"] = estimate.applied_joins
            info["incremental_cost"] = estimate.cost
            info["left_cardinality"] = estimate.left_cardinality
            info["explanation"] = estimate.explain()
        return info


def main() -> None:
    """Demonstration: walk one episode choosing the first legal action.

    No learning and no policy: just shows the interface.
    """
    from src.database import generate_database
    from src.query import generate_query

    print("=" * 70)
    print("MILESTONE 5 DEMONSTRATION")
    print("Join-order RL environment")
    print("=" * 70)
    env = JoinOrderEnv(generate_query(generate_database(4, seed=42), seed=42))
    obs, info = env.reset()
    print("Actions:", dict(enumerate(env.action_table_names)))
    print("Start:", info["start_table"], "| valid actions:", info["valid_actions"])
    done, total = False, 0.0
    while not done:
        action = info["valid_actions"][0]
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
        print(f"  action {action} -> {info['selected_table']:<10} "
              f"reward {reward:>14,.2f}  rows {info['estimated_cardinality']:>14,.0f}  "
              f"status {info['status']}")
    print(f"Return {total:,.2f} == -plan_cost {-env.cost_model.plan_cost(env.plan):,.2f}")


if __name__ == "__main__":
    main()

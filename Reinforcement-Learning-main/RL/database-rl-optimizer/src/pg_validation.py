"""Optional PostgreSQL validation adapter.

The simulator remains usable without PostgreSQL.  Set PGHOST, PGPORT, PGUSER,
PGPASSWORD, and PGDATABASE (or DATABASE_URL) before running this module.
"""

import os
from pathlib import Path
from statistics import median
import json
import random


def connection_config():
    return {
        "host": os.getenv("PGHOST", "localhost"),
        "port": int(os.getenv("PGPORT", "5432")),
        "user": os.getenv("PGUSER", "postgres"),
        "password": os.getenv("PGPASSWORD", "postgres"),
        "dbname": os.getenv("PGDATABASE", "postgres"),
    }


def connect():
    try:
        import psycopg2
    except ImportError:
        return None, "psycopg2 is not installed"
    try:
        return psycopg2.connect(os.getenv("DATABASE_URL"), **connection_config()) \
            if os.getenv("DATABASE_URL") else psycopg2.connect(**connection_config()), None
    except Exception as exc:
        return None, str(exc)


def _plan_tables(query, plan_or_order, start_table=None):
    if hasattr(plan_or_order, "join_sequence"):
        order = list(plan_or_order.join_sequence)
        added = {right for _, right in order}
        start = start_table or next(
            table for table in plan_or_order.joined_tables if table not in added)
        return start, [right for _, right in order]
    order = list(plan_or_order)
    if start_table is not None:
        return start_table, [right for _, right in order]
    if not order:
        return query.tables[0], []
    return order[0][0], [right for _, right in order]


def join_order_sql(query, plan_or_order, start_table=None, force_order=True):
    """Build explicit JOIN SQL in the exact left-deep plan order."""
    start, additions = _plan_tables(query, plan_or_order, start_table)
    sql = []
    if force_order:
        sql.extend(["SET join_collapse_limit = 1;",
                    "SET from_collapse_limit = 1;"])
    sql.extend(["SELECT *", f"FROM {start}"])
    joined = {start}
    for candidate in additions:
        join = next((item for table in joined for item in query.joins
                     if item.involves_tables(candidate, table)), None)
        if join is None:
            raise ValueError(f"No join predicate connects {candidate} to {joined}")
        sql.append(f"JOIN {candidate} ON {join}")
        joined.add(candidate)
    sql.append(";")
    return "\n".join(sql)


def parse_explain_plan(payload):
    """Extract stable metrics from PostgreSQL FORMAT JSON output."""
    root = payload[0] if isinstance(payload, list) else payload
    plan = root.get("Plan", root)
    peak_rows, memory, shared_buffers = 0.0, 0.0, 0.0

    def walk(node):
        nonlocal peak_rows, memory, shared_buffers
        peak_rows = max(peak_rows, float(node.get("Actual Rows", 0.0)))
        memory = max(memory, float(node.get("Memory Usage", 0.0)))
        shared_buffers += float(node.get("Shared Hit Blocks", 0.0))
        shared_buffers += float(node.get("Shared Read Blocks", 0.0))
        for child in node.get("Plans", []):
            walk(child)

    walk(plan)
    return {"planning_time_ms": root.get("Planning Time"),
            "execution_time_ms": root.get("Execution Time"),
            "total_cost": plan.get("Total Cost"),
            "peak_actual_rows": peak_rows, "memory_usage": memory,
            "shared_buffers": shared_buffers}


def explain_runs(connection, sql, runs=5):
    """Return parsed JSON plans after one warm-up execution."""
    cursor = connection.cursor()
    cursor.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql)
    cursor.fetchall()
    outputs = []
    for _ in range(max(5, runs)):
        cursor.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql)
        outputs.append(cursor.fetchone()[0][0])
    connection.commit()
    parsed = [parse_explain_plan(item) for item in outputs]
    execution = [item.get("execution_time_ms", 0.0) or 0.0 for item in parsed]
    return {"median_execution_time_ms": median(execution),
            "median_metrics": {key: median([item[key] for item in parsed
                                              if item[key] is not None])
                               for key in parsed[0] if key.endswith("_ms") or
                               key in ("total_cost", "peak_actual_rows",
                                       "memory_usage", "shared_buffers")},
            "runs": outputs}


def create_schema(connection, database, scale_factor=0.01):
    """Materialize scaled synthetic tables, keys, indexes, and ANALYZE."""
    cursor = connection.cursor()
    for table in database.tables:
        name = table.name.lower()
        cursor.execute(f'DROP TABLE IF EXISTS "{name}" CASCADE')
    for table in database.tables:
        name = table.name.lower()
        columns = [f'"{column.name}" INTEGER' for column in table.columns]
        columns.append(f'PRIMARY KEY ("{table.primary_key}")')
        for foreign_key in table.foreign_keys:
            columns.append(
                f'FOREIGN KEY ("{foreign_key.column}") REFERENCES '
                f'"{foreign_key.references_table.lower()}" '
                f'("{foreign_key.references_column}")')
        cursor.execute(f'CREATE TABLE "{name}" ({", ".join(columns)})')
        for column in table.indexed_columns:
            cursor.execute(f'CREATE INDEX ON "{name}" ("{column}")')
    rng = random.Random(database.seed)
    scaled_counts = {table.name: max(1, int(table.row_count * scale_factor))
                     for table in database.tables}
    for table in database.tables:
        name = table.name.lower()
        count = scaled_counts[table.name]
        columns = [column.name for column in table.columns]
        for row_id in range(1, count + 1):
            values = []
            foreign_columns = {foreign_key.column: foreign_key.references_table
                               for foreign_key in table.foreign_keys}
            for column in columns:
                if column == table.primary_key:
                    values.append(row_id)
                elif column in foreign_columns:
                    values.append(rng.randint(1, scaled_counts[foreign_columns[column]]))
                else:
                    values.append(rng.randint(1, count))
            placeholders = ", ".join(["%s"] * len(values))
            cursor.execute(f'INSERT INTO "{name}" ({", ".join(columns)}) '
                           f'VALUES ({placeholders})', values)
        cursor.execute(f'ANALYZE "{name}"')
    connection.commit()


def compare_plans(connection, query, named_plans, output_dir="results/pg"):
    """Run forced plans and the PostgreSQL default planner, saving raw JSON."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, plan in named_plans.items():
        sql = join_order_sql(query, plan, force_order=name != "postgres_default")
        results[name] = explain_runs(connection, sql, runs=5)
        (output / f"{name}.json").write_text(json.dumps(results[name], indent=2,
                                                        default=str))
    (output / "summary.json").write_text(json.dumps(results, indent=2,
                                                     default=str))
    return results


def spearman_rank_correlation(left, right):
    """Compute Spearman correlation without adding scipy as a dependency."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("rank inputs must have equal length >= 2")

    def ranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        output = [0.0] * len(values)
        for rank, index in enumerate(order):
            output[index] = rank + 1
        return output

    a, b = ranks(left), ranks(right)
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = (sum((x - mean_a) ** 2 for x in a) *
                  sum((y - mean_b) ** 2 for y in b)) ** 0.5
    return numerator / denominator if denominator else 0.0


def main():
    connection, error = connect()
    if connection is None:
        print(f"PostgreSQL validation skipped: {error}")
        return 0
    try:
        print("PostgreSQL connection is available; use explain_runs for a query.")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
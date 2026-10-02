"""Optional PostgreSQL validation adapter.

The simulator remains usable without PostgreSQL.  Set PGHOST, PGPORT, PGUSER,
PGPASSWORD, and PGDATABASE (or DATABASE_URL) before running this module.
"""

import json
import os
from pathlib import Path
from statistics import median


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


def join_order_sql(query, order):
    """Build explicit JOIN SQL; execution-order forcing is caller-controlled."""
    tables = [query.tables[0]] if not order else [order[0][0], order[0][1]]
    sql = ["SELECT *", f"FROM {tables[0]}"]
    joined = {tables[0]}
    if len(tables) > 1:
        joined.add(tables[1])
        join = query.get_join(tables[0], tables[1])
        sql.append(f"JOIN {tables[1]} ON {join}")
    for left, right in order[1:]:
        candidate = right if right not in joined else left
        join = query.get_join(left, right)
        sql.append(f"JOIN {candidate} ON {join}")
        joined.add(candidate)
    return "\n".join(sql)


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
    execution = [item.get("Execution Time", 0.0) for item in outputs]
    return {"median_execution_time_ms": median(execution), "runs": outputs}


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
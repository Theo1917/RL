# Demo

1. Install `requirements.txt` and run `python demo.py --seed 42`.
2. Use `python demo.py --seed 42 --no-postgres` when PostgreSQL is unavailable.
3. Inspect `results/plots/demo_comparison.png` and the printed MDP trace.
4. For optional database validation, start `docker compose up -d` and set the
   PostgreSQL environment variables before running `python -m src.pg_validation`.

The experiment uses fixed seeds and reports estimated simulator cost. It does
not claim measured execution-time results when PostgreSQL is unavailable.

## Browser Showcase

Run `python -m webapp.server --port 8000` and open
`http://127.0.0.1:8000`. The JoinLab interface calls the project’s real Python
API, renders the generated schema and join graph, replays the policy trace,
plots training history, and compares all five planning policies.
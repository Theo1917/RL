"""Tiny full-stack web server for the join-order demo."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import (ExhaustiveOptimal, GreedyMinImmediateCost,
                           GreedyMinIntermediate, RandomAgent, run_episode)
from src.database import generate_database
from src.environment import JoinOrderEnv
from src.q_learning import QLearningAgent
from src.query import generate_query


def build_payload(seed=42, tables=4, episodes=500):
    database = generate_database(tables, seed=seed, indexed=True)
    query = generate_query(database, seed=seed)
    env = JoinOrderEnv(query, start_mode="agent")
    learner = QLearningAgent(episodes=max(1, min(int(episodes), 2000)),
                             epsilon_schedule="exponential",
                             epsilon_decay=0.97, seed=seed)
    history = learner.train(env)
    learner.epsilon = 0.0
    optimal = ExhaustiveOptimal(env)
    trace_env = JoinOrderEnv(query, start_mode="agent")
    trace_env.reset()
    trace = []
    while True:
        state = sorted(trace_env.state_key())
        valid = trace_env.valid_actions()
        q_values = {str(action): round(learner._value(trace_env.state_key(), action), 2)
                    for action in valid}
        action = learner.choose_action(trace_env, greedy=True)
        _, reward, terminated, truncated, info = trace_env.step(action)
        trace.append({"state": state, "valid": valid, "q_values": q_values,
                      "action": action, "table": info.get("selected_table"),
                      "reward": reward, "rows": info.get("estimated_cardinality")})
        if terminated or truncated:
            break
    agents = {"Random": RandomAgent(seed),
              "Greedy intermediate": GreedyMinIntermediate(),
              "Greedy cost": GreedyMinImmediateCost(),
              "Q-learning": learner, "Exhaustive optimum": optimal}
    results = []
    for name, agent in agents.items():
        plan, cost, _ = run_episode(env, agent)
        results.append({"name": name, "cost": cost,
                        "ratio": cost / optimal.best_cost,
                        "plan": list(plan.join_sequence)})
    return {"seed": seed, "tables": tables,
            "schema": [{"name": t.name, "rows": t.row_count,
                        "indexes": t.indexed_columns} for t in database.tables],
            "joins": [{"text": str(j), "selectivity": j.selectivity} for j in query.joins],
            "sql": query.to_sql(), "history": history, "trace": trace,
            "results": results}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, content_type, body):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send(200, "application/json", b'{"status":"ok"}')
        elif path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", (WEB / "index.html").read_bytes())
        elif path in ("/app.js", "/styles.css"):
            asset = WEB / path.lstrip("/")
            content_type = "text/javascript" if asset.suffix == ".js" else "text/css"
            self._send(200, content_type + "; charset=utf-8", asset.read_bytes())
        else:
            self._send(404, "application/json", b'{"error":"not found"}')

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            self._send(404, "application/json", b'{"error":"not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = build_payload(int(payload.get("seed", 42)),
                                   int(payload.get("tables", 4)),
                                   int(payload.get("episodes", 500)))
            self._send(200, "application/json", json.dumps(result).encode())
        except Exception as exc:
            self._send(400, "application/json", json.dumps({"error": str(exc)}).encode())

    def log_message(self, format, *args):
        return


def main(host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"JoinLab running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(args.host, args.port)
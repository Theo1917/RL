from webapp.server import build_payload


def test_web_payload_uses_real_pipeline():
    payload = build_payload(seed=2, tables=3, episodes=2)
    assert len(payload["schema"]) == 3
    assert payload["trace"]
    assert {result["name"] for result in payload["results"]} == {
        "Random", "Greedy intermediate", "Greedy cost", "Q-learning",
        "Exhaustive optimum",
    }
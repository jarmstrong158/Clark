"""End-to-end test for the minimal local inference API.

Hits the real routes -> the real Clark inference path (the same
_run_one_plan_day the `clark plan` CLI uses, pinned by
test_plan_path.py). Uses an untrained ClarkAgent (no checkpoint) so the
test is fast and self-contained — exactly the pattern test_plan_path.py
uses. This closes the entrypoint test gap (con-012): every user-facing
surface gets an e2e test that exercises its real call path, not mocks.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from clark.agent.ppo import ClarkAgent
from clark.serve.app import build_app

CONFIGS = Path(__file__).resolve().parents[1] / "clark" / "data" / "configs"
FID = "example_small"


def _client():
    agent = ClarkAgent(device="cpu", use_amp=False)   # untrained, fast
    return TestClient(build_app(agent, CONFIGS, checkpoint_label="test"))


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["ready"] is True


def test_facilities_lists_shipped_configs():
    r = _client().get("/facilities")
    assert r.status_code == 200
    assert FID in r.json()["facilities"]


def test_facility_returns_config():
    r = _client().get(f"/facility/{FID}")
    assert r.status_code == 200
    cfg = r.json()["config"]
    assert "workers" in cfg and len(cfg["workers"]) >= 1


def test_facility_rejects_traversal():
    r = _client().get("/facility/..%2f..%2fsecret")
    assert r.status_code in (400, 404)


def test_plan_runs_real_inference():
    from clark.config.schema import FacilityConfig
    n_workers = len(FacilityConfig.from_yaml(CONFIGS / f"{FID}.yaml").workers)

    r = _client().post("/plan", json={"facility_id": FID, "date": "2026-04-01"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["facility_id"] == FID
    a = body["assignments"]
    assert len(a) == n_workers
    for row in a:
        assert set(row) == {"worker", "task", "hustle"}
        assert isinstance(row["hustle"], bool)


def test_what_if_returns_base_and_modified():
    r = _client().post("/what_if", json={
        "facility_id": FID,
        "date": "2026-04-01",
        "volume": 999,
        "absent_workers": [],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "base" in body and "modified" in body
    assert len(body["base"]) == len(body["modified"]) >= 1


def test_seed_makes_plan_deterministic():
    """Audit finding: plans were RNG-noisy. With a seed, identical."""
    c = _client()
    j = {"facility_id": FID, "date": "2026-04-01", "seed": 123}
    a1 = c.post("/plan", json=j).json()["assignments"]
    a2 = c.post("/plan", json=j).json()["assignments"]
    assert a1 == a2, "same seed must give an identical plan"


def test_what_if_absent_worker_actually_removed():
    """Audit CRITICAL: absent_workers was a silent no-op. A named
    absent worker must show task 'absent' in modified and NOT in base,
    with base/modified otherwise the same (shared seed)."""
    from clark.config.schema import FacilityConfig
    cfg = FacilityConfig.from_yaml(CONFIGS / f"{FID}.yaml")
    victim = cfg.workers[0].name

    r = _client().post("/what_if", json={
        "facility_id": FID, "date": "2026-04-01",
        "absent_workers": [victim],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    base = {x["worker"]: x for x in body["base"]}
    mod = {x["worker"]: x for x in body["modified"]}

    assert mod[victim]["task"] == "absent", (
        f"{victim} forced absent but modified shows "
        f"{mod[victim]['task']!r} — absent_workers is still a no-op"
    )
    assert base[victim]["task"] != "absent", (
        "base must NOT have the worker absent — else not a real contrast"
    )


def test_unknown_facility_404():
    r = _client().post("/plan", json={"facility_id": "does_not_exist"})
    assert r.status_code == 404

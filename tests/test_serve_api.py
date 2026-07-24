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

# No importorskip: fastapi/httpx are hard test dependencies (the `dev`
# extra pulls in `clark[serve]`). A missing dep must fail loudly here, not
# quietly delete the serve layer's coverage. See tests/test_optional_deps.py.
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
    facs = r.json()["facilities"]
    assert FID in facs
    # standard_vocab.yaml is a task-vocabulary reference doc, NOT a
    # plannable facility — it must never be advertised (it used to be,
    # and /plan on it 500'd, polluting the fine-tune dataset).
    assert "standard_vocab" not in facs


def test_non_facility_config_excluded_and_4xx():
    """standard_vocab.yaml loads as YAML but is not a plannable
    facility. Every route must give a clean 4xx, never an unhandled
    500."""
    c = _client()
    rf = c.get("/facility/standard_vocab")
    assert rf.status_code == 422, rf.text
    rp = c.post("/plan", json={"facility_id": "standard_vocab"})
    assert rp.status_code == 422, rp.text
    rw = c.post("/what_if", json={
        "facility_id": "standard_vocab", "absent_workers": []})
    assert rw.status_code == 422, rw.text


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


def test_concurrent_plan_safe():
    """The agent has a mutable LSTM hidden state and is reset per
    request. FastAPI runs sync handlers on a threadpool — without the
    agent_lock added in build_app, overlapping /plan requests would
    race through the same hidden state and either crash or produce
    nondeterministic output.

    Pin the fix: fire many concurrent identical /plan calls and
    require all to return 200 with the same well-formed shape (no
    exceptions, no shape drift). This wouldn't *guarantee* races
    are gone (concurrency tests rarely do) but it does fail loudly
    if any request errors under contention."""
    from concurrent.futures import ThreadPoolExecutor

    client = _client()
    body = {"facility_id": FID, "date": "2026-04-01", "seed": 7}
    n = 8

    def go(_):
        r = client.post("/plan", json=body)
        return r.status_code, r.json() if r.status_code == 200 else None

    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(go, range(n)))

    for code, body_ in results:
        assert code == 200, f"concurrent /plan returned {code}"
        assert body_ and "assignments" in body_
        assert isinstance(body_["assignments"], list)
        for row in body_["assignments"]:
            assert set(row) == {"worker", "task", "hustle"}


def test_simulate_capacity_lookup_and_capabilities():
    """Smoke /simulate (small n_days, no extras) and /capabilities.
    Pins both routes against the actual served app — both were added
    after the original test suite was written."""
    client = _client()
    r = client.get("/capabilities")
    assert r.status_code == 200
    cap = r.json()
    assert "limits" in cap and "n_workers" in cap["limits"]
    assert cap["limits"]["n_workers"]["max"] >= 2

    r = client.post("/simulate", json={
        "facility_id": FID, "extra_workers": 0,
        "n_days": 5, "seed": 1})
    assert r.status_code == 200, r.text
    s = r.json()["summary"]
    # 5 days run — should have a grade distribution with the right keys
    grades = s.get("grade_distribution", {})
    assert set(grades.keys()).issubset({"A", "B", "C", "D", "F"})
    assert sum(grades.values()) <= 5

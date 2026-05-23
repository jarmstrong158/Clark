"""Tests for the per-facility checkpoint auto-load in clark/serve/app.py.

When a request comes in for facility X, the server checks for
`<per_facility_ckpt_dir>/X.pt` and lazily loads + caches that agent
for X — otherwise falls back to the foundation agent it was
constructed with. Without coverage, this silently regressed in two
plausible ways:

1. Loader exception swallowed the wrong way → request 500s instead of
   falling back.
2. Cache key collision or sentinel handling wrong → wrong-agent leak
   between facilities.

These tests pin both. They use the real ClarkAgent untrained
(checkpoint dir is empty by default) so the path is exercised end-to-
end through the FastAPI route, not a mock of build_app's internals.
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


@pytest.fixture
def empty_ckpt_dir(tmp_path):
    d = tmp_path / "user_ckpts"
    d.mkdir()
    return d


@pytest.fixture
def foundation_agent():
    return ClarkAgent(device="cpu", use_amp=False)


def _client(agent, ckpt_dir):
    return TestClient(build_app(agent, CONFIGS, checkpoint_label="test",
                                per_facility_ckpt_dir=ckpt_dir))


# ── No per-facility checkpoint exists → foundation ─────────────────

def test_plan_reports_foundation_when_no_per_facility_ckpt(
        foundation_agent, empty_ckpt_dir):
    c = _client(foundation_agent, empty_ckpt_dir)
    r = c.post("/plan", json={"facility_id": FID, "seed": 1})
    assert r.status_code == 200
    assert r.json()["model"] == "foundation"


def test_health_lists_no_fine_tuned_facilities_when_dir_empty(
        foundation_agent, empty_ckpt_dir):
    c = _client(foundation_agent, empty_ckpt_dir)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["per_facility_ckpt_dir"] == str(empty_ckpt_dir)
    assert body["fine_tuned_facilities"] == []


# ── A per-facility checkpoint exists → loaded and reported ─────────

def test_plan_reports_fine_tuned_when_ckpt_exists(
        foundation_agent, empty_ckpt_dir, tmp_path):
    # Save a second (different) untrained agent to disk as the
    # "fine-tuned" checkpoint. ClarkAgent.save needs a FacilityConfig.
    from clark.config.schema import FacilityConfig
    cfg = FacilityConfig.from_yaml(CONFIGS / f"{FID}.yaml")
    ft = ClarkAgent(device="cpu", use_amp=False)
    ft.save(str(empty_ckpt_dir / f"{FID}.pt"), episode=0, config=cfg)

    c = _client(foundation_agent, empty_ckpt_dir)
    r = c.post("/plan", json={"facility_id": FID, "seed": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fine-tuned"
    # Health should now advertise the available fine-tuned facility.
    h = c.get("/health").json()
    assert FID in h["fine_tuned_facilities"]


def test_per_facility_load_failure_falls_back_to_foundation(
        foundation_agent, empty_ckpt_dir):
    # A garbage file at <fid>.pt should NOT 500 the request — the
    # loader is wrapped in try/except and falls back to foundation
    # with a printed warning. This is the resilience the production
    # bug-class needs.
    (empty_ckpt_dir / f"{FID}.pt").write_bytes(b"not a real torch checkpoint")
    c = _client(foundation_agent, empty_ckpt_dir)
    r = c.post("/plan", json={"facility_id": FID, "seed": 1})
    assert r.status_code == 200
    assert r.json()["model"] == "foundation"


# ── Cache behaviour ────────────────────────────────────────────────

def test_per_facility_cache_does_not_leak_across_facilities(
        foundation_agent, empty_ckpt_dir):
    """Two facilities with NO per-facility ckpt should both report
    'foundation'. Catches a regression where the cache sentinel for
    'no fine-tune available' got confused with 'load me'."""
    # Use two real existing facilities from data/configs.
    other = "example_medium"
    assert (CONFIGS / f"{other}.yaml").exists(), \
        "test assumes two configs are present"
    c = _client(foundation_agent, empty_ckpt_dir)
    r1 = c.post("/plan", json={"facility_id": FID, "seed": 1})
    r2 = c.post("/plan", json={"facility_id": other, "seed": 1})
    assert r1.json()["model"] == "foundation"
    assert r2.json()["model"] == "foundation"

"""Pin /compare and /calendar_check — the two new routes that make
the chat capable of (a) cross-facility ranking and (b) refusing to
silently plan a fake Saturday.

End-to-end through FastAPI's TestClient so the agent.py wrapper has
something stable to call against."""
from __future__ import annotations

from pathlib import Path

import pytest

# No importorskip: fastapi is a hard test dependency (the `dev` extra
# pulls in `clark[serve]`). A missing dep must fail loudly here, not
# quietly delete the serve layer's coverage. See tests/test_optional_deps.py.
from fastapi.testclient import TestClient

from clark.agent.ppo import ClarkAgent
from clark.serve.app import build_app

CONFIGS = Path(__file__).resolve().parents[1] / "clark" / "data" / "configs"


@pytest.fixture
def client(tmp_path):
    agent = ClarkAgent(device="cpu", use_amp=False)
    ckpts = tmp_path / "no_ckpts"
    ckpts.mkdir()
    return TestClient(build_app(agent, CONFIGS,
                                checkpoint_label="test",
                                per_facility_ckpt_dir=ckpts))


# ── /compare ───────────────────────────────────────────────────────

def test_compare_returns_one_row_per_facility(client):
    r = client.post("/compare", json={
        "facility_ids": ["example_small", "example_medium"],
        "date": "2026-06-01", "seed": 7})
    assert r.status_code == 200, r.text
    body = r.json()
    fids = [row["facility_id"] for row in body["facilities"]]
    assert fids == ["example_small", "example_medium"]
    for row in body["facilities"]:
        assert "n_workers" in row and row["n_workers"] > 0
        assert "tightness_score" in row
        assert "assignments" in row and len(row["assignments"]) == row["n_workers"]
        assert row["model"] in {"foundation", "fine-tuned"}


def test_compare_ranking_orders_by_tightness_desc(client):
    r = client.post("/compare", json={
        "facility_ids": ["example_small", "example_medium", "example_large"],
        "date": "2026-06-01", "seed": 7}).json()
    scores = [x["tightness_score"] for x in r["ranking"]]
    assert scores == sorted(scores, reverse=True), \
        f"ranking not sorted desc: {scores}"


def test_compare_unknown_facility_returns_in_errors_not_500(client):
    r = client.post("/compare", json={
        "facility_ids": ["example_small", "nope_xyz"],
        "date": "2026-06-01"}).json()
    fids_ok = [x["facility_id"] for x in r["facilities"]]
    assert "example_small" in fids_ok
    assert "nope_xyz" not in fids_ok
    err_fids = [e["facility_id"] for e in r["errors"]]
    assert "nope_xyz" in err_fids
    assert r["errors"][0]["status_code"] == 404


def test_compare_empty_list_rejected(client):
    r = client.post("/compare", json={"facility_ids": []})
    assert r.status_code == 422


def test_compare_oversize_list_rejected(client):
    r = client.post("/compare", json={
        "facility_ids": ["example_small"] * 33})
    assert r.status_code == 422


# ── /calendar_check ─────────────────────────────────────────────────

def test_calendar_check_workday_is_ok(client):
    # 2026-06-01 is a Monday
    r = client.post("/calendar_check", json={
        "facility_id": "example_small", "date": "2026-06-01"})
    assert r.status_code == 200
    body = r.json()
    assert body["weekday"] == "monday"
    assert body["is_workday"] is True
    assert body["flags"] == []
    assert "OK" in body["summary"]


def test_calendar_check_saturday_on_mon_fri_facility_flagged(client):
    # 2026-06-06 is a Saturday. example_small does NOT have work_saturday.
    r = client.post("/calendar_check", json={
        "facility_id": "example_small", "date": "2026-06-06"}).json()
    assert r["weekday"] == "saturday"
    assert r["is_workday"] is False
    assert r["is_saturday"] is True
    assert any("Saturday" in f for f in r["flags"])
    assert "Saturday" in r["summary"] or "saturday" in r["summary"]


def test_calendar_check_sunday_always_flagged(client):
    # 2026-06-07 is a Sunday
    r = client.post("/calendar_check", json={
        "facility_id": "example_small", "date": "2026-06-07"}).json()
    assert r["weekday"] == "sunday"
    assert r["is_workday"] is False
    assert r["is_sunday"] is True
    assert any("Sunday" in f for f in r["flags"])


def test_calendar_check_unknown_facility_404(client):
    r = client.post("/calendar_check", json={
        "facility_id": "nope_xyz", "date": "2026-06-01"})
    assert r.status_code == 404

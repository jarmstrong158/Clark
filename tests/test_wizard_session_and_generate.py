"""Cover two unprotected wizard surfaces:

  1. POST /sessions  → GET /sessions  → GET /sessions/{id} roundtrip.
     The wizard's UI saves state every step; a regression in the
     session shape would silently corrupt every in-progress facility
     until someone noticed.

  2. POST /generate (quick-mode) — the wizard's primary OUTPUT path.
     Bugs here ship a broken YAML that wastes hours of fine-tune.
     We verify the quick-mode minimal-task override actually takes
     effect (the 'test_v3 / 8-worker / 47-of-64-worker-hours on
     secondary tasks' regression that was fixed in cli/main.py:811-831).

Subprocess-based for the same reason test_wizard_security.py is:
WizardHandler is a closure inside cmd_wizard."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


@pytest.fixture
def wizard(tmp_path):
    port = _free_port()
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    repo_root = Path(__file__).resolve().parents[1]
    import os as _os
    env = {**_os.environ, "CLARK_NO_BROWSER": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "cli.main", "wizard",
         "--port", str(port),
         "--sessions-dir", str(sessions)],
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            if httpx.get(base + "/", timeout=2.0).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.25)
    else:
        proc.terminate(); proc.wait(timeout=10)
        raise RuntimeError(f"wizard didn't come up at {base}")
    try:
        yield base, sessions
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=5)


# ── /sessions roundtrip ────────────────────────────────────────────

def test_session_save_then_list_then_load_roundtrip(wizard):
    base, _ = wizard
    body = {
        "session_id": "smoke_test_v1",
        "facility_name": "Smoke Test Warehouse",
        "selected_archetype": "ecom_direct",
        "month_intensity": {"january": "slow", "november": "peak"},
        "busiest_weekday": "monday",
        "answers": {"q1": "a", "q2": "b"},
        "current_step": 4,
    }
    r = httpx.post(base + "/sessions", json=body, timeout=5.0)
    assert r.status_code == 200
    assert r.json()["session_id"] == "smoke_test_v1"

    # /sessions GET must list it.
    listed = httpx.get(base + "/sessions", timeout=5.0).json()
    ids = [s["session_id"] for s in listed["sessions"]]
    assert "smoke_test_v1" in ids

    # /sessions/{id} GET must return the persisted fields verbatim.
    loaded = httpx.get(base + "/sessions/smoke_test_v1", timeout=5.0).json()
    assert loaded["facility_name"] == "Smoke Test Warehouse"
    assert loaded["selected_archetype"] == "ecom_direct"
    assert loaded["month_intensity"]["november"] == "peak"
    assert loaded["answers"] == {"q1": "a", "q2": "b"}
    assert loaded["current_step"] == 4
    # And it must carry the auto-stamped timestamps.
    assert "created_at" in loaded and "updated_at" in loaded


def test_session_update_preserves_created_at(wizard):
    """Re-saving the same session_id must keep the original created_at,
    only updated_at moves. Pins a subtle UX contract — the session list
    sorts by updated_at, so corrupting this would scramble session order."""
    base, _ = wizard
    body = {"session_id": "ctime_test", "facility_name": "v1"}
    r1 = httpx.post(base + "/sessions", json=body, timeout=5.0).json()
    loaded1 = httpx.get(base + "/sessions/ctime_test", timeout=5.0).json()
    created_first = loaded1["created_at"]

    time.sleep(1.1)  # advance clock enough to change isoformat to seconds
    body["facility_name"] = "v2"
    httpx.post(base + "/sessions", json=body, timeout=5.0)
    loaded2 = httpx.get(base + "/sessions/ctime_test", timeout=5.0).json()
    assert loaded2["created_at"] == created_first, \
        "second save clobbered created_at"
    assert loaded2["updated_at"] >= created_first
    assert loaded2["facility_name"] == "v2"


# ── /generate quick-mode (the primary wizard output) ──────────────

def test_generate_quick_mode_forces_minimal_task_set(wizard, tmp_path,
                                                       request):
    """The 'test_v3 8-workers / 47-of-64 worker-hours on secondary
    tasks' regression: quick-mode must hardcode the 5-task minimal set,
    NOT honor the template's full task list. Pinned per cli/main.py:811-831."""
    import yaml as _yaml
    base_url, _ = wizard

    body = {
        "mode": "quick",
        "archetype": "ecom_direct",
        "facility_name": "Quick-Mode Pin Test",
        "resulting_weights": {},
    }
    r = httpx.post(base_url + "/generate", json=body, timeout=10.0)
    assert r.status_code == 200, r.text
    out = r.json()
    assert "yaml_path" in out and "yaml_text" in out

    # The wizard writes the YAML to the REAL configs/user/ dir (path
    # is hardcoded in cli/main.py:728, not parameterisable). Without
    # a teardown the file accumulates every test run as
    # quick-mode_pin_test_v2, _v3, ... and pollutes the repo. Clean up.
    written = Path(out["yaml_path"])
    def _cleanup():
        try: written.unlink()
        except FileNotFoundError: pass
    request.addfinalizer(_cleanup)

    parsed = _yaml.safe_load(out["yaml_text"])
    assert parsed["facility"]["name"] == "Quick-Mode Pin Test"

    enabled = set(parsed.get("tasks", {}).get("enabled", []))
    expected = {"pick", "pack", "restock", "management", "idle"}
    assert enabled == expected, (
        f"quick-mode emitted tasks {enabled}, expected exactly {expected} "
        f"— the minimal-task override regressed (see "
        f"cli/main.py:811-831).")

    # Worker task_eligibility lists must NOT reference disabled tasks
    # (otherwise schema validator emits a warning at fine-tune time).
    for w in parsed.get("workers", []):
        te = w.get("task_eligibility")
        if isinstance(te, list):
            assert set(te).issubset(expected), \
                f"worker {w.get('name')} eligibility {te} references " \
                f"disabled task(s)"


def test_generate_unknown_archetype_returns_400(wizard):
    base_url, _ = wizard
    r = httpx.post(base_url + "/generate",
                    json={"archetype": "does_not_exist",
                          "facility_name": "x",
                          "mode": "quick"},
                    timeout=5.0)
    assert r.status_code == 400
    assert "error" in r.json()

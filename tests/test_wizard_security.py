"""Security regression tests for the wizard web server.

External review (2026-05) flagged three real issues, all fixed in
cli/main.py:

1. Wildcard `Access-Control-Allow-Origin: *` let any web page in the
   user's browser read responses from the localhost wizard.
2. `session_id` from POST body was used directly as a filename
   (`sessions_dir / f"{sid}.json"`), so a malicious payload
   `{"session_id": "../../evil"}` would write outside sessions_dir.
3. `yaml_path` in `/train/start` was unsanitized — a same-origin
   attacker could spawn a finetune subprocess against arbitrary YAML.

These tests pin the fix by spinning up `clark wizard` as a real
subprocess on a random port + a tmp sessions_dir, and exercising the
adversarial inputs over HTTP. Subprocess-based (not in-process)
because WizardHandler is defined inside cmd_wizard as a closure over
its local state.
"""
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
    """Spawn `clark wizard` on a free port with an isolated
    sessions_dir; tear down after the test."""
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
    last = None
    while time.time() < deadline:
        try:
            r = httpx.get(base + "/", timeout=2.0)
            if r.status_code == 200:
                break
        except Exception as e:
            last = e
        time.sleep(0.25)
    else:
        proc.terminate()
        proc.wait(timeout=10)
        raise RuntimeError(f"wizard didn't come up at {base}: {last}")
    try:
        yield base, sessions
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_no_wildcard_cors_header(wizard):
    """The wildcard `*` CORS header was removed; verify it's not
    returned by GET / (the page response) or POST /sessions."""
    base, _ = wizard
    r = httpx.get(base + "/", timeout=5)
    assert r.status_code == 200
    assert r.headers.get("Access-Control-Allow-Origin") != "*"


def test_cross_origin_post_rejected(wizard):
    """POSTs with a foreign Origin header get 403. Defends against
    a malicious tab in the user's browser triggering wizard actions."""
    base, _ = wizard
    r = httpx.post(base + "/sessions",
                    json={"session_id": "abc"},
                    headers={"Origin": "http://evil.example"},
                    timeout=5)
    assert r.status_code == 403, (r.status_code, r.text[:200])


def test_same_origin_post_accepted(wizard):
    """No Origin header (curl-style same-origin) is allowed —
    that's the wizard's normal use case via the browser opening
    localhost directly."""
    base, _ = wizard
    r = httpx.post(base + "/sessions",
                    json={"session_id": "abc"}, timeout=5)
    assert r.status_code == 200, (r.status_code, r.text[:200])
    body = r.json()
    assert body.get("session_id") == "abc"


def test_session_id_rejects_path_traversal(wizard):
    """Old behavior: `{"session_id": "../evil"}` -> file written
    outside sessions_dir. New behavior: 400 + no file leaks."""
    base, sessions = wizard
    parent_before = set((sessions.parent).iterdir())
    for bad in ("../evil", "../../boom", "/abs/path",
                 "a/b/c", "x\\y", "long" + "x" * 200,
                 "with space", ""):
        r = httpx.post(base + "/sessions",
                        json={"session_id": bad}, timeout=5)
        assert r.status_code in (400, 200), (bad, r.status_code)
        if r.status_code == 200:
            # only the empty-string fallback (-> uuid) is allowed
            assert bad == ""
    parent_after = set((sessions.parent).iterdir())
    assert parent_before == parent_after, (
        "session-id sanitization let a file leak outside sessions_dir")


def test_train_start_rejects_yaml_outside_user_configs_dir(wizard,
                                                            tmp_path):
    """`/train/start` only accepts yaml_path under user_configs_dir;
    refuses to spawn finetune subprocess against an arbitrary path."""
    base, _ = wizard
    rogue = tmp_path / "rogue.yaml"
    rogue.write_text("facility: {name: x}\n", encoding="utf-8")
    r = httpx.post(base + "/train/start",
                    json={"yaml_path": str(rogue)}, timeout=5)
    assert r.status_code == 400, (r.status_code, r.text[:200])
    body = r.json()
    assert "user_configs_dir" in body.get("error", "").lower() or \
        "yaml_path" in body.get("error", "").lower()


def test_train_start_requires_yaml_path(wizard):
    """No yaml_path -> 400, not crash."""
    base, _ = wizard
    r = httpx.post(base + "/train/start", json={}, timeout=5)
    assert r.status_code == 400
    assert "yaml_path" in r.json().get("error", "")

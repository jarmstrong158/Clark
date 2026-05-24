"""Smoke tests for the operations dashboard.

The dashboard is a single static HTML file served by a tiny stdlib
HTTP server. These tests pin three things:

1. The HTML file exists in the package and has the expected tab
   structure (so a future refactor that drops/renames a tab fails
   loudly instead of silently shipping a broken dashboard).
2. The CLI subcommand wires up correctly and prints the launch URL.
3. The server binds, serves the page, and 404s on unknown paths.

We don't exercise the JS — it talks directly to clark serve from
the browser, which is integration territory better covered by
test_compare_and_calendar.py + test_serve_api.py.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[1]
OPS_HTML = REPO / "clark" / "ops" / "index.html"


# ── HTML structure ─────────────────────────────────────────────────

def test_ops_html_exists_and_has_all_tabs():
    """The dashboard's tab nav must include every promised tab. If a
    refactor drops one (or renames its data-tab key), this fails."""
    assert OPS_HTML.exists(), f"ops dashboard HTML missing at {OPS_HTML}"
    src = OPS_HTML.read_text(encoding="utf-8")
    for tab in ("plan", "compare",
                "calendar", "briefing", "training"):
        assert f'data-tab="{tab}"' in src, f"missing tab '{tab}'"


def test_ops_html_calls_real_clark_serve_routes():
    """The form handlers must call the actual clark serve routes,
    not stale placeholders. Pins each route by name so a route
    rename on the server side surfaces here."""
    src = OPS_HTML.read_text(encoding="utf-8")
    # /what_if intentionally absent — folded into Plan as modifiers.
    # /simulate intentionally absent — the Sweep tab that used it
    # was deleted; the "Find recommended staffing" button on Plan
    # uses /plan_outcome with extra_workers instead so it can
    # respect the Plan tab's date/volume/absent context.
    for route in ("/health", "/facilities", "/facility/",
                  "/plan", "/compare",
                  "/calendar_check", "/plan_outcome",
                  "/training_runs", "/launch_wizard"):
        assert route in src, f"dashboard doesn't reference {route}"


# ── CLI + server ───────────────────────────────────────────────────

def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


@pytest.fixture
def ops_server():
    port = _free_port()
    import os as _os
    env = {**_os.environ, "CLARK_NO_BROWSER": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "cli.main", "ops",
         "--port", str(port), "--no-browser"],
        cwd=str(REPO), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            if httpx.get(base + "/", timeout=2.0).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.2)
    else:
        proc.terminate(); proc.wait(timeout=5)
        raise RuntimeError(f"ops server didn't come up at {base}")
    try:
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=5)


def test_ops_server_serves_html_on_root(ops_server):
    r = httpx.get(ops_server + "/", timeout=5.0)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Clark — Operations Dashboard" in r.text


def test_ops_server_serves_html_on_index_html(ops_server):
    r = httpx.get(ops_server + "/index.html", timeout=5.0)
    assert r.status_code == 200


def test_ops_server_404s_unknown_path(ops_server):
    r = httpx.get(ops_server + "/nope", timeout=5.0)
    assert r.status_code == 404


def test_ops_server_training_runs_returns_list(ops_server):
    """The Training tab polls /training_runs every 5s. Endpoint must
    always return a JSON object with a 'runs' list, even when no
    training_metrics.json exists anywhere on disk (empty list).
    Without this, a fresh checkout with no prior runs would see the
    tab error out on first load instead of showing the friendly
    'No training runs detected' empty state."""
    r = httpx.get(ops_server + "/training_runs", timeout=5.0)
    assert r.status_code == 200
    body = r.json()
    assert "runs" in body
    assert isinstance(body["runs"], list)

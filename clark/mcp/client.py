"""Thin HTTP client over Clark's local inference API (`clark serve`).

This is the ONLY thing in clark.mcp that talks to Clark. It targets
the localhost endpoint `clark serve` exposes (default
http://127.0.0.1:8000). The http client is injectable so tests can
drive the real ASGI app in-process (httpx ASGITransport) without a
live socket.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import httpx


class ClarkAPIError(RuntimeError):
    pass


class ClarkClient:
    def __init__(self, base_url: Optional[str] = None,
                 http: Optional[httpx.Client] = None):
        self.base_url = (base_url
                         or os.environ.get("CLARK_API_URL",
                                           "http://127.0.0.1:8000")
                         ).rstrip("/")
        # 600s: a full-year /simulate on CPU is ~90s; generous margin.
        self._http = http or httpx.Client(timeout=600.0)

    def _get(self, path: str) -> Any:
        try:
            r = self._http.get(self.base_url + path)
        except httpx.HTTPError as e:
            raise ClarkAPIError(
                f"Clark API unreachable at {self.base_url} ({e}). "
                f"Is `clark serve` running?"
            ) from e
        if r.status_code != 200:
            raise ClarkAPIError(
                f"GET {path} -> {r.status_code}: {r.text[:200]}"
            )
        return r.json()

    def _post(self, path: str, body: dict) -> Any:
        try:
            r = self._http.post(self.base_url + path, json=body)
        except httpx.HTTPError as e:
            raise ClarkAPIError(
                f"Clark API unreachable at {self.base_url} ({e}). "
                f"Is `clark serve` running?"
            ) from e
        if r.status_code != 200:
            raise ClarkAPIError(
                f"POST {path} -> {r.status_code}: {r.text[:200]}"
            )
        return r.json()

    # ── Clark API surface (mirrors clark serve exactly) ──────────────

    def health(self) -> dict:
        return self._get("/health")

    def list_facilities(self) -> list[str]:
        return self._get("/facilities")["facilities"]

    def capabilities(self) -> dict:
        """Clark's architectural limits (worker/task caps, training
        envelope). Read from clark_limits.yaml + schema so the answer
        is always current with the code."""
        return self._get("/capabilities")

    def facility_info(self, facility_id: str) -> dict:
        return self._get(f"/facility/{facility_id}")

    def get_plan(self, facility_id: str, date: Optional[str] = None,
                 volume: Optional[int] = None) -> dict:
        return self._post("/plan", {"facility_id": facility_id,
                                    "date": date, "volume": volume})

    def what_if(self, facility_id: str, volume: Optional[int] = None,
                absent_workers: Optional[list[str]] = None,
                date: Optional[str] = None) -> dict:
        return self._post("/what_if", {
            "facility_id": facility_id, "date": date, "volume": volume,
            "absent_workers": absent_workers or [],
        })

    def compare_facilities(self, facility_ids: list[str],
                           date: Optional[str] = None) -> dict:
        return self._post("/compare", {"facility_ids": list(facility_ids),
                                       "date": date})

    def calendar_check(self, facility_id: str, date: str) -> dict:
        return self._post("/calendar_check",
                          {"facility_id": facility_id, "date": date})

    def plan_outcome(self, facility_id: str, date: Optional[str] = None,
                     volume: Optional[int] = None,
                     absent_workers: Optional[list[str]] = None,
                     extra_workers: int = 0,
                     n_samples: int = 20) -> dict:
        """Monte-Carlo single-day outcome projection — the same primitive
        the ops dashboard's "Project outcome" and "Find recommended
        staffing" buttons use."""
        return self._post("/plan_outcome", {
            "facility_id": facility_id, "date": date, "volume": volume,
            "absent_workers": absent_workers or [],
            "extra_workers": int(extra_workers),
            "n_samples": int(n_samples),
        })

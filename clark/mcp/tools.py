"""Tool implementations — pure functions over a ClarkClient.

These are the single source of truth for what the MCP server exposes
(server.py). Keeping them as plain functions makes them directly
testable against the real clark.serve app in-process, without any
MCP machinery.

Honest scope: every tool returns data verbatim from `clark serve`
plus a small `note` where the distinction matters (e.g. opening
assignments vs simulated outcome). The MCP host's model is
responsible for the prose; we do not synthesize plan content here.
"""
from __future__ import annotations

from typing import Optional

from .client import ClarkClient


def list_facilities(clark: ClarkClient) -> dict:
    """List the facility configs Clark can plan for."""
    return {"facilities": clark.list_facilities()}


def capabilities(clark: ClarkClient) -> dict:
    """Architectural capacity of the Clark RL agent itself: worker/task
    bounds it was trained inside, model architecture, training-envelope
    notes. Sourced from clark_limits.yaml + schema, NOT memorised.

    Use this for questions ABOUT Clark ("what's the largest roster
    Clark can plan?"). For facility-specific data use facility_info.
    """
    return clark.capabilities()


def facility_info(clark: ClarkClient, facility_id: str) -> dict:
    """Return a facility's full configuration (workers, tasks, rules)."""
    return clark.facility_info(facility_id)


def get_plan(clark: ClarkClient, facility_id: str,
             date: Optional[str] = None,
             volume: Optional[int] = None) -> dict:
    """Get Clark's opening shift plan (per-worker task + hustle) for a
    facility on a date (default today), optionally overriding volume."""
    return clark.get_plan(facility_id, date=date, volume=volume)


def what_if(clark: ClarkClient, facility_id: str,
            volume: Optional[int] = None,
            absent_workers: Optional[list[str]] = None,
            date: Optional[str] = None) -> dict:
    """Compare Clark's plan for the base scenario vs a modified one
    (changed volume and/or forced-absent workers). Returns both plans."""
    return clark.what_if(facility_id, volume=volume,
                         absent_workers=absent_workers, date=date)


def compare_facilities(clark: ClarkClient, facility_ids: list[str],
                       date: Optional[str] = None) -> dict:
    """Plan N facilities for the same date and rank them by who's most
    stretched. Honest scope: opening assignments, NOT end-of-day
    outcome projection."""
    return clark.compare_facilities(list(facility_ids), date=date)


def calendar_check(clark: ClarkClient, facility_id: str,
                   date: str) -> dict:
    """Sanity-check a date against a facility's schedule BEFORE planning,
    so the host can warn "your facility doesn't operate Saturdays"
    instead of silently returning a fabricated Saturday plan."""
    return clark.calendar_check(facility_id, date)


def plan_outcome(clark: ClarkClient, facility_id: str,
                 date: Optional[str] = None,
                 volume: Optional[int] = None,
                 absent_workers: Optional[list[str]] = None,
                 extra_workers: int = 0,
                 n_samples: int = 20) -> dict:
    """Monte-Carlo single-day outcome projection. Runs Clark on the
    given scenario `n_samples` times with different seeds and returns
    the grade distribution + completion stats. Use this when the
    operator asks "is this day at risk?" — `get_plan` only returns
    opening assignments and cannot answer that honestly."""
    return clark.plan_outcome(facility_id, date=date, volume=volume,
                              absent_workers=absent_workers,
                              extra_workers=extra_workers,
                              n_samples=n_samples)


def find_recommended_staffing(clark: ClarkClient, facility_id: str,
                              date: Optional[str] = None,
                              volume: Optional[int] = None,
                              absent_workers: Optional[list[str]] = None,
                              target_ab_pct: float = 80.0,
                              max_extras: int = 10,
                              n_samples: int = 20) -> dict:
    """Walk extra_workers from 0 upward until the A+B grade share on
    the projected outcome hits `target_ab_pct`. Returns the trajectory
    and the first roster size that meets the target. Same logic the
    ops dashboard's "Find recommended staffing" button uses."""
    trajectory = []
    hit = None
    for extras in range(max_extras + 1):
        r = plan_outcome(clark, facility_id, date=date, volume=volume,
                         absent_workers=absent_workers,
                         extra_workers=extras, n_samples=n_samples)
        dist = r.get("grade_distribution", {}) or {}
        ab = float(dist.get("A", 0) + dist.get("B", 0))
        trajectory.append({
            "extra_workers": extras,
            "n_workers": r.get("n_workers"),
            "grade_distribution": dist,
            "ab_pct": ab,
            "completion_mean": r.get("completion_mean"),
        })
        if hit is None and ab >= target_ab_pct:
            hit = extras
            break
    return {
        "facility_id": facility_id,
        "target_ab_pct": target_ab_pct,
        "recommended_extra_workers": hit,
        "hit_target": hit is not None,
        "trajectory": trajectory,
        "note": ("real Monte-Carlo projection per roster step using "
                 "the trained policy; not a heuristic"),
    }

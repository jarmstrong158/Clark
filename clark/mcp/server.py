"""MCP server exposing Clark as natural-language tools.

Real MCP server (any MCP-aware host can use it: Claude Desktop,
Cursor, Continue, Zed, ...). Thin layer: every tool delegates to
clark.mcp.tools, which calls a localhost `clark serve` over HTTP.

Run:  clark mcp                        (stdio transport)
Env:  CLARK_API_URL  (default http://127.0.0.1:8000)
"""
from __future__ import annotations

from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:                                     # pragma: no cover
    raise SystemExit(
        "The `mcp` package is required for `clark mcp`. "
        "Install with:  pip install -e \".[mcp]\""
    ) from e

from . import tools
from .client import ClarkClient

mcp = FastMCP("clark")
_clark = ClarkClient()


@mcp.tool()
def clark_list_facilities() -> dict:
    """List the warehouse facilities Clark can produce shift plans for."""
    return tools.list_facilities(_clark)


@mcp.tool()
def clark_capabilities() -> dict:
    """Clark's architectural capacity: worker/task bounds it was trained
    inside, model architecture, training-envelope notes. Use this for
    questions ABOUT the Clark RL agent (e.g. 'what's the largest
    roster Clark can plan?'); use clark_facility_info for questions
    about a specific facility."""
    return tools.capabilities(_clark)


@mcp.tool()
def clark_facility_info(facility_id: str) -> dict:
    """Get a facility's full configuration: workers, enabled tasks,
    and business rules (OT triggers, management requirements, etc.)."""
    return tools.facility_info(_clark, facility_id)


@mcp.tool()
def clark_get_plan(facility_id: str, date: Optional[str] = None,
                   volume: Optional[int] = None) -> dict:
    """Clark's opening shift plan for a facility: each worker's
    assigned task and whether they hustle. `date` is YYYY-MM-DD
    (default today); `volume` optionally overrides the day's
    order volume. Opening assignments only — for projected outcome
    use clark_plan_outcome."""
    return tools.get_plan(_clark, facility_id, date=date, volume=volume)


@mcp.tool()
def clark_what_if(facility_id: str, volume: Optional[int] = None,
                  absent_workers: Optional[list[str]] = None,
                  date: Optional[str] = None) -> dict:
    """Compare Clark's plan for the base scenario against a modified
    one (different order volume and/or specific workers forced absent).
    Returns both opening plans for side-by-side comparison."""
    return tools.what_if(_clark, facility_id, volume=volume,
                         absent_workers=absent_workers, date=date)


@mcp.tool()
def clark_compare_facilities(facility_ids: list[str],
                             date: Optional[str] = None) -> dict:
    """Plan N facilities for the same date and rank them by who's most
    stretched. Use for cross-facility prompts like 'of my 3 warehouses,
    where am I shortest on Monday?'. Honest scope: still opening-
    assignment plans, NOT end-of-day outcome."""
    return tools.compare_facilities(_clark, facility_ids, date=date)


@mcp.tool()
def clark_calendar_check(facility_id: str, date: str) -> dict:
    """Sanity-check a date BEFORE planning. Use this when the user
    asks for a date that might not be a workday (Saturday on a Mon-Fri
    facility, Sunday, off-calendar month). Returns flags + a one-
    sentence summary so the user isn't silently handed a fabricated
    plan for a non-workday."""
    return tools.calendar_check(_clark, facility_id, date)


@mcp.tool()
def clark_plan_outcome(facility_id: str, date: Optional[str] = None,
                       volume: Optional[int] = None,
                       absent_workers: Optional[list[str]] = None,
                       extra_workers: int = 0,
                       n_samples: int = 20) -> dict:
    """Monte-Carlo single-day outcome projection: runs Clark on the
    scenario `n_samples` times with different seeds and returns the
    grade distribution + completion stats. Use this when the user
    asks "is Tuesday at risk?" or "what grade should I expect?" —
    `clark_get_plan` only returns opening assignments and cannot
    answer outcome questions honestly."""
    return tools.plan_outcome(_clark, facility_id, date=date,
                              volume=volume,
                              absent_workers=absent_workers,
                              extra_workers=extra_workers,
                              n_samples=n_samples)


@mcp.tool()
def clark_find_recommended_staffing(
        facility_id: str, date: Optional[str] = None,
        volume: Optional[int] = None,
        absent_workers: Optional[list[str]] = None,
        target_ab_pct: float = 80.0,
        max_extras: int = 10,
        n_samples: int = 20) -> dict:
    """Walks +0, +1, +2, ... extra workers until the projected A+B
    grade share hits `target_ab_pct`. Answers "how many more workers
    do I need to reliably ship Tuesday?". Real Monte-Carlo projection
    at each step, not a heuristic. Same logic the ops dashboard's
    "Find recommended staffing" button uses."""
    return tools.find_recommended_staffing(
        _clark, facility_id, date=date, volume=volume,
        absent_workers=absent_workers, target_ab_pct=target_ab_pct,
        max_extras=max_extras, n_samples=n_samples,
    )


def main() -> None:
    """Entry point for `clark mcp` and direct `python -m clark.mcp.server`."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

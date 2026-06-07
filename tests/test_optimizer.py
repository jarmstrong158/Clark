"""CP-SAT completion-bound sanity checks (clark.inference.optimizer.plan_day).

Fast, no simulation: a hand-built episode/rules with all orders present at the
start, so completion turns purely on whether the roster has the capacity to
pick+pack everything by the operational close."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("ortools")

from clark.inference.optimizer import plan_day


def _episode(total, n_workers):
    # empty arrival_schedule -> all `total` orders present from the start
    return SimpleNamespace(
        total_orders=total,
        arrival_schedule={},
        eod_hour=16.0,
        workers=[SimpleNamespace(is_absent=False) for _ in range(n_workers)],
    )


def _rules():
    return SimpleNamespace(
        day_start_hour=8.0,
        ot_hard_stop_hour=18.0,
        order_cutoff_hour=17.0,
        management_daily_hours_required=0.0,
        lunch_hour=12.0,
        lunch_duration=0.5,
    )


def test_ample_capacity_is_complete():
    # 10 workers @ 15 oph over ~10h, only 300 orders -> trivially shippable
    r = plan_day(_episode(total=300, n_workers=10), _rules(), mean_oph=15.0)
    assert r["complete"] and r["shipped_max"] >= 298


def test_starved_roster_cannot_complete():
    # 1 worker cannot pack 5000 orders by the close -> infeasible, capped low
    r = plan_day(_episode(total=5000, n_workers=1), _rules(), mean_oph=15.0)
    assert not r["complete"] and r["shipped_max"] < 5000


def test_empty_day_is_trivially_complete():
    r = plan_day(_episode(total=0, n_workers=5), _rules(), mean_oph=15.0)
    assert r["complete"]

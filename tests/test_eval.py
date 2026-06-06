"""Held-out evaluation harness: percentile math, year summarization, and a
small end-to-end smoke that the eval pipeline wires up and returns the
expected structure.
"""
from __future__ import annotations

import os

import pytest

from clark.inference.evaluate import _pctile, _dist, summarize_year, evaluate


def test_pctile_endpoints_and_interpolation():
    vals = list(range(0, 101, 10))  # 0,10,...,100 (11 values)
    assert _pctile(vals, 0) == 0
    assert _pctile(vals, 100) == 100
    assert _pctile(vals, 50) == 50
    assert _pctile(vals, 10) == pytest.approx(10)
    assert _pctile(vals, 90) == pytest.approx(90)
    # single value -> that value at any percentile
    assert _pctile([7.0], 25) == 7.0
    assert _pctile([], 50) == 0.0


def test_dist_summary():
    d = _dist([10.0, 20.0, 30.0, 40.0])
    assert d["n"] == 4
    assert d["min"] == 10.0 and d["max"] == 40.0
    assert d["median"] == pytest.approx(25.0)
    assert d["p10"] < d["median"] < d["p90"]


def test_summarize_year_grades_and_rates():
    # 4 days: A, A, B, F. F day ships 50/100; others ship fully.
    def day(grade, total, shipped, ot=0.0):
        return {"footer": {"grade": grade, "orders_shipped": shipped, "ot_hours": ot},
                "header": {"total_orders": total}}
    summary = {"daily_summaries": [
        day("A", 100, 100), day("A", 100, 100), day("B", 100, 100), day("F", 100, 50, ot=2.0),
    ]}
    m = summarize_year(summary)
    assert m["n_days"] == 4
    assert m["a_pct"] == pytest.approx(50.0)     # 2/4
    assert m["ab_pct"] == pytest.approx(75.0)    # 3/4
    assert m["f_pct"] == pytest.approx(25.0)     # 1/4
    assert m["ship_win_pct"] == pytest.approx(75.0)        # 3/4 shipped fully
    assert m["completion_pct"] == pytest.approx(100 * 350 / 400)  # 350/400 orders
    assert m["ot_day_pct"] == pytest.approx(25.0)


def test_summarize_year_empty():
    m = summarize_year({"daily_summaries": []})
    assert m["n_days"] == 0
    assert m["a_pct"] == 0.0 and m["completion_pct"] == 0.0


@pytest.mark.skipif(not os.environ.get("CLARK_RUN_SLOW"),
                    reason="full-year sim is slow; set CLARK_RUN_SLOW=1 to run")
def test_evaluate_smoke_structure():
    """End-to-end on a fresh (untrained) agent: 1 stage-1 facility. We only
    assert the pipeline runs and the result structure is well-formed — not
    that an untrained policy scores well."""
    from clark.agent.ppo import ClarkAgent
    agent = ClarkAgent(device="cpu")
    res = evaluate(agent, n_per_stage=1, stages=(1,), base_seed=0)
    assert res["total_facilities"] == 1
    assert set(res["per_stage"].keys()) == {1}
    for metric in ("a_pct", "ab_pct", "f_pct", "ship_win_pct", "completion_pct"):
        assert metric in res["overall"]
        assert 0.0 <= res["overall"][metric]["median"] <= 100.0

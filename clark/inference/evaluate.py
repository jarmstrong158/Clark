"""Held-out evaluation of a Clark checkpoint.

Samples fresh synthetic facilities the policy has never seen, simulates a
full work-year on each with the in-env **production grader** (the same one
the training loop uses — not a probe rule), and reports **distributions**
(median / p10 / p90 / IQR), not single numbers. RL metrics are noisy across
configs and seeds; a point estimate on one facility is not evidence.

Reporting per curriculum stage (1 = small/easy → 3 = full complexity +
stress days) doubles as a **generalization view**: every sampled facility is
held-out (the model never trained on that exact config), so the stage-1 →
stage-3 trend shows how the foundation generalizes as facility complexity
rises.

`run_one_year` / `summarize_year` are the canonical full-year-sim primitives
(the production `/simulate` path and `tools/clark_eval.py` use the same
loop).
"""
from __future__ import annotations

import random
from collections import Counter
from statistics import median

import numpy as np

_METRICS = ("a_pct", "ab_pct", "f_pct", "ship_win_pct", "completion_pct")


def run_one_year(agent, cfg) -> dict:
    """Simulate one full work-year via YearEnv + the production action path.
    Returns the env's year summary (daily_summaries carry per-day production
    grades from facility_env)."""
    from clark.agent.actions import get_action_mask, get_hustle_mask
    from clark.agent.state import StateBuilder
    from clark.env.year_env import YearEnv

    env = YearEnv(cfg)
    builder = StateBuilder(cfg)
    env.reset()
    agent.reset_hidden()
    max_ticks = 200_000  # safety cap; a full year is well under this
    ticks = 0
    while not env.is_year_done and ticks < max_ticks:
        sd = builder.build(env.day_env)
        mask = get_action_mask(env.day_env)
        hmask = get_hustle_mask(env.day_env)
        ta, ha, *_ = agent.select_action_from_dict(sd, mask, hustle_mask=hmask)
        actions = [(i, int(ta[i]), bool(ha[i])) for i in range(len(ta))]
        env.step(actions)
        ticks += 1
    return env._get_year_summary()


def summarize_year(summary: dict) -> dict:
    """Headline metrics for one simulated year."""
    daily = summary.get("daily_summaries", [])
    n = len(daily)
    grades = Counter()
    ot_days = ship_full = total_o = shipped_o = 0
    for d in daily:
        f = d.get("footer", {})
        h = d.get("header", {})
        grades[f.get("grade", "?")] += 1
        if f.get("ot_hours", 0) > 0:
            ot_days += 1
        t = h.get("total_orders", 0)
        s = f.get("orders_shipped", 0)
        total_o += t
        shipped_o += s
        if t > 0 and s >= t:
            ship_full += 1
    nz = max(1, n)
    return {
        "n_days": n,
        "a_pct": 100 * grades["A"] / nz,
        "ab_pct": 100 * (grades["A"] + grades["B"]) / nz,
        "f_pct": 100 * grades["F"] / nz,
        "ship_win_pct": 100 * ship_full / nz,
        "completion_pct": 100 * shipped_o / max(1, total_o),
        "ot_day_pct": 100 * ot_days / nz,
    }


def _pctile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0, 100])."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(idx)
    frac = idx - lo
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _dist(vals: list[float]) -> dict:
    """Distribution summary for one metric across runs."""
    s = sorted(vals)
    return {
        "median": median(s) if s else 0.0,
        "p10": _pctile(s, 10),
        "p25": _pctile(s, 25),
        "p75": _pctile(s, 75),
        "p90": _pctile(s, 90),
        "min": s[0] if s else 0.0,
        "max": s[-1] if s else 0.0,
        "n": len(s),
    }


def _aggregate(runs: list[dict]) -> dict:
    return {m: _dist([r[m] for r in runs]) for m in _METRICS}


def evaluate(agent, n_per_stage: int = 10, stages=(1, 2, 3),
             base_seed: int = 0, progress=None, year_runner=None) -> dict:
    """Run held-out evaluation. For each stage, sample `n_per_stage` fresh
    synthetic facilities, simulate a full year on each, and aggregate metric
    distributions. Returns {"per_stage": {stage: {...}}, "overall": {...},
    "n_per_stage", "stages"}.

    `year_runner(agent, cfg) -> summary` defaults to `run_one_year` (the
    trained policy). Pass a different runner (e.g.
    `clark.inference.baseline.run_one_year_baseline`) to evaluate a non-RL
    scheduler over the *same* sampled facilities — apples-to-apples.

    Determinism: config sampling and the sim are seeded per (stage, i) off
    `base_seed`, so the same args reproduce the same facilities + numbers.
    """
    from clark.training.synthetic_gen import generate_random_facility

    runner = year_runner or run_one_year

    per_stage = {}
    all_runs = []
    for stage in stages:
        stage_runs = []
        for i in range(n_per_stage):
            seed = base_seed + stage * 100_000 + i
            random.seed(seed)
            np.random.seed(seed)
            cfg = generate_random_facility(stage=stage)
            try:
                import torch
                torch.manual_seed(seed)
            except Exception:
                pass
            m = summarize_year(runner(agent, cfg))
            stage_runs.append(m)
            all_runs.append(m)
            if progress:
                progress(stage, i + 1, n_per_stage)
        per_stage[stage] = _aggregate(stage_runs)
    return {
        "per_stage": per_stage,
        "overall": _aggregate(all_runs),
        "n_per_stage": n_per_stage,
        "stages": list(stages),
        "total_facilities": len(all_runs),
    }

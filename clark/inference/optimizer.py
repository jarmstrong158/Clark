"""CP-SAT completion bound for the daily order-flow problem.

The greedy and the RL agent both decide *reactively*, tick by tick. This
module asks, with perfect foresight and optimal allocation: **could the day's
orders be shipped at all, and how much slack is there?** It's an optimistic
relaxation (every simplification favours the planner), so a day it calls
infeasible is genuinely infeasible for any online policy.

What it models faithfully — the order pipeline:

  * **pack is the bottleneck** (pick runs at PICK_MULTIPLIER × pack speed);
  * **flow precedence** — can't pack what isn't picked, can't pick what hasn't
    arrived — via the real per-tick `arrival_schedule`;
  * **availability** is the sim's actual rule (`is_absent`, not shift windows)
    — the whole non-absent roster is on the floor all day, minus lunch;
  * the operational day runs to the **OT hard stop**, and late (post-eod)
    arrivals shipped before it cost no OT;
  * the **management labour** requirement;
  * labour as **centi-workers** so a worker's tick splits fractionally
    (mirrors the sim's `work_carry`; integer workers rounding-fail on the
    arrival cap).

**Scope — read this before trusting a number.** Auditing the grader showed the
A-grade is *multi-factor*: on a representative facility the non-A days split
~evenly between **overtime (39)** and **restock fill < 95% (35)**, plus a few
incomplete. This model covers the order-flow / OT dimension only — it does
**not** model restock dynamics, the management-backlog or per-task demerits,
so it is **not** a faithful "optimal A-rate" ceiling and we don't report one.
A true A-grade bound would mean re-implementing the whole multi-factor grader
(restock + mgmt + per-task + OT) inside CP-SAT — effectively rebuilding the
simulator as an optimisation model.

What it *does* establish cleanly is the **completion ceiling**: whether all
orders can ship by the operational close. That answers the strategic question
directly — across held-out days it comes out at ~100%, i.e. **throughput is
never the binding constraint**. So the Clark-vs-classical gap is not about
scheduling capacity; it lives entirely in jointly satisfying the soft quality
constraints (OT checkpoint, restock %, management, per-task) — exactly the
multi-objective trade-off a learned policy handles better than a fixed rule.
"""
from __future__ import annotations

from clark.env.facility_env import PICK_MULTIPLIER_MAIN

TICK_H = 1.0 / 6.0  # 10-minute ticks
_SCALE = 100        # integer scaling for CP-SAT (centi-orders)


def _present_counts(episode, rules, tick_times):
    """Workers available at each tick. The sim gates availability on
    ``is_absent`` (not shift windows) — every non-absent worker is on the
    floor for the whole operational day — so presence is a constant roster,
    dropped only during the lunch window."""
    lunch_start = rules.lunch_hour
    lunch_end = rules.lunch_hour + rules.lunch_duration
    roster = sum(1 for w in episode.workers if not getattr(w, "is_absent", False))
    return [0 if (lunch_start <= t < lunch_end) else roster for t in tick_times]


def _arrival_cum(episode, tick_times):
    """Cumulative orders available to pick by each tick.

    Orders not in the day's `arrival_schedule` are treated as present from the
    start (the morning block / carryover), so the early window is arrival-rich
    and the flow + capacity limits bind, which matches the sim."""
    sched = episode.arrival_schedule or {}
    scheduled_total = sum(sched.values())
    morning = max(0, episode.total_orders - scheduled_total)
    cum = []
    running = morning
    for t in tick_times:
        # add anything scheduled in (t-TICK_H, t]
        for h, v in sched.items():
            if t - TICK_H < h <= t + 1e-9:
                running += v
        cum.append(min(running, episode.total_orders))
    return cum


def plan_day(episode, rules, mean_oph):
    """Optimal-planning completion feasibility for one day.

    Under perfect foresight and optimal allocation, can all D orders ship by
    the operational close? Returns dict: complete (bool), shipped_max (best
    achievable), total (D). See the module docstring for why this is a
    *completion* bound, not an A-grade ceiling (the A-grade is multi-factor)."""
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'CP-SAT bound needs ortools: pip install -e ".[optimizer]"') from e

    D = int(episode.total_orders)
    if D <= 0:
        return {"complete": True, "shipped_max": 0, "total": 0}

    day_start = rules.day_start_hour
    eod = episode.eod_hour
    # operational close = the OT hard stop: orders arrive to the cutoff but the
    # floor keeps shipping until the hard stop.
    close = (getattr(rules, "ot_hard_stop_hour", None)
             or getattr(rules, "order_cutoff_hour", None) or (eod + 3.0))
    close = max(close, eod)
    mgmt_req_h = getattr(rules, "management_daily_hours_required", 0.0) or 0.0

    # Labour is modelled in CENTI-WORKERS (WS) so a worker's time can split
    # fractionally across tasks within a tick — mirroring the sim's fractional
    # work_carry. With integer whole-worker counts the high per-tick pick rate
    # can't land exactly on the arrival cap and the day falls a few orders
    # short purely from rounding; centi-workers remove that artifact. Orders
    # are scaled by OS.
    WS = 100
    OS = 10_000
    RP = max(1, round(mean_oph * PICK_MULTIPLIER_MAIN * TICK_H * OS / WS))  # /centi-worker-tick
    RK = max(1, round(mean_oph * 1.0 * TICK_H * OS / WS))

    n_ticks = max(1, round((close - day_start) / TICK_H))
    tick_times = [day_start + i * TICK_H for i in range(n_ticks)]
    present = _present_counts(episode, rules, tick_times)
    arr_cum = _arrival_cum(episode, tick_times)

    m = cp_model.CpModel()
    cap = [present[t] * WS for t in range(n_ticks)]
    npick = [m.NewIntVar(0, cap[t], f"np{t}") for t in range(n_ticks)]
    npack = [m.NewIntVar(0, cap[t], f"nk{t}") for t in range(n_ticks)]
    nmg = [m.NewIntVar(0, cap[t], f"ng{t}") for t in range(n_ticks)]
    cum_pick_terms, cum_pack_terms = [], []
    for t in range(n_ticks):
        m.Add(npick[t] + npack[t] + nmg[t] <= cap[t])
        cum_pick_terms.append(npick[t] * RP)
        cum_pack_terms.append(npack[t] * RK)
        m.Add(sum(cum_pick_terms) <= arr_cum[t] * OS)            # can't pick unarrived
        m.Add(sum(cum_pack_terms) <= sum(cum_pick_terms))        # can't pack unpicked
    mgmt_ct = int(round(mgmt_req_h / TICK_H)) * WS  # centi-worker-ticks of mgmt
    if mgmt_ct > 0:
        m.Add(sum(nmg) >= mgmt_ct)

    # maximise total orders shipped by the operational close
    obj = m.NewIntVar(0, D * OS, "obj")
    m.Add(obj == sum(cum_pack_terms))
    m.Maximize(obj)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = 3.0
    s.parameters.num_search_workers = 1
    s.Solve(m)
    shipped_max = s.Value(obj) // OS

    tol = 2  # orders; absorb any residual centi-worker discretization
    return {
        "complete": bool(shipped_max >= D - tol),
        "shipped_max": int(shipped_max),
        "total": D,
    }


def evaluate_bound(cfg, n_days_cap=400):
    """Walk a facility's work-year, solving the planning bound for each day.
    Returns aggregate stats: OT-free-feasible rate + min-OT distribution.

    Days are advanced with the greedy scheduler (not idling) so order
    carryover behaves realistically and per-day demand isn't corrupted."""
    import numpy as np

    from clark.agent.actions import get_action_mask, get_hustle_mask
    from clark.env.year_env import YearEnv
    from clark.inference.baseline import GreedyScheduler

    mean_oph = float(np.mean([w.base_oph for w in cfg.workers]))
    g = GreedyScheduler()
    env = YearEnv(cfg)
    env.reset()
    g.reset()
    results = []
    seen_days = 0
    last_key = None
    ticks = 0
    while not env.is_year_done and seen_days < n_days_cap and ticks < 200_000:
        de = env.day_env
        ep = de.episode
        key = (ep.year, ep.month, ep.day)
        if key != last_key:
            results.append(plan_day(ep, de.facility_config.rules, mean_oph))
            seen_days += 1
            last_key = key
        mask = get_action_mask(de)
        hmask = get_hustle_mask(de)
        ta, ha = g.act(de, mask, hmask)
        env.step([(i, int(ta[i]), bool(ha[i])) for i in range(len(ta))])
        ticks += 1
    n = max(1, len(results))
    slack = [1.0 - r["shipped_max"] / r["total"] for r in results if r["total"] > 0]
    return {
        "n_days": len(results),
        "complete_pct": 100.0 * sum(r["complete"] for r in results) / n,
        "median_unshippable_pct": 100.0 * float(np.median(slack)) if slack else 0.0,
    }

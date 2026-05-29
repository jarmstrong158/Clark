"""Single-day opening-assignment inference primitive.

Used by `clark plan` (cli/main.py) and `clark serve` (clark/serve/app.py).
Both used to import these from cli.main as `_run_one_plan_day` /
`_sample_volume_for_date` — relocated here so the server no longer
depends on CLI internals. The CLI re-exports the old underscore names
for backwards compatibility with any external script that imported
the private names directly.
"""
from __future__ import annotations

import calendar
import random
from datetime import date


def sample_volume_for_date(config, target_date: date) -> tuple[int, str]:
    """Sample a realistic order volume for a specific date using the
    config's seasonal_ranges + weekly_curve. Returns (volume, label)
    where label is "Normal Day" / "Moderate Volume Day" / "High Volume Day"."""
    month_name = calendar.month_name[target_date.month]
    dow = target_date.weekday()  # 0=Monday, 4=Friday

    dow_name_map = {
        0: "monday", 1: "tuesday", 2: "wednesday",
        3: "thursday", 4: "friday",
    }

    # Seasonal base range
    ranges = config.volume.seasonal_ranges
    vol_range = None
    if month_name in ranges:
        vol_range = tuple(ranges[month_name])
    else:
        for k, v in ranges.items():
            if k.lower() == month_name.lower():
                vol_range = tuple(v)
                break
    if vol_range is None:
        vol_range = (100, 300)

    vol_lo, vol_hi = vol_range
    vol_spread = vol_hi - vol_lo

    # Apply weekly curve
    curve = config.volume.weekly_curve
    dow_name = dow_name_map.get(dow, "monday")
    if dow_name in curve:
        pct_lo, pct_hi = curve[dow_name]
    else:
        pct_lo, pct_hi = 0.5, 1.0

    day_vol_lo = vol_lo + int(vol_spread * pct_lo)
    day_vol_hi = vol_lo + int(vol_spread * pct_hi)
    volume = random.randint(day_vol_lo, max(day_vol_lo, day_vol_hi))

    # Label
    high_threshold = config.rules.high_volume_day_orders
    if volume >= high_threshold:
        label = "High Volume Day"
    elif volume >= high_threshold * 0.6:
        label = "Moderate Volume Day"
    else:
        label = "Normal Day"

    return volume, label


def run_one_plan_day(config, agent, target_date: date, volume: int,
                     forced_absent: set | None = None,
                     seed: int | None = None) -> list[tuple]:
    """Simulate the opening assignment for a single day by running the
    first agent step. Returns a list of (worker_name, task_name, hustle).

    forced_absent: worker names deterministically forced absent (bypasses
      the probabilistic/capped call-off roll) — for faithful what-if
      scenarios.
    seed: if set, seeds RNG so the plan is reproducible and what-if
      base vs modified differ ONLY by the modification, not by episode
      noise.

    Absent workers (forced OR naturally called off) are reported with
    task "absent" instead of their masked-idle action — the prior code
    zipped the agent vector against the static config list and silently
    misreported absent workers as having a real assignment.
    """
    from clark.env.year_env import YearEnv
    from clark.agent.state import StateBuilder
    from clark.agent.actions import get_action_mask, get_hustle_mask

    if seed is not None:
        import random as _random
        import numpy as _np
        import torch as _torch
        _random.seed(seed)
        _np.random.seed(seed)
        _torch.manual_seed(seed)            # the agent samples via torch
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)

    env = YearEnv(config)
    builder = StateBuilder(config)

    dow = target_date.weekday()
    month = target_date.month

    env.reset()
    # Re-initialize the inner day_env for this specific date/volume
    env.day_env.reset(
        force_month=month,
        force_dow=dow,
        force_volume=volume,
    )

    # A planning tool answers "what's the plan for the scenario you
    # specified" — it must NOT inject unrequested random call-offs
    # (that made /plan irreproducible and /what_if base-vs-modified a
    # muddy comparison). Clear natural absence, then apply ONLY the
    # explicitly forced absences. 1:1 index alignment: episode.workers
    # is built in config.workers order (episode_generator.py:184).
    ws = env.day_env.episode.workers
    fa = forced_absent or set()
    for i, cw in enumerate(config.workers):
        if i < len(ws):
            ws[i].is_absent = cw.name in fa

    agent.reset_hidden()

    state_dict = builder.build(env.day_env)
    mask = get_action_mask(env.day_env)
    hmask = get_hustle_mask(env.day_env)
    # select_action_from_dict returns 5: task/hustle actions, task_lp,
    # hustle_lp, value. Previously unpacked 4 -> `clark plan` crashed
    # with "too many values to unpack (expected 4)" the moment a
    # checkpoint existed. The test suite missed it (no end-to-end
    # plan-path test); test_plan_path.py now pins the 5-tuple contract.
    task_actions, hustle_actions, _task_lp, _hustle_lp, _value = (
        agent.select_action_from_dict(state_dict, mask, hustle_mask=hmask)
    )

    assignments = []
    for i, worker in enumerate(config.workers):
        if i < len(ws) and ws[i].is_absent:
            # Honest: an absent worker has no assignment, not a fake one.
            assignments.append((worker.name, "absent", False))
            continue
        task_idx = task_actions[i]
        task_name = (config.task_ids[task_idx]
                     if task_idx < len(config.task_ids) else "unknown")
        hustle = bool(hustle_actions[i])
        assignments.append((worker.name, task_name, hustle))

    return assignments


def run_day_outcome_samples(config, agent, target_date: date,
                              volume: int,
                              n_samples: int = 20,
                              base_seed: int | None = None,
                              forced_absent: set | None = None,
                              ) -> dict:
    """Run the same day's simulation N times with different seeds
    and aggregate the outcome. Distinct from run_one_plan_day (which
    captures only the opening) and run_full_day_schedule (which
    captures one day's tick-by-tick assignments): this returns
    distributions over the day's *outcome*.

    The agent's action sampling is stochastic; volume is fixed
    (we pass it in), but the env's per-day RNG (which workers
    naturally call off, etc.) varies seed-by-seed. So the spread
    here is "given this scenario, how often does it ship".

    Returns:
        {
          "n_samples": int,
          "grades": {"A": int, "B": int, "C": int, "D": int, "F": int},
          "grade_pct": {"A": float, ...},
          "completion_rate_mean": float,   # mean orders_shipped/orders_total
          "completion_rate_p10": float,    # 10th percentile (downside)
          "completion_rate_p90": float,    # 90th percentile (upside)
          "full_completion_pct": float,    # % of samples that shipped 100%
          "samples": [{"seed": int, "grade": str, "shipped": int,
                       "total": int, "completion": float}, ...]
        }
    """
    from clark.env.year_env import YearEnv
    from clark.agent.state import StateBuilder
    from clark.agent.actions import get_action_mask, get_hustle_mask
    import random as _random
    import numpy as _np
    import torch as _torch

    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    base_seed = base_seed if base_seed is not None else 12345

    grades_counter = {g: 0 for g in "ABCDF"}
    completions: list[float] = []
    full_comps = 0
    sample_rows: list[dict] = []

    dow = target_date.weekday()
    month = target_date.month
    builder = StateBuilder(config)
    fa = forced_absent or set()
    MAX_TICKS = 200

    for i in range(n_samples):
        seed = base_seed + i
        _random.seed(seed); _np.random.seed(seed)
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)

        env = YearEnv(config)
        env.reset()
        env.day_env.reset(force_month=month, force_dow=dow,
                           force_volume=volume)
        ws = env.day_env.episode.workers
        for j, cw in enumerate(config.workers):
            if j < len(ws):
                ws[j].is_absent = cw.name in fa

        agent.reset_hidden()
        ticks = 0
        day_done = False
        while not day_done and ticks < MAX_TICKS:
            state_dict = builder.build(env.day_env)
            mask = get_action_mask(env.day_env)
            hmask = get_hustle_mask(env.day_env)
            ta, ha, *_ = agent.select_action_from_dict(
                state_dict, mask, hustle_mask=hmask)
            actions = [(j, int(ta[j]), bool(ha[j]))
                       for j in range(len(ta))]
            _, _, day_done, info = env.step(actions)
            ticks += 1
            if info.get("new_day") or day_done:
                break

        # YearEnv finalizes a day_summary into daily_summaries on day_done.
        # If the env wrapper didn't append (timeout, edge), fall back to F.
        if env.daily_summaries:
            ft = env.daily_summaries[-1].get("footer", {})
            grade = ft.get("grade", "F")
            shipped = int(ft.get("orders_shipped", 0))
            total = int(ft.get("orders_total", 0) or 1)
            comp = shipped / total
            reasons = ft.get("grade_reasons", [])
        else:
            grade, shipped, total, comp = "F", 0, 0, 0.0
            reasons = [{"code": "sim_error",
                        "detail": "simulation did not finalize a day"}]

        grades_counter[grade] = grades_counter.get(grade, 0) + 1
        completions.append(comp)
        if comp >= 0.999:
            full_comps += 1
        sample_rows.append({"seed": seed, "grade": grade,
                            "shipped": shipped, "total": total,
                            "completion": round(comp, 4),
                            "reasons": reasons})

    n = float(n_samples)
    grade_pct = {g: round(c / n, 4) for g, c in grades_counter.items()}
    sorted_c = sorted(completions)
    def _pct(p):
        if not sorted_c: return 0.0
        idx = max(0, min(len(sorted_c) - 1,
                          int(round(p / 100.0 * (len(sorted_c) - 1)))))
        return round(sorted_c[idx], 4)

    # Aggregate WHY the F runs failed so the UI can break them down and the
    # operator can trust (or interrogate) the model. Count each cause once
    # per failing run; keep an example detail + worst completion per cause.
    from collections import Counter
    f_rows = [s for s in sample_rows if s["grade"] == "F"]
    f_codes = Counter()
    f_detail, f_worst_comp = {}, {}
    for s in f_rows:
        for code in {r["code"] for r in s["reasons"]}:
            f_codes[code] += 1
        for r in s["reasons"]:
            f_detail.setdefault(r["code"], r["detail"])
            f_worst_comp[r["code"]] = min(
                f_worst_comp.get(r["code"], 1.0), s["completion"])
    f_count = len(f_rows)
    f_reasons = [{
        "code": code,
        "count": cnt,
        "runs_pct": round(cnt / max(1, f_count), 4),
        "example": f_detail.get(code, ""),
        "worst_completion": round(f_worst_comp.get(code, 0.0), 4),
    } for code, cnt in f_codes.most_common()]

    return {
        "n_samples": n_samples,
        "grades": grades_counter,
        "grade_pct": grade_pct,
        "completion_rate_mean": round(sum(completions) / n, 4),
        "completion_rate_p10": _pct(10),
        "completion_rate_p90": _pct(90),
        "full_completion_pct": round(full_comps / n, 4),
        "f_count": f_count,
        "f_reasons": f_reasons,
        "samples": sample_rows,
    }


def _coalesce_blocks(blocks: list[dict], min_h: float) -> list[dict]:
    """Turn a per-tick task timeline into human-readable blocks WITHOUT
    misattributing time. Two passes:

      1. Merge adjacent runs of the same (task, hustle) — always correct,
         they're literally the same contiguous work.
      2. Smooth genuine one-tick flicker (A → b → A, where b is shorter than
         min_h and both neighbors are the same task) by collapsing the trio
         into one A block.

    We deliberately do NOT fold an arbitrary short block into whatever block
    precedes it. The old behavior ("dur < min_h → extend the prior block")
    relabeled that time as the prior task, so a real task cap (e.g.
    cycle_count masked off after its daily-hours limit) followed by the
    policy thrashing across many one-tick tasks painted the whole stretch as
    the pre-cap task — making a 0.5h cap render as hours. The schedule must
    reflect what the worker actually did, even if that's a noisy run of short
    blocks.
    """
    # Pass 1: drop zero-duration, merge adjacent identical (task, hustle).
    merged: list[dict] = []
    for b in blocks:
        if b["end"] - b["start"] <= 0:
            continue
        if (merged and merged[-1]["task"] == b["task"]
                and merged[-1]["hustle"] == b["hustle"]):
            merged[-1]["end"] = b["end"]
        else:
            merged.append(dict(b))
    # Pass 2: collapse true A-b-A flicker. Repeat until stable.
    i = 1
    while i < len(merged) - 1:
        b = merged[i]
        same_neighbors = (
            merged[i - 1]["task"] == merged[i + 1]["task"]
            and merged[i - 1]["hustle"] == merged[i + 1]["hustle"]
        )
        if (b["end"] - b["start"]) < min_h and same_neighbors:
            merged[i - 1]["end"] = merged[i + 1]["end"]
            del merged[i:i + 2]
            i = max(1, i - 1)
        else:
            i += 1
    return merged


def run_full_day_schedule(config, agent, target_date: date,
                           volume: int,
                           forced_absent: set | None = None,
                           seed: int | None = None) -> dict:
    """Run a full day's simulation and return per-worker time blocks
    showing every task change. STEP_DURATION = 10 min, so a day has
    ~60-80 ticks; we aggregate runs of the same (task, hustle) into
    blocks so the result is human-readable instead of a 60-row noise.

    Returns:
        {
          "shift_start": float,     # 24h clock
          "shift_end":   float,
          "tick_minutes": int,
          "workers": [
            {"name": str, "blocks": [
              {"start": float, "end": float, "task": str, "hustle": bool},
              ...
            ]},
            ...
          ]
        }

    Same seed/forced_absent contract as run_one_plan_day: the schedule
    is fully reproducible and any forced absences are applied
    deterministically (not via the call-off roll).
    """
    from clark.env.year_env import YearEnv
    from clark.env.episode_generator import STEP_DURATION
    from clark.agent.state import StateBuilder
    from clark.agent.actions import get_action_mask, get_hustle_mask

    if seed is not None:
        import random as _random
        import numpy as _np
        import torch as _torch
        _random.seed(seed)
        _np.random.seed(seed)
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)

    env = YearEnv(config)
    builder = StateBuilder(config)
    dow = target_date.weekday()
    month = target_date.month
    env.reset()
    env.day_env.reset(force_month=month, force_dow=dow,
                       force_volume=volume)
    ws = env.day_env.episode.workers
    fa = forced_absent or set()
    for i, cw in enumerate(config.workers):
        if i < len(ws):
            ws[i].is_absent = cw.name in fa

    agent.reset_hidden()
    shift_start = float(env.day_env.current_hour)
    # Per-worker block list. Each entry is the OPEN block; closed
    # when the (task, hustle) tuple changes.
    n = len(config.workers)
    blocks_per_worker: list[list[dict]] = [[] for _ in range(n)]
    last_signature: list[tuple | None] = [None] * n

    # Run day to completion. Safety cap so a bug can't infinite-loop.
    MAX_TICKS = 200
    ticks = 0
    day_done = False
    while not day_done and ticks < MAX_TICKS:
        state_dict = builder.build(env.day_env)
        mask = get_action_mask(env.day_env)
        hmask = get_hustle_mask(env.day_env)
        task_actions, hustle_actions, *_ = agent.select_action_from_dict(
            state_dict, mask, hustle_mask=hmask)

        hour_before = float(env.day_env.current_hour)
        # Record what each worker is doing AT THIS TICK.
        for i, worker in enumerate(config.workers):
            if i < len(ws) and ws[i].is_absent:
                sig = ("absent", False)
            elif i >= len(task_actions):
                sig = ("unknown", False)
            else:
                ti = task_actions[i]
                tname = (config.task_ids[ti]
                         if ti < len(config.task_ids) else "unknown")
                sig = (tname, bool(hustle_actions[i]))
            if sig != last_signature[i]:
                # Close prior open block (if any) at hour_before.
                if blocks_per_worker[i]:
                    blocks_per_worker[i][-1]["end"] = hour_before
                # Open a new block.
                blocks_per_worker[i].append({
                    "start": hour_before,
                    "end": None,  # filled in when closed
                    "task": sig[0],
                    "hustle": sig[1],
                })
                last_signature[i] = sig

        # Advance one tick.
        actions = [(i, int(task_actions[i]), bool(hustle_actions[i]))
                   for i in range(len(task_actions))]
        _, _, day_done, info = env.step(actions)
        ticks += 1
        # YearEnv rolls to a new day when day_done; we stop there.
        if info.get("new_day") or day_done:
            break

    # End-of-shift: YearEnv auto-advances to the next day on day_done,
    # so env.day_env.current_hour right now points at the NEXT day's
    # start (e.g. 8.0) instead of the just-finished day's eod. Pull
    # the true eod from the config — that's what the policy was
    # planning against. BUT if the simulation ran into OT, blocks
    # may extend past eod; use the max of (eod, last block start +
    # one tick) so the timeline rendering covers everything.
    eod = float(getattr(config.rules, "eod_hour",
                          config.rules.day_start_hour + 8.0))
    last_block_starts = [b["start"] for w in blocks_per_worker for b in w]
    if last_block_starts:
        shift_end = max(eod, max(last_block_starts) + STEP_DURATION)
    else:
        shift_end = eod
    for blocks in blocks_per_worker:
        if blocks and blocks[-1]["end"] is None:
            blocks[-1]["end"] = shift_end

    # Snap every block boundary to the nearest tick — the env's
    # current_hour accumulates floating-point drift over 60+ ticks,
    # so a value that should be 12.0 becomes 11.999999..., which the
    # frontend then formats as "11:60" (Math.floor=11, round(0.999*
    # 60)=60). Snap to ticks here so all downstream rendering deals
    # with clean numbers, AND drop any zero-duration blocks the snap
    # might produce.
    def _snap(h: float) -> float:
        return round(h / STEP_DURATION) * STEP_DURATION

    for blocks in blocks_per_worker:
        for b in blocks:
            b["start"] = _snap(b["start"])
            b["end"] = _snap(b["end"])

    # Build human-readable blocks WITHOUT misattributing time.
    MIN_BLOCK_MINUTES = 20  # ≥ 2 ticks at 10-min STEP_DURATION
    min_h = MIN_BLOCK_MINUTES / 60.0
    cleaned_per_worker = [_coalesce_blocks(blocks, min_h)
                          for blocks in blocks_per_worker]

    # Snap shift bounds too so the frontend axis labels line up
    # with snapped block edges.
    shift_start = _snap(shift_start)
    shift_end = _snap(shift_end)

    return {
        "shift_start": shift_start,
        "shift_end": shift_end,
        "tick_minutes": int(round(STEP_DURATION * 60)),
        "workers": [
            {"name": config.workers[i].name,
             "blocks": cleaned_per_worker[i]}
            for i in range(n)
        ],
    }

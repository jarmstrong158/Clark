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

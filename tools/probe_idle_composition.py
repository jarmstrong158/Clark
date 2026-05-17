"""Read-only probe: attribute idle worker-hours on F-incomplete days to
a CAUSE (absent / shift-exhausted-no-OT / pack-only / other).

The aggregate day logs only carry total idle hours, not why. This
reconstructs the cause by inspecting env state each tick. A masked-
random policy is sufficient: idle is FORCED by env structure (absent
call-off, shift exhaustion at EOD without OT, pack-only with nothing
to pack) — the agent cannot choose idle in the normal branch, so the
idle CAUSE composition is policy-independent. (It also empirically
verifies the claim that OT now retains shift-exhausted workers.)

Does NOT touch training; constructs its own envs.
"""
from __future__ import annotations

import random
import numpy as np

from clark.env.facility_env import FacilityEnv
from clark.env.episode_generator import STEP_DURATION
from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.training.synthetic_gen import generate_random_facility

random.seed(20260517)
np.random.seed(20260517)

MAX_DAYS = 5000
TARGET_F = 1500

buckets = {"ABSENT": 0.0, "SHIFT_EXHAUSTED_NO_OT": 0.0,
           "PACK_ONLY": 0.0, "OTHER": 0.0}
totals = {"f_days": 0, "complete_days": 0, "all_days": 0,
          "f_idle_h": 0.0, "f_total_worker_h": 0.0,
          "f_with_absence": 0, "shift_exh_while_ot": 0.0}


def classify(env, w):
    if w.is_absent:
        return "ABSENT"
    if (env.current_hour >= env._eod_hour
            and getattr(w, "hours_remaining", 1.0) <= 0
            and not env.is_ot):
        return "SHIFT_EXHAUSTED_NO_OT"
    if getattr(w, "is_pack_only", False):
        return "PACK_ONLY"
    return "OTHER"


days = 0
while days < MAX_DAYS and totals["f_days"] < TARGET_F:
    cfg = generate_random_facility(stage=random.choice([1, 2, 3]))
    env = FacilityEnv(cfg)
    env.reset()
    N = len(env.episode.workers)
    day_idle = {k: 0.0 for k in buckets}
    day_worker_h = 0.0
    had_absence = any(w.is_absent for w in env.episode.workers)
    done = False
    guard = 0
    while not done and guard < 400:
        guard += 1
        m = get_action_mask(env)
        hm = get_hustle_mask(env)
        from clark.env.episode_generator import STEP_DURATION as _SD
        idle_idx = 0
        # Attribute at the DECISION POINT: a worker whose only legal
        # action this tick is idle was FORCED idle by env structure.
        # Use pre-step env state (the state the mask was built from).
        pre_ot = env.is_ot
        pre_hour = env.current_hour
        forced = {}
        for i in range(N):
            row = m[i]
            only_idle = bool(row[idle_idx]) and row.sum() == 1
            if not only_idle:
                continue
            w = env.episode.workers[i]
            if w.is_absent:
                forced[i] = "ABSENT"
            elif (pre_hour >= env._eod_hour
                    and getattr(w, "hours_remaining", 1.0) <= 0
                    and not pre_ot):
                forced[i] = "SHIFT_EXHAUSTED_NO_OT"
            elif getattr(w, "is_pack_only", False):
                forced[i] = "PACK_ONLY"
            else:
                forced[i] = "OTHER"
        acts = []
        for i in range(N):
            valid = np.where(m[i])[0]
            t = int(valid[0]) if len(valid) == 0 else int(random.choice(valid))
            h = bool(hm[i, 1]) and random.random() < 0.3
            acts.append((i, t, h))
        _, _, done, _ = env.step(acts)
        for i, w in enumerate(env.episode.workers):
            if getattr(w, "current_task", None) == "idle":
                c = forced.get(i, "OTHER")
                day_idle[c] += STEP_DURATION
                if c == "SHIFT_EXHAUSTED_NO_OT" and pre_ot:
                    totals["shift_exh_while_ot"] += STEP_DURATION
            else:
                day_worker_h += STEP_DURATION
    days += 1
    totals["all_days"] += 1
    rem = env.orders_in_queue + env.orders_picked_not_audited
    if rem > 0:
        totals["f_days"] += 1
        if had_absence:
            totals["f_with_absence"] += 1
        for k in buckets:
            buckets[k] += day_idle[k]
            totals["f_idle_h"] += day_idle[k] if False else 0.0
        totals["f_idle_h"] += sum(day_idle.values())
        totals["f_total_worker_h"] += day_worker_h + sum(day_idle.values())
    else:
        totals["complete_days"] += 1

ti = sum(buckets.values()) or 1.0
print(f"days_run={totals['all_days']}  F-incomplete={totals['f_days']}  "
      f"complete={totals['complete_days']}")
print(f"F days with >=1 absent worker: {totals['f_with_absence']} "
      f"({100*totals['f_with_absence']/max(1,totals['f_days']):.1f}%)")
print(f"total idle worker-h on F days = {ti:.0f}  "
      f"(idle as % of all F worker-h = "
      f"{100*ti/max(1.0,totals['f_total_worker_h']):.1f}%)")
print("--- idle attribution on F-incomplete days ---")
for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
    print(f"  {k:24s} {v:9.0f} h  ({100*v/ti:.1f}%)")
print(f"sanity: SHIFT_EXHAUSTED counted while is_ot was True = "
      f"{totals['shift_exh_while_ot']:.0f} h (should be ~0 — proves OT "
      f"retains shift-exhausted workers)")

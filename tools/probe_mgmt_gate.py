"""Read-only probe: how often is the new management gate actually OPEN?

The dec-028 gate pays the dense management bonus only while
`pending_orders <= 0.10 * total_orders` ("shipping under control").
The PRIOR gate (pending<=0) was inert because that condition was true
only in the last ~0.5h of a day. This measures whether the NEW gate
has a real, multi-hour, actionable window — using the actual trained
policy so the pending trajectory is faithful (a random policy never
ships enough to open the gate, which would mislead).

Does NOT touch training: own env instances, loads a STABLE backup
checkpoint (not the live file the trainer is writing).
"""
from __future__ import annotations
import random, statistics
import numpy as np

from clark.env.facility_env import FacilityEnv
from clark.env.episode_generator import STEP_DURATION
from clark.agent.ppo import ClarkAgent
from clark.agent.state import StateBuilder
from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.training.synthetic_gen import generate_random_facility

CKPT = r"C:\Users\jarms\repos\clark\clark\data\checkpoints\clark_foundation.pt.preMgmtGateFix.bak"
random.seed(7); np.random.seed(7)
agent = ClarkAgent.load(CKPT)

N_DAYS = 400
per_day_open_h = []      # hours/day the gate is open
per_day_open_actionable_h = []  # gate open AND a mgmt-eligible worker non-absent
per_day_total_h = []
days_ge2h = 0            # days with >=2h open (enough to accrue the 2-6h req)
days_zero = 0            # days the gate never opened
gate_open_tick_frac = []

d = 0
while d < N_DAYS:
    cfg = generate_random_facility(stage=random.choice([1, 2, 3]))
    env = FacilityEnv(cfg)
    env.reset()
    builder = StateBuilder(cfg)
    elig = env._mgmt_eligible_ids
    N = len(env.episode.workers)
    open_ticks = 0; open_actionable = 0; ticks = 0
    done = False; guard = 0
    while not done and guard < 400:
        guard += 1
        sd = builder.build(env)
        m = get_action_mask(env); hm = get_hustle_mask(env)
        ta, ha, *_ = agent.select_action_from_dict(sd, m, hustle_mask=hm)
        acts = [(i, int(ta[i]), bool(ha[i])) for i in range(N)]
        _, _, done, _ = env.step(acts)
        ticks += 1
        total = max(1, env.episode.total_orders)
        pending = env.orders_in_queue + env.orders_picked_not_audited
        if pending <= 0.10 * total:
            open_ticks += 1
            if any((w.worker_id in elig and not w.is_absent)
                   for w in env.episode.workers):
                open_actionable += 1
    oh = open_ticks * STEP_DURATION
    per_day_open_h.append(oh)
    per_day_open_actionable_h.append(open_actionable * STEP_DURATION)
    per_day_total_h.append(ticks * STEP_DURATION)
    gate_open_tick_frac.append(open_ticks / max(1, ticks))
    if oh >= 2.0: days_ge2h += 1
    if open_ticks == 0: days_zero += 1
    d += 1

n = len(per_day_open_h)
print(f"days={n}  (real trained policy from preMgmtGateFix.bak)")
print(f"gate-open hours/day: mean={statistics.mean(per_day_open_h):.2f} "
      f"median={statistics.median(per_day_open_h):.2f} "
      f"p10={sorted(per_day_open_h)[n//10]:.2f} "
      f"p90={sorted(per_day_open_h)[9*n//10]:.2f} "
      f"max={max(per_day_open_h):.2f}")
print(f"gate-open hours/day WITH an eligible worker present: "
      f"mean={statistics.mean(per_day_open_actionable_h):.2f}")
print(f"gate open as % of day ticks: mean={100*statistics.mean(gate_open_tick_frac):.1f}%")
print(f"days with >=2h open (can accrue the 2-6h mgmt req): "
      f"{days_ge2h}/{n} ({100*days_ge2h/n:.1f}%)")
print(f"days gate NEVER opened (still inert): {days_zero}/{n} "
      f"({100*days_zero/n:.1f}%)")

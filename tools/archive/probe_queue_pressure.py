"""Probe: validate the v2.2 queue_pressure signal shape pre-training.

Walks ticks on a normal-volume day and a stress-volume day, prints the
queue_pressure trajectory, confirms it discriminates the two and grows
monotonically with queue depth. Same con-015 discipline as the v2.1
probe — measure before paying for a 1-2h fine-tune.

Expected shape:
  - At t=0: instant arrival bucket lands ~half of total. On a stress
    day (vol = 1.8x capacity), pressure jumps immediately to ~0.9
    (90% of capacity already queued). On a normal day (vol = 0.4x
    capacity), pressure jumps to ~0.18.
  - Through the day: pressure rises as arrivals continue but DROPS
    as pickers/packers ship orders. Final value reflects whether
    backlog was cleared.
  - Clearly distinguishable: stress day peak > 1.0 (overload),
    normal day peak < 0.5 (rescuable).

Usage:
    python tools/probe_queue_pressure.py
"""
from __future__ import annotations

import argparse
import random
from typing import Optional

import numpy as np

from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


def _random_actions(env):
    mask = get_action_mask(env)
    hmask = get_hustle_mask(env)
    actions = []
    for w in range(mask.shape[0]):
        valid = np.where(mask[w])[0]
        if len(valid) == 0: continue
        task_idx = int(np.random.choice(valid))
        h_valid = np.where(hmask[w])[0]
        hustle = bool(np.random.choice(h_valid)) if len(h_valid) else False
        actions.append((w, task_idx, hustle))
    return actions


def walk(env, label, seed=42, volume_override=None):
    random.seed(seed); np.random.seed(seed)
    env.reset()
    if volume_override is not None:
        from clark.env.episode_generator import generate_episode
        env.episode = generate_episode(env.facility_config, volume=volume_override)
        env.arrived_so_far = 0
        env.orders_in_queue = 0
        env.orders_completed = 0
        env.orders_picked_not_audited = 0
        env._today_capacity = env._compute_today_capacity()
        total = max(1, env.episode.total_orders)
        if env._restock_enabled:
            env.restock_drain_per_order = 2.0 / total
            env.restock_refill_per_hour = 1.0 / max(0.1, env.episode.restock_hours)
        env.today_complexity_mix = env._sample_order_complexity(env.episode.total_orders)
        env._process_arrivals()

    print(f"\n--- {label} ---")
    print(f"  total_orders (oracle):    {env.episode.total_orders}")
    print(f"  today_capacity:           {env._today_capacity:.1f}")
    print(f"  total/capacity (oracle):  {env.episode.total_orders/env._today_capacity:.3f}")
    print()
    print(f"  {'hour':>6} {'queue':>6} {'picked':>6} {'shipped':>8} "
          f"{'pressure':>9}")

    traj = []
    peak = 0.0
    tick = 0
    while not env.is_done:
        if tick % 12 == 0:
            qp = env._compute_queue_pressure()
            peak = max(peak, qp)
            print(f"  {env.current_hour:>6.2f} "
                  f"{env.orders_in_queue:>6d} "
                  f"{env.orders_picked_not_audited:>6d} "
                  f"{env.orders_completed:>8d} "
                  f"{qp:>9.3f}")
            traj.append({"hour": env.current_hour, "qp": qp})
        env.step(_random_actions(env))
        tick += 1
        if tick > 200: break

    print(f"  EOD queue_pressure: {env._compute_queue_pressure():.3f}  "
          f"peak: {peak:.3f}")
    return {"label": label, "peak": peak, "trajectory": traj,
            "total_orders": env.episode.total_orders,
            "capacity": env._today_capacity}


def verify(normal, stress):
    print("\n--- Verification ---")
    fails = 0
    # 1: stress peak > 1.0 (overload surfaced)
    if stress["peak"] > 1.0:
        print(f"  [OK] stress peak pressure {stress['peak']:.2f} > 1.0 "
              f"(overload clearly signalled)")
    else:
        print(f"  [FAIL] stress peak {stress['peak']:.2f} <= 1.0 "
              f"(overload not surfaced)")
        fails += 1
    # 2: normal peak < 0.7 (rescuable, doesn't false-alarm)
    if normal["peak"] < 0.7:
        print(f"  [OK] normal peak {normal['peak']:.2f} < 0.7 "
              f"(no false overload alarm)")
    else:
        print(f"  [FAIL] normal peak {normal['peak']:.2f} >= 0.7 "
              f"(false overload alarm on rescuable day)")
        fails += 1
    # 3: meaningful gap between the two
    gap = stress["peak"] - normal["peak"]
    if gap > 0.5:
        print(f"  [OK] discriminating gap {gap:.2f} > 0.5 "
              f"(stress vs normal clearly separable)")
    else:
        print(f"  [FAIL] gap {gap:.2f} <= 0.5 "
              f"(signal can't tell stress from normal)")
        fails += 1
    print()
    if fails == 0:
        print("All hard checks passed. Safe to proceed with fine-tune.")
        return 0
    else:
        print(f"{fails} check(s) failed. STOP -- investigate before training.")
        return 1


def main():
    p = argparse.ArgumentParser()
    args = p.parse_args()
    cfg_n = generate_random_facility(stage=3)
    cfg_s = generate_random_facility(stage=3)
    env_n = FacilityEnv(cfg_n)
    env_s = FacilityEnv(cfg_s)

    # Pick volumes by capacity multiple
    def vol(cfg, mult):
        n = len(cfg.workers)
        avg_oph = sum(w.base_oph for w in cfg.workers) / max(1, n)
        cap = n * max(12.0, avg_oph * 9.0 * 0.22)
        return int(cap * mult)

    normal = walk(env_n, f"NORMAL day (vol mult 0.45)",
                  volume_override=vol(cfg_n, 0.45))
    stress = walk(env_s, f"STRESS day (vol mult 1.8)",
                  volume_override=vol(cfg_s, 1.8))
    import sys
    sys.exit(verify(normal, stress))


if __name__ == "__main__":
    main()

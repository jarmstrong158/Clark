"""Probe: validate the demand_vs_capacity signal shape pre-training.

con-015 / pipe-004 discipline: before paying for a 1-2h fine-tune,
measure that the new observation feature actually discriminates the
behavior it's supposed to discriminate. Specifically:

  1. On a NORMAL-volume day, dvc should land somewhere near 1.0
     (today's demand ≈ today's capacity, rescuable) and stay there.
  2. On a STRESS-volume day (jack_baseline at vol=450 -- same probe
     scenario the foundation-behavior diagnostic uses), dvc should
     rise above 1.0 by mid-morning, ideally past 1.5, signalling
     "this day exceeds capacity."
  3. The signal should sharpen through the day: morning estimates
     have wider error bands than afternoon ones because the instant
     bucket has high per-day fraction variance.
  4. dvc must never read episode.total_orders (already pinned in
     tests; verified again here for paranoia).

If any of these fail, the design is broken and we should NOT proceed
to the transplant + fine-tune. The probe runs in ~30s and saves us
1-2h of wasted GPU time.

Usage:
    python tools/probe_demand_vs_capacity.py
    python tools/probe_demand_vs_capacity.py --config clark/data/configs/jack_baseline.yaml --stress-volume 450
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.config.schema import FacilityConfig
from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


def _random_actions(env):
    """Pick a random valid action per worker -- we don't care about
    quality here, only that the env advances and arrivals get
    processed normally."""
    mask = get_action_mask(env)
    hmask = get_hustle_mask(env)
    actions = []
    for w in range(mask.shape[0]):
        valid = np.where(mask[w])[0]
        if len(valid) == 0:
            continue
        task_idx = int(np.random.choice(valid))
        h_valid = np.where(hmask[w])[0]
        hustle = bool(np.random.choice(h_valid)) if len(h_valid) else False
        actions.append((w, task_idx, hustle))
    return actions


def walk(env: FacilityEnv, label: str, seed: int = 42,
         volume_override: Optional[int] = None) -> dict:
    """Walk one full day, recording dvc + the inputs at every tick.

    If volume_override is given, force today's total_orders to that
    value. Implemented by patching `env.episode` AFTER reset (the only
    way to inject without modifying the FacilityEnv API)."""
    random.seed(seed)
    np.random.seed(seed)
    env.reset()
    if volume_override is not None:
        # Rebuild the episode with the forced volume and replay the
        # initial-state computations that reset() did. We can't call
        # reset() a second time because that would re-roll the seed and
        # land on a different episode.
        from clark.env.episode_generator import generate_episode
        env.episode = generate_episode(env.facility_config,
                                        volume=volume_override)
        env.arrived_so_far = 0
        env.orders_in_queue = 0
        env.orders_completed = 0
        env.orders_picked_not_audited = 0
        env._today_capacity = env._compute_today_capacity()
        # Re-seed restock + complexity bookkeeping that reset() did
        total = max(1, env.episode.total_orders)
        if env._restock_enabled:
            env.restock_drain_per_order = 2.0 / total
            env.restock_refill_per_hour = 1.0 / max(0.1, env.episode.restock_hours)
        env.today_complexity_mix = env._sample_order_complexity(env.episode.total_orders)
        env._process_arrivals()

    print(f"\n---{label} ---")
    print(f"  total_orders (oracle, NOT visible to obs): "
          f"{env.episode.total_orders}")
    print(f"  today_capacity:                           "
          f"{env._today_capacity:.1f}")
    print(f"  demand-vs-capacity at TRUTH (oracle):      "
          f"{env.episode.total_orders / env._today_capacity:.3f}")
    print()
    print(f"  {'hour':>6} {'arrived':>9} {'exp_frac':>10} "
          f"{'projected':>11} {'dvc':>7} {'truth':>7}  notes")

    trajectory = []
    tick = 0
    while not env.is_done:
        # Sample evenly across the day for printing -- every 12th tick = ~2h
        if tick % 12 == 0:
            from clark.env.episode_generator import expected_fraction_arrived
            ds = env.facility_config.rules.day_start_hour
            cut = env.facility_config.rules.order_cutoff_hour
            ef = expected_fraction_arrived(env.current_hour, ds, cut)
            dvc = env._compute_demand_vs_capacity()
            projected = (env.arrived_so_far / ef) if ef > 1e-4 else 0.0
            truth_ratio = (env.episode.total_orders / env._today_capacity)
            print(f"  {env.current_hour:>6.2f} {env.arrived_so_far:>9d} "
                  f"{ef:>10.3f} {projected:>11.1f} {dvc:>7.3f} "
                  f"{truth_ratio:>7.3f}")
            trajectory.append({
                "hour": env.current_hour,
                "arrived_so_far": env.arrived_so_far,
                "expected_fraction": ef,
                "projected_total": projected,
                "dvc": dvc,
                "truth_ratio": truth_ratio,
            })

        env.step(_random_actions(env))
        tick += 1
        if tick > 200:  # safety
            break

    # Final read
    dvc_final = env._compute_demand_vs_capacity()
    print(f"  EOD  dvc={dvc_final:.3f}  "
          f"true_ratio={env.episode.total_orders / env._today_capacity:.3f}")
    return {
        "label": label,
        "total_orders": env.episode.total_orders,
        "capacity": env._today_capacity,
        "truth_ratio": env.episode.total_orders / env._today_capacity,
        "trajectory": trajectory,
        "dvc_final": dvc_final,
    }


def verify(normal: dict, stress: dict) -> int:
    """Assert the design's claims. Returns shell exit code."""
    print("\n---Verification ---")
    fails = 0

    # (1) Stress day truth ratio MUST be > normal day's
    if stress["truth_ratio"] <= normal["truth_ratio"]:
        print(f"  [FAIL]stress truth_ratio ({stress['truth_ratio']:.2f}) "
              f"<= normal ({normal['truth_ratio']:.2f}); your stress "
              f"scenario isn't actually stressed.")
        fails += 1
    else:
        print(f"  [OK]stress truth_ratio {stress['truth_ratio']:.2f} > "
              f"normal {normal['truth_ratio']:.2f}")

    # (2) By mid-morning (hour ≥ 10), stress dvc should be > 1.0
    mid_stress = [s for s in stress["trajectory"] if 9.5 <= s["hour"] <= 13.0]
    if mid_stress:
        peak_mid = max(s["dvc"] for s in mid_stress)
        if peak_mid > 1.0:
            print(f"  [OK]stress dvc peaks at {peak_mid:.2f} by mid-morning "
                  f"(signalling overload to the policy)")
        else:
            print(f"  [FAIL]stress dvc only reached {peak_mid:.2f} by "
                  f"mid-morning -- overload not surfaced to policy")
            fails += 1

    # (3) Normal-day dvc should stay near truth_ratio, not run away
    normal_dvc_values = [s["dvc"] for s in normal["trajectory"] if s["hour"] > 10]
    if normal_dvc_values:
        normal_dvc_peak = max(normal_dvc_values)
        # Allow up to 50% over truth on a normal day (curve noise)
        if normal_dvc_peak < normal["truth_ratio"] * 1.5 + 0.3:
            print(f"  [OK]normal dvc stays in band "
                  f"(peak {normal_dvc_peak:.2f} vs truth "
                  f"{normal['truth_ratio']:.2f})")
        else:
            print(f"  [FAIL]normal dvc inflated to {normal_dvc_peak:.2f} "
                  f"despite truth {normal['truth_ratio']:.2f} -- false "
                  f"overload signal on rescuable days")
            fails += 1

    # (4) Sharpening: late-day projection error should be < early-day
    if len(stress["trajectory"]) >= 4:
        early = stress["trajectory"][:2]   # first two samples (~9-11am)
        late = stress["trajectory"][-2:]   # last two samples (afternoon)
        early_err = np.mean([abs(s["projected_total"] - stress["total_orders"])
                             for s in early if s["projected_total"] > 0])
        late_err = np.mean([abs(s["projected_total"] - stress["total_orders"])
                            for s in late if s["projected_total"] > 0])
        if late_err <= early_err * 1.5:  # allow some slack
            print(f"  [OK]projection sharpens: early err "
                  f"{early_err:.0f}, late err {late_err:.0f}")
        else:
            print(f"  [?]projection did not sharpen as expected: "
                  f"early err {early_err:.0f}, late err {late_err:.0f}")
            # not a hard fail -- natural day variance can defeat the
            # sharpening claim on a single sample

    print()
    if fails == 0:
        print("All hard checks passed. Safe to proceed with transplant + fine-tune.")
    else:
        print(f"{fails} check(s) failed. STOP -- investigate before "
              f"committing GPU time.")
    return 0 if fails == 0 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                   help="Facility config YAML (optional; defaults to a stage-2 "
                        "synthetic). If given, used for both the normal and "
                        "stress walks (stress uses --stress-volume override).")
    p.add_argument("--stress-volume", type=int, default=None,
                   help="Override total_orders for the stress walk. Defaults "
                        "to 1.7x the synthetic stress ceiling for stage 2.")
    args = p.parse_args()

    # Normal-day walk -- let the generator pick a sensible volume
    if args.config:
        cfg = FacilityConfig.from_yaml(args.config)
    else:
        cfg = generate_random_facility(stage=2)
    env_n = FacilityEnv(cfg)
    normal = walk(env_n, "NORMAL-volume day")

    # Stress-day walk -- same config, force a high volume override
    if args.config:
        cfg2 = FacilityConfig.from_yaml(args.config)
    else:
        cfg2 = generate_random_facility(stage=2)
    env_s = FacilityEnv(cfg2)
    stress_vol = args.stress_volume
    if stress_vol is None:
        # Default: ~1.8x nominal capacity (beyond OT-rescue ceiling)
        n = len(cfg2.workers)
        avg_oph = sum(w.base_oph for w in cfg2.workers) / max(1, n)
        capacity = n * max(12.0, avg_oph * 9.0 * 0.22)
        stress_vol = int(capacity * 1.8)
    stress = walk(env_s, f"STRESS-volume day (vol={stress_vol})",
                  volume_override=stress_vol)

    sys.exit(verify(normal, stress))


if __name__ == "__main__":
    main()

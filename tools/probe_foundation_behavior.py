"""Probe the current foundation policy's behavior on jack_baseline.

Three measurements, each justifying or refuting a planned Restart:

  P1: per_filler_unit contribution per day, split by grade.
      Justifies Restart 1 (dec-012 cleanup, zero out filler reward).
      If P1 shows large positive filler reward accumulating on F-days,
      removing it is the right move. If F-days already have ~zero
      filler reward, the change is cosmetic.

  P2: Crunch-cap binding frequency + max breach.
      Justifies Restart 2 (cap fix + tighten).
      If cap binds rarely OR doesn't overshoot, the "free filler
      after cap" exploit isn't actually being learned and Restart 2
      is solving a non-problem. If cap binds often and leaks far past
      -5000, it's the smoking gun for the afternoon-side_project
      pattern in jack_baseline.

  P3: Buffer-full incidents during non-OT hours.
      Justifies Restart 3 (mask out non-pack when buffer full).
      Counts non-OT ticks where pick_buffer_full was true, and what
      task each affected worker actually chose. If "filler" shows up
      frequently in those ticks, structural enforcement is needed.

Per con-015 (verification-discipline): measure before shipping any
gated/threshold-based change.

Run:
    python tools/probe_foundation_behavior.py [--days N] [--seed S] [--facility ID]

Outputs a human-readable report to stdout + writes
tools/_probe_results.json for downstream comparison after each Restart.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facility", default="jack_baseline",
                    help="Facility id (must exist in clark/data/configs).")
    ap.add_argument("--ckpt",
                    default="clark/data/checkpoints/clark_foundation.pt",
                    help="Foundation checkpoint to probe.")
    ap.add_argument("--days", type=int, default=40,
                    help="Number of simulated days to run.")
    ap.add_argument("--seed", type=int, default=20260524,
                    help="RNG seed for reproducibility.")
    ap.add_argument("--volume", type=int, default=None,
                    help="Force this exact daily volume (else season-sampled).")
    ap.add_argument("--out", default="tools/_probe_results.json")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "clark" / "data" / "configs" / f"{args.facility}.yaml"
    if not cfg_path.exists():
        # try user/ dir as fallback
        cfg_path = repo_root / "clark" / "data" / "configs" / "user" / f"{args.facility}.yaml"
    if not cfg_path.exists():
        print(f"ERROR: facility {args.facility!r} not found", file=sys.stderr)
        sys.exit(2)
    ckpt_path = repo_root / args.ckpt
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not at {ckpt_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Probing {args.facility} with {ckpt_path.name} for {args.days} days...")
    print()
    _seed_everything(args.seed)

    from clark.agent.ppo import ClarkAgent
    from clark.agent.state import StateBuilder
    from clark.agent.actions import get_action_mask, get_hustle_mask
    from clark.config.schema import FacilityConfig
    from clark.env.year_env import YearEnv

    cfg = FacilityConfig.from_yaml(str(cfg_path))
    agent = ClarkAgent.load(str(ckpt_path), device="cpu")
    env = YearEnv(cfg)
    builder = StateBuilder(cfg)
    env.reset()
    # Force every day's volume to the override (else season-sampled).
    # YearEnv pre-builds a work_days list at reset(); patch each entry.
    if args.volume is not None:
        for wd in env.work_days:
            wd["volume"] = args.volume
        print(f"  forcing daily volume = {args.volume}")
        print()
    agent.reset_hidden()

    # -- Per-tick instrumentation (P3) ----------------------------
    # We capture the (task, hustle) chosen by each worker on each
    # tick along with the env state we care about. Done in-loop
    # rather than from daily_summaries because the daily summary
    # doesn't expose per-tick action/state.
    FILLER_TASKS = {"side_project", "loading", "training",
                    "quality_check", "returns_processing", "receiving"}
    buffer_full_normal_ticks = 0
    buffer_full_normal_choices: Counter = Counter()
    buffer_full_normal_filler_picks = 0

    # P4: per-day task distribution (worker-tick counts per task), so
    # we can split A vs F days and see if the policy is doing very
    # different things or just diffusing.
    per_day_task_counts: list[dict[str, int]] = []
    current_day_counts: dict[str, int] = defaultdict(int)
    current_day_idx_seen = -1

    # P5: time-of-day for side_project assignments — morning (mistake
    # under pressure) vs afternoon (defeatism after the day is lost).
    # Buckets: <11h, 11-13h, 13-15h, >=15h
    sp_by_hour_bucket: Counter = Counter()
    # Also: side_project assignments per day relative to grade
    sp_by_day_grade: dict[str, list[int]] = defaultdict(list)
    current_day_sp_count = 0

    MAX_TICKS = 200_000
    ticks = 0
    while not env.is_year_done and ticks < MAX_TICKS:
        sd = builder.build(env.day_env)
        mask = get_action_mask(env.day_env)
        hmask = get_hustle_mask(env.day_env)
        ta, ha, *_ = agent.select_action_from_dict(sd, mask, hustle_mask=hmask)

        denv = env.day_env

        # P3: buffer-full + non-OT tick check
        cap = cfg.rules.pick_buffer_capacity
        if cap is not None and denv.orders_picked_not_audited >= cap \
                and not getattr(denv, "is_ot", False):
            buffer_full_normal_ticks += 1
            for i, _w in enumerate(denv.episode.workers):
                if i >= len(ta):
                    break
                ti = int(ta[i])
                tname = (cfg.task_ids[ti] if ti < len(cfg.task_ids)
                         else "unknown")
                buffer_full_normal_choices[tname] += 1
                if tname in FILLER_TASKS:
                    buffer_full_normal_filler_picks += 1

        # P4 + P5: per-day task counts + side_project hour buckets.
        # New day = daily_summaries grew since last tick (we capture
        # the current day's data before stepping into the next).
        if len(env.daily_summaries) != current_day_idx_seen:
            # We're now on day = len(daily_summaries) (0-indexed for prior closed)
            # Close out previous day if we have one
            if current_day_idx_seen >= 0:
                per_day_task_counts.append(dict(current_day_counts))
            current_day_counts = defaultdict(int)
            current_day_sp_count = 0
            current_day_idx_seen = len(env.daily_summaries)

        for i in range(len(ta)):
            ti = int(ta[i])
            tname = (cfg.task_ids[ti] if ti < len(cfg.task_ids)
                     else "unknown")
            current_day_counts[tname] += 1
            if tname == "side_project":
                current_day_sp_count += 1
                h = denv.current_hour
                if h < 11.0:
                    sp_by_hour_bucket["08-11"] += 1
                elif h < 13.0:
                    sp_by_hour_bucket["11-13"] += 1
                elif h < 15.0:
                    sp_by_hour_bucket["13-15"] += 1
                else:
                    sp_by_hour_bucket["15+"] += 1

        actions = [(i, int(ta[i]), bool(ha[i]))
                   for i in range(len(ta))]
        _, _, day_done, info = env.step(actions)
        # If a day just closed, attribute the side_project count to its grade.
        if (info or {}).get("new_day") or day_done:
            if env.daily_summaries:
                g = env.daily_summaries[-1].get("footer", {}).get("grade", "?")
                sp_by_day_grade[g].append(current_day_sp_count)
                current_day_sp_count = 0
        ticks += 1
        if len(env.daily_summaries) >= args.days:
            # Flush final day's counts
            if current_day_counts:
                per_day_task_counts.append(dict(current_day_counts))
            break

    # -- Per-day rollup (P1 + P2) ----------------------------------
    # daily_summaries[i].footer has reward_breakdown + grade
    by_grade_filler: dict[str, list[float]] = defaultdict(list)
    by_grade_filler_bonus: dict[str, list[float]] = defaultdict(list)
    by_grade_crunch: dict[str, list[float]] = defaultdict(list)
    cap_binds = 0       # days that hit exactly the cap (-5000)
    cap_breaches = 0    # days where crunch penalty went past -5000
    max_breach = 0.0
    cap_value = -5000.0

    grade_counts: Counter = Counter()
    # P6: per-grade reward-component sums (find the dominant signal)
    by_grade_rewards: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for s in env.daily_summaries[:args.days]:
        ft = s.get("footer", {})
        rb = ft.get("reward_breakdown", {}) or {}
        g = ft.get("grade", "?")
        grade_counts[g] += 1
        by_grade_filler[g].append(rb.get("per_filler_unit", 0.0))
        by_grade_filler_bonus[g].append(rb.get("filler_completion_bonus", 0.0))
        crunch = rb.get("side_project_during_crunch", 0.0)
        by_grade_crunch[g].append(crunch)
        # P6: every reward component
        for k, v in rb.items():
            by_grade_rewards[g][k].append(float(v))
        # Cap analysis
        if crunch <= cap_value:
            if abs(crunch - cap_value) < 0.5:
                cap_binds += 1
            else:
                cap_breaches += 1
                max_breach = min(max_breach, crunch)

    # -- REPORT ---------------------------------------------------
    def _stats(xs):
        if not xs: return "n=0"
        return f"n={len(xs)} mean={np.mean(xs):+.2f} median={np.median(xs):+.2f} max={max(xs):+.2f} min={min(xs):+.2f}"

    print("=" * 70)
    print(f"  Foundation probe: {args.facility}")
    print(f"  ckpt: {ckpt_path.name}    days simulated: {len(env.daily_summaries)}")
    print("=" * 70)
    print()
    print(f"Grade distribution:  ",
          " ".join(f"{g}:{grade_counts.get(g,0)}" for g in "ABCDF"))
    print()
    print("-- P1: per_filler_unit + filler_completion_bonus per day, by grade")
    print("   (R1 zeros these - if F-days accumulate large positive filler reward,")
    print("    Restart 1 is justified.)")
    for g in "ABCDF":
        if grade_counts.get(g, 0):
            print(f"    {g}: per_filler_unit   {_stats(by_grade_filler[g])}")
            print(f"       filler_completion {_stats(by_grade_filler_bonus[g])}")
    print()
    print("-- P2: side_project_during_crunch per day, by grade")
    print("   (R2 tightens cap and fixes the leak. If cap rarely binds OR")
    print("    breaches are small, Restart 2 is solving a non-problem.)")
    for g in "ABCDF":
        if grade_counts.get(g, 0):
            print(f"    {g}: {_stats(by_grade_crunch[g])}")
    print(f"   cap value: {cap_value}")
    print(f"   days that hit cap exactly:  {cap_binds}")
    print(f"   days that breached cap:     {cap_breaches}")
    print(f"   worst breach (most-neg):    {max_breach:+.2f}")
    print()
    # -- P4: per-day task distribution, split by grade ----------------
    print("-- P4: task share of worker-ticks per day, A vs F")
    print("   (If F-days show pick/pack still dominant, the policy is")
    print("    trying but failing. If side_project/idle is large, the")
    print("    policy is giving up.)")
    # Match per_day_task_counts back to grades
    grade_for_day = [s.get("footer", {}).get("grade", "?")
                     for s in env.daily_summaries[:args.days]]
    a_totals: Counter = Counter()
    f_totals: Counter = Counter()
    a_days = f_days = 0
    for g, counts in zip(grade_for_day, per_day_task_counts):
        if g == "A":
            for k, v in counts.items(): a_totals[k] += v
            a_days += 1
        elif g == "F":
            for k, v in counts.items(): f_totals[k] += v
            f_days += 1
    def _share_table(totals: Counter, n_days: int, label: str):
        total = sum(totals.values()) or 1
        print(f"    {label} ({n_days} days, {total} worker-ticks):")
        for task, n in totals.most_common():
            pct = (n / total) * 100
            mark = " <-- filler" if task in FILLER_TASKS else ""
            print(f"      {task:22s} {n:8d}  {pct:5.1f}%{mark}")
    if a_days: _share_table(a_totals, a_days, "A-day task share")
    if f_days: _share_table(f_totals, f_days, "F-day task share")
    print()

    # -- P5: time-of-day for side_project assignments -----------------
    print("-- P5: side_project chosen at what time-of-day")
    print("   (Morning <11h = mistake under peak pressure. Afternoon")
    print("    15+ = defeatism / value-function gave up.)")
    sp_total = sum(sp_by_hour_bucket.values()) or 1
    for bucket in ("08-11", "11-13", "13-15", "15+"):
        n = sp_by_hour_bucket.get(bucket, 0)
        pct = (n / sp_total) * 100
        print(f"    hour {bucket:5s}: {n:6d}  {pct:5.1f}%")
    print()
    print("   side_project assignments per day by grade:")
    for g in "ABCDF":
        if sp_by_day_grade.get(g):
            xs = sp_by_day_grade[g]
            print(f"      {g}: n_days={len(xs)} mean={np.mean(xs):.1f} "
                  f"median={np.median(xs):.1f} max={max(xs)}")
    print()

    # -- P6: reward-component dominance on F-days ---------------------
    print("-- P6: top reward components on F-days (sorted by abs(mean))")
    print("   (What's actually driving the gradient on failed days?)")
    if by_grade_rewards.get("F"):
        means = [(k, float(np.mean(vs)))
                 for k, vs in by_grade_rewards["F"].items() if vs]
        means.sort(key=lambda kv: -abs(kv[1]))
        for k, m in means[:12]:
            print(f"      {k:32s} mean={m:+10.2f}")
    print()

    print("-- P3: buffer-full incidents during non-OT hours")
    print("   (R3 masks non-pack when buffer is full. If 'filler' shows up")
    print("    frequently here, structural enforcement is needed.)")
    print(f"   non-OT buffer-full ticks (any worker): {buffer_full_normal_ticks}")
    total_choices = sum(buffer_full_normal_choices.values())
    if total_choices:
        print(f"   worker-task choices during those ticks:")
        for task, n in buffer_full_normal_choices.most_common():
            pct = (n / total_choices) * 100
            mark = " !!" if task in FILLER_TASKS else ""
            print(f"      {task:22s} {n:6d}  {pct:5.1f}%{mark}")
        filler_pct = (buffer_full_normal_filler_picks / total_choices) * 100
        print(f"   filler-task choices when buffer full: "
              f"{buffer_full_normal_filler_picks} / {total_choices} = {filler_pct:.1f}%")
    else:
        print(f"   no buffer-full incidents during non-OT hours in this sample.")
    print()

    # Persisted machine-readable summary for after/before comparison
    result = {
        "facility": args.facility,
        "ckpt": str(ckpt_path),
        "days": len(env.daily_summaries),
        "grade_counts": dict(grade_counts),
        "p1_filler_unit_by_grade_mean": {
            g: float(np.mean(by_grade_filler[g])) if by_grade_filler[g] else 0.0
            for g in "ABCDF"},
        "p1_filler_completion_by_grade_mean": {
            g: float(np.mean(by_grade_filler_bonus[g])) if by_grade_filler_bonus[g] else 0.0
            for g in "ABCDF"},
        "p2_crunch_by_grade_mean": {
            g: float(np.mean(by_grade_crunch[g])) if by_grade_crunch[g] else 0.0
            for g in "ABCDF"},
        "p2_cap_binds_days": cap_binds,
        "p2_cap_breaches_days": cap_breaches,
        "p2_worst_breach": float(max_breach),
        "p3_buffer_full_normal_ticks": buffer_full_normal_ticks,
        "p3_buffer_full_choices": dict(buffer_full_normal_choices),
        "p3_filler_picks_on_buffer_full": buffer_full_normal_filler_picks,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"machine-readable results written to {args.out}")


if __name__ == "__main__":
    main()

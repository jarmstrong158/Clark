"""Clark — proper checkpoint evaluator.

Runs full-year simulations against the in-env production grader (NOT a
probe rule). Supports:

  - Single facility config (e.g. clark/data/configs/jack_baseline.yaml)
  - Synthetic stage 1 / 2 / 3 sampling (when --stage is set)
  - Multiple seeds for variance estimation
  - Paired head-to-head between two checkpoints

This replaces the ad-hoc probe scripts in tools/archive/. Use this for any
"how good is checkpoint X" or "is checkpoint X better than Y" question —
it uses the same simulation loop and grader the training loop uses, so
the numbers are directly comparable to training-time `year-day grades`
lines.

The signature lesson the archived probes taught: a probe that exits at
`env.is_done` (day 1) and uses a coarse home-rolled grade rule will
disagree with the production grader on exactly the bands that matter for
promotion decisions. The right comparison is full-year + production
grader. That's what this script does, full stop.

Examples
--------
  # Evaluate v2.10 foundation on Jack (full year, 1 seed)
  python tools/clark_eval.py \\
      --ckpt clark/data/checkpoints/clark_foundation_v210_ft.pt \\
      --config clark/data/configs/jack_baseline.yaml \\
      --label "v2.10 on Jack"

  # Evaluate v2.10 + 50ep ft on Jack across 5 seeds
  python tools/clark_eval.py \\
      --ckpt clark/data/checkpoints/clark_jack_v210_ft.pt \\
      --config clark/data/configs/jack_baseline.yaml \\
      --seeds 1,2,3,4,5 \\
      --label "v2.10 + 50ep ft on Jack"

  # Evaluate on 10 synthetic stage-3 configs (representative of curriculum)
  python tools/clark_eval.py \\
      --ckpt clark/data/checkpoints/clark_foundation_v210_ft.pt \\
      --stage 3 --n-configs 10 --seed 42 \\
      --label "v2.10 stage-3 sweep"

  # Head-to-head: two checkpoints, same configs/seeds (paired)
  python tools/clark_eval.py \\
      --ckpt clark/data/checkpoints/clark_foundation_v28_ft.pt \\
      --vs clark/data/checkpoints/clark_foundation_v210_ft.pt \\
      --config clark/data/configs/jack_baseline.yaml \\
      --seeds 1,2,3 --label "v2.8 vs v2.10 on Jack"
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

import numpy as np
import torch

from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.agent.ppo import ClarkAgent
from clark.agent.state import StateBuilder
from clark.config.schema import FacilityConfig
from clark.env.year_env import YearEnv
from clark.training.synthetic_gen import generate_random_facility


# ----- core simulation -----

def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_one_year(agent: ClarkAgent, cfg: FacilityConfig) -> dict:
    """One full year via YearEnv + the same select_action_from_dict
    path the production /simulate endpoint uses. Returns the env's
    year summary (with daily_summaries that carry per-day production
    grades from facility_env._compute_grade)."""
    env = YearEnv(cfg)
    builder = StateBuilder(cfg)
    env.reset()
    agent.reset_hidden()
    max_ticks = 200_000  # safety cap; full year is well under this
    ticks = 0
    while not env.is_year_done and ticks < max_ticks:
        sd = builder.build(env.day_env)
        mask = get_action_mask(env.day_env)
        hmask = get_hustle_mask(env.day_env)
        ta, ha, *_ = agent.select_action_from_dict(
            sd, mask, hustle_mask=hmask)
        actions = [(i, int(ta[i]), bool(ha[i])) for i in range(len(ta))]
        env.step(actions)
        ticks += 1
    return env._get_year_summary(), ticks


def _summarize(summary: dict) -> dict:
    """Extract the headline metrics from a year summary."""
    daily = summary.get("daily_summaries", [])
    n = len(daily)
    grades = Counter()
    ot_days = 0
    ship_full = 0
    total_o = 0
    shipped_o = 0
    for d in daily:
        f = d.get("footer", {}); h = d.get("header", {})
        grades[f.get("grade", "?")] += 1
        if f.get("ot_hours", 0) > 0: ot_days += 1
        t = h.get("total_orders", 0); s = f.get("orders_shipped", 0)
        total_o += t; shipped_o += s
        if t > 0 and s >= t: ship_full += 1
    return {
        "n_days": n,
        "ship_win_pct": 100 * ship_full / max(1, n),
        "completion_pct": 100 * shipped_o / max(1, total_o),
        "ot_day_pct": 100 * ot_days / max(1, n),
        "grade_counts": {g: grades[g] for g in "ABCDF"},
        "grade_pct": {g: 100 * grades[g] / max(1, n) for g in "ABCDF"},
        "ab_pct": 100 * (grades["A"] + grades["B"]) / max(1, n),
    }


# ----- run plans -----

def _configs_from_args(args) -> list[tuple[str, FacilityConfig]]:
    """Build (label, cfg) pairs from CLI args."""
    out = []
    if args.config:
        cfg = FacilityConfig.from_yaml(args.config)
        name = Path(args.config).stem
        out.append((name, cfg))
    if args.stage:
        # Synthetic sweep: n configs from stage
        n = args.n_configs
        # Seed deterministic config sampling separately from seed list
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        for i in range(n):
            cfg = generate_random_facility(stage=args.stage)
            out.append((f"stage{args.stage}_cfg{i}", cfg))
    if not out:
        raise SystemExit("Must pass --config and/or --stage")
    return out


def _eval_ckpt(ckpt_path: Path, configs: list[tuple[str, FacilityConfig]],
               seeds: list[int]) -> dict:
    """Evaluate one checkpoint across all (config, seed) pairs.
    Returns aggregated stats."""
    agent = ClarkAgent.load(str(ckpt_path), device="cpu")
    runs = []  # list of (config_label, seed, summary)
    for cfg_label, cfg in configs:
        for seed in seeds:
            _seed_all(seed)
            summary, ticks = _run_one_year(agent, cfg)
            s = _summarize(summary)
            s["config_label"] = cfg_label
            s["seed"] = seed
            s["ticks"] = ticks
            runs.append(s)
    # Aggregate
    agg = {
        "ckpt": str(ckpt_path),
        "n_runs": len(runs),
        "runs": runs,
        "mean_ship_win_pct": mean(r["ship_win_pct"] for r in runs),
        "mean_completion_pct": mean(r["completion_pct"] for r in runs),
        "mean_ab_pct": mean(r["ab_pct"] for r in runs),
        "mean_grade_pct": {
            g: mean(r["grade_pct"][g] for r in runs) for g in "ABCDF"
        },
    }
    if len(runs) > 1:
        agg["std_ship_win_pct"] = stdev(r["ship_win_pct"] for r in runs)
        agg["std_ab_pct"] = stdev(r["ab_pct"] for r in runs)
    return agg


def _print_block(label: str, agg: dict) -> None:
    print(f"\n=== {label} ({agg['n_runs']} runs) ===")
    if "std_ship_win_pct" in agg:
        print(f"  ship_win:     {agg['mean_ship_win_pct']:5.1f}% "
              f"(±{agg['std_ship_win_pct']:.1f})")
        print(f"  A+B share:    {agg['mean_ab_pct']:5.1f}% "
              f"(±{agg['std_ab_pct']:.1f})")
    else:
        print(f"  ship_win:     {agg['mean_ship_win_pct']:5.1f}%")
        print(f"  A+B share:    {agg['mean_ab_pct']:5.1f}%")
    print(f"  completion:   {agg['mean_completion_pct']:5.1f}%")
    g = agg["mean_grade_pct"]
    print(f"  grade dist:   "
          f"A={g['A']:.1f}%  B={g['B']:.1f}%  C={g['C']:.1f}%  "
          f"D={g['D']:.1f}%  F={g['F']:.1f}%")


def _print_head_to_head(a_label: str, a: dict, b_label: str, b: dict) -> None:
    print(f"\n=== {a_label} vs {b_label} (paired) ===")
    print(f"{'metric':<14} {a_label:>20} {b_label:>20} {'delta':>10}")
    print("-" * 68)
    pairs = [
        ("ship_win %", a["mean_ship_win_pct"], b["mean_ship_win_pct"]),
        ("A+B %", a["mean_ab_pct"], b["mean_ab_pct"]),
        ("completion %", a["mean_completion_pct"], b["mean_completion_pct"]),
        ("A %", a["mean_grade_pct"]["A"], b["mean_grade_pct"]["A"]),
        ("F %", a["mean_grade_pct"]["F"], b["mean_grade_pct"]["F"]),
    ]
    for k, va, vb in pairs:
        delta = vb - va
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "·")
        print(f"  {k:<12} {va:>19.1f}% {vb:>19.1f}% {delta:>+9.1f} {arrow}")


# ----- entry -----

def main():
    p = argparse.ArgumentParser(
        description="Full-year evaluator using the in-env production grader."
    )
    p.add_argument("--ckpt", required=True,
                   help="Primary checkpoint to evaluate.")
    p.add_argument("--vs", default=None,
                   help="Second checkpoint for paired head-to-head.")
    p.add_argument("--label", default=None,
                   help="Display label for the primary checkpoint.")
    p.add_argument("--vs-label", default=None,
                   help="Display label for the second checkpoint.")
    p.add_argument("--config", default=None,
                   help="Facility config YAML to run.")
    p.add_argument("--stage", type=int, default=None, choices=[1, 2, 3],
                   help="Sample synthetic configs from this curriculum stage.")
    p.add_argument("--n-configs", type=int, default=5,
                   help="Number of synthetic configs to sample (with --stage).")
    p.add_argument("--seeds", default="42",
                   help="Comma-separated seed list, e.g. '1,2,3'.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for synthetic config sampling (separate from --seeds).")
    p.add_argument("--out", default=None,
                   help="Optional JSON path to write the full result blob.")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    configs = _configs_from_args(args)

    a_label = args.label or Path(args.ckpt).stem
    print(f"\n### Evaluating {a_label}")
    print(f"### configs: {len(configs)}  seeds: {seeds}  "
          f"total runs: {len(configs) * len(seeds)}")
    a_agg = _eval_ckpt(Path(args.ckpt), configs, seeds)
    _print_block(a_label, a_agg)

    if args.vs:
        b_label = args.vs_label or Path(args.vs).stem
        print(f"\n### Evaluating {b_label} (paired)")
        b_agg = _eval_ckpt(Path(args.vs), configs, seeds)
        _print_block(b_label, b_agg)
        _print_head_to_head(a_label, a_agg, b_label, b_agg)
        if args.out:
            Path(args.out).write_text(json.dumps(
                {"a": a_agg, "b": b_agg,
                 "a_label": a_label, "b_label": b_label},
                indent=2))
            print(f"\nWrote: {args.out}")
    else:
        if args.out:
            Path(args.out).write_text(json.dumps(
                {"a": a_agg, "a_label": a_label}, indent=2))
            print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()

"""Sample the synthetic curriculum and measure how often jack-baseline-shape
configs appear.

A "jack-baseline-shape" config:
  - N in [5, 10]  (small-mid roster)
  - filler tasks enabled (side_project / loading / training / qc / returns / receiving)
  - peak-month vol-to-capacity ratio in [0.35, 0.70]  (enough load that
    crunch should fire, not so much that the day is unwinnable)

If this shape appears at <5% per stage, the foundation literally didn't
get trained on jack-style scenarios enough to learn the right behavior
on them — that's the curriculum gap, and the fix is to oversample.

If >10%, jack-shape coverage is fine and the issue is elsewhere.
"""
from __future__ import annotations
import argparse
import random
from collections import Counter
from pathlib import Path
import sys

import numpy as np


FILLER_TASKS = {"side_project", "loading", "training",
                "quality_check", "returns_processing", "receiving"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-stage", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from clark.training.synthetic_gen import (
        generate_random_facility, CURRICULUM_STAGES,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Sampling {args.samples_per_stage} configs per stage...\n")

    overall = {}
    for stage_num in (1, 2, 3):
        stats = {
            "N_dist": Counter(),
            "has_filler": 0,
            "n_workers_range": [],
            "n_tasks": Counter(),
            "vol_cap_ratio_peak": [],
            "tasks_seen": Counter(),
            "jack_shape": 0,
            "jack_shape_with_high_pressure": 0,
        }
        for i in range(args.samples_per_stage):
            try:
                cfg = generate_random_facility(stage=stage_num)
            except Exception:
                continue
            N = len(cfg.workers)
            stats["N_dist"][N] += 1
            stats["n_workers_range"].append(N)
            enabled_tasks = set(cfg.tasks.enabled or [])
            stats["n_tasks"][len(enabled_tasks)] += 1
            for t in enabled_tasks:
                stats["tasks_seen"][t] += 1
            has_filler = bool(enabled_tasks & FILLER_TASKS)
            if has_filler:
                stats["has_filler"] += 1
            # Estimate capacity = sum(worker.base_oph × shift_hours)
            capacity = sum(w.base_oph * w.shift_hours for w in cfg.workers)
            # Peak-month volume = max of all seasonal_ranges' high
            ranges = cfg.volume.seasonal_ranges or {}
            if ranges:
                peak_vol = max(hi for (_lo, hi) in ranges.values())
                ratio = peak_vol / max(1.0, capacity)
                stats["vol_cap_ratio_peak"].append(ratio)
            else:
                ratio = 0.0

            # Jack-shape: N in [5, 10], has filler, peak ratio in [0.35, 0.70]
            is_jack_N = 5 <= N <= 10
            is_pressure = 0.35 <= ratio <= 0.70
            if is_jack_N and has_filler:
                stats["jack_shape"] += 1
                if is_pressure:
                    stats["jack_shape_with_high_pressure"] += 1

        print(f"=== STAGE {stage_num} (sampled {sum(stats['N_dist'].values())}) ===")
        cs = CURRICULUM_STAGES[stage_num]
        print(f"  config bounds: N in [{cs.min_workers},{cs.max_workers}], M <= {cs.max_tasks}")
        nws = stats["n_workers_range"]
        print(f"  N distribution: mean={np.mean(nws):.1f} median={int(np.median(nws))} "
              f"p10={int(np.percentile(nws,10))} p90={int(np.percentile(nws,90))}")
        print(f"  N=5-10 facilities: {sum(c for n,c in stats['N_dist'].items() if 5<=n<=10)} "
              f"({sum(c for n,c in stats['N_dist'].items() if 5<=n<=10)/max(1,sum(stats['N_dist'].values()))*100:.1f}%)")
        filler_pct = stats["has_filler"] / max(1, sum(stats["N_dist"].values())) * 100
        print(f"  configs with ANY filler task enabled: {stats['has_filler']} ({filler_pct:.1f}%)")
        if stats["vol_cap_ratio_peak"]:
            vc = stats["vol_cap_ratio_peak"]
            print(f"  peak-month vol/capacity ratio: "
                  f"mean={np.mean(vc):.3f} median={np.median(vc):.3f} "
                  f"p10={np.percentile(vc,10):.3f} p90={np.percentile(vc,90):.3f}")
        print(f"  enabled-task frequencies (top 10):")
        for t, n in stats["tasks_seen"].most_common(10):
            pct = n / max(1, sum(stats["N_dist"].values())) * 100
            mark = " <-- filler" if t in FILLER_TASKS else ""
            print(f"      {t:22s} {n:4d}  {pct:5.1f}%{mark}")
        js = stats["jack_shape"]
        jp = stats["jack_shape_with_high_pressure"]
        n_samples = sum(stats["N_dist"].values())
        print(f"  jack-baseline-shape (N=5-10 + filler enabled): "
              f"{js} ({js/max(1,n_samples)*100:.1f}%)")
        print(f"  jack-shape + peak vol/cap in [0.35, 0.70]:      "
              f"{jp} ({jp/max(1,n_samples)*100:.1f}%)")
        print()
        overall[stage_num] = (js, jp, n_samples)

    # Pretrain stage weighting
    print("=== Pretrain effective coverage ===")
    print("Foundation pre-trains across all 3 stages in sequence.")
    print("Stage-weighted jack-shape coverage assumes equal stage exposure:")
    total_js = sum(s[0] for s in overall.values())
    total_jp = sum(s[1] for s in overall.values())
    total_n = sum(s[2] for s in overall.values())
    print(f"  jack-shape across all stages:        {total_js}/{total_n} = {total_js/max(1,total_n)*100:.1f}%")
    print(f"  jack-shape + high-pressure:          {total_jp}/{total_n} = {total_jp/max(1,total_n)*100:.1f}%")
    print()
    print("Interpretation:")
    print(f"  - jack-shape >5%: curriculum covers jack-style configs adequately")
    print(f"  - jack-shape +high-pressure <5%: model rarely sees jack-style HARD days")
    print(f"  - if <5% for high-pressure: that's the gap. Oversample those scenarios.")


if __name__ == "__main__":
    main()

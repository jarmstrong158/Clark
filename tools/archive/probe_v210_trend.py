"""Is v2.10 actually improving, or oscillating around its v2.8 warm-start?

Splits the v2.10 phase (eps >= 15800) into early / mid / late thirds
and compares grade dist, win, ship, OT, and reward composition. Also
fits a slope on A% and F% over the run.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np

d = json.loads(Path("clark/data/logs/pretrain/training_metrics.json").read_text())
eps = [e for e in d["episodes"] if e["ep"] >= 15800]
wins = [w for w in d["windows"] if w["ep"] >= 15800]
ppo = [p for p in d["ppo_updates"] if p["ep"] >= 15800]

print(f"v2.10 phase: {len(eps)} eps  |  {len(wins)} window pts  |  "
      f"{len(ppo)} PPO updates")
print(f"ep range: {eps[0]['ep']} -> {eps[-1]['ep']}\n")

# Split into thirds
n = len(eps)
i1, i2 = n // 3, 2 * n // 3
parts = {
    "EARLY (first 1/3)": eps[:i1],
    "MID   (mid 1/3)":   eps[i1:i2],
    "LATE  (last 1/3)":  eps[i2:],
}

print(f"{'phase':<22} {'eps':>5}  {'A%':>5} {'B%':>5} {'C%':>5} {'D%':>5} {'F%':>5}  "
      f"{'win':>5} {'ship':>5} {'OT':>5} {'Cmp':>5}")
print("-" * 88)
summaries = {}
for name, slice_ in parts.items():
    gs = Counter(e["last_grade"] for e in slice_)
    nslice = len(slice_)
    win = np.mean([e["win_year"] for e in slice_])
    ship = np.mean([e["ship_win"] for e in slice_])
    ot = np.mean([e["ot_year"] for e in slice_])
    cmp = np.mean([e["cmp_year"] for e in slice_])
    pct = {g: 100 * gs[g] / nslice for g in "ABCDF"}
    summaries[name] = (pct, win, ship, ot, cmp)
    print(f"{name:<22} {nslice:>5}  "
          f"{pct['A']:>5.1f} {pct['B']:>5.1f} {pct['C']:>5.1f} "
          f"{pct['D']:>5.1f} {pct['F']:>5.1f}  "
          f"{win:>5.2f} {ship:>5.2f} {ot:>5.2f} {cmp:>5.2f}")

print("\n=== TREND (LATE - EARLY) ===")
e_pct, e_win, e_ship, e_ot, e_cmp = summaries["EARLY (first 1/3)"]
l_pct, l_win, l_ship, l_ot, l_cmp = summaries["LATE  (last 1/3)"]
for g in "ABCDF":
    diff = l_pct[g] - e_pct[g]
    arrow = "+" if diff > 0 else ""
    sig = " <-- gain" if (g in "AB" and diff > 1) else (" <-- worse" if (g in "DF" and diff > 1) else "")
    print(f"  {g}%:  {e_pct[g]:5.1f} -> {l_pct[g]:5.1f}  ({arrow}{diff:+.1f}pp){sig}")
print(f"  win:  {e_win:.3f} -> {l_win:.3f}  ({l_win-e_win:+.3f})")
print(f"  ship: {e_ship:.3f} -> {l_ship:.3f}  ({l_ship-e_ship:+.3f})")
print(f"  OT:   {e_ot:.3f} -> {l_ot:.3f}  ({l_ot-e_ot:+.3f})")

# Slope fit
print("\n=== LINEAR SLOPE FIT over v2.10 phase ===")
x = np.array([e["ep"] for e in eps])
for label, ys in [
    ("win_year", [e["win_year"] for e in eps]),
    ("ship_win", [e["ship_win"] for e in eps]),
    ("ot_year",  [e["ot_year"]  for e in eps]),
]:
    y = np.array(ys)
    slope, intercept = np.polyfit(x, y, 1)
    per_100 = slope * 100
    direction = "improving" if (
        (label in ("win_year","ship_win") and slope > 0) or
        (label == "ot_year" and slope < 0)
    ) else "regressing" if abs(per_100) > 0.001 else "flat"
    print(f"  {label:>10}: slope = {per_100:+.4f}/100eps  ({direction})")

# Slope on A% and F% from windows
print("\n=== Windowed grade % slope ===")
wx = np.array([w["ep"] for w in wins])
A_pct = np.array([100 * w["grades"].get("A",0) / max(1, sum(w["grades"].values())) for w in wins])
F_pct = np.array([100 * w["grades"].get("F",0) / max(1, sum(w["grades"].values())) for w in wins])
ab_pct = A_pct + np.array([100 * w["grades"].get("B",0) / max(1, sum(w["grades"].values())) for w in wins])
for label, y in [("A%", A_pct), ("F%", F_pct), ("A+B%", ab_pct)]:
    slope, _ = np.polyfit(wx, y, 1)
    print(f"  {label:>5}: {y[0]:.1f} -> {y[-1]:.1f}  |  slope {slope*100:+.3f}pp/100eps")

# PPO health drift
print("\n=== PPO health drift ===")
px = np.array([p["ep"] for p in ppo])
for label, ys in [
    ("clip", [p["clip"] for p in ppo]),
    ("v_loss", [min(p["v_loss"], 5) for p in ppo]),  # clip outliers
    ("entropy", [p["entropy"] for p in ppo]),
]:
    y = np.array(ys)
    # early vs late thirds
    n_ppo = len(y)
    early_mean = y[:n_ppo//3].mean()
    late_mean  = y[2*n_ppo//3:].mean()
    print(f"  {label:>8}: early {early_mean:.3f}  late {late_mean:.3f}  "
          f"(delta {late_mean-early_mean:+.3f})")

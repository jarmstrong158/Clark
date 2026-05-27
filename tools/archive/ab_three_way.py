"""Three-way A/B: v2 baseline vs v2.5 vs v2.6, same scenarios.

Runs each checkpoint through N identical synthetic stage-3 episodes
(same seeds → same configs), uses temperature-0.5 sampling (matches
production serve behavior, not deterministic argmax), and reports
the metrics that matter:
- A-rate, F-rate, mean ship rate
- Heavy-day metrics (ship rate, filler%, clean %, drown %)
- Mean OT hours
- A-rate among perfect-ship days (the v2.6-targeted metric)

Output decides which checkpoint to promote as the new canonical
clark_foundation.pt.
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.agent.state import StateBuilder
from clark.agent.ppo import ClarkAgent
from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


FILLER = {"side_project", "loading", "training", "quality_check",
          "returns_processing", "receiving"}


def run_episode(model, cfg, seed: int, tau: float = 0.5):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = FacilityEnv(cfg)
    env.reset()
    sb = StateBuilder(cfg)
    lstm_hidden = None
    tick = 0
    while not env.is_done and tick < 250:
        state = sb.build(env)
        amask = get_action_mask(env)
        hmask = get_hustle_mask(env)
        with torch.no_grad():
            a_logits, h_logits, _v, lstm_hidden = model.forward(
                state, action_mask=amask, hustle_mask=hmask,
                lstm_hidden=lstm_hidden,
            )
        am = torch.from_numpy(amask).bool()
        a_logits = a_logits.masked_fill(~am, -1e9)
        probs = torch.softmax(a_logits / tau, dim=-1)
        actions = []
        for w in range(a_logits.shape[0]):
            t_idx = int(torch.multinomial(probs[w], 1).item())
            actions.append((w, t_idx, False))
        env.step(actions)
        tick += 1

    agg = {}
    for w in env.episode.workers:
        for t, h in w.task_hours_today.items():
            agg[t] = agg.get(t, 0) + h
    total_h = sum(agg.values()) or 1
    filler_h = sum(h for t, h in agg.items() if t in FILLER)
    shipped = env.orders_completed
    total_o = env.episode.total_orders
    ot_h = env.ot_hours
    ship = shipped / max(1, total_o)
    # Crude grade matching prior probe
    if shipped < total_o:
        grade = "F"
    elif ot_h > 1.5:
        grade = "C" if ot_h < 3 else "D"
    elif ot_h > 0:
        grade = "B"
    else:
        grade = "A"
    n_w = len(env.episode.workers)
    return {
        "n_workers": n_w, "total_orders": total_o,
        "shipped": shipped, "ship_pct": 100 * ship,
        "grade": grade, "ot_h": ot_h,
        "filler_pct": 100 * filler_h / total_h,
        "heavy": (n_w >= 20 or total_o >= 400),
    }


def summarize(results, label):
    n = len(results)
    from collections import Counter
    gc = Counter(r["grade"] for r in results)
    avg_filler = sum(r["filler_pct"] for r in results) / n
    avg_ot = sum(r["ot_h"] for r in results) / n
    perfect = [r for r in results if r["ship_pct"] >= 99.99]
    perf_a = sum(1 for r in perfect if r["grade"] == "A")
    heavy = [r for r in results if r["heavy"]]
    heavy_filler_avg = sum(r["filler_pct"] for r in heavy) / max(1, len(heavy))
    heavy_ship_avg = sum(r["ship_pct"] for r in heavy) / max(1, len(heavy))
    heavy_clean = sum(1 for r in heavy if r["filler_pct"] < 15)
    heavy_drown = sum(1 for r in heavy if r["filler_pct"] > 40)
    print(f"\n=== {label} (n={n}) ===")
    print(f"  grade dist:     ", end="")
    for g in "ABCDF": print(f"{g}={gc[g]}({100*gc[g]/n:.0f}%) ", end="")
    print()
    print(f"  A-rate:                {100*gc['A']/n:.1f}%")
    print(f"  F-rate:                {100*gc['F']/n:.1f}%")
    print(f"  A-rate / perfect-ship: {100*perf_a/max(1,len(perfect)):.1f}% ({perf_a}/{len(perfect)})")
    print(f"  aggregate filler:      {avg_filler:.1f}%")
    print(f"  mean OT hours:         {avg_ot:.2f}h")
    if heavy:
        print(f"  heavy days ({len(heavy)}):")
        print(f"    avg filler:       {heavy_filler_avg:.1f}%")
        print(f"    mean ship:        {heavy_ship_avg:.1f}%")
        print(f"    clean (<15%):     {heavy_clean}/{len(heavy)} ({100*heavy_clean/len(heavy):.0f}%)")
        print(f"    drown (>40%):     {heavy_drown}/{len(heavy)} ({100*heavy_drown/len(heavy):.0f}%)")
    return {
        "A": 100*gc['A']/n, "F": 100*gc['F']/n,
        "A_perfect_ship": 100*perf_a/max(1,len(perfect)),
        "filler": avg_filler, "ot": avg_ot,
        "heavy_ship": heavy_ship_avg, "heavy_clean": 100*heavy_clean/max(1,len(heavy)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seeds_base", type=int, default=98765)
    args = p.parse_args()

    print(f"Loading checkpoints...")
    a_v2 = ClarkAgent.load("clark/data/checkpoints/clark_foundation.pt", device="cpu")
    a_v25 = ClarkAgent.load("clark/data/checkpoints/clark_foundation_v25_ft.pt", device="cpu")
    a_v26 = ClarkAgent.load("clark/data/checkpoints/clark_foundation_v26_ft.pt", device="cpu")

    # Same configs for all three checkpoints
    print(f"Generating {args.n} matched scenarios...")
    cfgs_and_seeds = []
    for i in range(args.n):
        s = args.seeds_base + i
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        cfg = generate_random_facility(stage=3)
        cfgs_and_seeds.append((cfg, s))

    print(f"Running v2 baseline...")
    r_v2 = [run_episode(a_v2.model, cfg, s) for cfg, s in cfgs_and_seeds]
    print(f"Running v2.5...")
    r_v25 = [run_episode(a_v25.model, cfg, s) for cfg, s in cfgs_and_seeds]
    print(f"Running v2.6...")
    r_v26 = [run_episode(a_v26.model, cfg, s) for cfg, s in cfgs_and_seeds]

    s_v2 = summarize(r_v2,  "v2 BASELINE")
    s_v25 = summarize(r_v25, "v2.5 (ep 15300)")
    s_v26 = summarize(r_v26, "v2.6 (ep 15500)")

    print("\n\n=== SIDE BY SIDE ===")
    print(f"{'metric':<26} {'v2':>10} {'v2.5':>10} {'v2.6':>10} {'v2.6 vs v2':>13}")
    for k, label in [
        ("A", "A-rate"),
        ("F", "F-rate"),
        ("A_perfect_ship", "A / perfect-ship"),
        ("filler", "agg filler"),
        ("ot", "mean OT (h)"),
        ("heavy_ship", "heavy ship rate"),
        ("heavy_clean", "heavy clean %"),
    ]:
        v = s_v2[k]; v5 = s_v25[k]; v6 = s_v26[k]
        delta = v6 - v
        unit = "h" if k == "ot" else "%"
        print(f"{label:<26} {v:>9.1f}{unit} {v5:>9.1f}{unit} {v6:>9.1f}{unit} {delta:>+11.1f}pp")


if __name__ == "__main__":
    main()

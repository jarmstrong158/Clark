"""Quick comparison: v2 baseline checkpoint vs v2.5 fine-tune so far.

Runs v2 baseline through N synthetic stage-3 episodes and prints the
same metrics we've been tracking on the live v2.5 fine-tune. Both
checkpoints are clark-v2 arch (v2.5 didn't change env_feat_dim or
ARCH_VERSION — only the action mask differs), so both load via the
standard path.

The trick: we need to load v2 baseline WITHOUT the v2.5 mask applied.
The mask code is in clark/agent/actions.py and is always active. To
simulate "v2 baseline behavior" without the mask, we'd need to either:
  (a) revert the actions.py change temporarily (risky, affects live run),
  (b) run the comparison against a different checkpoint that doesn't
      need the new mask logic to behave like v2 baseline.

Cheaper: just use the v2 BASELINE checkpoint and accept that the mask
is biting on its decisions too. This gives us "v2 baseline weights
with v2.5 mask" which is the actual upper bound on how much the mask
alone helps without the policy adapting via PPO. The difference between
that and "v2.5 fine-tuned weights with v2.5 mask" is what PPO learned
on top.

For a TRUE "pre-training" v2 reference, we'd need to checkout the
actions.py from before v2.4 and run that. Not worth the trouble for a
30-episode quick comparison — the v2.5 fine-tune log already shows
us where v2.5 has converged.
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


def run_episodes(ckpt_path: Path, n_eps: int = 30, seed: int = 12345):
    print(f"\n--- Loading {ckpt_path.name} ---")
    agent = ClarkAgent.load(str(ckpt_path), device="cpu")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    results = []
    for ep_i in range(n_eps):
        cfg = generate_random_facility(stage=3)
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
                a_logits, h_logits, _v, lstm_hidden = agent.model.forward(
                    state, action_mask=amask, hustle_mask=hmask,
                    lstm_hidden=lstm_hidden,
                )
            am = torch.from_numpy(amask).bool()
            a_logits = a_logits.masked_fill(~am, -1e9)
            # Sample with temperature 0.5 (sharper than training-time but
            # not full argmax) to match the policy's actual operating mode
            # in serve. argmax-deterministic at this training stage gives
            # misleadingly bad results because the policy hasn't yet
            # converged to sharp per-worker logits.
            tau = 0.5
            probs = torch.softmax(a_logits / tau, dim=-1)
            actions = []
            for w in range(a_logits.shape[0]):
                t_idx = int(torch.multinomial(probs[w], 1).item())
                actions.append((w, t_idx, False))
            env.step(actions)
            tick += 1

        # Aggregate
        agg = {}
        for w in env.episode.workers:
            for t, h in w.task_hours_today.items():
                agg[t] = agg.get(t, 0) + h
        total_h = sum(agg.values()) or 1
        filler_h = sum(h for t, h in agg.items() if t in FILLER)
        pick_h = agg.get("pick", 0); pack_h = agg.get("pack", 0)
        off_h = agg.get("off", 0); idle_h = agg.get("idle", 0)
        shipped = env.orders_completed
        total_o = env.episode.total_orders
        ot_h = env.ot_hours
        if shipped < total_o:
            grade = "F"
        elif ot_h > 1.5:
            grade = "C" if ot_h < 3 else "D"
        else:
            grade = "A"
        n_w = len(env.episode.workers)
        results.append({
            "n_workers": n_w, "total_orders": total_o,
            "shipped": shipped, "ship_pct": 100 * shipped / max(1, total_o),
            "grade": grade, "ot_h": ot_h,
            "filler_pct": 100 * filler_h / total_h,
            "pp_pct": 100 * (pick_h + pack_h) / total_h,
            "off_pct": 100 * off_h / total_h,
            "idle_pct": 100 * idle_h / total_h,
            "heavy": (n_w >= 20 or total_o >= 400),
        })
    return results


def summarize(results, label):
    n = len(results)
    grades = [r["grade"] for r in results]
    from collections import Counter
    gc = Counter(grades)
    avg_filler = sum(r["filler_pct"] for r in results) / n
    avg_pp = sum(r["pp_pct"] for r in results) / n
    heavy = [r for r in results if r["heavy"]]
    heavy_avg_filler = sum(r["filler_pct"] for r in heavy) / max(1, len(heavy))
    heavy_clean = sum(1 for r in heavy if r["filler_pct"] < 15)
    heavy_drown = sum(1 for r in heavy if r["filler_pct"] > 40)
    heavy_ship = sum(r["ship_pct"] for r in heavy) / max(1, len(heavy))
    print(f"\n=== {label} ({n} eps) ===")
    print(f"  aggregate filler:    {avg_filler:5.1f}%")
    print(f"  pick + pack:         {avg_pp:5.1f}%")
    print(f"  grade dist:          ", end="")
    for g in "ABCDF":
        print(f"{g}={gc[g]}({100*gc[g]/n:.0f}%) ", end="")
    print()
    if heavy:
        print(f"  heavy days ({len(heavy)}):")
        print(f"    avg filler:        {heavy_avg_filler:5.1f}%")
        print(f"    clean (<15%):      {heavy_clean}/{len(heavy)} ({100*heavy_clean/len(heavy):.0f}%)")
        print(f"    drown (>40%):      {heavy_drown}/{len(heavy)} ({100*heavy_drown/len(heavy):.0f}%)")
        print(f"    mean ship rate:    {heavy_ship:.1f}%")
    return {
        "filler": avg_filler, "pp": avg_pp,
        "A": 100*gc["A"]/n, "F": 100*gc["F"]/n,
        "heavy_filler": heavy_avg_filler,
        "heavy_clean": 100*heavy_clean/max(1,len(heavy)),
        "heavy_ship": heavy_ship,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="clark/data/checkpoints/clark_foundation.pt")
    p.add_argument("--current",  default="clark/data/checkpoints/clark_foundation_v25_ft.pt")
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()

    r_base = run_episodes(Path(args.baseline), args.n)
    r_curr = run_episodes(Path(args.current),  args.n)
    s_base = summarize(r_base, "v2 BASELINE (clark_foundation.pt)")
    s_curr = summarize(r_curr, "v2.5 IN-PROGRESS (clark_foundation_v25_ft.pt)")

    print("\n\n=== SIDE BY SIDE ===")
    print(f"{'metric':<26} {'baseline':>10}  {'in-progress':>11}  {'delta':>8}")
    for k, label in [
        ("filler", "aggregate filler"),
        ("pp", "pick + pack"),
        ("A", "A-grade rate"),
        ("F", "F-grade rate"),
        ("heavy_filler", "heavy avg filler"),
        ("heavy_clean", "heavy clean %"),
        ("heavy_ship", "heavy ship rate"),
    ]:
        bv = s_base[k]; cv = s_curr[k]
        delta = cv - bv
        print(f"{label:<26} {bv:>9.1f}%  {cv:>10.1f}%  {delta:>+7.1f}pp")


if __name__ == "__main__":
    main()

"""Multi-temperature serve probe — head-to-head between two checkpoints.

For each checkpoint, runs N stage-3 episodes at each temperature in the
sweep and reports the serve-time metrics that matter for promotion:
  - ship_win (the primary KPI)
  - grade distribution (A/B/C/D/F on episode-final day)
  - OT frequency
  - filler% of worker-hours
  - mgmt% of worker-hours
  - per-worker task-distinct count (churn signal)

Save the JSON results so the v2.10 run can be compared to the same
v2.8 baseline without rerunning v2.8.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.agent.ppo import ClarkAgent
from clark.agent.state import StateBuilder
from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


FILLER = {"side_project", "loading", "training", "quality_check",
          "returns_processing", "receiving"}


def run_one_temp(ckpt_path: Path, tau: float, n_eps: int, seed: int):
    agent = ClarkAgent.load(str(ckpt_path), device="cpu")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    eps_data = []
    for ep_i in range(n_eps):
        cfg = generate_random_facility(stage=3)
        env = FacilityEnv(cfg)
        env.reset()
        sb = StateBuilder(cfg)
        n_w = len(env.episode.workers)
        history = [[] for _ in range(n_w)]
        lstm_hidden = None
        tick = 0
        while not env.is_done and tick < 250:
            state = sb.build(env)
            amask = get_action_mask(env)
            hmask = get_hustle_mask(env)
            with torch.no_grad():
                a_logits, _hl, _v, lstm_hidden = agent.model.forward(
                    state, action_mask=amask, hustle_mask=hmask,
                    lstm_hidden=lstm_hidden,
                )
            am = torch.from_numpy(amask).bool()
            a_logits = a_logits.masked_fill(~am, -1e9)
            if tau <= 1e-4:
                act_ids = a_logits.argmax(dim=-1).tolist()
            else:
                probs = torch.softmax(a_logits / tau, dim=-1)
                act_ids = [int(torch.multinomial(probs[w], 1).item())
                           for w in range(a_logits.shape[0])]
            actions = [(w, t_idx, False) for w, t_idx in enumerate(act_ids)]
            for w, t_idx in enumerate(act_ids):
                history[w].append(t_idx)
            env.step(actions)
            tick += 1

        # Episode-level aggregation
        agg = {}
        for w in env.episode.workers:
            for t, h in w.task_hours_today.items():
                agg[t] = agg.get(t, 0) + h
        total_h = sum(agg.values()) or 1
        filler_h = sum(h for t, h in agg.items() if t in FILLER)
        mgmt_h = agg.get("management", 0)
        pick_h = agg.get("pick", 0); pack_h = agg.get("pack", 0)

        shipped = env.orders_completed
        total_o = env.episode.total_orders
        ot_h = env.ot_hours

        if shipped < total_o:
            grade = "F"
        elif ot_h > 1.5:
            grade = "C" if ot_h < 3 else "D"
        else:
            grade = "A"

        # task churn
        distinct = [len(set(h)) for h in history if h]
        mean_distinct = mean(distinct) if distinct else 0.0

        eps_data.append({
            "n_workers": n_w, "total_orders": total_o, "shipped": shipped,
            "ship_pct": 100 * shipped / max(1, total_o),
            "grade": grade, "ot_h": ot_h,
            "filler_pct": 100 * filler_h / total_h,
            "mgmt_pct": 100 * mgmt_h / total_h,
            "pp_pct": 100 * (pick_h + pack_h) / total_h,
            "mean_distinct_tasks": mean_distinct,
        })
    return eps_data


def summarize(label, eps_data):
    n = len(eps_data)
    grades = Counter(e["grade"] for e in eps_data)
    ship = mean(e["ship_pct"] for e in eps_data)
    ot = mean(e["ot_h"] for e in eps_data)
    filler = mean(e["filler_pct"] for e in eps_data)
    mgmt = mean(e["mgmt_pct"] for e in eps_data)
    pp = mean(e["pp_pct"] for e in eps_data)
    churn = mean(e["mean_distinct_tasks"] for e in eps_data)
    ship_100 = sum(1 for e in eps_data if e["ship_pct"] >= 99.999)

    print(f"\n=== {label} ({n} eps) ===")
    print(f"  ship_win:           {100*ship_100/n:5.1f}%  ({ship_100}/{n})")
    print(f"  mean ship%:         {ship:5.2f}%")
    print(f"  mean OT hours:      {ot:5.2f}")
    print(f"  grade dist:         " +
          " ".join(f"{g}={grades[g]}({100*grades[g]/n:.0f}%)" for g in "ABCDF"))
    print(f"  pick+pack%:         {pp:5.1f}%")
    print(f"  filler%:            {filler:5.1f}%")
    print(f"  management%:        {mgmt:5.2f}%")
    print(f"  mean tasks/worker:  {churn:.2f}")

    return {
        "n": n,
        "ship_win_pct": 100 * ship_100 / n,
        "mean_ship_pct": ship,
        "mean_ot_h": ot,
        "grade_counts": dict(grades),
        "grade_pct": {g: 100 * grades[g] / n for g in "ABCDF"},
        "pp_pct": pp,
        "filler_pct": filler,
        "mgmt_pct": mgmt,
        "mean_distinct_tasks": churn,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--label", required=True, help="e.g. v2.8 or v2.10")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--temps", default="0.0001,0.1,0.5")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    print(f"\n### {args.label}: {ckpt.name} ###")
    print(f"### {args.n} stage-3 eps/temp, temps={args.temps}, seed={args.seed} ###")

    all_results = {"label": args.label, "ckpt": str(ckpt), "n": args.n,
                   "seed": args.seed, "by_temp": {}}
    for tau_s in args.temps.split(","):
        tau = float(tau_s.strip())
        eps_data = run_one_temp(ckpt, tau, args.n, args.seed)
        summary = summarize(f"{args.label} @ tau={tau}", eps_data)
        all_results["by_temp"][str(tau)] = summary

    out_path = args.out or f"tools/_probe_{args.label.replace('.','')}.json"
    Path(out_path).write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

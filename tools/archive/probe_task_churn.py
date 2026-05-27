"""Per-worker task-churn probe (temperature sweep).

Diagnostic: training-time worker_time logs show ~9 distinct tasks per
worker per day at typical training-time sampling temperature (~1.0).
Question: is this a learned erratic policy, or a stochastic-sampling
artifact that disappears under serve-time low-temperature inference?

This probe runs N stage-3 episodes at each of several temperatures and
reports, per episode/per worker, how many DISTINCT tasks each worker
performed during the day, and the entropy of their per-tick task choice.

Interpretation:
  - If at tau<=0.2 workers commit to 1-2 tasks/day -> churn is a
    training-time artifact only; deploy with low temperature.
  - If at tau<=0.2 workers still hop across 5+ tasks/day -> policy
    has genuinely learned to churn; need a task-switching penalty or
    a sticky mask.
"""
from __future__ import annotations

import argparse
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


def run_at_temp(ckpt_path: Path, tau: float, n_eps: int, seed: int):
    agent = ClarkAgent.load(str(ckpt_path), device="cpu")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    per_ep_worker_taskcounts = []   # list[ list[int] ] -> distinct tasks per worker per ep
    per_ep_worker_entropy = []      # list[ list[float] ] -> shannon entropy of task distribution

    for ep_i in range(n_eps):
        cfg = generate_random_facility(stage=3)
        env = FacilityEnv(cfg)
        env.reset()
        sb = StateBuilder(cfg)
        n_w = len(env.episode.workers)
        # per-worker per-tick action history
        history = [[] for _ in range(n_w)]

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
            if tau <= 1e-4:
                act_ids = a_logits.argmax(dim=-1).tolist()
            else:
                probs = torch.softmax(a_logits / tau, dim=-1)
                act_ids = [int(torch.multinomial(probs[w], 1).item())
                           for w in range(a_logits.shape[0])]
            actions = []
            for w, t_idx in enumerate(act_ids):
                actions.append((w, t_idx, False))
                history[w].append(t_idx)
            env.step(actions)
            tick += 1

        # Per-worker stats for this episode
        distinct_counts = []
        entropies = []
        for w_hist in history:
            if not w_hist:
                distinct_counts.append(0); entropies.append(0.0); continue
            cnt = Counter(w_hist)
            distinct_counts.append(len(cnt))
            total = sum(cnt.values())
            ent = -sum((c/total) * np.log2(c/total) for c in cnt.values() if c > 0)
            entropies.append(float(ent))
        per_ep_worker_taskcounts.append(distinct_counts)
        per_ep_worker_entropy.append(entropies)

    return per_ep_worker_taskcounts, per_ep_worker_entropy


def summarize(counts, ents, label):
    flat_c = [c for ep in counts for c in ep]
    flat_e = [e for ep in ents for e in ep]
    n_workers = len(flat_c)
    if n_workers == 0:
        print(f"{label}: no workers"); return
    # Buckets of distinct task count
    bk = Counter()
    for c in flat_c:
        if c <= 2:
            bk["1-2 (committed)"] += 1
        elif c <= 4:
            bk["3-4 (some switching)"] += 1
        elif c <= 6:
            bk["5-6 (high churn)"] += 1
        else:
            bk["7+ (severe churn)"] += 1
    print(f"\n=== {label} ===")
    print(f"  workers sampled:        {n_workers}")
    print(f"  mean distinct tasks/wk: {mean(flat_c):.2f}")
    print(f"  median:                 {sorted(flat_c)[n_workers//2]}")
    print(f"  mean entropy (bits):    {mean(flat_e):.2f}")
    print(f"  distribution:")
    for k in ["1-2 (committed)", "3-4 (some switching)",
             "5-6 (high churn)", "7+ (severe churn)"]:
        c = bk.get(k, 0)
        print(f"    {k:25s} {c:4d}  ({100*c/n_workers:5.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="clark/data/checkpoints/clark_foundation_v210_ft.pt")
    p.add_argument("--n", type=int, default=8, help="eps per temperature")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--temps", default="0.1,0.5,1.0",
                   help="comma-separated temperatures to sweep")
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    print(f"Probing {ckpt.name} at temperatures: {args.temps}")
    print(f"Stage 3, {args.n} eps/temp, seed={args.seed}")

    for tau_s in args.temps.split(","):
        tau = float(tau_s.strip())
        counts, ents = run_at_temp(ckpt, tau, args.n, args.seed)
        summarize(counts, ents, f"tau={tau}")


if __name__ == "__main__":
    main()

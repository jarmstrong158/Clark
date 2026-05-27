"""Audit the probe_head_to_head.py methodology.

Concerns to check:
1) What does env.is_done MEAN here? Year? Day? After how many ticks does it fire?
2) Are orders_completed / total_orders per-day or per-year?
3) The grading rule we used (A/C/D/F, no B) — does it match reality?
4) With same seed, do v2.8 vs v2.8 produce truly identical results
   (sanity check on determinism)?
5) Across temperatures with same seed, do we face the SAME configs?
"""
from __future__ import annotations
import random
import numpy as np
import torch
from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.agent.ppo import ClarkAgent
from clark.agent.state import StateBuilder
from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility

CKPT = "clark/data/checkpoints/clark_foundation_v210_ft.pt"


def show_one_episode():
    print("\n=== A: what does is_done mean? ===")
    random.seed(12345); np.random.seed(12345); torch.manual_seed(12345)
    cfg = generate_random_facility(stage=3)
    env = FacilityEnv(cfg)
    env.reset()
    print(f"  facility: N={len(env.episode.workers)}")
    print(f"  total_orders at reset: {env.episode.total_orders}")
    print(f"  total_work_days (if attr): {getattr(env, 'total_work_days', '<no attr>')}")

    agent = ClarkAgent.load(CKPT, device="cpu")
    sb = StateBuilder(cfg)
    lstm_hidden = None
    tick = 0
    snapshots = []
    while not env.is_done and tick < 250:
        state = sb.build(env)
        amask = get_action_mask(env); hmask = get_hustle_mask(env)
        with torch.no_grad():
            a_logits, _h, _v, lstm_hidden = agent.model.forward(
                state, action_mask=amask, hustle_mask=hmask, lstm_hidden=lstm_hidden)
        am = torch.from_numpy(amask).bool()
        a_logits = a_logits.masked_fill(~am, -1e9)
        probs = torch.softmax(a_logits / 1.0, dim=-1)
        act_ids = [int(torch.multinomial(probs[w], 1).item()) for w in range(a_logits.shape[0])]
        actions = [(w, t_idx, False) for w, t_idx in enumerate(act_ids)]
        env.step(actions)
        if tick in (0, 10, 40, 78, 100, 156, 200, 249):
            snapshots.append({
                "tick": tick,
                "orders_completed": env.orders_completed,
                "total_orders": env.episode.total_orders,
                "is_done": env.is_done,
                "day_idx": getattr(env, "current_day_idx", "?"),
                "ot_hours": getattr(env, "ot_hours", "?"),
            })
        tick += 1
    snapshots.append({
        "tick": tick, "FINAL": True,
        "orders_completed": env.orders_completed,
        "total_orders": env.episode.total_orders,
        "is_done": env.is_done,
        "day_idx": getattr(env, "current_day_idx", "?"),
        "ot_hours": getattr(env, "ot_hours", "?"),
    })
    for s in snapshots:
        print(f"  {s}")
    print(f"  Total ticks executed: {tick}")
    print(f"  Loop exit reason: {'is_done' if env.is_done else 'tick cap (250)'}")


def determinism_check():
    print("\n=== B: determinism check (same model, same seed, two runs) ===")
    rs = []
    for trial in range(2):
        random.seed(99); np.random.seed(99); torch.manual_seed(99)
        agent = ClarkAgent.load(CKPT, device="cpu")
        cfg = generate_random_facility(stage=3)
        env = FacilityEnv(cfg); env.reset()
        sb = StateBuilder(cfg)
        lstm_hidden = None
        for tick in range(80):
            state = sb.build(env)
            amask = get_action_mask(env); hmask = get_hustle_mask(env)
            with torch.no_grad():
                al, _h, _v, lstm_hidden = agent.model.forward(
                    state, action_mask=amask, hustle_mask=hmask, lstm_hidden=lstm_hidden)
            am = torch.from_numpy(amask).bool()
            al = al.masked_fill(~am, -1e9)
            probs = torch.softmax(al / 1.0, dim=-1)
            act_ids = [int(torch.multinomial(probs[w], 1).item()) for w in range(al.shape[0])]
            env.step([(w, t_idx, False) for w, t_idx in enumerate(act_ids)])
        rs.append({
            "orders_completed": env.orders_completed,
            "total_orders": env.episode.total_orders,
            "ot_hours": env.ot_hours,
        })
    print(f"  trial 1: {rs[0]}")
    print(f"  trial 2: {rs[1]}")
    print(f"  identical: {rs[0] == rs[1]}")


def cross_temp_config_check():
    print("\n=== C: across-temperature config identity ===")
    seen = {}
    for tau in [0.0001, 0.5, 1.0]:
        random.seed(12345); np.random.seed(12345); torch.manual_seed(12345)
        # Just sample 3 configs after the seed reset (mimicking probe loop start)
        cfgs = [generate_random_facility(stage=3) for _ in range(3)]
        seen[tau] = [c.num_workers for c in cfgs]
    for tau, info in seen.items():
        print(f"  tau={tau}: first 3 configs (N,M) = {info}")
    print(f"  all same: {len(set(tuple(v) for v in seen.values())) == 1}")


def grading_rule_audit():
    print("\n=== D: probe's grading rule vs production ===")
    print("  Probe rule (from probe_head_to_head.py):")
    print("    shipped < total: F")
    print("    elif ot > 1.5:   C (if ot<3) or D")
    print("    else:            A")
    print("  --> NO B grade. No partial-completion. No mgmt-deficit demerit.")
    print("  --> Anything that shipped and stayed under 1.5h OT lumps to A.")
    print("  This is INSENSITIVE to many policy differences (small OT trim,")
    print("  mgmt-backlog drift, B-grade-band behaviors).")
    print()
    print("  Production grading: env.facility_env._compute_grade (or similar)")
    print("  uses A/B/C/D/F with multiple demerits (OT, backlog, restock, mgmt,")
    print("  starvation, etc.). Real probe should call the real grader.")


if __name__ == "__main__":
    show_one_episode()
    determinism_check()
    cross_temp_config_check()
    grading_rule_audit()

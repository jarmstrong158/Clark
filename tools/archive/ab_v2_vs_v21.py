"""Head-to-head A/B: v2 baseline vs current v2.1 in-progress checkpoint.

Runs both policies on IDENTICAL synthetic scenarios (same seeds, same
generated episodes) and reports filler% + grade side-by-side. This is
the controlled test for "is the v2.1 policy moving in the direction
we designed for, or is it actively regressing?"

Three measurements per checkpoint, on each of three scenario types:
  - STRESS (overload, dvc > 1.2):     does v2.1 filler% drop vs v2?
  - MID    (rescuable, dvc ~ 0.6):    does v2.1 filler% stay roughly equal?
  - EASY   (loose, dvc < 0.5):        does v2.1 filler% stay equal?

Verdict logic:
  v2.1 wins ALL three  -> "yes, design works, keep training"
  v2.1 wins stress, ties mid/easy -> "yes, design works"
  v2.1 wins stress, loses mid OR easy -> "over-correction confirmed"
  v2.1 doesn't win stress -> "design failed, roll back"

The v2 checkpoint is loaded with explicit env_feat_dim=17 to match its
saved weights. At inference we slice feats[17] off the state before
passing to the v2 model. The v2.1 checkpoint loads normally.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# IMPORTANT: import everything before any seeding so module-level RNG
# state doesn't drift between the two halves of the A/B.
from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.agent.state import StateBuilder
from clark.agent.transformer import ClarkActorCritic
from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


SCENARIOS = {
    "STRESS": {"stage": 3, "vol_mult": 1.8, "n_eps": 6, "label": "dvc ~1.8 (overload)"},
    "MID":    {"stage": 3, "vol_mult": 0.70, "n_eps": 6, "label": "dvc ~0.7 (rescuable)"},
    "EASY":   {"stage": 3, "vol_mult": 0.40, "n_eps": 6, "label": "dvc ~0.4 (loose)"},
}

FILLER = {"side_project", "loading", "training", "quality_check",
          "returns_processing", "receiving"}


def load_v2(ckpt_path: Path) -> ClarkActorCritic:
    """Raw-load a v2 checkpoint into a v2-arch model (env_feat_dim=17)."""
    data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if data.get("arch_version") != "clark-v2":
        raise SystemExit(f"expected clark-v2, got {data.get('arch_version')}")
    hparams = data.get("hparams", {})
    import inspect
    model_keys = set(inspect.signature(ClarkActorCritic.__init__).parameters)
    model_keys.discard("self")
    mh = {k: v for k, v in hparams.items() if k in model_keys}
    mh["env_feat_dim"] = 17                # explicit, since module default is now 18
    model = ClarkActorCritic(**mh)
    model.load_state_dict(data["model_state_dict"], strict=True)
    model.eval()
    return model


def load_v21(ckpt_path: Path) -> ClarkActorCritic:
    """Load a v2.1 checkpoint directly (env_feat_dim=18)."""
    data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if data.get("arch_version") != "clark-v2.1":
        raise SystemExit(f"expected clark-v2.1, got {data.get('arch_version')}")
    hparams = data.get("hparams", {})
    import inspect
    model_keys = set(inspect.signature(ClarkActorCritic.__init__).parameters)
    model_keys.discard("self")
    mh = {k: v for k, v in hparams.items() if k in model_keys}
    mh["env_feat_dim"] = 18
    model = ClarkActorCritic(**mh)
    model.load_state_dict(data["model_state_dict"], strict=True)
    model.eval()
    return model


def policy_step(model, state, action_mask, hustle_mask, lstm_hidden,
                slice_feats_to_17: bool):
    """One inference + action sample. If slice_feats_to_17, lop off the
    18th env_feats column before passing to the v2 model."""
    if slice_feats_to_17:
        state = dict(state)
        state["env_feats"] = state["env_feats"][:17].copy()
    with torch.no_grad():
        a_logits, h_logits, _v, new_hidden = model.forward(
            state, action_mask=action_mask, hustle_mask=hustle_mask,
            lstm_hidden=lstm_hidden,
        )
    # Mask + argmax for deterministic eval (no sampling noise)
    am = torch.from_numpy(action_mask).bool()
    a_logits = a_logits.masked_fill(~am, -1e9)
    a_choice = a_logits.argmax(dim=-1)             # (N,)
    hm = torch.from_numpy(hustle_mask).bool()
    h_logits_masked = h_logits.masked_fill(~hm, -1e9)
    h_choice = h_logits_masked.argmax(dim=-1)      # (N,)
    actions = [(i, int(a_choice[i].item()), bool(h_choice[i].item()))
               for i in range(a_choice.shape[0])]
    return actions, new_hidden


def run_one_episode(model, cfg, volume_override: Optional[int],
                    seed: int, slice_feats: bool) -> dict:
    """Run one full simulated day with the given policy. Return per-
    task hours + final grade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = FacilityEnv(cfg)
    env.reset()
    if volume_override is not None:
        from clark.env.episode_generator import generate_episode
        env.episode = generate_episode(env.facility_config, volume=volume_override)
        env.arrived_so_far = 0
        env.orders_in_queue = 0
        env.orders_completed = 0
        env.orders_picked_not_audited = 0
        env._today_capacity = env._compute_today_capacity()
        total = max(1, env.episode.total_orders)
        if env._restock_enabled:
            env.restock_drain_per_order = 2.0 / total
            env.restock_refill_per_hour = 1.0 / max(0.1, env.episode.restock_hours)
        env.today_complexity_mix = env._sample_order_complexity(env.episode.total_orders)
        env._process_arrivals()

    sb = StateBuilder(cfg)
    lstm_hidden = None
    tick = 0
    while not env.is_done:
        state = sb.build(env)
        amask = get_action_mask(env)
        hmask = get_hustle_mask(env)
        actions, lstm_hidden = policy_step(
            model, state, amask, hmask, lstm_hidden, slice_feats
        )
        env.step(actions)
        tick += 1
        if tick > 200:  # safety
            break

    # Aggregate task hours from per-worker dicts maintained in env
    agg = {}
    for w in env.episode.workers:
        for t, h in w.task_hours_today.items():
            agg[t] = agg.get(t, 0) + h
    total_h = sum(agg.values()) or 1
    filler_h = sum(h for t, h in agg.items() if t in FILLER)
    pick_h = agg.get("pick", 0)
    pack_h = agg.get("pack", 0)
    # Grade is computed at EOD by env._check_eod; pull from reward breakdown
    # via the simulator's own grading logic
    from clark.env.facility_env import INCOMPLETE_CAP
    orders_shipped = env.orders_completed
    orders_total = env.episode.total_orders
    cmp_pct = orders_shipped / max(1, orders_total)
    # Crude grade reproduction (matches env logic in _finalize_episode)
    if orders_shipped < orders_total:
        grade = "F"
    elif env.ot_hours > 1.5:
        grade = "C" if env.ot_hours < 3 else "D"
    else:
        grade = "A"
    return {
        "total_orders": orders_total,
        "shipped": orders_shipped,
        "cmp_pct": cmp_pct,
        "ot_hours": env.ot_hours,
        "grade": grade,
        "total_h": total_h,
        "filler_h": filler_h,
        "filler_pct": 100 * filler_h / total_h,
        "pickpack_pct": 100 * (pick_h + pack_h) / total_h,
        "capacity": env._today_capacity,
        "dvc_truth": orders_total / max(1, env._today_capacity),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2",
                    default="clark/data/checkpoints/clark_foundation.pt.preDvCRatio.bak")
    ap.add_argument("--v21",
                    default="clark/data/checkpoints/clark_foundation_dvc_ft.pt")
    args = ap.parse_args()

    v2_path = Path(args.v2)
    v21_path = Path(args.v21)
    if not v2_path.exists():
        raise SystemExit(f"missing v2 checkpoint: {v2_path}")
    if not v21_path.exists():
        raise SystemExit(f"missing v2.1 checkpoint: {v21_path}")

    print(f"v2  source: {v2_path.name}")
    print(f"v2.1 source: {v21_path.name}")
    print()

    m_v2 = load_v2(v2_path)
    m_v21 = load_v21(v21_path)

    # Pre-generate scenarios (cfg + vol_override + seed) so both
    # policies see IDENTICAL episodes
    scenarios = []
    base_seed = 7777
    for sname, sdef in SCENARIOS.items():
        for i in range(sdef["n_eps"]):
            seed = base_seed + len(scenarios)
            # generate cfg under a fixed seed so cfg is identical across runs
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            cfg = generate_random_facility(stage=sdef["stage"])
            n = len(cfg.workers)
            avg_oph = sum(w.base_oph for w in cfg.workers) / max(1, n)
            cap = n * max(12.0, avg_oph * 9.0 * 0.22)
            vol_override = int(cap * sdef["vol_mult"])
            scenarios.append((sname, sdef["label"], cfg, vol_override, seed))

    results = []
    for sname, label, cfg, vol, seed in scenarios:
        n_workers = len(cfg.workers)
        r_v2 = run_one_episode(m_v2, cfg, vol, seed, slice_feats=True)
        r_v21 = run_one_episode(m_v21, cfg, vol, seed, slice_feats=False)
        results.append({
            "scenario": sname, "label": label, "seed": seed,
            "n_workers": n_workers, "vol": vol,
            "dvc_truth": r_v2["dvc_truth"],
            "v2": r_v2, "v21": r_v21,
        })

    # Report per-episode
    print("=" * 105)
    print(f"{'sce':<6} {'seed':>5} {'work':>4} {'vol':>5} {'dvc':>5}  "
          f"|  {'v2_grd':>6} {'v2_fl%':>7} {'v2_pp%':>7} {'v2_cmp':>6}  "
          f"|  {'21_grd':>6} {'21_fl%':>7} {'21_pp%':>7} {'21_cmp':>6}  "
          f"|  {'dFL':>5} {'dPP':>5}")
    print("=" * 105)
    for r in results:
        v2 = r["v2"]; v21 = r["v21"]
        d_fl = v21["filler_pct"] - v2["filler_pct"]
        d_pp = v21["pickpack_pct"] - v2["pickpack_pct"]
        print(f"{r['scenario']:<6} {r['seed']:>5} {r['n_workers']:>4} {r['vol']:>5} "
              f"{r['dvc_truth']:>5.2f}  |  "
              f"{v2['grade']:>6} {v2['filler_pct']:>6.1f}% {v2['pickpack_pct']:>6.1f}% "
              f"{v2['cmp_pct']*100:>5.0f}%  |  "
              f"{v21['grade']:>6} {v21['filler_pct']:>6.1f}% {v21['pickpack_pct']:>6.1f}% "
              f"{v21['cmp_pct']*100:>5.0f}%  |  "
              f"{d_fl:+5.1f} {d_pp:+5.1f}")

    # Aggregate by scenario
    print()
    print("=" * 90)
    print("SCENARIO MEANS")
    print("=" * 90)
    print(f"{'scenario':<10}  {'n':>3}   {'v2_filler%':>11}   {'v21_filler%':>12}   "
          f"{'delta':>7}   {'verdict':<20}")
    by_sce = {}
    for r in results:
        by_sce.setdefault(r["scenario"], []).append(r)
    for sname, rs in by_sce.items():
        mean_v2 = sum(r["v2"]["filler_pct"] for r in rs) / len(rs)
        mean_v21 = sum(r["v21"]["filler_pct"] for r in rs) / len(rs)
        delta = mean_v21 - mean_v2
        if sname == "STRESS":
            verdict = "WIN (lower is better)" if delta < -3 else (
                "tie" if abs(delta) <= 3 else "LOSE")
        else:
            verdict = "OK (stays equal)" if abs(delta) <= 3 else (
                "REGRESS" if delta > 3 else "over-suppressed")
        print(f"{sname:<10}  {len(rs):>3}   {mean_v2:>10.1f}%   {mean_v21:>11.1f}%   "
              f"{delta:>+6.1f}   {verdict}")

    # Final binary verdict
    print()
    print("=" * 90)
    print("FINAL VERDICT")
    print("=" * 90)
    stress_delta = (sum(r["v21"]["filler_pct"] for r in by_sce["STRESS"])
                    - sum(r["v2"]["filler_pct"] for r in by_sce["STRESS"])) / len(by_sce["STRESS"])
    mid_delta = (sum(r["v21"]["filler_pct"] for r in by_sce["MID"])
                 - sum(r["v2"]["filler_pct"] for r in by_sce["MID"])) / len(by_sce["MID"])
    easy_delta = (sum(r["v21"]["filler_pct"] for r in by_sce["EASY"])
                  - sum(r["v2"]["filler_pct"] for r in by_sce["EASY"])) / len(by_sce["EASY"])

    if stress_delta < -3 and mid_delta < 3 and easy_delta < 3:
        verdict = "YES: design works. v2.1 commits more on stress days without inflating easy-day filler."
    elif stress_delta < -3 and (mid_delta > 5 or easy_delta > 5):
        verdict = "OVER-CORRECTION CONFIRMED: stress wins but easy/mid filler climbed."
    elif stress_delta >= 0:
        verdict = "DESIGN FAILED: v2.1 did NOT reduce stress-day filler. Roll back."
    else:
        verdict = "MIXED: small effects, longer training needed or marginal regression."

    print(f"  stress filler delta: {stress_delta:+.1f}pp  (want: negative)")
    print(f"  mid    filler delta: {mid_delta:+.1f}pp  (want: ~0)")
    print(f"  easy   filler delta: {easy_delta:+.1f}pp  (want: ~0)")
    print()
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()

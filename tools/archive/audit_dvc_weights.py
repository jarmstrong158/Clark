"""Honest audit of how much the demand_vs_capacity feature is actually
influencing the policy.

Three measurements:

  1. Per-column stats on env_linear.weight. Column 17 is the new
     feature's input column. Comparing its magnitude to the other 17
     columns tells us whether the optimizer has lifted it to a
     "real" level or whether it's still dormant relative to peers.

  2. Behavioral ablation. Take a real stress-day scenario (high dvc),
     run the model forward TWICE:
        a) feats[17] = current dvc (e.g. ~1.8 on a stress day)
        b) feats[17] = 0 (the dormant transplant baseline)
     Measure the resulting shift in:
       - assignment logits (per worker, per task)
       - action probabilities (softmax of masked logits)
       - value head output
     If the shifts are tiny (e.g. < 0.01 KL on assignment dist), the
     feature is structurally there but behaviorally inert.

  3. Same ablation on a NORMAL-volume day (low dvc ~0.5). If the
     shift is larger on the stress day than the normal day, the model
     IS learning to use the signal differentially — the policy's
     response is conditional on dvc magnitude, which is the whole
     point of adding it.

If (1) shows col-17 < peer scale AND (2) shows tiny logit shifts
AND (3) shows no differential response, the answer is "weights are
nowhere near drastic enough yet — the policy is barely seeing the
signal." If (2) shows large shifts the answer is "weights are
small in magnitude but well-aligned with the right inputs and ARE
moving the policy."
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from clark.agent.ppo import ClarkAgent
from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.env.episode_generator import generate_episode
from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


def column_audit(model) -> None:
    """Measure 1: per-column stats on env_linear.weight."""
    w = model.env_linear.weight.detach().cpu().numpy()  # (d_model=512, env_feat_dim=18)
    print("=" * 70)
    print("MEASUREMENT 1: env_linear.weight per-column magnitude")
    print("=" * 70)
    print(f"{'col':>3}  {'feature':<28} {'max_abs':>8} {'mean_abs':>9} {'std':>8} {'L2':>8}")
    feat_names = [
        "time_of_day_norm",
        "orders_in_queue/total",
        "orders_completed/total",
        "orders_picked_not_audited/total",
        "restock_remaining/restock_hours",
        "side_project_progress",
        "season_winter", "season_spring", "season_summer", "season_fall",
        "is_high_volume",
        "management_progress",
        "picker_needs_replacement",
        "restock_level",
        "is_ot",
        "carrier_urgency",
        "order_complexity_load",
        "demand_vs_capacity (v2.1)",  # NEW
    ]
    col17 = None
    others = []
    for c in range(w.shape[1]):
        col = w[:, c]
        max_abs = abs(col).max()
        mean_abs = abs(col).mean()
        std = col.std()
        l2 = np.linalg.norm(col)
        marker = "  <-- NEW" if c == 17 else ""
        print(f"{c:>3}  {feat_names[c]:<28} {max_abs:>8.4f} {mean_abs:>9.4f} "
              f"{std:>8.4f} {l2:>8.3f}{marker}")
        if c == 17:
            col17 = (max_abs, mean_abs, l2)
        else:
            others.append((max_abs, mean_abs, l2))
    others_max = np.array([o[0] for o in others])
    others_mean = np.array([o[1] for o in others])
    others_l2 = np.array([o[2] for o in others])
    print()
    print(f"col 17 max_abs ratio to other-column median: "
          f"{col17[0] / np.median(others_max):.3f}")
    print(f"col 17 L2-norm ratio to other-column median: "
          f"{col17[2] / np.median(others_l2):.3f}")
    print(f"col 17 L2-norm percentile among 17 other cols: "
          f"{(others_l2 < col17[2]).sum()}/17")


def behavioral_ablation(model, env: FacilityEnv,
                        volume_override: int | None = None,
                        label: str = "scenario") -> dict:
    """Measurements 2/3: forward pass with feats[17] set vs zeroed,
    measure how much the policy distribution actually shifts."""
    env.reset()
    if volume_override is not None:
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

    # Advance a few ticks so dvc has stabilized (instant bucket landed)
    for _ in range(6):
        amask = get_action_mask(env)
        hmask = get_hustle_mask(env)
        actions = []
        for w in range(amask.shape[0]):
            valid = np.where(amask[w])[0]
            if len(valid) == 0:
                continue
            task_idx = int(np.random.choice(valid))
            h_valid = np.where(hmask[w])[0]
            hustle = bool(np.random.choice(h_valid)) if len(h_valid) else False
            actions.append((w, task_idx, hustle))
        env.step(actions)

    # Build the state
    from clark.agent.state import StateBuilder
    sb = StateBuilder(env.facility_config)
    state = sb.build(env)
    dvc_real = float(state["env_feats"][17])
    action_mask = get_action_mask(env)
    hustle_mask = get_hustle_mask(env)

    # Forward with real dvc
    with torch.no_grad():
        a_logits_real, h_logits_real, v_real, _ = model.forward(
            state, action_mask=action_mask, hustle_mask=hustle_mask,
            lstm_hidden=None,
        )

    # Forward with feats[17] zeroed
    state_zero = {k: v.copy() if hasattr(v, "copy") else v for k, v in state.items()}
    state_zero["env_feats"] = state["env_feats"].copy()
    state_zero["env_feats"][17] = 0.0
    with torch.no_grad():
        a_logits_zero, h_logits_zero, v_zero, _ = model.forward(
            state_zero, action_mask=action_mask, hustle_mask=hustle_mask,
            lstm_hidden=None,
        )

    # Compute deltas
    a_diff = (a_logits_real - a_logits_zero).abs()
    v_diff = float((v_real - v_zero).abs().item())

    # KL divergence on per-worker action distributions (proper measure)
    def softmax_masked(logits, mask):
        m = torch.from_numpy(mask).bool()
        logits = logits.masked_fill(~m, -1e9)
        return torch.softmax(logits, dim=-1)
    p_real = softmax_masked(a_logits_real, action_mask)
    p_zero = softmax_masked(a_logits_zero, action_mask)
    # KL(p_real || p_zero) per worker, then mean
    eps = 1e-10
    kl = (p_real * (torch.log(p_real + eps) - torch.log(p_zero + eps))).sum(dim=-1)
    kl_mean = float(kl.mean().item())
    kl_max = float(kl.max().item())

    # Largest argmax flip: how many workers would the model assign a different task to?
    flips = (p_real.argmax(dim=-1) != p_zero.argmax(dim=-1)).sum().item()
    n_workers = a_logits_real.shape[0]

    print()
    print("=" * 70)
    print(f"MEASUREMENT: behavioral ablation -- {label}")
    print("=" * 70)
    print(f"  total_orders (oracle):    {env.episode.total_orders}")
    print(f"  today_capacity:           {env._today_capacity:.1f}")
    print(f"  dvc at observation:       {dvc_real:.3f}")
    print(f"  truth ratio (orders/cap): {env.episode.total_orders/env._today_capacity:.3f}")
    print()
    print(f"  assignment_logits max abs change: {float(a_diff.max()):.4f}")
    print(f"  assignment_logits mean abs change: {float(a_diff.mean()):.4f}")
    print(f"  per-worker action-distribution KL (mean): {kl_mean:.4f}")
    print(f"  per-worker action-distribution KL (max):  {kl_max:.4f}")
    print(f"  argmax-task FLIPS due to feature: {flips}/{n_workers} workers")
    print(f"  value head shift: {v_diff:.4f}")
    return {"dvc": dvc_real, "kl_mean": kl_mean, "kl_max": kl_max,
            "flips": flips, "n_workers": n_workers, "v_diff": v_diff}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="clark/data/checkpoints/clark_foundation_dvc_ft.pt")
    args = p.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    agent = ClarkAgent.load(args.ckpt, device="cpu")
    print(f"  arch_version OK; ep recorded: {agent.episode if hasattr(agent, 'episode') else '?'}")
    print()
    column_audit(agent.model)

    # Two scenarios: a stress day (high dvc) and a normal day (low dvc).
    cfg_stress = generate_random_facility(stage=3)
    env_stress = FacilityEnv(cfg_stress)
    n = len(cfg_stress.workers)
    avg_oph = sum(w.base_oph for w in cfg_stress.workers) / max(1, n)
    cap = n * max(12.0, avg_oph * 9.0 * 0.22)
    normal_vol = int(cap * 0.45)
    stress_vol = int(cap * 1.8)

    res_n = behavioral_ablation(agent.model, env_stress,
                                volume_override=normal_vol,
                                label=f"NORMAL day (vol={normal_vol})")

    cfg_stress2 = generate_random_facility(stage=3)
    env_stress2 = FacilityEnv(cfg_stress2)
    n2 = len(cfg_stress2.workers)
    avg_oph2 = sum(w.base_oph for w in cfg_stress2.workers) / max(1, n2)
    cap2 = n2 * max(12.0, avg_oph2 * 9.0 * 0.22)
    stress_vol2 = int(cap2 * 1.8)
    res_s = behavioral_ablation(agent.model, env_stress2,
                                volume_override=stress_vol2,
                                label=f"STRESS day (vol={stress_vol2})")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  stress-day KL / normal-day KL ratio: "
          f"{res_s['kl_mean'] / max(1e-9, res_n['kl_mean']):.2f}x")
    print(f"  stress-day flips: {res_s['flips']}/{res_s['n_workers']}")
    print(f"  normal-day flips: {res_n['flips']}/{res_n['n_workers']}")
    print()
    if res_s["flips"] == 0 and res_n["flips"] == 0:
        print("  Feature is barely moving the policy yet. Weights are off zero")
        print("  but not yet large enough to flip any worker's argmax task.")
    elif res_s["flips"] > res_n["flips"]:
        print("  Feature IS conditional: causes more decision change on stress days")
        print("  than on normal days. Aligned with the design intent.")
    else:
        print("  Feature causes similar shifts on stress and normal days -- not yet")
        print("  learning the conditional response we want.")


if __name__ == "__main__":
    main()

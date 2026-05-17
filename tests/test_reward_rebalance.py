"""Completion-dominant reward regression tests (re-tuned, audit #2 2026-05-17).

Audit #1 found the root cause — a FAILED day netted net-POSITIVE reward,
so PPO had no gradient to stop leaving orders unshipped — and the SIGN
fix is kept. But audit #1's implementation (per_order_shipped=1.0 + a
pure terminal +4xT lump + UNCAPPED per_order_incomplete) was over-tuned:
three further independent audits found it (a) over-sparsified (no
learnable gradient on the ~21% of days that end incomplete) and (b) the
uncap recreated the fat-tailed value-target instability symlog can't
absorb. Re-tune: dense completion-graded shipped (3.0), completion
premium lump (3.0 x total_orders), and a REINSTATED looser cap (200).

These tests pin the re-tuned contract so it can't silently regress:
  1. weights are the re-tuned values,
  2. the completion bonus SCALES with the day's order total,
  3. the incomplete penalty is CAPPED at INCOMPLETE_CAP (200), and
  4. finishing a day STRICTLY DOMINATES a near-miss by >= the lump.
"""
from __future__ import annotations

from clark.config.schema import DEFAULT_REWARDS
from clark.env.facility_env import FacilityEnv, INCOMPLETE_CAP
from clark.training.synthetic_gen import generate_random_facility


def _env():
    cfg = generate_random_facility(stage=1)
    env = FacilityEnv(cfg)
    env.reset()
    env.ot_hours = 0.0  # drop OT term so deltas isolate order reward
    return env


def test_retuned_weights():
    assert DEFAULT_REWARDS["per_order_shipped"] == 3.0
    assert DEFAULT_REWARDS["all_orders_complete_bonus"] == 3.0
    assert DEFAULT_REWARDS["per_order_incomplete"] == -10.0
    assert INCOMPLETE_CAP == 200


def test_completion_bonus_scales_with_order_total():
    env = _env()
    T = env.episode.total_orders
    env.reward_breakdown = {k: 0.0 for k in env._rewards}
    env.is_done = False
    env.orders_in_queue = 0
    env.orders_picked_not_audited = 0
    env._finalize_episode()
    assert env.reward_breakdown["all_orders_complete_bonus"] == 3.0 * T, (
        "completion bonus must be +3 x total_orders (the finishing premium)"
    )


def test_incomplete_penalty_is_capped():
    """Below the cap it scales with remaining; above it, it floors at
    -10 x INCOMPLETE_CAP (bounds the value-target tail)."""
    env = _env()
    env.reward_breakdown = {k: 0.0 for k in env._rewards}
    env.is_done = False
    env.orders_in_queue = 120  # < cap (200) → scales
    env.orders_picked_not_audited = 0
    env._finalize_episode()
    assert env.reward_breakdown["per_order_incomplete"] == -10.0 * 120

    env2 = _env()
    env2.reward_breakdown = {k: 0.0 for k in env2._rewards}
    env2.is_done = False
    env2.orders_in_queue = 500  # > cap (200) → floored
    env2.orders_picked_not_audited = 0
    env2._finalize_episode()
    assert env2.reward_breakdown["per_order_incomplete"] == -10.0 * INCOMPLETE_CAP, (
        f"penalty must floor at -10 x {INCOMPLETE_CAP} = "
        f"{-10.0 * INCOMPLETE_CAP} (the value-tail bound), not -5000"
    )


def test_finishing_strictly_dominates_near_miss():
    """Core invariant: a finished day must out-reward a 1-order-short day
    by >= the completion lump (3 x total_orders). All other finalize
    terms are held identical, so they cancel in the delta."""
    env = _env()
    T = env.episode.total_orders

    env.reward_breakdown = {k: 0.0 for k in env._rewards}
    env.is_done = False
    env.orders_in_queue = 0
    env.orders_picked_not_audited = 0
    r_complete = env._finalize_episode()

    env.reward_breakdown = {k: 0.0 for k in env._rewards}
    env.is_done = False
    env.orders_in_queue = 1  # cheapest possible miss
    env.orders_picked_not_audited = 0
    r_one_short = env._finalize_episode()

    assert r_complete - r_one_short >= 3.0 * T, (
        f"finishing must dominate a 1-short day by >= the completion "
        f"lump (3*{T}={3*T}); got delta {r_complete - r_one_short:.1f}."
    )
    assert r_complete > 0, "a fully completed day must be net positive"

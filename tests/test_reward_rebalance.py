"""Completion-dominant reward rebalance regression tests (2026-05-17).

Three independent audits found the dominant failure: a FAILED day
netted net-POSITIVE reward, because per_order_shipped was paid linearly
and banked during the day while the binary finish/fail outcome was tiny
(flat +50 bonus, -500-capped penalty) and idle/filler were ~free. PPO
optimised correctly; the reward was mis-specified, so it could never
learn to suppress idle/filler on under-shipped days.

These tests pin the fix so it can't silently regress:
  1. the completion bonus SCALES with the day's order total,
  2. the incomplete penalty is UNCAPPED (no -500 floor), and
  3. finishing a day STRICTLY DOMINATES a near-miss by a large margin
     (the completion outcome, not the linear stream, controls reward).
"""
from __future__ import annotations

from clark.config.schema import DEFAULT_REWARDS
from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


def _env():
    cfg = generate_random_facility(stage=1)
    env = FacilityEnv(cfg)
    env.reset()
    env.ot_hours = 0.0  # drop OT term so deltas isolate order reward
    return env


def test_weights_are_completion_dominant():
    assert DEFAULT_REWARDS["per_order_shipped"] == 1.0
    assert DEFAULT_REWARDS["all_orders_complete_bonus"] == 4.0
    assert DEFAULT_REWARDS["per_order_incomplete"] == -10.0


def test_completion_bonus_scales_with_order_total():
    env = _env()
    T = env.episode.total_orders
    env.reward_breakdown = {k: 0.0 for k in env._rewards}
    env.is_done = False
    env.orders_in_queue = 0
    env.orders_picked_not_audited = 0
    env._finalize_episode()
    assert env.reward_breakdown["all_orders_complete_bonus"] == 4.0 * T, (
        "completion bonus must be +4 x total_orders, not a flat +50"
    )


def test_incomplete_penalty_is_uncapped():
    env = _env()
    env.reward_breakdown = {k: 0.0 for k in env._rewards}
    env.is_done = False
    env.orders_in_queue = 120  # > old INCOMPLETE_CAP of 50
    env.orders_picked_not_audited = 0
    env._finalize_episode()
    assert env.reward_breakdown["per_order_incomplete"] == -10.0 * 120, (
        "incomplete penalty must scale with full orders_remaining "
        "(was floored at -500 by INCOMPLETE_CAP=50)"
    )


def test_finishing_strictly_dominates_near_miss():
    """The core invariant: a finished day must out-reward a 1-order-short
    day by at least the completion lump (~4 x total_orders). All other
    finalize terms are held identical, so they cancel in the delta."""
    env = _env()
    T = env.episode.total_orders

    env.reward_breakdown = {k: 0.0 for k in env._rewards}
    env.is_done = False
    env.orders_in_queue = 0
    env.orders_picked_not_audited = 0
    r_complete = env._finalize_episode()

    env.reward_breakdown = {k: 0.0 for k in env._rewards}
    env.is_done = False
    env.orders_in_queue = 1  # one order short — the cheapest possible miss
    env.orders_picked_not_audited = 0
    r_one_short = env._finalize_episode()

    assert r_complete - r_one_short >= 4.0 * T, (
        f"finishing must dominate a 1-short day by >= the completion "
        f"lump (4*{T}={4*T}); got delta {r_complete - r_one_short:.1f}. "
        f"If this fails the binary outcome no longer controls reward."
    )
    assert r_complete > 0, "a fully completed day must be net positive"

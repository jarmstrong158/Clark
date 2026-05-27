"""Gated dense management-bonus regression tests (triple-audit 2026-05-18).

Root cause: the decisive management signal was a sparse end-of-day
grade-tier lump (+10 -> +30) on a day already worth ~3,600 in dense
shipped/completion reward — after PPO advantage normalization it was
noise, so the policy was numerically indifferent and dumped ~118 spare
worker-h/day into filler instead of the ~3h of management needed.

Fix: `management_progress_gated` pays a dense per-management-hour bonus
on the same per-tick scale as the shipped stream, but ONLY once the
day's orders are essentially shipped (gate) and only up to the required
hours (cap via the existing `< daily_cap` branch). It must be
structurally inert during crunch / before completion so it can never
trade against shipping.

These tests pin: the weight, that the bonus pays when orders are done,
and — critically — that it does NOT pay while orders are pending.
"""
from __future__ import annotations

from clark.config.schema import DEFAULT_REWARDS
from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


def _env_and_mgmt_worker():
    cfg = generate_random_facility(stage=1)
    env = FacilityEnv(cfg)
    env.reset()
    w = next(
        (x for x in env.episode.workers
         if x.worker_id in env._mgmt_eligible_ids and not x.is_absent),
        None,
    )
    if w is not None:
        w.is_absent = False
        w.management_hours = 0.0
    return env, w


def test_weight_present():
    assert DEFAULT_REWARDS["management_progress_gated"] == 6.0


def test_bonus_pays_when_shipping_under_control():
    """Gate opens while pending orders are <= 10% of the day's total
    (shipping on-track) — a real multi-hour window, not the dead
    pending==0 boundary the original gate used."""
    env, w = _env_and_mgmt_worker()
    if w is None:
        return  # no eligible non-absent worker this sample; skip silently
    env.episode.total_orders = 1000
    env.reward_breakdown = {k: 0.0 for k in env._rewards}
    env.orders_in_queue = 50                    # 5% pending -> under control
    env.orders_picked_not_audited = 0
    env._process_management(w, 1.0)
    assert env.reward_breakdown["management_progress_gated"] == 6.0, (
        "dense management bonus must pay while shipping is under control "
        "(pending <= 10% of total), not only at pending==0"
    )
    assert env.reward_breakdown["per_management_hour"] == 1.0  # v2.10 (gentler bump than v2.9's 1.5)


def test_bonus_blocked_during_crunch():
    """The decisive guarantee: inert during real backlog/crunch, so it
    can NEVER trade against shipping."""
    env, w = _env_and_mgmt_worker()
    if w is None:
        return
    env.episode.total_orders = 1000
    env.reward_breakdown = {k: 0.0 for k in env._rewards}
    env.orders_in_queue = 200                   # 20% pending -> crunch
    env.orders_picked_not_audited = 0
    env._process_management(w, 1.0)
    assert env.reward_breakdown["management_progress_gated"] == 0.0, (
        "gated bonus must be inert during crunch (pending > 10% of "
        "total) — it must never incentivize management over shipping"
    )
    assert env.reward_breakdown["per_management_hour"] == 1.0  # v2.10 (gentler bump than v2.9's 1.5)

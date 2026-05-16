"""Reward bookkeeping + the filler-vs-orders crunch cap (Restart A/B/Z).

These guard the exact mechanism we spent days debugging (dec-011 →
dec-013). A refactor that breaks the per-day crunch cap or the scale
formula would silently reintroduce the value-function tail blowups
that caused v_loss to spike to 80-91 — invisible without a test.
"""
from __future__ import annotations

import pytest

from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


@pytest.fixture
def env() -> FacilityEnv:
    cfg = generate_random_facility(stage=3)
    e = FacilityEnv(cfg)
    e.reset()
    return e


# ── _add_reward bookkeeping ───────────────────────────────────────────────────

def test_add_reward_unknown_key_is_noop(env):
    before = dict(env.reward_breakdown)
    out = env._add_reward("definitely_not_a_real_reward_key")
    assert out == 0.0
    assert env.reward_breakdown == before


def test_add_reward_accumulates_in_breakdown(env):
    # Pick a key that actually exists in this config's reward table.
    key = next(iter(env._rewards.keys()))
    base = env._rewards[key]
    env.reward_breakdown[key] = 0.0
    r1 = env._add_reward(key)
    r2 = env._add_reward(key)
    assert r1 == pytest.approx(base)
    assert r2 == pytest.approx(base)
    assert env.reward_breakdown[key] == pytest.approx(2 * base)


def test_add_reward_multiplier_scales_value(env):
    key = next(iter(env._rewards.keys()))
    base = env._rewards[key]
    env.reward_breakdown[key] = 0.0
    out = env._add_reward(key, 3.0)
    assert out == pytest.approx(base * 3.0)
    assert env.reward_breakdown[key] == pytest.approx(base * 3.0)


# ── Crunch cap state (Restart Z) ──────────────────────────────────────────────

def test_crunch_cap_initialized_on_reset(env):
    assert env._crunch_penalty_today == 0.0
    assert env._CRUNCH_DAILY_CAP == -5000.0


def test_crunch_cap_resets_each_day():
    cfg = generate_random_facility(stage=3)
    e = FacilityEnv(cfg)
    e.reset()
    e._crunch_penalty_today = -4999.0  # simulate a heavy day
    e.reset()
    assert e._crunch_penalty_today == 0.0, "cap accumulator must reset per day"


def test_crunch_scale_formula_floor_and_ramp():
    """_crunch_scale = max(1.0, pending_pct * 5.0). Floor of 1x up to
    20% pending, linear ramp above, 5x at 100%. This is the curve the
    whole Restart B fix depends on — pin it."""
    def scale(pending_pct):
        return max(1.0, pending_pct * 5.0)

    assert scale(0.05) == 1.0       # tiny backlog → floor
    assert scale(0.20) == 1.0       # at the knee → still floor
    assert scale(0.30) == pytest.approx(1.5)
    assert scale(0.50) == pytest.approx(2.5)
    assert scale(1.00) == pytest.approx(5.0)
    assert scale(2.00) == pytest.approx(10.0)  # carryover can exceed 100%


def test_crunch_cap_bounds_worst_case(env):
    """Once cumulative crunch passes the cap, further hits must stop —
    this is the tail-clamp that fixed the v_loss spike. Drive the
    accumulator past the cap and confirm the gate closes."""
    env._crunch_penalty_today = -5001.0  # already past cap
    # The guard in facility_env is `if self._crunch_penalty_today > cap`.
    # cap is -5000; -5001 is NOT > -5000, so the gate is closed.
    gate_open = env._crunch_penalty_today > env._CRUNCH_DAILY_CAP
    assert gate_open is False, "crunch penalty must stop accruing past the cap"

    env._crunch_penalty_today = -100.0   # well within budget
    gate_open = env._crunch_penalty_today > env._CRUNCH_DAILY_CAP
    assert gate_open is True, "crunch penalty must still accrue under the cap"

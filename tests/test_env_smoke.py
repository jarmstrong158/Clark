"""End-to-end env smoke: reset + many steps must not crash or NaN.

This is the integration backstop. The unit tests pin individual
mechanisms; this proves the whole step loop holds together across a
realistic slice of a day with random valid actions — the exact thing
a silent refactor regression would break without a unit test
noticing.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


def _random_valid_actions(env, rng):
    """Build the (worker_id, task_idx, hustle) tuple list step() expects."""
    mask = get_action_mask(env)
    hustle_mask = get_hustle_mask(env)
    actions = []
    for w in range(mask.shape[0]):
        task_idx = int(rng.choice(np.where(mask[w])[0]))
        hustle = bool(rng.choice(np.where(hustle_mask[w])[0]))
        actions.append((w, task_idx, hustle))
    return actions


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_reset_returns_finite_state(stage):
    env = FacilityEnv(generate_random_facility(stage=stage))
    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert np.isfinite(state).all(), "reset() produced non-finite state"


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_full_day_step_loop_no_crash(stage):
    """Run a full day to terminal with random valid actions. Asserts:
    state stays finite, reward stays finite, episode terminates."""
    env = FacilityEnv(generate_random_facility(stage=stage))
    env.reset()
    rng = np.random.default_rng(stage)

    done = False
    steps = 0
    MAX_STEPS = 500  # a day is well under this many ticks
    while not done and steps < MAX_STEPS:
        actions = _random_valid_actions(env, rng)
        state, reward, done, _info = env.step(actions)
        assert np.isfinite(np.asarray(state)).all(), f"NaN state at step {steps}"
        assert math.isfinite(float(reward)), f"NaN reward at step {steps}"
        steps += 1

    assert done, f"episode did not terminate within {MAX_STEPS} steps"


def test_reward_breakdown_sums_are_finite():
    """After a full day, every reward-breakdown component must be finite
    — an inf/nan in any single component is the signature of the value
    blowups we just fixed."""
    env = FacilityEnv(generate_random_facility(stage=3))
    env.reset()
    rng = np.random.default_rng(42)
    done = False
    steps = 0
    while not done and steps < 500:
        actions = _random_valid_actions(env, rng)
        _state, _reward, done, _info = env.step(actions)
        steps += 1
    for k, v in env.reward_breakdown.items():
        assert math.isfinite(v), f"reward component {k!r} = {v} is not finite"


def test_task_hours_tracked_when_workers_active():
    """The per-worker task_hours_today diagnostic (con-009, permanent)
    must actually accumulate — it's load-bearing for the audits that
    found the filler-task escape valves."""
    env = FacilityEnv(generate_random_facility(stage=2))
    env.reset()
    rng = np.random.default_rng(7)
    for _ in range(30):
        actions = _random_valid_actions(env, rng)
        _state, _reward, done, _info = env.step(actions)
        if done:
            break
    total_tracked = sum(
        sum(w.task_hours_today.values()) for w in env.episode.workers
    )
    assert total_tracked > 0.0, (
        "task_hours_today never accumulated — the diagnostic that found "
        "the filler escape valves is broken"
    )

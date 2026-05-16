"""Action-mask invariants.

The single most dangerous failure here is an all-False row: that
NaNs the policy softmax and kills a training run with a cryptic
error hours later. The mask also encodes business rules (absent →
idle only, pack-only, OT branch) that silently change agent
behavior if broken.
"""
from __future__ import annotations

import numpy as np
import pytest

from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


@pytest.fixture(params=[1, 2, 3])
def env(request) -> FacilityEnv:
    cfg = generate_random_facility(stage=request.param)
    e = FacilityEnv(cfg)
    e.reset()
    return e


# ── Shape + dtype ─────────────────────────────────────────────────────────────

def test_mask_shape_matches_workers_and_tasks(env):
    mask = get_action_mask(env)
    n = len(env.episode.workers)
    m = env._num_tasks
    assert mask.shape == (n, m)
    assert mask.dtype == bool


def test_hustle_mask_shape(env):
    mask = get_hustle_mask(env)
    assert mask.shape == (len(env.episode.workers), 2)
    assert mask.dtype == bool


# ── The NaN guard: no all-False rows ──────────────────────────────────────────

def test_no_worker_has_zero_valid_actions(env):
    """Every worker must have >=1 valid task every step or the softmax
    NaNs. This is THE invariant that matters most."""
    mask = get_action_mask(env)
    rows_with_no_action = np.where(~mask.any(axis=1))[0]
    assert len(rows_with_no_action) == 0, (
        f"workers {rows_with_no_action.tolist()} have an all-False mask "
        f"row — policy softmax will NaN"
    )


def test_no_zero_mask_after_stepping(env):
    """Drive the env forward a few steps with random valid actions and
    re-check — the invariant must hold mid-episode too, not just at
    reset (OT branch, pick-buffer-full, shift-exhaustion all kick in
    later in the day)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        mask = get_action_mask(env)
        assert mask.any(axis=1).all(), "all-False row appeared mid-episode"
        hustle_mask = get_hustle_mask(env)
        actions = []
        for w in range(mask.shape[0]):
            task_idx = int(rng.choice(np.where(mask[w])[0]))
            hustle = bool(rng.choice(np.where(hustle_mask[w])[0]))
            actions.append((w, task_idx, hustle))
        _state, _reward, done, _info = env.step(actions)
        if done:
            break


def test_hustle_mask_no_hustle_column_always_valid(env):
    """Column 0 (no-hustle) must be True for every worker — otherwise a
    worker who can't hustle has no valid hustle action."""
    mask = get_hustle_mask(env)
    assert mask[:, 0].all(), "no-hustle option must always be available"


# ── Business-rule encoding ────────────────────────────────────────────────────

def test_absent_worker_only_idle(env):
    idle_idx = env._task_to_idx.get("idle", 0)
    w = env.episode.workers[0]
    w.is_absent = True
    mask = get_action_mask(env)
    assert mask[0, idle_idx]
    # Every non-idle column must be False for the absent worker.
    non_idle = [j for j in range(env._num_tasks) if j != idle_idx]
    assert not mask[0, non_idle].any()


def test_pack_only_worker_restricted(env):
    if "pack" not in env._task_to_idx:
        pytest.skip("facility has no pack task")
    pack_idx = env._task_to_idx["pack"]
    idle_idx = env._task_to_idx.get("idle", 0)
    w = env.episode.workers[0]
    w.is_pack_only = True
    w.is_absent = False
    env.is_ot = False
    mask = get_action_mask(env)
    allowed = set(np.where(mask[0])[0].tolist())
    assert allowed.issubset({pack_idx, idle_idx})
    assert pack_idx in allowed


def test_idle_not_valid_in_normal_branch(env):
    """Idle is deliberately INVALID in the normal branch — without this
    a deterministic-at-init policy collapses to 'everyone idle' forever
    (idle never accrues the reward signals that would push it out)."""
    idle_idx = env._task_to_idx.get("idle", None)
    if idle_idx is None:
        pytest.skip("no idle task")
    # Ensure a clean normal-branch state: not OT, nobody absent/pack-only,
    # not shift-exhausted.
    env.is_ot = False
    for w in env.episode.workers:
        w.is_absent = False
        w.is_pack_only = False
        w.hours_worked = 0.0
    mask = get_action_mask(env)
    # At least the workers in the normal branch should NOT have idle set.
    # (Absent / exhausted branches can set idle, but we cleared those.)
    assert not mask[:, idle_idx].any(), (
        "idle should be masked off in the normal branch"
    )

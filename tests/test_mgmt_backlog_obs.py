"""Tests for v2.5's mgmt_backlog observation feature (env_feats[17]).

v2.5 adds management_backlog as an observation so the policy can learn
to prevent the multi-day backlog accumulator from firing the grading
demerit. Without this, v2.7's policy could not even SEE the failure
mode it was triggering (audit found 80% of v2.7 C days had no other
measurable demerit firing — the missing demerit was backlog).

This file pins:
1. env_feats shape is 18 dims (the arch contract).
2. feats[17] is the mgmt_backlog normalized by week threshold.
3. The value clips to [0, 1] even when backlog exceeds threshold.
4. No-oracle property: feature does NOT read episode.total_orders.
"""
from __future__ import annotations

import numpy as np
import pytest

from clark.agent.state import StateBuilder
from clark.env.facility_env import FacilityEnv, ENV_STATE_SIZE
from clark.training.synthetic_gen import generate_random_facility


def test_env_state_size_is_18():
    """v2.5 contract: ENV_STATE_SIZE = 18 (was 17 in v2)."""
    assert ENV_STATE_SIZE == 18


def test_state_builder_emits_18_dim_env_feats():
    cfg = generate_random_facility(stage=2)
    env = FacilityEnv(cfg)
    env.reset()
    state = StateBuilder(cfg).build(env)
    assert state["env_feats"].shape == (18,)
    assert np.isfinite(state["env_feats"]).all()


def test_feature_17_zero_at_episode_start():
    """Fresh episode: backlog is 0 (no prior days carried over), so
    feats[17] should be 0.0."""
    cfg = generate_random_facility(stage=2)
    env = FacilityEnv(cfg)
    env.reset()
    state = StateBuilder(cfg).build(env)
    assert state["env_feats"][17] == 0.0


def test_feature_17_reflects_injected_backlog():
    """When year_env injects _mgmt_backlog, the observation should
    reflect it. Forces a backlog of half the threshold and verifies
    feats[17] = 0.5."""
    cfg = generate_random_facility(stage=2)
    env = FacilityEnv(cfg)
    env.reset()
    threshold = cfg.rules.management_backlog_week_threshold
    env._mgmt_backlog = threshold * 0.5
    state = StateBuilder(cfg).build(env)
    assert abs(state["env_feats"][17] - 0.5) < 1e-6


def test_feature_17_clips_at_one():
    """Backlog can exceed threshold (env scales the penalty
    quadratically) but the observation must stay bounded for network
    stability."""
    cfg = generate_random_facility(stage=2)
    env = FacilityEnv(cfg)
    env.reset()
    threshold = cfg.rules.management_backlog_week_threshold
    env._mgmt_backlog = threshold * 5  # way above threshold
    state = StateBuilder(cfg).build(env)
    assert state["env_feats"][17] == 1.0


def test_feature_17_no_oracle():
    """The feature is computed from env state, not from total_orders.
    Mutating total_orders must not change the feature."""
    cfg = generate_random_facility(stage=2)
    env = FacilityEnv(cfg)
    env.reset()
    env._mgmt_backlog = 3.0
    r_before = StateBuilder(cfg).build(env)["env_feats"][17]
    env.episode.total_orders = env.episode.total_orders // 2
    r_after = StateBuilder(cfg).build(env)["env_feats"][17]
    assert r_before == r_after, (
        "mgmt_backlog feature must not read total_orders — would be an "
        "oracle leak that doesn't transfer to production"
    )

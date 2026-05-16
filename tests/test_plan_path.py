"""Plan / inference-path regression tests.

The 392-test suite covered the training path but never exercised
`clark plan` end-to-end, so a contract drift shipped on master:
select_action_from_dict returns a 5-tuple but cli/main.py and
api/main.py unpacked 4, crashing `clark plan` with
`too many values to unpack (expected 4)` the moment a checkpoint
existed. (Caught by an external review, not by us — this file
closes that gap.)

These tests pin:
  1. the exact 5-tuple contract of select_action_from_dict, and
  2. that the cli/api unpack pattern matches it,
so this class of return-arity drift can't silently ship again.
"""
from __future__ import annotations

import numpy as np
import torch

from clark.agent.ppo import ClarkAgent
from clark.agent.state import StateBuilder
from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


def _plan_inputs(stage=1):
    """Reproduce exactly what cli.cmd_plan / api /plan feed the agent."""
    cfg = generate_random_facility(stage=stage)
    env = FacilityEnv(cfg)
    env.reset()
    builder = StateBuilder(cfg)
    state_dict = builder.build(env)
    mask = get_action_mask(env)
    hmask = get_hustle_mask(env)
    agent = ClarkAgent(device="cpu", use_amp=False)
    return agent, state_dict, mask, hmask, env


def test_select_action_from_dict_returns_exactly_five():
    """The contract that broke: 5 values, in order, correct types."""
    agent, sd, mask, hmask, env = _plan_inputs()
    out = agent.select_action_from_dict(sd, mask, hustle_mask=hmask)

    assert isinstance(out, tuple), "must return a tuple"
    assert len(out) == 5, (
        f"select_action_from_dict must return a 5-tuple "
        f"(task_actions, hustle_actions, task_lp, hustle_lp, value); "
        f"got {len(out)}. cli/main.py + api/main.py unpack exactly this."
    )
    task_actions, hustle_actions, task_lp, hustle_lp, value = out
    n = len(env.episode.workers)
    assert isinstance(task_actions, list) and len(task_actions) == n
    assert isinstance(hustle_actions, list) and len(hustle_actions) == n
    assert all(isinstance(a, int) for a in task_actions)
    assert all(h in (0, 1) for h in hustle_actions)
    for t in (task_lp, hustle_lp, value):
        assert isinstance(t, torch.Tensor)
        assert torch.isfinite(t).all()
        assert not t.requires_grad, "must be detached (no rollout-time graph)"


def test_cli_plan_unpack_pattern_matches():
    """Literally the cli/main.py:355 + api/main.py:448 unpack. If the
    return arity drifts again, this fails instead of `clark plan`."""
    agent, sd, mask, hmask, env = _plan_inputs()
    # cli/main.py form:
    task_actions, hustle_actions, _task_lp, _hustle_lp, _value = (
        agent.select_action_from_dict(sd, mask, hustle_mask=hmask)
    )
    assert len(task_actions) == len(hustle_actions) == len(env.episode.workers)


def test_plan_path_runs_without_mask_arg():
    """action_mask is Optional in the signature — the no-mask path must
    also yield the 5-tuple (defensive: a caller might omit it)."""
    agent, sd, _mask, _hmask, env = _plan_inputs()
    out = agent.select_action_from_dict(sd)
    assert len(out) == 5

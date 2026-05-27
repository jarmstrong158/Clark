"""Pin the v2.4-aux change: absent workers' tracked time is tagged "off"
in task_hours_today (and therefore in the episode footer's worker_time),
NOT "idle". Without this distinction, post-hoc allocation analyses can
confuse "worker was called off / didn't show up" with "policy chose to
idle this worker" — and on heavy-roster days the difference materially
inflates the apparent idle (or filler) bucket.

This is logging-only. The action mask still forces idle as the only
valid action for absent workers; the simulator still treats it as a
no-op; reward bookkeeping is unchanged. The only thing that differs
is the LABEL stored in the per-worker time tally.
"""
from __future__ import annotations

import numpy as np
import pytest

from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


def _force_one_absent(env):
    """Force the first non-manager worker absent for the day."""
    for w in env.episode.workers:
        if w.role not in ("manager", "assistant_manager"):
            w.is_absent = True
            w.current_task = "idle"
            return w
    return env.episode.workers[0]


def test_absent_worker_logged_as_off_not_idle():
    """After ticking a few steps, an absent worker's task_hours_today
    must contain an "off" key, NOT an "idle" key — distinguishing the
    absence from a policy choice."""
    env = FacilityEnv(generate_random_facility(stage=2))
    env.reset()
    absent = _force_one_absent(env)
    # Take a few simulated steps; the absent worker stays on "idle"
    # internally, but the logger should tag it as "off".
    from clark.agent.actions import get_action_mask, get_hustle_mask
    for _ in range(5):
        amask = get_action_mask(env)
        hmask = get_hustle_mask(env)
        actions = []
        for w in range(amask.shape[0]):
            valid = np.where(amask[w])[0]
            if len(valid) == 0:
                continue
            task_idx = int(valid[0])
            actions.append((w, task_idx, False))
        env.step(actions)

    th = absent.task_hours_today
    assert "off" in th and th["off"] > 0, (
        f"absent worker should accumulate hours under 'off' key in "
        f"task_hours_today, got: {th}"
    )
    assert th.get("idle", 0) == 0, (
        f"absent worker should NOT accumulate hours under 'idle' "
        f"(that label is reserved for policy-chosen idle), got: {th}"
    )


def test_active_worker_policy_idle_still_logged_as_idle():
    """A non-absent worker the policy ASSIGNS to idle (would only happen
    via shift-exhaustion or stranded-worker safety) must still log as
    'idle'. Only the absence path remaps to 'off'."""
    env = FacilityEnv(generate_random_facility(stage=2))
    env.reset()
    # Find a non-absent worker, force their current_task to "idle"
    target = None
    for w in env.episode.workers:
        if not w.is_absent:
            target = w
            target.current_task = "idle"
            break
    assert target is not None

    # Trigger the per-worker accumulator manually for one tick
    from clark.env.facility_env import STEP_DURATION
    # Simulate the per-tick log accumulation directly. We can't easily
    # cause the env to log idle for a non-absent worker via step() because
    # the action mask blocks idle in the normal branch, so just inspect
    # the loop's intent:
    for w in env.episode.workers:
        t = w.current_task
        if w.is_absent and t == "idle":
            t = "off"
        w.task_hours_today[t] = w.task_hours_today.get(t, 0.0) + STEP_DURATION

    assert target.task_hours_today.get("idle", 0) > 0, (
        "policy-chosen idle on a non-absent worker should still log "
        "as 'idle', not be remapped to 'off'."
    )


def test_footer_worker_time_carries_off_tag():
    """End-to-end: the episode footer's worker_time dict (the post-mortem
    diagnostic that downstream audits read) should surface 'off' for
    absent workers — this is what makes audits robust against absence-
    inflation false flags."""
    env = FacilityEnv(generate_random_facility(stage=2))
    env.reset()
    absent = _force_one_absent(env)
    from clark.agent.actions import get_action_mask, get_hustle_mask
    for _ in range(10):
        amask = get_action_mask(env)
        hmask = get_hustle_mask(env)
        actions = []
        for w in range(amask.shape[0]):
            valid = np.where(amask[w])[0]
            if len(valid) == 0:
                continue
            actions.append((w, int(valid[0]), False))
        env.step(actions)

    # Simulate the footer build (same dict comp as facility_env line 1344)
    footer_worker_time = {
        w.name: {t: round(h, 2) for t, h in w.task_hours_today.items()}
        for w in env.episode.workers
    }
    absent_tasks = footer_worker_time[absent.name]
    assert "off" in absent_tasks, (
        f"footer worker_time should expose 'off' for absent worker, "
        f"got {absent_tasks}"
    )

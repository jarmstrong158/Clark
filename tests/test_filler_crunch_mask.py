"""Tests for v2.5's multi-gate hard-mask on filler.

v2.5 mask fires when ANY of FOUR gates trips:
  1. projected_dvc > 0.65   (proactive: morning curve says heavy)
  2. pending_pct > 0.25     (reactive: queue piling up)
  3. schedule_pressure > 0.20 (reactive: behind schedule)
  4. time_pressure > 0.85   (reactive: not enough time for remaining work)

Gate 4 is the "manager-clock check" that v2.4-only-projection missed:
on a mid-range day where projection says "rescuable," the policy
could still drown in filler until the math of "orders remaining
divided by capacity remaining" exceeded 1.0 — too late to recover.

This file pins the four-gate behavior plus the safety invariants
(no oracle leak on the projection, no stranded workers, no zero
masks).
"""
from __future__ import annotations

import numpy as np
import pytest

from clark.agent.actions import get_action_mask
from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility


FILLER_TASKS = {"side_project", "loading", "training", "quality_check",
                "returns_processing", "receiving"}


def _filler_indices(env) -> list[int]:
    return [i for i, tid in enumerate(env._task_ids) if tid in FILLER_TASKS]


def _force_projection(env, ratio: float):
    from clark.env.episode_generator import expected_fraction_arrived
    day_start = env.facility_config.rules.day_start_hour
    cutoff = env.facility_config.rules.order_cutoff_hour
    expected = expected_fraction_arrived(env.current_hour, day_start, cutoff)
    if expected <= 1e-4:
        env.current_hour = day_start + 1e-3
        expected = expected_fraction_arrived(env.current_hour, day_start, cutoff)
    target_arrived = int(ratio * env._today_capacity * expected)
    env.arrived_so_far = max(1, target_arrived)


def _zero_all_pressures(env):
    """Suppress every other gate so we can test one in isolation."""
    env.arrived_so_far = 0
    env.orders_in_queue = 0
    env.orders_picked_not_audited = 0
    env.orders_completed = env.episode.total_orders
    env.current_hour = env.facility_config.rules.day_start_hour
    # Clear the minimum-dwell lock so filler-mask gates are tested in
    # isolation (a mid-dwell worker is locked to their current task).
    for w in env.episode.workers:
        w.ticks_on_task = 999


def test_gate1_projection_fires_mask():
    env = FacilityEnv(generate_random_facility(stage=3))
    env.reset()
    if not _filler_indices(env):
        pytest.skip("no filler tasks")
    _zero_all_pressures(env)
    _force_projection(env, 1.5)  # well above 0.65
    # Reset completion so the schedule_pressure gate doesn't ALSO fire
    env.orders_completed = 0
    mask = get_action_mask(env)
    for j in _filler_indices(env):
        for w in range(mask.shape[0]):
            if env.episode.workers[w].is_absent: continue
            assert not mask[w, j], (
                f"projection gate (proj_dvc=1.5) should mask filler"
            )


def test_gate2_pending_fires_mask():
    """Even with low projection, if queue piles up >25% the mask fires."""
    env = FacilityEnv(generate_random_facility(stage=3))
    env.reset()
    if not _filler_indices(env):
        pytest.skip("no filler tasks")
    _zero_all_pressures(env)
    # No arrivals -> projection=0. Force queue.
    env.orders_in_queue = int(0.40 * env.episode.total_orders)
    env.orders_completed = 0
    mask = get_action_mask(env)
    for j in _filler_indices(env):
        for w in range(mask.shape[0]):
            if env.episode.workers[w].is_absent: continue
            assert not mask[w, j], (
                "pending_pct=0.40 should fire the reactive queue gate"
            )


def test_gate3_schedule_pressure_fires_mask():
    """At mid-day with no completion progress, schedule_pressure fires."""
    env = FacilityEnv(generate_random_facility(stage=3))
    env.reset()
    if not _filler_indices(env):
        pytest.skip("no filler tasks")
    _zero_all_pressures(env)
    # Mid-day, 0% complete -> schedule_pressure ~= time_progress (0.5)
    env.current_hour = env.facility_config.rules.day_start_hour + 0.5 * (
        env._eod_hour - env.facility_config.rules.day_start_hour
    )
    env.orders_completed = 0
    # Suppress pending so gate 2 doesn't carry it
    env.orders_in_queue = 0
    env.orders_picked_not_audited = 0
    mask = get_action_mask(env)
    # schedule_pressure = 0.5 - 0 = 0.5 > 0.20 -> fires
    for j in _filler_indices(env):
        for w in range(mask.shape[0]):
            if env.episode.workers[w].is_absent: continue
            assert not mask[w, j]


def test_gate4_time_pressure_fires_mask():
    """The manager-clock check: low projection AND low pending_pct AND
    fine schedule, but orders/remaining-capacity is high because time
    is short. Mask should still fire."""
    env = FacilityEnv(generate_random_facility(stage=3))
    env.reset()
    if not _filler_indices(env):
        pytest.skip("no filler tasks")
    _zero_all_pressures(env)
    # Late in the day: drain almost all worker hours_remaining.
    # hours_remaining is a derived property: shift_hours - hours_worked - hours_lost.
    # Bump hours_worked so each worker has only ~0.5h remaining.
    for w in env.episode.workers:
        if not w.is_absent:
            w.hours_worked = max(0.0, w.shift_hours - w.hours_lost - 0.5)
    # Modest queue (low pending_pct) but capacity is tiny so time_pressure spikes
    env.orders_in_queue = max(int(0.15 * env.episode.total_orders), 10)
    env.orders_completed = int(0.85 * env.episode.total_orders)
    # Push the clock forward so schedule_pressure stays bounded
    env.current_hour = env._eod_hour - 0.5
    tp = env._compute_time_pressure()
    if tp <= 0.85:
        pytest.skip(f"setup didn't produce time_pressure>0.85, got {tp:.2f}")
    mask = get_action_mask(env)
    for j in _filler_indices(env):
        for w in range(mask.shape[0]):
            if env.episode.workers[w].is_absent: continue
            assert not mask[w, j], (
                f"time_pressure={tp:.2f} should fire mask"
            )


def test_filler_available_when_all_gates_clear():
    """All four gates dormant -> filler stays valid. Tests that we
    haven't broken the easy-day case."""
    env = FacilityEnv(generate_random_facility(stage=3))
    env.reset()
    if not _filler_indices(env):
        pytest.skip("no filler tasks")
    _zero_all_pressures(env)
    # Low projection, low pending, no schedule pressure, no time pressure
    _force_projection(env, 0.30)
    env.orders_in_queue = int(0.05 * env.episode.total_orders)
    env.orders_completed = int(0.30 * env.episode.total_orders)
    env.current_hour = env.facility_config.rules.day_start_hour + 0.3 * (
        env._eod_hour - env.facility_config.rules.day_start_hour
    )
    mask = get_action_mask(env)
    any_filler_valid = False
    for w in range(mask.shape[0]):
        if env.episode.workers[w].is_absent: continue
        if any(mask[w, j] for j in _filler_indices(env)):
            any_filler_valid = True
            break
    assert any_filler_valid, (
        "all gates clear -> filler should remain a valid choice "
        "(rescuable days need filler as a productive option)"
    )


def test_no_zero_mask_under_combined_stress():
    """Worst case: every gate firing. Workers must still have at least
    one valid option (pick/pack/restock/management)."""
    env = FacilityEnv(generate_random_facility(stage=3))
    env.reset()
    _force_projection(env, 2.0)
    env.orders_in_queue = int(0.80 * env.episode.total_orders)
    env.orders_completed = 0
    env.current_hour = env._eod_hour - 0.1
    mask = get_action_mask(env)
    for w in range(mask.shape[0]):
        assert mask[w].any(), f"worker {w} stranded under all-gate stress"


def test_projection_is_no_oracle():
    env = FacilityEnv(generate_random_facility(stage=3))
    env.reset()
    env.arrived_so_far = 100
    r_before = env._compute_projected_demand_ratio()
    env.episode.total_orders = env.episode.total_orders // 4
    r_after = env._compute_projected_demand_ratio()
    assert r_before == r_after, (
        f"_compute_projected_demand_ratio leaked from oracle: "
        f"{r_before} -> {r_after}"
    )


def test_gate5_restock_pressure_fires_mask():
    """v2.6: when restock_level drops below 0.35 the mask should fire
    even if no other stress gate is tripped. This is the proactive
    restock-collapse gate — fires BEFORE the 0.20 cliff so restockers
    can recover at normal speed."""
    env = FacilityEnv(generate_random_facility(stage=3))
    env.reset()
    if not _filler_indices(env):
        pytest.skip("no filler tasks")
    _zero_all_pressures(env)
    if not env._restock_enabled:
        pytest.skip("facility has no restock task")
    # Drop the stock to the gate-firing range. Suppress other gates.
    env.restock_level = 0.25  # below 0.35 threshold
    env.orders_in_queue = 0  # don't trip pending gate
    env.orders_completed = int(0.50 * env.episode.total_orders)
    env.current_hour = env.facility_config.rules.day_start_hour + 0.5 * (
        env._eod_hour - env.facility_config.rules.day_start_hour
    )
    mask = get_action_mask(env)
    for j in _filler_indices(env):
        for w in range(mask.shape[0]):
            if env.episode.workers[w].is_absent: continue
            assert not mask[w, j], (
                "restock_level=0.25 should fire restock_pressure gate, "
                "masking filler so restock proactivity wins"
            )


def test_gate5_dormant_when_restock_full():
    """At normal stock levels the gate doesn't fire — easy days should
    keep their normal filler optionality."""
    env = FacilityEnv(generate_random_facility(stage=3))
    env.reset()
    if not _filler_indices(env):
        pytest.skip("no filler tasks")
    _zero_all_pressures(env)
    env.restock_level = 0.85  # well above threshold
    env.orders_in_queue = int(0.05 * env.episode.total_orders)
    env.orders_completed = int(0.30 * env.episode.total_orders)
    mask = get_action_mask(env)
    any_filler = False
    for w in range(mask.shape[0]):
        if env.episode.workers[w].is_absent: continue
        if any(mask[w, j] for j in _filler_indices(env)):
            any_filler = True; break
    assert any_filler, (
        "restock at 0.85 should not fire the gate — easy days keep "
        "filler optionality"
    )


def test_v26_thresholds_pinned():
    """Pin all five thresholds so regressions surface in tests."""
    import inspect
    from clark.agent import actions
    src = inspect.getsource(actions)
    assert "_projected_dvc > 0.65" in src, "projection threshold = 0.65"
    assert "_pending_pct > 0.25" in src, "pending threshold = 0.25"
    assert "_schedule_pressure > 0.20" in src, "schedule threshold = 0.20"
    assert "_time_pressure > 0.85" in src, "time_pressure threshold = 0.85"
    assert "restock_level < 0.35" in src, (
        "v2.6 restock_pressure threshold = 0.35 (0.15 buffer above the "
        "0.20 picking-speed-collapse cliff)"
    )
    assert "_compute_time_pressure" in src, (
        "mask must call the env's _compute_time_pressure (manager-clock check)"
    )

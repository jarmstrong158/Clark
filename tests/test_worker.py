"""WorkerState — OPH math, fatigue, soreness, eligibility, hustle.

Pure logic with no env coupling, so cheap to test exhaustively. These
multipliers feed directly into the reward calc; a silent change to
the OPH chain would skew every training run without an obvious error.
"""
from __future__ import annotations

import pytest

from clark.env.worker import (
    FATIGUE_OPH_PENALTY,
    HUSTLE_EXHAUSTION_MULTIPLIER,
    HUSTLE_MODE_BONUS,
    PACK_FATIGUE_THRESHOLD,
    PICK_FATIGUE_THRESHOLD,
    SORENESS_HOUR_THRESHOLD,
    SORENESS_OPH_PENALTY,
    WorkerState,
)


def _w(**kw) -> WorkerState:
    base = dict(worker_id=0, name="W", base_oph=20.0, shift_hours=8.0, role="warehouse")
    base.update(kw)
    return WorkerState(**base)


# ── Properties ────────────────────────────────────────────────────────────────

def test_shift_end_and_hours_remaining():
    w = _w(shift_hours=8.0, shift_start=9.0)
    assert w.shift_end == 17.0
    assert w.hours_remaining == 8.0
    w.hours_worked = 3.0
    assert w.hours_remaining == 5.0
    w.hours_lost = 1.0
    assert w.hours_remaining == 4.0


def test_hours_remaining_never_negative():
    w = _w(shift_hours=8.0)
    w.hours_worked = 10.0
    assert w.hours_remaining == 0.0


def test_hustle_daily_cap_default_and_override():
    assert _w().hustle_daily_cap == 8.0
    assert _w(_hustle_daily_cap=5.0).hustle_daily_cap == 5.0


def test_hustle_weekly_threshold_is_double_daily():
    w = _w(_hustle_daily_cap=6.0)
    assert w.hustle_weekly_threshold == 12.0


def test_hustle_remaining_and_can_hustle():
    w = _w(_hustle_daily_cap=8.0)
    assert w.can_hustle is True
    w.hustle_hours_today = 8.0
    assert w.hustle_daily_remaining == 0.0
    assert w.can_hustle is False
    w.hustle_hours_today = 3.0
    assert w.hustle_daily_remaining == 5.0
    assert w.can_hustle is True
    w.is_hustle_exhausted = True
    assert w.can_hustle is False


# ── OPH computation ───────────────────────────────────────────────────────────

def test_effective_oph_baseline_is_base():
    assert _w(base_oph=22.0).effective_oph("pick") == pytest.approx(22.0)


def test_effective_oph_hustle_bonus():
    w = _w(base_oph=20.0)
    w.hustle_mode = True
    assert w.effective_oph("pick") == pytest.approx(20.0 * HUSTLE_MODE_BONUS)


def test_effective_oph_hustle_exhaustion_penalty():
    w = _w(base_oph=20.0)
    w.is_hustle_exhausted = True
    assert w.effective_oph("pick") == pytest.approx(20.0 * HUSTLE_EXHAUSTION_MULTIPLIER)


def test_effective_oph_fatigue_and_soreness_stack_multiplicatively():
    w = _w(base_oph=20.0)
    w.is_fatigued = True
    w.has_soreness = True
    w.soreness_activated = True
    expected = 20.0 * FATIGUE_OPH_PENALTY * SORENESS_OPH_PENALTY
    assert w.effective_oph("pick") == pytest.approx(expected)


def test_effective_oph_defaults_to_current_task():
    w = _w(base_oph=15.0)
    w.current_task = "pack"
    w.ticks_on_task = 2  # steady state -> flow multiplier 1.0 (isolate the default behavior)
    assert w.effective_oph() == pytest.approx(15.0)
    # The no-arg form must equal the explicit current-task form.
    assert w.effective_oph() == pytest.approx(w.effective_oph("pack"))


def test_flow_ramp_dip_then_bonus():
    from clark.env.worker import TASK_FLOW_RAMP
    w = _w(base_oph=20.0)
    w.current_task = "pick"
    # Just switched in: setup-cost dip.
    w.ticks_on_task = 0
    assert w.effective_oph("pick") == pytest.approx(20.0 * TASK_FLOW_RAMP[0])
    # Settled: recovers to baseline.
    w.ticks_on_task = 2
    assert w.effective_oph("pick") == pytest.approx(20.0 * 1.0)
    # In flow (well past the ramp): slight bonus, clamped to the last entry.
    w.ticks_on_task = 50
    assert w.effective_oph("pick") == pytest.approx(20.0 * TASK_FLOW_RAMP[-1])
    assert TASK_FLOW_RAMP[-1] > 1.0  # bonus exists


def test_flow_only_applies_to_current_task():
    w = _w(base_oph=20.0)
    w.current_task = "pick"
    w.ticks_on_task = 0  # dip on pick
    # Querying a DIFFERENT task gets no flow penalty/bonus (you aren't on it).
    assert w.effective_oph("pack") == pytest.approx(20.0)
    assert w.effective_oph("pick") < 20.0  # current task carries the dip


def test_flow_exempts_time_based_tasks():
    w = _w(base_oph=20.0)
    for t in ("management", "idle", "cycle_count"):
        w.current_task = t
        w.ticks_on_task = 0  # would be a dip if it applied
        assert w.effective_oph(t) == pytest.approx(20.0), f"{t} should be flow-exempt"


def test_bad_headspace_uses_per_task_then_default():
    w = _w(base_oph=10.0)
    w.health_debuff = "bad_headspace"
    w.bad_headspace_effects = {"pick": 0.5, "default": 0.8}
    assert w.effective_oph("pick") == pytest.approx(10.0 * 0.5)
    assert w.effective_oph("pack") == pytest.approx(10.0 * 0.8)


def test_bad_headspace_empty_effects_is_neutral():
    w = _w(base_oph=10.0)
    w.health_debuff = "bad_headspace"
    w.bad_headspace_effects = {}
    assert w.effective_oph("pick") == pytest.approx(10.0)


# ── Fatigue / soreness state transitions ──────────────────────────────────────

def test_check_fatigue_triggers_at_pack_threshold():
    w = _w()
    w.orders_packed = PACK_FATIGUE_THRESHOLD - 1
    w.check_fatigue()
    assert w.is_fatigued is False
    w.orders_packed = PACK_FATIGUE_THRESHOLD
    w.check_fatigue()
    assert w.is_fatigued is True


def test_check_fatigue_triggers_at_pick_threshold():
    w = _w()
    w.orders_picked = PICK_FATIGUE_THRESHOLD
    w.check_fatigue()
    assert w.is_fatigued is True


def test_check_soreness_requires_flag_and_threshold():
    w = _w()
    w.non_side_project_hours = SORENESS_HOUR_THRESHOLD + 1
    w.check_soreness()  # has_soreness False → no activation
    assert w.soreness_activated is False
    w.has_soreness = True
    w.non_side_project_hours = SORENESS_HOUR_THRESHOLD - 0.1
    w.check_soreness()
    assert w.soreness_activated is False
    w.non_side_project_hours = SORENESS_HOUR_THRESHOLD
    w.check_soreness()
    assert w.soreness_activated is True


# ── Eligibility ───────────────────────────────────────────────────────────────

def test_eligible_for_empty_set_means_all():
    w = _w(task_eligibility_set=set())
    assert w.eligible_for("pick") is True
    assert w.eligible_for("anything") is True


def test_eligible_for_restricted_set():
    w = _w(task_eligibility_set={"pick", "pack"})
    assert w.eligible_for("pick") is True
    assert w.eligible_for("restock") is False


def test_can_do_task_absent_blocks_everything():
    w = _w()
    w.is_absent = True
    assert w.can_do_task("pick") is False
    assert w.can_do_task("idle") is False


def test_can_do_task_pack_only_restriction():
    w = _w()
    w.is_pack_only = True
    assert w.can_do_task("pack") is True
    assert w.can_do_task("pick") is False


def test_management_eligible_by_role():
    assert _w(role="manager").is_management_eligible(None) is True
    assert _w(role="assistant_manager").is_management_eligible(None) is True
    assert _w(role="warehouse").is_management_eligible(None) is False

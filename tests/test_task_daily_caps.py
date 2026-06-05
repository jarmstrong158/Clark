"""Per-task daily-hours caps + unmet-penalty grading.

A facility can give an optional task a `daily_hours` target. Two effects:
  1. AUTO-OFF: once the total hours spent on the task (summed across all
     workers today) reach the target, the action mask removes the task
     for the rest of the day — no further labor is wasted on it.
  2. UNMET PENALTY: if the target ISN'T met by end of day, the facility's
     chosen penalty applies to the grade — "none" / "letter" (−1) /
     "two_letters" (−2) / "fail" (force F).

These tests pin both behaviors plus the schema validation guards.
"""
from __future__ import annotations

import yaml
import pytest

from clark.config.schema import FacilityConfig
from clark.env.facility_env import FacilityEnv
from clark.agent.actions import get_action_mask

CFG_PATH = "clark/data/configs/example_small.yaml"


def _load() -> FacilityConfig:
    return FacilityConfig.from_yaml(CFG_PATH)


def _unstress(env) -> None:
    """Suppress every filler-mask stress gate so the daily-hours cap is
    the only thing that can mask the task (mirrors the filler-mask test's
    _zero_all_pressures)."""
    env.arrived_so_far = 0
    env.orders_in_queue = 0
    env.orders_picked_not_audited = 0
    env.orders_completed = env.episode.total_orders
    env.restock_level = 1.0
    rules = env.facility_config.rules
    env.current_hour = (rules.day_start_hour + env._eod_hour) / 2
    # Clear the minimum-dwell lock so the cap is the only thing constraining
    # task choice (these tests isolate cap behavior, not the dwell mask).
    for w in env.episode.workers:
        w.ticks_on_task = 999


def _clean_A_day(env, cfg) -> None:
    """Force an otherwise-spotless A-grade end-of-day state."""
    env.orders_completed = env.episode.total_orders
    env.orders_in_queue = 0
    env.orders_picked_not_audited = 0
    env.restock_remaining = 0.0          # all_restock satisfied
    env.ot_hours = 0.0                   # no OT demerit
    env.episode.workers[0].management_hours = (
        cfg.rules.management_daily_hours_required + 1.0
    )
    for w in env.episode.workers:
        w.task_hours_today["side_project"] = 0.0


def _reason_codes(env, cfg):
    ft = env.get_episode_summary(management_backlog=0.0)["footer"]
    return ft["grade"], {r["code"] for r in ft["grade_reasons"]}


def test_grade_reasons_empty_on_clean_A():
    cfg = _load()
    cfg.tasks.daily_hours = {"side_project": 2.0}
    cfg.tasks.unmet_penalty = {"side_project": "none"}
    env = FacilityEnv(cfg); env.reset(force_month=12, force_dow=4, force_volume=20)
    _clean_A_day(env, cfg)
    env.episode.workers[0].task_hours_today["side_project"] = 2.5  # cap met
    grade, codes = _reason_codes(env, cfg)
    assert grade == "A"
    assert codes == set(), f"a clean A day should have no grade_reasons, got {codes}"


def test_grade_reasons_incomplete_orders():
    cfg = _load()
    env = FacilityEnv(cfg); env.reset(force_month=12, force_dow=4, force_volume=20)
    _clean_A_day(env, cfg)
    # Leave 5 orders unshipped -> incomplete-orders hard fail.
    env.orders_completed = env.episode.total_orders - 5
    env.orders_in_queue = 5
    grade, codes = _reason_codes(env, cfg)
    assert grade == "F"
    assert "incomplete_orders" in codes


def test_grade_reasons_task_fail():
    cfg = _load()
    cfg.tasks.daily_hours = {"side_project": 2.0}
    cfg.tasks.unmet_penalty = {"side_project": "fail"}
    env = FacilityEnv(cfg); env.reset(force_month=12, force_dow=4, force_volume=20)
    _clean_A_day(env, cfg)  # side_project left at 0h -> unmet
    grade, codes = _reason_codes(env, cfg)
    assert grade == "F"
    assert "task_fail" in codes


def test_task_capped_off_when_hours_met():
    cfg = _load()
    cfg.tasks.daily_hours = {"side_project": 2.0}
    cfg.tasks.unmet_penalty = {"side_project": "none"}
    assert "side_project" in cfg.task_ids
    sp = cfg.task_ids.index("side_project")

    env = FacilityEnv(cfg)
    env.reset(force_month=12, force_dow=4, force_volume=20)
    _unstress(env)

    # Below the cap: the task is a valid choice for at least one worker.
    for w in env.episode.workers:
        w.task_hours_today.pop("side_project", None)
    m0 = get_action_mask(env)
    assert m0[:, sp].any(), "uncapped side_project should be available on an unstressed day"

    # Target met (2.5h >= 2.0): masked off for every worker.
    env.episode.workers[0].task_hours_today["side_project"] = 2.5
    m1 = get_action_mask(env)
    assert not m1[:, sp].any(), "side_project should be off once its 2.0h target is met"
    assert all(m1[w].any() for w in range(m1.shape[0])), "no worker may be left with an all-zero mask"


def test_cap_does_not_fire_below_target():
    cfg = _load()
    cfg.tasks.daily_hours = {"side_project": 2.0}
    sp = cfg.task_ids.index("side_project")
    env = FacilityEnv(cfg)
    env.reset(force_month=12, force_dow=4, force_volume=20)
    _unstress(env)
    env.episode.workers[0].task_hours_today["side_project"] = 1.0  # below 2.0
    m = get_action_mask(env)
    assert m[:, sp].any(), "below-target hours must NOT mask the task"


@pytest.mark.parametrize("penalty,expected", [
    ("none", "A"),
    ("letter", "B"),
    ("two_letters", "C"),
    ("fail", "F"),
])
def test_unmet_penalty_grades(penalty, expected):
    cfg = _load()
    cfg.tasks.daily_hours = {"side_project": 2.0}
    cfg.tasks.unmet_penalty = {"side_project": penalty}
    env = FacilityEnv(cfg)
    env.reset(force_month=12, force_dow=4, force_volume=20)
    _clean_A_day(env, cfg)  # side_project left at 0.0h -> target unmet
    grade = env.get_episode_summary(management_backlog=0.0)["footer"]["grade"]
    assert grade == expected, f"unmet penalty '{penalty}' should grade {expected}, got {grade}"


def test_met_target_skips_penalty_even_when_fail():
    cfg = _load()
    cfg.tasks.daily_hours = {"side_project": 2.0}
    cfg.tasks.unmet_penalty = {"side_project": "fail"}
    env = FacilityEnv(cfg)
    env.reset(force_month=12, force_dow=4, force_volume=20)
    _clean_A_day(env, cfg)
    env.episode.workers[0].task_hours_today["side_project"] = 2.5  # target MET
    grade = env.get_episode_summary(management_backlog=0.0)["footer"]["grade"]
    assert grade == "A", "meeting the target must avoid the penalty even when penalty=fail"


def test_schema_rejects_bad_penalty():
    d = yaml.safe_load(open(CFG_PATH))
    d["tasks"]["daily_hours"] = {"side_project": 2.0}
    d["tasks"]["unmet_penalty"] = {"side_project": "bogus"}
    with pytest.raises(ValueError):
        FacilityConfig._from_dict(d)


def test_schema_warns_capacity_and_dangling_penalty():
    cfg = _load()
    cfg.tasks.daily_hours = {"side_project": 9999.0}      # exceeds roster capacity
    cfg.tasks.unmet_penalty = {"receiving": "letter"}     # no matching daily_hours target
    errs, warns = cfg.validate()
    assert not errs
    assert any("exceeds total roster" in w for w in warns), "should warn on impossible target"
    assert any("no matching tasks.daily_hours" in w for w in warns), "should warn on dangling penalty"

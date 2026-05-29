"""Sunday operations: a facility can opt into working Sundays, mirroring
the existing Saturday support.

The year schedule (year_env) is the only place day-of-week enters the
simulation — the policy's observation has no day-of-week feature, so a
Sunday is just another (low-volume) work day to the model. These tests
pin the schedule inclusion/exclusion, the Monday-scaled volume, and the
schema validation guard.
"""
from __future__ import annotations

import random
from collections import Counter

import pytest

from clark.config.schema import FacilityConfig
from clark.env.year_env import YearEnv

CFG_PATH = "clark/data/configs/example_small.yaml"


def _schedule(cfg, seed=0):
    random.seed(seed)
    return YearEnv(cfg)._generate_year_schedule()


def _dow_counts(cfg, seed=0) -> Counter:
    return Counter(d["day_of_week"] for d in _schedule(cfg, seed))


def test_sunday_excluded_by_default():
    cfg = FacilityConfig.from_yaml(CFG_PATH)
    cfg.rules.work_saturday = False
    cfg.rules.work_sunday = False
    c = _dow_counts(cfg)
    assert c.get(6, 0) == 0, "Sunday (dow=6) must not appear when work_sunday is False"
    assert c.get(5, 0) == 0, "Saturday must not appear when work_saturday is False"
    assert set(c) == {0, 1, 2, 3, 4}, "only Mon-Fri should be scheduled"


def test_sunday_included_when_enabled():
    cfg = FacilityConfig.from_yaml(CFG_PATH)
    cfg.rules.work_sunday = True
    cfg.rules.sunday_volume_fraction = 0.4
    c = _dow_counts(cfg)
    assert c.get(6, 0) >= 50, f"a full year should schedule ~52 Sundays, got {c.get(6, 0)}"


def test_sunday_volume_scaled_from_monday():
    cfg = FacilityConfig.from_yaml(CFG_PATH)
    cfg.rules.work_sunday = True
    cfg.rules.sunday_volume_fraction = 0.4
    sched = _schedule(cfg)
    sun = [d["volume"] for d in sched if d["day_of_week"] == 6]
    mon = [d["volume"] for d in sched if d["day_of_week"] == 0]
    ratio = (sum(sun) / len(sun)) / (sum(mon) / len(mon))
    # Sunday volume is drawn from [frac*0.8, frac] of Monday -> mean ~0.36*frac-band.
    assert 0.2 <= ratio <= 0.5, f"Sunday/Monday volume ratio {ratio:.2f} should track ~0.4 fraction"


def test_saturday_and_sunday_coexist():
    cfg = FacilityConfig.from_yaml(CFG_PATH)
    cfg.rules.work_saturday = True
    cfg.rules.saturday_volume_fraction = 0.5
    cfg.rules.work_sunday = True
    cfg.rules.sunday_volume_fraction = 0.4
    c = _dow_counts(cfg)
    assert c.get(5, 0) >= 50 and c.get(6, 0) >= 50, "both weekend days should be scheduled"


def test_schema_validates_sunday_fraction():
    cfg = FacilityConfig.from_yaml(CFG_PATH)
    cfg.rules.work_sunday = True
    cfg.rules.sunday_volume_fraction = 2.0  # out of [0.1, 1.0]
    errs, _ = cfg.validate()
    assert any("sunday_volume_fraction" in e for e in errs), "bad Sunday fraction must be a config error"


def test_sunday_fields_round_trip_from_dict():
    import yaml
    raw = yaml.safe_load(open(CFG_PATH))
    raw.setdefault("business_rules", {})
    raw["business_rules"]["work_sunday"] = True
    raw["business_rules"]["sunday_volume_fraction"] = 0.3
    parsed = FacilityConfig._from_dict(raw)
    assert parsed.rules.work_sunday is True
    assert parsed.rules.sunday_volume_fraction == 0.3

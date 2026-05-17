"""End-of-day OT regression tests.

Two structural defects let near-miss days finalize as an instant,
unrecoverable F (binary `order_incomplete_threshold=0` grade) even
though end-of-day overtime existed precisely to rescue them:

  Bug B (synthetic_gen): `eod_hour` and `ot_hard_stop_hour` were
    sampled on independent clocks, so `ot_hard_stop <= eod` on ~45%
    of generated facilities (OT window <= 0) and a sub-tick window on
    ~5% more — ~half the training distribution could never use OT.

  Bug A (facility_env._check_eod): `orders_remaining > ot_trigger`
    abandoned days *exactly* `ot_trigger` short; with the synthetic
    constant ot_trigger=1 a 1-order miss hit `1 > 1 == False` and the
    day finalized with zero OT.

These tests pin:
  1. every synthetic facility has a usable OT runway, and
  2. a 1-order-short day at EOD enters OT instead of finalizing,
so this class of "OT structurally impossible / never fires" defect
can't silently ship again.
"""
from __future__ import annotations

import random

from clark.env.facility_env import FacilityEnv
from clark.training.synthetic_gen import generate_random_facility

MIN_OT_WINDOW = 0.5  # must match synthetic_gen.MIN_OT_WINDOW


def test_every_synthetic_facility_has_usable_ot_window():
    """Bug B regression: ot_hard_stop must sit a real window past eod
    for EVERY facility across all curriculum stages."""
    random.seed(1234)
    degenerate = []
    for _ in range(3000):
        cfg = generate_random_facility(stage=random.choice([1, 2, 3]))
        r = cfg.rules
        window = float(r.ot_hard_stop_hour) - float(r.eod_hour)
        if window < MIN_OT_WINDOW - 1e-9:
            degenerate.append((r.eod_hour, r.ot_hard_stop_hour, window))
    assert not degenerate, (
        f"{len(degenerate)} facilities have an OT window < {MIN_OT_WINDOW}h "
        f"(eod, ot_hard_stop, window). First few: {degenerate[:5]}. "
        f"ot_hard_stop_hour must be anchored >= eod_hour + MIN_OT_WINDOW."
    )


def test_one_order_short_enters_overtime():
    """Bug A regression: a day that is exactly `ot_trigger` (=1) orders
    short at EOD must ENTER overtime, not finalize with zero OT."""
    cfg = generate_random_facility(stage=1)
    env = FacilityEnv(cfg)
    env.reset()

    assert cfg.rules.ot_trigger_orders_remaining == 1

    # Park the clock just past regular EOD but well inside the OT
    # window (Bug B fix guarantees the window is >= MIN_OT_WINDOW).
    env.current_hour = env._eod_hour + 0.01
    env.orders_in_queue = 1
    env.orders_picked_not_audited = 0
    env.is_ot = False
    env.is_done = False
    env._morning_ot_hours = 0.0

    env._check_eod()

    assert env.is_ot is True, (
        "a 1-order-short day must enter OT (orders_remaining >= ot_trigger)"
    )
    assert env.is_done is False, (
        "must NOT finalize on the first tick past EOD while orders remain "
        "and the OT window is open"
    )


def test_fully_complete_day_does_not_spuriously_enter_ot():
    """Guard: 0 orders remaining must never trigger OT even if a config
    sets ot_trigger=0 (the `orders_remaining > 0` clause)."""
    cfg = generate_random_facility(stage=1)
    env = FacilityEnv(cfg)
    env.reset()

    env.current_hour = env._eod_hour + 0.01
    env.orders_in_queue = 0
    env.orders_picked_not_audited = 0
    env.is_ot = False
    env.is_done = False
    env._morning_ot_hours = 0.0

    env._check_eod()

    assert env.is_ot is False, "a fully complete day must not enter OT"
    assert env.is_done is True, "a complete day at EOD must finalize"

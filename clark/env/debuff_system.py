"""
Generalized debuff engine.

Rolls sleep, health, and individual debuffs for all workers at day start.
All logic is config-driven — no hardcoded worker names.

Sleep and health distributions are universal (same across all workers).
Individual debuffs and bad_headspace effects are read from WorkerConfig.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from clark.env.worker import WorkerState, make_worker_state

if TYPE_CHECKING:
    from clark.config.schema import FacilityConfig, WorkerConfig


# ─── Universal debuff tables (not user-configurable) ──────────────────────────

SLEEP_DEBUFFS = [
    {"name": "well_rested", "multiplier": 1.05, "probability": 0.10},
    {"name": "normal",      "multiplier": 1.00, "probability": 0.75},
    {"name": "bad_sleep",   "multiplier": 0.95, "probability": 0.15},
]

HEALTH_DEBUFFS = [
    {"name": "locked_in",     "multiplier": 1.20, "probability": 0.05},
    {"name": "normal",        "multiplier": 1.00, "probability": 0.69},
    {"name": "bad_headspace", "multiplier": None,  "probability": 0.08},  # per-worker task effects
    {"name": "sick_mild",     "multiplier": 0.85, "probability": 0.05},
    {"name": "very_sick",     "multiplier": 0.60, "probability": 0.01},
    {"name": "injured",       "multiplier": 0.50, "probability": 0.02},
    {"name": "buffer",        "multiplier": 1.00, "probability": 0.10},
]

# High-volume day call-off probability (applied uniformly across all workers)
CALL_OFF_PROBABILITY_HIGH_VOLUME = 0.02


def roll_day_start(
    workers: list[WorkerState],
    configs: list["WorkerConfig"],
    season: str,
    cooldown_tracker: dict,
    facility_config: "FacilityConfig",
    is_high_volume: bool = False,
) -> None:
    """
    Roll debuffs for all workers at the start of a day.
    Modifies workers in place.

    Args:
        workers: List of WorkerState objects (already constructed).
        configs: Matching list of WorkerConfig objects (same order/index as workers).
        season: Current season string ("winter", "spring", "summer", "fall").
        cooldown_tracker: Dict keyed by worker name — tracks cooldown days between debuffs.
        facility_config: FacilityConfig for max call-off limits.
        is_high_volume: True if today is a high-volume day.
    """
    config_by_id = {cfg.worker_id: cfg for cfg in configs}

    for w in workers:
        cfg = config_by_id[w.worker_id]

        # Roll sleep category
        sleep = _roll_category(SLEEP_DEBUFFS)
        w.sleep_debuff = sleep["name"]
        w.sleep_multiplier = sleep["multiplier"]

        # Roll health category
        health = _roll_category(HEALTH_DEBUFFS)
        w.health_debuff = health["name"]
        if health["name"] == "bad_headspace":
            w.health_multiplier = 1.0  # effects handled per-task via bad_headspace_effects dict
        elif health["name"] == "buffer":
            w.health_debuff = "normal"
            w.health_multiplier = 1.0
        else:
            w.health_multiplier = health["multiplier"]

        # Roll individual debuff
        if cfg.individual_debuff is not None:
            _roll_individual_debuff(w, cfg, season, cooldown_tracker)

    # Roll daily call-offs
    _roll_call_offs(workers, configs, facility_config, is_high_volume)


def _roll_category(debuff_list: list[dict]) -> dict:
    """Pick a category from a weighted probability list."""
    roll = random.random()
    cumulative = 0.0
    for debuff in debuff_list:
        cumulative += debuff["probability"]
        if roll < cumulative:
            return debuff
    return debuff_list[-1]


def _roll_call_offs(
    workers: list[WorkerState],
    configs: list["WorkerConfig"],
    facility_config: "FacilityConfig",
    is_high_volume: bool,
) -> None:
    """
    Roll call-offs for each worker.

    High-volume days: max_call_offs_high_volume, 2% chance per worker.
    Normal days: max_call_offs_per_day, per-worker probability from config.
    """
    max_call_offs = (
        facility_config.rules.max_call_offs_high_volume
        if is_high_volume
        else facility_config.rules.max_call_offs_per_day
    )
    call_offs = 0

    config_by_id = {cfg.worker_id: cfg for cfg in configs}

    # Shuffle order so bias doesn't favor early-ID workers
    indices = list(range(len(workers)))
    random.shuffle(indices)

    for i in indices:
        w = workers[i]
        if w.is_absent:
            call_offs += 1
            continue

        cfg = config_by_id[w.worker_id]
        prob = CALL_OFF_PROBABILITY_HIGH_VOLUME if is_high_volume else cfg.call_off_probability

        if prob > 0 and random.random() < prob:
            if call_offs < max_call_offs:
                w.is_absent = True
                w.individual_debuff = "call_off"
                w.individual_multiplier = 0.0
                w.individual_effect = "absent"
                call_offs += 1
            # else: denied — at capacity


def _roll_individual_debuff(
    w: WorkerState,
    cfg: "WorkerConfig",
    season: str,
    cooldown_tracker: dict,
) -> None:
    """
    Roll this worker's individual debuff based on their WorkerConfig.
    Modifies w in place.
    """
    from clark.config.schema import IndividualDebuff

    key = w.name
    cooldown_remaining = cooldown_tracker.get(key, 0)

    if cooldown_remaining > 0:
        cooldown_tracker[key] = cooldown_remaining - 1
        return

    debuff_cfg = cfg.individual_debuff

    # Determine probability — use season_weights if present (e.g. pack_only flare)
    if debuff_cfg.season_weights:
        prob = debuff_cfg.season_weights.get(season, debuff_cfg.probability)
    else:
        prob = debuff_cfg.probability

    if random.random() >= prob:
        return

    # Debuff fires
    w.individual_debuff = debuff_cfg.debuff_type

    if debuff_cfg.cooldown_days > 0:
        cooldown_tracker[key] = debuff_cfg.cooldown_days

    effect = debuff_cfg.effect

    if effect == "lose_hours":
        # Same as Jack's family_needs logic
        lo, hi = debuff_cfg.hours_lost_range
        w.hours_lost = random.uniform(lo, hi)
        w.individual_multiplier = 1.0
        w.individual_effect = "lose_hours"

    elif effect == "oph_penalty":
        # Same as Jack's stomach_issues logic
        w.individual_multiplier = debuff_cfg.oph_multiplier
        w.individual_effect = "oph_penalty"

    elif effect == "pack_only":
        # Same as Jack's Andrew flare logic
        w.is_pack_only = True
        w.individual_multiplier = 1.0
        w.individual_effect = "pack_only"

    elif effect == "absent":
        w.is_absent = True
        w.individual_multiplier = 0.0
        w.individual_effect = "absent"

    elif effect == "soreness":
        # Same as Jack's Guy soreness logic
        w.has_soreness = True
        w.individual_multiplier = 1.0
        w.individual_effect = "soreness"

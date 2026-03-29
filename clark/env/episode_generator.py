"""
Config-driven episode generator.

Replaces Jack's hardcoded episode_generator.py.
All parameters come from FacilityConfig — no hardcoded worker IDs or names.
"""
from __future__ import annotations

import random
import math
import calendar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from clark.env.worker import WorkerState, make_worker_state
from clark.env.debuff_system import roll_day_start

if TYPE_CHECKING:
    from clark.config.schema import FacilityConfig


# ─── Constants (not user-configurable) ────────────────────────────────────────

RESTOCK_BASE_HOURS = 2.5
RESTOCK_VOLUME_COEFF = 0.008     # max ~6.5h at 500 orders
RESTOCK_NOISE_RANGE = (-0.5, 0.5)

DELIBERATE_PROJECT_CHANCE = 0.25
DELIBERATE_PROJECT_SIZE_RANGE = (2.0, 6.0)  # worker-hours
FILLER_PROJECT_SIZE = float("inf")           # never depletes
FILLER_COMPLETION_THRESHOLD = 4.0            # hours to count as "complete"

MORNING_PICK_CARTS_MIN = 1
MORNING_PICK_CARTS_MAX = 2
PER_CART_MIN = 1
PER_CART_MAX = 6

# Order arrival timing constants (fixed — no config override needed)
DAY_START_HOUR = 9.0
MORNING_END_HOUR = 14.0
AFTERNOON_END_HOUR = 16.25
ORDER_ARRIVAL_END_HOUR = 16.83   # last sim step before 5 PM
ORDER_CUTOFF_HOUR = 17.0
STEP_DURATION = 1 / 6            # 10-minute intervals

HIGH_VOLUME_PERCENTILE = 0.75

ORDER_ARRIVAL_BUCKETS = [
    ("day_start",    0.40, 0.50),
    ("morning",      0.25, 0.30),
    ("afternoon",    0.10, 0.15),
    ("late_surge",   0.15, 0.20),
]

ORDER_ARRIVAL_BUCKETS_HIGH_VOLUME = [
    ("day_start",    0.45, 0.55),
    ("morning",      0.30, 0.38),
    ("afternoon",    0.08, 0.12),
    ("late_trickle", 0.03, 0.07),
]

SEASONS = ["winter", "spring", "summer", "fall"]

MONTH_TO_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring",  4: "spring", 5: "spring",
    6: "summer",  7: "summer", 8: "summer",
    9: "fall",   10: "fall",  11: "fall",
}

# Peak seasons for pre-sim cycle count credit
PEAK_SEASONS = {"spring", "summer"}
PEAK_EARLY_CYCLE_COUNT = 0.5   # hours credited to manager at sim start on peak-season days

# Pre-sim management credit for manager role (7:45-9:00 equivalent = 1.25h)
MANAGER_PRE_SIM_MANAGEMENT = 1.25

# Weekly day names for volume curve lookup
DOW_NAMES = {0: "monday", 1: "tuesday", 2: "wednesday", 3: "thursday", 4: "friday"}


# ─── EpisodeParams dataclass ──────────────────────────────────────────────────

@dataclass
class EpisodeParams:
    # Date info
    year: int
    month: int
    day: int
    day_of_week: int      # 0=Monday
    month_name: str
    season: str

    # Orders
    total_orders: int
    is_high_volume: bool
    arrival_schedule: dict  # {hour: count}

    # Restock
    restock_hours: float

    # Side projects
    has_deliberate_project: bool
    deliberate_project_size: float
    filler_available: bool

    # Workers (already rolled with debuffs)
    workers: list
    picker_id: int
    picker_needs_replacement: bool

    # Debuffs summary
    debuffs_fired: list

    # EOD / timing
    eod_hour: float


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_episode(
    facility_config: "FacilityConfig",
    volume: int = None,
    day_of_week: int = None,
    season: str = None,
    month: int = None,
    cooldown_tracker: dict = None,
) -> EpisodeParams:
    """
    Generate one day's episode parameters from a FacilityConfig.

    Args:
        facility_config: The loaded facility configuration.
        volume: Override total order count (used by YearEnv for pre-computed schedule).
        day_of_week: Override day of week (0=Monday).
        season: Override season string.
        month: Override month (1-12).
        cooldown_tracker: Mutable dict keyed by worker name for debuff cooldowns.
    """
    if cooldown_tracker is None:
        cooldown_tracker = {}

    # Date
    year = random.randint(2024, 2026)
    if month is None:
        month = random.randint(1, 12)
    month_name = calendar.month_name[month]
    resolved_season = season or MONTH_TO_SEASON[month]

    if day_of_week is None:
        day_of_week = random.randint(0, 4)
    day = _find_day_for_dow(year, month, day_of_week)

    # Order volume
    vol_range = _get_volume_range(facility_config, month_name)
    total_orders = volume if volume is not None else random.randint(vol_range[0], vol_range[1])
    is_high_vol = _is_high_volume_day(total_orders, vol_range, facility_config)

    # Arrival schedule
    arrival = _generate_arrival_schedule(total_orders, is_high_vol)

    # Restock hours
    noise = random.uniform(*RESTOCK_NOISE_RANGE)
    restock_hours = max(0.0, RESTOCK_BASE_HOURS + total_orders * RESTOCK_VOLUME_COEFF + noise)

    # Side projects
    has_deliberate = random.random() < DELIBERATE_PROJECT_CHANCE
    deliberate_size = 0.0
    if has_deliberate:
        deliberate_size = random.uniform(*DELIBERATE_PROJECT_SIZE_RANGE)

    # Build workers and roll debuffs
    workers = [make_worker_state(cfg) for cfg in facility_config.workers]
    roll_day_start(
        workers=workers,
        configs=facility_config.workers,
        season=resolved_season,
        cooldown_tracker=cooldown_tracker,
        facility_config=facility_config,
        is_high_volume=is_high_vol,
    )

    # Determine picker for today — round-robin among non-manager warehouse workers
    # sorted by day_of_week, or use the config-driven schedule approach
    picker_id, picker_needs_replacement = _assign_picker(
        workers, facility_config, day_of_week
    )

    # Pre-sim credits for managers (replaces Jonathan's 7:45-9:00 arrival)
    for w in workers:
        if w.role == "manager" and not w.is_absent:
            w.management_hours += MANAGER_PRE_SIM_MANAGEMENT

    # Peak season early cycle count credit for managers
    if resolved_season in PEAK_SEASONS:
        for w in workers:
            if w.role == "manager" and not w.is_absent:
                w.cycle_count_hours_today += PEAK_EARLY_CYCLE_COUNT

    # Collect debuff summary
    debuffs_fired = []
    for w in workers:
        if w.sleep_debuff != "normal" or w.health_debuff != "normal" or w.individual_debuff:
            debuffs_fired.append({
                "worker": w.name,
                "sleep": w.sleep_debuff,
                "health": w.health_debuff,
                "individual": w.individual_debuff,
            })

    # EOD hour: use the latest shift end among all workers
    eod_hour = max(
        (cfg.shift_start + cfg.shift_hours for cfg in facility_config.workers),
        default=17.5,
    )

    return EpisodeParams(
        year=year,
        month=month,
        day=day,
        day_of_week=day_of_week,
        month_name=month_name,
        season=resolved_season,
        total_orders=total_orders,
        is_high_volume=is_high_vol,
        arrival_schedule=arrival,
        restock_hours=restock_hours,
        has_deliberate_project=has_deliberate,
        deliberate_project_size=deliberate_size,
        filler_available=True,
        workers=workers,
        picker_id=picker_id,
        picker_needs_replacement=picker_needs_replacement,
        debuffs_fired=debuffs_fired,
        eod_hour=eod_hour,
    )


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _get_volume_range(
    facility_config: "FacilityConfig",
    month_name: str,
) -> tuple[int, int]:
    """Look up volume range by month name, case-insensitive."""
    ranges = facility_config.volume.seasonal_ranges
    # Try exact match first, then lowercase
    if month_name in ranges:
        return tuple(ranges[month_name])
    key = month_name.lower()
    for k, v in ranges.items():
        if k.lower() == key:
            return tuple(v)
    raise KeyError(f"No seasonal_range for month '{month_name}' in facility config.")


def _is_high_volume_day(
    total_orders: int,
    vol_range: tuple[int, int],
    facility_config: "FacilityConfig",
) -> bool:
    lo, hi = vol_range
    threshold = lo + (hi - lo) * facility_config.rules.high_volume_percentile
    return total_orders >= threshold


def _assign_picker(
    workers: list[WorkerState],
    facility_config: "FacilityConfig",
    day_of_week: int,
) -> tuple[int, bool]:
    """
    Determine the picker for today.
    Uses round-robin among non-manager warehouse workers sorted by worker_id,
    cycling through days of week (same spirit as Jack's PICKER_SCHEDULE).
    Returns (picker_id, picker_needs_replacement).
    """
    # Eligible pickers: non-manager workers who can pick
    eligible_configs = [
        cfg for cfg in facility_config.workers
        if cfg.role not in ("manager", "assistant_manager")
        and cfg.eligible_for("pick")
    ]

    if not eligible_configs:
        # Fallback: any worker who can pick
        eligible_configs = [cfg for cfg in facility_config.workers if cfg.eligible_for("pick")]

    if not eligible_configs:
        return 0, False  # degenerate case

    # Round-robin by day_of_week index
    picker_cfg = eligible_configs[day_of_week % len(eligible_configs)]
    picker_id = picker_cfg.worker_id

    # Find the worker state for this picker
    picker_worker = next((w for w in workers if w.worker_id == picker_id), None)
    if picker_worker is None:
        return picker_id, False

    picker_needs_replacement = False

    # Check if picker has pack_only debuff (can't pick today)
    if picker_worker.is_pack_only:
        picker_needs_replacement = True
    else:
        picker_worker.is_picker = True

    return picker_id, picker_needs_replacement


def _find_day_for_dow(year: int, month: int, target_dow: int) -> int:
    """Find the first day in month/year matching the target day-of-week."""
    for day in range(1, 29):
        try:
            dow = calendar.weekday(year, month, day)
            if dow == target_dow:
                return day
        except ValueError:
            continue
    return 1


# ─── Order arrival schedule ───────────────────────────────────────────────────

def _generate_arrival_schedule(total_orders: int, is_high_volume: bool) -> dict[float, int]:
    """
    Generate {simulated_hour: num_orders_arriving} for each 10-min step.
    High-volume days front-load heavier.
    """
    buckets = ORDER_ARRIVAL_BUCKETS_HIGH_VOLUME if is_high_volume else ORDER_ARRIVAL_BUCKETS
    return _curved_distribution(total_orders, ORDER_ARRIVAL_END_HOUR, buckets)


def _curved_distribution(
    total_orders: int,
    arrival_end: float,
    buckets: list,
) -> dict[float, int]:
    fractions = []
    for _, lo, hi in buckets:
        fractions.append(random.uniform(lo, hi))

    total_frac = sum(fractions)
    fractions = [f / total_frac for f in fractions]

    bucket_boundaries = [
        (DAY_START_HOUR, DAY_START_HOUR),        # instant at day start
        (DAY_START_HOUR, MORNING_END_HOUR),       # morning flow
        (MORNING_END_HOUR, AFTERNOON_END_HOUR),   # afternoon slow
        (AFTERNOON_END_HOUR, arrival_end),        # late trickle
    ]

    schedule: dict[float, int] = {}
    remaining = total_orders

    for i, (frac, (start, end)) in enumerate(zip(fractions, bucket_boundaries)):
        if i == len(fractions) - 1:
            bucket_orders = max(0, remaining)
        else:
            bucket_orders = min(round(total_orders * frac), remaining)
            remaining -= bucket_orders

        if i == 0:
            schedule[DAY_START_HOUR] = bucket_orders
        else:
            steps = _get_steps_in_range(start, end)
            if not steps:
                schedule[DAY_START_HOUR] = schedule.get(DAY_START_HOUR, 0) + bucket_orders
                continue
            _distribute_to_steps(schedule, steps, bucket_orders)

    return schedule


def _get_steps_in_range(start: float, end: float) -> list[float]:
    steps = []
    t = start
    while t < end - 0.01:
        steps.append(round(t, 2))
        t += STEP_DURATION
    return steps


def _distribute_to_steps(schedule: dict, steps: list[float], num_orders: int):
    if not steps:
        return
    base_per_step = num_orders // len(steps)
    remainder = num_orders - base_per_step * len(steps)
    bonus_steps = set(random.sample(range(len(steps)), min(remainder, len(steps))))
    for i, step in enumerate(steps):
        count = base_per_step + (1 if i in bonus_steps else 0)
        schedule[step] = schedule.get(step, 0) + count

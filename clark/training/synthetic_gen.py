"""Random facility config generator for pre-training domain randomization.

Every call to generate_random_facility() returns a fully valid FacilityConfig
with randomized but plausible parameters. Used during pre-training to expose
the model to diverse facility shapes (N workers, M tasks, volume curves, etc.)
so the foundation model generalizes across real-world facility deployments.
"""
from __future__ import annotations

import random
from pathlib import Path

from clark.config.schema import (
    FacilityConfig,
    WorkerConfig,
    TasksConfig,
    VolumeConfig,
    BusinessRules,
    RewardOverrides,
    OrderComplexityConfig,
    OrderComplexityTier,
)
from clark.config.task_vocab import STANDARD_VOCAB, CORE_TASK_IDS
from clark.config.bounds import BOUNDS


# ── Constants ──────────────────────────────────────────────────────────────────

_MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

_WEEK_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]

# Tasks that can be optionally sampled (excludes core tasks pick/pack/idle)
_OPTIONAL_TASK_IDS = [
    t_id for t_id, td in STANDARD_VOCAB.items()
    if t_id not in CORE_TASK_IDS
]

_ROLE_POOL = ["manager", "assistant_manager", "lead", "warehouse"]

_TIMEZONES = [
    "America/New_York", "America/Chicago", "America/Denver",
    "America/Los_Angeles", "America/Phoenix",
]


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _sample_oph() -> float:
    """Sample worker OPH from N(16, 2.5), clamped within BOUNDS['base_oph']."""
    lo, hi = BOUNDS["base_oph"]
    return _clamp(random.gauss(16.0, 2.5), max(lo, 8.0), min(hi, 28.0))


def generate_random_facility() -> FacilityConfig:
    """
    Generate a random but valid FacilityConfig for pre-training.

    Returns a fully valid config — FacilityConfig.validate() will pass with
    no errors (warnings may appear for edge cases like no manager with
    management task).
    """
    n_workers = random.randint(*BOUNDS["n_workers"])
    workers = _generate_workers(n_workers)

    tasks = _generate_tasks()
    volume = _generate_volume()
    rules = _generate_rules()
    complexity = _sample_random_complexity()

    return FacilityConfig(
        name=f"synthetic_{random.randint(1000, 9999)}",
        timezone=random.choice(_TIMEZONES),
        workers=workers,
        tasks=tasks,
        volume=volume,
        rules=rules,
        reward_overrides=RewardOverrides(),
        order_complexity=complexity,
    )


# ── Sub-generators ─────────────────────────────────────────────────────────────

def _generate_workers(n_workers: int) -> list[WorkerConfig]:
    """
    Assign roles: exactly 1 manager, 0-2 assistant managers, remaining
    warehouse/lead. Shuffle so manager isn't always worker 0.
    """
    n_am = random.randint(0, min(2, n_workers - 1))
    n_remaining = n_workers - 1 - n_am

    # How many leads (0-20% of remaining, at least 0)
    n_leads = random.randint(0, max(0, n_remaining // 5))
    n_warehouse = n_remaining - n_leads

    roles = (
        ["manager"] * 1
        + ["assistant_manager"] * n_am
        + ["lead"] * n_leads
        + ["warehouse"] * n_warehouse
    )
    random.shuffle(roles)

    # Sample tasks that will be enabled (for eligibility assignment)
    # We'll use "all" task eligibility for simplicity — the action mask
    # handles per-role business constraints at runtime.
    shift_hours_base = _sample_shift_hours()
    workers: list[WorkerConfig] = []
    for i, role in enumerate(roles):
        oph = round(_sample_oph(), 1)
        # Small variation in shift hours per worker
        sh_lo, sh_hi = BOUNDS["shift_hours"]
        shift_h = round(_clamp(shift_hours_base + random.uniform(-0.5, 0.5), max(sh_lo, 6.0), sh_hi), 1)
        hustle_lo, hustle_hi = BOUNDS["hustle_daily_cap"]
        hustle_cap = round(random.uniform(max(hustle_lo, 2.0), min(hustle_hi, 9.0)), 1)
        co_lo, co_hi = BOUNDS["call_off_probability"]
        call_off_prob = round(random.uniform(max(co_lo, 0.01), min(co_hi, 0.05)), 3)

        # 30% chance of task_oph_overrides for 1-3 tasks
        task_oph_overrides = None
        if random.random() < 0.3:
            task_oph_overrides = _sample_task_oph_overrides()

        workers.append(WorkerConfig(
            worker_id=i,
            name=f"Worker_{i}",
            base_oph=oph,
            shift_hours=shift_h,
            shift_start=9.0,
            role=role,
            task_eligibility="all",
            call_off_probability=call_off_prob,
            max_ot_hours=round(random.uniform(0.5, 2.0), 1),
            hustle_daily_cap=hustle_cap,
            task_oph_overrides=task_oph_overrides,
        ))

    return workers


def _sample_shift_hours() -> float:
    """Sample facility-level shift length from [7.0, 10.0], within BOUNDS."""
    sh_lo, sh_hi = BOUNDS["shift_hours"]
    return round(random.uniform(max(sh_lo, 7.0), min(sh_hi, 10.0)), 1)


def _sample_task_oph_overrides() -> dict[str, float]:
    """Sample per-task OPH overrides for 1-3 randomly chosen tasks."""
    candidate_tasks = ["pick", "pack", "restock", "side_project", "receiving"]
    n_tasks = random.randint(1, min(3, len(candidate_tasks)))
    chosen = random.sample(candidate_tasks, n_tasks)
    lo, hi = BOUNDS["task_oph"]
    return {t: round(random.uniform(lo, min(hi, 50.0)), 1) for t in chosen}


def _generate_tasks() -> TasksConfig:
    """
    Always include pick/pack/idle (core). Sample n_tasks-3 additional tasks
    from standard vocab. Always include management to enable business-rule
    reward paths.
    """
    # n_tasks = how many total tasks (including core 3)
    max_optional = len(_OPTIONAL_TASK_IDS)
    # At minimum just core tasks; at most 15 or vocab size
    n_tasks = random.randint(3, min(15, 3 + max_optional))
    n_optional = n_tasks - 3  # number of optional tasks to add

    optional_selected = random.sample(_OPTIONAL_TASK_IDS, min(n_optional, max_optional))

    # Always include management if we're adding optional tasks — this ensures
    # the business-rule reward path is reachable during training.
    if n_optional > 0 and "management" not in optional_selected:
        # Replace the last selected optional with management
        optional_selected[-1] = "management"

    enabled = list(optional_selected)  # core tasks are auto-added by TasksConfig

    return TasksConfig(enabled=enabled, custom=[])


def _generate_volume() -> VolumeConfig:
    """
    Sample base volume range and apply seasonal and weekly curves.

    Seasonal pattern: spring/summer peaks (1.5-3x winter).
    Weekly pattern: Monday heaviest, Friday lightest.
    """
    # Base winter volume
    winter_lo = random.randint(30, 600)
    winter_hi = random.randint(winter_lo + 20, min(winter_lo + 400, 1000))

    # Spring/summer multiplier
    peak_mult = random.uniform(1.5, 3.0)
    spring_lo = int(winter_lo * 1.2)
    spring_hi = int(winter_hi * 1.5)
    summer_lo = int(winter_lo * peak_mult * 0.9)
    summer_hi = min(int(winter_hi * peak_mult), 2000)
    fall_lo = int(winter_lo * 1.1)
    fall_hi = int(winter_hi * 1.3)

    # Assign months to seasons
    seasonal_ranges: dict[str, tuple[int, int]] = {
        "january":   (winter_lo, winter_hi),
        "february":  (winter_lo, winter_hi),
        "march":     (spring_lo, spring_hi),
        "april":     (spring_lo, spring_hi),
        "may":       (spring_lo, spring_hi),
        "june":      (summer_lo, summer_hi),
        "july":      (summer_lo, summer_hi),
        "august":    (summer_lo, summer_hi),
        "september": (fall_lo, fall_hi),
        "october":   (fall_lo, fall_hi),
        "november":  (fall_lo, fall_hi),
        "december":  (winter_lo, winter_hi),
    }

    # Weekly curve: Monday heaviest, Friday lightest.
    # Sample from reasonable priors. Each value is (low_pct, high_pct) of
    # the month's spread that gets added to the monthly low.
    mon_lo = round(random.uniform(0.65, 0.80), 2)
    mon_hi = round(random.uniform(0.85, 1.00), 2)

    tue_lo = round(random.uniform(0.55, 0.70), 2)
    tue_hi = round(random.uniform(0.75, 0.90), 2)

    wed_lo = round(random.uniform(0.45, 0.60), 2)
    wed_hi = round(random.uniform(0.65, 0.80), 2)

    thu_lo = round(random.uniform(0.35, 0.55), 2)
    thu_hi = round(random.uniform(0.60, 0.75), 2)

    fri_lo = round(random.uniform(0.20, 0.40), 2)
    fri_hi = round(random.uniform(0.45, 0.65), 2)

    weekly_curve: dict[str, tuple[float, float]] = {
        "monday":    (mon_lo, mon_hi),
        "tuesday":   (tue_lo, tue_hi),
        "wednesday": (wed_lo, wed_hi),
        "thursday":  (thu_lo, thu_hi),
        "friday":    (fri_lo, fri_hi),
    }

    return VolumeConfig(seasonal_ranges=seasonal_ranges, weekly_curve=weekly_curve)


def _generate_rules() -> BusinessRules:
    """Sample business rules from reasonable priors, using BOUNDS for timing fields."""
    # Sample shift timing coherently (start before end, lunch between them)
    ds_lo, ds_hi = BOUNDS["day_start_hour"]
    eod_lo, eod_hi = BOUNDS["eod_hour"]
    lunch_lo, lunch_hi = BOUNDS["lunch_hour"]
    lunch_dur_lo, lunch_dur_hi = BOUNDS["lunch_duration"]

    day_start = round(random.uniform(ds_lo, ds_hi) * 2) / 2  # 30-min increments
    eod = round(random.uniform(max(eod_lo, day_start + 6), eod_hi) * 2) / 2
    lunch_h = round(random.uniform(day_start + 2, eod - 2) * 2) / 2
    lunch_h = max(lunch_lo, min(lunch_hi, lunch_h))
    lunch_dur = round(random.uniform(lunch_dur_lo, lunch_dur_hi) * 4) / 4  # 15-min increments

    # Carrier pickup: 50% of facilities have one
    carrier = None
    if random.random() < 0.5:
        cpu_lo, cpu_hi = BOUNDS["carrier_pickup_hour"]
        carrier = round(random.uniform(cpu_lo, min(cpu_hi, eod)) * 2) / 2

    # Morning pick: 60% of facilities use it
    morning_pick = random.random() < 0.6

    # OT
    otw_lo, otw_hi = BOUNDS["ot_wall_clock_max"]
    ot_stop_lo, ot_stop_hi = BOUNDS["ot_hard_stop_hour"]
    ot_wall = round(random.uniform(max(otw_lo, 0.5), min(otw_hi, 2.0)), 1)
    ot_stop = round(random.uniform(max(ot_stop_lo, 17.5), min(ot_stop_hi, 20.0)), 1)

    # Cycle counts
    cc_lo, cc_hi = BOUNDS["cycle_count_weekly_hours"]
    ccow_lo, ccow_hi = BOUNDS["cycle_count_max_overdue_weeks"]

    return BusinessRules(
        management_daily_hours_required=round(random.uniform(2.0, 6.0), 1),
        management_min_daily_hours=round(random.uniform(1.0, 2.0), 1),
        management_backlog_week_threshold=round(random.uniform(5.0, 15.0), 1),
        management_backlog_weekly_penalty=round(random.uniform(-75.0, -25.0), 1),
        ot_wall_clock_max=ot_wall,
        ot_hard_stop_hour=ot_stop,
        ot_trigger_orders_remaining=random.randint(5, 25),
        cycle_count_weekly_hours=round(random.uniform(max(cc_lo, 2.0), min(cc_hi, 5.0)), 1),
        cycle_count_max_overdue_weeks=random.randint(max(ccow_lo, 2), min(ccow_hi, 6)),
        min_staffing_floor=random.randint(1, 3),
        high_volume_day_orders=random.randint(200, 800),
        high_volume_percentile=round(random.uniform(0.65, 0.85), 2),
        target_daily_orders=None,
        order_incomplete_threshold=0,
        max_call_offs_per_day=random.randint(1, 4),
        max_call_offs_high_volume=random.randint(1, 2),
        # New timing fields
        day_start_hour=day_start,
        eod_hour=eod,
        order_cutoff_hour=round(random.uniform(eod - 2, eod) * 2) / 2,
        carrier_pickup_hour=carrier,
        lunch_hour=lunch_h,
        lunch_duration=lunch_dur,
        morning_pick_enabled=morning_pick,
        morning_pick_carts_min=random.randint(0, 2),
        morning_pick_carts_max=random.randint(1, 4),
        morning_pick_per_cart_min=random.randint(1, 5),
        morning_pick_per_cart_max=random.randint(4, 12),
    )


def _sample_random_complexity() -> OrderComplexityConfig:
    """
    Sample a random order complexity config.
    40% single tier, 40% two tiers, 20% three tiers.
    """
    roll = random.random()
    mult_lo, mult_hi = BOUNDS["order_complexity_oph_multiplier"]

    if roll < 0.40:
        # Single tier: all standard
        return OrderComplexityConfig(tiers=[
            OrderComplexityTier("standard", 1.0, 1.0)
        ])
    elif roll < 0.80:
        # Two tiers
        if random.random() < 0.5:
            # simple + standard
            w_simple = round(random.uniform(0.1, 0.5), 2)
            w_standard = round(1.0 - w_simple, 2)
            simple_mult = round(random.uniform(1.1, min(mult_hi, 2.0)), 2)
            return OrderComplexityConfig(tiers=[
                OrderComplexityTier("simple", w_simple, simple_mult),
                OrderComplexityTier("standard", w_standard, 1.0),
            ])
        else:
            # standard + complex
            w_standard = round(random.uniform(0.5, 0.9), 2)
            w_complex = round(1.0 - w_standard, 2)
            complex_mult = round(random.uniform(max(mult_lo, 0.3), 0.9), 2)
            return OrderComplexityConfig(tiers=[
                OrderComplexityTier("standard", w_standard, 1.0),
                OrderComplexityTier("complex", w_complex, complex_mult),
            ])
    else:
        # Three tiers: simple + standard + complex
        w_simple = round(random.uniform(0.1, 0.3), 2)
        w_complex = round(random.uniform(0.1, 0.3), 2)
        w_standard = round(1.0 - w_simple - w_complex, 2)
        # Clamp to ensure weights sum exactly
        w_standard = max(0.01, w_standard)
        total = w_simple + w_standard + w_complex
        w_simple = round(w_simple / total, 3)
        w_standard = round(w_standard / total, 3)
        w_complex = round(1.0 - w_simple - w_standard, 3)
        simple_mult = round(random.uniform(1.1, min(mult_hi, 2.0)), 2)
        complex_mult = round(random.uniform(max(mult_lo, 0.3), 0.9), 2)
        return OrderComplexityConfig(tiers=[
            OrderComplexityTier("simple", w_simple, simple_mult),
            OrderComplexityTier("standard", w_standard, 1.0),
            OrderComplexityTier("complex", w_complex, complex_mult),
        ])


# ── Convenience loader ─────────────────────────────────────────────────────────

def get_volt_config() -> FacilityConfig:
    """Load the example_1.yaml config (formerly volt_warehouse.yaml) from the standard data directory."""
    config_path = (
        Path(__file__).parent.parent / "data" / "configs" / "example_1.yaml"
    )
    return FacilityConfig.from_yaml(config_path)

"""Random facility config generator for pre-training domain randomization.

Every call to generate_random_facility() returns a fully valid FacilityConfig
with randomized but plausible parameters. Used during pre-training to expose
the model to diverse facility shapes (N workers, M tasks, volume curves, etc.)
so the foundation model generalizes across real-world facility deployments.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from clark.config.schema import (
    FacilityConfig,
    WorkerConfig,
    TasksConfig,
    VolumeConfig,
    BusinessRules,
    RewardOverrides,
    OrderComplexityConfig,
    OrderComplexityTier,
    PeakStaffingConfig,
    BreakConfig,
)
from clark.config.task_vocab import STANDARD_VOCAB, CORE_TASK_IDS
from clark.config.bounds import BOUNDS


# ── Curriculum Learning ────────────────────────────────────────────────────────

@dataclass
class CurriculumStage:
    stage_num: int
    min_workers: int
    max_workers: int
    max_tasks: int
    carryover_prob: float
    peak_staffing_prob: float
    max_complexity_tiers: int
    saturday_prob: float
    # Operational choices — intentionally NOT here.
    # Picker strategy, breaks, carrier deadline, restock availability, and
    # late order exceptions are always sampled at full probability regardless
    # of stage. The model must learn all operational strategies from day one.


STAGE_1 = CurriculumStage(
    stage_num=1,
    # min_workers=5 — N=3 and N=4 facilities have a structural near-zero win
    # ceiling: too few workers to cover pick+pack+restock+management AND
    # absorb a single absent or debuffed worker. After 1183 eps of training
    # those configs averaged win 0.6%/8.7% with R/W -437k/-282k while the
    # model was hitting 28% wins on N=8-10. They were teaching the model
    # "lose" and dragging the gradient toward defeatist OT-everywhere
    # behavior. Raised min from 3→5 so stage 1 is still small-facility
    # focused but every config is at least theoretically winnable.
    min_workers=5, max_workers=10,
    max_tasks=5,
    carryover_prob=0.0,
    peak_staffing_prob=0.0,
    max_complexity_tiers=1,
    saturday_prob=0.0,
)

STAGE_2 = CurriculumStage(
    stage_num=2,
    min_workers=5, max_workers=25,
    max_tasks=10,
    carryover_prob=0.30,
    peak_staffing_prob=0.30,
    max_complexity_tiers=2,
    saturday_prob=0.15,
)

STAGE_3 = CurriculumStage(
    stage_num=3,
    min_workers=5, max_workers=50,
    max_tasks=15,
    carryover_prob=0.40,
    peak_staffing_prob=0.50,
    max_complexity_tiers=3,
    saturday_prob=0.25,
)

CURRICULUM_STAGES: dict[int, CurriculumStage] = {1: STAGE_1, 2: STAGE_2, 3: STAGE_3}


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


def generate_random_facility(stage: int = 3) -> FacilityConfig:
    """
    Generate a random but valid FacilityConfig for pre-training.

    Args:
        stage: Curriculum stage (1, 2, or 3). Defaults to 3 (full complexity)
               so all existing callers that don't pass a stage continue to work.

    Returns a fully valid config — FacilityConfig.validate() will pass with
    no errors (warnings may appear for edge cases like no manager with
    management task).
    """
    cs = CURRICULUM_STAGES.get(stage, STAGE_3)

    n_workers = random.randint(cs.min_workers, cs.max_workers)
    workers = _generate_workers(n_workers, cs)
    avg_oph = sum(w.base_oph for w in workers) / max(1, len(workers))

    tasks = _generate_tasks(cs)
    volume = _generate_volume(n_workers, avg_oph)
    rules = _generate_rules(cs)
    complexity = _sample_random_complexity(cs)

    # Generate break schedule and peak staffing
    day_start = rules.day_start_hour
    eod = rules.eod_hour
    breaks = _generate_breaks(day_start, eod)
    peak_staffing = _generate_peak_staffing(cs)

    return FacilityConfig(
        name=f"synthetic_{random.randint(1000, 9999)}",
        timezone=random.choice(_TIMEZONES),
        workers=workers,
        tasks=tasks,
        volume=volume,
        rules=rules,
        reward_overrides=RewardOverrides(),
        order_complexity=complexity,
        breaks=breaks,
        peak_staffing=peak_staffing,
    )


# ── Sub-generators ─────────────────────────────────────────────────────────────

def _generate_workers(n_workers: int, cs: CurriculumStage) -> list[WorkerConfig]:
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

    # Facility shift start: most workers start together, some start late (split shift)
    ds_lo, ds_hi = BOUNDS["day_start_hour"]
    facility_start = round(random.uniform(ds_lo, ds_hi) * 2) / 2

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

        # 20% of warehouse workers get a staggered start (0.5–3h after facility start)
        if role == "warehouse" and random.random() < 0.2:
            shift_start = round((facility_start + random.uniform(0.5, 3.0)) * 2) / 2
        else:
            shift_start = facility_start

        # Pick OPH is always set — realistic range is 1.5x to 3.5x base packing rate
        # This reflects real warehouses: picking is faster because you're pulling, not packing
        lo_task, hi_task = BOUNDS["task_oph"]
        pick_oph = round(oph * random.uniform(1.5, 3.5), 1)
        pick_oph = min(pick_oph, hi_task)
        task_oph_overrides: dict = {"pick": pick_oph}
        # 30% chance of additional task overrides
        if random.random() < 0.3:
            extra = _sample_task_oph_overrides(exclude=["pick"])
            task_oph_overrides.update(extra)

        workers.append(WorkerConfig(
            worker_id=i,
            name=f"Worker_{i}",
            base_oph=oph,
            shift_hours=shift_h,
            shift_start=shift_start,
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


def _sample_task_oph_overrides(exclude: list[str] = []) -> dict[str, float]:
    """Sample per-task OPH overrides for 1-3 randomly chosen tasks."""
    candidate_tasks = [t for t in ["pick", "pack", "restock", "side_project", "receiving"]
                       if t not in exclude]
    if not candidate_tasks:
        return {}
    n_tasks = random.randint(1, min(3, len(candidate_tasks)))
    chosen = random.sample(candidate_tasks, n_tasks)
    lo, hi = BOUNDS["task_oph"]
    return {t: round(random.uniform(lo, min(hi, 50.0)), 1) for t in chosen}


def _generate_tasks(cs: CurriculumStage) -> TasksConfig:
    """
    Always include pick/pack/idle (core) plus restock and management. Sample
    any remaining optional tasks from standard vocab.

    `restock` is force-included because the env's restock drain/refill system
    runs unconditionally — a facility without a restock task is structurally
    unwinnable (pick speed decays to 5% once the restock level drains).

    `management` is force-included so the business-rule reward path is
    reachable during training.
    """
    max_optional = len(_OPTIONAL_TASK_IDS)
    # Bump the minimum to 5 (core 3 + restock + management) so both
    # force-included tasks always fit inside n_tasks.
    min_tasks = min(5, 3 + max_optional)
    n_tasks = random.randint(min_tasks, min(cs.max_tasks, 3 + max_optional))
    n_tasks = max(n_tasks, min_tasks)
    n_optional = n_tasks - 3  # number of optional tasks to add

    # Start with the two mandatory optional tasks, then fill the rest randomly.
    mandatory = [t for t in ("restock", "management") if t in _OPTIONAL_TASK_IDS]
    remaining_pool = [t for t in _OPTIONAL_TASK_IDS if t not in mandatory]

    n_extra = max(0, n_optional - len(mandatory))
    extras = random.sample(remaining_pool, min(n_extra, len(remaining_pool)))

    optional_selected = mandatory + extras

    enabled = list(optional_selected)  # core tasks are auto-added by TasksConfig

    return TasksConfig(enabled=enabled, custom=[])


def _generate_volume(n_workers: int, avg_oph: float) -> VolumeConfig:
    """
    Sample base volume range and apply seasonal and weekly curves.

    Seasonal pattern: spring/summer peaks (1.5-3x winter).
    Weekly pattern: Monday heaviest, Friday lightest.

    Volume scales with N. Previously volume was sampled independently of
    worker count — a 5-worker facility could randomly draw 600 winter /
    1800 summer daily orders, the same as a 25-worker facility, which is
    physically unwinnable. That made small-N look "broken" when the real
    problem was unfair config generation.

    Per-worker daily order capacity is roughly base_oph × shift_hours × ~0.4
    (split between pick + pack + restock + management; pickers don't pack
    while picking, etc.). With base_oph≈16 and 9-hour shifts that's
    ~58 orders/worker/day realistic max. We sample winter baseline from
    [30, 60] orders/worker/day (comfortable to needs-OT range), and the
    seasonal multiplier (up to 3x) stacks on top — so a peak-summer day
    can be physically demanding but never impossible.
    """
    # Per-worker per-day capacity: base_oph × shift_hours × 0.4 (split across
    # pick + pack + restock + management; you can't pick while packing).
    capacity_per_worker = max(20.0, avg_oph * 9.0 * 0.4)
    # Sample the SUMMER peak multiplier first so we can cap winter to keep
    # peak-summer achievable. Without this cap winter * peak_mult could
    # easily exceed 3× capacity, which is physically unwinnable and trains
    # the model to "panic OT" / "give up" on impossible days. That bad
    # pattern then bleeds into days that ARE achievable.
    peak_mult = random.uniform(1.5, 3.0)
    # Target: even the SUMMER HIGH stays at or under ~110% capacity.
    # winter_high * peak_mult <= n_workers * capacity * 1.10
    # → winter_high_max = n_workers * capacity * 1.10 / peak_mult
    winter_hi_cap = n_workers * capacity_per_worker * 1.10 / peak_mult
    fraction_hi = random.uniform(0.85, 1.00)
    fraction_lo = random.uniform(0.50, 0.70)
    base_hi = int(n_workers * capacity_per_worker * fraction_hi)
    base_lo = int(n_workers * capacity_per_worker * fraction_lo)
    # Apply the peak-aware cap so we don't generate impossible summers.
    base_hi = min(base_hi, int(winter_hi_cap))
    base_lo = min(base_lo, max(30, base_hi - 50))
    winter_lo = max(30, base_lo)
    winter_hi = max(winter_lo + 20, min(base_hi, 2000))

    # Spring/summer multiplier — peak_mult was sampled above so we could
    # cap winter accordingly. Don't re-roll it here.
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


def _generate_rules(cs: CurriculumStage) -> BusinessRules:
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

    # Order carryover — stage-gated
    order_carryover_enabled = random.random() < cs.carryover_prob
    order_carryover_max_days = random.randint(1, 5) if order_carryover_enabled else 3
    order_carryover_max_backlog = random.choice([0, 0, 0, random.randint(50, 500)])

    # Picker assignment
    picker_assignment = random.choices(
        ["round_robin", "fixed", "agent"],
        weights=[0.5, 0.3, 0.2]
    )[0]

    # Saturday — stage-gated
    # NOTE: carrier_pickup, breaks, picker_assignment, restock_availability,
    # late_order_exception_rate — these are NOT gated. Always sampled at full
    # probability regardless of stage.
    work_saturday = random.random() < cs.saturday_prob
    saturday_vol = round(random.uniform(0.2, 0.6), 2) if work_saturday else 0.4

    # Late order exceptions (30% of facilities)
    late_exception_rate = round(random.uniform(0.02, 0.15), 3) if random.random() < 0.3 else 0.0

    # Restock availability (40% of facilities have a delivery window)
    restock_avail = 0.0
    if random.random() < 0.4:
        avail_lo, avail_hi = BOUNDS["restock_availability_hour"]
        restock_avail = round(random.uniform(max(avail_lo, 7.0), min(avail_hi, 13.0)) * 2) / 2

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
        # Timing fields
        day_start_hour=day_start,
        eod_hour=eod,
        order_cutoff_hour=round(random.uniform(eod - 2, eod) * 2) / 2,
        carrier_pickup_hour=carrier,
        lunch_hour=lunch_h,
        lunch_duration=lunch_dur,
        morning_pick_enabled=morning_pick,
        morning_pick_carts_min=(_carts_min := random.randint(0, 2)),
        morning_pick_carts_max=random.randint(_carts_min, 4),
        morning_pick_per_cart_min=(_per_min := random.randint(1, 5)),
        morning_pick_per_cart_max=random.randint(_per_min, 12),
        # New parameters
        order_carryover_enabled=order_carryover_enabled,
        order_carryover_max_days=order_carryover_max_days,
        order_carryover_max_backlog=order_carryover_max_backlog,
        picker_assignment=picker_assignment,
        work_saturday=work_saturday,
        saturday_volume_fraction=saturday_vol,
        late_order_exception_rate=late_exception_rate,
        restock_availability_hour=restock_avail,
    )


def _generate_breaks(day_start: float, eod: float) -> list:
    """
    Generate a break schedule relative to the shift.
    30% no breaks (use lunch_hour fallback), 40% single lunch, 30% full schedule.
    """
    roll = random.random()
    if roll < 0.30:
        return []  # use lunch_hour fallback
    # Calculate sensible break times relative to shift
    shift_len = eod - day_start
    lunch_time = day_start + shift_len * random.uniform(0.35, 0.55)
    lunch_time = round(lunch_time * 2) / 2
    lunch_dur = round(random.choice([0.25, 0.5, 0.75]), 2)
    if roll < 0.70:
        # Just lunch
        return [BreakConfig(hour=lunch_time, duration=lunch_dur,
                            staggered=random.random() < 0.3)]
    else:
        # Full schedule: morning, lunch, afternoon
        morning_time = round((day_start + shift_len * random.uniform(0.15, 0.30)) * 2) / 2
        afternoon_time = round((day_start + shift_len * random.uniform(0.65, 0.80)) * 2) / 2
        return [
            BreakConfig(hour=morning_time, duration=0.25, staggered=False),
            BreakConfig(hour=lunch_time, duration=lunch_dur, staggered=random.random() < 0.4),
            BreakConfig(hour=afternoon_time, duration=0.25, staggered=False),
        ]


def _generate_peak_staffing(cs: CurriculumStage) -> Optional[PeakStaffingConfig]:
    """Generate peak staffing config — probability controlled by curriculum stage."""
    if random.random() > cs.peak_staffing_prob:
        return None
    # Peak months: typically spring and/or summer
    all_months = ["march", "april", "may", "june", "july", "august", "september"]
    n_peak_months = random.randint(2, 4)
    peak_months = random.sample(all_months, n_peak_months)
    t_lo, t_hi = BOUNDS["temp_oph"]
    return PeakStaffingConfig(
        months=peak_months,
        extra_workers=random.randint(1, min(5, BOUNDS["peak_extra_workers"][1])),
        temp_oph_range=(round(random.uniform(max(t_lo, 8.0), min(t_hi, 15.0)), 1),
                        round(random.uniform(max(t_lo, 12.0), min(t_hi, 20.0)), 1)),
        temp_call_off_probability=round(random.uniform(0.05, 0.12), 3),
        temp_shift_hours=round(random.uniform(6.0, 8.5), 1),
    )


def _sample_random_complexity(cs: CurriculumStage) -> OrderComplexityConfig:
    """
    Sample a random order complexity config.
    Stage constrains maximum number of tiers.
    Stage 1: always single tier.
    Stage 2: single or two tiers (50/50).
    Stage 3: 40% single, 40% two tiers, 20% three tiers.
    """
    mult_lo, mult_hi = BOUNDS["order_complexity_oph_multiplier"]

    if cs.max_complexity_tiers == 1:
        return OrderComplexityConfig(tiers=[OrderComplexityTier("standard", 1.0, 1.0)])

    roll = random.random()

    if cs.max_complexity_tiers == 2:
        # Only single or two tiers
        if roll < 0.50:
            return OrderComplexityConfig(tiers=[OrderComplexityTier("standard", 1.0, 1.0)])
        # else fall through to two-tier logic
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

    # max_complexity_tiers == 3: full distribution
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

def get_example1_config() -> FacilityConfig:
    """Load the example_1.yaml config from the standard data directory."""
    config_path = (
        Path(__file__).parent.parent / "data" / "configs" / "example_1.yaml"
    )
    return FacilityConfig.from_yaml(config_path)

"""
FacilityConfig dataclass and YAML validation.

Load a facility config from YAML:
    config = FacilityConfig.from_yaml("my_warehouse.yaml")

Validate and get human-readable warnings:
    errors, warnings = config.validate()

The config drives everything: environment dimensions (N workers, M tasks),
volume curves, business rules, and reward signal overrides.
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from clark.config.task_vocab import STANDARD_VOCAB, CORE_TASK_IDS, TaskDef
from clark.config.bounds import validate_value


# ─── Sub-dataclasses ─────────────────────────────────────────────────────────

@dataclass
class BreakConfig:
    hour: float           # 24h clock (e.g. 10.0 = 10:00 AM)
    duration: float       # hours (0.25 = 15 min, 0.5 = 30 min)
    staggered: bool = False  # if True, workers rotate in two groups instead of all at once


@dataclass
class PeakStaffingConfig:
    months: list[str]                    # e.g. ["april", "may", "june"]
    extra_workers: int                   # how many temps to add during peak
    temp_oph_range: tuple[float, float]  # (min, max) OPH for temps — lower than regulars
    temp_call_off_probability: float = 0.08
    temp_shift_hours: float = 8.0


@dataclass
class OrderComplexityTier:
    name: str             # "simple", "standard", "complex"
    weight: float         # fraction of orders (all tiers must sum to ~1.0)
    oph_multiplier: float # speed multiplier vs base OPH


@dataclass
class OrderComplexityConfig:
    tiers: list[OrderComplexityTier]

    @classmethod
    def default(cls) -> "OrderComplexityConfig":
        """Standard single-tier config (all orders equal) — backwards compatible."""
        return cls(tiers=[OrderComplexityTier("standard", 1.0, 1.0)])

    def validate(self) -> list[str]:
        errors = []
        total = sum(t.weight for t in self.tiers)
        if abs(total - 1.0) > 0.01:
            errors.append(f"Order complexity weights sum to {total:.3f}, must sum to 1.0")
        return errors


@dataclass
class IndividualDebuff:
    debuff_type: str                    # e.g. "family_needs", "soreness", "stomach_issues"
    probability: float                  # daily roll probability (0-1)
    cooldown_days: int                  # min days between occurrences
    effect: str                         # "lose_hours" | "oph_penalty" | "pack_only" | "soreness"
    hours_lost_range: Optional[tuple[float, float]] = None  # for "lose_hours"
    oph_multiplier: Optional[float] = None                  # for "oph_penalty"
    season_weights: Optional[dict[str, float]] = None       # override probability by season


@dataclass
class WorkerConfig:
    worker_id: int
    name: str
    base_oph: float                     # base orders-per-hour (packing rate)
    shift_hours: float                  # regular shift length in hours
    shift_start: float                  # shift start in 24h decimal (e.g. 9.0 = 9:00 AM)
    role: str                           # "manager" | "assistant_manager" | "lead" | "warehouse"
    task_eligibility: str | list[str]   # "all" or list of task_ids
    call_off_probability: float = 0.035
    max_ot_hours: Optional[float] = None  # personal OT cap; None = facility default
    hustle_daily_cap: Optional[float] = None  # None = role-based default
    bad_headspace_effects: Optional[dict[str, float]] = None  # task_id → multiplier
    individual_debuff: Optional[IndividualDebuff] = None
    task_oph_overrides: Optional[dict[str, float]] = None
    # Per-task OPH rates. Overrides the base_oph × task_multiplier calculation.
    # Example: {"pick": 42.5, "restock": 8.0}
    # Tasks not listed use base_oph × standard multiplier.
    # Management, idle, cycle_count are time-based — OPH overrides ignored for them.

    def eligible_for(self, task_id: str) -> bool:
        if self.task_eligibility == "all":
            return True
        return task_id in self.task_eligibility


@dataclass
class CustomTaskConfig:
    task_id: str
    display_name: str
    output_type: str            # "orders_per_hour" | "units_per_hour" | "hours" | "none"
    hustle_eligible: bool
    eligible_roles: list[str]   # roles that can perform this task
    reward_weight: float = 1.0  # scales step reward relative to orders


@dataclass
class TasksConfig:
    enabled: list[str]                      # standard task IDs to activate
    custom: list[CustomTaskConfig] = field(default_factory=list)

    def all_task_ids(self) -> list[str]:
        """Ordered list of all active task IDs (standard + custom)."""
        ids = list(CORE_TASK_IDS)  # pick, pack, idle always first
        for t_id in self.enabled:
            if t_id not in ids:
                ids.append(t_id)
        for ct in self.custom:
            if ct.task_id not in ids:
                ids.append(ct.task_id)
        return ids


@dataclass
class VolumeConfig:
    seasonal_ranges: dict[str, tuple[int, int]]  # month_name → (low, high)
    weekly_curve: dict[str, tuple[float, float]]  # day_name → (low_pct, high_pct)


@dataclass
class BusinessRules:
    management_daily_hours_required: float = 4.0
    management_min_daily_hours: float = 1.5       # minimum before carryover penalty
    management_backlog_week_threshold: float = 10.0
    management_backlog_weekly_penalty: float = -50.0
    ot_wall_clock_max: float = 1.0
    ot_hard_stop_hour: float = 18.5
    ot_trigger_orders_remaining: int = 10
    cycle_count_weekly_hours: float = 3.0
    cycle_count_max_overdue_weeks: int = 4
    min_staffing_floor: int = 2
    high_volume_day_orders: int = 400
    high_volume_percentile: float = 0.75
    target_daily_orders: Optional[int] = None
    order_incomplete_threshold: int = 0
    max_call_offs_per_day: int = 2
    max_call_offs_high_volume: int = 1

    # Shift timing
    day_start_hour: float = 9.0
    eod_hour: float = 17.5
    order_cutoff_hour: float = 17.0
    carrier_pickup_hour: Optional[float] = None  # None = no hard deadline

    # Breaks
    lunch_hour: float = 13.0
    lunch_duration: float = 0.5   # 0.0 = no lunch break

    # Morning pick round
    morning_pick_enabled: bool = True
    morning_pick_carts_min: int = 1
    morning_pick_carts_max: int = 2
    morning_pick_per_cart_min: int = 1
    morning_pick_per_cart_max: int = 6

    # Equipment constraints
    pack_stations: Optional[int] = None    # None = unlimited
    carts_available: Optional[int] = None  # None = unlimited

    # Order carryover — user controls all three knobs
    order_carryover_enabled: bool = False
    order_carryover_max_days: int = 3        # orders expire after this many days unshipped
    order_carryover_max_backlog: int = 0     # 0 = unlimited backlog; >0 = hard cap before penalty

    # Picker assignment strategy
    picker_assignment: str = "round_robin"   # "round_robin" | "fixed" | "agent"
    fixed_picker_id: Optional[int] = None    # used when picker_assignment == "fixed"

    # Weekend operations
    work_saturday: bool = False
    saturday_volume_fraction: float = 0.4   # fraction of Monday's volume applied to Saturday

    # Late order exceptions (post-cutoff orders that still must ship)
    late_order_exception_rate: float = 0.0  # 0.05 = 5% of post-cutoff orders still count

    # Inbound freight timing
    restock_availability_hour: float = 0.0  # 0.0 = restock materials always available
                                             # e.g. 10.0 = truck arrives at 10 AM, can't restock before then


@dataclass
class RewardOverrides:
    """Optional per-facility reward signal overrides. Unset = Clark defaults."""
    per_order_shipped: Optional[float] = None
    all_orders_complete_bonus: Optional[float] = None
    per_order_incomplete: Optional[float] = None
    per_ot_hour: Optional[float] = None
    per_restock_completed: Optional[float] = None
    per_cycle_count_hour: Optional[float] = None
    cycle_count_week_complete: Optional[float] = None
    cycle_count_week_missed: Optional[float] = None
    cycle_count_critical_overdue: Optional[float] = None
    management_duty_met: Optional[float] = None
    management_duty_missed: Optional[float] = None

    def apply(self, defaults: dict) -> dict:
        """Merge overrides into a defaults dict, returning merged copy."""
        merged = dict(defaults)
        for key, val in vars(self).items():
            if val is not None and key in merged:
                merged[key] = val
        return merged


# ─── Default reward signals ──────────────────────────────────────────────────

DEFAULT_REWARDS: dict[str, float] = {
    "per_order_shipped":              1.0,
    "all_orders_complete_bonus":     50.0,
    "per_order_incomplete":         -10.0,
    "per_ot_hour":                   -0.5,
    "ot_incomplete_flat":           -25.0,
    "ot_per_order_incomplete":      -10.0,
    "per_restock_completed":          1.0,
    "all_restock_bonus":             25.0,
    "per_restock_bleed":             -2.0,
    "restock_pick_interruption":    -10.0,
    "restock_level_low":             -2.0,
    "restock_level_empty":           -5.0,
    "per_filler_unit":                0.1,
    "filler_completion_bonus":        5.0,
    "per_deliberate_unit":            0.1,
    "deliberate_completion_bonus":    8.0,
    "side_project_during_crunch":    -2.0,
    "per_productive_hour":            0.3,
    "per_management_hour":            0.5,
    "per_idle_hour":                 -0.5,
    "packers_starved":               -1.0,
    "picked_backlog":                -2.0,
    "management_duty_met":           30.0,
    "management_duty_missed":       -50.0,
    "per_cycle_count_hour":           0.5,
    "cycle_count_week_complete":     20.0,
    "cycle_count_week_missed":       -15.0,
    "cycle_count_critical_overdue": -150.0,
}


# ─── Top-level FacilityConfig ─────────────────────────────────────────────────

@dataclass
class FacilityConfig:
    name: str
    timezone: str
    workers: list[WorkerConfig]
    tasks: TasksConfig
    volume: VolumeConfig
    rules: BusinessRules
    reward_overrides: RewardOverrides = field(default_factory=RewardOverrides)
    order_complexity: OrderComplexityConfig = field(default_factory=OrderComplexityConfig.default)
    breaks: list[BreakConfig] = field(default_factory=list)
    # If empty, falls back to BusinessRules.lunch_hour + lunch_duration (backwards compat).
    # If populated, these replace the single lunch. Listed in chronological order.
    peak_staffing: Optional[PeakStaffingConfig] = None

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    @property
    def task_ids(self) -> list[str]:
        return self.tasks.all_task_ids()

    @property
    def num_tasks(self) -> int:
        return len(self.task_ids)

    @property
    def rewards(self) -> dict[str, float]:
        return self.reward_overrides.apply(DEFAULT_REWARDS)

    def manager_worker_ids(self) -> list[int]:
        return [w.worker_id for w in self.workers if w.role == "manager"]

    def assistant_manager_ids(self) -> list[int]:
        return [w.worker_id for w in self.workers if w.role == "assistant_manager"]

    def cycle_count_eligible_ids(self) -> list[int]:
        """Workers eligible for cycle count: managers and assistant managers."""
        return [w.worker_id for w in self.workers
                if w.role in ("manager", "assistant_manager")
                and w.eligible_for("cycle_count")]

    def management_eligible_ids(self) -> list[int]:
        """Workers whose hours count toward daily management quota."""
        return [w.worker_id for w in self.workers
                if w.role in ("manager", "assistant_manager")]

    # ── Loader ────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FacilityConfig":
        """Load and parse a facility config YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> "FacilityConfig":
        facility = d.get("facility", {})
        workers = [cls._parse_worker(w) for w in d.get("workers", [])]
        tasks = cls._parse_tasks(d.get("tasks", {}))
        volume = cls._parse_volume(d.get("volume", {}))
        rules = cls._parse_rules(d.get("business_rules", {}))
        reward_overrides = cls._parse_reward_overrides(d.get("rewards", {}))
        order_complexity = cls._parse_order_complexity(d.get("order_complexity"))
        breaks = cls._parse_breaks(d.get("breaks", []))
        peak_staffing = cls._parse_peak_staffing(d.get("peak_staffing"))

        return cls(
            name=facility.get("name", "Unnamed Facility"),
            timezone=facility.get("timezone", "UTC"),
            workers=workers,
            tasks=tasks,
            volume=volume,
            rules=rules,
            reward_overrides=reward_overrides,
            order_complexity=order_complexity,
            breaks=breaks,
            peak_staffing=peak_staffing,
        )

    @staticmethod
    def _parse_worker(d: dict) -> WorkerConfig:
        debuff_raw = d.get("individual_debuff")
        debuff = None
        if debuff_raw:
            hrs = debuff_raw.get("hours_lost_range")
            season_w = debuff_raw.get("season_weights")
            debuff = IndividualDebuff(
                debuff_type=debuff_raw["type"],
                probability=debuff_raw.get("probability", 0.0),
                cooldown_days=debuff_raw.get("cooldown_days", 0),
                effect=debuff_raw["effect"],
                hours_lost_range=tuple(hrs) if hrs else None,
                oph_multiplier=debuff_raw.get("oph_multiplier"),
                season_weights=season_w,
            )
        raw_oph_overrides = d.get("task_oph_overrides")
        task_oph_overrides = None
        if raw_oph_overrides:
            task_oph_overrides = {k: float(v) for k, v in raw_oph_overrides.items()}

        return WorkerConfig(
            worker_id=d["id"],
            name=d["name"],
            base_oph=float(d["base_oph"]),
            shift_hours=float(d["shift_hours"]),
            shift_start=float(d.get("shift_start", 9.0)),
            role=d.get("role", "warehouse"),
            task_eligibility=d.get("task_eligibility", "all"),
            call_off_probability=float(d.get("call_off_probability", 0.035)),
            max_ot_hours=d.get("max_ot_hours"),
            hustle_daily_cap=d.get("hustle_daily_cap"),
            bad_headspace_effects=d.get("bad_headspace_effects"),
            individual_debuff=debuff,
            task_oph_overrides=task_oph_overrides,
        )

    @staticmethod
    def _parse_tasks(d: dict) -> TasksConfig:
        enabled = list(d.get("enabled", ["pick", "pack"]))
        # Always include core tasks
        for core_id in CORE_TASK_IDS:
            if core_id not in enabled:
                enabled.append(core_id)
        customs = []
        for ct in d.get("custom", []):
            customs.append(CustomTaskConfig(
                task_id=ct["id"],
                display_name=ct["display_name"],
                output_type=ct.get("output_type", "hours"),
                hustle_eligible=ct.get("hustle_eligible", False),
                eligible_roles=ct.get("eligible_roles", ["warehouse"]),
                reward_weight=float(ct.get("reward_weight", 1.0)),
            ))
        return TasksConfig(enabled=enabled, custom=customs)

    @staticmethod
    def _parse_volume(d: dict) -> VolumeConfig:
        raw_seasonal = d.get("seasonal_ranges", {})
        seasonal = {k: tuple(v) for k, v in raw_seasonal.items()}
        raw_weekly = d.get("weekly_curve", {})
        weekly = {k: tuple(v) for k, v in raw_weekly.items()}
        return VolumeConfig(seasonal_ranges=seasonal, weekly_curve=weekly)

    @staticmethod
    def _parse_rules(d: dict) -> BusinessRules:
        return BusinessRules(
            management_daily_hours_required=float(d.get("management_daily_hours_required", 4.0)),
            management_min_daily_hours=float(d.get("management_min_daily_hours", 1.5)),
            management_backlog_week_threshold=float(d.get("management_backlog_week_threshold", 10.0)),
            management_backlog_weekly_penalty=float(d.get("management_backlog_weekly_penalty", -50.0)),
            ot_wall_clock_max=float(d.get("ot_wall_clock_max", 1.0)),
            ot_hard_stop_hour=float(d.get("ot_hard_stop_hour", 18.5)),
            ot_trigger_orders_remaining=int(d.get("ot_trigger_orders_remaining", 10)),
            cycle_count_weekly_hours=float(d.get("cycle_count_weekly_hours", 3.0)),
            cycle_count_max_overdue_weeks=int(d.get("cycle_count_max_overdue_weeks", 4)),
            min_staffing_floor=int(d.get("min_staffing_floor", 2)),
            high_volume_day_orders=int(d.get("high_volume_day_orders", 400)),
            high_volume_percentile=float(d.get("high_volume_percentile", 0.75)),
            target_daily_orders=d.get("target_daily_orders"),
            order_incomplete_threshold=int(d.get("order_incomplete_threshold", 0)),
            max_call_offs_per_day=int(d.get("max_call_offs_per_day", 2)),
            max_call_offs_high_volume=int(d.get("max_call_offs_high_volume", 1)),
            # Shift timing
            day_start_hour=float(d.get("day_start_hour", 9.0)),
            eod_hour=float(d.get("eod_hour", 17.5)),
            order_cutoff_hour=float(d.get("order_cutoff_hour", 17.0)),
            carrier_pickup_hour=(float(d["carrier_pickup_hour"]) if d.get("carrier_pickup_hour") is not None else None),
            # Breaks
            lunch_hour=float(d.get("lunch_hour", 13.0)),
            lunch_duration=float(d.get("lunch_duration", 0.5)),
            # Morning pick round
            morning_pick_enabled=bool(d.get("morning_pick_enabled", True)),
            morning_pick_carts_min=int(d.get("morning_pick_carts_min", 1)),
            morning_pick_carts_max=int(d.get("morning_pick_carts_max", 2)),
            morning_pick_per_cart_min=int(d.get("morning_pick_per_cart_min", 1)),
            morning_pick_per_cart_max=int(d.get("morning_pick_per_cart_max", 6)),
            # Equipment
            pack_stations=(int(d["pack_stations"]) if d.get("pack_stations") is not None else None),
            carts_available=(int(d["carts_available"]) if d.get("carts_available") is not None else None),
            # Order carryover
            order_carryover_enabled=bool(d.get("order_carryover_enabled", False)),
            order_carryover_max_days=int(d.get("order_carryover_max_days", 3)),
            order_carryover_max_backlog=int(d.get("order_carryover_max_backlog", 0)),
            # Picker assignment
            picker_assignment=str(d.get("picker_assignment", "round_robin")),
            fixed_picker_id=(int(d["fixed_picker_id"]) if d.get("fixed_picker_id") is not None else None),
            # Weekend operations
            work_saturday=bool(d.get("work_saturday", False)),
            saturday_volume_fraction=float(d.get("saturday_volume_fraction", 0.4)),
            # Late order exceptions
            late_order_exception_rate=float(d.get("late_order_exception_rate", 0.0)),
            # Inbound freight timing
            restock_availability_hour=float(d.get("restock_availability_hour", 0.0)),
        )

    @staticmethod
    def _parse_order_complexity(d: Optional[dict]) -> OrderComplexityConfig:
        if d is None:
            return OrderComplexityConfig.default()
        tiers = []
        for t in d.get("tiers", []):
            tiers.append(OrderComplexityTier(
                name=str(t["name"]),
                weight=float(t["weight"]),
                oph_multiplier=float(t["oph_multiplier"]),
            ))
        if not tiers:
            return OrderComplexityConfig.default()
        return OrderComplexityConfig(tiers=tiers)

    @staticmethod
    def _parse_breaks(raw: list) -> list[BreakConfig]:
        if not raw:
            return []
        result = []
        for b in raw:
            result.append(BreakConfig(
                hour=float(b["hour"]),
                duration=float(b["duration"]),
                staggered=bool(b.get("staggered", False)),
            ))
        return result

    @staticmethod
    def _parse_peak_staffing(raw: Optional[dict]) -> Optional[PeakStaffingConfig]:
        if not raw:
            return None
        oph_range = raw.get("temp_oph_range", [8.0, 15.0])
        return PeakStaffingConfig(
            months=list(raw["months"]),
            extra_workers=int(raw["extra_workers"]),
            temp_oph_range=(float(oph_range[0]), float(oph_range[1])),
            temp_call_off_probability=float(raw.get("temp_call_off_probability", 0.08)),
            temp_shift_hours=float(raw.get("temp_shift_hours", 8.0)),
        )

    @staticmethod
    def _parse_reward_overrides(d: dict) -> RewardOverrides:
        if not d:
            return RewardOverrides()
        return RewardOverrides(**{k: v for k, v in d.items() if hasattr(RewardOverrides, k)})

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> tuple[list[str], list[str]]:
        """
        Return (errors, warnings).
        Errors are blocking (config is unusable).
        Warnings are advisory (config will work but may behave unexpectedly).
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Worker count
        if self.num_workers < 2:
            errors.append(f"Need at least 2 workers, got {self.num_workers}.")
        if self.num_workers > 50:
            warnings.append(f"{self.num_workers} workers is above recommended max (50). Training will be slower.")

        # Task count
        if self.num_tasks < 2:
            errors.append("Need at least 2 active tasks (pick + pack minimum).")

        # At least one manager if management task is enabled
        if "management" in self.task_ids and not self.manager_worker_ids():
            warnings.append("'management' task is enabled but no worker has role='manager'. "
                            "Management duty will never be fulfilled.")

        # Worker IDs must be unique and 0-indexed
        ids = [w.worker_id for w in self.workers]
        if len(ids) != len(set(ids)):
            errors.append("Worker IDs must be unique.")
        if sorted(ids) != list(range(self.num_workers)):
            errors.append(f"Worker IDs must be 0-indexed contiguous integers (0..{self.num_workers-1}).")

        # OPH sanity + bounds check
        for w in self.workers:
            if w.base_oph < 1.0 or w.base_oph > 50.0:
                warnings.append(f"Worker '{w.name}' base_oph={w.base_oph} is outside typical range [1, 50].")
            ok, msg = validate_value("base_oph", w.base_oph)
            if not ok:
                errors.append(f"base_oph (Worker '{w.name}'): {w.base_oph} exceeds max of 40.0 — see clark_limits.yaml")
            ok, msg = validate_value("shift_hours", w.shift_hours)
            if not ok:
                errors.append(f"shift_hours (Worker '{w.name}'): {w.shift_hours} — {msg} — see clark_limits.yaml")
            # task_oph_overrides bounds
            if w.task_oph_overrides:
                for task_id, oph_val in w.task_oph_overrides.items():
                    ok, msg = validate_value("task_oph", oph_val)
                    if not ok:
                        errors.append(
                            f"task_oph_overrides['{task_id}'] (Worker '{w.name}'): {oph_val} — {msg} — see clark_limits.yaml"
                        )

        # n_workers bounds check
        ok, msg = validate_value("n_workers", self.num_workers)
        if not ok:
            errors.append(f"n_workers: {msg} — see clark_limits.yaml")

        # Shift timing validations
        rules = self.rules
        if rules.eod_hour <= rules.day_start_hour + 4:
            errors.append(
                f"eod_hour ({rules.eod_hour}) must be > day_start_hour + 4 ({rules.day_start_hour + 4}). "
                "Minimum 4-hour shift required."
            )
        if not (rules.day_start_hour < rules.lunch_hour < rules.eod_hour):
            errors.append(
                f"lunch_hour ({rules.lunch_hour}) must be between day_start_hour ({rules.day_start_hour}) "
                f"and eod_hour ({rules.eod_hour})."
            )
        if rules.carrier_pickup_hour is not None:
            if not (rules.day_start_hour <= rules.carrier_pickup_hour <= rules.eod_hour):
                errors.append(
                    f"carrier_pickup_hour ({rules.carrier_pickup_hour}) must be between "
                    f"day_start_hour ({rules.day_start_hour}) and eod_hour ({rules.eod_hour})."
                )

        # Bounds checks for key rules fields
        for field_name, value in [
            ("day_start_hour", rules.day_start_hour),
            ("eod_hour", rules.eod_hour),
            ("ot_wall_clock_max", rules.ot_wall_clock_max),
            ("ot_hard_stop_hour", rules.ot_hard_stop_hour),
            ("management_daily_hours_required", rules.management_daily_hours_required),
            ("cycle_count_weekly_hours", rules.cycle_count_weekly_hours),
            ("cycle_count_max_overdue_weeks", rules.cycle_count_max_overdue_weeks),
        ]:
            ok, msg = validate_value(field_name, value)
            if not ok:
                errors.append(f"{msg} — see clark_limits.yaml")

        # Order complexity validation
        complexity_errors = self.order_complexity.validate()
        errors.extend(complexity_errors)

        # Seasonal ranges present for all 12 months
        expected_months = {
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        }
        missing_months = expected_months - {k.lower() for k in self.volume.seasonal_ranges}
        if missing_months:
            errors.append(f"Missing seasonal_ranges for months: {sorted(missing_months)}")

        # Weekly curve present for all 5 weekdays
        expected_days = {"monday", "tuesday", "wednesday", "thursday", "friday"}
        missing_days = expected_days - {k.lower() for k in self.volume.weekly_curve}
        if missing_days:
            errors.append(f"Missing weekly_curve for days: {sorted(missing_days)}")

        # Unknown task IDs in enabled list
        known_ids = set(STANDARD_VOCAB.keys()) | {ct.task_id for ct in self.tasks.custom}
        for t_id in self.tasks.enabled:
            if t_id not in known_ids:
                errors.append(f"Unknown task_id in tasks.enabled: '{t_id}'. "
                               "Use standard vocab IDs or define it in tasks.custom.")

        # Task eligibility references valid task IDs
        for w in self.workers:
            if isinstance(w.task_eligibility, list):
                active_ids = set(self.task_ids)
                for t_id in w.task_eligibility:
                    if t_id not in active_ids:
                        warnings.append(f"Worker '{w.name}' has task_eligibility entry '{t_id}' "
                                        f"which is not in the active task list.")

        # Picker assignment validation
        rules = self.rules
        if rules.picker_assignment == "fixed" and rules.fixed_picker_id is not None:
            valid_ids = [w.worker_id for w in self.workers]
            if rules.fixed_picker_id not in valid_ids:
                errors.append(
                    f"fixed_picker_id ({rules.fixed_picker_id}) is not a valid worker ID. "
                    f"Valid IDs: {sorted(valid_ids)}"
                )

        # Order carryover validation
        if rules.order_carryover_enabled and rules.order_carryover_max_days < 1:
            errors.append(
                f"order_carryover_max_days ({rules.order_carryover_max_days}) must be >= 1 "
                "when order_carryover_enabled is True."
            )

        # Saturday operations validation
        if rules.work_saturday:
            if not (0.1 <= rules.saturday_volume_fraction <= 1.0):
                errors.append(
                    f"saturday_volume_fraction ({rules.saturday_volume_fraction}) must be between "
                    "0.1 and 1.0 when work_saturday is True."
                )

        return errors, warnings

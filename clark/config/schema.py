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


# ─── Sub-dataclasses ─────────────────────────────────────────────────────────

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

        return cls(
            name=facility.get("name", "Unnamed Facility"),
            timezone=facility.get("timezone", "UTC"),
            workers=workers,
            tasks=tasks,
            volume=volume,
            rules=rules,
            reward_overrides=reward_overrides,
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

        # OPH sanity
        for w in self.workers:
            if w.base_oph < 1.0 or w.base_oph > 50.0:
                warnings.append(f"Worker '{w.name}' base_oph={w.base_oph} is outside typical range [1, 50].")

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

        return errors, warnings

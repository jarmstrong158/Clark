"""
WorkerState dataclass — generalized from Jack's WorkerState.
No hardcoded names, IDs, or roles. Driven entirely by WorkerConfig.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from clark.config.schema import WorkerConfig, FacilityConfig

# ─── Constants (not user-configurable) ────────────────────────────────────────
HUSTLE_MODE_BONUS = 1.12            # +12% OPH while hustling
HUSTLE_EXHAUSTION_MULTIPLIER = 0.85  # -15% OPH when exhausted from overuse
FATIGUE_OPH_PENALTY = 0.85          # 15% drop when fatigued
PACK_FATIGUE_THRESHOLD = 110        # orders packed before fatigue kicks in
PICK_FATIGUE_THRESHOLD = 230        # orders picked before fatigue kicks in
SORENESS_OPH_PENALTY = 0.50         # 50% drop when soreness activates
SORENESS_HOUR_THRESHOLD = 4.0       # hours of non-side-project work before soreness activates

# ── Task flow / ramp ──────────────────────────────────────────────────────────
# Humans don't context-switch for free, and they don't switch every 10 minutes.
# Each entry is the OPH multiplier for a throughput task as a function of how
# many consecutive 10-min ticks the worker has been on it. Index 0 = the tick
# they just switched in (setup/walk-over cost), climbing to a slight "flow"
# bonus once they've settled. A worker who thrashes tasks every tick is stuck
# at the 0.85 floor; one who holds a task for ~40 min runs at 1.05. This makes
# sustained allocation genuinely more productive, so the policy is rewarded
# (via throughput → completion) for keeping people on tasks — without a new
# reward term to game. Time-based tasks (management/idle/cycle_count) have no
# physical ramp and are exempt.
TASK_FLOW_RAMP = (0.85, 0.92, 1.0, 1.03, 1.05)  # 10-min ticks; clamps to last
FLOW_EXEMPT_TASKS = frozenset({"management", "idle", "cycle_count", "off"})

DAY_START_HOUR = 9.0


@dataclass
class WorkerState:
    worker_id: int
    name: str
    base_oph: float
    shift_hours: float
    role: str

    # task_eligibility set: populated from WorkerConfig at creation
    task_eligibility_set: set = field(default_factory=set)
    # bad_headspace_effects: {task_id: multiplier, "default": multiplier}
    bad_headspace_effects: dict = field(default_factory=dict)
    # hustle_daily_cap: set from config (None → role-based default 8.0)
    _hustle_daily_cap: Optional[float] = field(default=None, repr=False)

    # Current state
    current_task: str = "idle"
    # Consecutive 10-min ticks the worker has been on `current_task`. Drives
    # the flow/ramp multiplier (see effective_oph): a fresh switch starts at
    # 0 (setup-time dip), sustained work ramps up to a slight flow bonus.
    ticks_on_task: int = 0
    hours_worked: float = 0.0
    shift_start: float = DAY_START_HOUR
    is_picker: bool = False

    # Debuffs
    sleep_debuff: str = "normal"
    sleep_multiplier: float = 1.0
    health_debuff: str = "normal"
    health_multiplier: float = 1.0
    individual_debuff: Optional[str] = None
    individual_multiplier: float = 1.0
    individual_effect: Optional[str] = None

    # Individual debuff specifics
    hours_lost: float = 0.0       # for lose_hours effect
    is_absent: bool = False        # for absent effect (call-off / NCNS)
    is_pack_only: bool = False     # for pack_only effect (flare)
    has_soreness: bool = False     # for soreness effect

    # Fatigue tracking
    orders_packed: int = 0
    orders_picked: int = 0
    is_fatigued: bool = False

    # Fractional work accumulator — avoids int truncation waste on short steps
    work_carry: float = 0.0

    # Soreness tracking
    non_side_project_hours: float = 0.0
    soreness_activated: bool = False

    # Management tracking
    management_hours: float = 0.0

    # Cycle count tracking
    cycle_count_hours_today: float = 0.0

    # Per-task time tracking (diagnostic — populated by facility_env each tick).
    # Lets us spot patterns like "model assigned this worker to pack but no
    # picked orders ever showed up so they had 5 hours of effective idle time."
    # Reset at day start. Keys are task_ids; values are hours.
    task_hours_today: dict = field(default_factory=dict)

    # Hustle mode — agent-controlled per-step toggle
    hustle_mode: bool = False
    hustle_hours_today: float = 0.0    # accumulated hustle hours this day
    weekly_hustle_hours: float = 0.0   # accumulated hustle hours this week (set from YearEnv)
    is_hustle_exhausted: bool = False  # true if weekly threshold crossed — lasts until Monday

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def hustle_daily_cap(self) -> float:
        if self._hustle_daily_cap is not None:
            return self._hustle_daily_cap
        return 8.0  # role-based default

    @property
    def hustle_weekly_threshold(self) -> float:
        """Weekly threshold = 2× daily cap (matches Jack's formula)."""
        return self.hustle_daily_cap * 2

    @property
    def hustle_daily_remaining(self) -> float:
        return max(0.0, self.hustle_daily_cap - self.hustle_hours_today)

    @property
    def can_hustle(self) -> bool:
        return (not self.is_hustle_exhausted and
                self.hustle_hours_today < self.hustle_daily_cap)

    @property
    def shift_end(self) -> float:
        return self.shift_start + self.shift_hours

    @property
    def hours_remaining(self) -> float:
        return max(0.0, self.shift_hours - self.hours_worked - self.hours_lost)

    # ── OPH computation ────────────────────────────────────────────────────────

    def effective_oph(self, task: str = None) -> float:
        if task is None:
            task = self.current_task

        base = self.base_oph
        sleep_mod = self.sleep_multiplier
        health_mod = self._health_modifier_for_task(task)
        individual_mod = self.individual_multiplier
        fatigue_mod = self._fatigue_modifier()

        hustle_mod = HUSTLE_MODE_BONUS if self.hustle_mode else 1.0
        exhaustion_mod = HUSTLE_EXHAUSTION_MULTIPLIER if self.is_hustle_exhausted else 1.0
        flow_mod = self.flow_multiplier(task)

        return base * sleep_mod * health_mod * individual_mod * fatigue_mod * hustle_mod * exhaustion_mod * flow_mod

    def flow_multiplier(self, task: str = None) -> float:
        """Setup-cost-then-flow ramp for sustained work on a throughput task.

        Only applies to the task the worker is *currently* sustaining — a
        query for any other task gets 1.0 (you can't be 'in flow' on a task
        you aren't doing; the dip is paid when you actually switch to it and
        ticks_on_task resets). Time-based tasks are exempt.
        """
        if task is None:
            task = self.current_task
        if task != self.current_task or task in FLOW_EXEMPT_TASKS:
            return 1.0
        i = self.ticks_on_task
        return TASK_FLOW_RAMP[i] if i < len(TASK_FLOW_RAMP) else TASK_FLOW_RAMP[-1]

    def _health_modifier_for_task(self, task: str) -> float:
        if self.health_debuff != "bad_headspace":
            return self.health_multiplier

        effects = self.bad_headspace_effects
        if not effects:
            return 1.0
        if task in effects:
            return effects[task]
        return effects.get("default", 1.0)

    def _fatigue_modifier(self) -> float:
        mod = 1.0
        if self.is_fatigued:
            mod *= FATIGUE_OPH_PENALTY
        if self.has_soreness and self.soreness_activated:
            mod *= SORENESS_OPH_PENALTY
        return mod

    # ── State checks ───────────────────────────────────────────────────────────

    def check_fatigue(self):
        if not self.is_fatigued:
            if self.orders_packed >= PACK_FATIGUE_THRESHOLD or \
               self.orders_picked >= PICK_FATIGUE_THRESHOLD:
                self.is_fatigued = True

    def check_soreness(self):
        if self.has_soreness and not self.soreness_activated:
            if self.non_side_project_hours >= SORENESS_HOUR_THRESHOLD:
                self.soreness_activated = True

    def can_do_task(self, task: str) -> bool:
        if self.is_absent:
            return False
        if self.is_pack_only and task != "pack":
            return False
        return True

    # ── Eligibility helpers ────────────────────────────────────────────────────

    def eligible_for(self, task_id: str) -> bool:
        """True if this worker can perform the given task_id."""
        if not self.task_eligibility_set:
            return True  # "all" — no restrictions
        return task_id in self.task_eligibility_set

    def is_management_eligible(self, config: "FacilityConfig") -> bool:
        """True if this worker's role qualifies them for management duty."""
        return self.role in ("manager", "assistant_manager")


def make_worker_state(worker_cfg: "WorkerConfig") -> WorkerState:
    """
    Construct a fresh WorkerState from a WorkerConfig.
    Called at the start of each episode.
    """
    from clark.config.schema import WorkerConfig  # local import to avoid circular

    # Resolve task_eligibility_set
    if worker_cfg.task_eligibility == "all":
        eligibility_set: set = set()  # empty set = "all" (checked in eligible_for)
    else:
        eligibility_set = set(worker_cfg.task_eligibility)

    return WorkerState(
        worker_id=worker_cfg.worker_id,
        name=worker_cfg.name,
        base_oph=worker_cfg.base_oph,
        shift_hours=worker_cfg.shift_hours,
        role=worker_cfg.role,
        task_eligibility_set=eligibility_set,
        bad_headspace_effects=dict(worker_cfg.bad_headspace_effects) if worker_cfg.bad_headspace_effects else {},
        _hustle_daily_cap=worker_cfg.hustle_daily_cap,
        shift_start=worker_cfg.shift_start,
    )

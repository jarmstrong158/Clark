"""
YearEnv — config-driven year wrapper.

One episode = one full year (~260 work days).
Manages carryover between days: restock level, management backlog, cycle counts, hustle.
Delegates intra-day simulation to FacilityEnv.
"""
from __future__ import annotations

import random
import calendar
import numpy as np
from typing import Optional, TYPE_CHECKING

from clark.env.facility_env import FacilityEnv, RESTOCK_STARTING_LEVEL
from clark.env.episode_generator import MONTH_TO_SEASON, SEASONS

if TYPE_CHECKING:
    from clark.config.schema import FacilityConfig


class YearEnv:
    """
    Wraps FacilityEnv to run a full calendar year.
    The agent still makes 10-minute decisions — same action space.
    Year-level state (backlog, cycle counts, etc.) is appended to the state vector.
    """

    # Extra state variables beyond the daily env
    YEAR_STATE_SIZE = 7  # day_of_year, week_of_year, mgmt_backlog,
                         # cycle_counts_done_this_week, cycle_counts_overdue,
                         # month_progress, is_monday

    def __init__(self, facility_config: "FacilityConfig"):
        self.facility_config = facility_config
        self.day_env = FacilityEnv(facility_config)

        n = facility_config.num_workers

        # Year-level state
        self.year: int = 0
        self.work_days: list[dict] = []
        self.current_day_idx: int = 0
        self.total_work_days: int = 0

        # Carryover mechanics
        self.restock_level: float = RESTOCK_STARTING_LEVEL
        self.management_backlog: float = 0.0
        self.mgmt_carryover_hours: float = 0.0
        self.cycle_counts_done_this_week: int = 0
        self.cycle_counts_overdue: int = 0
        self.weekly_cycle_count_hours: float = 0.0

        # Weekly hustle tracking — resets every Monday
        self.weekly_hustle_hours: dict[int, float] = {i: 0.0 for i in range(n)}
        self.hustle_exhausted: dict[int, bool] = {i: False for i in range(n)}

        # Tracking
        self.year_reward: float = 0.0
        self.year_reward_breakdown: dict = {}
        self.daily_grades: list[str] = []
        self.daily_summaries: list[dict] = []
        self.is_year_done: bool = False

        # Week tracking
        self.current_week: int = 0

        # Order carryover state
        self.order_backlog: int = 0               # current unshipped orders carried forward
        self.backlog_ages: list[tuple[int, int]] = []  # list of (day_index, order_count)

    def reset(self) -> np.ndarray:
        """Start a new year. Returns initial state vector."""
        n = self.facility_config.num_workers
        self.year = random.randint(2024, 2026)
        self.work_days = self._generate_year_schedule()
        self.total_work_days = len(self.work_days)
        self.current_day_idx = 0

        self.restock_level = RESTOCK_STARTING_LEVEL
        self.management_backlog = 0.0
        self.mgmt_carryover_hours = 0.0
        self.cycle_counts_done_this_week = 0
        self.cycle_counts_overdue = 0
        self.weekly_cycle_count_hours = 0.0
        self.current_week = 0

        self.weekly_hustle_hours = {i: 0.0 for i in range(n)}
        self.hustle_exhausted = {i: False for i in range(n)}

        self.year_reward = 0.0
        self.year_reward_breakdown = {}
        self.daily_grades = []
        self.daily_summaries = []
        self.is_year_done = False

        # Reset order carryover state
        self.order_backlog = 0
        self.backlog_ages = []

        return self._start_new_day()

    def step(
        self,
        actions: list[tuple[int, int, bool]],
    ) -> tuple[np.ndarray, float, bool, dict]:
        """
        Same interface as FacilityEnv.step().
        Returns (state, reward, done, info).
        When a day ends, automatically starts the next day.
        """
        state, reward, day_done, info = self.day_env.step(actions)

        if day_done:
            day_reward = self._process_end_of_day()
            reward += day_reward

            self.year_reward += self.day_env.total_reward + day_reward
            self.current_day_idx += 1

            if self.current_day_idx >= self.total_work_days:
                year_end_reward = self._process_end_of_year()
                reward += year_end_reward
                self.is_year_done = True
                info["year_done"] = True
                info["year_summary"] = self._get_year_summary()
                return self._get_year_state(), reward, True, info

            state = self._start_new_day()
            info["new_day"] = True
            info["day_number"] = self.current_day_idx + 1
            return state, reward, False, info

        return self._augment_state(state), reward, False, info

    # ── Year schedule generation ───────────────────────────────────────────────

    def _generate_year_schedule(self) -> list[dict]:
        """Generate all ~260 work days for the year with pre-computed volumes."""
        schedule = []

        # Build day-of-week index map from config's weekly_curve
        # Config uses day names; we need a dow-index lookup
        dow_name_map = {
            0: "monday", 1: "tuesday", 2: "wednesday",
            3: "thursday", 4: "friday",
        }

        for month in range(1, 13):
            month_name = calendar.month_name[month]
            season = MONTH_TO_SEASON[month]

            # Get volume range for this month (case-insensitive lookup)
            vol_range = self._get_volume_range(month_name)
            vol_lo, vol_hi = vol_range
            vol_spread = vol_hi - vol_lo

            rules = self.facility_config.rules
            cal = calendar.Calendar()
            # Track Monday volume for Saturday scaling
            last_monday_vol_lo: int = vol_lo
            last_monday_vol_hi: int = vol_hi
            for day, dow in cal.itermonthdays2(self.year, month):
                if day == 0:
                    continue
                if dow > 5:
                    continue  # skip Sunday only

                if dow == 5:
                    # Saturday — only include if work_saturday is enabled
                    if not rules.work_saturday:
                        continue
                    sat_vol_lo = int(last_monday_vol_lo * rules.saturday_volume_fraction * 0.8)
                    sat_vol_hi = int(last_monday_vol_hi * rules.saturday_volume_fraction)
                    daily_volume = random.randint(sat_vol_lo, max(sat_vol_lo, sat_vol_hi))
                    schedule.append({
                        "year": self.year,
                        "month": month,
                        "month_name": month_name,
                        "day": day,
                        "day_of_week": dow,
                        "season": season,
                        "volume": daily_volume,
                        "vol_range": vol_range,
                    })
                    continue

                # Apply weekly volume curve (weekdays 0-4)
                curve = self.facility_config.volume.weekly_curve
                dow_name = dow_name_map[dow]
                if dow_name in curve:
                    pct_lo, pct_hi = curve[dow_name]
                else:
                    pct_lo, pct_hi = 0.5, 1.0

                day_vol_lo = vol_lo + int(vol_spread * pct_lo)
                day_vol_hi = vol_lo + int(vol_spread * pct_hi)
                daily_volume = random.randint(day_vol_lo, max(day_vol_lo, day_vol_hi))

                # Track Monday volume for Saturday scaling
                if dow == 0:
                    last_monday_vol_lo = day_vol_lo
                    last_monday_vol_hi = day_vol_hi

                schedule.append({
                    "year": self.year,
                    "month": month,
                    "month_name": month_name,
                    "day": day,
                    "day_of_week": dow,
                    "season": season,
                    "volume": daily_volume,
                    "vol_range": vol_range,
                })

        return schedule

    def _inject_temp_workers(self, peak) -> None:
        """Append temporary WorkerState objects to the day env for peak staffing."""
        from clark.env.worker import WorkerState
        if self.day_env.episode is None:
            return
        # Find the highest existing worker ID
        existing_ids = [w.worker_id for w in self.day_env.episode.workers]
        next_id = max(existing_ids) + 1 if existing_ids else 0

        oph_lo, oph_hi = peak.temp_oph_range
        for i in range(peak.extra_workers):
            temp_oph = round(random.uniform(oph_lo, oph_hi), 1)
            # Decide if temp calls off
            is_absent = random.random() < peak.temp_call_off_probability
            temp_worker = WorkerState(
                worker_id=next_id + i,
                name=f"Temp_{next_id + i}",
                base_oph=temp_oph,
                shift_hours=peak.temp_shift_hours,
                role="warehouse",
                task_eligibility_set={"pick", "pack", "idle"},
                is_absent=is_absent,
            )
            temp_worker.current_task = "idle" if is_absent else "pack"
            self.day_env.episode.workers.append(temp_worker)

    def _get_volume_range(self, month_name: str) -> tuple[int, int]:
        """Look up seasonal volume range from config, case-insensitive."""
        ranges = self.facility_config.volume.seasonal_ranges
        if month_name in ranges:
            return tuple(ranges[month_name])
        key = month_name.lower()
        for k, v in ranges.items():
            if k.lower() == key:
                return tuple(v)
        raise KeyError(f"No seasonal_range for '{month_name}' in facility config.")

    # ── Day management ─────────────────────────────────────────────────────────

    def _start_new_day(self) -> np.ndarray:
        """Initialize the daily env for the current day with carryover state."""
        day_info = self.work_days[self.current_day_idx]

        # Check for new week (Monday)
        if day_info["day_of_week"] == 0:
            self._process_new_week()

        state = self.day_env.reset(
            force_month=day_info["month"],
            force_dow=day_info["day_of_week"],
            force_volume=day_info["volume"],
        )

        # Inject carried backlog into today's order count
        if self.facility_config.rules.order_carryover_enabled and self.order_backlog > 0:
            self.day_env.episode.total_orders += self.order_backlog
            self.day_env.orders_in_queue += self.order_backlog
            # Don't reset backlog here — it carries until shipped or expired

        # Peak staffing injection
        peak = self.facility_config.peak_staffing
        if peak is not None:
            current_month = day_info["month_name"].lower()
            if current_month in [m.lower() for m in peak.months]:
                self._inject_temp_workers(peak)

        # Apply carryover restock level from previous day
        self.day_env.restock_level = self.restock_level

        # Pass backlog so management cap logic can lift it
        self.day_env._mgmt_backlog = self.management_backlog

        # Management carryover: if yesterday's minimum wasn't met,
        # management-eligible workers must catch up at day start
        if self.mgmt_carryover_hours > 0:
            mgmt_ids = set(self.facility_config.management_eligible_ids())
            for w in self.day_env.episode.workers:
                if w.worker_id in mgmt_ids and not w.is_absent:
                    catch_up = min(self.mgmt_carryover_hours, 1.0)
                    w.management_hours += catch_up
                    w.hours_worked += catch_up
                    self.mgmt_carryover_hours -= catch_up
                    if self.mgmt_carryover_hours <= 0:
                        break

        # Inject weekly hustle state
        for w in self.day_env.episode.workers:
            w.weekly_hustle_hours = self.weekly_hustle_hours.get(w.worker_id, 0.0)
            w.is_hustle_exhausted = self.hustle_exhausted.get(w.worker_id, False)

        return self._augment_state(state)

    def _process_end_of_day(self) -> float:
        """Handle end-of-day carryover and penalties. Returns extra reward."""
        reward = 0.0
        rules = self.facility_config.rules

        # Carry restock level
        self.restock_level = self.day_env.restock_level

        # Management backlog
        total_mgmt = self.day_env._get_effective_management_hours()
        required = rules.management_daily_hours_required
        delta = required - total_mgmt  # positive = shortfall, negative = surplus
        self.management_backlog = max(0.0, self.management_backlog + delta)

        # If under minimum, next day loses hours to catch up
        min_hours = rules.management_min_daily_hours
        min_shortfall = max(0.0, min_hours - total_mgmt)
        self.mgmt_carryover_hours = min_shortfall

        # Daily escalating backlog penalty — quadratic scaling
        if self.management_backlog > 0:
            penalty = -0.1 * (self.management_backlog ** 2) / 10.0
            reward += penalty
            self._add_year_reward("management_backlog_daily", penalty)

        # Accumulate cycle count hours from eligible workers
        cycle_eligible_ids = set(self.facility_config.cycle_count_eligible_ids())
        for w in self.day_env.episode.workers:
            if w.worker_id in cycle_eligible_ids:
                self.weekly_cycle_count_hours += w.cycle_count_hours_today

        # Accumulate weekly hustle hours and check for exhaustion
        for w in self.day_env.episode.workers:
            if w.hustle_hours_today > 0:
                self.weekly_hustle_hours[w.worker_id] = (
                    self.weekly_hustle_hours.get(w.worker_id, 0.0) + w.hustle_hours_today
                )
                # Threshold: 2× daily cap (config-driven per worker)
                w_cfg = next(
                    (c for c in self.facility_config.workers if c.worker_id == w.worker_id), None
                )
                daily_cap = w_cfg.hustle_daily_cap if w_cfg and w_cfg.hustle_daily_cap else w.hustle_daily_cap
                threshold = daily_cap * 2
                if (not self.hustle_exhausted.get(w.worker_id, False) and
                        self.weekly_hustle_hours[w.worker_id] >= threshold):
                    self.hustle_exhausted[w.worker_id] = True

        # Friday EOD: clear hustle exhaustion so it doesn't bleed into next week
        current_day_info = self.work_days[self.current_day_idx]
        if current_day_info["day_of_week"] == 4:
            for w_id in self.hustle_exhausted:
                self.hustle_exhausted[w_id] = False

        # Order carryover
        rules = self.facility_config.rules
        if rules.order_carryover_enabled:
            remaining = self.day_env.orders_in_queue + self.day_env.orders_picked_not_audited
            if remaining > 0:
                self.backlog_ages.append((self.current_day_idx, remaining))
                self.order_backlog += remaining

            # Expire orders older than max_carryover_days
            if rules.order_carryover_max_days > 0:
                cutoff = self.current_day_idx - rules.order_carryover_max_days
                expired = [(d, c) for d, c in self.backlog_ages if d <= cutoff]
                for day_idx, count in expired:
                    self.order_backlog -= count
                    # Expired unshipped orders = severe penalty
                    penalty = self.facility_config.rewards["per_order_incomplete"] * count * 1.5
                    reward += penalty
                    self._add_year_reward("backlog_expired", penalty)
                self.backlog_ages = [(d, c) for d, c in self.backlog_ages if d > cutoff]
                self.order_backlog = max(0, self.order_backlog)

            # Hard backlog cap penalty
            if rules.order_carryover_max_backlog > 0 and self.order_backlog > rules.order_carryover_max_backlog:
                excess = self.order_backlog - rules.order_carryover_max_backlog
                penalty = self.facility_config.rewards["per_order_incomplete"] * excess
                reward += penalty
                self._add_year_reward("backlog_cap_exceeded", penalty)

        # Record daily summary
        summary = self.day_env.get_episode_summary(management_backlog=self.management_backlog)
        self.daily_grades.append(summary["footer"]["grade"])
        self.daily_summaries.append(summary)

        return reward

    def _process_new_week(self) -> float:
        """Called on Monday — check previous week's cycle counts and backlog."""
        reward = 0.0
        rules = self.facility_config.rules
        rewards = self.facility_config.rewards

        if self.current_week > 0:
            # Check cycle count hours for previous week
            if self.weekly_cycle_count_hours >= rules.cycle_count_weekly_hours:
                bonus = rewards["cycle_count_week_complete"]
                reward += bonus
                self._add_year_reward("cycle_count_week_complete", bonus)
                if self.cycle_counts_overdue > 0:
                    self.cycle_counts_overdue -= 1
            else:
                self.cycle_counts_overdue += 1
                if self.cycle_counts_overdue <= rules.cycle_count_max_overdue_weeks:
                    penalty = rewards["cycle_count_week_missed"]
                else:
                    penalty = rewards["cycle_count_critical_overdue"]
                reward += penalty
                self._add_year_reward("cycle_count_week_missed", penalty)

            # Management backlog weekly threshold check
            if self.management_backlog > rules.management_backlog_week_threshold:
                penalty = rules.management_backlog_weekly_penalty
                self._add_year_reward("management_backlog_weekly", penalty)
                reward += penalty

        # Reset weekly counters
        n = self.facility_config.num_workers
        self.cycle_counts_done_this_week = 0
        self.weekly_cycle_count_hours = 0.0
        self.weekly_hustle_hours = {i: 0.0 for i in range(n)}
        self.current_week += 1

        self.year_reward += reward
        return reward

    def _process_end_of_year(self) -> float:
        """Final year-end scoring."""
        reward = 0.0
        rewards = self.facility_config.rewards
        rules = self.facility_config.rules

        # Final week cycle count check
        if self.weekly_cycle_count_hours < rules.cycle_count_weekly_hours:
            self.cycle_counts_overdue += 1

        # Year-end penalty if critically overdue
        if self.cycle_counts_overdue > rules.cycle_count_max_overdue_weeks:
            penalty = rewards.get("cycle_count_critical_overdue", -150.0)
            reward += penalty
            self._add_year_reward("cycle_count_year_miss", penalty)

        self.year_reward += reward
        return reward

    # ── State augmentation ─────────────────────────────────────────────────────

    def _augment_state(self, daily_state: np.ndarray) -> np.ndarray:
        """Append year-level features to the daily state vector."""
        rules = self.facility_config.rules

        year_features = np.array([
            self.current_day_idx / max(1, self.total_work_days),
            self.current_week / 52.0,
            min(self.management_backlog / 20.0, 1.0),
            self.cycle_counts_done_this_week / max(1.0, rules.cycle_count_weekly_hours),
            min(self.cycle_counts_overdue / 10.0, 1.0),
            self.restock_level,
            1.0 if (
                self.current_day_idx < self.total_work_days
                and self.work_days[self.current_day_idx]["day_of_week"] == 0
            ) else 0.0,
        ], dtype=np.float32)

        return np.concatenate([daily_state, year_features])

    def _get_year_state(self) -> np.ndarray:
        """Get state when year is done (final state)."""
        daily_state = self.day_env._get_state()
        return self._augment_state(daily_state)

    # ── Reward tracking ────────────────────────────────────────────────────────

    def _add_year_reward(self, key: str, value: float):
        self.year_reward_breakdown[key] = self.year_reward_breakdown.get(key, 0.0) + value

    # ── Year summary ───────────────────────────────────────────────────────────

    def _get_year_summary(self) -> dict:
        grade_counts = {g: self.daily_grades.count(g) for g in ["A", "B", "C", "D", "F"]}
        total_days = len(self.daily_grades)

        win_pct = (grade_counts["A"] + grade_counts["B"]) / max(1, total_days)
        if win_pct >= 0.90 and grade_counts["F"] == 0:
            year_grade = "A"
        elif win_pct >= 0.80:
            year_grade = "B"
        elif win_pct >= 0.65:
            year_grade = "C"
        elif win_pct >= 0.50:
            year_grade = "D"
        else:
            year_grade = "F"

        return {
            "year": self.year,
            "total_work_days": total_days,
            "grade_distribution": grade_counts,
            "year_grade": year_grade,
            "total_reward": round(self.year_reward, 2),
            "year_reward_breakdown": {k: round(v, 2) for k, v in self.year_reward_breakdown.items()},
            "management_backlog_final": round(self.management_backlog, 2),
            "cycle_counts_overdue_final": self.cycle_counts_overdue,
            "order_backlog_final": self.order_backlog,
            "daily_summaries": self.daily_summaries,
        }

    @property
    def state_size(self) -> int:
        """Total state size: daily worker+env state + year features."""
        cfg = self.facility_config
        daily_size = (
            cfg.num_workers * (cfg.num_tasks + 13) + 15
        )
        return daily_size + self.YEAR_STATE_SIZE

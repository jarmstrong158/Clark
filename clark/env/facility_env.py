"""
FacilityEnv — generalized warehouse simulation.

Directly parallels Jack's WarehouseEnv with all simulation logic preserved.
The ONLY changes from Jack:
- Reads from FacilityConfig instead of hardcoded config.py
- State vector built dynamically from N * (NUM_TASKS + 13) + ENV_STATE_SIZE
- Management eligibility uses config.management_eligible_ids()
- Reward signals use facility_config.rewards dict
- No hardcoded worker IDs or names
"""
from __future__ import annotations

import random
import numpy as np
from typing import Optional, TYPE_CHECKING

from clark.env.episode_generator import (
    generate_episode, EpisodeParams,
    DAY_START_HOUR, ORDER_CUTOFF_HOUR, STEP_DURATION,
    MORNING_PICK_CARTS_MIN, MORNING_PICK_CARTS_MAX,
    PER_CART_MIN, PER_CART_MAX,
    FILLER_COMPLETION_THRESHOLD,
    MANAGER_PRE_SIM_MANAGEMENT,
    SEASONS,
)
from clark.env.worker import WorkerState

if TYPE_CHECKING:
    from clark.config.schema import FacilityConfig, WorkerConfig


# ─── Constants (not user-configurable) ────────────────────────────────────────

# Task OPH multipliers (fallback when no task_oph_overrides set)
PICK_MULTIPLIER_MAIN = 2.5
PICK_MULTIPLIER_SUPPLEMENT = 2.25
TASK_OPH_MULTIPLIERS = {
    "pack": 1.0,
    "restock": 1.0,
    "side_project": 1.0,
    "management": 0.0,
    "idle": 0.0,
    "cycle_count": 0.0,
}
# Keep old name as alias for any external code still referencing it
TASK_OPH_MULTIPLIER = TASK_OPH_MULTIPLIERS

# Restock level system
RESTOCK_STARTING_LEVEL = 1.0
RESTOCK_DRAIN_FACTOR = 2.0
RESTOCK_PICK_PENALTY_THRESHOLD = 0.2
RESTOCK_PICK_PENALTY_MULTIPLIER = 0.05

# State vector sizing
WORKER_STATE_SCALARS = 14   # expanded from 13 (added task_oph_normalized at index 13)
ENV_STATE_SIZE = 17          # expanded from 15 (added carrier_urgency, order_complexity_load)

# Tasks that cannot be hustled
HUSTLE_BLOCKED_TASKS = {"management", "idle", "cycle_count"}

# Management fallback: when all management-eligible workers are absent,
# the first warehouse lead/worker handles minimum management
MANAGEMENT_FALLBACK_ROLE_PRIORITY = ("lead", "warehouse")


class FacilityEnv:
    """
    Simulates one work day in 10-minute intervals.
    Driven entirely by FacilityConfig — no hardcoded worker or facility parameters.
    """

    def __init__(self, facility_config: "FacilityConfig"):
        self.facility_config = facility_config
        self.episode: Optional[EpisodeParams] = None
        self.cooldown_tracker: dict = {}

        # OT config from rules
        self._eod_hour: float = facility_config.rules.ot_hard_stop_hour - facility_config.rules.ot_wall_clock_max
        self._ot_hard_stop: float = facility_config.rules.ot_hard_stop_hour

        # Order complexity mix for the current day: {tier_name: count}
        self.today_complexity_mix: dict[str, int] = {}

        # Order tracking
        self.orders_in_queue: int = 0
        self.orders_completed: int = 0
        self.orders_picked_not_audited: int = 0

        # Restock tracking
        self.restock_remaining: float = 0.0

        # Side project tracking
        self.deliberate_progress: float = 0.0
        self.filler_progress: float = 0.0
        self.deliberate_complete: bool = False
        self.filler_complete: bool = False

        # Reward tracking
        self.total_reward: float = 0.0
        self.reward_breakdown: dict = {}
        self.ot_hours: float = 0.0
        self.is_ot: bool = False
        self.is_done: bool = False

        # Restock interruption tracking
        self.restock_interrupted_pick: bool = False

        # Restock level
        self.restock_level: float = RESTOCK_STARTING_LEVEL
        self.restock_drain_per_order: float = 0.0
        self.restock_refill_per_hour: float = 0.0

        # Step log
        self.step_log: list = []

        # Management backlog carryover (set by YearEnv)
        self._mgmt_backlog: float = 0.0

        # Current simulation hour
        self.current_hour: float = DAY_START_HOUR

        # Derived config
        self._num_tasks: int = facility_config.num_tasks
        self._task_ids: list[str] = facility_config.task_ids
        self._task_to_idx: dict[str, int] = {t: i for i, t in enumerate(self._task_ids)}
        self._num_workers: int = facility_config.num_workers
        self._rewards: dict = facility_config.rewards
        self._mgmt_eligible_ids: set[int] = set(facility_config.management_eligible_ids())
        self._cycle_count_eligible_ids: set[int] = set(facility_config.cycle_count_eligible_ids())

        # Management fallback worker (first non-management warehouse worker)
        self._mgmt_fallback_id: Optional[int] = self._resolve_mgmt_fallback()

    def _resolve_mgmt_fallback(self) -> Optional[int]:
        """Find the fallback management worker (first lead or warehouse worker)."""
        for priority_role in MANAGEMENT_FALLBACK_ROLE_PRIORITY:
            for cfg in self.facility_config.workers:
                if cfg.role == priority_role:
                    return cfg.worker_id
        return None

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        force_month: int = None,
        force_dow: int = None,
        force_volume: int = None,
    ) -> np.ndarray:
        self.episode = generate_episode(
            facility_config=self.facility_config,
            volume=force_volume,
            day_of_week=force_dow,
            month=force_month,
            cooldown_tracker=self.cooldown_tracker,
        )
        self.current_hour = self.facility_config.rules.day_start_hour
        self.orders_in_queue = 0
        self.orders_completed = 0
        self.orders_picked_not_audited = 0
        self.restock_remaining = self.episode.restock_hours
        self.deliberate_progress = 0.0
        self.filler_progress = 0.0
        self.deliberate_complete = False
        self.filler_complete = False
        self.total_reward = 0.0
        self.reward_breakdown = {k: 0.0 for k in self._rewards}
        self.ot_hours = 0.0
        self.is_ot = False
        self.is_done = False
        self.restock_interrupted_pick = False

        # Restock level
        self.restock_level = RESTOCK_STARTING_LEVEL
        total = max(1, self.episode.total_orders)
        self.restock_drain_per_order = RESTOCK_DRAIN_FACTOR / total
        self.restock_refill_per_hour = 1.0 / max(0.1, self.episode.restock_hours)

        self.step_log = []

        # Sample today's order complexity mix
        self.today_complexity_mix = self._sample_order_complexity(self.episode.total_orders)

        # Process initial order arrivals
        self._process_arrivals()

        # Morning pick round: everyone grabs 1-2 carts before assignments
        if self.facility_config.rules.morning_pick_enabled:
            self._morning_pick_round()

        # Default assignments
        mgmt_eligible = self._mgmt_eligible_ids
        for w in self.episode.workers:
            if w.is_absent:
                w.current_task = "idle"
            elif w.is_picker:
                w.current_task = "pick"
            elif w.worker_id in mgmt_eligible:
                w.current_task = "management"
            else:
                w.current_task = "pack"

        return self._get_state()

    def _morning_pick_round(self):
        """All available workers pick 1-2 carts at day start. Driven by config rules."""
        rules = self.facility_config.rules
        carts_min = rules.morning_pick_carts_min
        carts_max = rules.morning_pick_carts_max
        per_cart_min = rules.morning_pick_per_cart_min
        per_cart_max = rules.morning_pick_per_cart_max
        for w in self.episode.workers:
            if w.is_absent:
                continue
            num_carts = random.randint(carts_min, carts_max)
            cart_total = sum(
                random.randint(per_cart_min, per_cart_max)
                for _ in range(num_carts)
            )
            picked = min(cart_total, self.orders_in_queue)
            if picked > 0:
                self.orders_in_queue -= picked
                self.orders_picked_not_audited += picked
                w.orders_picked += picked

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(
        self,
        actions: list[tuple[int, int, bool]],
    ) -> tuple[np.ndarray, float, bool, dict]:
        """
        actions: list of (worker_id, task_idx, hustle) tuples — one per worker.
        Returns: (state, reward, done, info)
        """
        step_reward = 0.0

        for worker_id, task_idx, hustle in actions:
            if 0 <= worker_id < self._num_workers and 0 <= task_idx < self._num_tasks:
                worker = self.episode.workers[worker_id]
                new_task = self._task_ids[task_idx]

                # Enforce pack-only constraint — redirect to pack, never idle
                if worker.is_pack_only and new_task != "pack":
                    new_task = "pack"

                if worker.can_do_task(new_task):
                    worker.current_task = new_task

                    # Apply hustle flag
                    can_hustle = (
                        hustle
                        and not worker.is_hustle_exhausted
                        and worker.hustle_hours_today < worker.hustle_daily_cap
                        and new_task not in HUSTLE_BLOCKED_TASKS
                    )
                    worker.hustle_mode = can_hustle

        # Check lunch — driven by config (duration > 0 enables it)
        rules = self.facility_config.rules
        is_lunch = (
            rules.lunch_duration > 0
            and abs(self.current_hour - rules.lunch_hour) < 0.01
        )
        if is_lunch:
            # Advance through lunch_duration worth of steps
            steps_for_lunch = max(1, round(rules.lunch_duration / STEP_DURATION))
            for _ in range(steps_for_lunch):
                self._process_arrivals()
                self.current_hour += STEP_DURATION
            for w in self.episode.workers:
                w.current_task = "idle"
        else:
            self._process_arrivals()
            step_reward += self._simulate_step()
            self.current_hour += STEP_DURATION

        finalize_reward = self._check_eod()
        step_reward += finalize_reward

        self._log_step(step_reward)
        self.total_reward += step_reward

        return self._get_state(), step_reward, self.is_done, self._get_info()

    # ── Simulation step ────────────────────────────────────────────────────────

    def _simulate_step(self) -> float:
        reward = 0.0
        duration = STEP_DURATION
        eod_hour = self._eod_hour

        active_workers = []
        for w in self.episode.workers:
            if w.is_absent:
                if w.current_task != "idle":
                    w.current_task = "idle"
                continue
            if self.is_ot:
                active_workers.append(w)
                continue
            if self.current_hour >= eod_hour and w.hours_remaining <= 0:
                w.current_task = "idle"
                continue
            active_workers.append(w)

        # --- Phase 1: Pickers ---
        for w in active_workers:
            if w.current_task != "pick":
                continue
            oph = w.effective_oph("pick")
            pick_mult = PICK_MULTIPLIER_MAIN if w.is_picker else PICK_MULTIPLIER_SUPPLEMENT

            # Restock level gates picking speed
            restock_mult = 1.0
            if self.restock_level < RESTOCK_PICK_PENALTY_THRESHOLD:
                t = self.restock_level / max(0.001, RESTOCK_PICK_PENALTY_THRESHOLD)
                restock_mult = RESTOCK_PICK_PENALTY_MULTIPLIER + t * (1.0 - RESTOCK_PICK_PENALTY_MULTIPLIER)

            raw_output = oph * pick_mult * restock_mult * duration + w.work_carry
            picked = min(int(raw_output), self.orders_in_queue)
            if self.orders_in_queue > 0:
                w.work_carry = raw_output - picked
            else:
                w.work_carry = 0.0
            self.orders_in_queue -= picked
            self.orders_picked_not_audited += picked
            w.orders_picked += picked

            # Drain restock level
            self.restock_level = max(0.0, self.restock_level - picked * self.restock_drain_per_order)
            w.check_fatigue()
            w.hours_worked += duration
            reward += self._add_reward("per_productive_hour", duration)

            # Soreness check
            if w.has_soreness:
                w.non_side_project_hours += duration
                w.check_soreness()

            # Restock interruption check
            if self.restock_remaining > 2.0 and self.orders_in_queue > 0:
                if not self.restock_interrupted_pick:
                    self.restock_interrupted_pick = True
                    reward += self._add_reward("restock_pick_interruption")

        # --- Phase 2: Packers ---
        packers = [w for w in active_workers if w.current_task == "pack"]
        if packers and self.orders_picked_not_audited > 0:
            packer_raws = []
            packer_capacities = []
            for w in packers:
                oph = w.effective_oph("pack")
                task_mult = TASK_OPH_MULTIPLIER.get("pack", 1.0)
                raw = oph * task_mult * duration + w.work_carry
                capacity = int(raw)
                packer_raws.append(raw)
                packer_capacities.append(capacity)

            total_capacity = sum(packer_capacities)
            available = self.orders_picked_not_audited

            for i, w in enumerate(packers):
                if available <= 0:
                    share = 0
                elif total_capacity <= available:
                    share = packer_capacities[i]
                else:
                    proportion = packer_capacities[i] / max(1, total_capacity)
                    share = min(packer_capacities[i], int(available * proportion))
                    if i == len(packers) - 1:
                        share = min(packer_capacities[i], available)

                actual = min(share, self.orders_picked_not_audited)
                if actual > 0:
                    w.work_carry = min(packer_raws[i] - actual, 1.0)
                else:
                    w.work_carry = 0.0
                self.orders_picked_not_audited -= actual
                self.orders_completed += actual
                w.orders_packed += actual
                available -= actual
                w.check_fatigue()

                reward += self._add_reward("per_order_shipped", actual)

        # Packers starved: assigned but nothing to pack while queue has orders
        if packers and self.orders_picked_not_audited == 0 and self.orders_in_queue > 0:
            starved_count = len(packers)
            reward += self._add_reward("packers_starved", starved_count)

        # Picked backlog piling up
        if self.orders_picked_not_audited > 20:
            backlog_units = self.orders_picked_not_audited // 10
            reward += self._add_reward("picked_backlog", backlog_units)

        # Update packer hours
        for w in packers:
            w.hours_worked += duration
            reward += self._add_reward("per_productive_hour", duration)
            if w.has_soreness:
                w.non_side_project_hours += duration
                w.check_soreness()

        # --- Phase 3: Restock, side_project, management, cycle_count, idle ---
        for w in active_workers:
            task = w.current_task
            if task in ("pick", "pack"):
                continue

            if task == "idle":
                reward += self._add_reward("per_idle_hour", duration)
                continue

            if task == "management":
                reward += self._process_management(w, duration)
                continue

            # Scale output by effectiveness ratio
            effectiveness = w.effective_oph(task) / max(1.0, w.base_oph)
            effective_duration = duration * effectiveness

            if task == "restock":
                prev_remaining = self.restock_remaining
                self.restock_remaining = max(0.0, self.restock_remaining - effective_duration)
                self.restock_level = min(
                    1.0,
                    self.restock_level + effective_duration * self.restock_refill_per_hour,
                )
                if prev_remaining > 0 and self.restock_remaining <= 0:
                    reward += self._add_reward("per_restock_completed", 1)

            elif task == "side_project":
                if self.orders_in_queue + self.orders_picked_not_audited > self.episode.total_orders * 0.3:
                    reward += self._add_reward("side_project_during_crunch")

                if self.episode.has_deliberate_project and not self.deliberate_complete:
                    self.deliberate_progress += effective_duration
                    reward += self._add_reward("per_deliberate_unit", effective_duration)
                    if self.deliberate_progress >= self.episode.deliberate_project_size:
                        self.deliberate_complete = True
                        reward += self._add_reward("deliberate_completion_bonus")
                else:
                    self.filler_progress += effective_duration
                    reward += self._add_reward("per_filler_unit", effective_duration)

            elif task == "cycle_count":
                if w.worker_id in self._cycle_count_eligible_ids:
                    w.cycle_count_hours_today += duration
                    w.hours_worked += duration
                    reward += self._add_reward("per_cycle_count_hour", duration)
                else:
                    reward += self._add_reward("per_idle_hour", duration)
                    w.hours_worked += duration
                continue  # skip generic hours_worked block below

            w.hours_worked += duration
            reward += self._add_reward("per_productive_hour", duration)

            if w.has_soreness and task != "side_project":
                w.non_side_project_hours += duration
                w.check_soreness()

        # Accumulate hustle hours
        for w in active_workers:
            if w.hustle_mode:
                w.hustle_hours_today += duration
                if w.hustle_hours_today >= w.hustle_daily_cap:
                    w.hustle_mode = False

        # Restock level penalties
        if self.restock_level <= 0.001:
            reward += self._add_reward("restock_level_empty")
        elif self.restock_level < RESTOCK_PICK_PENALTY_THRESHOLD:
            reward += self._add_reward("restock_level_low")

        return reward

    def _process_management(self, w: WorkerState, duration: float) -> float:
        """Handle management task for a worker. Returns reward."""
        reward = 0.0

        all_mgmt_absent = all(
            mw.is_absent for mw in self.episode.workers
            if mw.worker_id in self._mgmt_eligible_ids
        )

        if all_mgmt_absent and self._mgmt_fallback_id is not None and w.worker_id == self._mgmt_fallback_id:
            # Fallback worker handling minimum management
            fallback_worker = next(
                (mw for mw in self.episode.workers if mw.worker_id == self._mgmt_fallback_id), None
            )
            if fallback_worker:
                min_hours = self.facility_config.rules.management_min_daily_hours
                if fallback_worker.management_hours < min_hours:
                    w.management_hours += duration
                    w.hours_worked += duration
                    reward += self._add_reward("per_productive_hour", duration)
                    reward += self._add_reward("per_management_hour", duration)
                else:
                    reward += self._add_reward("per_idle_hour", duration)
                    w.hours_worked += duration
        else:
            total_mgmt = sum(
                mw.management_hours for mw in self.episode.workers
                if mw.worker_id in self._mgmt_eligible_ids
            )
            backlog = getattr(self, "_mgmt_backlog", 0.0)
            daily_cap = self.facility_config.rules.management_daily_hours_required + backlog
            if total_mgmt < daily_cap:
                w.management_hours += duration
                w.hours_worked += duration
                reward += self._add_reward("per_productive_hour", duration)
                reward += self._add_reward("per_management_hour", duration)
            else:
                reward += self._add_reward("per_idle_hour", duration)
                w.hours_worked += duration

        return reward

    # ── Order arrivals ─────────────────────────────────────────────────────────

    def _process_arrivals(self):
        if self.episode is None:
            return
        cutoff = self.facility_config.rules.order_cutoff_hour
        current = round(self.current_hour, 2)
        to_remove = []
        for key, count in self.episode.arrival_schedule.items():
            if key >= cutoff:
                to_remove.append(key)
            elif key <= current + 0.01:
                self.orders_in_queue += count
                to_remove.append(key)
        for key in to_remove:
            del self.episode.arrival_schedule[key]

    # ── End of day ─────────────────────────────────────────────────────────────

    def _check_eod(self) -> float:
        """Returns finalization reward (0.0 if day isn't over yet)."""
        orders_remaining = self.orders_in_queue + self.orders_picked_not_audited
        eod_hour = self._eod_hour
        ot_hard_stop = self._ot_hard_stop

        if self.current_hour >= eod_hour:
            if orders_remaining > 0:
                if not self.is_ot:
                    self.is_ot = True
                self.ot_hours = self.current_hour - eod_hour
                # Hard stop — use epsilon to guard against FP accumulation
                # (57 steps of 1/6 from 9.0 can land at 18.4999...93 instead of 18.5)
                if self.current_hour >= ot_hard_stop - 1e-9:
                    return self._finalize_episode()
            else:
                return self._finalize_episode()
        return 0.0

    def _finalize_episode(self) -> float:
        """End-of-day scoring. Returns total finalization reward."""
        self.is_done = True
        reward = 0.0

        orders_remaining = self.orders_in_queue + self.orders_picked_not_audited

        # OT penalty
        if self.ot_hours > 0:
            reward += self._add_reward("per_ot_hour", self.ot_hours)

        # Order completion
        if orders_remaining <= 0:
            reward += self._add_reward("all_orders_complete_bonus")
        else:
            reward += self._add_reward("per_order_incomplete", orders_remaining)
            if self.ot_hours > 0:
                reward += self._add_reward("ot_incomplete_flat")

        # Restock EOD check
        if self.restock_remaining <= 0:
            reward += self._add_reward("all_restock_bonus")
        else:
            restock_tasks_remaining = max(1, int(self.restock_remaining / STEP_DURATION))
            reward += self._add_reward("per_restock_bleed", restock_tasks_remaining)

        # Management check
        total_mgmt = self._get_effective_management_hours()
        required = self.facility_config.rules.management_daily_hours_required
        if total_mgmt >= required:
            reward += self._add_reward("management_duty_met")
        elif self.episode.total_orders >= self.facility_config.rules.high_volume_day_orders:
            pass  # heavy day — management backlog acceptable
        else:
            reward += self._add_reward("management_duty_missed")

        # Filler completion check
        if self.filler_progress >= FILLER_COMPLETION_THRESHOLD:
            self.filler_complete = True
            reward += self._add_reward("filler_completion_bonus")

        return reward

    def _get_effective_management_hours(self) -> float:
        """
        Returns total management hours for today's EOD check.
        Primary: all management-eligible workers.
        If ALL management-eligible workers are absent, fallback worker's hours count.
        """
        all_mgmt_absent = all(
            w.is_absent for w in self.episode.workers
            if w.worker_id in self._mgmt_eligible_ids
        )
        if all_mgmt_absent and self._mgmt_fallback_id is not None:
            fallback = next(
                (w for w in self.episode.workers if w.worker_id == self._mgmt_fallback_id), None
            )
            return fallback.management_hours if fallback else 0.0
        return sum(
            w.management_hours for w in self.episode.workers
            if w.worker_id in self._mgmt_eligible_ids
        )

    # ── Reward helper ──────────────────────────────────────────────────────────

    def _add_reward(self, key: str, multiplier: float = 1.0) -> float:
        if key not in self._rewards:
            return 0.0
        value = self._rewards[key] * multiplier
        self.reward_breakdown[key] = self.reward_breakdown.get(key, 0.0) + value
        return value

    # ── State vector ───────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        num_tasks = self._num_tasks
        total_size = self._num_workers * (num_tasks + WORKER_STATE_SCALARS) + ENV_STATE_SIZE
        state = np.zeros(total_size, dtype=np.float32)
        idx = 0

        required = self.facility_config.rules.management_daily_hours_required

        for w in self.episode.workers:
            # One-hot task encoding
            task_idx = self._task_to_idx.get(w.current_task, self._task_to_idx.get("idle", 0))
            state[idx + task_idx] = 1.0
            idx += num_tasks

            # 14 scalar features (13 original + task_oph_normalized at index 13)
            state[idx] = w.effective_oph() / 25.0
            idx += 1
            state[idx] = w.hours_worked / 10.0
            idx += 1
            state[idx] = w.hours_remaining / 10.0
            idx += 1
            state[idx] = 1.0 if (w.sleep_debuff != "normal" or w.health_debuff != "normal") else 0.0
            idx += 1
            state[idx] = 1.0 if w.individual_debuff is not None else 0.0
            idx += 1
            state[idx] = 1.0 if w.is_fatigued else 0.0
            idx += 1
            state[idx] = 1.0 if w.is_picker else 0.0
            idx += 1
            state[idx] = 1.0 if w.is_pack_only else 0.0
            idx += 1
            # Soreness progress (0-1)
            from clark.env.worker import SORENESS_HOUR_THRESHOLD
            state[idx] = w.non_side_project_hours / SORENESS_HOUR_THRESHOLD if w.has_soreness else 0.0
            idx += 1
            state[idx] = w.management_hours / max(1.0, required)
            idx += 1
            # Hustle daily ratio
            state[idx] = w.hustle_hours_today / max(0.01, w.hustle_daily_cap)
            idx += 1
            # Weekly hustle pressure
            state[idx] = w.weekly_hustle_hours / max(0.01, w.hustle_weekly_threshold)
            idx += 1
            # Exhaustion flag
            state[idx] = 1.0 if w.is_hustle_exhausted else 0.0
            idx += 1
            # Feature 13: task_oph_normalized — OPH on current task / max_possible_oph
            task_oph = self._get_worker_task_oph_by_id(w.worker_id, w.current_task)
            max_oph = TASK_OPH_MULTIPLIERS.get("pick", 1.0) * PICK_MULTIPLIER_MAIN * max(1.0, w.base_oph)
            state[idx] = min(1.0, task_oph / max(1.0, max_oph))
            idx += 1

        eod_hour = self._eod_hour
        day_start = self.facility_config.rules.day_start_hour

        # ENV features (17 total)
        state[idx] = (self.current_hour - day_start) / max(0.01, eod_hour - day_start)
        idx += 1
        state[idx] = self.orders_in_queue / max(1, self.episode.total_orders)
        idx += 1
        state[idx] = self.orders_completed / max(1, self.episode.total_orders)
        idx += 1
        state[idx] = self.orders_picked_not_audited / max(1, self.episode.total_orders)
        idx += 1
        state[idx] = self.restock_remaining / max(1.0, self.episode.restock_hours)
        idx += 1

        # Side project progress
        if self.episode.has_deliberate_project and self.episode.deliberate_project_size > 0:
            state[idx] = self.deliberate_progress / self.episode.deliberate_project_size
        else:
            state[idx] = self.filler_progress / 10.0
        idx += 1

        # Season one-hot (4)
        season_idx = SEASONS.index(self.episode.season) if self.episode.season in SEASONS else 0
        state[idx + season_idx] = 1.0
        idx += 4

        state[idx] = 1.0 if self.episode.is_high_volume else 0.0
        idx += 1

        total_mgmt = self._get_effective_management_hours()
        state[idx] = total_mgmt / max(0.01, required)
        idx += 1

        state[idx] = 1.0 if self.episode.picker_needs_replacement else 0.0
        idx += 1

        # Restock level (0-1)
        state[idx] = self.restock_level
        idx += 1

        # Feature 15: carrier_urgency
        state[idx] = self._compute_carrier_urgency()
        idx += 1

        # Feature 16: order_complexity_load
        state[idx] = self._compute_complexity_load()
        idx += 1

        return state

    # ── OPH resolution ─────────────────────────────────────────────────────────

    def _get_worker_task_oph(
        self,
        worker: WorkerState,
        worker_config: "WorkerConfig",
        task_id: str,
    ) -> float:
        """
        Get effective OPH for a worker on a task.
        Priority: task_oph_overrides[task_id] > base_oph × standard_multiplier
        """
        if worker_config.task_oph_overrides and task_id in worker_config.task_oph_overrides:
            return worker_config.task_oph_overrides[task_id]
        multiplier = TASK_OPH_MULTIPLIERS.get(task_id, 1.0)
        return worker.base_oph * multiplier

    def _get_worker_task_oph_by_id(self, worker_id: int, task_id: str) -> float:
        """Convenience: look up worker config by ID then delegate to _get_worker_task_oph."""
        worker_cfg = next(
            (w for w in self.facility_config.workers if w.worker_id == worker_id), None
        )
        if worker_cfg is None or self.episode is None:
            return 1.0
        worker_state = next(
            (w for w in self.episode.workers if w.worker_id == worker_id), None
        )
        if worker_state is None:
            return 1.0
        return self._get_worker_task_oph(worker_state, worker_cfg, task_id)

    # ── Carrier deadline ───────────────────────────────────────────────────────

    @property
    def hours_until_carrier_pickup(self) -> Optional[float]:
        if self.facility_config.rules.carrier_pickup_hour is None:
            return None
        return max(0.0, self.facility_config.rules.carrier_pickup_hour - self.current_hour)

    # ── Order complexity ───────────────────────────────────────────────────────

    def _sample_order_complexity(self, total_orders: int) -> dict[str, int]:
        """
        Sample a complexity mix for today's orders.
        Returns {tier_name: count} summing to total_orders.
        """
        tiers = self.facility_config.order_complexity.tiers
        if not tiers:
            return {}
        counts: dict[str, int] = {}
        remaining = total_orders
        for i, tier in enumerate(tiers):
            if i == len(tiers) - 1:
                counts[tier.name] = remaining
            else:
                n = int(round(tier.weight * total_orders))
                n = max(0, min(n, remaining))
                counts[tier.name] = n
                remaining -= n
        return counts

    def _compute_complexity_load(self) -> float:
        """
        Weighted average OPH multiplier for today's order mix.
        1.0 = standard, <1.0 = complex-heavy, >1.0 = simple-heavy.
        """
        tiers = self.facility_config.order_complexity.tiers
        if not tiers:
            return 1.0
        total = sum(self.today_complexity_mix.get(t.name, 0) for t in tiers)
        if total == 0:
            return 1.0
        weighted = sum(
            self.today_complexity_mix.get(t.name, 0) * t.oph_multiplier
            for t in tiers
        )
        return weighted / total

    def _compute_carrier_urgency(self) -> float:
        """
        Carrier urgency: 0.0 when no carrier deadline, else 1 - (hours_remaining / shift_span).
        Rises toward 1.0 as the carrier pickup approaches.
        """
        pickup = self.facility_config.rules.carrier_pickup_hour
        if pickup is None:
            return 0.0
        day_start = self.facility_config.rules.day_start_hour
        shift_span = max(0.01, pickup - day_start)
        elapsed = self.current_hour - day_start
        return max(0.0, min(1.0, elapsed / shift_span))

    # ── Info / logging ─────────────────────────────────────────────────────────

    def _get_info(self) -> dict:
        return {
            "current_hour": self.current_hour,
            "orders_in_queue": self.orders_in_queue,
            "orders_completed": self.orders_completed,
            "orders_picked_not_audited": self.orders_picked_not_audited,
            "total_orders": self.episode.total_orders,
            "restock_remaining": self.restock_remaining,
            "total_reward": self.total_reward,
            "is_ot": self.is_ot,
            "picker_needs_replacement": self.episode.picker_needs_replacement,
        }

    def _log_step(self, step_reward: float):
        workers_state = []
        total_mgmt = self._get_effective_management_hours()
        for w in self.episode.workers:
            logged_task = w.current_task
            required = self.facility_config.rules.management_daily_hours_required
            if logged_task == "management" and total_mgmt >= required:
                logged_task = "idle"
            workers_state.append({
                "name": w.name,
                "task": logged_task,
                "effective_oph": round(w.effective_oph(), 2),
                "debuffs": {
                    "sleep": w.sleep_debuff,
                    "health": w.health_debuff,
                    "individual": w.individual_debuff,
                },
            })

        self.step_log.append({
            "hour": round(self.current_hour, 2),
            "workers": workers_state,
            "orders_remaining": self.orders_in_queue,
            "orders_completed": self.orders_completed,
            "picked_not_audited": self.orders_picked_not_audited,
            "restock_remaining": round(self.restock_remaining, 2),
            "side_project_progress": round(
                self.deliberate_progress if self.episode.has_deliberate_project
                else self.filler_progress, 2
            ),
            "running_reward": round(self.total_reward, 2),
            "restock_level": round(self.restock_level, 3),
        })

    def get_episode_summary(self, management_backlog: float = 0.0) -> dict:
        ep = self.episode
        total = ep.total_orders
        shipped = self.orders_completed
        remaining = self.orders_in_queue + self.orders_picked_not_audited
        required = self.facility_config.rules.management_daily_hours_required
        min_hours = self.facility_config.rules.management_min_daily_hours

        all_orders = (shipped >= total)
        restock_pct_raw = max(0.0, min(1.0, 1.0 - (self.restock_remaining / max(0.01, ep.restock_hours))))
        all_restock = (restock_pct_raw >= 0.95)

        total_mgmt_grade = self._get_effective_management_hours()
        mgmt_full = (total_mgmt_grade >= required)
        mgmt_minimum = (total_mgmt_grade >= min_hours)

        no_ot = (self.ot_hours <= 0)

        backlog_threshold = self.facility_config.rules.management_backlog_week_threshold

        if not all_orders or not mgmt_minimum:
            grade = "F"
        else:
            demerits = 0
            if not all_restock:
                demerits += 1
            if not mgmt_full:
                demerits += 1
            if not no_ot:
                demerits += 1
            if management_backlog > backlog_threshold:
                demerits += 1
            grades = ["A", "B", "C", "D", "F"]
            grade = grades[min(demerits, 4)]

        restock_pct = max(0.0, min(1.0, 1.0 - (self.restock_remaining / max(0.01, ep.restock_hours))))

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        picker_worker = next((w for w in ep.workers if w.worker_id == ep.picker_id), None)

        return {
            "header": {
                "date": f"{ep.year}-{ep.month:02d}-{ep.day:02d}",
                "season": ep.season,
                "month": ep.month_name,
                "day_of_week": day_names[ep.day_of_week] if ep.day_of_week < 5 else str(ep.day_of_week),
                "total_orders": total,
                "debuffs_fired": ep.debuffs_fired,
                "picker": picker_worker.name if picker_worker and not ep.picker_needs_replacement else "NEEDS_REPLACEMENT",
                "is_high_volume": ep.is_high_volume,
            },
            "footer": {
                "orders_shipped": shipped,
                "orders_total": total,
                "orders_remaining": remaining,
                "reward": round(self.total_reward, 2),
                "reward_breakdown": {k: round(v, 2) for k, v in self.reward_breakdown.items() if v != 0},
                "ot_hours": round(self.ot_hours, 2),
                "restock_pct": round(restock_pct * 100, 1),
                "deliberate_complete": self.deliberate_complete,
                "filler_complete": self.filler_complete,
                "grade": grade,
            },
            "steps": self.step_log,
        }

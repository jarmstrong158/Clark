"""Variable-dimension action masking: (N, M) bool tensor.

Generalizes Jack's get_valid_action_mask() from a fixed (7, 12) shape to
a config-driven (N, M) shape where N = num_workers and M = num_tasks.

Key differences from Jack's version:
  - Task column = task index in facility_config.task_ids (not hardcoded TASK_TO_IDX)
  - Hustle is a *separate* head — this mask covers task assignment only (N, M)
  - Management eligibility and cycle-count eligibility come from FacilityConfig
  - No combined (task + hustle) encoding; that was Jack's (N, M*2) format
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clark.env.facility_env import FacilityEnv

from clark.env.facility_env import HUSTLE_BLOCKED_TASKS


def get_action_mask(env: "FacilityEnv") -> np.ndarray:
    """
    Build a (N, M) boolean action mask for the current step.

    mask[worker_i][task_j] = True  ⟹  worker i may be assigned task j.

    Rules (mirroring Jack's get_valid_action_mask logic):
      - Absent worker → only idle column True
      - OT with orders remaining → only pick/pack True (pack-only → only pack)
      - Shift exhausted at EOD → only idle True
      - Pack-only worker → only pack + idle True
      - Otherwise: apply task eligibility + business rule constraints

    Works with both FacilityEnv and YearEnv (accesses .day_env if needed).

    Returns:
        np.ndarray of dtype=bool, shape (N, M)
    """
    # Support YearEnv wrapper
    day_env = getattr(env, "day_env", env)
    cfg = day_env.facility_config
    # Use actual worker list — temp peak-staffing workers may push N above _num_workers
    N = len(day_env.episode.workers)
    M = day_env._num_tasks
    task_ids: list[str] = day_env._task_ids
    task_to_idx: dict[str, int] = day_env._task_to_idx

    idle_idx: int = task_to_idx.get("idle", 0)
    pick_idx: int = task_to_idx.get("pick", -1)
    pack_idx: int = task_to_idx.get("pack", -1)

    mask = np.zeros((N, M), dtype=bool)

    mgmt_eligible_ids: set[int] = day_env._mgmt_eligible_ids
    cycle_eligible_ids: set[int] = day_env._cycle_count_eligible_ids
    mgmt_fallback_id: int | None = day_env._mgmt_fallback_id

    orders_remaining = (day_env.orders_in_queue + day_env.orders_picked_not_audited) > 0
    eod_hour = day_env._eod_hour

    # Management quota check (used in the normal-case branch)
    def _management_available(worker_id: int) -> bool:
        """True if this worker may still do meaningful management work."""
        all_mgmt_absent = all(
            w.is_absent for w in day_env.episode.workers
            if w.worker_id in mgmt_eligible_ids
        )
        if all_mgmt_absent:
            if worker_id != mgmt_fallback_id:
                return False
            fallback = next(
                (w for w in day_env.episode.workers if w.worker_id == mgmt_fallback_id),
                None,
            )
            if fallback is None:
                return False
            return fallback.management_hours < cfg.rules.management_min_daily_hours
        else:
            if worker_id not in mgmt_eligible_ids:
                return False
            total_mgmt = sum(
                w.management_hours for w in day_env.episode.workers
                if w.worker_id in mgmt_eligible_ids
            )
            backlog = getattr(day_env, "_mgmt_backlog", 0.0)
            daily_cap = cfg.rules.management_daily_hours_required + backlog
            return total_mgmt < daily_cap

    for w_id in range(N):
        worker = day_env.episode.workers[w_id]

        # ── Absent ────────────────────────────────────────────────────────────
        if worker.is_absent:
            mask[w_id, idle_idx] = True
            continue

        # ── OT: orders still open → pick + pack only ──────────────────────────
        if day_env.is_ot and orders_remaining:
            if worker.is_pack_only:
                if pack_idx >= 0:
                    mask[w_id, pack_idx] = True
            else:
                if pick_idx >= 0:
                    mask[w_id, pick_idx] = True
                if pack_idx >= 0:
                    mask[w_id, pack_idx] = True
            continue

        # ── Shift exhausted (EOD, no OT) → idle only ─────────────────────────
        if (day_env.current_hour >= eod_hour
                and worker.hours_remaining <= 0
                and not day_env.is_ot):
            mask[w_id, idle_idx] = True
            continue

        # ── Pack-only restriction ─────────────────────────────────────────────
        if worker.is_pack_only:
            if pack_idx >= 0:
                mask[w_id, pack_idx] = True
            mask[w_id, idle_idx] = True
            continue

        # ── Normal case: apply per-task eligibility ──────────────────────────
        for j, t_id in enumerate(task_ids):
            if t_id == "idle":
                # Idle allowed for any non-absent, non-exhausted worker
                # (agent can choose to idle; it may be suboptimal but valid)
                mask[w_id, j] = True
                continue

            if t_id == "management":
                mask[w_id, j] = _management_available(w_id)
                continue

            if t_id == "cycle_count":
                mask[w_id, j] = (w_id in cycle_eligible_ids)
                continue

            if t_id == "restock":
                # Only useful when stock isn't full
                mask[w_id, j] = (
                    day_env.restock_level < 1.0
                    and worker.eligible_for(t_id)
                )
                continue

            # General task: check worker eligibility
            mask[w_id, j] = worker.eligible_for(t_id)

    return mask


def get_hustle_mask(env: "FacilityEnv") -> np.ndarray:
    """
    Build a (N, 2) boolean hustle mask.

    Column 0 (no hustle) = always True for non-absent workers.
    Column 1 (hustle)    = True if worker is hustle-capable.

    Note: This mask is conservative — it marks hustle=True whenever the
    *worker* can hustle, regardless of which task they will be assigned.
    Per-task hustle blocking (management/idle/cycle_count) is enforced inside
    the environment when actions are applied, and may also be applied inside
    PPO during log-prob evaluation.

    Works with both FacilityEnv and YearEnv.

    Returns:
        np.ndarray of dtype=bool, shape (N, 2)
    """
    day_env = getattr(env, "day_env", env)
    N = len(day_env.episode.workers)
    mask = np.zeros((N, 2), dtype=bool)

    for w_id in range(N):
        worker = day_env.episode.workers[w_id]
        if worker.is_absent:
            # Absent workers: neither hustle variant matters, but set no-hustle
            mask[w_id, 0] = True
            mask[w_id, 1] = False
        else:
            mask[w_id, 0] = True            # no-hustle always valid
            mask[w_id, 1] = worker.can_hustle  # hustle if worker hasn't hit cap

    return mask

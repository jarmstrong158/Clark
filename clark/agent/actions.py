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

from clark.env.facility_env import HUSTLE_BLOCKED_TASKS, RESTOCK_PICK_PENALTY_THRESHOLD

# Minimum-dwell: ticks a worker must stay on a task before a non-emergency
# switch is allowed. 3 ticks = 30 min at 10-min ticks. Humans don't
# context-switch every 10 minutes, and a pure productivity ramp was too weak
# to stop the policy thrashing — this makes continuity structural.
DWELL_MIN_TICKS = 3


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
    # Pick buffer cap — when too many orders are picked-but-not-packed,
    # carts/staging is full and no one can pick any more until packing
    # drains some. Real-warehouse cart-space proxy. None = unlimited
    # (legacy behavior).
    pick_buffer_cap = day_env.facility_config.rules.pick_buffer_capacity
    pick_buffer_full = (
        pick_buffer_cap is not None
        and day_env.orders_picked_not_audited >= pick_buffer_cap
    )

    # v2.5/v2.6: multi-gate hard-mask on filler. Real warehouse managers
    # check the situation continuously — they BOTH project the day from
    # the morning AND react to current state ("look at the clock at 3pm
    # — we have 200 unshipped and 2 hours, no more side-projects").
    # A single projection-only gate (v2.4) missed mid-range days where
    # the projection said "fine" but the policy spent the day on filler
    # and then ran out of time. Multi-gate fires when ANY of:
    #
    #   1. _projected_dvc > 0.65 -- proactive: morning curve says heavy
    #   2. _pending_pct > 0.25   -- reactive: queue is piling up now
    #   3. _schedule_pressure > 0.20 -- reactive: behind where we should be
    #   4. _time_pressure > 0.85 -- reactive: not enough time left for
    #                               remaining work (the missing manager-
    #                               clock check)
    #   5. _restock_pressure (v2.6) -- restock_level < 0.35: stock is
    #      approaching the 0.20 cliff where picking speed crashes to 5%.
    #      Mask filler PROACTIVELY so the policy is forced to allocate
    #      restockers BEFORE collapse — preventing the cascade that
    #      forces OT and drops the grade. v2.5 audit showed: any OT use
    #      = automatic A-grade disqualification, and the dominant cause
    #      of OT cascade was restock-level collapse. 0.35 gives a 0.15
    #      buffer above the cliff so the mask fires with time for
    #      restockers to catch up at normal speed.
    #
    # Pending_pct AND total_orders inside the mask are OK to read
    # directly — the mask is env-side code, not an observation the
    # policy can game. The "no oracle" rule applies only to obs feats.
    _FILLER_TASKS = {"side_project", "loading", "training",
                     "quality_check", "returns_processing", "receiving"}
    _total_orders = max(1, day_env.episode.total_orders)
    _orders_pending = day_env.orders_in_queue + day_env.orders_picked_not_audited
    _projected_dvc = day_env._compute_projected_demand_ratio()
    _pending_pct = _orders_pending / _total_orders
    _eod_h = day_env._eod_hour
    _day_start_h = cfg.rules.day_start_hour
    _shift_span = max(0.01, _eod_h - _day_start_h)
    _time_progress = max(0.0, min(1.0,
        (day_env.current_hour - _day_start_h) / _shift_span))
    _completion_progress = day_env.orders_completed / _total_orders
    _schedule_pressure = max(0.0, _time_progress - _completion_progress)
    _time_pressure = day_env._compute_time_pressure()
    _restock_pressure = (
        day_env._restock_enabled and day_env.restock_level < 0.35
    )
    _filler_in_stress = (
        _projected_dvc > 0.65
        or _pending_pct > 0.25
        or _schedule_pressure > 0.20
        or _time_pressure > 0.85
        or _restock_pressure
    )

    # Per-task daily-hours auto-off: once the total hours spent on a task
    # (summed across all workers today) reach its configured target, the
    # task is removed from the action space for the rest of the day so no
    # further labor is wasted on it. Same idea as the management cap below,
    # generalized to any task the facility put a `daily_hours` target on.
    # Management is intentionally excluded — it has its own quota logic in
    # `_management_available` (driven by management_daily_hours_required).
    _capped_met: set[str] = set()
    _task_caps = cfg.tasks.daily_hours
    if _task_caps:
        for _t_id, _target in _task_caps.items():
            if _t_id == "management":
                continue
            _done = sum(
                w.task_hours_today.get(_t_id, 0.0)
                for w in day_env.episode.workers
            )
            if _done >= _target:
                _capped_met.add(_t_id)

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

        # ── OT: orders still open → pick + pack (+ restock if depleted) ──────
        # Audit found 33% of F days were "restock collapse": stock ran
        # empty mid-day, picking dropped to 5% speed, OT triggered to
        # catch up — but the OT mask blocked restock, so workers had
        # no way to refill. Picking stayed at 5% through OT, hard-stop
        # hit, day failed with hundreds of orders remaining. Real
        # warehouses obviously DO restock during OT in this scenario.
        # Now: restock is masked True during OT only when restock_level
        # is below the pick-penalty threshold (i.e. picking is being
        # crippled by lack of stock). Otherwise it stays blocked so
        # workers focus on shipping.
        if day_env.is_ot and orders_remaining:
            restock_idx = task_to_idx.get("restock", -1)
            restock_critical = (
                restock_idx >= 0
                and day_env._restock_enabled
                and day_env.restock_level < RESTOCK_PICK_PENALTY_THRESHOLD
            )
            if worker.is_pack_only:
                if pack_idx >= 0:
                    mask[w_id, pack_idx] = True
            else:
                # Buffer-full: pick removed, force pack. If pack also
                # unavailable (no pack task in this facility) fall back
                # to allowing pick so the worker isn't stranded.
                pack_available = pack_idx >= 0
                if pick_idx >= 0 and not (pick_buffer_full and pack_available):
                    mask[w_id, pick_idx] = True
                if pack_available:
                    mask[w_id, pack_idx] = True
                # Restock allowed only when stock is critically low and
                # this worker is actually eligible. Lets the model
                # break the restock-collapse cascade during OT.
                if restock_critical and worker.eligible_for("restock"):
                    mask[w_id, restock_idx] = True
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
                # Idle is INVALID in the normal branch — matches Jack's
                # action mask. Idle is only available via the absent /
                # shift-exhausted special-case branches above. Without
                # this constraint, a deterministic-at-init policy can
                # commit to "everyone idle" as the lowest-penalty path
                # and never escape, since idle never accumulates the
                # reward signals (per_order_shipped, per_productive_hour)
                # that would push it toward better behaviors.
                mask[w_id, j] = False
                continue

            # Per-task daily-hours cap reached → task is off for the day.
            if t_id in _capped_met:
                mask[w_id, j] = False
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

            # Pick buffer cap — block "pick" when buffer is full so
            # the model is forced toward pack/restock/etc. Eligibility
            # check still applies. (Stranded-worker safety check below
            # the per-task loop re-enables pick if no other action ended
            # up valid.)
            if t_id == "pick" and pick_buffer_full:
                mask[w_id, j] = False
                continue

            # v2.4 hard-mask: filler tasks are structurally invalid on
            # days projected to exceed 85% of capacity. Forces the policy
            # toward pick / pack / restock / management instead of
            # letting it indulge the "filler is OK" attractor learned
            # during pretrain. Fires from tick 1 of stress days because
            # the projection is sharp early (canonical curve has 45% of
            # the day's orders landing at day_start instant).
            if t_id in _FILLER_TASKS and _filler_in_stress:
                mask[w_id, j] = False
                continue

            # General task: check worker eligibility
            mask[w_id, j] = worker.eligible_for(t_id)

        # Minimum-dwell lock: once started, a worker must hold a task for
        # DWELL_MIN_TICKS before a non-emergency switch. The OT / EOD /
        # absent / pack-only branches above already `continue` past here, so
        # real emergencies still override. We lock ONLY when the current task
        # is still a valid choice this tick (its mask bit is True): if it just
        # became invalid (hit a daily-hours cap, restock filled, pick buffer
        # full, stress-gated filler, ...) that's a legitimate reason to
        # switch, so the normal options stay open.
        if worker.ticks_on_task < DWELL_MIN_TICKS:
            ct_idx = task_to_idx.get(worker.current_task, -1)
            if ct_idx >= 0 and mask[w_id, ct_idx]:
                locked = np.zeros(M, dtype=bool)
                locked[ct_idx] = True
                mask[w_id] = locked

        # Stranded-worker safety: never emit an all-False row (it would
        # NaN the policy softmax). If pick was blocked by the buffer cap,
        # prefer re-enabling pick (better a small buffer overflow than a
        # stranded worker). Otherwise — e.g. a worker whose only eligible
        # task just hit its daily-hours cap — fall back to idle.
        if not mask[w_id].any():
            if pick_buffer_full and pick_idx >= 0:
                mask[w_id, pick_idx] = True
            else:
                mask[w_id, idle_idx] = True

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

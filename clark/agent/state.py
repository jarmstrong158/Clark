"""State builder: converts FacilityEnv state into structured token dicts for the transformer."""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clark.config.schema import FacilityConfig
    from clark.env.facility_env import FacilityEnv

from clark.env.facility_env import ENV_STATE_SIZE, WORKER_STATE_SCALARS, TASK_OPH_MULTIPLIERS, PICK_MULTIPLIER_MAIN
from clark.env.worker import SORENESS_HOUR_THRESHOLD
from clark.env.episode_generator import DAY_START_HOUR, SEASONS


class StateBuilder:
    """
    Converts a live FacilityEnv into structured token dicts for the transformer.

    Output shapes:
      worker_feats:    (N, 14)
      task_feats:      (M, 3)
      env_feats:       (17,)
      worker_role_ids: (N,) int64
      task_type_ids:   (M,) int64
    """

    # Canonical role → integer mapping
    ROLE_IDS: dict[str, int] = {
        "manager": 0,
        "assistant_manager": 1,
        "lead": 2,
        "warehouse": 3,
    }

    def __init__(self, facility_config: "FacilityConfig"):
        self.config = facility_config
        self.N: int = facility_config.num_workers
        self.M: int = facility_config.num_tasks
        self.task_ids: list[str] = facility_config.task_ids
        # task_id → column index in task dimension
        self.task_type_ids: dict[str, int] = {
            t_id: i for i, t_id in enumerate(self.task_ids)
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self, env: "FacilityEnv") -> dict:
        """
        Build a structured state dict from a live FacilityEnv after reset/step.

        Uses the actual worker list from the env (not self.N) so that peak-staffing
        temp workers injected at runtime are included without an IndexError.

        Returns:
            {
                "worker_feats":    np.ndarray (actual_N, 14)  float32
                "task_feats":      np.ndarray (M, 3)          float32
                "env_feats":       np.ndarray (17,)            float32
                "worker_role_ids": np.ndarray (actual_N,)     int64
                "task_type_ids":   np.ndarray (M,)            int64
            }
        """
        actual_workers = env.episode.workers
        worker_role_ids = np.array(
            [self.ROLE_IDS.get(w.role, 3) for w in actual_workers],
            dtype=np.int64,
        )
        return {
            "worker_feats": self._build_worker_feats(env),
            "task_feats": self._build_task_feats(env),
            "env_feats": self._build_env_feats(env),
            "worker_role_ids": worker_role_ids,
            "task_type_ids": np.arange(self.M, dtype=np.int64),
        }

    def validate(self, state_dict: dict) -> bool:
        """Check shapes and that no NaN values are present.

        N is read FROM the state itself, not from the config-time `self.N`,
        because peak-staffing days inject temp workers that push the actual
        worker count above the config baseline.
        """
        wf = state_dict.get("worker_feats")
        if wf is None or wf.ndim != 2 or wf.shape[1] != WORKER_STATE_SCALARS:
            return False
        actual_N = wf.shape[0]
        expected_shapes = {
            "worker_feats": (actual_N, WORKER_STATE_SCALARS),
            "task_feats": (self.M, 3),
            "env_feats": (ENV_STATE_SIZE,),
            "worker_role_ids": (actual_N,),
            "task_type_ids": (self.M,),
        }
        for key, shape in expected_shapes.items():
            if key not in state_dict:
                return False
            arr = state_dict[key]
            if arr.shape != shape:
                return False
            if np.issubdtype(arr.dtype, np.floating) and np.any(np.isnan(arr)):
                return False
        return True

    # ── Worker features ────────────────────────────────────────────────────────

    def _build_worker_feats(self, env: "FacilityEnv") -> np.ndarray:
        """
        14 scalar features per worker.

        Feature index:
          0  normalized OPH (base_oph / 30.0)
          1  hours_worked / shift_hours
          2  hours_remaining / shift_hours
          3  generic_debuff (sleep+health, 1.0 if any active)
          4  individual_debuff (1.0 if active)
          5  fatigue flag (1.0 if fatigued)
          6  is_picker (1.0 if designated picker today)
          7  is_pack_only (1.0 if restricted)
          8  soreness_progress (non_side_project_hours / SORENESS_HOUR_THRESHOLD)
          9  management_hours / management_daily_hours_required
         10  hustle_today_ratio (hustle_hours_today / hustle_daily_cap)
         11  hustle_weekly_ratio (weekly_hustle_hours / hustle_weekly_threshold)
         12  is_hustle_exhausted (1.0 or 0.0)
         13  task_oph_normalized — OPH on current task / max_possible_oph, normalized [0,1]
        """
        required = self.config.rules.management_daily_hours_required
        actual_workers = env.episode.workers
        feats = np.zeros((len(actual_workers), WORKER_STATE_SCALARS), dtype=np.float32)

        for i, w in enumerate(actual_workers):
            shift_h = max(0.01, w.shift_hours)
            feats[i, 0] = w.base_oph / 30.0
            feats[i, 1] = w.hours_worked / shift_h
            feats[i, 2] = w.hours_remaining / shift_h
            feats[i, 3] = 1.0 if (w.sleep_debuff != "normal" or w.health_debuff != "normal") else 0.0
            feats[i, 4] = 1.0 if w.individual_debuff is not None else 0.0
            feats[i, 5] = 1.0 if w.is_fatigued else 0.0
            feats[i, 6] = 1.0 if w.is_picker else 0.0
            feats[i, 7] = 1.0 if w.is_pack_only else 0.0
            feats[i, 8] = (
                w.non_side_project_hours / SORENESS_HOUR_THRESHOLD
                if w.has_soreness else 0.0
            )
            feats[i, 9] = w.management_hours / max(0.01, required)
            feats[i, 10] = w.hustle_hours_today / max(0.01, w.hustle_daily_cap)
            feats[i, 11] = w.weekly_hustle_hours / max(0.01, w.hustle_weekly_threshold)
            feats[i, 12] = 1.0 if w.is_hustle_exhausted else 0.0
            # Feature 13: task_oph_normalized
            task_oph = env._get_worker_task_oph_by_id(w.worker_id, w.current_task)
            max_oph = PICK_MULTIPLIER_MAIN * max(1.0, w.base_oph)
            feats[i, 13] = min(1.0, task_oph / max(1.0, max_oph))

        return feats

    # ── Task features ──────────────────────────────────────────────────────────

    def _build_task_feats(self, env: "FacilityEnv") -> np.ndarray:
        """
        3 features per task:
          0  demand_signal  — urgency/load (float 0-1)
          1  availability_flag — 1.0 if any worker can do this task
          2  task_type_id   — integer index (used as embedding key; stored as float)
        """
        feats = np.zeros((self.M, 3), dtype=np.float32)
        orders_total = max(1, env.episode.total_orders)
        orders_pending = env.orders_in_queue + env.orders_picked_not_audited

        # Pre-compute which workers are available (not absent)
        available_workers = [w for w in env.episode.workers if not w.is_absent]

        # Filler-task suppression. Per-worker time tracking exposed that
        # the model was sending 16-26h/day to quality_check / training /
        # receiving / returns_processing on bad-Cmp days while pick was
        # 21h short of good days. Root cause: filler tasks had a
        # CONSTANT demand=0.5 regardless of pick/pack queue state, so
        # when pick demand briefly dropped (early morning, low queue),
        # the model committed workers to filler — then orders flooded
        # in but workers were locked into the wrong tasks.
        # Fix: suppress filler demand based on the larger of two
        # pressures:
        #   pending_pressure = current outstanding orders / total
        #   schedule_pressure = how far behind schedule we are
        # When EITHER is significant, filler demand drops toward 0.
        pending_pressure = orders_pending / orders_total
        eod_hour = env._eod_hour
        day_start = self.config.rules.day_start_hour
        time_progress = max(0.0, min(1.0,
            (env.current_hour - day_start) / max(0.01, eod_hour - day_start)
        ))
        completion_progress = max(0.0, min(1.0, env.orders_completed / orders_total))
        schedule_pressure = max(0.0, time_progress - completion_progress)
        # Combined pressure: the bigger concern wins. Multiplier 1.7 means
        # filler demand hits 0 around 30% pending OR 30% behind schedule.
        # At 0 pressure (slack day, on schedule), filler demand = 0.5.
        filler_demand = max(0.0, 0.5 - max(pending_pressure, schedule_pressure) * 1.7)

        for j, t_id in enumerate(self.task_ids):
            # --- demand_signal ---
            # Pick and pack demand are SEPARATE signals — previously both got
            # `(orders_in_queue + orders_picked_not_audited) / total` so the
            # model could not differentiate "pickers needed" from "packers
            # needed" at the per-task level. Result: model picked aggressively
            # then left orders sitting picked-but-not-packed (large
            # `picked_backlog` penalty in audit). With separate signals the
            # cross-attention can route workers to whichever stage of the
            # pipeline is actually backlogged.
            if t_id == "pick":
                demand = min(1.0, env.orders_in_queue / orders_total)
            elif t_id == "pack":
                demand = min(1.0, env.orders_picked_not_audited / orders_total)
            elif t_id == "restock":
                demand = max(0.0, 1.0 - env.restock_level)
            elif t_id == "management":
                required = self.config.rules.management_daily_hours_required
                total_mgmt = sum(
                    w.management_hours for w in env.episode.workers
                    if w.worker_id in env._mgmt_eligible_ids
                )
                demand = max(0.0, min(1.0, 1.0 - total_mgmt / max(0.01, required)))
            elif t_id == "idle":
                demand = 0.0
            else:
                # Filler tasks (side_project, cycle_count, receiving,
                # quality_check, training, returns_processing, loading).
                # Demand scales DOWN as pick/pack pressure or schedule
                # pressure rises — see filler_demand computation above.
                demand = filler_demand

            feats[j, 0] = demand

            # --- availability_flag ---
            available = False
            for w in available_workers:
                if w.eligible_for(t_id) and not (w.is_pack_only and t_id != "pack"):
                    available = True
                    break
            feats[j, 1] = 1.0 if available else 0.0

            # --- task_type_id (as float for consistency; cast to int for embedding) ---
            feats[j, 2] = float(j)

        return feats

    # ── Environment features ───────────────────────────────────────────────────

    def _build_env_feats(self, env: "FacilityEnv") -> np.ndarray:
        """
        17 global environment features (mirrors _get_state() ENV block in FacilityEnv).

          0   time_of_day_norm        (current_hour - DAY_START) / shift_span
          1   orders_in_queue_norm    orders_in_queue / total_orders
          2   orders_completed_norm   orders_completed / total_orders
          3   orders_audited_norm     orders_picked_not_audited / total_orders
          4   restock_remaining_norm  restock_remaining / restock_hours
          5   side_project_progress   deliberate or filler progress normalized
          6-9 season_one_hot          [winter, spring, summer, fall]
         10   is_high_volume          (1.0 / 0.0)
         11   management_progress     total_mgmt_hours / required
         12   picker_needs_replacement (1.0 / 0.0)
         13   restock_level           (0-1)
         14   is_ot                   (1.0 / 0.0)
         15   carrier_urgency         1-(hours_until_pickup/shift_span) if deadline set, else 0.0
         16   order_complexity_load   weighted avg OPH multiplier for today's order mix
        """
        ep = env.episode
        eod_hour = env._eod_hour
        day_start = self.config.rules.day_start_hour
        required = self.config.rules.management_daily_hours_required
        total_orders = max(1, ep.total_orders)

        feats = np.zeros(ENV_STATE_SIZE, dtype=np.float32)

        feats[0] = (env.current_hour - day_start) / max(0.01, eod_hour - day_start)
        feats[1] = env.orders_in_queue / total_orders
        feats[2] = env.orders_completed / total_orders
        feats[3] = env.orders_picked_not_audited / total_orders
        feats[4] = env.restock_remaining / max(1.0, ep.restock_hours)

        # Side project progress
        if ep.has_deliberate_project and ep.deliberate_project_size > 0:
            feats[5] = env.deliberate_progress / ep.deliberate_project_size
        else:
            feats[5] = env.filler_progress / 10.0

        # Season one-hot (4 slots: indices 6-9)
        season_idx = SEASONS.index(ep.season) if ep.season in SEASONS else 0
        feats[6 + season_idx] = 1.0

        feats[10] = 1.0 if ep.is_high_volume else 0.0

        total_mgmt = env._get_effective_management_hours()
        feats[11] = total_mgmt / max(0.01, required)

        feats[12] = 1.0 if ep.picker_needs_replacement else 0.0
        feats[13] = env.restock_level
        feats[14] = 1.0 if env.is_ot else 0.0

        # Feature 15: carrier_urgency
        feats[15] = env._compute_carrier_urgency()

        # Feature 16: order_complexity_load
        feats[16] = env._compute_complexity_load()

        return feats

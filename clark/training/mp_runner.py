"""MultiprocessRunner — one OS subprocess per env slot.

Mirrors the BatchedRunner public API (states/masks/hustle_masks/step/
pop_just_reset_flags/configs/__len__/n_envs/slots-via-snapshot) but pushes
env stepping AND state/mask construction into worker processes. The main
process only owns the GPU model and the per-env stats.

Why this exists: profiling showed env.step + StateBuilder.build +
get_action_mask cost ~510 μs per env-step single-threaded, and the in-process
BatchedRunner walks all envs sequentially. With N worker processes pinned to
N CPU cores, that work fans out N-way and the GPU forward overlaps with the
next batch of env steps via pipe round-trips.

Protocol (one Pipe per worker, main↔worker, request/response):
  Request from main:   ("STEP", actions_list)  |  ("SWAP", new_config)
  Response from worker: dict with keys
       reward, done, info, state, mask, hmask, finished_episode

`finished_episode` is None unless the env's year completed on this step,
in which case it's {"reward": episode_total, "steps": int,
"daily_summaries": list, "config": FacilityConfig}. After a finished episode
the worker stays on the SAME config (env.reset()); rotation to a new config
happens via an explicit SWAP command from main when years_per_config is hit.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Optional

import numpy as np

from clark.config.schema import FacilityConfig


# ── Worker entrypoint (must be module-level for Windows spawn) ────────────────

def _worker_loop(conn, init_config: FacilityConfig) -> None:
    """One env slot living inside its own process."""
    # Imports here so the parent's import doesn't pay this on every spawn.
    from clark.env.year_env import YearEnv
    from clark.agent.state import StateBuilder
    from clark.agent.actions import get_action_mask, get_hustle_mask

    cfg = init_config
    env = YearEnv(cfg)
    builder = StateBuilder(cfg)
    env.reset()
    episode_reward = 0.0
    episode_steps = 0

    def _emit_state(extra: dict) -> dict:
        return {
            **extra,
            "state": builder.build(env.day_env),
            "mask": get_action_mask(env.day_env),
            "hmask": get_hustle_mask(env.day_env),
            "current_day_idx": env.current_day_idx,
            "total_work_days": env.total_work_days,
        }

    # Initial handshake: emit starting state for tick 0.
    conn.send(_emit_state({
        "reward": 0.0, "done": False, "info": {}, "finished_episode": None,
        "just_reset": True,
    }))

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break

        kind = msg[0]
        if kind == "STOP":
            break

        if kind == "SWAP":
            new_cfg = msg[1]
            cfg = new_cfg
            env = YearEnv(cfg)
            builder = StateBuilder(cfg)
            env.reset()
            episode_reward = 0.0
            episode_steps = 0
            conn.send(_emit_state({
                "reward": 0.0, "done": False, "info": {}, "finished_episode": None,
                "just_reset": True,
            }))
            continue

        if kind == "STEP":
            task_a, hustle_a = msg[1], msg[2]
            acts = [(i, task_a[i], bool(hustle_a[i])) for i in range(len(task_a))]
            _, reward, done, info = env.step(acts)
            episode_reward += float(reward)
            episode_steps += 1

            # If a day boundary just fired, surface that day's summary back
            # to main so it can show real-time grade signal during the
            # 30-60 min that an in-progress year takes to complete. Without
            # this, the main process is blind to grades until episode_done.
            day_just_finished = None
            if info.get("new_day") and env.daily_summaries:
                completed_day = env.daily_summaries[-1]
                day_footer = completed_day.get("footer", {})
                day_header = completed_day.get("header", {})
                # Send a slim digest, not the full summary — keep IPC cheap.
                rb = day_footer.get("reward_breakdown", {}) or {}
                day_just_finished = {
                    "grade": day_footer.get("grade", "?"),
                    "ot":    day_footer.get("ot_hours", 0.0) > 0.0,
                    "ot_h":  float(day_footer.get("ot_hours", 0.0)),
                    "completion": (
                        max(0.0, (day_header.get("total_orders", 1)
                                  - day_footer.get("orders_remaining", 0))
                            / max(1, day_header.get("total_orders", 1)))
                    ),
                    "orders_total":     int(day_header.get("total_orders", 0)),
                    "orders_remaining": int(day_footer.get("orders_remaining", 0)),
                    "restock_pct":      float(day_footer.get("restock_pct", 0.0)),
                    "reward":           float(day_footer.get("reward", 0.0)),
                    # Slim reward breakdown — only the components that
                    # tend to blow up in spirals. Keep IPC tiny.
                    "rb": {
                        "incomplete":  float(rb.get("per_order_incomplete", 0.0)),
                        "ot_flat":     float(rb.get("ot_incomplete_flat", 0.0)),
                        "backlog":     float(rb.get("picked_backlog", 0.0)),
                        "rs_low":      float(rb.get("restock_level_low", 0.0)),
                        "rs_empty":    float(rb.get("restock_level_empty", 0.0)),
                        "rs_bleed":    float(rb.get("per_restock_bleed", 0.0)),
                        "rs_intr":     float(rb.get("restock_pick_interruption", 0.0)),
                        "starved":     float(rb.get("packers_starved", 0.0)),
                        "idle":        float(rb.get("per_idle_hour", 0.0)),
                        "prod":        float(rb.get("per_productive_hour", 0.0)),
                        "mgmt":        float(rb.get("per_management_hour", 0.0)),
                        "shipped":     float(rb.get("per_order_shipped", 0.0)),
                        # Restart B: surface the filler-during-crunch penalty
                        # and per-filler-unit base so we can SEE whether the
                        # scaled penalty is biting. Pre-Restart-B this was
                        # firing at -1890/day but invisible on dashboards —
                        # flew blind for weeks. Now it shows up in rb["crunch"]
                        # and rb["filler"] alongside everything else.
                        "crunch":      float(rb.get("side_project_during_crunch", 0.0)),
                        "filler":      float(rb.get("per_filler_unit", 0.0)),
                    },
                }
                # Aggregate task-time totals across all workers so the
                # day digest can show "of 135 worker-hours today, 60 went
                # to pick, 50 to pack, 18 to idle, ..." without paying
                # per-worker IPC cost. The full per-worker breakdown is
                # still in env.daily_summaries[-1]['footer']['worker_time']
                # if we ever want to deep-dive a specific day.
                wt = day_footer.get("worker_time", {})
                task_totals: dict = {}
                for worker_tasks in wt.values():
                    for t, h in worker_tasks.items():
                        task_totals[t] = task_totals.get(t, 0.0) + float(h)
                day_just_finished["task_hours"] = {
                    k: round(v, 1) for k, v in task_totals.items()
                }

            finished = None
            just_reset = False
            if done:
                # Capture year summary BEFORE resetting.
                finished = {
                    "reward": episode_reward,
                    "steps": episode_steps,
                    "daily_summaries": env.daily_summaries,
                    "config": cfg,
                }
                # Reset to start a new year on the SAME config; main process
                # decides via SWAP whether to rotate to a different config.
                env.reset()
                episode_reward = 0.0
                episode_steps = 0
                just_reset = True

            conn.send(_emit_state({
                "reward": float(reward),
                "done": done,
                "info": info,
                "finished_episode": finished,
                "day_just_finished": day_just_finished,
                "just_reset": just_reset,
            }))
            continue

        # Unknown message — protocol error.
        conn.send({"error": f"unknown message kind: {kind}"})

    conn.close()


# ── Public runner ─────────────────────────────────────────────────────────────

class MultiprocessRunner:
    """Drop-in alternative to BatchedRunner backed by N worker processes."""

    def __init__(
        self,
        n_envs: int,
        config_factory: Callable[[], FacilityConfig],
        years_per_config: int = 5,
        mp_context: str = "spawn",
    ):
        if n_envs <= 0:
            raise ValueError(f"n_envs must be positive, got {n_envs}")
        self.n_envs = n_envs
        self.config_factory = config_factory
        self.years_per_config = years_per_config
        self.ctx = mp.get_context(mp_context)

        self._configs: list[FacilityConfig] = []
        self._years_on_config: list[int] = [0] * n_envs
        self._just_reset_flags: list[bool] = [True] * n_envs

        # Cached most-recent state/mask/hmask per slot (filled by INIT and step).
        self._states: list[dict] = [None] * n_envs           # type: ignore
        self._masks: list[np.ndarray] = [None] * n_envs      # type: ignore
        self._hmasks: list[np.ndarray] = [None] * n_envs     # type: ignore
        self._current_day_idxs: list[int] = [0] * n_envs
        self._total_work_days: int = 0
        # Rolling buffer of completed-day digests across all slots —
        # gives main-process visibility into mid-episode grades. Drained
        # by drain_recent_days() at heartbeat time.
        self._recent_days: list[dict] = []

        # Spawn one worker per slot.
        self._procs: list[mp.Process] = []
        self._pipes = []
        for _ in range(n_envs):
            cfg = config_factory()
            self._configs.append(cfg)
            parent_conn, child_conn = self.ctx.Pipe(duplex=True)
            p = self.ctx.Process(target=_worker_loop, args=(child_conn, cfg), daemon=True)
            p.start()
            child_conn.close()  # parent doesn't need this end
            self._procs.append(p)
            self._pipes.append(parent_conn)

        # Drain initial-state handshake.
        for b, pipe in enumerate(self._pipes):
            r = pipe.recv()
            self._states[b] = r["state"]
            self._masks[b] = r["mask"]
            self._hmasks[b] = r["hmask"]
            self._current_day_idxs[b] = r["current_day_idx"]
            # Track the MAX across slots so heartbeat percentage never
            # overshoots when slots have different year lengths (Saturdays).
            self._total_work_days = max(self._total_work_days, r["total_work_days"])

    # ── Same shape as BatchedRunner ──────────────────────────────────────────

    def states(self) -> list[dict]:
        return self._states

    def masks(self) -> list[np.ndarray]:
        return self._masks

    def hustle_masks(self) -> list[np.ndarray]:
        return self._hmasks

    def configs(self) -> list[FacilityConfig]:
        return self._configs

    def current_day_idxs(self) -> list[int]:
        return list(self._current_day_idxs)

    @property
    def total_work_days(self) -> int:
        return self._total_work_days

    def pop_just_reset_flags(self) -> list[bool]:
        flags = list(self._just_reset_flags)
        self._just_reset_flags = [False] * self.n_envs
        return flags

    def __len__(self) -> int:
        return self.n_envs

    # ── Stepping ─────────────────────────────────────────────────────────────

    def step_send(
        self,
        task_actions: list[list[int]],
        hustle_actions: list[list[int]],
    ) -> None:
        """Fan-out actions to all workers without waiting. Pair with step_recv."""
        for b, pipe in enumerate(self._pipes):
            pipe.send(("STEP", task_actions[b], hustle_actions[b]))

    def step_recv(self) -> list[dict]:
        """Block until all workers respond. Updates cached state/mask/hmask."""
        out: list[dict] = []
        for b, pipe in enumerate(self._pipes):
            r = pipe.recv()
            self._states[b] = r["state"]
            self._masks[b] = r["mask"]
            self._hmasks[b] = r["hmask"]
            self._current_day_idxs[b] = r["current_day_idx"]

            episode_done = r["finished_episode"] is not None
            if episode_done:
                self._years_on_config[b] += 1
                self._just_reset_flags[b] = True
                if self._years_on_config[b] >= self.years_per_config:
                    # Rotate config: ask worker to swap, then refresh cache.
                    new_cfg = self.config_factory()
                    self._configs[b] = new_cfg
                    self._years_on_config[b] = 0
                    pipe.send(("SWAP", new_cfg))
                    swap_r = pipe.recv()
                    self._states[b] = swap_r["state"]
                    self._masks[b] = swap_r["mask"]
                    self._hmasks[b] = swap_r["hmask"]
                    self._current_day_idxs[b] = swap_r["current_day_idx"]
                    self._total_work_days = swap_r["total_work_days"]

            # Accumulate any per-day digest the worker sent back so the
            # main loop's heartbeat can show real-time grade signal.
            day_digest = r.get("day_just_finished")
            if day_digest is not None:
                self._recent_days.append(day_digest)

            out.append({
                "reward": r["reward"],
                "done": r["done"],
                "info": r["info"],
                "episode_done": episode_done,
                "finished_episode": r["finished_episode"],
            })
        return out

    def drain_recent_days(self) -> list[dict]:
        """Return + clear the buffer of completed-day digests. Called by the
        training loop's heartbeat to surface mid-episode grade signal."""
        out = self._recent_days
        self._recent_days = []
        return out

    def step(
        self,
        task_actions: list[list[int]],
        hustle_actions: list[list[int]],
    ) -> list[dict]:
        """Convenience: send + recv in one call (non-pipelined callers)."""
        self.step_send(task_actions, hustle_actions)
        return self.step_recv()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        for pipe in self._pipes:
            try:
                pipe.send(("STOP",))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        self._pipes.clear()
        self._procs.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

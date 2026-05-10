"""BatchedRunner — owns N parallel YearEnvs for Tier-2 batched pretraining.

Each slot in the batch runs an independent year on an independent facility.
The runner exposes list-of-state-dict interfaces so the model can do one
batched forward over all N envs per tick (see ClarkActorCritic.forward_batched).

Design notes:
  * Envs are stepped sequentially in Python. Env.step is cheap (~150μs); the
    real cost is the model forward, and *that* is the part we batch. Spawning
    subprocesses would add pickle overhead to buy back ~2% of wall-clock.
  * When an env's year finishes, we immediately reset it with a fresh random
    facility and continue. The agent's LSTM hidden state for that env slot
    must also be zeroed — the runner tracks which slots just reset via the
    `just_reset` flag so the caller can zero the corresponding hidden slices.
  * StateBuilder is rebuilt whenever a new facility is sampled (it caches
    task-id ordering, which is config-specific).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.agent.state import StateBuilder
from clark.config.schema import FacilityConfig
from clark.env.year_env import YearEnv


class EnvSlot:
    """One slot in a BatchedRunner — env + builder + bookkeeping."""

    def __init__(self, config: FacilityConfig):
        self.config = config
        self.env = YearEnv(config)
        self.builder = StateBuilder(config)
        self.env.reset()
        self.years_on_config = 0
        self.just_reset = True   # first call also counts as "just reset"

        # Per-episode stats — the batched pretrain loop reads these when an
        # episode completes to populate progress logs.
        self.episode_reward = 0.0
        self.episode_steps = 0

    def state(self) -> dict:
        return self.builder.build(self.env.day_env)

    def mask(self) -> np.ndarray:
        return get_action_mask(self.env.day_env)

    def hustle_mask(self) -> np.ndarray:
        return get_hustle_mask(self.env.day_env)

    def swap_config(self, new_config: FacilityConfig) -> None:
        self.config = new_config
        self.env = YearEnv(new_config)
        self.builder = StateBuilder(new_config)
        self.env.reset()
        self.years_on_config = 0
        self.just_reset = True
        self.episode_reward = 0.0
        self.episode_steps = 0


class BatchedRunner:
    """Run B parallel YearEnvs with periodic facility rotation.

    Parameters
    ----------
    n_envs : int
        Batch size B — number of parallel envs.
    config_factory : callable
        Zero-argument callable that returns a fresh FacilityConfig.
        Called at startup and whenever an env slot needs a new config.
    years_per_config : int
        Number of full years to run on a config before rotating to a new one.
        Matches the single-env pretrain cadence.
    """

    def __init__(
        self,
        n_envs: int,
        config_factory: Callable[[], FacilityConfig],
        years_per_config: int = 5,
    ):
        if n_envs <= 0:
            raise ValueError(f"n_envs must be positive, got {n_envs}")
        self.n_envs = n_envs
        self.config_factory = config_factory
        self.years_per_config = years_per_config
        self.slots: list[EnvSlot] = [
            EnvSlot(config_factory()) for _ in range(n_envs)
        ]
        # Mirrors MultiprocessRunner — gives the heartbeat code visibility
        # into per-day grades that complete during an in-progress year.
        self._recent_days: list[dict] = []

    # ── Batch introspection ──────────────────────────────────────────────────

    def states(self) -> list[dict]:
        return [s.state() for s in self.slots]

    def masks(self) -> list[np.ndarray]:
        return [s.mask() for s in self.slots]

    def hustle_masks(self) -> list[np.ndarray]:
        return [s.hustle_mask() for s in self.slots]

    def configs(self) -> list[FacilityConfig]:
        return [s.config for s in self.slots]

    def current_day_idxs(self) -> list[int]:
        return [s.env.current_day_idx for s in self.slots]

    @property
    def total_work_days(self) -> int:
        # Slots may have different year lengths (e.g. one with Saturdays
        # enabled, another without). Use max so the heartbeat percentage
        # never overshoots, then `avg_day / total_work_days` reads sensibly
        # for the fastest slot and pessimistically for the others.
        if not self.slots:
            return 0
        return max(s.env.total_work_days for s in self.slots)

    def pop_just_reset_flags(self) -> list[bool]:
        """Return per-slot `just_reset` flags, then clear them.

        The caller uses this to know which LSTM hidden state slots to zero.
        """
        flags = [s.just_reset for s in self.slots]
        for s in self.slots:
            s.just_reset = False
        return flags

    # ── Stepping ─────────────────────────────────────────────────────────────

    # Pipelined API used by pretrain_batched. For in-proc there's no real
    # overlap (no worker procs to do work in the background), so step_send
    # just stashes the actions and step_recv runs the actual env.step calls.
    def step_send(
        self,
        task_actions: list[list[int]],
        hustle_actions: list[list[int]],
    ) -> None:
        self._pending_ta = task_actions
        self._pending_ha = hustle_actions

    def step_recv(self) -> list[dict]:
        return self.step(self._pending_ta, self._pending_ha)

    def step(
        self,
        task_actions: list[list[int]],
        hustle_actions: list[list[int]],
    ) -> list[dict]:
        """Step all B envs one tick in parallel.

        Args:
            task_actions:   list length B, each a list[int] of length N_i
            hustle_actions: list length B, each a list[int] of length N_i

        Returns:
            List of B dicts with keys:
                reward (float), done (bool), info (dict),
                episode_done (bool) — True if the env's YEAR completed on this step,
                finished_episode (dict | None) — populated when episode_done,
                    containing {'reward', 'steps', 'daily_summaries', 'config'}
        """
        out = []
        for b, slot in enumerate(self.slots):
            ta = task_actions[b]
            ha = hustle_actions[b]
            # Build env action tuples. len(ta) handles temp peak-staffing
            # workers pushing N above config.num_workers mid-year.
            acts = [(i, ta[i], bool(ha[i])) for i in range(len(ta))]
            _, reward, done, info = slot.env.step(acts)
            slot.episode_reward += float(reward)
            slot.episode_steps += 1

            # Surface per-day digest at every day boundary so the heartbeat
            # can show real-time grade signal — same idea as the MP runner.
            if info.get("new_day") and slot.env.daily_summaries:
                completed_day = slot.env.daily_summaries[-1]
                d_footer = completed_day.get("footer", {})
                d_header = completed_day.get("header", {})
                d_total = d_header.get("total_orders", 1) or 1
                d_remaining = d_footer.get("orders_remaining", 0)
                self._recent_days.append({
                    "grade": d_footer.get("grade", "?"),
                    "ot": d_footer.get("ot_hours", 0.0) > 0.0,
                    "completion": max(0.0, (d_total - d_remaining) / d_total),
                })

            episode_done = False
            finished: Optional[dict] = None

            if done:
                episode_done = True
                # Capture episode summary for the training-loop logger BEFORE
                # we swap the config out from under it.
                finished = {
                    "reward": slot.episode_reward,
                    "steps": slot.episode_steps,
                    "daily_summaries": slot.env.daily_summaries,
                    "config": slot.config,
                }

                slot.years_on_config += 1
                if slot.years_on_config >= self.years_per_config:
                    slot.swap_config(self.config_factory())
                else:
                    # Same config — just reset the year.
                    slot.env.reset()
                    slot.just_reset = True
                    slot.episode_reward = 0.0
                    slot.episode_steps = 0

            out.append({
                "reward": float(reward),
                "done": done,
                "info": info,
                "episode_done": episode_done,
                "finished_episode": finished,
            })

        return out

    def drain_recent_days(self) -> list[dict]:
        """Return + clear the buffer of completed-day digests."""
        out = self._recent_days
        self._recent_days = []
        return out

    def __len__(self) -> int:
        return self.n_envs

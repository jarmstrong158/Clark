"""Per-update training metrics logger for Clark.

Writes a rolling JSON file (`training_metrics.json`) containing time-series
data the dashboard needs but `episode_log.json` doesn't capture:

- PPO update health (clip fraction, P-loss, V-loss, entropy, grad norm)
- Curriculum stage per episode
- Per-episode facility shape (N workers, M tasks, stage)
- Reward-component dominance per window
- Per-day reward breakdown digests (drained from runner mid-episode)
- Per-stage N-distribution counts (sampler health check)

Designed for live tail by the dashboard. Each `record_*` call appends a
single point and rewrites the file atomically (small enough that this is
fine — capped at `max_points` to bound disk).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


# Cap on time-series length; older points are dropped to keep file size bounded
# and dashboard load times snappy.
DEFAULT_MAX_POINTS = 5000


class TrainingMetricsLogger:
    """Append-only-ish time series store, JSON-on-disk for the dashboard."""

    def __init__(
        self,
        log_dir: str = "clark/data/logs",
        max_points: int = DEFAULT_MAX_POINTS,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_points = max_points
        self.path = self.log_dir / "training_metrics.json"

        # Time-series buckets — each is a list of dicts, oldest-first.
        # Bucket choice is per-event-type so dashboard can render each
        # without joining across mismatched x-axes.
        self.ppo_updates: list[dict] = []   # one entry per PPO update cycle
        self.episodes:    list[dict] = []   # one entry per completed episode
        self.windows:     list[dict] = []   # one entry per log_interval window
        # Per-day grade digests collected mid-episode — gives the dashboard
        # signal long before any year (~13k steps) completes. Drained from
        # the runner's drain_recent_days() at heartbeat time.
        self.day_grades:  list[dict] = []   # {grade, ot, completion, t}
        # Per-stage N-distribution counter — counts how many episodes the
        # curriculum sampler actually drew at each (stage, N) pair. Lets us
        # spot bugs like "stage 2 supposedly samples N=5-25 but never draws
        # N>10" without waiting hundreds of episodes for it to be obvious
        # in the by-N table. Structure: {"1": {"5": 12, "6": 8, ...}, "2": {...}}
        self.n_distribution: dict = {}

        # Heartbeat/status — overwritten, not appended.
        self.status: dict = {
            "started_at": time.time(),
            "last_heartbeat": time.time(),
            "current_episode": 0,
            "n_episodes_target": 0,
            "current_stage": 1,
            "ticks": 0,
            "elapsed_s": 0.0,
            "alive": True,
        }

        self._load_existing()

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.ppo_updates = data.get("ppo_updates", [])[-self.max_points:]
            self.episodes    = data.get("episodes", [])[-self.max_points:]
            self.windows     = data.get("windows", [])[-self.max_points:]
            self.day_grades  = data.get("day_grades", [])[-self.max_points:]
            self.n_distribution = data.get("n_distribution", {}) or {}
            # Keep the last status's started_at if restoring — useful for
            # cumulative elapsed across resumes.
            prev_status = data.get("status", {})
            if prev_status:
                self.status["started_at"] = time.time()  # reset clock per launch
                self.status["n_episodes_target"] = prev_status.get(
                    "n_episodes_target", 0
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # start fresh

    # ── Public recorders ──────────────────────────────────────────────────────

    def record_ppo_update(
        self,
        episode: int,
        clip_fraction: float,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        n_updates: int = 1,
    ) -> None:
        self.ppo_updates.append({
            "ep": episode,
            "clip": round(clip_fraction, 4),
            "p_loss": round(policy_loss, 5),
            "v_loss": round(value_loss, 5),
            "entropy": round(entropy, 4),
            "n": n_updates,
            "t": time.time(),
        })
        self._trim(self.ppo_updates)

    def record_episode(
        self,
        episode: int,
        stage: int,
        n_workers: int,
        n_tasks: int,
        win_rate_year: float,        # fraction of A/B days across this year
        ot_rate_year: float,
        completion_rate_year: float,
        reward_per_worker: float,
        last_grade: str,
        config_name: str = "",
    ) -> None:
        self.episodes.append({
            "ep": episode,
            "stage": stage,
            "N": n_workers,
            "M": n_tasks,
            "win_year": round(win_rate_year, 4),
            "ot_year": round(ot_rate_year, 4),
            "cmp_year": round(completion_rate_year, 4),
            "rw_per_worker": round(reward_per_worker, 2),
            "last_grade": last_grade,
            "cfg": config_name,
            "t": time.time(),
        })
        self._trim(self.episodes)
        # Bump the per-stage N counter — JSON dict keys must be strings.
        stage_key = str(stage)
        n_key = str(n_workers)
        bucket = self.n_distribution.setdefault(stage_key, {})
        bucket[n_key] = int(bucket.get(n_key, 0)) + 1

    def record_window(
        self,
        episode: int,
        avg_reward_per_worker: float,
        win_rate: float,
        ot_rate: float,
        completion_rate: float,
        reward_dominance: dict,      # {component_name: value}
        grade_distribution: dict,    # {A: n, B: n, ...}
    ) -> None:
        # Top 5 dominance components for compactness
        top_dominance = dict(sorted(
            reward_dominance.items(), key=lambda kv: -abs(kv[1])
        )[:5]) if reward_dominance else {}
        self.windows.append({
            "ep": episode,
            "avg_rw": round(avg_reward_per_worker, 2),
            "win": round(win_rate, 4),
            "ot": round(ot_rate, 4),
            "cmp": round(completion_rate, 4),
            "dominance": {k: round(v, 2) for k, v in top_dominance.items()},
            "grades": grade_distribution,
            "t": time.time(),
        })
        self._trim(self.windows)

    def record_day_grades(self, day_digests: list[dict]) -> None:
        """Append per-day grade digests drained from the runner. Used by
        the dashboard to show learning signal long before the first year
        completes."""
        now = time.time()
        for d in day_digests:
            entry = {
                "grade": d.get("grade", "?"),
                "ot": bool(d.get("ot", False)),
                "ot_h": round(d.get("ot_h", 0.0), 2),
                "completion": round(d.get("completion", 0.0), 4),
                "orders_total":     int(d.get("orders_total", 0)),
                "orders_remaining": int(d.get("orders_remaining", 0)),
                "restock_pct":      round(d.get("restock_pct", 0.0), 2),
                "reward":           round(d.get("reward", 0.0), 2),
                "t": now,
            }
            rb = d.get("rb")
            if rb is not None:
                entry["rb"] = {k: round(float(v), 2) for k, v in rb.items()}
            self.day_grades.append(entry)
        self._trim(self.day_grades)

    def update_status(self, **kwargs) -> None:
        """Overwrite-style status fields. Common keys: current_episode,
        n_episodes_target, current_stage, ticks, elapsed_s, alive."""
        self.status.update(kwargs)
        self.status["last_heartbeat"] = time.time()

    def write(self) -> None:
        """Atomic JSON write. Dashboard polls this file.

        Windows + concurrent reader gotcha: if the dashboard server happens
        to hold the JSON open for a read at the moment os.replace() fires,
        Windows raises PermissionError [WinError 5] and the trainer crashes.
        POSIX is fine with this; Windows isn't. We retry the rename a few
        times with brief sleeps — the reader's lock is typically held for
        single-digit ms, so a handful of 50ms backoffs covers it without
        materially delaying the trainer's write cadence.
        """
        import os, time as _time
        out = {
            "status": self.status,
            "ppo_updates": self.ppo_updates,
            "episodes": self.episodes,
            "windows": self.windows,
            "day_grades": self.day_grades,
            "n_distribution": self.n_distribution,
        }
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(out, f)
        last_err: Exception | None = None
        for attempt in range(8):
            try:
                tmp.replace(self.path)
                return
            except PermissionError as e:
                last_err = e
                _time.sleep(0.05 * (attempt + 1))   # 50ms, 100ms, ... ~1.8s total worst case
        # All retries failed — surface the original error rather than crashing
        # silently. Caller (training loop) already has try/except wrapping the
        # heartbeat; emergency checkpoint will fire.
        raise last_err if last_err else RuntimeError("metrics replace failed")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _trim(self, series: list) -> None:
        if len(series) > self.max_points:
            del series[:len(series) - self.max_points]

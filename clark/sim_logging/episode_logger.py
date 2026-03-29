"""
Episode logger for Clark.

Generalizes Jack's EpisodeLogger for use with any FacilityConfig.

Two logging modes:
  - "pretrain": lightweight. Records reward, grade, N/M, facility name per episode.
                No per-step data. Rolls over a window of recent episodes.
                Designed for long runs (10k+ episodes) without ballooning disk use.

  - "finetune": full detail. Records step assignments, debuffs, year snapshots,
                and facility config metadata. This is what the dashboard reads.

Usage:
    # Pre-training
    logger = EpisodeLogger(log_dir="clark/data/logs", mode="pretrain")
    logger.log_episode(summary, ep_num, facility_config)

    # Fine-tuning (full detail, dashboard-ready)
    logger = EpisodeLogger(log_dir="clark/data/logs/my_facility", mode="finetune",
                           facility_config=cfg)
    logger.log_episode(summary, ep_num)
    logger.write_year_snapshot(n_days, year_summary, facility_config)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from clark.sim_logging.log_schema import (
    episode_log_entry,
    notable_episode_entry,
    facility_config_full,
)

if TYPE_CHECKING:
    from clark.config.schema import FacilityConfig


# Rolling window size — how many episodes to keep in memory/on disk
ROLLING_WINDOW = 522   # ~2 full years of daily logs at 261 days/year


class EpisodeLogger:
    """
    Logs training episodes to disk for dashboard consumption.

    In "pretrain" mode, the facility config changes every episode, so compact
    per-episode metadata is stored with each entry. The rolling log shows the
    diversity of configs the foundation model has trained on.

    In "finetune" mode, the facility is fixed. Full step data and yearly
    snapshots are written so the dashboard can show worker-level detail.
    """

    def __init__(
        self,
        log_dir: str = "clark/data/logs",
        mode: str = "finetune",           # "pretrain" | "finetune"
        facility_config: Optional["FacilityConfig"] = None,
        rolling_window: int = ROLLING_WINDOW,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.facility_config = facility_config   # fixed for finetune, None for pretrain
        self.rolling_window = rolling_window

        self.log_path = self.log_dir / "episode_log.json"
        self.notable_dir = self.log_dir / "notable_episodes"
        self.notable_dir.mkdir(exist_ok=True)

        self.episodes: list[dict] = []

        # Notable tracking
        self.best_reward: float = float("-inf")
        self.worst_reward: float = float("inf")
        self.most_debuffs: int = 0
        self.first_perfect: Optional[dict] = None

        # Aggregate stats
        self.all_rewards: list[float] = []
        self.all_grades: list[str] = []
        self.ot_count: int = 0
        self.total_episodes: int = 0

        self._load_existing()

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load_existing(self):
        """Resume from a previous run if log file exists."""
        if not self.log_path.exists():
            return
        try:
            with open(self.log_path) as f:
                data = json.load(f)
            self.episodes = data.get("episodes", [])[-self.rolling_window:]
            for ep in self.episodes:
                reward = ep["footer"]["reward"]
                grade = ep["footer"]["grade"]
                ot = ep["footer"].get("ot_hours", 0.0)
                self.all_rewards.append(reward)
                self.all_grades.append(grade)
                self.total_episodes += 1
                if ot > 0:
                    self.ot_count += 1
                if reward > self.best_reward:
                    self.best_reward = reward
                if reward < self.worst_reward:
                    self.worst_reward = reward
                debuffs = len(ep["header"].get("debuffs_fired", []))
                if debuffs > self.most_debuffs:
                    self.most_debuffs = debuffs
                if (self.first_perfect is None
                        and ep["footer"].get("orders_remaining", 1) == 0
                        and ot == 0):
                    self.first_perfect = ep
            print(f"  [Logger] Loaded {len(self.episodes)} existing episodes from {self.log_path}")
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"  [Logger] Existing log unreadable — starting fresh.")
            self.episodes = []

    # ── Public API ────────────────────────────────────────────────────────────

    def log_episode(
        self,
        episode_summary: dict,
        episode_num: int,
        facility_config: Optional["FacilityConfig"] = None,
        write: bool = True,
    ):
        """
        Log a completed episode.

        Args:
            episode_summary: Output of FacilityEnv.get_episode_summary() or
                             YearEnv._get_year_summary() daily_summaries entry.
            episode_num:     Global episode counter.
            facility_config: Pass per-episode in pretrain mode (changes each ep).
                             In finetune mode, uses self.facility_config.
            write:           Write to disk immediately. Set False for batch writes.
        """
        cfg = facility_config or self.facility_config

        # In pretrain mode, strip steps to keep log size manageable
        if self.mode == "pretrain" and "steps" in episode_summary:
            episode_summary = {k: v for k, v in episode_summary.items() if k != "steps"}

        entry = episode_log_entry(episode_summary, episode_num, cfg)
        self.episodes.append(entry)

        # Maintain rolling window
        if len(self.episodes) > self.rolling_window:
            self.episodes = self.episodes[-self.rolling_window:]

        # Update aggregate stats
        footer = episode_summary["footer"]
        reward = footer["reward"]
        grade = footer["grade"]
        ot_hours = footer.get("ot_hours", 0.0)
        num_debuffs = len(episode_summary["header"].get("debuffs_fired", []))

        self.all_rewards.append(reward)
        self.all_grades.append(grade)
        self.total_episodes += 1
        if ot_hours > 0:
            self.ot_count += 1

        # Notable episode tracking
        if reward > self.best_reward:
            self.best_reward = reward
            self._save_notable(episode_summary, episode_num, "best_reward", cfg)

        if reward < self.worst_reward:
            self.worst_reward = reward
            self._save_notable(episode_summary, episode_num, "worst_reward", cfg)

        if num_debuffs > self.most_debuffs:
            self.most_debuffs = num_debuffs
            self._save_notable(episode_summary, episode_num, "most_debuffs", cfg)

        if (self.first_perfect is None
                and footer.get("orders_remaining", 1) == 0
                and ot_hours == 0):
            self.first_perfect = entry
            self._save_notable(episode_summary, episode_num, "first_perfect_day", cfg)

        if write:
            self._write_log()

    def write_year_snapshot(
        self,
        n_days: int,
        year_summary: Optional[dict] = None,
        facility_config: Optional["FacilityConfig"] = None,
    ):
        """
        Write the last n_days episodes (one complete year) to year_snapshot.json.

        The dashboard reads this file. Written at year-end so the dashboard
        always sees a complete year — never a partial view.

        Args:
            n_days:          Number of days in this year (usually ~260).
            year_summary:    Output of YearEnv._get_year_summary().
            facility_config: The facility config to embed in the snapshot.
        """
        cfg = facility_config or self.facility_config
        year_episodes = (
            self.episodes[-n_days:] if n_days <= len(self.episodes)
            else self.episodes
        )

        output = {
            "episodes": year_episodes,
            "training_stats": self.get_training_stats(),
            "year_summary": year_summary or {},
            "facility_config": facility_config_full(cfg) if cfg else {},
        }

        snapshot_path = self.log_dir / "year_snapshot.json"
        with open(snapshot_path, "w") as f:
            json.dump(output, f, indent=2)

    def get_training_stats(self) -> dict:
        """Aggregate stats over the rolling window."""
        window = self.rolling_window
        recent_rewards = self.all_rewards[-window:]
        recent_grades = self.all_grades[-window:]

        win_count = sum(1 for g in recent_grades if g in ("A", "B"))
        avg_reward = sum(recent_rewards) / max(1, len(recent_rewards))
        recent_ot = sum(
            1 for ep in self.episodes
            if ep["footer"].get("ot_hours", 0.0) > 0
        )

        return {
            "total_episodes": self.total_episodes,
            "avg_reward_last_window": round(avg_reward, 2),
            "win_rate_last_window": round(win_count / max(1, len(recent_grades)), 3),
            "ot_frequency": round(recent_ot / max(1, len(self.episodes)), 3),
            "best_reward_ever": round(self.best_reward, 2) if self.best_reward != float("-inf") else None,
            "worst_reward_ever": round(self.worst_reward, 2) if self.worst_reward != float("inf") else None,
            "grade_distribution": {
                g: recent_grades.count(g) for g in ("A", "B", "C", "D", "F")
            },
            "mode": self.mode,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save_notable(
        self,
        episode_summary: dict,
        episode_num: int,
        reason: str,
        facility_config=None,
    ):
        entry = notable_episode_entry(episode_summary, episode_num, reason, facility_config)
        path = self.notable_dir / f"{reason}.json"
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)

    def _write_log(self):
        output = {
            "episodes": self.episodes,
            "training_stats": self.get_training_stats(),
        }
        with open(self.log_path, "w") as f:
            json.dump(output, f, indent=2)

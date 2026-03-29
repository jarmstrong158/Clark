"""
Log schema helpers for Clark episode logging.

Episode log structure:
  {
    "episode": int,
    "facility": { name, n_workers, n_tasks, task_ids },   ← facility metadata
    "header":  { date, season, total_orders, debuffs, picker, is_high_volume },
    "footer":  { orders_shipped, reward, grade, ot_hours, ... },
    "steps":   [ { step, assignments: [{worker, task, hustle}], ... } ],  ← optional
  }

Year snapshot structure:
  {
    "episodes":       [ ...episode entries for this year... ],
    "training_stats": { total_episodes, avg_reward, grade_dist, ... },
    "year_summary":   { year_grade, grade_distribution, cycle_counts, ... },
    "facility_config": { name, workers, tasks, rules_summary },
  }
"""
from __future__ import annotations


def facility_metadata(facility_config) -> dict:
    """Compact facility metadata to embed in each log entry."""
    return {
        "name": facility_config.name,
        "n_workers": facility_config.num_workers,
        "n_tasks": facility_config.num_tasks,
        "task_ids": facility_config.task_ids,
        "workers": [
            {
                "id": w.worker_id,
                "name": w.name,
                "role": w.role,
                "base_oph": w.base_oph,
            }
            for w in facility_config.workers
        ],
    }


def facility_config_full(facility_config) -> dict:
    """Full facility config snapshot for year-level log (dashboard config panel)."""
    rules = facility_config.rules
    return {
        "facility": {
            "name": facility_config.name,
            "timezone": facility_config.timezone,
        },
        "workers": [
            {
                "id": w.worker_id,
                "name": w.name,
                "role": w.role,
                "base_oph": w.base_oph,
                "shift_hours": w.shift_hours,
                "task_eligibility": w.task_eligibility,
                "task_oph_overrides": w.task_oph_overrides or {},
            }
            for w in facility_config.workers
        ],
        "tasks": {
            "enabled": facility_config.tasks.enabled,
        },
        "volume": {
            "seasonal_ranges": {
                k: list(v) for k, v in facility_config.volume.seasonal_ranges.items()
            },
        },
        "rules_summary": {
            "day_start_hour": rules.day_start_hour,
            "eod_hour": rules.eod_hour,
            "lunch_hour": rules.lunch_hour,
            "lunch_duration": rules.lunch_duration,
            "morning_pick_enabled": rules.morning_pick_enabled,
            "carrier_pickup_hour": rules.carrier_pickup_hour,
            "ot_wall_clock_max": rules.ot_wall_clock_max,
            "management_daily_hours_required": rules.management_daily_hours_required,
            "cycle_count_weekly_hours": rules.cycle_count_weekly_hours,
            "cycle_count_max_overdue_weeks": rules.cycle_count_max_overdue_weeks,
            "high_volume_day_orders": rules.high_volume_day_orders,
            "pack_stations": rules.pack_stations,
            "carts_available": rules.carts_available,
        },
        "order_complexity": [
            {"name": t.name, "weight": t.weight, "oph_multiplier": t.oph_multiplier}
            for t in facility_config.order_complexity.tiers
        ],
    }


def episode_log_entry(
    episode_summary: dict,
    episode_num: int,
    facility_config=None,
) -> dict:
    """Build a complete log entry from an episode summary."""
    entry = {
        "episode": episode_num,
        "header": episode_summary["header"],
        "footer": episode_summary["footer"],
        "steps": episode_summary.get("steps", []),
    }
    if facility_config is not None:
        entry["facility"] = facility_metadata(facility_config)
    return entry


def notable_episode_entry(
    episode_summary: dict,
    episode_num: int,
    reason: str,
    facility_config=None,
) -> dict:
    """Build a notable episode log entry with a reason tag."""
    entry = episode_log_entry(episode_summary, episode_num, facility_config)
    entry["notable_reason"] = reason
    return entry

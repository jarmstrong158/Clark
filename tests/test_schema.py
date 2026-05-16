"""FacilityConfig schema — validation rules + YAML round-trip.

The wizard (cmd_wizard) and finetune both depend on validate()
correctly rejecting broken configs. If validation silently passes a
bad config, a user burns hours of fine-tune compute before finding
out. These tests pin the error/warning contract.
"""
from __future__ import annotations

import copy

import pytest

from clark.config.schema import FacilityConfig


# ── Example configs are valid ─────────────────────────────────────────────────

def test_small_example_is_valid(small_config):
    errors, _warnings = small_config.validate()
    assert errors == [], f"example_small.yaml should be valid, got: {errors}"


def test_large_example_is_valid(large_config):
    errors, _warnings = large_config.validate()
    assert errors == [], f"example_large.yaml should be valid, got: {errors}"


def test_derived_counts_match_workers(small_config):
    assert small_config.num_workers == len(small_config.workers)
    assert small_config.num_tasks == len(small_config.task_ids)


# ── Validation catches structural breakage ────────────────────────────────────

def test_too_few_workers_is_error(small_config):
    cfg = copy.deepcopy(small_config)
    cfg.workers = cfg.workers[:1]  # 1 worker
    errors, _ = cfg.validate()
    assert any("at least 2 workers" in e.lower() for e in errors)


def test_non_contiguous_worker_ids_is_error(small_config):
    cfg = copy.deepcopy(small_config)
    cfg.workers[0].worker_id = 99  # break 0-indexed contiguity
    errors, _ = cfg.validate()
    assert any("0-indexed" in e or "unique" in e for e in errors)


def test_duplicate_worker_ids_is_error(small_config):
    cfg = copy.deepcopy(small_config)
    cfg.workers[1].worker_id = cfg.workers[0].worker_id
    errors, _ = cfg.validate()
    assert any("unique" in e.lower() for e in errors)


def test_too_many_tasks_is_error(small_config):
    cfg = copy.deepcopy(small_config)
    # task_ids is a derived property (tasks.enabled + custom + core).
    # Force it over the model's 20-task embedding cap via tasks.enabled.
    cfg.tasks.enabled = [f"extra_task_{i}" for i in range(25)]
    assert cfg.num_tasks > 20  # sanity: the derivation actually grew
    errors, _ = cfg.validate()
    assert any("embedding capacity" in e or "exceeds" in e for e in errors)


def test_shift_window_too_short_is_error(small_config):
    cfg = copy.deepcopy(small_config)
    cfg.rules.eod_hour = cfg.rules.day_start_hour + 2  # < 4h shift
    errors, _ = cfg.validate()
    assert any("4-hour shift" in e or "day_start_hour + 4" in e for e in errors)


def test_lunch_outside_shift_is_error(small_config):
    cfg = copy.deepcopy(small_config)
    cfg.rules.lunch_hour = cfg.rules.eod_hour + 1  # lunch after EOD
    errors, _ = cfg.validate()
    assert any("lunch_hour" in e for e in errors)


def test_extreme_oph_is_warning_not_error(small_config):
    cfg = copy.deepcopy(small_config)
    cfg.workers[0].base_oph = 0.5  # below typical [1, 50] but not over hard cap
    errors, warnings = cfg.validate()
    assert any("base_oph" in w for w in warnings)


# ── YAML round-trip ───────────────────────────────────────────────────────────

def test_from_yaml_then_validate_clean(tmp_path, small_config):
    """Loading a config from YAML must produce the same validity verdict
    as the in-memory object — the wizard writes YAML then fine-tunes."""
    import yaml

    src = (
        __import__("pathlib").Path(__file__).parent.parent
        / "clark" / "data" / "configs" / "example_small.yaml"
    )
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    rt_path = tmp_path / "roundtrip.yaml"
    rt_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    reloaded = FacilityConfig.from_yaml(rt_path)
    errors, _ = reloaded.validate()
    assert errors == [], f"round-tripped config should still validate: {errors}"
    assert reloaded.num_workers == small_config.num_workers

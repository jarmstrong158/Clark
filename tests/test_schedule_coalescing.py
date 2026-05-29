"""Schedule block coalescing must not misattribute time.

Regression for the bug where a capped task (e.g. cycle_count limited to
0.5h/day) rendered as hours: the old coalescer folded every sub-20-min
block into the *previous* block under the previous block's task label, so
post-cap task-thrashing got painted as the pre-cap task.

The fix: only merge same-task runs and collapse true A-b-A one-tick
flicker — never relabel sustained churn.
"""
from __future__ import annotations

from clark.inference.plan import _coalesce_blocks

MIN_H = 20 / 60.0  # 20-minute minimum block, matching run_full_day_schedule


def _blk(start, end, task, hustle=False):
    return {"start": start, "end": end, "task": task, "hustle": hustle}


def _dur(blocks, task):
    return sum(b["end"] - b["start"] for b in blocks if b["task"] == task)


def test_capped_task_not_inflated_by_following_churn():
    # cycle_count for 0.5h, then the policy thrashes one tick each across
    # several other tasks (each 10 min < 20 min minimum).
    blocks = [
        _blk(8.0, 8.5, "cycle_count"),
        _blk(8.5, 8.6667, "pick"),
        _blk(8.6667, 8.8333, "pack"),
        _blk(8.8333, 9.0, "management"),
        _blk(9.0, 9.1667, "pick"),
        _blk(9.1667, 9.3333, "restock"),
    ]
    out = _coalesce_blocks(blocks, MIN_H)
    # cycle_count must stay ~0.5h, NOT absorb the trailing churn.
    assert abs(_dur(out, "cycle_count") - 0.5) < 1e-6, (
        f"cycle_count inflated to {_dur(out, 'cycle_count'):.2f}h by following churn"
    )
    # The first block is still cycle_count ending at 8.5 (not extended).
    assert out[0]["task"] == "cycle_count" and abs(out[0]["end"] - 8.5) < 1e-6


def test_adjacent_same_task_runs_merge():
    blocks = [
        _blk(8.0, 8.1667, "pick"),
        _blk(8.1667, 8.3333, "pick"),
        _blk(8.3333, 8.5, "pick"),
    ]
    out = _coalesce_blocks(blocks, MIN_H)
    assert len(out) == 1
    assert out[0]["task"] == "pick"
    assert abs(out[0]["start"] - 8.0) < 1e-6 and abs(out[0]["end"] - 8.5) < 1e-6


def test_one_tick_flicker_between_same_task_is_smoothed():
    # pick → (10-min management blip) → pick  should render as one pick block.
    blocks = [
        _blk(8.0, 8.5, "pick"),
        _blk(8.5, 8.6667, "management"),
        _blk(8.6667, 9.5, "pick"),
    ]
    out = _coalesce_blocks(blocks, MIN_H)
    assert len(out) == 1, f"expected A-b-A flicker collapsed, got {out}"
    assert out[0]["task"] == "pick"
    assert abs(out[0]["end"] - 9.5) < 1e-6


def test_distinct_long_blocks_preserved():
    blocks = [
        _blk(8.0, 10.0, "pick"),
        _blk(10.0, 12.0, "pack"),
        _blk(12.0, 14.0, "restock"),
    ]
    out = _coalesce_blocks(blocks, MIN_H)
    assert [b["task"] for b in out] == ["pick", "pack", "restock"]


def test_zero_duration_blocks_dropped():
    blocks = [
        _blk(8.0, 8.0, "pick"),       # zero-duration (snap collapse)
        _blk(8.0, 9.0, "pack"),
    ]
    out = _coalesce_blocks(blocks, MIN_H)
    assert [b["task"] for b in out] == ["pack"]

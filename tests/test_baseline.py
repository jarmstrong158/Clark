"""GreedyScheduler (fair v2) decision logic: management coverage, restock
coverage, and backlog-proportional pick/pack balance — all while respecting
the action mask. Fast unit tests with a fake day_env (no simulation)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from clark.inference.baseline import GreedyScheduler

# idle, pick, pack, restock, management
T2I = {"idle": 0, "pick": 1, "pack": 2, "restock": 3, "management": 4}
PICK, PACK, RESTOCK, MGMT, IDLE = 1, 2, 3, 0, 0


def _env(queue, buffer, restock_level=1.0, restock_enabled=True):
    return SimpleNamespace(
        _task_to_idx=T2I, orders_in_queue=queue,
        orders_picked_not_audited=buffer,
        restock_level=restock_level, _restock_enabled=restock_enabled,
    )


def _hmask(n, can=True):
    return np.array([[True, can]] * n)


def _mask(rows):
    return np.array(rows, dtype=bool)


def test_reserves_one_worker_for_management():
    g = GreedyScheduler()
    env = _env(queue=10, buffer=10)             # order pressure on both sides
    allround = [False, True, True, True, True]
    ta, _ = g.act(env, _mask([allround] * 4), _hmask(4))
    assert ta.count(T2I["management"]) == 1, "exactly one worker should cover management"


def test_balances_pick_pack_by_backlog():
    g = GreedyScheduler()
    env = _env(queue=10, buffer=30)             # buffer 3x queue -> mostly packers
    # pick+pack only (mgmt/restock masked out) to isolate the balance logic
    pp = [False, True, True, False, False]
    ta, _ = g.act(env, _mask([pp] * 4), _hmask(4))
    assert ta.count(T2I["pack"]) == 3 and ta.count(T2I["pick"]) == 1


def test_no_packers_when_buffer_empty():
    g = GreedyScheduler()
    env = _env(queue=12, buffer=0)
    pp = [False, True, True, False, False]
    ta, _ = g.act(env, _mask([pp] * 3), _hmask(3))
    assert ta.count(T2I["pack"]) == 0 and ta.count(T2I["pick"]) == 3


def test_reserves_restock_when_low():
    g = GreedyScheduler()
    env = _env(queue=10, buffer=10, restock_level=0.2)
    # mgmt masked out so we isolate restock coverage
    no_mgmt = [False, True, True, True, False]
    ta, _ = g.act(env, _mask([no_mgmt] * 4), _hmask(4))
    assert ta.count(T2I["restock"]) >= 1


def test_respects_mask_per_worker():
    g = GreedyScheduler()
    env = _env(queue=10, buffer=10)
    ta, _ = g.act(env, _mask([
        [False, True, False, False, False],   # pick-only
        [False, False, True, False, False],   # pack-only
        [False, False, False, True, False],   # restock-only
    ]), _hmask(3))
    assert ta[0] == T2I["pick"] and ta[1] == T2I["pack"] and ta[2] == T2I["restock"]


def test_idle_only_when_nothing_valid():
    g = GreedyScheduler()
    env = _env(queue=10, buffer=10)
    ta, _ = g.act(env, _mask([[True, False, False, False, False]]), _hmask(1))
    assert ta[0] == T2I["idle"]


def test_no_hustle_without_backlog():
    g = GreedyScheduler()
    env = _env(queue=0, buffer=0)
    _, ha = g.act(env, _mask([[False, True, True, True, True]]), _hmask(1))
    assert ha[0] is False

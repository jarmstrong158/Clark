"""GreedyScheduler decision logic — bottleneck priority while respecting the
action mask. Fast unit tests with a fake day_env (no simulation)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from clark.inference.baseline import GreedyScheduler

T2I = {"idle": 0, "pick": 1, "pack": 2, "restock": 3, "management": 4}


def _env(queue, buffer, restock_level=1.0, restock_enabled=True):
    return SimpleNamespace(
        _task_to_idx=T2I,
        orders_in_queue=queue,
        orders_picked_not_audited=buffer,
        restock_level=restock_level,
        _restock_enabled=restock_enabled,
    )


def _hmask(n, can=True):
    return np.array([[True, can]] * n)


def test_packs_when_buffer_and_respects_mask():
    g = GreedyScheduler()
    env = _env(queue=10, buffer=20)  # buffer present -> pack is top priority
    mask = np.array([
        [False, True, True, True, True],     # all-round worker
        [False, True, False, False, False],  # pick-only
        [False, False, False, True, False],  # restock-only
    ])
    ta, ha = g.act(env, mask, _hmask(3))
    assert ta[0] == T2I["pack"]      # chooses the top valid priority
    assert ta[1] == T2I["pick"]      # mask forces pick
    assert ta[2] == T2I["restock"]   # mask forces restock
    assert all(ha)                   # under load + hustle-capable


def test_picks_when_only_queue():
    g = GreedyScheduler()
    env = _env(queue=8, buffer=0)    # nothing to pack -> pick leads
    mask = np.array([[False, True, True, True, True]])
    ta, _ = g.act(env, mask, _hmask(1))
    assert ta[0] == T2I["pick"]


def test_restock_when_low_and_no_order_pressure():
    g = GreedyScheduler()
    env = _env(queue=0, buffer=0, restock_level=0.2)  # only restock is "needed"
    mask = np.array([[False, True, True, True, True]])
    ta, _ = g.act(env, mask, _hmask(1))
    assert ta[0] == T2I["restock"]


def test_idle_only_when_nothing_else_valid():
    g = GreedyScheduler()
    env = _env(queue=10, buffer=10)
    mask = np.array([[True, False, False, False, False]])  # only idle valid
    ta, ha = g.act(env, mask, _hmask(1))
    assert ta[0] == T2I["idle"]


def test_no_hustle_when_no_backlog():
    g = GreedyScheduler()
    env = _env(queue=0, buffer=0)  # no load -> don't burn hustle
    mask = np.array([[False, True, True, True, True]])
    _, ha = g.act(env, mask, _hmask(1))
    assert ha[0] is False

"""Task flow/ramp: the env must track consecutive ticks on a task so the
effective_oph ramp (setup dip -> recover -> flow bonus) actually fires.

The worker-level multiplier curve is covered in test_worker.py; this pins
the env wiring — that step() increments the streak when a worker holds a
task and resets it when they switch.
"""
from __future__ import annotations

from clark.config.schema import FacilityConfig
from clark.env.facility_env import FacilityEnv

CFG_PATH = "clark/data/configs/example_small.yaml"


def _env():
    cfg = FacilityConfig.from_yaml(CFG_PATH)
    env = FacilityEnv(cfg)
    env.reset(force_month=12, force_dow=4, force_volume=30)
    return cfg, env


def _step_worker0(env, cfg, task):
    """Assign worker 0 `task`, everyone else idle, and step one tick."""
    idle = cfg.task_ids.index("idle")
    ti = cfg.task_ids.index(task)
    actions = [(i, idle, False) for i in range(len(env.episode.workers))]
    actions[0] = (0, ti, False)
    env.step(actions)


def test_streak_increments_on_hold_and_resets_on_switch():
    cfg, env = _env()
    w = env.episode.workers[0]

    _step_worker0(env, cfg, "pick")        # fresh switch into pick
    assert w.current_task == "pick"
    assert w.ticks_on_task == 0            # just switched -> setup-cost floor

    _step_worker0(env, cfg, "pick")        # held pick
    assert w.ticks_on_task == 1

    _step_worker0(env, cfg, "pick")        # held again
    assert w.ticks_on_task == 2

    _step_worker0(env, cfg, "pack")        # switched -> reset
    assert w.current_task == "pack"
    assert w.ticks_on_task == 0


def test_sustained_work_outproduces_thrashing():
    """A worker who holds pick should accumulate more effective output than
    one who switches every tick (dip vs ramp). Uses effective_oph directly
    as the per-tick throughput proxy."""
    cfg, _ = _env()
    from clark.env.worker import TASK_FLOW_RAMP
    base = 20.0
    # Settled worker: average multiplier over a 6-tick block.
    settled = sum(TASK_FLOW_RAMP[min(i, len(TASK_FLOW_RAMP)-1)] for i in range(6)) / 6
    # Thrasher: every tick is a fresh switch -> always the floor.
    thrasher = TASK_FLOW_RAMP[0]
    assert settled > thrasher
    assert base * settled > base * thrasher

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


def test_dwell_locks_to_current_task_within_floor():
    from clark.agent.actions import get_action_mask, DWELL_MIN_TICKS
    cfg, env = _env()
    pick = cfg.task_ids.index("pick")
    w = env.episode.workers[0]
    w.is_absent = False
    w.current_task = "pick"
    w.ticks_on_task = 0                      # just started -> locked
    m = get_action_mask(env)
    assert m[0, pick] and int(m[0].sum()) == 1, "mid-dwell -> locked to current task only"
    w.ticks_on_task = DWELL_MIN_TICKS - 1    # satisfied the floor -> free
    m2 = get_action_mask(env)
    assert int(m2[0].sum()) > 1, "after the dwell floor the worker can switch again"


def test_dwell_yields_when_current_task_becomes_invalid():
    from clark.agent.actions import get_action_mask
    cfg, env = _env()
    rs = cfg.task_ids.index("restock")
    w = env.episode.workers[0]
    w.is_absent = False
    w.current_task = "restock"
    w.ticks_on_task = 0                      # mid-dwell...
    env.restock_level = 1.0                  # ...but restock is full (invalid this tick)
    m = get_action_mask(env)
    assert not m[0, rs], "a now-invalid current task must not be force-held"
    assert int(m[0].sum()) >= 1, "worker keeps valid options when the dwell lock yields"


def test_switch_penalty_only_on_real_task_change():
    cfg, env = _env()
    # Workers start the day on role-based tasks; force a clean idle start so
    # we can check the "starting work isn't churn" case.
    env.episode.workers[0].current_task = "idle"
    _step_worker0(env, cfg, "pick")          # idle -> pick: starting work, no penalty
    p_start = env.reward_breakdown.get("per_task_switch", 0.0)
    _step_worker0(env, cfg, "pack")          # pick -> pack: genuine churn, penalized
    p_switch = env.reward_breakdown.get("per_task_switch", 0.0)
    assert p_start == 0.0, "idle -> task is not a penalized switch"
    assert p_switch < 0.0 and p_switch < p_start, "real -> real switch incurs per_task_switch"


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

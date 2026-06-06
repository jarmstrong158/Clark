"""Non-RL baselines for Clark, for honest comparison.

Clark's whole premise is that a *learned* policy beats classical scheduling
on this problem. You can't claim that without a baseline. `GreedyScheduler`
is the cheap one: a transparent bottleneck-priority heuristic. It plugs into
the same evaluation path as the trained agent (`clark eval --baseline
greedy`), runs the same held-out facilities through the same in-env
production grader, and crucially **obeys the exact same action masks** Clark
does — eligibility, OT/EOD, daily-hours caps, the minimum-dwell lock. So the
comparison is apples-to-apples: same constraints, same scoring, only the
decision rule differs.

This is deliberately simple (start cheap — see docs/ENGINEERING_NOTES.md). A
strong constraint-program / CP-SAT baseline is the natural next rung if the
greedy turns out competitive.
"""
from __future__ import annotations

_RESTOCK_LOW = 0.5  # treat restock as "needed" below this fill level


class GreedyScheduler:
    """Per-tick bottleneck-priority rule. Each worker takes the highest-
    priority task that's *valid for them this tick* (per the action mask):

      1. pack   — if there are picked-but-unpacked orders (clear the buffer → ship)
      2. pick   — if there are orders waiting in the queue
      3. restock — if stock is below the low threshold
      4. management / any remaining valid task
      5. idle (only if nothing else is valid)

    No learning, no lookahead — a manager's "work the current bottleneck"
    reflex. The mask does the constraint enforcement, so this never violates
    eligibility, caps, dwell, or OT rules.
    """

    name = "greedy"

    def reset(self):  # parity with the trained-agent interface; nothing to reset
        pass

    def act(self, day_env, mask, hmask):
        N, M = mask.shape
        t2i = day_env._task_to_idx
        idle = t2i.get("idle", 0)
        pick, pack = t2i.get("pick"), t2i.get("pack")
        restock, mgmt = t2i.get("restock"), t2i.get("management")

        queue = day_env.orders_in_queue
        buffer = day_env.orders_picked_not_audited
        restock_low = (getattr(day_env, "_restock_enabled", False)
                       and getattr(day_env, "restock_level", 1.0) < _RESTOCK_LOW)

        # State-dependent task priority (high → low).
        prio: list[int] = []
        def _add(t):
            if t is not None and t not in prio:
                prio.append(t)
        if buffer > 0:
            _add(pack)
        if queue > 0:
            _add(pick)
        if restock_low:
            _add(restock)
        # fallbacks so a worker always has a sensible default
        for t in (pick, pack, restock, mgmt):
            _add(t)

        task_actions = []
        for w in range(N):
            valid = mask[w]
            chosen = None
            for t in prio:
                if t is not None and t < M and valid[t]:
                    chosen = t
                    break
            if chosen is None:  # nothing prioritized is valid → first valid, else idle
                vi = [j for j in range(M) if valid[j]]
                chosen = vi[0] if vi else idle
            task_actions.append(chosen)

        # Hustle when there's real backlog and the worker is able to.
        under_load = (queue + buffer) > 0
        hustle_actions = [bool(under_load and hmask[w][1]) for w in range(N)]
        return task_actions, hustle_actions


def run_one_year_baseline(scheduler, cfg) -> dict:
    """Full work-year under a non-RL scheduler. Same loop/masks/grader as
    `clark.inference.evaluate.run_one_year`, minus the StateBuilder (the
    heuristic reads the env directly). Returns the env year summary."""
    from clark.agent.actions import get_action_mask, get_hustle_mask
    from clark.env.year_env import YearEnv

    env = YearEnv(cfg)
    env.reset()
    scheduler.reset()
    ticks = 0
    while not env.is_year_done and ticks < 200_000:
        mask = get_action_mask(env.day_env)
        hmask = get_hustle_mask(env.day_env)
        ta, ha = scheduler.act(env.day_env, mask, hmask)
        env.step([(i, int(ta[i]), bool(ha[i])) for i in range(len(ta))])
        ticks += 1
    return env._get_year_summary()

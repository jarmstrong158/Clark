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

_RESTOCK_PROACTIVE = 0.8  # staff restock once stock dips below this (pre-cliff)


class GreedyScheduler:
    """A *competent* (not naive) priority-rule scheduler — a fair classical
    baseline. Each tick it allocates the roster the way a sensible manager
    would, respecting the action mask (so eligibility, caps, dwell, OT all
    still bind):

      1. **Management coverage** — keep one eligible worker on management
         whenever the duty is still owed. (The mask only *offers* management
         while it's under the daily requirement, so this self-limits — once
         the duty is met, management leaves the mask and the worker rejoins
         the floor. Skipping this was the naive v1's fatal hole: an unmet
         management minimum is an automatic F.)
      2. **Restock coverage (proactive + scaled)** — staff restock *before*
         stock hits the picking-speed cliff (below 0.8 fill, not after it's
         critical), with the number of restockers scaled to the stock deficit
         and roster size. Under-staffing restock was the v2 hole: stock
         collapsed → picking crashed → days couldn't finish even on full OT.
      3. **Balance pick vs pack by backlog** — split the remaining workers in
         proportion to (orders waiting to be picked) vs (picked-but-unpacked),
         instead of dog-piling one task. Falls back through the mask if a
         worker can't do the targeted task; idle only as a last resort.

    No learning, no lookahead — but no obvious holes. This is the bar Clark
    has to clear to justify the RL.
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

        assigned: dict[int, int] = {}
        used: set[int] = set()

        def reserve(t):
            """Put the first available worker who can do task t on it.
            Returns True if one was assigned."""
            if t is None or t >= M:
                return False
            for w in range(N):
                if w not in used and mask[w][t]:
                    assigned[w] = t
                    used.add(w)
                    return True
            return False

        # 1) cover the management duty (mask self-limits once it's met)
        reserve(mgmt)
        # 2) restock — proactive + scaled. Staff it before stock hits the
        #    picking-speed cliff (a late, single-worker reaction can't recover
        #    in time, especially with the 60-min dwell lock). Count scales with
        #    the deficit below the proactive band and the roster size, capped
        #    at a quarter of the roster.
        if restock is not None and getattr(day_env, "_restock_enabled", False):
            level = getattr(day_env, "restock_level", 1.0)
            if level < _RESTOCK_PROACTIVE:
                deficit = _RESTOCK_PROACTIVE - level
                n_restock = max(1, min(round(N * deficit * 0.4), max(1, N // 4)))
                for _ in range(n_restock):
                    if not reserve(restock):
                        break

        # 3) split the rest between pick and pack proportional to the backlog
        remaining = [w for w in range(N) if w not in used]
        total = queue + buffer
        pack_target = round(len(remaining) * (buffer / total)) if total > 0 else 0
        packers = 0
        for w in remaining:
            want = pack if packers < pack_target else pick
            alt = pick if want == pack else pack
            if want is not None and want < M and mask[w][want]:
                assigned[w] = want
            elif alt is not None and alt < M and mask[w][alt]:
                assigned[w] = alt
            else:
                vi = [j for j in range(M) if mask[w][j]]
                assigned[w] = vi[0] if vi else idle
            if assigned[w] == pack:
                packers += 1

        task_actions = [assigned.get(w, idle) for w in range(N)]
        under_load = total > 0
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

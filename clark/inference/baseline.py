"""Non-RL baselines for Clark, for honest comparison.

Clark's whole premise is that a *learned* policy beats classical scheduling
on this problem. You can't claim that without a baseline. `GreedyScheduler`
is a **deliberately strong** bottleneck-priority heuristic — iterated under
audit (v1 naive → v5; see docs/ENGINEERING_NOTES.md §9) until it stopped
losing to its own structure, not a strawman. It plugs into the same
evaluation path as the trained agent (`clark eval --baseline heuristic`,
alias `greedy`), runs the same held-out facilities through the same in-env
production grader, and crucially **obeys the exact same action masks** Clark
does — eligibility, OT/EOD, daily-hours caps, the minimum-dwell lock. So the
comparison is apples-to-apples: same constraints, same scoring, only the
decision rule differs.

It turned out competitive: on 20 held-out stage-3 facilities this heuristic
*ties* Clark on A+B (98.3 vs 97.5) and completion (100%); Clark's edge is
overtime avoidance, tail robustness, and zero-per-facility-tuning. The
constraint-program rung that "if competitive" anticipated was then built as a
CP-SAT *completion bound* (clark/inference/optimizer.py, ENGINEERING_NOTES
§10) — it showed throughput is never the binding constraint.
"""
from __future__ import annotations

_RESTOCK_PROACTIVE = 0.95  # staff restock below this — matches the grade's 95%
#                            restock-fill threshold (targeting 0.8 parked stock
#                            right in the demerit zone; see ENGINEERING_NOTES §10)
_BUFFER_HEALTHY_FLOOR = 8  # buffer above max(this, roster) -> only trickle pickers
_PICK_TRICKLE_DIV = 6      # healthy-buffer pickers ≈ remaining // this
_PICK_REFILL_FRAC = 0.30   # buffer thinning -> ramp pickers to this frac of remaining
_HUSTLE_CRUNCH_FRAC = 0.6  # hustle only when (queue+buffer) exceeds this * roster


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
         stock hits the picking-speed cliff, with the number of restockers
         scaled to the stock deficit and roster size. Under-staffing restock
         was the v2 hole: stock collapsed → picking crashed → days couldn't
         finish even on full OT. The target tracks the grade's 95% restock-fill
         line (v5): parking stock at 80% sat right inside the demerit band and
         cost A-grades on otherwise-clean days.
      3. **Pack-bottleneck-aware pick/pack split** — picking runs at ~2.5×
         pack speed (PICK_MULTIPLIER vs pack's 1.0), and the morning-pick
         round front-loads the buffer, so **pack is the perennial constraint**
         and daily throughput ≈ pack capacity. Balancing by *backlog size*
         (v3) over-chased the big WIP buffer with packers while still missing
         days. Instead: keep only enough pickers to keep the buffer non-empty
         (trickle when it's healthy, ramp when it's thinning), and throw
         everyone else at pack to drain WIP. Audited A/B: A+B 78→90, F 21→9.
      4. **Concentrate hustle on the crunch** — the daily hustle budget is
         capped, so hustling whenever *any* backlog exists (v3) burned it out
         early and left nothing for the afternoon pack crunch. Hustle only
         when the pile (queue+buffer) is large relative to the roster.

    No learning, no lookahead — but no obvious holes, and grounded in the
    sim's actual bottleneck. This is the bar Clark has to clear.
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

        # 3) pack-bottleneck-aware split of the rest. Pack is the constraint
        #    (2.5x slower than pick), so size pickers only to keep the buffer
        #    fed; everyone else packs to drain WIP.
        remaining = [w for w in range(N) if w not in used]
        R = len(remaining)
        if queue <= 0:                       # nothing left to pick -> all pack
            target_pick = 0
        elif buffer <= 0:                    # buffer empty -> refill hard
            target_pick = R
        elif buffer > max(_BUFFER_HEALTHY_FLOOR, R):   # healthy buffer -> trickle
            target_pick = max(1, R // _PICK_TRICKLE_DIV)
        else:                                # buffer thinning -> ramp pickers
            target_pick = max(1, round(R * _PICK_REFILL_FRAC))

        pickers = 0
        for w in remaining:
            want = pick if pickers < target_pick else pack
            alt = pack if want == pick else pick
            if want is not None and want < M and mask[w][want]:
                assigned[w] = want
            elif alt is not None and alt < M and mask[w][alt]:
                assigned[w] = alt
            else:
                vi = [j for j in range(M) if mask[w][j]]
                assigned[w] = vi[0] if vi else idle
            if assigned[w] == pick:
                pickers += 1

        task_actions = [assigned.get(w, idle) for w in range(N)]
        # 4) concentrate the capped hustle budget on the crunch
        crunch = (queue + buffer) > N * _HUSTLE_CRUNCH_FRAC
        hustle_actions = [bool(crunch and hmask[w][1]) for w in range(N)]
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

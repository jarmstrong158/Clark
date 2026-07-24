# Engineering notes — what building Clark actually taught

Honest, transferable lessons from building Clark, written down because the
*reasoning* (and the times the data proved a confident assumption wrong)
is worth more than the final numbers. Milestone history lives in
[CHANGELOG.md](../CHANGELOG.md); this is the "what I'd tell the next person"
version.

---

## 1. Constrain what must be true; shape what you want to optimize

The biggest recurring lesson, learned twice the hard way.

- **Filler-during-crunch (v2.5):** two attempts to fix it with *reward
  shaping* (a new observation feature, then a reward term) both regressed.
  A **structural action mask** (zero out filler tasks under stress gates)
  fixed it immediately.
- **Task thrashing (v2.11):** the policy switched tasks almost every 10-min
  tick. I added a *productivity ramp* (sustained work → higher effective
  output → more reward) — a clean, implicit incentive. It moved the cadence
  **~zero** after 500 episodes of training. The fix was again **structural**:
  a minimum-dwell mask that locks a worker to a task for 60 min.

The principle: a **hard constraint** ("a worker can't context-switch every
10 minutes", "filler is forbidden when we're drowning") should be *enforced
in the action space*, not *learned via reward*. Reward is for **soft
objectives** you want optimized (ship more, finish the day). Trying to teach
a hard rule through reward is fragile and slow; masking it is reliable and
free. The mature instinct is to ask "is this a rule or a preference?" first.

**Corollary — why the soft incentive lost:** the PPO **entropy bonus**
actively rewards spreading probability across actions every tick. A diffuse,
delayed signal (a switch costs a little future throughput) cannot beat an
immediate, dense one (entropy paid every step). If you *must* shape, the
signal has to be immediate and creditable — otherwise the structural route
wins.

## 2. Inference-time masks are model-agnostic and free

The dwell mask cut switching from ~29 to ~14 per worker/day **at serve time,
on any checkpoint, with no retraining** — because a mask is applied in
`get_action_mask`, not baked into weights. I burned ~6.5 h of GPU on two
retrains chasing the cadence before realizing the structural mask already
did the job and the *training* added nothing. Before you retrain to change
behavior, check whether a constraint at inference gets you there for free.

## 3. Measure distributions, not point estimates — and trust the real grader

- An early head-to-head probe reported two checkpoints "essentially tied."
  It was **wrong**: it ran single-day episodes and used a coarse 3-grade
  rule that collapsed exactly the bands being optimized. The full-year
  **production grader** told a clear story the probe couldn't see. Lesson:
  evaluate through the *same* path/grader as training; audit any
  convenient probe against it before believing it.
- Claims that rested on one facility were repeatedly shaky. RL metrics are
  noisy across configs and seeds; a number on one config is an anecdote.
  This is why `clark eval` reports **median / p10–p90 across many held-out
  facilities**, per stage. (See [evaluate.py](../clark/inference/evaluate.py).)
- Throughout, the useful reflex was "go measure it." More than once a
  confident claim ("this retrain will fix the cadence", "fine-tune takes
  ~30 min") was simply false, and only the data caught it.

## 4. How you sample at inference must match how you trained

Argmax inference on this policy scored **~13% ship-win** on stage-3 configs
vs **~93% at temperature ≈ 1.0**. PPO with an entropy bonus trains a
*distribution-mixing* regime — the per-tick action values assume you're
sampling, not committing to the single highest-logit action. Deploy at the
temperature the policy was trained under. (A symptom of this: "task churn
looks high" in training logs was partly a sampling artifact, not a learned
pathology.)

## 5. Separate what the model *does* from how you *display* it

A daily-hours cap (e.g. cycle_count ≤ 0.5 h) was enforced correctly by the
mask — verified tick-by-tick — yet the schedule **rendered** it as multiple
hours. The bug was in the timeline's block-coalescing, which folded short
blocks into the previous block under the wrong label. The model was right;
the renderer lied. Fix: render **ground-truth executed hours** from the
simulator's own tally, not a lossy reconstruction. When a metric looks
wrong, separate "is the behavior wrong?" from "is the *view* wrong?" before
touching the model.

## 6. Make sure the training signal is learnable

Stage-1 of the curriculum originally sampled facilities down to N=3 workers.
Audit found N=3/N=4 configs had a near-zero structural win ceiling — they
were teaching the policy *"you lose"* rather than building competence. Raising
the floor to N=5 fixed it. If a slice of your data is effectively unwinnable,
it's not training, it's noise (or worse, anti-signal). Bound difficulty to
what's actually achievable, then add stress deliberately.

## 7. Value-head saturation: the DreamerV3 recipe earned its keep

Recurring value-head saturation that EMA-normalization and PopArt couldn't
fix was permanently solved by **symlog value targets** (compress targets
with a signed log, predict in that space). When a standard normalization
trick keeps failing on a heavy-tailed target, a compressing transform on the
target itself is often the real fix.

## 8. Systems bugs hide in platform details

The wizard's training runs were dying ~0.3 s after launch with
"Fine-tuning interrupted." Root cause: on **Windows, `os.kill(pid, 0)` is
`CTRL_C_EVENT`** — not a liveness probe — so the status-poll loop was sending
Ctrl-C to the trainer's process group every 5 seconds, killing it on the
first poll (the child also wasn't detached). The timing clue (death at
launch, not after a long run) is what cracked it. Two lessons: detach
child processes you want to outlive the parent, and never assume a
cross-platform call means the same thing on every platform.

## 9. A baseline you didn't try to break isn't a baseline

The whole point of Clark is that a *learned* policy beats classical
scheduling — a claim that's worthless without a real baseline. The first
greedy was naive (no management coverage) and scored **A+B 29%**; it would
have been easy, and dishonest, to stop there and say "RL crushes classical."

Auditing the greedy instead of defending it is what produced the real story:

- **v1 → v2:** add management coverage (an unmet management minimum is an
  auto-F). A+B 29 → ~55.
- **v2 → v3:** staff restock *proactively and scaled to the deficit*, before
  stock hits the picking-speed cliff a single late restocker can't recover.
  A+B 55 → 93.
- **v3 → v4:** the decisive one — **balance against the bottleneck rate, not
  the queue size.** v3 split pick/pack proportional to *backlog*, chasing a
  giant work-in-progress buffer with packers. But picking runs **2.5×** pack
  speed (`PICK_MULTIPLIER` vs pack's 1.0) and the morning-pick round
  front-loads the buffer, so **pack is the perennial constraint** and daily
  throughput ≈ pack capacity. The fat buffer was a *symptom*, not the
  problem. Fix: keep only enough pickers to keep the buffer non-empty, throw
  everyone else at pack, and spend the **capped hustle budget on the crunch**
  instead of dribbling it whenever any backlog exists. A+B 93 → **98.1** on
  the same 20 held-out facilities, F 6.3 → 1.9, A-rate → 46.5.
- **v4 → v5:** the audit (via the CP-SAT `grade_reasons`, §10) showed ~half
  the *non-overtime* A-losses were restock-fill-under-95% demerits, and the
  greedy was targeting only 80% stock — parked inside the penalty band.
  Raising the restock target to the grade's 95% line nearly eliminated that
  demerit. Net on the 20 facilities: A+B 98.1 → **98.3**, F 1.9 → **1.7**,
  p10 55 → 57 — but A-rate 46.5 → **43.3** (holding 95% stock costs some
  order-throughput labour → a few A's become OT B's). A deliberate trade of
  the OT axis for fewer failures + better tail; v5 is the shipped baseline.

The lesson is classic theory-of-constraints / flow-shop, and it's the kind of
structural fact a reactive rule misses until you **instrument it** — a
per-tick audit (workers-per-task, buffer trajectory, hustle rate, split by
day outcome) is what exposed that the greedy was over-serving the non-
bottleneck stage. "Go measure it" beat three rounds of plausible guessing.

**The honest result this produced:** the shipped (v5) heuristic reaches
**A+B 98.3%** — statistically indistinguishable from Clark's 97.5% — at 100%
order completion. So the foundation model does *not* win on the headline
metric. Where it still earns its keep is narrow and specific:
**(1) overtime avoidance** (A-rate 76.5% vs 43.3% — Clark finishes within
regular hours where the heuristic leans on OT to hit the same A+B), and
**(2) worst-case robustness** (F 0.5% vs 1.7%, A+B p10 65 vs 57 — it degrades
more gracefully on the hardest configs). That is a far more credible and
useful claim than "RL crushes classical," and we only have it because we
built the baseline to win, not to lose. (See
[baseline.py](../clark/inference/baseline.py); reproduce with
`clark eval --baseline heuristic --stages 3 --n-per-stage 20 --seed 0`.)

## 10. The CP-SAT bound: when the model you built answers a different question

To check whether Clark's A-rate (OT-free days) was near optimal, I built a
CP-SAT planner (`clark/inference/optimizer.py`) — perfect foresight, optimal
allocation, an optimistic relaxation that should *upper-bound* any online
policy. (The solver is an optional extra: `pip install -e ".[optimizer]"`.) Getting it faithful was a tour of the sim's real mechanics, each
revealed by the model contradicting ground truth:

- **0% feasible (v1):** I gated worker presence on `shift_start/shift_end`.
  The sim gates on `is_absent` — the whole roster is on the floor all day. My
  model sent everyone home at 13:00.
- **Still 0% (v2):** I treated `eod` as a hard wall. It isn't — the day runs
  to the **OT hard stop**, and late arrivals shipped after eod cost no OT.
- **Off by ~8 orders (v3):** integer whole-worker counts can't land on the
  arrival cap when picking runs 6.75 orders/worker-tick. Modelling labour as
  **centi-workers** (fractional, like the sim's `work_carry`) fixed it.

Each fix made the bound *more* faithful — and then the data delivered the
real lesson. A "B" day in the trace had **`ot_hours = 0`**. Auditing the
grader: on a representative facility the non-A days split **overtime 39 /
restock-fill-under-95% 35 / incomplete 6**. The A-grade is **multi-factor**.
An order-flow optimiser — however carefully built — can model the OT
dimension but not restock dynamics, management backlog, or per-task demerits,
so it is **not** a faithful "optimal A-rate" ceiling, and reporting one would
have been a confident lie. The honest move was to *scope it down*: report only
what it bounds soundly.

What it bounds soundly is **completion**, and that turns out to be the
valuable answer: across held-out days, ~**100%** are completion-feasible
(median unshippable ≈ 0.1%). **Throughput is never the binding constraint.**
So the entire Clark-vs-classical gap lives not in raw scheduling capacity but
in jointly satisfying the *soft* quality constraints — the OT checkpoint,
restock %, management, per-task caps — which is exactly the multi-objective
trade-off a learned policy is meant to handle better than a fixed rule. The
"failed" bound reframed the whole question: stop asking "can it schedule
enough?" (always yes) and start asking "can it balance four soft objectives at
once?" (the real game).

Meta-lesson: when a model you trust keeps disagreeing with ground truth, each
disagreement is free knowledge about the system — but once it's faithful,
believe what it tells you even when that's "you built the wrong tool for this
question." A bound that honestly answers a narrower question beats one that
dishonestly answers the question you wanted.

## 11. How much of Clark is linear? A value-distillation probe

A reviewer suggested an alternative to the deep net: an Approximate-LP
approach — represent the value as an *affine* function of features and solve
an LP over simulated snapshots (Powell, *Approximate Dynamic Programming*,
§10.8). It only works if the value is roughly linear in known features. That's
a cheap, decidable question, so instead of arguing it we measured it
([`tools/distill_value_probe.py`](../tools/distill_value_probe.py)): run the
trained foundation through held-out facilities, log `(φ(s), V(s))` at every
tick where `V(s)` is Clark's own critic estimate (carrying full LSTM history),
and fit a linear model `φ(s) → V(s)`. `φ` is the 18 env features + the
mean-pooled 14 worker features + two grade-cliff hinge features (a
`max(0, 0.95 − restock)` kink, schedule pressure). R² = the fraction of
Clark's value *variance* a memoryless affine value recovers.

The result is a clean two-part answer:

- **Within a single facility, ~0.58 mean R²** (range 0.24–0.88 across 5
  facilities; env-only features 0.41, so the worker aggregates + the restock
  hinge carry real signal). Better than chance, but well short of the ~0.9
  that would mean "the value is basically linear" — **~40% of the
  within-facility value is nonlinear / LSTM-history** an affine value can't
  see. And it's *inconsistent*: linear on some facilities, not others, with
  the worst fit on the highest-value-variance facility.
- **Across facilities, a single global linear value does not transfer at all**
  (held-out R² ≈ −8, worse than predicting the mean). Clark's cross-facility
  value structure is genuinely nonlinear — that is the foundation-model edge,
  quantified. The 3 months bought *that*.

Two lessons. First, **a confident architectural opinion ("a linear model would
do") is a measurement, not a debate** — the probe cost one afternoon and a
checkpoint we already had, and it converted hand-waving into 0.58 / −8.
Second, mind what the number *is*: **R² of the value is an upper-bound proxy
for policy quality, not policy quality.** A 58%-explained value can still
yield a serviceable greedy policy (the decision only needs the right
*ordering* of actions, not exact magnitudes); conversely a high R² doesn't
guarantee good actions. So the probe bounds the ALP's *ceiling* and says "this
would likely beat the hand-tuned heuristic but not match Clark, and
unpredictably so" — but the only way to know the realized policy quality is to
build the ALP and run it head-to-head through the same grader. We stopped at
the cheap upper-bound because it already answered the decision: pursue a
linear/ALP policy only if the goal is a GPU-free, interpretable, per-facility
"edge Clark," accepting it will be sub-Clark and inconsistent.

---

*Throughout: the project's value came less from any single result than from
the discipline around it — honest evals, documented failures, and changing
course when the data disagreed with the plan (which it did, more than once).*

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

---

*Throughout: the project's value came less from any single result than from
the discipline around it — honest evals, documented failures, and changing
course when the data disagreed with the plan (which it did, more than once).*

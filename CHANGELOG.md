# Clark — Changelog

Major shipped items. For day-to-day commit history use `git log`.

## Lessons learned

- **Minimal `tasks.enabled` beats over-spec'd by ~95pp on A-grade rate.**
  On `test_v3` (8 workers), a config with the full 12 standard
  tasks enabled spent 47 of 64 worker-hours/day on secondary tasks
  (loading / receiving / returns / QC / training / side_project)
  and only ~0.5 h/day on management. Result after 25 fine-tune
  episodes: 84% ship_win but **0% A-grade days** (capped at C
  because management duty was structurally starved). Same model
  and same workers, trimmed to the minimal 5-task primary set
  (pick / pack / restock / management / idle): **94% A-grade days
  + 97% ship_win after just 5 episodes**, ~30 min wall time. The
  config was the bottleneck, not the model. Wizard's Quick mode
  was updated to hardcode the minimal set; Advanced mode groups
  secondary tasks behind a warning ("consumes worker-hours · may
  demote grades") with a "Show all" expander, so users can't
  accidentally enable them.

## Architecture and training

- **Variable-shape transformer + LSTM architecture** (`clark-v2.5`) — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Synthetic facility generator** with 3-stage curriculum (N=5→10 / 5→25 / 5→50; carryover, peak-staffing, Saturday operations introduced progressively).
- **Pre-train + fine-tune CLI** (`clark pretrain`, `clark finetune`).
- **Per-worker PPO ratio + per-(worker, head) clipping** (IPPO-style) — fixes the ratio-variance-scaling-with-N pathology that saturated `clip_eps=0.2` at large N.
- **Symlog value-target compression** (DreamerV3 recipe) + `vf_clip` — replaced EMA-only normalization and PopArt; permanently fixed the recurring value-head saturation.
- **Assignment logits scaled by `1/√d_model`** for well-tempered softmax at init.
- **Per-worker mean entropy** (N-invariant exploration bonus).
- **Hustle action masks threaded through rollout AND PPO update.**
- **fp32 log-prob storage** (eliminates bf16 ratio noise on sum-over-workers log-probs).
- **Dropout disabled in the policy network** (PPO importance-ratio consistency).
- **`γ=0.999`, `λ=0.98`, `chunk_size=64`** tuned for the 13k-step year.
- **Reward / return clips** wide enough to preserve catastrophic-day signal while bounding value-target tail.
- **Completion-dominant order reward** (dense `per_order_shipped=+3`, `+3 × total_orders` bonus only on full completion, `per_order_incomplete` capped at -10 × min(N_unshipped, 200) / floor -2000) — replaces flat completion bonus that let 95%-shipped failed days net positive reward.
- **Scaled, per-day-capped filler-during-crunch penalty** (orders prioritized over filler under load).
- **Physical pick-buffer cap** (`pick_buffer_capacity`) prevents unbounded over-picking.
- **Restock allowed during OT when stock is critically low** (breaks restock-collapse cascade).
- **Feasibility-bounded synthetic volume** — daily orders tied to OT-rescuable workforce capacity so no generated year is physically unwinnable.
- **Vectorized within-update PPO log-probs** (single GPU sync per buffer update).
- **Vectorized batched action sampler** (one GPU→CPU transfer per tick).
- **Permanent production-tick profiler** (recv/act/ppo wall-clock breakdown).
- **Facility-aware order-arrival schedule** (no silent drops).
- **Float-comparison epsilon at order-cutoff boundary** (no stranded orders).
- **Multi-process env runner** with pipelined CPU/GPU overlap.
- **Multi-process runner protocol smoke test + soft-fail metrics write** (closes the untested-path crash class).
- **N-split TBPTT chunker for peak-staffing days.**
- **Live training metrics + dashboard** (PPO health, day-grade trends, sliding windows, per-N rollup) — see [docs/DASHBOARD.md](docs/DASHBOARD.md).
- **Operational `ship_win` metric** logged separately from the conflated grade-based `win` (audit-driven eval split).
- **Curriculum-counter resume bug fixed** — stage advancement persists across restarts.

## Infrastructure

- **Episode logging + live dashboard** (single-file HTML; reads `training_metrics.json`).
- **Local facility-setup wizard** (stdlib HTTP, no service layer — see [NOTE.md](NOTE.md) on why a hosted API is deliberately not built). Now has a **Quick / Advanced mode split** (Quick ~2 min for archetype-driven setup; Advanced ~15–20 min adds full worker roster + tasks + equipment + Saturday + peak-staffing editing).
- **Minimal localhost inference API** (`clark serve`) — fenced to one real consumer ([clark-mcp](https://github.com/jarmstrong158/clark-mcp)).
- **Natural-language interface** ([clark-mcp](https://github.com/jarmstrong158/clark-mcp)) — local LLM + 7-tool MCP server + web UI + staffing dashboard; QLoRA fine-tune (`clark-hermes3:ft`) deployed locally. Chat auto-launches `clark serve` if it's down, auto-relaunches on death (15s probe, 60s cooldown), and surfaces a banner if it stays down.
- **Wizard `--episodes` default lowered 500 → 50** (the Jack-validation floor; ~3.3 h on a consumer GPU vs the old ~36 h). CLI default stays 500 for backwards compatibility.
- **Foundation pre-training run completed** at episode 15 000 / 15 000, clean termination, value head stable.
- **Validated on Jack's facility** — Clark foundation + 50 fine-tune episodes beats Jack's failure rate and A+B share on Jack's exact config (~0.2 sim years vs Jack's ~9.4). See README's *Results* section.
- **Trained foundation weights publicly released** under [PolyForm Noncommercial 1.0.0](LICENSE) — free for research / evaluation / personal / educational use. `clark_foundation.pt` is on the [GitHub release](https://github.com/jarmstrong158/Clark/releases/latest) and [Hugging Face](https://huggingface.co/Roflimjonny/clark-foundation). Commercial / for-profit production use requires a separate agreement (see [Use Clark](README.md#use-clark--commercial-access)). *(Earlier in the project the weights were held back as commercial-only; that was reversed — locking them away only blocked the noncommercial users who'd benefit, while the license already protects the commercial line.)*

## Facility controls & operator UX

- **Per-task daily-hours caps** (`tasks.daily_hours`) with **auto-off** — once a task's summed worker-hours reach its target, the action mask removes it for the rest of the day so no labor is wasted on it (same pattern as the management quota, generalized). **Configurable unmet penalty** (`tasks.unmet_penalty`: `none` / `letter` / `two_letters` / `fail`) demerits or fails the grade if the target isn't met by EOD. Env-side mask, so it holds on any checkpoint; a cap-aware fine-tune additionally stops the policy fighting it. Surfaced in the wizard as per-task "stop after N hrs/day · penalty" controls.
- **Sunday operations** (`work_sunday` + `sunday_volume_fraction`), mirroring Saturday across the year schedule, `clark serve`'s calendar check, `clark plan`, and the wizard's "Weekend operations" step.
- **Summary-first full-day schedule** — the ops dashboard leads with a per-worker **task-mix** (hours per task, from the simulator's ground-truth per-tick tally), with the per-10-min timeline behind a toggle. Per-tick switching is `tau≈1.0` sampling noise, not an operational plan, so the mix is the default and the timeline is opt-in.
- **F-day explanations** — the outcome projection breaks down *why* each F happened (incomplete orders + shipped %, management minimum, restock, non-peak OT, management backlog, unmet task caps), counted once per failing run, behind a toggle. Makes an F-rate legible as "narrow 99% misses" vs "model fell over."
- **Wizard training progress bar + completion notification** — `/train/{job}/status` returns episode / target / ETA from `training_metrics.json`; the wizard renders a live bar and fires a desktop notification when the run finishes.

## Reliability fixes (operator path)

- **The wizard no longer kills the training it monitors.** On Windows `os.kill(pid, 0)` is `CTRL_C_EVENT` — *not* a liveness probe — so the 5-second status poll was delivering a Ctrl-C to the trainer's process group, killing freshly-started runs on the first poll (with the non-detached child sharing the wizard's console). Fixed two ways: the trainer is spawned **detached** (`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`; `start_new_session` on POSIX) so wizard signals can't reach it, and liveness now uses the `Popen` handle (`proc.poll()`), never `os.kill(pid,0)`. Training genuinely survives closing the wizard now.
- **`Run Clark Dashboard.bat` made robust** — rewritten with `goto`-label flow (no nested-parenthesized-block parser abort, no delayed-expansion trap that silently skipped the serve auto-launch) and always pauses on exit so a startup error stays readable. Canonical `clark_foundation.pt` promoted to the current-architecture (`clark-v2.5`) weights so the launcher, `clark serve`, and `clark mcp` all load the newest model under one name.
- **Schedule duration fidelity** — block coalescing no longer folds short blocks into the previous block under its label (which mis-attributed a capped task's post-cap churn, e.g. cycle_count rendering as hours). It now only merges same-task runs + true one-tick A-b-A flicker, and the task-mix summary reads ground-truth executed hours from the day summary.

## Post-pretrain refinement chain (v2.5 → v2.11)

The 15k-episode pretrain finished with the policy stuck in a "filler during crunch is OK" attractor on heavy days. Six iterations on top of the foundation closed that gap and produced the current production checkpoint. Full rationale + measured deltas in the README's *Post-pretrain refinements* section.

- **v2.5 — structural multi-gate filler mask.** Two prior attempts (v2.1 observation feature, v2.2 reward shaping) both regressed in A/B vs baseline. v2.5 replaces gradient pressure with a hard mask: filler tasks zeroed (`-1e9` pre-softmax) when ANY of four stress gates fires — projection (projected demand/capacity > 0.65), pending (queue > 25% of day total), schedule (completion >20pp behind time-elapsed), time-pressure (orders_remaining / remaining_worker_hours·throughput > 0.85). ~250-ep fine-tune from v2 → v2.5 lifted heavy-day ship rate from ~89% to 99.1% and cut F-rate from ~20% to 5.6%.
- **v2.6 — restock-proactivity gate (5th mask).** OT cascades on hardest days traced to stock falling below the 0.2 picking-speed cliff in mid-day, a feedback loop the 4-gate mask couldn't reach. Added a 5th gate that suppresses filler when `restock_level < 0.35`, preempting the cliff rather than reacting to it.
- **v2.7 — `per_ot_hour` reward `−1.5 → −5.0`.** Grading rubric is OT-binary (any OT use disqualifies A). At the old cost, OT was invisible to PPO next to the +3 per shipped order signal; the policy shipped via OT rather than without. New cost is at the same scale as the shipped reward — what closing the B → A gap actually requires.
- **v2.8 — `mgmt_backlog_norm` observation (`env_feats[17]`).** A v2.7 C-day audit found ~80% of downgrades had no single-day measurable demerit — the demerit was the multi-day backlog accumulator firing in week 2-3. The policy literally couldn't see the failure mode it was triggering. `env_feats` extended 17 → 18 dims; `arch_version` bumped `clark-v2` → `clark-v2.5`. Old v2 checkpoints upgrade via [`tools/transplant_obs_extension.py`](tools/transplant_obs_extension.py) (zero-init col 17 keeps the policy bit-identical on day one).
- **v2.10 — `per_management_hour` `0.5 → 1.0` (gentle 2×).** v2.8 made the management backlog *observable*; v2.10 reinforces the corresponding action signal. v2.9 attempted a 3× bump and destabilized PPO (vloss spiked to 6.85); v2.10 retries with 2× from the stable v2.8 checkpoint. +500 episodes warm-started from v2.8 ep 15800 to ep 16300 (~3.5 h on RTX 5070 Ti, clean termination).
- **Re-validated on Jack** — v2.10 foundation **alone** matches Jack-from-scratch on A-grade (57.5% vs 58%), zero per-facility training. With 50ep fine-tune on Jack: A=62.1% / A+B=95.8% / F=4.2% — *beats* Jack-from-scratch (58% A) at ~0.2 sim years vs Jack's ~9.4. Strongest Jack-facility result Clark has produced.
- **v2.11 — task-flow ramp + minimum-dwell mask + `per_task_switch` penalty.** A per-worker timeline audit found the policy thrashing — ~29 task switches/worker/day, operationally unusable though grade-neutral. Soft fix first: a `ticks_on_task` flow ramp (`TASK_FLOW_RAMP = (0.85, 0.92, 1.0, 1.03, 1.05)`, indexed on consecutive 10-min ticks; 0.85× setup floor on a fresh switch → 1.05× at ~40 min; `management`/`idle`/`cycle_count`/`off` exempt) makes sustained work genuinely faster via throughput, no new reward term to game. **The structural constraint, not the training, was the lever** (the v2.5 lesson again): two retrains — flow-only (`clark_foundation_flow.pt`) and mask+penalty (`clark_foundation_dwell.pt`), ~6.5 h GPU total — *neither* meaningfully cut switching beyond what the mask does at inference; the throughput signal is too diffuse to beat the entropy bonus. The win was a **minimum-dwell mask** (model-agnostic, applies to any served checkpoint): a worker is locked to its current task for `DWELL_MIN_TICKS = 6` (60 min) before a non-emergency switch, releasing **only for hard stops** (idle/off, met daily-hours cap, ineligibility, satisfied management quota, cycle-count ineligibility; OT / EOD / absent / pack-only already `continue` past it) — *not* for soft fluctuations (pick-buffer wobble, restock top-out, filler gate), which are overridden so the block holds. Plus `per_task_switch = -1.5` on genuine churn (real → different-real only). Tuning the **mask** (free, no GPU) is what worked: 30-min soft-yield reached ~21 switches/worker/day; **60-min hard-stop-only reached ~14** (≈ every 38 min, down from ~10) with A+B 90%, 100% completion, zero C/D, and F-days that are 99.7% near-misses — continuity made structural and grade-neutral at no ongoing training cost. (Early off-by-one in the floor counter + soft-gate releases both fixed; regression-tested in `tests/test_flow_ramp.py`.)
- **Serve-time temperature finding.** Argmax inference catastrophically underperforms (13% ship_win on stage-3 vs 93% at tau=1.0). PPO + entropy bonus trains a distribution-mixing regime; the right serve recipe is **tau ≈ 1.0**, matching how PPO saw the policy during training, not argmax.

## Methodology lessons from the iteration chain

- **Trust the in-env production grader over custom probes.** During the v2.10 evaluation I built a fresh head-to-head probe (single-day episodes, 3-grade A/C/D/F rule with no B/restock/mgmt/backlog demerits) and reported v2.8 ≈ v2.10 "essentially tied." The probe was structurally insensitive to exactly the bands these iterations were optimizing. The training-time grader had been telling a much clearer story (A+B share climbing across the run). Future evaluations route through the same production grader as training; one-off probes get audited against it before claims ship.

## Evaluation: classical baseline + optimality bound

- **Held-out eval harness (`clark eval`).** Samples fresh synthetic facilities the model never trained on, simulates a full work-year each through the in-env production grader, and reports metric *distributions* (median / p10–p90) per curriculum stage. Current `clark-v2.5` foundation on 20 held-out stage-3 facilities: median **A+B 97.5%** (p10–p90 65.1–100), **A 76.5%**, **F 0.5%**, completion 100%.
- **Heuristic scheduler baseline ([`clark/inference/baseline.py`](clark/inference/baseline.py)).** A deliberately *strong* rule-based dispatcher — not a strawman — run through the same masks, grader, and held-out facilities; only the decision rule differs. Iterated under audit: v1 naive (29% A+B) → +management coverage → +proactive scaled restock → **v4: balance against the *bottleneck rate*** (pick runs 2.5× pack, so pack is the perennial constraint; trickle pickers to feed the buffer, throw the rest at pack, and concentrate the capped hustle budget on the crunch) → **v5: align restock target to the grade's 95% fill line** (parking stock at 80% sat inside the demerit band). Head-to-head on the same 20 facilities: heuristic **A+B 98.3%, A 43.3%, F 1.7%, completion 100%**. **Result: a well-built heuristic ties Clark on throughput (A+B, completion); Clark's edge is overtime avoidance (A-rate 76.5 vs 43.3), ~3× fewer catastrophic days (F 0.5 vs 1.7), a better worst-case tail (p10 65 vs 57), and zero-per-facility-tuning generalization.** Reproduce: `clark eval --baseline heuristic --stages 3 --n-per-stage 20 --seed 0`. (CLI accepts `heuristic`; `greedy` kept as an alias.)
- **CP-SAT completion bound ([`clark/inference/optimizer.py`](clark/inference/optimizer.py)).** A perfect-foresight constraint-programming planner (Google OR-Tools CP-SAT). Building it faithfully surfaced the sim's real mechanics (availability = `is_absent` not shift windows; day runs to the OT hard stop; labour modelled as centi-workers for fractional `work_carry`). Key finding: the **A-grade is multi-factor** — non-A days split overtime / restock-fill<95% / incomplete — so an order-flow optimizer can't be a faithful "optimal A-rate" ceiling. Scoped to what it bounds soundly: **completion, which comes out ~100% feasible — throughput is never the binding constraint.** The Clark-vs-classical gap is entirely in jointly satisfying the soft quality objectives. Transferable write-up: [ENGINEERING_NOTES §9–§10](docs/ENGINEERING_NOTES.md).

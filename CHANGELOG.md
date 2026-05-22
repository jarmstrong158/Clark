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

- **Variable-shape transformer + LSTM architecture** (`clark-v2`) — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
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
- **Trained foundation weights** — *commercial, not publicly released by design* (see [Use Clark](README.md#use-clark--commercial-access)).

# Clark — Foundation Model for Warehouse Workforce Optimization

> This file is the primary reference for humans and agents working on Clark.
> Read it before touching any code. Keep it up to date as decisions are made.

---

## What Clark Is

Clark is a foundation model for warehouse workforce scheduling and optimization. Given a description of any warehouse facility — number of workers, their roles and capabilities, task types, seasonal order volumes, and business rules — Clark trains a reinforcement learning agent that learns to make near-optimal shift scheduling decisions.

Clark operates as a **pre-train → fine-tune** system:
1. **Pre-training** on thousands of synthetically generated facility configurations builds a general understanding of warehouse dynamics: when to call OT, how to balance task priorities, how hustle affects throughput and worker stamina.
2. **Fine-tuning** on a specific facility's configuration adapts the foundation model to the real-world constraints of that facility in 200–500 episodes (fast).

The result is a CLI tool and eventually a cloud API: give Clark your facility config, get back a trained agent and daily shift plans.

---

## Relationship to Jack

Jack is a complete, working RL agent for one specific warehouse. Clark is the generalization of Jack.

**What was reused (~70%):**
- Reward signal architecture (`_add_reward` pattern)
- Hustle mechanic (daily cap, weekly exhaustion threshold)
- Restock level system (carry between days)
- OT logic (wall-clock max + FP epsilon hard stop)
- Three-phase step loop (pickers first → packers → other workers)
- PPO loss + GAE + TBPTT (chunk_size=64)
- Episode logger structure and dashboard visual design
- Year carryover mechanics (restock, mgmt backlog, cycle count overdue)

**What was rebuilt:**
- `config.py` (hardcoded) → `FacilityConfig` YAML schema (variable)
- `ActorCritic` (LSTM only) → `ClarkActorCritic` (Transformer + LSTM hybrid)
- Worker debuff system (hardcoded names) → config-driven debuff engine
- Action masking (fixed 7×14 in Jack) → variable `(N, M)` bool tensor
- State builder (flat vector) → structured token dict for transformer input
- `config.py` hardcoded worker/task IDs → role strings + task_eligibility sets

---

## Architecture: Why Transformer + LSTM Hybrid

Jack's LSTM handles temporal memory well but requires a fixed-size state vector. To handle variable N workers and M tasks, Clark uses:

```
Input per step:
  worker_feats: (N, worker_feat_dim)   — 14 scalars per worker
  task_feats:   (M, task_feat_dim)     — demand signal + task type embedding
  env_feats:    (env_size,)            — time, orders, restock, season, carrier urgency, etc.

Encoder:
  W = WorkerLinear(worker_feats) + RoleEmbed(roles)     → (N, 256)
  T = TaskLinear(task_feats) + TaskTypeEmbed(types)     → (M, 256)
  E = EnvLinear(env_feats)                              → (256,)
  W = W + E.unsqueeze(0)       # condition workers on global env state
  W = SelfAttention(W) × 2    # workers attend to each other
  W = CrossAttention(W, T)    # workers attend to tasks

Temporal LSTM:
  g = mean(W)                              # (256,) global pooled state
  h, lstm_hidden = LSTM(g, lstm_hidden)    # temporal memory across steps/days
  W_final = W + h.unsqueeze(0)            # broadcast temporal context to workers

Outputs:
  assignment_logits = W_final @ T.T        # (N, M) — mask + sample per worker
  hustle_logits     = Linear(W_final)      # (N, 2) — independent binary per worker
  value             = Linear(h)            # (1,) — global value estimate
```

Key parameters: `d_model = 512`, 4 SA layers, 1 CA layer, 8 attention heads, LSTM hidden = 512, TBPTT chunk = 64.
Estimated ~18M parameters (vs Jack's ~800K).
Architecture version: `clark-v2`

**v2 vs v1:** d_model 256 → 512, self-attention layers 2 → 4, LSTM hidden 256 → 512.
v1 checkpoints are not loadable under v2 (strict arch_version check at load time).

**Pseudocode above is illustrative.** Real implementation:
- Assignment matmul is divided by `√d_model` to keep softmax well-tempered at init (no learnable scaler)
- Entropy bonus is averaged over workers, not summed (N-invariant)
- Value-loss return normalization uses an EMA running mean/var (not per-batch, not PopArt)
- See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full PPO config + rationale on each piece

---

## Standard Task Vocabulary

These are the built-in task types. All facilities get `pick`, `pack`, and `idle`. Others are opt-in.

| Task ID | Display Name | Output Type | Hustle Eligible | Notes |
|---|---|---|---|---|
| `pick` | Picking | orders/hr | Yes | Core production |
| `pack` | Packing | orders/hr | Yes | Core production |
| `idle` | Idle | — | No | Absent/done workers |
| `restock` | Restocking | hours | Yes | Replenish pick locations |
| `management` | Management | hours | No | Admin, floor oversight |
| `cycle_count` | Cycle Count | hours | No | Inventory audit |
| `side_project` | Side Project | hours | Yes | Facility improvement |
| `receiving` | Receiving | units/hr | Yes | Inbound freight |
| `loading` | Loading | units/hr | No | Outbound trucks |
| `returns_processing` | Returns | units/hr | No | Customer returns |
| `quality_check` | QC | orders/hr | No | Outbound inspection |
| `training` | Training | hours | No | Onboarding new hires |

Custom tasks can be added via `custom_tasks` in the facility config YAML.

---

## Config Schema Reference

The facility config is a YAML file. Key sections:

### `facility`
- `name`: display name
- `timezone`: IANA timezone string

### `workers`
List of worker entries:
- `id`, `name`, `base_oph` (orders/hr), `shift_hours`, `shift_start` (24h)
- `role`: `manager` | `assistant_manager` | `lead` | `warehouse`
- `task_eligibility`: `all` or list of task IDs
- `max_ot_hours`: personal OT cap (null = no limit)
- `call_off_probability`: daily absence rate
- `individual_debuff`: optional config-driven debuff
- `task_oph_overrides`: optional per-task OPH rates (e.g. `{pick: 42.5, restock: 8.0}`). Overrides `base_oph × multiplier` for listed tasks.

### `tasks`
- `enabled`: list of standard task IDs to activate
- `custom`: list of custom task definitions

### `volume`
- `seasonal_ranges`: per-month `[low, high]` order counts — **set these to your real numbers**
- `weekly_curve`: per-day-of-week `[low_pct, high_pct]` fractions

### `business_rules`
- `management_daily_hours_required`: target management hours/day
- `ot_wall_clock_max`: max OT hours past normal shift end
- `ot_hard_stop_hour`: absolute latest (24h clock)
- `ot_trigger_orders_remaining`: orders remaining at EOD that trigger OT
- `cycle_count_weekly_hours`: hours/week required for cycle count compliance
- `cycle_count_max_overdue_weeks`: grace period before critical penalty
- `high_volume_day_orders`: order count defining a "high volume" day
- `target_daily_orders`: optional hard target (null = ship everything)
- `order_incomplete_threshold`: remaining orders below this = no penalty
- **Shift timing** (new):
  - `day_start_hour`: shift start in 24h decimal (default 9.0)
  - `eod_hour`: regular end of day in 24h decimal (default 17.5)
  - `order_cutoff_hour`: last hour orders are accepted (default 17.0)
  - `carrier_pickup_hour`: hard carrier deadline (null = no deadline)
- **Breaks** (new):
  - `lunch_hour`: lunch start in 24h decimal (default 13.0)
  - `lunch_duration`: lunch duration in hours — 0.0 disables (default 0.5)
- **Morning pick round** (new):
  - `morning_pick_enabled`: whether workers do an initial pick round (default true)
  - `morning_pick_carts_min/max`: cart count range per worker
  - `morning_pick_per_cart_min/max`: orders-per-cart range
- **Equipment** (new):
  - `pack_stations`: max simultaneous packers (null = unlimited)
  - `carts_available`: total carts in facility (null = unlimited)

### `order_complexity`
Optional. Controls per-day order difficulty mix. Default = all orders equal.
```yaml
order_complexity:
  tiers:
    - name: simple
      weight: 0.30      # fraction of orders (all weights must sum to 1.0)
      oph_multiplier: 1.4  # faster than base OPH
    - name: standard
      weight: 0.55
      oph_multiplier: 1.0
    - name: complex
      weight: 0.15
      oph_multiplier: 0.55  # slower than base OPH
```

### `rewards`
Optional overrides for reward signal weights. Unset = Clark defaults.

---

## Training Phases

### Phase 1: Pre-training (Foundation Model)
- Each episode: sample a random facility config (5–50 workers, 3–15 tasks, random volume curve)
- Run a full simulated year (~260 work days)
- PPO update every TBPTT chunk (16 steps)
- Goal: learn general warehouse dynamics, not facility-specific habits
- ~10,000 episodes recommended; saves `clark_foundation.pt`

### Phase 2: Fine-tuning (Facility-Specific)
- Load `clark_foundation.pt`
- Run 200–500 episodes on the target facility config
- Lower learning rate (5e-5 vs 3e-4 pre-train)
- Optional: freeze encoder layers, only update LSTM + output heads
- Goal: adapt general knowledge to specific constraints

---

## CLI Reference

```
clark init <output.yaml>          — scaffold a new facility config with prompts
clark validate <config.yaml>      — validate config, show warnings and suggestions
clark pretrain                    — run foundation pre-training
  --episodes 10000
  --output clark_foundation.pt
clark finetune                    — fine-tune on a specific facility
  --config my_warehouse.yaml
  --base clark_foundation.pt
  --episodes 500
  --output my_agent.pt
clark plan                        — generate shift plan for a given date
  --config my_warehouse.yaml
  --model my_agent.pt
  --date 2026-04-01
clark dashboard                   — launch local dashboard server
  --config my_warehouse.yaml
  --model my_agent.pt
```

---

## Key Design Constraints

1. **Variable N/M**: All model weights are N/M-agnostic. Attention operates over token sequences; output logits are `(N, M)` via `W_final @ T.T`.
2. **TBPTT**: Hidden state persists across the full year; gradients truncated every 16 steps. Same as Jack.
3. **Action masking**: Applied as `-1e9` fill before softmax, not by zeroing logits. Ensures valid action probabilities sum to 1.
4. **Hustle flag**: Separate `(N, 2)` head, independent of task assignment. A worker can hustle any task (unless task is hustle-blocked).
5. **Padding**: Variable-length batches padded to `max_N` / `max_M`; `key_padding_mask` prevents attention to pad tokens.
6. **Checkpoint versioning**: Every checkpoint includes `arch_version = "clark-v2"` and `facility_config` metadata. Stale checkpoints are rejected at load time.
7. **FP epsilon**: OT hard stop uses `>= OT_HARD_STOP - 1e-9` to guard against float accumulation (learned from Jack's bug).
8. **State dimensions**: `ENV_FEAT_DIM = 17` (was 15 — added `carrier_urgency` at index 15, `order_complexity_load` at index 16). `WORKER_FEAT_DIM = 14` (was 13 — added `task_oph_normalized` at index 13).
9. **Clark limits**: `clark/config/clark_limits.yaml` defines the absolute bounds for all configurable parameters. The synthetic pre-training generator samples within these bounds — everything outside this envelope is out-of-distribution. Expand bounds before retraining when adding new facilities with unusual parameters.

---

## Cloud / API layer — scrapped; minimal local inference endpoint sanctioned

A FastAPI + Celery/Redis + S3 deployment was prototyped and **scrapped**.
The skeleton stubbed training-via-API, faked auth, and used local files
for the "registry" — it overstated readiness and had **zero consumers**.
That speculative cloud/multi-tenant/WMS skeleton **stays dead**: no
Celery, no Redis, no S3, no auth, no facility registry, no
training-via-API. Do not resurrect any of it.

The original rule was: build a serving layer *only when a real consumer
exists to shape it*. **That condition is now met.** The local AI
warehouse system (a domain-fine-tuned LLM in Ollama → `clark-mcp` MCP
server) is a concrete, single, local consumer that needs to call Clark
for inference. So a **minimal, local-only inference endpoint** is now
sanctioned and built — scoped strictly to that consumer:

- **In scope:** localhost only; stateless `facility/scenario in → Clark
  plan out` (+ what-if). Loads the foundation/finetuned weights once,
  serves inference. That is the entire surface.
- **Explicitly still out of scope:** anything multi-tenant, networked
  beyond localhost, authenticated, queue-backed, cloud, or
  training-via-API. If a future need pushes past localhost inference,
  it is a new decision with its own real consumer — not licence to
  rebuild the scrapped skeleton.

This is deliberately recorded as an *evolution* of the rule, not a
reversal: the discipline (no speculative surface; build only for a real
consumer) is intact — the consumer simply now exists.

---

## Pointer to Jack

Jack is the single-facility predecessor to Clark. When in doubt about how a mechanic should work, read Jack's implementation first.
- Key files: `env/warehouse_env.py`, `env/year_env.py`, `agent/ppo.py`, `agent/actions.py`
- Jack's config: `config.py` — the hardcoded version of what Clark's YAML schema replaces

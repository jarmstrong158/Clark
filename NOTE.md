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

Jack (`C:\Users\jarms\repos\volt-warehouse\`) is a complete, working RL agent for one specific warehouse (Volt). Clark is the generalization of Jack.

**What was reused (~70%):**
- Reward signal architecture (`_add_reward` pattern)
- Hustle mechanic (daily cap, weekly exhaustion threshold)
- Restock level system (carry between days)
- OT logic (wall-clock max + FP epsilon hard stop)
- Three-phase step loop (pickers first → packers → other workers)
- PPO loss + GAE + TBPTT (chunk_size=16)
- Episode logger structure and dashboard visual design
- Year carryover mechanics (restock, mgmt backlog, cycle count overdue)

**What was rebuilt:**
- `config.py` (hardcoded) → `FacilityConfig` YAML schema (variable)
- `ActorCritic` (LSTM only) → `ClarkActorCritic` (Transformer + LSTM hybrid)
- Worker debuff system (hardcoded names) → config-driven debuff engine
- Action masking (fixed 7×14) → variable `(N, M)` bool tensor
- State builder (flat vector) → structured token dict for transformer input
- `config.py` hardcoded worker/task IDs → role strings + task_eligibility sets

---

## Architecture: Why Transformer + LSTM Hybrid

Jack's LSTM handles temporal memory well but requires a fixed-size state vector. To handle variable N workers and M tasks, Clark uses:

```
Input per step:
  worker_feats: (N, worker_feat_dim)   — 13 scalars per worker
  task_feats:   (M, task_feat_dim)     — demand signal + task type embedding
  env_feats:    (env_size,)            — time, orders, restock, season, etc.

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

Key parameters: `d_model = 256`, 2 SA layers, 1 CA layer, 8 attention heads, TBPTT chunk = 16.
Estimated ~8–12M parameters (vs Jack's ~800K).
Architecture version: `clark-v1`

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
6. **Checkpoint versioning**: Every checkpoint includes `arch_version = "clark-v1"` and `facility_config` metadata. Stale checkpoints are rejected at load time.
7. **FP epsilon**: OT hard stop uses `>= OT_HARD_STOP - 1e-9` to guard against float accumulation (learned from Jack's bug).

---

## Cloud Architecture (Phase 7)

FastAPI backend + Celery/Redis for training jobs + S3/GCS for storage.

```
POST /facilities              — register facility config
POST /facilities/{id}/train   — trigger fine-tuning job
GET  /facilities/{id}/status  — job status
POST /facilities/{id}/plan    — get shift plan for a date
GET  /facilities/{id}/logs    — dashboard data
```

Docker: separate containers for API and training workers. Scales horizontally.

---

## Pointer to Jack

Jack's reference implementations live at:
- `C:\Users\jarms\repos\volt-warehouse\volt_sim\`
- Key files: `env/warehouse_env.py`, `env/year_env.py`, `agent/ppo.py`, `agent/actions.py`
- Jack's config: `volt_sim/config.py` — the hardcoded version of what Clark's YAML schema replaces

When in doubt about how a mechanic should work, read Jack's implementation first.

# Clark — Architecture Reference

This document describes Clark's technical design for integration engineers and operators evaluating deployment.

---

## Overview

Clark is a foundation reinforcement learning system for warehouse workforce scheduling. Unlike its predecessor [Jack](https://github.com/jarmstrong158/Jack) — which was a single-facility PPO + LSTM agent operating on a fixed-shape state vector — Clark is built around a transformer + LSTM hybrid that handles variable numbers of workers and tasks. The same model weights generalize across facilities; per-facility adaptation happens through a short fine-tune step.

Clark's responsibilities span the full RL lifecycle:

1. **Config ingestion** — parse and validate a facility YAML into a typed `FacilityConfig`
2. **State construction** — translate current facility state into per-worker / per-task / per-env tokens (shape varies with `N` workers and `M` tasks)
3. **Inference** — run the transformer + LSTM forward pass to produce task assignment and hustle decisions
4. **Training** — pre-train on synthetic facilities (foundation model) and fine-tune per facility
5. **API exposure** — wrap registration, training jobs, and shift planning in a FastAPI server
6. **Episode logging** — persist episode-level outcomes and curriculum metadata to a database

---

## State Representation

Clark's encoder consumes a structured state dict at every decision step. Shapes vary per facility (variable `N`, `M`).

### Per-step inputs

```
worker_feats     : (N, 14)    one row per worker
task_feats       : (M, 3)     one row per task
env_feats        : (17,)      global facility state
worker_role_ids  : (N,)       int role index per worker
task_type_ids    : (M,)       int task-type index per task
```

### Worker features (14 dims per worker)

| Index | Feature | Notes |
|---|---|---|
| 0 | Effective OPH | Normalized 0–1 against `clark_limits.max_oph` |
| 1 | Hours worked today | Normalized against shift length |
| 2 | Hours remaining in shift | Normalized |
| 3 | Sleep debuff multiplier | Float ∈ [0, 1] |
| 4 | Health debuff multiplier | Float ∈ [0, 1] |
| 5 | Individual debuff active | Binary |
| 6 | Fatigue flag | Binary |
| 7 | Today's picker | Binary |
| 8 | Pack-only restriction | Binary |
| 9 | Soreness progress | Float ∈ [0, 1] |
| 10 | Hustle hours used today | Normalized |
| 11 | Hustle exhausted | Binary |
| 12 | Absent today | Binary |
| 13 | Per-task OPH normalized | Reflects `task_oph_overrides` if set |

### Task features (3 dims per task)

| Index | Feature | Notes |
|---|---|---|
| 0 | Demand signal | Normalized backlog/queue depth for this task |
| 1 | Availability flag | Binary — task currently runnable |
| 2 | Task type id (float) | Embedded separately via `task_type_ids` |

### Environment features (17 dims)

| Index | Feature |
|---|---|
| 0 | Current hour (normalized over shift) |
| 1 | Orders remaining (normalized) |
| 2 | Orders completed today (normalized) |
| 3 | Picked, not yet packed (normalized) |
| 4 | Restock tasks remaining (normalized) |
| 5 | Restock level (0–1) |
| 6 | Side project progress (0–1) |
| 7–10 | Season one-hot (winter/spring/summer/fall) |
| 11 | Is high-volume day (binary) |
| 12 | Management hours completed (normalized) |
| 13 | OT authorized (binary) |
| 14 | Workers absent today (normalized) |
| 15 | Carrier urgency (binary) |
| 16 | Order complexity load (normalized) |

Worker IDs in the state are positional and match the `id` field order in `config.yaml`. Task IDs are positional and match the order of the resolved task list (standard tasks enabled in the config, followed by custom tasks).

---

## Model Architecture

```
Encoder
  W = WorkerLinear(worker_feats) + RoleEmbed(worker_role_ids)     → (N, d_model)
  T = TaskLinear(task_feats[:,:2]) + TaskTypeEmbed(task_type_ids) → (M, d_model)
  E = EnvLinear(env_feats)                                        → (d_model,)
  W = W + E.unsqueeze(0)                       # condition workers on env
  W = SelfAttention(W) × 4                     # workers attend to each other
  W = CrossAttention(W, T) + FF                # workers attend to tasks

Temporal memory
  g = mean(W)                                  # (d_model,) global pooled
  h, lstm_state = LSTM(g, lstm_state)          # carries across the year
  W_final = W + LSTMProj(h).unsqueeze(0)       # broadcast temporal context

Outputs
  assignment_logits = W_final @ T.T            # (N, M)
  hustle_logits     = HustleHead(W_final)      # (N, 2)
  value             = ValueHead(h)             # ()
```

### Hyperparameters

| Parameter | Value |
|---|---|
| `d_model` | 512 |
| Self-attention layers | 4 |
| Cross-attention layers | 1 |
| Attention heads | 8 |
| LSTM hidden size | 512 |
| TBPTT chunk size | 64 |
| Approx parameters | ~18M |
| Architecture version tag | `clark-v2` |

Every checkpoint embeds an `arch_version` field. Loading a checkpoint with a mismatched arch version raises an error — old checkpoints are not silently coerced.

### Action Masking

Two masks are computed per step from facility state and applied as `-1e9` fills before softmax:

- **Assignment mask** — shape `(N, M)`, `True` where worker `i` is eligible for task `j`. Encodes absence, OT-only-pick-and-pack restriction (with a restock exception when stock is critically low so OT can't deadlock on an empty warehouse), shift exhaustion, pack-only workers, per-task eligibility, management quota gating, cycle-count eligibility, restock-level-driven gating, and pick-buffer-capacity gating (pick is masked off when picked-but-unpacked orders exceed the cart-space proxy, forcing pack/restock — with a stranded-worker fallback that re-enables pick rather than emit an all-False row). Idle is deliberately invalid in the normal branch (only reachable via the absent / shift-exhausted branches) so a deterministic-at-init policy can't collapse to "everyone idle" forever.
- **Hustle mask** — shape `(N, 2)`. Column 0 (no-hustle) is always valid for non-absent workers; column 1 (hustle) is `True` only when the worker hasn't hit their daily hustle cap and isn't absent.

Both masks are applied at sample time AND during PPO log-prob recomputation, so the policy distribution is consistent across rollout and update.

### Padding and Variable Shapes

In batched (multi-environment) forward passes, per-env shapes are padded to `max_N × max_M` across the batch. `key_padding_mask` arguments to the attention layers exclude pad positions. For the assignment logits, padded positions are filled with `-1e9` alongside the action mask, so they cannot be sampled.

---

## Training

Clark trains in two stages: a one-time pre-training run that produces the foundation model, and per-facility fine-tunes that adapt it.

### Pre-training (foundation)

Pre-training exposes the model to thousands of synthetically generated `FacilityConfig` instances. The model never sees the same config twice in a row — `years_per_config = 5` years are simulated on each generated config before sampling a new one.

A 3-stage curriculum advances by share of total configs:

| Stage | Share | Workers | Tasks | Carryover | Peak staffing | Saturday |
|---|---|---|---|---|---|---|
| 1 | first 15% | 3–10 | up to 5 | 0% | 0% | 0% |
| 2 | next 30% | 5–25 | up to 10 | 30% | 30% | 15% |
| 3 | remaining 55% | 5–50 | up to 15 | 40% | 50% | 25% |

The synthetic generator stays within bounds defined by `clark/config/clark_limits.yaml`. Configs outside those bounds are explicitly out-of-distribution; expanding the limits requires retraining.

**Feasibility-bounded volume.** Generated daily order volume is tied to the paired workforce's throughput, not sampled independently. Comfortable capacity ≈ `N × avg_oph × 9 × 0.22` (the 0.22 factor is empirically calibrated — every order needs a sequential pick *and* pack labor unit, minus restock/management/break/complexity overhead and the env's pick→pack coordination cost). Every generated season's peak is hard-capped at `comfortable × 1.25` (max-OT rescue headroom); peak staffing, when present, is additional safety margin and is deliberately *not* counted toward feasibility (it is stage-probabilistic and month-specific). This guarantees every sampled year is winnable with reasonable OT while still producing genuinely demanding peak days. Prior to this, the curriculum emitted configs whose volume physically exceeded workforce capacity (~2× too optimistic), and the agent was being graded on unwinnable years — a ceiling no amount of training could break.

### Fine-tuning (per facility)

Fine-tuning loads the foundation checkpoint and runs 200–500 episodes on a single user-supplied `FacilityConfig`. Default learning rate is `5e-5` (vs `1e-5` in pre-train). Encoder layers can optionally be frozen via `--freeze-encoder` to prevent catastrophic forgetting on facilities very different from the pre-training distribution.

A fresh-init Clark can also be trained directly on a single facility (no foundation), but this requires substantially more episodes — comparable to training Jack from scratch.

### PPO Setup

| Parameter | Value | Notes |
|---|---|---|
| Algorithm | PPO with GAE | |
| `gamma` | **0.999** | Effective horizon ~1000 steps (≈2 simulated days) — sized for the 13,050-step year |
| `gae_lambda` | **0.98** | Effective TD horizon ~50 steps (≈one full day) |
| `clip_epsilon` | 0.2 | Policy ratio clip — applied per (worker × head), not on the joint sum |
| `entropy_coeff` | 0.05 | Compensates for the entropy mean-over-workers normalization; halved from 0.10 when `lr` dropped 2e-5→1e-5 so the entropy bonus didn't dominate the shrunken policy gradient (see [Per-worker mean entropy](#per-worker-mean-entropy)) |
| `value_loss_coeff` | 0.5 | |
| `max_grad_norm` | 0.5 | |
| `epochs_per_update` | 4 | |
| `vf_clip` | 0.2 | PPO-style value-prediction clip, applied **in symlog space**. Bounds the per-update value-head step (see [Symlog value-target compression](#symlog-value-target-compression)) |
| `lr` | **1e-5** | 5e-5 → 2e-5 → 1e-5. Second drop after a 21-hr run hit best-ever metrics then showed 12-hr monotonic drift; the lower LR tightens the trust region around what works |
| Update cadence | per day boundary | matches Jack's daily-update strategy |
| Reward clip (per step) | ±20,000 | guards single-step numerical blow-up; the dominant end-of-day penalties are themselves bounded at the source (`per_order_incomplete` capped at `-10 × min(N, 50)`; cumulative `side_project_during_crunch` capped at -5000/day) |
| Returns clip (post-GAE) | ±500,000 | guards numerical blow-up; symlog compression (below) is the primary mechanism that keeps catastrophic-year signal learnable |
| AMP | bf16 on CUDA | with explicit fp16 fallback for hardware without bf16 |
| Old log-prob storage | fp32 | bf16 quantization noise is comparable to `clip_epsilon` |
| Dropout | **0.0** | Cross-rollout/update dropout-mask difference would saturate `clip_epsilon` even with frozen weights |

The LSTM hidden state persists across the full simulated year. Gradients are truncated every `chunk_size = 64` steps via TBPTT. The PPO update walks the rollout buffer in homogeneous-`N` segments (worker counts can change mid-day under peak staffing) — chunks end at either `chunk_size` steps OR the next `N`-change, whichever comes first. (Earlier values of 16 truncated day-long cause→effect 3-4× per day, so the agent could not learn that an 8AM pick-heavy assignment causes packer-starvation at 11AM.)

### Per-worker PPO ratio

The standard PPO importance-sampling ratio `exp(new_log_prob - old_log_prob)` is computed and clipped **per (step, worker, head)**, not as a sum across all 2N decisions. Concretely: for each step the value head produces `N` per-worker log-probs for task choice and `N` for hustle, and the ratio + clip is applied to each independently before averaging the surrogate.

The motivation is variance scaling. A naive joint ratio sums 2N independent log-prob deltas; even if each per-worker delta has tiny stdev (~0.05), the joint sum has stdev `~0.05 × sqrt(2N)`, and `exp` of that exceeds the `clip_epsilon=0.2` boundary at N≥10. The clip threshold then saturates *structurally* — not because of a bad policy update, but because the dimensionality of the action space exceeded the trust region. Per-worker ratio (this is the IPPO formulation; see [Independent Learning All You Need in StarCraft Multi-Agent Challenge](https://arxiv.org/pdf/2011.09533)) keeps the per-decision ratio variance independent of N.

### Symlog value-target compression

The value head is a simple two-layer MLP (`Linear → Tanh → Linear`). Value targets span ~8 orders of magnitude — a normal day scores in the single digits while a catastrophic one can reach -120,000 — and no fixed-scale normalizer kept the head learnable across that range. The recurring failure mode was **value-head saturation**: the output-layer weights blew out trying to represent the range (observed `value_head.2.weight` std ≈ 14 in a saturated checkpoint vs ≈ 0.08 healthy), value estimates collapsed into noise, and policy improvement stalled. It recurred three times under EMA-only normalization; resetting the head only treated the symptom.

The fix is symlog target compression (the [DreamerV3](https://arxiv.org/abs/2301.04104) recipe). Targets are transformed by a reversible curve that squashes large magnitudes and barely touches small ones:

```
symlog(x) = sign(x) · log(1 + |x|)          # compress (e.g. -120,000 → ≈ -11.7)
symexp(y) = sign(y) · (exp(|y|) - 1)         # decompress, |y| clamped ≤ 20

returns are symexp'd before GAE; the value loss is computed in symlog space:
  v_loss = max( (v − symlog(R))²,  (v_clip − symlog(R))² )   # PPO-style, vf_clip = 0.2
```

The head only ever learns the compressed signal, bounded by construction to roughly ±13 — fully inside its natural operating range. Combined with `vf_clip` (bounding the per-update value step in symlog space), the saturation pathology has not recurred across thousands of episodes since.

**Rejected alternatives.** *PopArt* ([DeepMind 2018](https://arxiv.org/abs/1809.04474)) didn't fit the single-policy / per-day update / single-trajectory-buffer setting — μ/σ oscillated chasing per-day batch noise. *Per-batch standardization* (vanilla PPO) explodes here: TBPTT chunks of 64 correlated steps have tiny intra-chunk σ. *EMA-only normalization* (β=0.01 running mean/var, the prior approach) was an improvement on both but still let the head saturate whenever the return distribution shifted faster than the EMA could track — symlog bounds the target by construction instead of chasing a moving normalizer, which is why it finally held.

### Per-worker mean entropy

The entropy bonus is averaged over workers, not summed. With sum-over-workers the entropy magnitude scaled with N (~7 for N=8), dwarfing the policy gradient (~0.005) — the optimizer was effectively just being told to spread the policy. Per-worker mean keeps the bonus magnitude consistent across N=5 vs N=25 facilities.

The `entropy_coeff = 0.05` is set well above typical PPO defaults to compensate for the mean-over-workers normalization (which is ~Nx smaller in absolute terms than sum-over-workers). It was halved from 0.10 when `lr` dropped 2e-5→1e-5: halving the LR shrank the policy gradient without changing the entropy term, so the bonus went from ~20× to ~40× the policy gradient and the policy drifted toward random. Halving `entropy_coeff` restored the balance.

### Assignment logits scaled by `1/√d_model`

Assignment logits come from `W_final @ T.T`. With both factors at gain=√2 orthogonal init, the raw matmul output magnitude scales with √d_model (~22 at d_model=512), producing near-deterministic softmax at init and immediate entropy collapse. The matmul is divided by `√d_model` (the same trick scaled-dot-product attention uses for its scores) so initial logits are O(1) and softmax is well-tempered. No learnable scaler — earlier experiments with one ended up either freezing gradients (small init) or saturating (large init).

---

## Reward Design

The reward function is the primary behavior lever. Weights live in `FacilityConfig.rewards` (overridable per facility); the env applies them in `facility_env.py`. Two mechanisms are load-bearing enough to call out architecturally:

### Filler-vs-orders priority

"Filler" tasks (side_project, loading, training, quality_check, returns_processing, receiving) are work that doesn't ship today's orders. The agent must not do them while orders are at risk. Three coupled controls enforce this:

1. **No "look busy" floor.** `per_productive_hour` (the small positive reward for any non-idle work) is *not* paid for filler tasks — only for order-flow and management work. Filler is reward-neutral by default, not reward-positive. (Previously the floor let a pure-filler day net positive reward even on a grade-F day.)
2. **Severity-scaled crunch penalty.** When pending order pressure exceeds 10%, filler incurs `side_project_during_crunch × max(1.0, pending_pct × 5.0)` — a flat floor up to ~20% backlog, then a linear ramp to 5× at 100% pending. Acute backlog is punished proportionally harder than chronic.
3. **Per-day crunch cap.** Cumulative crunch penalty is bounded at **-5000/day**. Beyond that the signal is already saturated; the cap prevents a single pathological day from injecting an extreme-magnitude tail that destabilizes the value function (even with symlog upstream).

### Bounded catastrophic penalties

`per_order_incomplete` is capped at `-10 × min(N_unshipped, 50)` (max -500/day) rather than scaling unbounded with unshipped count. Uncapped, a small-N disaster produced a secondary mode in the return distribution that re-saturated the value head. Capping at the source is cheaper and more stable than relying on downstream clips alone.

These were arrived at through a sequenced, attribution-disciplined debugging arc (one or two changes per restart, never bundled) and an independent multi-agent audit pattern — see the context-keeper decision log (`dec-011` … `dec-015`) for the full rationale and rejected alternatives.

---

## Facility Setup Wizard

`clark wizard` launches a local web UI (stdlib `http.server`, vanilla JS, zero new dependencies — same pattern as the dashboard) that walks a non-technical warehouse operator from "describe my facility" to a validated `FacilityConfig` and a kicked-off fine-tune, with no YAML editing.

- **`clark/wizard/presets.json`** is the single source of truth: archetypes, pain-point questions, and validation rules. Each question maps user-friendly choices to specific reward-weight deltas via a hand-curated table — **no LLM calls at runtime**, so refining the question library is a content edit, not an engineering change.
- **Flow:** name/resume → archetype → volume profile (per-intensity actual order ranges + month/weekday shape) → pain-point questions (OT tolerance, incomplete severity, stockout severity, filler tolerance, backlog tolerance) → review with defensive validation → generate YAML → kick off fine-tune subprocess.
- **Endpoints:** `/presets`, `/sessions` (GET/POST/by-id, save+resume), `/validate` (catches broken weight combos, e.g. OT cost dominating incomplete cost), `/generate` (materializes the YAML), `/train/start` + `/train/{id}/status` (subprocess + liveness).
- Generated configs land under `clark/data/configs/user/`; per-job logs under `clark/data/logs/user/{job_id}/`. Launchers: `Run Clark Wizard.bat` (root, double-click) and `clark/wizard/wizard.bat`.

---

## Testing

`pytest` from the repo root runs the full suite (config in `pyproject.toml` under `[tool.pytest.ini_options]`). Coverage targets the silent-regression risks — code where a refactor breaks training without an obvious error:

| Area | What's pinned |
|---|---|
| `test_symlog.py` | symlog/symexp exact round-trip, sign preservation, monotonicity, overflow clamps |
| `test_rewards.py` | `_add_reward` bookkeeping, the crunch cap, the `max(1.0, pending_pct×5)` scale formula |
| `test_worker.py` | OPH multiplier chain (hustle/fatigue/soreness stack multiplicatively), eligibility |
| `test_actions.py` | the no-all-False-row invariant (an all-False mask NaNs the policy softmax), business-rule encoding |
| `test_schema.py` | `validate()` error/warning contract + YAML round-trip |
| `test_synthetic_gen.py` | every generated config validates; volume never exceeds the OT-rescue ceiling; no inverted ranges |
| `test_env_smoke.py` | full-day reset+step loop stays finite and terminates |

The synthetic-gen tests have already caught one latent crash (an inverted volume range that would have killed a stage-3 run mid-flight) before it reached training.

---

## REST API

Clark's API is facility-oriented: register a facility once, queue training jobs, request shift plans, retrieve logs.

### `POST /facilities`

Register a new facility from a YAML config string.

**Request:**
```json
{
  "name": "My Warehouse",
  "config_yaml": "facility:\n  name: My Warehouse\n  ..."
}
```

**Response:**
```json
{
  "facility_id": "abc-123",
  "name": "My Warehouse",
  "registered_at": "2026-05-09T14:00:00Z",
  "has_model": false
}
```

The config is validated against `clark_limits.yaml` at this step. Configs with structural errors (missing required fields, invalid task IDs) are rejected. Configs outside the training distribution produce a warning but are accepted.

### `GET /facilities` and `GET /facilities/{id}`

List all registered facilities, or fetch one. Returns `FacilityInfo` objects (id, name, registered_at, has_model, latest job status).

### `POST /facilities/{id}/train`

Queue a fine-tune job.

**Request:**
```json
{
  "episodes": 500,
  "lr": 5e-5,
  "freeze_encoder": false
}
```

**Response (HTTP 202):**
```json
{
  "job_id": "train-7f2e1a",
  "status": "pending",
  "queued_at": "2026-05-09T14:22:00Z"
}
```

In production the job is enqueued to Redis and consumed by a worker container (Celery). In dev the job runs via `BackgroundTasks` in the API process.

### `GET /facilities/{id}/train/status`

Returns the latest fine-tune job's status, episodes completed, and elapsed time.

```json
{
  "job_id": "train-7f2e1a",
  "status": "running",
  "episodes_done": 312,
  "episodes_total": 500,
  "started_at": "2026-05-09T14:22:05Z"
}
```

`status` ∈ `pending | running | complete | failed`.

### `POST /facilities/{id}/plan`

Generate a shift plan for a given date.

**Request:**
```json
{ "date": "2026-06-01" }
```

**Response:**
```json
{
  "facility_id": "abc-123",
  "date": "2026-06-01",
  "forecast_orders": 412,
  "assignments": [
    {"worker_id": 0, "name": "Manager",  "task": "management", "hustle": false},
    {"worker_id": 1, "name": "Picker A", "task": "pick",       "hustle": true}
  ],
  "value_estimate": 847.3
}
```

Internally: the API loads the facility's fine-tuned model, instantiates a `YearEnv` for the requested date, fast-forwards to that day, runs one simulated day of inference, and returns the resulting per-worker assignments.

### `GET /facilities/{id}/logs`

Tail the latest training logs (for the dashboard).

### `DELETE /facilities/{id}`

Remove a facility (config, checkpoint, logs). Hard delete in dev, soft delete in production.

### Authentication

All endpoints require an `X-API-Key` header. In the MVP skeleton, any non-empty key is accepted. Production target: per-key identity stored hashed in Postgres, with rate limiting via Redis token buckets.

Full OpenAPI spec served at `/docs` once the API is running.

---

## Database Schema

Clark logs every fine-tune episode and notable summary stats. The schema is SQLite-compatible (default) and Postgres-compatible (production).

### `facilities`

One row per registered facility.

```sql
CREATE TABLE facilities (
    id              TEXT PRIMARY KEY,           -- UUID
    name            TEXT NOT NULL,
    config_yaml     TEXT NOT NULL,
    registered_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    has_model       BOOLEAN DEFAULT FALSE
);
```

### `train_jobs`

One row per fine-tune job (queued, running, or complete).

```sql
CREATE TABLE train_jobs (
    id              TEXT PRIMARY KEY,
    facility_id     TEXT REFERENCES facilities(id),
    status          TEXT,                       -- pending | running | complete | failed
    episodes_total  INTEGER,
    episodes_done   INTEGER DEFAULT 0,
    lr              REAL,
    freeze_encoder  BOOLEAN,
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,
    error           TEXT                        -- traceback if failed
);
```

### `episodes`

One row per completed simulated year during fine-tuning.

```sql
CREATE TABLE episodes (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    facility_id        TEXT REFERENCES facilities(id),
    job_id             TEXT REFERENCES train_jobs(id),
    episode_index      INTEGER,
    grade              TEXT,                    -- A/B/C/D/F (last day's grade)
    total_reward       REAL,
    reward_per_worker  REAL,
    orders_shipped     INTEGER,
    ot_hours           REAL,
    win                BOOLEAN,                 -- grade in (A, B)
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### `notable_episodes`

Sparse table — one row each for best reward, worst reward, first perfect day, most debuffs fired.

```sql
CREATE TABLE notable_episodes (
    facility_id    TEXT,
    reason         TEXT,                       -- 'best_reward', 'first_perfect', etc.
    episode_id     INTEGER REFERENCES episodes(id),
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (facility_id, reason)
);
```

### Retention

Episode rows are kept indefinitely. Heavy per-day step data from training (worker timelines, etc.) is sampled to one snapshot per `save_interval` episodes to bound disk growth during long pre-train / fine-tune runs.

---

## Storage Layout

### Production (S3)

```
s3://clark-prod/
  checkpoints/
    clark_foundation.pt          # shared base model
  facilities/
    <facility_uuid>/
      config.yaml                # validated facility YAML
      model.pt                   # latest fine-tuned checkpoint
      logs/
        <job_id>.log             # training log per job
```

### Dev (local filesystem)

```
clark/data/
  checkpoints/
    clark_foundation.pt
  facilities/
    <facility_uuid>/
      config.yaml
      model.pt
      meta.json                  # replaces facilities row in SQLite/PG dev mode
      logs/
        <job_id>.log
```

---

## Docker Compose Structure

Clark ships as three containers in production: API, training worker, database.

```yaml
services:
  api:
    image: clark-api:latest
    ports: ["8000:8000"]
    volumes:
      - ./model:/app/model:ro
    environment:
      - CLARK_DB_URL=postgresql://clark:${DB_PASSWORD}@db/clark
      - CLARK_REDIS_URL=redis://redis:6379/0
    depends_on: [db, redis]

  worker:
    image: clark-worker:latest
    command: celery -A clark.tasks worker --loglevel=info
    depends_on: [db, redis]
    deploy:
      replicas: 2

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: clark
      POSTGRES_USER: clark
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - clark_db:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  clark_db:
```

The `worker` container holds the full training stack (PyTorch, env simulation, model). The `api` container is lighter and can run on a smaller instance.

**SQLite mode (dev):** Drop the `db`, `redis`, and `worker` services and set `CLARK_DB_URL=sqlite:////data/clark.db`. Fine-tune jobs run in-process via FastAPI `BackgroundTasks`. Suitable for single-facility, single-server deployments.

**Postgres + Redis + Celery mode (prod):** Required for multi-facility deployments, parallel fine-tunes, and HA setups.

---

## Resource Profile

Per fine-tune job (500 episodes on a typical facility config):

| Hardware | Wall clock | RAM | Notes |
|---|---|---|---|
| Modern laptop CPU | 25–35 min | ~1.5 GB | bf16 disabled, single-env |
| Consumer GPU (RTX 4070+) | 3–8 min | ~2 GB | bf16 enabled, 8 parallel envs |
| T4 / spot GPU | 5–10 min | ~2 GB | bf16 enabled |

Pre-training (foundation, ~10k episodes): a multi-day GPU job, run once. Re-running is only required when expanding `clark_limits.yaml` bounds or revising the architecture (`arch_version` bump).

---

## Inference vs. Training Boundary

Clark does both — but they run in different containers and have different resource profiles.

**API container (always-on):**
- Forward pass through encoder + LSTM (planning)
- LSTM hidden state initialization for new sessions
- Config-to-state mapping
- Episode summary logging

**Training worker container (on-demand):**
- PPO loss computation, GAE advantage estimation, gradient updates
- Synthetic facility generation (pre-train only)
- Rollout buffer management
- Checkpoint serialization

Model weights flow one way: a fine-tune job uploads a new `model.pt` to S3 (or writes to `clark/data/facilities/.../model.pt` in dev), and the next API request reads it. No live weight sharing — the API picks up new weights at request time.

---

## Config Validation

Configs are validated at registration (`POST /facilities`) and at fine-tune start. Validation has two layers:

1. **Structural** — required fields present, types correct, IDs contiguous, custom tasks well-formed. Failures are HTTP 400 errors and prevent registration.
2. **Bounds** — values within `clark_limits.yaml` ranges (per-worker max OPH, per-month order ranges, OT-related thresholds, etc.). Out-of-bounds values produce warnings but do not block registration. The model may still produce reasonable behavior, especially after fine-tuning.

The `clark validate <config.yaml>` CLI command runs both layers locally without registering.

---

## Reproducibility and Versioning

Every checkpoint embeds:

- `arch_version` — must match the running code's version constant
- `episode` — episode counter at save time (used for resume)
- `facility_config` — YAML string of the config it was trained on (informational)
- `hparams` — full PPO hyperparameter dict
- `model_state_dict` — model weights
- `optimizer_state_dict` — Adam state for resume

Resuming a run reads `episode` and continues curriculum / fine-tune from there. Random seeds are not currently saved, so resumed runs see a different sequence of synthetic configs than their original — acceptable for pre-training (the curriculum stage is what matters), worth being aware of for fine-tune determinism.

# Clark Cloud Architecture

## Overview

Clark is deployed as a cloud-native API that allows facility operators to register configs,
trigger fine-tuning jobs, and request shift plans over HTTP.
This document describes the production-target architecture and the MVP shortcuts taken
for the initial skeleton.

---

## Stack

| Layer | Production | MVP / Dev |
|---|---|---|
| API server | FastAPI + Uvicorn | FastAPI + Uvicorn (same) |
| Background jobs | Celery + Redis | FastAPI `BackgroundTasks` |
| Checkpoint storage | AWS S3 / GCS | Local filesystem `clark/data/` |
| Facility registry + job tracking | PostgreSQL | `meta.json` files on disk |
| Containers | Docker (API + worker images) | Direct `uvicorn` process |
| Frontend | React (later phase) | Existing dashboard HTML |

---

## Deployment Diagram

```
                    ┌──────────────┐
     HTTPS          │   FastAPI    │
User ──────────────>│   API Server │
(CLI / Web)         │   (Docker)   │
                    └──────┬───────┘
                           │ enqueue
                    ┌──────▼───────┐
                    │  Redis Queue │
                    └──────┬───────┘
                           │ consume
                    ┌──────▼───────┐
                    │  Training    │
                    │  Workers     │──── GPU (optional)
                    │  (Docker ×N) │
                    └──────┬───────┘
                           │ store
                    ┌──────▼───────┐
                    │  S3 / GCS    │
                    │  Checkpoints │
                    │  + Logs      │
                    └──────────────┘
```

PostgreSQL (or SQLite for dev) sits alongside Redis for durable job records and facility
metadata — Redis alone is ephemeral.

---

## Data Flow

1. **Register facility** — `POST /facilities`
   - Client sends YAML config string.
   - API parses + validates via `FacilityConfig.from_yaml()` / `.validate()`.
   - UUID assigned; config written to `s3://<bucket>/facilities/<id>/config.yaml`.
   - Row inserted into `facilities` table (PostgreSQL) or `meta.json` (dev).
   - Returns `FacilityInfo` with the assigned UUID.

2. **Queue training** — `POST /facilities/{id}/train`
   - API writes a `TrainJob` record to PostgreSQL with `status=pending`.
   - Publishes job message to Redis queue.
   - Returns `job_id` immediately (HTTP 202 Accepted).

3. **Worker processes job**
   - Training worker (Celery task) pops job from Redis.
   - Downloads `config.yaml` from S3.
   - Resolves `base_model`: either the facility's previous `model.pt` or
     `clark_foundation.pt` from the shared checkpoints bucket.
   - Calls `finetune(config_path, base_model_path, output_path, ...)`.
   - Streams log lines to `s3://.../logs/<job_id>.log` in real time.
   - On completion: uploads `model.pt` to S3, updates job status to `complete`.
   - On exception: updates status to `failed`, stores traceback in job record.

4. **Poll status** — `GET /facilities/{id}/train/status`
   - API queries PostgreSQL for the latest `TrainJob` row for this facility.
   - Returns `TrainStatus` with `status`, `episodes_done`, `started_at`.

5. **Request shift plan** — `POST /facilities/{id}/plan`
   - API downloads `model.pt` and `config.yaml` from S3.
   - Instantiates `FacilityConfig`, `YearEnv`, `StateBuilder`, `ClarkAgent`.
   - Fast-forwards the environment to the requested date.
   - Runs one simulated day; collects per-worker assignments.
   - Returns `PlanResponse` with `forecast_orders` and `assignments` list.

6. **View logs** — `GET /facilities/{id}/logs`
   - API reads the latest `*.log` from S3 (or local `logs/` dir in dev).
   - Returns tail of log lines for the dashboard.

---

## Scaling Model

Training workers are **stateless** — each worker:
1. Pops one job from Redis.
2. Downloads inputs from S3.
3. Calls `finetune()` (CPU or CUDA).
4. Uploads outputs to S3.
5. Updates job status.
6. Is ready for the next job.

This means the worker fleet can be scaled horizontally to handle N parallel fine-tune jobs
simply by launching more worker containers (`docker-compose scale worker=N` or a Kubernetes
HPA rule). Workers share no local state; all coordination is through Redis + PostgreSQL.

**Resource profile per job:**
- 500 episodes on a modern laptop CPU: ~25–35 minutes
- 500 episodes with PyTorch CUDA (T4 GPU): ~3–5 minutes
- RAM: ~1–2 GB peak (model weights ~50 MB, env simulation overhead)

The API server itself is lightweight (no model weights loaded at request time for train
endpoints) and can run on a t3.small.

---

## Authentication & Security

**MVP (current skeleton):**
- Any non-empty `X-API-Key` header is accepted.
- No per-key identity or rate limiting.

**Production target:**
- API keys stored hashed in PostgreSQL, associated with a `facility_owner` record.
- Each key grants access only to facilities owned by that key's owner.
- JWT tokens issued for the web dashboard (short expiry + refresh tokens).
- Rate limiting per API key via Redis token buckets (e.g. 60 req/min, 5 train jobs/day).
- HTTPS enforced at the load balancer (ALB or Cloudflare).

---

## Storage Layout

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

Local dev mirrors this under `clark/data/`:
```
clark/data/
  facilities/
    <facility_uuid>/
      config.yaml
      model.pt
      meta.json                  # replaces PostgreSQL row in dev
      logs/
        <job_id>.log
```

---

## Cost Estimate (AWS)

| Resource | Spec | Cost/hr | Concurrent jobs | Notes |
|---|---|---|---|---|
| API server | t3.small (1 vCPU, 2 GB) | ~$0.02/hr | — | Always-on |
| Training worker | t3.medium (2 vCPU, 4 GB) | ~$0.04/hr | 2–3 | Scale to zero when idle |
| Training worker | t3.large (2 vCPU, 8 GB) | ~$0.08/hr | 5–6 | For larger configs (20+ workers) |
| GPU worker | g4dn.xlarge (T4 GPU) | ~$0.53/hr | 1 | ~5-min fine-tune; spot pricing available |
| Redis | ElastiCache t3.micro | ~$0.02/hr | — | Queue + job state |
| PostgreSQL | RDS t3.micro | ~$0.02/hr | — | Facility + job registry |
| S3 checkpoint storage | ~50 MB per model | ~$0.001/mo | — | Per facility, negligible |

**Typical per-fine-tune cost:** ~$0.02–0.05 (CPU worker) or ~$0.05 (spot GPU).
**Monthly baseline (always-on):** ~$35–45 (API + Redis + RDS, 0 training workers idle).

---

## Docker Layout (planned)

```
docker-compose.yml
  services:
    api:
      build: ./docker/api
      image: clark-api:latest
      ports: ["8000:8000"]
      command: uvicorn api.main:app --host 0.0.0.0 --port 8000
      depends_on: [redis, postgres]

    worker:
      build: ./docker/worker
      image: clark-worker:latest
      command: celery -A clark.tasks worker --loglevel=info
      depends_on: [redis, postgres]
      deploy:
        replicas: 2

    redis:
      image: redis:7-alpine
      ports: ["6379:6379"]

    postgres:
      image: postgres:16-alpine
      environment:
        POSTGRES_DB: clark
        POSTGRES_USER: clark
        POSTGRES_PASSWORD: clark_dev
```

API and worker use the same base Python image but the worker image includes the full
`clark` package with training dependencies (PyTorch, numpy, etc.). The API image can
be kept lighter if the planning endpoint is moved to a separate service.

---

## Next Steps (Phase 8)

- [ ] Add SQLite/PostgreSQL facility registry (replace `meta.json`)
- [ ] Implement `POST /facilities/{id}/train` with real Celery task
- [ ] Implement `POST /facilities/{id}/plan` with checkpoint loading
- [ ] Add per-key auth database lookup
- [ ] Write Dockerfile for API + worker
- [ ] CI: build + import-test on push

"""
Clark REST API — Cloud deployment entry point.
Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /facilities                — register a new facility config (YAML upload)
  GET  /facilities                — list all registered facilities
  GET  /facilities/{id}           — get facility info
  POST /facilities/{id}/train     — trigger fine-tuning job
  GET  /facilities/{id}/train/status — job status
  POST /facilities/{id}/plan      — get shift plan for a date
  GET  /facilities/{id}/logs      — latest training logs (for dashboard)
  DELETE /facilities/{id}         — remove facility

Auth: API key in header `X-API-Key`. For now, accept any non-empty key (placeholder for real auth).
Storage: Local filesystem under `clark/data/` (S3-compatible interface in production).
"""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from clark.config.schema import FacilityConfig


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Clark API",
    description="REST API for Clark — warehouse workforce optimisation foundation model.",
    version="0.1.0",
)


# ── Storage paths ─────────────────────────────────────────────────────────────

# Root lives at clark/data/facilities/<facility_id>/
_DATA_ROOT = Path(__file__).parent.parent / "clark" / "data" / "facilities"
_DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _facility_dir(facility_id: str) -> Path:
    return _DATA_ROOT / facility_id


def _config_path(facility_id: str) -> Path:
    return _facility_dir(facility_id) / "config.yaml"


def _checkpoint_path(facility_id: str) -> Path:
    return _facility_dir(facility_id) / "model.pt"


def _logs_dir(facility_id: str) -> Path:
    return _facility_dir(facility_id) / "logs"


def _meta_path(facility_id: str) -> Path:
    return _facility_dir(facility_id) / "meta.json"


# ── Auth ──────────────────────────────────────────────────────────────────────

def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """
    Placeholder auth: accept any non-empty API key.
    In production, look up the key in a database and return the owner identity.
    """
    if not x_api_key or not x_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or empty X-API-Key header.",
        )
    return x_api_key


# ── Request / Response models ─────────────────────────────────────────────────

class FacilityRegistration(BaseModel):
    """Payload for registering a new facility."""
    name: str = Field(..., description="Human-readable facility name.")
    config_yaml: str = Field(..., description="Full YAML content of the facility config.")


class FacilityInfo(BaseModel):
    """Summary info returned for a registered facility."""
    id: str
    name: str
    n_workers: int
    n_tasks: int
    created_at: str
    has_checkpoint: bool


class TrainRequest(BaseModel):
    """Parameters for a fine-tuning job."""
    base_model: str = Field(
        "foundation",
        description=(
            "Path to a .pt checkpoint, or 'foundation' to use the default "
            "clark/data/checkpoints/clark_foundation.pt."
        ),
    )
    episodes: int = Field(500, ge=1, description="Number of fine-tuning episodes.")
    lr: float = Field(5e-5, gt=0.0, description="Learning rate.")
    freeze_encoder: bool = Field(
        False,
        description="If True, freeze encoder layers to prevent catastrophic forgetting.",
    )


class TrainStatus(BaseModel):
    """Status of a fine-tuning job."""
    job_id: str
    status: Literal["pending", "running", "complete", "failed"]
    episodes_done: int
    started_at: str


class PlanRequest(BaseModel):
    """Parameters for requesting a shift plan."""
    date: str = Field(..., description="Target date in YYYY-MM-DD format.")
    n_days: int = Field(1, ge=1, description="Number of days to plan ahead.")


class PlanResponse(BaseModel):
    """Shift plan result."""
    date: str
    forecast_orders: int
    assignments: list[dict]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_meta(facility_id: str) -> dict:
    """Load facility metadata JSON, or raise 404."""
    mp = _meta_path(facility_id)
    if not mp.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Facility '{facility_id}' not found.",
        )
    with open(mp, "r") as f:
        return json.load(f)


def _save_meta(facility_id: str, meta: dict) -> None:
    with open(_meta_path(facility_id), "w") as f:
        json.dump(meta, f, indent=2)


def _facility_info_from_meta(meta: dict) -> FacilityInfo:
    fid = meta["id"]
    return FacilityInfo(
        id=fid,
        name=meta["name"],
        n_workers=meta["n_workers"],
        n_tasks=meta["n_tasks"],
        created_at=meta["created_at"],
        has_checkpoint=_checkpoint_path(fid).exists(),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/facilities",
    response_model=FacilityInfo,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new facility",
)
def register_facility(
    body: FacilityRegistration,
    _key: str = Depends(require_api_key),
) -> FacilityInfo:
    """
    Parse and validate a facility config YAML, assign a UUID, and persist it.

    Returns the new facility's info including its assigned `id`.
    """
    # Parse and validate the YAML
    import tempfile, os
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(body.config_yaml)
        tmp_path = tmp.name

    try:
        config = FacilityConfig.from_yaml(tmp_path)
        errors, warnings = config.validate()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse facility YAML: {exc}",
        )
    finally:
        os.unlink(tmp_path)

    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Facility config validation failed.", "errors": errors},
        )

    # Assign UUID and set up directory
    facility_id = str(uuid.uuid4())
    fdir = _facility_dir(facility_id)
    fdir.mkdir(parents=True, exist_ok=True)
    _logs_dir(facility_id).mkdir(exist_ok=True)

    # Write canonical YAML
    with open(_config_path(facility_id), "w", encoding="utf-8") as f:
        f.write(body.config_yaml)

    # Persist metadata
    created_at = datetime.now(timezone.utc).isoformat()
    meta: dict = {
        "id": facility_id,
        "name": config.name,
        "n_workers": config.num_workers,
        "n_tasks": config.num_tasks,
        "created_at": created_at,
        "validation_warnings": warnings,
    }
    _save_meta(facility_id, meta)

    return _facility_info_from_meta(meta)


@app.get(
    "/facilities",
    response_model=list[FacilityInfo],
    summary="List all registered facilities",
)
def list_facilities(
    _key: str = Depends(require_api_key),
) -> list[FacilityInfo]:
    """Return summary info for every registered facility."""
    results: list[FacilityInfo] = []
    if not _DATA_ROOT.exists():
        return results
    for entry in sorted(_DATA_ROOT.iterdir()):
        mp = entry / "meta.json"
        if mp.exists():
            with open(mp, "r") as f:
                meta = json.load(f)
            results.append(_facility_info_from_meta(meta))
    return results


@app.get(
    "/facilities/{facility_id}",
    response_model=FacilityInfo,
    summary="Get facility info",
)
def get_facility(
    facility_id: str,
    _key: str = Depends(require_api_key),
) -> FacilityInfo:
    """Return full info for a single facility by its UUID."""
    meta = _load_meta(facility_id)
    return _facility_info_from_meta(meta)


@app.post(
    "/facilities/{facility_id}/train",
    summary="Trigger a fine-tuning job",
)
def trigger_training(
    facility_id: str,
    body: TrainRequest,
    background_tasks: BackgroundTasks,
    _key: str = Depends(require_api_key),
) -> JSONResponse:
    """
    Queue a fine-tuning job for this facility.

    MVP: runs in a FastAPI BackgroundTask.
    Production: enqueue to Celery + Redis; return job_id immediately.
    """
    # Confirm facility exists
    _load_meta(facility_id)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "not_implemented",
            "message": (
                "Training job queuing is stubbed. "
                "In production this enqueues a Celery task and returns a job_id. "
                f"Facility: {facility_id}, base_model: {body.base_model}, "
                f"episodes: {body.episodes}, lr: {body.lr}, "
                f"freeze_encoder: {body.freeze_encoder}."
            ),
        },
    )


@app.get(
    "/facilities/{facility_id}/train/status",
    response_model=TrainStatus,
    summary="Get training job status",
)
def get_train_status(
    facility_id: str,
    _key: str = Depends(require_api_key),
) -> JSONResponse:
    """Return status of the most recent training job for this facility."""
    _load_meta(facility_id)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "not_implemented",
            "message": (
                "Job status tracking is stubbed. "
                "In production, query Redis/PostgreSQL for job state. "
                f"Facility: {facility_id}."
            ),
        },
    )


@app.post(
    "/facilities/{facility_id}/plan",
    response_model=PlanResponse,
    summary="Get a shift plan for a given date",
)
def get_plan(
    facility_id: str,
    body: PlanRequest,
    _key: str = Depends(require_api_key),
) -> JSONResponse:
    """
    Load the facility's fine-tuned checkpoint and run one simulated day to
    produce worker assignments.

    Requires a trained checkpoint at clark/data/facilities/{id}/model.pt.
    """
    import calendar
    import random
    from datetime import date as _date, timedelta

    _load_meta(facility_id)
    config_p = _config_path(facility_id)
    ckpt_p = _checkpoint_path(facility_id)

    if not config_p.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Config file missing for facility '{facility_id}'.",
        )
    if not ckpt_p.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"No trained checkpoint found for facility '{facility_id}'. "
                "Run POST /facilities/{id}/train first."
            ),
        )

    # Parse date
    try:
        start_date = _date.fromisoformat(body.date)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid date format: {body.date!r}. Use YYYY-MM-DD.",
        )

    # Load config + model
    try:
        from clark.config.schema import FacilityConfig
        config = FacilityConfig.from_yaml(str(config_p))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load facility config: {exc}",
        )

    try:
        from clark.agent.ppo import ClarkAgent
        agent = ClarkAgent.load(str(ckpt_p))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model checkpoint: {exc}",
        )

    # Sample order volume for a date using facility volume curves
    def _sample_volume(target: _date) -> tuple[int, str]:
        month_name = calendar.month_name[target.month].lower()
        dow_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        dow_name = dow_names[target.weekday()]

        ranges = config.volume.seasonal_ranges
        vol_range = ranges.get(month_name, [100, 300])
        vol_lo, vol_hi = vol_range[0], vol_range[1]
        vol_spread = vol_hi - vol_lo

        curve = config.volume.weekly_curve
        pct_lo, pct_hi = curve.get(dow_name, [0.5, 1.0])
        day_lo = vol_lo + int(vol_spread * pct_lo)
        day_hi = vol_lo + int(vol_spread * pct_hi)
        volume = random.randint(day_lo, max(day_lo, day_hi))

        high_threshold = config.rules.high_volume_day_orders
        label = (
            "High Volume Day" if volume >= high_threshold
            else "Moderate Volume Day" if volume >= high_threshold * 0.6
            else "Normal Day"
        )
        return volume, label

    # Run one day through the model, return opening assignments
    def _run_day(target: _date, volume: int) -> list[dict]:
        from clark.env.year_env import YearEnv
        from clark.agent.state import StateBuilder
        from clark.agent.actions import get_action_mask

        env = YearEnv(config)
        builder = StateBuilder(config)
        env.reset()
        env.day_env.reset(
            force_month=target.month,
            force_dow=target.weekday(),
            force_volume=volume,
        )
        agent.reset_hidden()

        state_dict = builder.build(env.day_env)
        mask = get_action_mask(env.day_env)
        task_actions, hustle_actions, _lp, _v = agent.select_action_from_dict(state_dict, mask)

        assignments = []
        for i, worker in enumerate(config.workers):
            task_idx = task_actions[i]
            task_name = (
                config.task_ids[task_idx]
                if task_idx < len(config.task_ids)
                else "unknown"
            )
            assignments.append({
                "worker_id": worker.id,
                "worker_name": worker.name,
                "task": task_name,
                "hustle": bool(hustle_actions[i]),
            })
        return assignments

    # Generate plan for requested days (skip weekends)
    days_planned = []
    for offset in range(max(1, body.n_days)):
        target = start_date + timedelta(days=offset)
        if target.weekday() >= 5:
            continue
        volume, vol_label = _sample_volume(target)
        try:
            assignments = _run_day(target, volume)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Planning failed for {target.isoformat()}: {exc}",
            )
        days_planned.append({
            "date": target.isoformat(),
            "day_of_week": calendar.day_name[target.weekday()],
            "forecast_orders": volume,
            "volume_label": vol_label,
            "assignments": assignments,
        })

    if not days_planned:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No work days in the requested date range (all weekend).",
        )

    first = days_planned[0]
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "date": first["date"],
            "forecast_orders": first["forecast_orders"],
            "assignments": first["assignments"],
            "days": days_planned,
        },
    )


@app.get(
    "/facilities/{facility_id}/logs",
    summary="Get latest training logs",
)
def get_logs(
    facility_id: str,
    _key: str = Depends(require_api_key),
) -> JSONResponse:
    """
    Return the most recent training log lines for the dashboard.
    In production, stream from S3 or a log aggregator (e.g. CloudWatch, Loki).
    """
    _load_meta(facility_id)

    logs_dir = _logs_dir(facility_id)
    log_files = sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not log_files:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"facility_id": facility_id, "lines": [], "message": "No logs yet."},
        )

    # Read last 100 lines of the most recent log file
    with open(log_files[0], "r", encoding="utf-8") as f:
        lines = f.readlines()
    tail = [line.rstrip() for line in lines[-100:]]

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "facility_id": facility_id,
            "log_file": log_files[0].name,
            "lines": tail,
        },
    )


@app.delete(
    "/facilities/{facility_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a facility",
)
def delete_facility(
    facility_id: str,
    _key: str = Depends(require_api_key),
) -> None:
    """
    Permanently remove a facility and all its associated data
    (config, checkpoint, logs).
    """
    # Confirm it exists first (raises 404 if not)
    _load_meta(facility_id)

    fdir = _facility_dir(facility_id)
    shutil.rmtree(fdir)

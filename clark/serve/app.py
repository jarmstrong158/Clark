"""Minimal local Clark inference API — the entire surface.

Five stateless routes, localhost only, no auth, no queue, no registry
DB, no cloud. Weights are loaded ONCE (the caller passes a ready
`agent`); every request is load-config -> run Clark's existing,
already-tested inference primitive -> return JSON.

`/plan` and `/what_if` call `_run_one_plan_day` from `cli.main` — the
exact path `clark plan` uses and `tests/test_plan_path.py` pins. This
module is a thin HTTP adapter, NOT a reimplementation of inference.

Scope is fenced by NOTE.md / dec-029: anything beyond localhost
inference is a new decision, never licence to rebuild the scrapped
skeleton.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Thin adapter: reuse the CLI's real inference path verbatim.
from cli.main import _run_one_plan_day, _sample_volume_for_date
from clark.config.schema import FacilityConfig


class PlanRequest(BaseModel):
    facility_id: str
    date: Optional[str] = None          # YYYY-MM-DD; default = today
    volume: Optional[int] = None        # default = season-sampled


class WhatIfRequest(BaseModel):
    facility_id: str
    date: Optional[str] = None
    volume: Optional[int] = None        # override the base volume
    absent_workers: list[str] = []      # worker names forced absent


def _resolve_date(s: Optional[str]) -> date:
    if not s:
        return date.today()
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise HTTPException(422, f"Invalid date {s!r}; use YYYY-MM-DD")


def build_app(agent: Any, facilities_dir: str | Path,
              checkpoint_label: Optional[str] = None) -> FastAPI:
    """Construct the app. `agent` is a ready ClarkAgent (weights loaded
    once by the caller); `facilities_dir` holds the *.yaml configs."""
    fdir = Path(facilities_dir)
    app = FastAPI(title="Clark local inference", version="0.1.0")

    def _config_path(fid: str) -> Path:
        # Reject path-traversal; only flat <id>.yaml/.yml in fdir.
        if "/" in fid or "\\" in fid or ".." in fid:
            raise HTTPException(400, "invalid facility_id")
        for ext in (".yaml", ".yml"):
            p = fdir / f"{fid}{ext}"
            if p.exists():
                return p
        raise HTTPException(404, f"no facility config {fid!r} in {fdir}")

    def _plan_for(cfg: FacilityConfig, when: date,
                  volume: Optional[int]) -> list[dict]:
        vol = volume if volume is not None else _sample_volume_for_date(cfg, when)[0]
        rows = _run_one_plan_day(cfg, agent, when, vol)
        return [{"worker": w, "task": t, "hustle": h} for (w, t, h) in rows]

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "checkpoint": checkpoint_label,
            "facilities_dir": str(fdir),
            "ready": True,
        }

    @app.get("/facilities")
    def facilities():
        if not fdir.is_dir():
            return {"facilities": []}
        ids = sorted({p.stem for p in fdir.iterdir()
                      if p.suffix in (".yaml", ".yml")})
        return {"facilities": ids}

    @app.get("/facility/{fid}")
    def facility(fid: str):
        p = _config_path(fid)
        with open(p) as f:
            return {"facility_id": fid, "config": yaml.safe_load(f)}

    @app.post("/plan")
    def plan(req: PlanRequest):
        cfg = FacilityConfig.from_yaml(_config_path(req.facility_id))
        when = _resolve_date(req.date)
        return {
            "facility_id": req.facility_id,
            "date": when.isoformat(),
            "assignments": _plan_for(cfg, when, req.volume),
        }

    @app.post("/what_if")
    def what_if(req: WhatIfRequest):
        """Returns the opening-assignment plan for the BASE scenario and
        the MODIFIED scenario, for comparison. Honest scope: this is the
        agent's opening assignment under each scenario — NOT a simulated
        end-of-day outcome/grade projection (Clark would have to run a
        full day for that; out of Phase 0 scope, not faked here)."""
        when = _resolve_date(req.date)
        base_cfg = FacilityConfig.from_yaml(_config_path(req.facility_id))
        base = _plan_for(base_cfg, when, None)

        mod_cfg = FacilityConfig.from_yaml(_config_path(req.facility_id))
        absent = set(req.absent_workers)
        for w in mod_cfg.workers:
            if w.name in absent and hasattr(w, "call_off_probability"):
                w.call_off_probability = 1.0
        modified = _plan_for(mod_cfg, when, req.volume)

        return {
            "facility_id": req.facility_id,
            "date": when.isoformat(),
            "base": base,
            "modified": modified,
            "modifications": {
                "volume": req.volume,
                "absent_workers": req.absent_workers,
            },
            "note": "opening-assignment plans for comparison; not an "
                    "end-of-day outcome projection",
        }

    return app

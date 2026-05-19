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
    seed: Optional[int] = None          # set for a reproducible plan


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

    def _try_plannable(path: Path) -> Optional[FacilityConfig]:
        """Load a YAML and return it ONLY if it's a usable, plannable
        FacilityConfig. Returns None for files that exist and are valid
        YAML but are not facilities — e.g. `standard_vocab.yaml`, the
        task-vocabulary reference doc (no workers; `tasks` is a list,
        not a facility mapping). Never raises: callers decide the HTTP
        shape so a non-facility yields a clean 4xx, not a 500."""
        try:
            cfg = FacilityConfig.from_yaml(path)
        except Exception:
            return None
        if not cfg.workers:
            return None
        errors, _ = cfg.validate()
        if errors:
            return None
        return cfg

    def _load_facility(fid: str) -> FacilityConfig:
        """Resolve + load a plannable facility, or raise the right 4xx:
        404 if no config by that id, 422 if the config exists but is not
        a plannable facility (so /plan never leaks an unhandled 500)."""
        cfg = _try_plannable(_config_path(fid))
        if cfg is None:
            raise HTTPException(
                422, f"{fid!r} is not a plannable facility config "
                     f"(no workers / failed validation)")
        return cfg

    def _plan_for(cfg: FacilityConfig, when: date, volume: Optional[int],
                  forced_absent: Optional[set] = None,
                  seed: Optional[int] = None) -> list[dict]:
        if volume is not None:
            vol = volume
        else:
            # The seed contract is "same seed -> same plan", but the
            # season volume draw uses the global `random` module and
            # ran BEFORE _run_one_plan_day's reseed — so an omitted
            # volume sampled off un-seeded global RNG, making a seeded
            # /plan (and what-if base vs modified) silently irreproducible
            # depending on prior RNG history. Seed the volume draw too.
            if seed is not None:
                import random as _r
                import numpy as _np
                import torch as _t
                _r.seed(seed)
                _np.random.seed(seed)
                _t.manual_seed(seed)
            vol = _sample_volume_for_date(cfg, when)[0]
        rows = _run_one_plan_day(cfg, agent, when, vol,
                                 forced_absent=forced_absent, seed=seed)
        return [{"worker": w, "task": t, "hustle": h} for (w, t, h) in rows]

    def _stable_seed(fid: str, when: date) -> int:
        import hashlib
        return int(hashlib.sha1(f"{fid}|{when.isoformat()}".encode()
                                ).hexdigest()[:8], 16)

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
                      if p.suffix in (".yaml", ".yml")
                      and _try_plannable(p) is not None})
        return {"facilities": ids}

    @app.get("/facility/{fid}")
    def facility(fid: str):
        p = _config_path(fid)
        if _try_plannable(p) is None:
            raise HTTPException(
                422, f"{fid!r} is not a plannable facility config "
                     f"(no workers / failed validation)")
        with open(p) as f:
            return {"facility_id": fid, "config": yaml.safe_load(f)}

    @app.post("/plan")
    def plan(req: PlanRequest):
        cfg = _load_facility(req.facility_id)
        when = _resolve_date(req.date)
        return {
            "facility_id": req.facility_id,
            "date": when.isoformat(),
            "assignments": _plan_for(cfg, when, req.volume, seed=req.seed),
        }

    @app.post("/what_if")
    def what_if(req: WhatIfRequest):
        """Returns the opening-assignment plan for the BASE scenario and
        the MODIFIED scenario, for comparison.

        Honest scope: opening assignment under each scenario — NOT a
        simulated end-of-day outcome/grade projection.

        Faithful comparison: base and modified share one deterministic
        seed, so they differ ONLY by the modification (volume and/or the
        forced-absent workers) — not by episode RNG. Absences are forced
        deterministically (NOT via the probabilistic, max-2/day-capped
        call-off roll, which silently ignored most requested absences)."""
        when = _resolve_date(req.date)
        seed = _stable_seed(req.facility_id, when)
        cfg = _load_facility(req.facility_id)
        base = _plan_for(cfg, when, None, seed=seed)

        modified = _plan_for(cfg, when, req.volume,
                             forced_absent=set(req.absent_workers),
                             seed=seed)

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

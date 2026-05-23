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
from clark.config.schema import FacilityConfig, WorkerConfig


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


class SimulateRequest(BaseModel):
    """Year-long end-to-end simulation. Unlike /plan and /what_if
    (which are honest about being OPENING ASSIGNMENTS only), this
    actually runs the policy through ~260 work days so the grade
    distribution is a real outcome, not a projection from a snapshot.
    `extra_workers` lets you sweep roster size — the staffing-
    sufficiency primitive: 'how does +N staff change the F-rate?'"""
    facility_id: str
    extra_workers: int = 0              # additive temps (>=0)
    n_days: Optional[int] = None        # cap days for interactive use
    seed: Optional[int] = None


def _resolve_date(s: Optional[str]) -> date:
    if not s:
        return date.today()
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise HTTPException(422, f"Invalid date {s!r}; use YYYY-MM-DD")


def build_app(agent: Any, facilities_dir: str | Path,
              checkpoint_label: Optional[str] = None,
              per_facility_ckpt_dir: Optional[str | Path] = None,
              ) -> FastAPI:
    """Construct the app. `agent` is the foundation ClarkAgent (weights
    loaded once by the caller); `facilities_dir` holds the *.yaml configs.

    `per_facility_ckpt_dir` (default: `<facilities_dir>/../checkpoints/user`)
    is checked on each request — if `<facility_id>.pt` exists there, that
    fine-tuned checkpoint is lazily loaded and reused for that facility,
    so plans for a wizard-trained facility use its trained weights instead
    of falling back to the generic foundation. Agents are cached after
    first load (~10s warm) so subsequent calls are free."""
    fdir = Path(facilities_dir)
    if per_facility_ckpt_dir is None:
        per_facility_ckpt_dir = fdir.parent / "checkpoints" / "user"
    pf_dir = Path(per_facility_ckpt_dir)
    app = FastAPI(title="Clark local inference", version="0.1.0")
    # The agent carries mutable LSTM hidden state and is reset per
    # request inside _run_one_plan_day. FastAPI runs sync handlers in
    # a threadpool, so overlapping /plan, /what_if, or /simulate
    # requests would race through the same agent. Serialize agent use
    # with a process-wide lock. This is a localhost single-consumer
    # service (clark-mcp); throughput isn't the bottleneck, correctness
    # is.
    import threading as _threading
    _agent_lock = _threading.Lock()

    # Per-facility agent cache. Keys = facility_id; values = ClarkAgent
    # loaded from <pf_dir>/<fid>.pt. None sentinel means "checked, no
    # fine-tune exists — use foundation". Cache survives process lifetime.
    _facility_agents: dict[str, Any] = {}
    _facility_agents_lock = _threading.Lock()

    def _agent_for(fid: str):
        """Return the right agent for a facility: per-facility fine-tune
        if one exists on disk, else the foundation agent. Loads on first
        hit, then cached."""
        with _facility_agents_lock:
            if fid in _facility_agents:
                hit = _facility_agents[fid]
                return hit if hit is not None else agent
            ckpt = pf_dir / f"{fid}.pt"
            if not ckpt.exists():
                _facility_agents[fid] = None
                return agent
            try:
                from clark.agent.ppo import ClarkAgent
                loaded = ClarkAgent.load(str(ckpt))
                _facility_agents[fid] = loaded
                print(f"[serve] loaded fine-tuned weights for {fid!r}: "
                      f"{ckpt.name}", flush=True)
                return loaded
            except Exception as e:
                print(f"[serve] WARN failed to load {ckpt}: {e}; "
                      f"falling back to foundation", flush=True)
                _facility_agents[fid] = None
                return agent

    # Scan order: top-level templates first, then the user/ subdir
    # where the wizard writes user-authored facility YAMLs. iterdir
    # alone misses user/ entirely (subdirs are filtered out by the
    # `.yaml` suffix check), so user-authored facilities were
    # silently invisible to /facilities, /facility/{id}, /plan, etc.
    # — see test_v3.yaml + wizard-Quick-mode UX.
    def _scan_dirs():
        out = [fdir]
        user_sub = fdir / "user"
        if user_sub.is_dir():
            out.append(user_sub)
        return out

    def _config_path(fid: str) -> Path:
        # Reject path-traversal; only flat <id>.yaml/.yml in scanned dirs.
        if "/" in fid or "\\" in fid or ".." in fid:
            raise HTTPException(400, "invalid facility_id")
        for scan in _scan_dirs():
            for ext in (".yaml", ".yml"):
                p = scan / f"{fid}{ext}"
                if p.exists():
                    return p
        raise HTTPException(404, f"no facility config {fid!r} in {fdir} or user/")

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
                  seed: Optional[int] = None,
                  use_agent: Any = None) -> list[dict]:
        ag = use_agent if use_agent is not None else agent
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
        with _agent_lock:  # serialize mutable agent state across requests
            rows = _run_one_plan_day(cfg, ag, when, vol,
                                     forced_absent=forced_absent,
                                     seed=seed)
        return [{"worker": w, "task": t, "hustle": h} for (w, t, h) in rows]

    def _stable_seed(fid: str, when: date) -> int:
        import hashlib
        return int(hashlib.sha1(f"{fid}|{when.isoformat()}".encode()
                                ).hexdigest()[:8], 16)

    @app.get("/health")
    def health():
        ft = []
        if pf_dir.is_dir():
            ft = sorted(p.stem for p in pf_dir.iterdir() if p.suffix == ".pt")
        return {
            "status": "ok",
            "checkpoint": checkpoint_label,
            "facilities_dir": str(fdir),
            "per_facility_ckpt_dir": str(pf_dir),
            "fine_tuned_facilities": ft,
            "ready": True,
        }

    @app.get("/facilities")
    def facilities():
        if not fdir.is_dir():
            return {"facilities": []}
        ids = set()
        for scan in _scan_dirs():
            for p in scan.iterdir():
                if p.suffix in (".yaml", ".yml") and _try_plannable(p) is not None:
                    ids.add(p.stem)
        return {"facilities": sorted(ids)}

    @app.get("/capabilities")
    def capabilities():
        """Architectural capacity / limits of the Clark RL agent itself
        (NOT facility data). Reads clark_limits.yaml + schema so the
        answer cannot drift from the code — the LLM looks this up
        instead of memorizing facts that change."""
        from pathlib import Path as _P
        import importlib
        limits_path = (_P(__file__).resolve().parents[1] / "config"
                       / "clark_limits.yaml")
        try:
            with open(limits_path) as f:
                lim = yaml.safe_load(f) or {}
        except Exception:
            lim = {}
        fac = (lim.get("facility") or {})
        # MAX_TASKS hard-cap lives in the schema validate() (the
        # transformer task_type_embed has 20 rows). Pull it from there
        # rather than hard-coding here.
        try:
            sch = importlib.import_module("clark.config.schema")
            mt_src = open(sch.__file__).read()
            import re as _re
            m = _re.search(r"num_tasks\s*>\s*(\d+)", mt_src)
            n_tasks_hard_max = int(m.group(1)) if m else None
        except Exception:
            n_tasks_hard_max = None
        return {
            "architecture": (
                "Clark v2 — variable-N variable-M transformer + LSTM "
                "PPO foundation model. One pre-trained model "
                "fine-tunes per facility."),
            "limits": {
                "n_workers": fac.get("n_workers"),
                "n_tasks": {"hard_max": n_tasks_hard_max,
                             "note": "model's task-embedding capacity"},
                "base_oph": fac.get("base_oph"),
                "shift_hours": fac.get("shift_hours"),
                "call_off_probability": fac.get("call_off_probability"),
            },
            "training_envelope": (
                "Foundation pre-trained inside these bounds via a "
                "3-stage synthetic curriculum (~15k episodes). "
                "Configs outside bounds are rejected by validation."),
            "sources": ["clark/config/clark_limits.yaml",
                         "clark/config/schema.py"],
            "note": ("Architectural facts ABOUT Clark, not about any "
                     "facility. For facility specifics use "
                     "clark_facility_info."),
        }

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
        ag = _agent_for(req.facility_id)
        return {
            "facility_id": req.facility_id,
            "date": when.isoformat(),
            "assignments": _plan_for(cfg, when, req.volume,
                                     seed=req.seed, use_agent=ag),
            "model": ("fine-tuned" if ag is not agent else "foundation"),
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
        ag = _agent_for(req.facility_id)
        base = _plan_for(cfg, when, None, seed=seed, use_agent=ag)

        modified = _plan_for(cfg, when, req.volume,
                             forced_absent=set(req.absent_workers),
                             seed=seed, use_agent=ag)

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

    def _with_extras(cfg: FacilityConfig, n: int) -> FacilityConfig:
        """Return a copy of cfg with n synthetic temp workers appended.
        Disables peak_staffing on the copy so `extra_workers` is the
        SOLE roster delta (otherwise the env's peak temps double-count)."""
        if n <= 0:
            return cfg
        import copy
        c = copy.deepcopy(cfg)
        c.peak_staffing = None
        next_id = max((w.worker_id for w in c.workers), default=-1) + 1
        for i in range(n):
            c.workers.append(WorkerConfig(
                worker_id=next_id + i, name=f"Temp {i + 1}",
                base_oph=12.0, shift_hours=8.0, shift_start=9.0,
                role="warehouse", task_eligibility="all",
                call_off_probability=0.08))
        return c

    def _simulate_year(cfg: FacilityConfig,
                       seed: Optional[int],
                       max_days: Optional[int] = None,
                       use_agent: Any = None) -> dict:
        ag = use_agent if use_agent is not None else agent
        """End-to-end year simulation: run the trained policy through
        ~260 work days and return the real grade distribution. The
        honest staffing-sufficiency primitive — NOT a snapshot
        projection."""
        from clark.env.year_env import YearEnv
        from clark.agent.state import StateBuilder
        from clark.agent.actions import (get_action_mask,
                                          get_hustle_mask)
        if seed is not None:
            import random as _r
            import numpy as _np
            import torch as _t
            _r.seed(seed)
            _np.random.seed(seed)
            _t.manual_seed(seed)
            if _t.cuda.is_available():
                _t.cuda.manual_seed_all(seed)
        env = YearEnv(cfg)
        builder = StateBuilder(cfg)
        env.reset()
        # Hold the agent lock for the whole year simulation — the
        # LSTM state needs to evolve linearly day-by-day without
        # another request resetting it mid-year. `with` ensures the
        # lock releases even if the inner loop raises.
        with _agent_lock:
            ag.reset_hidden()
            max_ticks = 200_000   # safety cap (~year is well under this)
            ticks = 0
            while not env.is_year_done and ticks < max_ticks:
                sd = builder.build(env.day_env)
                mask = get_action_mask(env.day_env)
                hmask = get_hustle_mask(env.day_env)
                ta, ha, *_ = ag.select_action_from_dict(
                    sd, mask, hustle_mask=hmask)
                actions = [(i, int(ta[i]), bool(ha[i]))
                           for i in range(len(ta))]
                env.step(actions)
                ticks += 1
                # Honest partial sim: stop once we've completed
                # max_days. _get_year_summary() reads daily_grades,
                # which holds whatever's been finalized so far — the
                # grade distribution is correct for the actual days
                # run (sample, not full yr).
                if max_days is not None and \
                        len(env.daily_grades) >= max_days:
                    break
        return env._get_year_summary()

    @app.post("/simulate")
    def simulate(req: SimulateRequest):
        """Full-year end-to-end simulation at a given roster size.
        Real outcome statistics (grade distribution, OT use,
        completion) — the staffing-sufficiency primitive."""
        if req.extra_workers < 0:
            raise HTTPException(422, "extra_workers must be >= 0")
        cfg = _load_facility(req.facility_id)
        cfg = _with_extras(cfg, req.extra_workers)
        ag = _agent_for(req.facility_id)
        summary = _simulate_year(cfg, seed=req.seed,
                                 max_days=req.n_days, use_agent=ag)
        return {
            "facility_id": req.facility_id,
            "extra_workers": req.extra_workers,
            "n_workers": len(cfg.workers),
            "days_run": len(summary.get("daily_summaries", [])),
            "model": ("fine-tuned" if ag is not agent else "foundation"),
            "summary": summary,
            "note": ("real end-to-end simulation; outcome distribution "
                     "on the days run, not a snapshot projection"),
        }

    return app

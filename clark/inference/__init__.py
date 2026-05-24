"""Inference primitives shared by the CLI (`clark plan`) and the
local HTTP service (`clark serve`).

These functions were previously private to `cli/main.py` and imported
across the layering boundary by `clark/serve/app.py` — a smell because
the server has no business depending on CLI internals. Both callers
now import from here so the dependency direction is correct: CLI →
inference, serve → inference, no edge between CLI and serve.
"""
from clark.inference.plan import (
    run_one_plan_day,
    run_full_day_schedule,
    sample_volume_for_date,
)

__all__ = ["run_one_plan_day", "run_full_day_schedule",
           "sample_volume_for_date"]

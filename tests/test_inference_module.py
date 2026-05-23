"""Pin the clark.inference.plan module surface + the cli.main
backwards-compat re-export.

The two helpers (run_one_plan_day / sample_volume_for_date) were
relocated from cli/main.py to clark/inference/plan.py to fix a layering
smell. The CLI still re-exports them under their old leading-underscore
names for any external script that imported the private symbols
directly. If either link breaks, this test fails loudly instead of
shipping a silent ImportError to whoever was relying on the old path.
"""
from datetime import date


def test_inference_module_exports_public_names():
    from clark.inference import plan as P
    assert callable(P.run_one_plan_day)
    assert callable(P.sample_volume_for_date)


def test_inference_package_reexports():
    from clark.inference import run_one_plan_day, sample_volume_for_date
    from clark.inference import plan as P
    # The package __init__ should re-export the same objects, not
    # rebound copies.
    assert run_one_plan_day is P.run_one_plan_day
    assert sample_volume_for_date is P.sample_volume_for_date


def test_cli_main_backcompat_aliases():
    """Old underscore names must still import from cli.main and resolve
    to the new module's functions — same callable, just aliased."""
    from cli.main import _run_one_plan_day, _sample_volume_for_date
    from clark.inference.plan import (run_one_plan_day,
                                       sample_volume_for_date)
    assert _run_one_plan_day is run_one_plan_day
    assert _sample_volume_for_date is sample_volume_for_date


def test_sample_volume_for_date_returns_int_and_label():
    """Smoke-test the pure helper against a real config so a signature
    drift (e.g. someone changes the return shape to a dict) fails here
    instead of in the serve route."""
    from pathlib import Path
    from clark.config.schema import FacilityConfig
    from clark.inference.plan import sample_volume_for_date

    cfg = FacilityConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "clark" / "data"
        / "configs" / "example_small.yaml")
    vol, label = sample_volume_for_date(cfg, date(2026, 5, 22))
    assert isinstance(vol, int) and vol > 0
    assert label in {"Normal Day", "Moderate Volume Day",
                     "High Volume Day"}

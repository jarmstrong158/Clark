"""
Loads clark_limits.yaml and exposes typed bounds for validation and synthetic generation.

Usage:
    from clark.config.bounds import BOUNDS, clamp, validate_value

    lo, hi = BOUNDS["base_oph"]           # (5.0, 40.0)
    clamped = clamp("base_oph", 99.0)     # → 40.0
    ok, msg = validate_value("n_workers", 7)   # → (True, "")
    ok, msg = validate_value("n_workers", 1)   # → (False, "n_workers: 1 is below min 2")
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

Number = Union[float, int]


def _load_bounds(path: Path) -> dict[str, tuple]:
    """
    Read clark_limits.yaml and return a flat dict of {key: (min, max)}.
    The YAML has a top-level 'facility' key; we flatten all entries under it.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    facility = raw.get("facility", {})
    result: dict[str, tuple] = {}
    for key, spec in facility.items():
        if isinstance(spec, dict) and "min" in spec and "max" in spec:
            result[key] = (spec["min"], spec["max"])
    return result


def _find_limits_yaml() -> Path:
    """
    Locate clark_limits.yaml.
    Search order:
      1. clark/config/ directory (same directory as this file)
      2. Package root (two levels up from this file)
    """
    this_dir = Path(__file__).parent
    candidate = this_dir / "clark_limits.yaml"
    if candidate.exists():
        return candidate

    pkg_root = this_dir.parent.parent
    candidate = pkg_root / "clark_limits.yaml"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "clark_limits.yaml not found. Expected at "
        f"{Path(__file__).parent / 'clark_limits.yaml'} "
        "or package root."
    )


# ── Module-level singleton loaded at import time ───────────────────────────────

BOUNDS: dict[str, tuple] = _load_bounds(_find_limits_yaml())


# ── Public helpers ─────────────────────────────────────────────────────────────

def clamp(key: str, value: Number) -> Number:
    """
    Clamp value to the bounds for key.
    Returns value unchanged if key is not in BOUNDS.
    """
    if key not in BOUNDS:
        return value
    lo, hi = BOUNDS[key]
    return max(lo, min(hi, value))


def validate_value(key: str, value: Number) -> tuple[bool, str]:
    """
    Check that value is within bounds for key.

    Returns:
        (True, "")           — value is within bounds
        (False, error_msg)   — value is out of bounds
        (True, "")           — key not in BOUNDS (no constraint; treated as valid)
    """
    if key not in BOUNDS:
        return True, ""
    lo, hi = BOUNDS[key]
    if value < lo:
        return False, f"{key}: {value} is below min {lo}"
    if value > hi:
        return False, f"{key}: {value} exceeds max {hi}"
    return True, ""


def validate_range(key: str, lo: Number, hi: Number) -> list[str]:
    """
    Validate a min/max pair:
      - lo must be <= hi
      - both lo and hi must be within BOUNDS[key]

    Returns a (possibly empty) list of error strings.
    """
    errors: list[str] = []

    if lo > hi:
        errors.append(f"{key}: min {lo} is greater than max {hi}")

    ok_lo, msg_lo = validate_value(key, lo)
    if not ok_lo:
        errors.append(msg_lo)

    ok_hi, msg_hi = validate_value(key, hi)
    if not ok_hi:
        errors.append(msg_hi)

    return errors

"""Shared fixtures for the Clark test suite.

Strategy: lean on the real config factories (synthetic_gen +
example YAMLs) rather than hand-building configs, so tests exercise
the same objects training does. Fixtures are function-scoped unless
the underlying object is expensive and provably immutable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `clark` importable when pytest is run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from clark.config.schema import FacilityConfig
from clark.training.synthetic_gen import generate_random_facility

REPO_ROOT = Path(__file__).parent.parent
CONFIG_DIR = REPO_ROOT / "clark" / "data" / "configs"


@pytest.fixture
def small_config() -> FacilityConfig:
    """The hand-written small example facility (5 workers)."""
    return FacilityConfig.from_yaml(CONFIG_DIR / "example_small.yaml")


@pytest.fixture
def large_config() -> FacilityConfig:
    """The hand-written large example facility."""
    return FacilityConfig.from_yaml(CONFIG_DIR / "example_large.yaml")


@pytest.fixture
def stage1_config() -> FacilityConfig:
    """A randomly generated stage-1 curriculum facility (small N, no carryover)."""
    return generate_random_facility(stage=1)


@pytest.fixture
def stage3_config() -> FacilityConfig:
    """A randomly generated stage-3 curriculum facility (full difficulty)."""
    return generate_random_facility(stage=3)

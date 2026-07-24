"""The suite's optional dependencies must actually be installed.

Why this file exists. Four test modules (serve API, per-facility agent,
compare/calendar, CP-SAT optimizer) used to open with
`pytest.importorskip("fastapi" | "httpx" | "ortools")`. None of those
packages were declared in the `dev` extra that CI installs, so CI
resolved every one of those guards to "skip" and reported green while
never once executing the serve/API layer or the optimizer — ~27 tests
that looked covered and were not.

`importorskip` is the right tool for a dependency that is genuinely
optional to the thing under test. It is the wrong tool for a dependency
the test *requires*, because a skip is indistinguishable from a pass in
a CI summary. The guards are gone; `dev` now pulls in `clark[serve,
optimizer]`. This module is the loud, named tripwire: if the install is
incomplete, you get one unambiguous failure that says which extra is
missing, instead of a wall of collection errors.
"""
from __future__ import annotations

import importlib

import pytest

# module name -> the extra that provides it
REQUIRED = [
    ("fastapi", "serve"),
    ("uvicorn", "serve"),
    ("httpx", "dev"),
    ("ortools", "optimizer"),
]


@pytest.mark.parametrize("module,extra", REQUIRED)
def test_test_dependency_is_installed(module, extra):
    try:
        importlib.import_module(module)
    except ImportError as e:  # pragma: no cover - only fires on a bad install
        pytest.fail(
            f"`{module}` is not installed, so the tests that need it cannot "
            f"run. It is provided by the `{extra}` extra, which `dev` pulls "
            f"in. Install the full test environment with:\n\n"
            f'    pip install -e ".[dev]"\n\n'
            f"(Original error: {e})"
        )


def test_cp_sat_solver_is_usable():
    """ortools being importable is not enough — the CP-SAT native solver
    must load, which is the part that breaks on unsupported platforms."""
    from ortools.sat.python import cp_model

    m = cp_model.CpModel()
    x = m.NewIntVar(0, 5, "x")
    m.Add(x >= 3)
    solver = cp_model.CpSolver()
    assert solver.Solve(m) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.Value(x) >= 3

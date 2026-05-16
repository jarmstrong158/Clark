"""Symlog / symexp value-target compression.

This is the single highest-leverage thing to lock down: the entire
value-learning stability fix (dec-004 era) rests on these four
functions being exact inverses and bounded. A refactor that subtly
breaks the round-trip would silently reintroduce the value-head
saturation pathology that recurred three times before symlog.

Reference: Hafner et al. DreamerV3 (arXiv:2301.04104), §4.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from clark.agent.ppo import _symexp_np, _symexp_t, _symlog_np, _symlog_t


# ── Round-trip: symexp(symlog(x)) == x ────────────────────────────────────────

@pytest.mark.parametrize("x", [
    0.0, 1.0, -1.0, 5.0, -5.0, 100.0, -100.0,
    1234.5, -9876.5, 1e4, -1e4, 50_000.0, -50_000.0,
])
def test_symlog_symexp_roundtrip_np(x):
    """symexp must invert symlog across 5 orders of magnitude (the real
    reward range Clark sees: single-digit to ~100k/day)."""
    recovered = _symexp_np(_symlog_np(np.float64(x)))
    assert recovered == pytest.approx(x, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 250.0, -3461.0, 123_456.0, -123_456.0])
def test_symlog_symexp_roundtrip_torch(x):
    t = torch.tensor([x], dtype=torch.float64)
    recovered = _symexp_t(_symlog_t(t))
    assert recovered.item() == pytest.approx(x, rel=1e-6, abs=1e-6)


def test_torch_numpy_agreement():
    """The torch and numpy implementations must agree — PPO mixes them
    (returns computed in numpy, value loss in torch)."""
    xs = np.array([0.0, 3.0, -7.0, 999.0, -45000.0, 88_888.0], dtype=np.float64)
    np_out = _symlog_np(xs)
    t_out = _symlog_t(torch.tensor(xs, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(np_out, t_out, rtol=1e-9, atol=1e-9)


# ── Sign preservation ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("x", [3.0, -3.0, 1e5, -1e5, 0.001, -0.001])
def test_symlog_preserves_sign(x):
    assert np.sign(_symlog_np(x)) == np.sign(x)


def test_symlog_zero_is_zero():
    assert _symlog_np(0.0) == 0.0
    assert _symexp_np(0.0) == 0.0
    assert _symlog_t(torch.zeros(1)).item() == 0.0


# ── Compression property ──────────────────────────────────────────────────────

def test_symlog_compresses_large_magnitudes():
    """The whole point: a -123k reward day must land in the value head's
    natural operating range (|symlog| well under ~20), not blow it out."""
    extreme = _symlog_np(-123_000.0)
    assert abs(extreme) < 13.0, f"symlog(-123k)={extreme} should be ~-11.7"
    # And the worst capped crunch day (-5000) should be even tamer.
    capped = _symlog_np(-5000.0)
    assert abs(capped) < 9.0


def test_symlog_monotonic_increasing():
    """Monotonicity is required for the value ordering to be preserved
    under compression — otherwise the policy's advantage signs flip."""
    xs = np.linspace(-100_000, 100_000, 5000)
    ys = _symlog_np(xs)
    assert np.all(np.diff(ys) > 0), "symlog must be strictly increasing"


# ── Overflow safety ───────────────────────────────────────────────────────────

def test_symexp_clamps_against_overflow():
    """symexp must not inf/nan even on absurd inputs — the clamp at 20
    is a deliberate guard (symexp(20) ~ 5e8 already)."""
    for bad in [50.0, -50.0, 1e6, -1e6]:
        out = _symexp_np(np.float64(bad))
        assert np.isfinite(out), f"symexp({bad}) = {out} is not finite"
    t_out = _symexp_t(torch.tensor([1e6, -1e6, 50.0]))
    assert torch.isfinite(t_out).all()


def test_symexp_clamp_boundary_is_continuous():
    """Just inside vs at the clamp boundary shouldn't produce a wild jump
    that would create a value-target discontinuity during training."""
    just_under = _symexp_np(np.float64(19.9))
    at_clamp = _symexp_np(np.float64(20.0))
    assert at_clamp >= just_under
    assert np.isfinite(at_clamp)

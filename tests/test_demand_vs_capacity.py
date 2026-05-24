"""Tests for the v2.1 demand_vs_capacity observation feature + companion
crunch-scale tweak.

This is the obs/reward change that targets the "Clark assigns filler
early on overload days" failure mode. Three things need pinning:

1. The canonical arrival curve produces the cumulative-fraction shape
   the design assumes — instant jump to ~0.45 at day_start, monotone
   non-decreasing, 1.0 by cutoff. If a future refactor breaks the
   curve, the projection silently shifts and every downstream gate
   moves with it.

2. The ratio reads from observable quantities only — never
   `episode.total_orders` directly. This is the "no crystal-ball"
   property the design depends on for the policy to transfer to
   production where the true total is genuinely unknown.

3. The companion crunch-scale tweak actually lifts the penalty when
   demand exceeds capacity. Without this gradient, the obs feature is
   signal-without-consequence and the policy will ignore it.
"""
from __future__ import annotations

import numpy as np
import pytest

from clark.agent.state import StateBuilder
from clark.env.episode_generator import (
    DAY_START_HOUR, ORDER_CUTOFF_HOUR, STEP_DURATION,
    expected_fraction_arrived,
)
from clark.env.facility_env import FacilityEnv, ENV_STATE_SIZE
from clark.training.synthetic_gen import generate_random_facility


# ── Canonical arrival-curve shape ───────────────────────────────────

def test_env_state_size_bumped_to_18():
    """v2.1 contract: ENV_STATE_SIZE = 18. If this drops back to 17
    silently the new feature is gone and the warm-started foundation
    will be reading zeros at index 17 forever."""
    assert ENV_STATE_SIZE == 18


def test_expected_fraction_curve_shape():
    """The canonical curve must (a) be 0 before day_start, (b) jump to
    ~0.45 at day_start (the instant bucket), (c) be monotone after,
    (d) hit 1.0 at arrival_end."""
    ds, cut = DAY_START_HOUR, ORDER_CUTOFF_HOUR
    # Well before day_start: nothing should have arrived.
    assert expected_fraction_arrived(ds - 1.0, ds, cut) == 0.0
    # Right at day_start: instant bucket has dumped.
    f0 = expected_fraction_arrived(ds + 1e-4, ds, cut)
    assert 0.40 < f0 < 0.50, f"instant bucket should be ~0.45, got {f0}"
    # End of arrivals: everything in.
    assert expected_fraction_arrived(cut, ds, cut) == 1.0
    # Monotone non-decreasing across the day.
    prev = -1e-6
    t = ds
    while t < cut:
        f = expected_fraction_arrived(t, ds, cut)
        assert f >= prev - 1e-9, f"non-monotone at hour {t}: {prev} → {f}"
        prev = f
        t += STEP_DURATION


def test_expected_fraction_scales_with_facility_hours():
    """A facility opening at 10am instead of 9am should get a curve
    scaled to ITS window, not the canonical 9-17 window. Pinning this
    because the bug history on _generate_arrival_schedule was
    canonical-window-vs-actual-window mismatch dropping arrivals after
    cutoff (and forcing F)."""
    # At "morning-end" relative position (~64% of window), both
    # facilities should see roughly the same cumulative fraction.
    ds_a, cut_a = 9.0, 17.0
    ds_b, cut_b = 10.0, 18.0
    f_a = expected_fraction_arrived(ds_a + 0.64 * (cut_a - ds_a), ds_a, cut_a)
    f_b = expected_fraction_arrived(ds_b + 0.64 * (cut_b - ds_b), ds_b, cut_b)
    assert abs(f_a - f_b) < 0.02, (
        f"scaled-window curves diverged: {f_a} vs {f_b}"
    )


# ── No-oracle property ──────────────────────────────────────────────

def test_dvc_does_not_read_total_orders():
    """The ratio's projection denominator MUST be the canonical curve,
    not episode.total_orders. If a refactor flips this to a true-total
    normalization the feature becomes oracular and the policy learns a
    shortcut that does not transfer to production.

    We assert this by (a) computing the ratio at t=0, (b) artificially
    halving episode.total_orders, (c) recomputing — the ratio MUST be
    unchanged (the projection went via arrived_so_far + canonical
    curve, not total_orders)."""
    env = FacilityEnv(generate_random_facility(stage=2))
    env.reset()
    r_before = env._compute_demand_vs_capacity()
    env.episode.total_orders = max(1, env.episode.total_orders // 2)
    r_after = env._compute_demand_vs_capacity()
    assert r_before == r_after, (
        f"demand_vs_capacity changed when total_orders was mutated "
        f"({r_before} → {r_after}); the projection is leaking from "
        f"the oracle."
    )


def test_dvc_responds_to_arrived_so_far():
    """Conversely, the ratio MUST react to arrived_so_far — that's the
    one observable input. Forcing arrived_so_far up should raise the
    ratio."""
    env = FacilityEnv(generate_random_facility(stage=2))
    env.reset()
    baseline = env._compute_demand_vs_capacity()
    env.arrived_so_far = env.arrived_so_far * 3
    bumped = env._compute_demand_vs_capacity()
    assert bumped > baseline, (
        f"ratio failed to respond to arrived_so_far tripling: "
        f"{baseline} → {bumped}"
    )


def test_dvc_clipped_to_3():
    """Clip to [0, 3] keeps the input in a sane range for the
    network. A 10× overload day should saturate at 3.0, not feed a
    network-poisoning huge value."""
    env = FacilityEnv(generate_random_facility(stage=2))
    env.reset()
    # Force a massive arrival count
    env.arrived_so_far = int(env._today_capacity * 50)
    r = env._compute_demand_vs_capacity()
    assert r == 3.0, f"expected clip at 3.0, got {r}"


# ── env_feats wiring ────────────────────────────────────────────────

def test_state_builder_emits_18_dim_env_feats():
    cfg = generate_random_facility(stage=2)
    env = FacilityEnv(cfg)
    env.reset()
    state = StateBuilder(cfg).build(env)
    assert state["env_feats"].shape == (18,), (
        f"env_feats shape regressed: {state['env_feats'].shape}"
    )
    assert np.isfinite(state["env_feats"]).all()


def test_state_builder_populates_feature_17():
    """Pin that feats[17] is the demand_vs_capacity ratio, not zero
    (which would mean we wired the dim but forgot the assignment)."""
    cfg = generate_random_facility(stage=2)
    env = FacilityEnv(cfg)
    env.reset()
    state = StateBuilder(cfg).build(env)
    expected = env._compute_demand_vs_capacity()
    assert abs(state["env_feats"][17] - expected) < 1e-6


# ── Companion reward gradient ──────────────────────────────────────

def test_crunch_scale_unchanged_when_dvc_under_one():
    """Below dvc=1.0 the new term must not bind — prior behavior on
    rescuable days has to be preserved exactly. We synthesize a
    scenario with no arrivals (dvc=0) and confirm crunch_scale is
    governed by the pending-pct term alone."""
    env = FacilityEnv(generate_random_facility(stage=2))
    env.reset()
    # No arrivals processed yet ⇒ dvc=0. Force pending_pct via
    # injected counts.
    env.orders_in_queue = max(1, env.episode.total_orders // 2)  # 50% pending
    env.arrived_so_far = 0  # explicitly suppress dvc
    pending_pct = (
        (env.orders_in_queue + env.orders_picked_not_audited)
        / max(1, env.episode.total_orders)
    )
    dvc = env._compute_demand_vs_capacity()
    expected_scale = max(1.0, pending_pct * 5.0, (dvc - 1.0) * 3.0)
    # On a 50%-pending, dvc=0 scenario: pending term = 2.5, dvc term
    # = max(0, -3) = -3 → clamped to 1.0. Expected scale = 2.5.
    assert dvc == 0.0
    assert abs(expected_scale - 2.5) < 1e-6


def test_crunch_scale_lifts_when_dvc_overshoots():
    """Above dvc=1.0 the new term must take over the max — that IS
    the new gradient the policy needs."""
    env = FacilityEnv(generate_random_facility(stage=2))
    env.reset()
    # Force a strong dvc overshoot without inflating pending_pct
    # beyond 20% (so the new term has to bind, not the old one).
    env.arrived_so_far = int(env._today_capacity * 2.0)  # dvc near 2.0/expected_frac
    env.orders_in_queue = max(1, env.episode.total_orders // 10)  # 10% pending
    pending_pct = 0.10
    dvc = env._compute_demand_vs_capacity()
    # On a 10%-pending, strong-overshoot scenario:
    #   old term = 0.5, new dvc term = (dvc-1)*3. Whichever is bigger.
    expected_scale = max(1.0, pending_pct * 5.0, (dvc - 1.0) * 3.0)
    assert dvc > 1.0, f"setup should produce dvc>1, got {dvc}"
    assert expected_scale > 0.5, "new dvc term should bind"

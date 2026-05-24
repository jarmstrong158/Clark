"""Synthetic curriculum config generation.

generate_random_facility() feeds every pretrain episode. If it ever
emits a config that fails validate(), training crashes mid-run. If
the stage knobs drift (e.g. stage 1 starts sampling N=50), the
curriculum silently stops being a curriculum. Pin both.
"""
from __future__ import annotations

import pytest

from clark.training.synthetic_gen import generate_random_facility


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("trial", range(8))
def test_generated_config_always_validates(stage, trial):
    """Every generated config across stages + many random draws must
    pass its own validator — a single bad draw crashes a pretrain run
    hours in."""
    cfg = generate_random_facility(stage=stage)
    errors, _warnings = cfg.validate()
    assert errors == [], (
        f"stage={stage} trial={trial} produced an invalid config: {errors}"
    )


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_generated_config_has_pick_and_pack(stage):
    """Order flow requires both pick and pack to exist — a facility
    without them can't ship anything and is a degenerate training env."""
    cfg = generate_random_facility(stage=stage)
    assert "pick" in cfg.task_ids
    assert "pack" in cfg.task_ids


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_worker_count_within_model_capacity(stage):
    """N must stay within the transformer's worker capacity. Stage 3
    is the widest; if it ever exceeds the model embed cap, training
    silently corrupts representations."""
    for _ in range(10):
        cfg = generate_random_facility(stage=stage)
        assert 2 <= cfg.num_workers <= 50


def test_stage1_is_smaller_than_stage3_on_average():
    """The curriculum must actually be a curriculum: stage 1 should
    draw smaller facilities than stage 3 on average. Sample enough to
    beat per-draw noise."""
    s1 = [generate_random_facility(stage=1).num_workers for _ in range(30)]
    s3 = [generate_random_facility(stage=3).num_workers for _ in range(30)]
    assert sum(s1) / len(s1) < sum(s3) / len(s3), (
        f"stage1 avg N ({sum(s1)/len(s1):.1f}) should be < "
        f"stage3 avg N ({sum(s3)/len(s3):.1f})"
    )


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_task_count_within_embedding_cap(stage):
    cfg = generate_random_facility(stage=stage)
    assert cfg.num_tasks <= 20, "exceeds transformer _MAX_TASKS embedding cap"


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("trial", range(40))
def test_volume_ranges_never_inverted(stage, trial):
    """Regression: summer_hi was clamped to 2000 but summer_lo was not,
    so a high winter_lo * peak_mult produced an inverted range like
    (2062, 2000) that crashed episode generation with
    `ValueError: empty range in randrange(...)` mid-pretrain. Every
    season range must satisfy low < high, across many random draws and
    all stages."""
    cfg = generate_random_facility(stage=stage)
    for month, (lo, hi) in cfg.volume.seasonal_ranges.items():
        assert lo < hi, (
            f"stage={stage} trial={trial}: {month} range ({lo}, {hi}) "
            f"is inverted/degenerate — would crash episode generation"
        )
        assert lo >= 30, f"{month} low {lo} below floor"
        assert hi <= 2000, f"{month} high {hi} above ceiling"


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("trial", range(40))
def test_volume_never_exceeds_stress_ceiling(stage, trial):
    """Generalized Fork C invariant. Non-stress configs must stay within
    the OT-rescue ceiling (n_workers × avg_oph × 9 × 0.22 × 1.25). Stress
    configs (introduced to teach overload handling) may go up to that
    × STRESS_MULT=1.7, but not beyond. Anything past the stress ceiling
    is a config-generation bug: structurally unwinnable + outside what
    the curriculum is supposed to expose."""
    cfg = generate_random_facility(stage=stage)
    avg_oph = sum(w.base_oph for w in cfg.workers) / max(1, len(cfg.workers))
    comfortable_pw = max(12.0, avg_oph * 9.0 * 0.22)
    rescue_ceiling = len(cfg.workers) * comfortable_pw * 1.25
    stress_ceiling = rescue_ceiling * 1.7
    worst_hi = max(hi for _lo, hi in cfg.volume.seasonal_ranges.values())
    assert worst_hi <= stress_ceiling + 5, (
        f"stage={stage} trial={trial}: peak volume {worst_hi} exceeds the "
        f"stress ceiling {stress_ceiling:.0f} for {len(cfg.workers)} "
        f"workers @ avg_oph={avg_oph:.1f} -- bigger than the curriculum "
        f"intends to generate, even on stress days."
    )


@pytest.mark.parametrize("stage,expected_min_rate,expected_max_rate", [
    (1, 0.00, 0.00),   # stage 1: no stress, ever
    (2, 0.08, 0.22),   # stage 2: ~15% (allow ±0.07 sampling slack)
    (3, 0.17, 0.33),   # stage 3: ~25% (same slack)
])
def test_stress_tier_fires_at_expected_rate(stage, expected_min_rate,
                                              expected_max_rate):
    """Sample many configs and count how many breach the OT-rescue ceiling
    (i.e. are stress configs). Confirms the new stress_prob actually fires
    at roughly the configured rate per stage. Without this, the foundation
    could silently regress to all-easy-config training and never see
    overload scenarios — the exact gap that motivated this change."""
    n_samples = 300
    stress_hits = 0
    for _ in range(n_samples):
        cfg = generate_random_facility(stage=stage)
        avg_oph = sum(w.base_oph for w in cfg.workers) / max(1, len(cfg.workers))
        comfortable_pw = max(12.0, avg_oph * 9.0 * 0.22)
        rescue_ceiling = len(cfg.workers) * comfortable_pw * 1.25
        worst_hi = max(hi for _lo, hi in cfg.volume.seasonal_ranges.values())
        if worst_hi > rescue_ceiling + 5:
            stress_hits += 1
    rate = stress_hits / n_samples
    assert expected_min_rate <= rate <= expected_max_rate, (
        f"stage={stage}: stress tier fired at {rate:.2%}, expected "
        f"[{expected_min_rate:.0%}, {expected_max_rate:.0%}]. If 0% on "
        f"stage 2/3, the stress_prob isn't actually being used in "
        f"_generate_volume. If >>expected on stage 1, stress fired where "
        f"it shouldn't."
    )


def test_generate_random_facility_can_reset_an_env():
    """The ultimate integration check for the volume bug: a generated
    config must actually drive FacilityEnv.reset() without raising.
    Run enough draws to hit the previously-crashing path."""
    from clark.env.facility_env import FacilityEnv

    for _ in range(25):
        env = FacilityEnv(generate_random_facility(stage=3))
        env.reset()  # raised ValueError pre-fix on inverted summer ranges

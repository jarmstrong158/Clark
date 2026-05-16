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


def test_generate_random_facility_can_reset_an_env():
    """The ultimate integration check for the volume bug: a generated
    config must actually drive FacilityEnv.reset() without raising.
    Run enough draws to hit the previously-crashing path."""
    from clark.env.facility_env import FacilityEnv

    for _ in range(25):
        env = FacilityEnv(generate_random_facility(stage=3))
        env.reset()  # raised ValueError pre-fix on inverted summer ranges

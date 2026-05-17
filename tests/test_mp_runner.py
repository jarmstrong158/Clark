"""MultiprocessRunner protocol smoke test.

The two worst latent bugs this project shipped (the `clark plan`
5-tuple unpack crash and the removed-INCOMPLETE_CAP import crash) were
both *untested code paths* — and the MP/batched runner, the path the
live foundation run uses (`--mp --n-envs 32`), had ZERO coverage. A
full-year integration test is too slow for the suite (~13k steps/year),
so this is a fast contract/smoke test of the spawn → handshake →
step_send/step_recv round-trip → cached-state update → close path. It
would have caught the import-crash class (a worker that dies on import
makes the first `recv()` raise instead of returning state).
"""
from __future__ import annotations

import numpy as np
import pytest

from clark.training.mp_runner import MultiprocessRunner
from clark.training.synthetic_gen import generate_random_facility


def _small_cfg():
    return generate_random_facility(stage=1)


def test_mp_runner_protocol_roundtrip():
    # years_per_config high so no SWAP fires during the short run.
    runner = MultiprocessRunner(
        n_envs=2, config_factory=_small_cfg,
        years_per_config=999, mp_context="spawn",
    )
    try:
        assert len(runner) == 2
        assert len(runner.states()) == 2
        assert len(runner.configs()) == 2

        # Handshake state/mask must be present and shape-consistent.
        for b in range(2):
            N = runner.configs()[b].num_workers
            assert runner.states()[b] is not None
            assert runner.masks()[b].shape[0] == N
            assert runner.hustle_masks()[b].shape[0] == N

        # Reset flags: all True at construction, all False after pop.
        flags = runner.pop_just_reset_flags()
        assert flags == [True, True]
        assert runner.pop_just_reset_flags() == [False, False]

        # Step round-trip: pick the first valid task per worker from the
        # mask, no hustle. Assert the recv contract holds every step.
        for _ in range(10):
            masks = runner.masks()
            task_actions, hustle_actions = [], []
            for b in range(2):
                m = masks[b]
                ta = [int(np.argmax(m[i])) for i in range(m.shape[0])]
                task_actions.append(ta)
                hustle_actions.append([0] * m.shape[0])
            runner.step_send(task_actions, hustle_actions)
            results = runner.step_recv()
            assert len(results) == 2
            for r in results:
                assert {"reward", "done", "info",
                        "episode_done", "finished_episode"} <= set(r)
            # Cached state/mask views update in lockstep with the recv and
            # stay shape-consistent with each slot's config.
            assert len(runner.states()) == 2
            for b in range(2):
                assert runner.masks()[b].shape[0] == \
                    runner.configs()[b].num_workers
    finally:
        runner.close()


def test_mp_runner_rejects_bad_n_envs():
    with pytest.raises(ValueError):
        MultiprocessRunner(n_envs=0, config_factory=_small_cfg)

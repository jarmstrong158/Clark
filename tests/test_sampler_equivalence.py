"""Equivalence gate for the vectorized batched sampler.

The per-env Python loop in select_action_batched (the training-throughput
bottleneck: ~235ms/tick vs 27ms GPU) is being replaced with a single
batched Categorical over the padded (B, max_N, K) logits. A vectorized
sampler that is NOT distribution-identical to the per-env loop silently
corrupts the policy with no error — the worst possible failure mode.

This test pins the equivalence two ways:
  1. log_prob is EXACTLY equal (deterministic): given the same logits and
     the same fixed action, per-env Categorical.log_prob must equal the
     batched Categorical.log_prob slice to float precision. This is the
     strong guarantee that the underlying distribution math is identical.
  2. The sampled empirical distribution matches over many draws (the
     batched RNG draws in a different order than B sequential .sample()
     calls, so individual samples differ but the distribution must not).
  3. Degenerate rows (all-but-one masked, fully masked padded rows) do
     not NaN and respect the mask.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

MASK_FILL = -1e9


def _make_logits(B, max_N, max_M, seed):
    """Random logits with realistic masking: per (b, worker) at least one
    valid task, some invalid (-1e9), plus padded worker rows and padded
    task columns filled -1e9 — mirrors forward_batched's output."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, max_N, max_M, generator=g)
    n_w = []
    n_t = []
    for b in range(B):
        Ni = int(torch.randint(2, max_N + 1, (1,), generator=g))
        Mi = int(torch.randint(2, max_M + 1, (1,), generator=g))
        n_w.append(Ni)
        n_t.append(Mi)
        # Padded task columns -> masked
        logits[b, :, Mi:] = MASK_FILL
        # Padded worker rows -> masked (discarded downstream, must not NaN)
        logits[b, Ni:, :] = MASK_FILL
        # Within real rows, mask a random subset of valid tasks but always
        # leave at least one open (the no-all-False invariant).
        for w in range(Ni):
            keep = int(torch.randint(1, Mi + 1, (1,), generator=g))
            perm = torch.randperm(Mi, generator=g)
            drop = perm[keep:]
            logits[b, w, drop] = MASK_FILL
    return logits, n_w, n_t


def _per_env_logprob(logits, n_w, n_t, actions):
    """Reference: the original per-env loop's log_prob for a fixed action."""
    out = []
    for b in range(len(n_w)):
        Ni, Mi = n_w[b], n_t[b]
        d = Categorical(logits=logits[b, :Ni, :Mi])
        out.append(d.log_prob(actions[b][:Ni]))
    return out


def _batched_logprob(logits, n_w, actions_full):
    """Vectorized: one Categorical over (B, max_N, K), sliced per env."""
    d = Categorical(logits=logits)
    lp_full = d.log_prob(actions_full)  # (B, max_N)
    return [lp_full[b, : n_w[b]] for b in range(len(n_w))]


# ── 1. Deterministic log_prob equivalence ─────────────────────────────────────

@pytest.mark.parametrize("seed", range(12))
def test_logprob_exactly_matches_per_env(seed):
    B, max_N, max_M = 6, 12, 9
    logits, n_w, n_t = _make_logits(B, max_N, max_M, seed)

    # Pick an arbitrary valid action per (b, worker): argmax of the masked
    # row is guaranteed valid (the -1e9 entries can't be the max).
    actions_full = logits.argmax(dim=-1)  # (B, max_N)
    per_env_actions = [actions_full[b] for b in range(B)]

    ref = _per_env_logprob(logits, n_w, n_t, per_env_actions)
    got = _batched_logprob(logits, n_w, actions_full)

    for b in range(B):
        torch.testing.assert_close(
            ref[b], got[b], rtol=1e-6, atol=1e-6,
            msg=f"seed={seed} env={b}: batched log_prob != per-env log_prob",
        )


def test_logprob_finite_on_degenerate_rows():
    """Fully-masked padded rows and all-but-one-masked real rows must give
    finite log_prob for the valid action and never NaN."""
    B, max_N, max_M = 3, 8, 6
    logits, n_w, n_t = _make_logits(B, max_N, max_M, 99)
    # Force env 0 worker 0 to have exactly one valid task.
    logits[0, 0, :] = MASK_FILL
    logits[0, 0, 2] = 1.234
    actions_full = logits.argmax(dim=-1)
    d = Categorical(logits=logits)
    lp = d.log_prob(actions_full)
    assert torch.isfinite(lp[0, 0]), "single-valid-task row produced non-finite log_prob"
    # The forced row must select the only valid task.
    assert int(actions_full[0, 0]) == 2
    # No NaN anywhere, including discarded padded rows.
    assert not torch.isnan(lp).any()


# ── 2. Distributional equivalence ─────────────────────────────────────────────

def test_sample_distribution_matches_per_env():
    """Batched sampling draws from the RNG in a different order than B
    sequential .sample() calls, so individual draws differ. The empirical
    action distribution per (b, worker) must still match within sampling
    error over many draws."""
    B, max_N, max_M = 4, 6, 7
    logits, n_w, n_t = _make_logits(B, max_N, max_M, 7)
    DRAWS = 20000

    # Per-env empirical frequencies.
    torch.manual_seed(0)
    ref_counts = [torch.zeros(n_w[b], max_M) for b in range(B)]
    for _ in range(DRAWS):
        for b in range(B):
            Ni, Mi = n_w[b], n_t[b]
            s = Categorical(logits=logits[b, :Ni, :Mi]).sample()
            ref_counts[b][torch.arange(Ni), s] += 1

    # Batched empirical frequencies.
    torch.manual_seed(0)
    got_counts = [torch.zeros(n_w[b], max_M) for b in range(B)]
    bd = Categorical(logits=logits)
    for _ in range(DRAWS):
        s = bd.sample()  # (B, max_N)
        for b in range(B):
            Ni = n_w[b]
            got_counts[b][torch.arange(Ni), s[b, :Ni]] += 1

    for b in range(B):
        pr = ref_counts[b] / DRAWS
        pg = got_counts[b] / DRAWS
        # 20k draws: sampling error on a frequency is ~1/sqrt(20000) ≈ 0.007.
        # 0.03 absolute tolerance is comfortably above noise, well below any
        # real distribution mismatch.
        assert torch.max(torch.abs(pr - pg)).item() < 0.03, (
            f"env {b}: empirical action distribution diverges between "
            f"per-env and batched sampling (max |Δfreq| "
            f"{torch.max(torch.abs(pr - pg)).item():.4f})"
        )


def test_sampled_actions_are_always_valid():
    """A sampled action must never land on a masked (-1e9) entry, for real
    workers, across many draws."""
    B, max_N, max_M = 5, 10, 8
    logits, n_w, n_t = _make_logits(B, max_N, max_M, 3)
    bd = Categorical(logits=logits)
    for _ in range(2000):
        s = bd.sample()
        for b in range(B):
            for w in range(n_w[b]):
                assert logits[b, w, s[b, w]].item() > MASK_FILL / 2, (
                    f"env {b} worker {w} sampled a masked task"
                )

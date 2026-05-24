"""Surgical checkpoint transplant: v2 foundation -> v2.1 (env_feat_dim 17->18).

The arch_version bump from "clark-v2" to "clark-v2.1" added one new
input dim to `self.env_linear` (the only layer that touches env_feats
directly). Every other layer is bit-identical. This tool:

  1. Loads a v2 checkpoint raw (bypassing ClarkAgent.load's strict
     version check, since the version is what's about to change).
  2. Builds a v2.1 model with env_feat_dim=18.
  3. Copies env_linear.weight (512, 17) -> new_weight[:, :17];
     ZERO-initialises the 18th column.
  4. Copies env_linear.bias unchanged.
  5. Copies every other tensor verbatim.
  6. Drops the optimizer state (would have shape-mismatched (512, 17)
     moments for env_linear). The new optimizer warms up in the first
     few hundred fine-tune steps — cheaper than maintaining matching
     opt state across an obs-shape change.
  7. Sanity-checks: a fresh forward pass on a dummy state with
     feats[17]=0 must produce IDENTICAL outputs to the source
     checkpoint (because zero-init column 17 contributes nothing).
  8. Saves a new .pt with arch_version="clark-v2.1".

Usage:
    python tools/transplant_obs_extension.py \\
        --src clark/data/checkpoints/clark_foundation.pt \\
        --dst clark/data/checkpoints/clark_foundation_dvc.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from clark.agent.ppo import ClarkAgent
from clark.agent.transformer import ARCH_VERSION, ClarkActorCritic, _ENV_FEAT_DIM


SRC_ARCH = "clark-v2"
DST_ARCH = "clark-v2.1"
SRC_ENV_DIM = 17
DST_ENV_DIM = 18


def transplant(src_path: Path, dst_path: Path) -> None:
    assert _ENV_FEAT_DIM == DST_ENV_DIM, (
        f"transformer._ENV_FEAT_DIM is {_ENV_FEAT_DIM}, expected {DST_ENV_DIM}. "
        f"This tool is calibrated for the v2->v2.1 bump only."
    )
    assert ARCH_VERSION == DST_ARCH, (
        f"transformer.ARCH_VERSION is {ARCH_VERSION!r}, expected {DST_ARCH!r}. "
        f"This tool only emits {DST_ARCH} checkpoints."
    )

    print(f"Loading source checkpoint: {src_path}")
    data = torch.load(src_path, map_location="cpu", weights_only=True)

    saved_version = data.get("arch_version", "unknown")
    if saved_version != SRC_ARCH:
        raise SystemExit(
            f"Source checkpoint arch_version is {saved_version!r}, "
            f"expected {SRC_ARCH!r}. This tool transplants v2 -> v2.1 only."
        )

    old_msd = dict(data["model_state_dict"])

    # ── Verify the env_linear shapes ────────────────────────────────
    env_w_key = "env_linear.weight"
    env_b_key = "env_linear.bias"
    if env_w_key not in old_msd:
        raise SystemExit(
            f"{env_w_key} not in source checkpoint. Either the model "
            f"architecture changed unexpectedly or this is the wrong file."
        )
    old_w = old_msd[env_w_key]
    if old_w.shape[1] != SRC_ENV_DIM:
        raise SystemExit(
            f"Source {env_w_key} has shape {tuple(old_w.shape)}, expected "
            f"(_, {SRC_ENV_DIM}). Refusing to transplant ambiguous tensor."
        )
    d_model = old_w.shape[0]
    print(f"  env_linear.weight: {tuple(old_w.shape)} (d_model={d_model})")

    # ── Build the new model (allocates env_linear at (d_model, 18)) ─
    print("Constructing v2.1 model...")
    hparams = data.get("hparams", {})
    # hparams contains BOTH model-arch hparams (d_model, n_heads, ...)
    # AND optimizer/training hparams (lr, clip_eps, ...) that get
    # passed to ClarkAgent.__init__. Filter to the model-arch subset
    # since we're constructing a bare ClarkActorCritic.
    import inspect
    model_param_names = set(inspect.signature(ClarkActorCritic.__init__).parameters)
    model_param_names.discard("self")
    model_hparams = {k: v for k, v in hparams.items() if k in model_param_names}
    print(f"  model hparams: {model_hparams}")
    new_model = ClarkActorCritic(**model_hparams)
    new_msd = new_model.state_dict()

    # ── Surgical copy + zero-init column 17 ─────────────────────────
    print("Transplanting weights...")
    transplanted = 0
    skipped = []
    for k, v in old_msd.items():
        if k == env_w_key:
            # Expand from (d_model, 17) -> (d_model, 18). First 17 cols
            # = old weights; col 17 = zeros so the new feature is
            # dormant on first forward pass.
            new_w = torch.zeros((d_model, DST_ENV_DIM), dtype=v.dtype)
            new_w[:, :SRC_ENV_DIM] = v
            new_msd[k] = new_w
            transplanted += 1
            print(f"  + {k}: {tuple(v.shape)} -> {tuple(new_w.shape)} (col 17 zero-init)")
            continue
        if k not in new_msd:
            skipped.append(k)
            continue
        if new_msd[k].shape != v.shape:
            raise SystemExit(
                f"Shape mismatch on {k}: source {tuple(v.shape)} vs "
                f"target {tuple(new_msd[k].shape)}. v2->v2.1 should only "
                f"change env_linear; this is unexpected."
            )
        new_msd[k] = v
        transplanted += 1

    if skipped:
        print(f"  ! {len(skipped)} key(s) in source but not in target: "
              f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    new_model.load_state_dict(new_msd, strict=True)
    print(f"  copied {transplanted}/{len(old_msd)} tensors")

    # ── Sanity check: dormant-feature equivalence ───────────────────
    print("Verifying zero-init dormancy (forward pass with feats[17]=0 "
          "must equal source model output)...")
    _verify_dormant(old_msd, new_model, d_model, hparams)
    print("  passed: outputs match to 1e-6.")

    # ── Save ────────────────────────────────────────────────────────
    print(f"Saving target checkpoint: {dst_path}")
    out = {
        "arch_version": DST_ARCH,
        "episode": data.get("episode", 0),
        "facility_config": data.get("facility_config", ""),
        "model_state_dict": new_msd,
        # Drop optimizer_state_dict — shape would have been (d_model, 17)
        # for env_linear's Adam moments. ClarkAgent.load tolerates a
        # missing key and builds a fresh optimizer.
        "hparams": hparams,
        "transplant_note": (
            f"warm-started from {src_path.name} via "
            f"tools/transplant_obs_extension.py; env_linear.weight col 17 "
            f"zero-init, optimizer reset. On first forward pass this "
            f"checkpoint is bit-identical to its source."
        ),
    }
    torch.save(out, dst_path)
    print(f"  size: {dst_path.stat().st_size / 1e6:.1f} MB")
    print("Done.")


def _verify_dormant(old_msd: dict, new_model: ClarkActorCritic,
                    d_model: int, hparams: dict) -> None:
    """Confirm zero-init col 17 makes new_model bit-identical to a v2
    model on env_feats[17]=0 input. If this fails we have a transplant
    bug and should NOT save the checkpoint."""
    # Build the source v2 model and load its state in-place.
    import inspect
    model_param_names = set(inspect.signature(ClarkActorCritic.__init__).parameters)
    model_param_names.discard("self")
    src_hparams = {k: v for k, v in hparams.items() if k in model_param_names}
    src_model = _build_v2_model(src_hparams)
    src_model.load_state_dict(old_msd, strict=False)
    src_model.eval()
    new_model.eval()

    # Synthesize a minimally-shaped state dict — 2 workers, 3 tasks.
    N, M = 2, 3
    state_old = {
        "worker_feats": np.zeros((N, 14), dtype=np.float32),
        "task_feats": np.zeros((M, 3), dtype=np.float32),
        "env_feats": np.zeros((SRC_ENV_DIM,), dtype=np.float32),
        "worker_role_ids": np.zeros((N,), dtype=np.int64),
        "task_type_ids": np.arange(M, dtype=np.int64),
    }
    state_new = dict(state_old)
    state_new["env_feats"] = np.zeros((DST_ENV_DIM,), dtype=np.float32)

    action_mask = np.ones((N, M), dtype=bool)
    hustle_mask = np.ones((N, 2), dtype=bool)

    with torch.no_grad():
        _, _, v_old, _ = src_model.forward(
            state_old, action_mask=action_mask, hustle_mask=hustle_mask,
            lstm_hidden=None,
        )
        _, _, v_new, _ = new_model.forward(
            state_new, action_mask=action_mask, hustle_mask=hustle_mask,
            lstm_hidden=None,
        )

    diff = float(torch.abs(v_old - v_new).max())
    if diff > 1e-6:
        raise SystemExit(
            f"Dormancy verification FAILED: value-head outputs differ by "
            f"{diff} (must be < 1e-6). Refusing to save the transplant — "
            f"the zero-init contract is broken and warm-starting from "
            f"this checkpoint would degrade the policy."
        )


def _build_v2_model(hparams: dict) -> ClarkActorCritic:
    """Build a *v2-shape* model (env_feat_dim=17) for the dormancy
    check. Python evaluates default-arg expressions at function-
    definition time, so monkey-patching _ENV_FEAT_DIM doesn't help —
    pass env_feat_dim explicitly instead."""
    kwargs = dict(hparams)
    kwargs["env_feat_dim"] = SRC_ENV_DIM
    return ClarkActorCritic(**kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path,
                   help="Source v2 checkpoint (.pt)")
    p.add_argument("--dst", required=True, type=Path,
                   help="Target v2.1 checkpoint path (.pt)")
    args = p.parse_args()

    if not args.src.exists():
        raise SystemExit(f"source not found: {args.src}")
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    if args.dst.exists():
        print(f"WARNING: {args.dst} exists and will be overwritten.")

    transplant(args.src, args.dst)


if __name__ == "__main__":
    main()

"""Surgical checkpoint transplant: clark-v2 source -> clark-v2.5 target.

v2.5 adds env_feat 17 = mgmt_backlog_norm. The env_linear input layer
expands from (d_model, 17) -> (d_model, 18). Every other parameter is
bit-identical to v2/v2.7. This tool:

  1. Loads a v2 checkpoint raw (bypassing ClarkAgent.load's version check).
  2. Builds a v2.5 model with env_feat_dim=18.
  3. Copies env_linear.weight (d_model, 17) -> new_weight[:, :17]; ZERO-init col 17.
  4. Copies env_linear.bias and every other tensor verbatim.
  5. Drops optimizer state (shape mismatch on env_linear moments).
  6. Sanity-checks: feats[17]=0 -> identical output (zero-init dormancy).
  7. Saves with arch_version="clark-v2.5".

Usage:
    python tools/transplant_obs_extension.py \\
        --src clark/data/checkpoints/clark_foundation_v27_ft.pt \\
        --dst clark/data/checkpoints/clark_foundation_v28_ft.pt
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch

from clark.agent.transformer import ARCH_VERSION, ClarkActorCritic, _ENV_FEAT_DIM


ALLOWED_SRC = "clark-v2"
DST_ARCH = "clark-v2.5"
SRC_ENV_DIM = 17
DST_ENV_DIM = 18


def transplant(src_path: Path, dst_path: Path) -> None:
    assert _ENV_FEAT_DIM == DST_ENV_DIM
    assert ARCH_VERSION == DST_ARCH

    print(f"Loading source: {src_path}")
    data = torch.load(src_path, map_location="cpu", weights_only=True)
    saved = data.get("arch_version", "unknown")
    if saved != ALLOWED_SRC:
        raise SystemExit(f"Source arch {saved!r}; only {ALLOWED_SRC!r} accepted.")

    old_msd = dict(data["model_state_dict"])
    old_w = old_msd["env_linear.weight"]
    if old_w.shape[1] != SRC_ENV_DIM:
        raise SystemExit(f"env_linear shape {tuple(old_w.shape)} unexpected.")
    d_model = old_w.shape[0]

    hparams = data.get("hparams", {})
    model_keys = set(inspect.signature(ClarkActorCritic.__init__).parameters)
    model_keys.discard("self")
    model_hparams = {k: v for k, v in hparams.items() if k in model_keys}
    new_model = ClarkActorCritic(**model_hparams)
    new_msd = new_model.state_dict()

    print(f"Transplanting {len(old_msd)} tensors...")
    for k, v in old_msd.items():
        if k == "env_linear.weight":
            new_w = torch.zeros((d_model, DST_ENV_DIM), dtype=v.dtype)
            new_w[:, :SRC_ENV_DIM] = v
            new_msd[k] = new_w
            print(f"  + {k}: {tuple(v.shape)} -> {tuple(new_w.shape)} (col 17 zero-init)")
            continue
        if k in new_msd and new_msd[k].shape == v.shape:
            new_msd[k] = v
    new_model.load_state_dict(new_msd, strict=True)

    print("Verifying dormancy (feats[17]=0 must match source)...")
    _verify_dormant(old_msd, new_model, hparams)
    print("  passed.")

    print(f"Saving: {dst_path}")
    out = {
        "arch_version": DST_ARCH,
        "episode": data.get("episode", 0),
        "facility_config": data.get("facility_config", ""),
        "model_state_dict": new_msd,
        "hparams": hparams,
        "transplant_note": f"v2 source ({src_path.name}) -> v2.5; env_linear col 17 zero-init.",
    }
    torch.save(out, dst_path)
    print(f"  size: {dst_path.stat().st_size / 1e6:.1f} MB")
    print("Done.")


def _verify_dormant(old_msd, new_model, hparams):
    model_keys = set(inspect.signature(ClarkActorCritic.__init__).parameters)
    model_keys.discard("self")
    src_hp = {k: v for k, v in hparams.items() if k in model_keys}
    src_hp["env_feat_dim"] = SRC_ENV_DIM
    src_model = ClarkActorCritic(**src_hp)
    src_model.load_state_dict(old_msd, strict=False)
    src_model.eval(); new_model.eval()

    N, M = 2, 3
    s_old = {
        "worker_feats": np.zeros((N, 14), dtype=np.float32),
        "task_feats": np.zeros((M, 3), dtype=np.float32),
        "env_feats": np.zeros((SRC_ENV_DIM,), dtype=np.float32),
        "worker_role_ids": np.zeros((N,), dtype=np.int64),
        "task_type_ids": np.arange(M, dtype=np.int64),
    }
    s_new = dict(s_old); s_new["env_feats"] = np.zeros((DST_ENV_DIM,), dtype=np.float32)
    am = np.ones((N, M), dtype=bool); hm = np.ones((N, 2), dtype=bool)
    with torch.no_grad():
        _, _, v_old, _ = src_model.forward(s_old, action_mask=am, hustle_mask=hm, lstm_hidden=None)
        _, _, v_new, _ = new_model.forward(s_new, action_mask=am, hustle_mask=hm, lstm_hidden=None)
    diff = float(torch.abs(v_old - v_new).max())
    if diff > 1e-6:
        raise SystemExit(f"Dormancy verification FAILED: diff={diff}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    args = p.parse_args()
    if not args.src.exists():
        raise SystemExit(f"missing: {args.src}")
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    transplant(args.src, args.dst)

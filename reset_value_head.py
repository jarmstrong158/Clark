"""One-shot value-head reset for Clark.

Per round-2 audit: value_head.2 was found catastrophically saturated
(weight std=28.9, absmax=29, bias -28.84 cancels weight). Probe test
showed value_head outputs std≈3000 across random states — the head
became a high-gain near-linear map of an already-saturated Tanh. EMA
return normalization in PPO was hiding this from v_loss metrics.

This script:
1. Loads the checkpoint
2. Re-initializes BOTH linears in value_head (orthogonal gain=1)
3. Zeros their Adam moments so optimizer state restarts for these tensors
4. Saves a backup of the original, writes the patched checkpoint in place

The agent's _ret_mean / _ret_var are NOT in the checkpoint (instance
attrs only) so they reset on next launch automatically — no need to
touch them here. The brief warmup period (~50-100 updates) where they
re-converge will give the fresh value head room to anchor against the
new return distribution.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

CKPT = Path(r"C:/Users/jarms/repos/clark/clark/data/checkpoints/clark_foundation.pt")
BACKUP = CKPT.with_suffix(".pt.preValueHeadReset.bak")


def main() -> int:
    if not CKPT.exists():
        print(f"checkpoint not found: {CKPT}")
        return 1

    print(f"Loading {CKPT.name}...")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    msd = ckpt["model_state_dict"]
    osd = ckpt["optimizer_state_dict"]

    # Sanity dump of CURRENT value head state
    print("\n=== BEFORE ===")
    for key in ("value_head.0.weight", "value_head.0.bias",
                "value_head.2.weight", "value_head.2.bias"):
        if key in msd:
            t = msd[key]
            print(f"  {key:>30}: shape={tuple(t.shape)} mean={t.mean().item():+.4f} "
                  f"std={t.std().item():.4f} absmax={t.abs().max().item():.4f}")

    # Re-init both linears
    print("\nReinitializing value_head[0] and value_head[2]...")
    for layer_idx in (0, 2):
        wkey = f"value_head.{layer_idx}.weight"
        bkey = f"value_head.{layer_idx}.bias"
        if wkey not in msd:
            print(f"  WARN: {wkey} not in state_dict, skipping")
            continue
        w = msd[wkey]
        # Orthogonal init, gain=1 for the value head (matches model._init_weights for value_head)
        new_w = torch.empty_like(w)
        nn.init.orthogonal_(new_w, gain=1.0)
        msd[wkey] = new_w
        if bkey in msd:
            msd[bkey] = torch.zeros_like(msd[bkey])

    # Zero Adam moments for the reset tensors. Optimizer state is keyed
    # by param INDEX (an int), not by name. We need to find which indices
    # correspond to value_head[0] and value_head[2] params. The model's
    # named_parameters() iteration order matches optimizer param-group
    # order (Adam was constructed via self.model.parameters()).
    #
    # We don't have the model object here; rebuild the iteration order
    # by counting from the state_dict keys. Easier: optimizer state_dict
    # gives us the keys directly under osd["state"], and each entry has
    # exp_avg / exp_avg_sq that match the parameter shape. So we match
    # by SHAPE: there are exactly 4 value-head tensors with their unique
    # shapes (Linear weight + bias for each of the 2 linears), and the
    # value head MLP is the only place with these shape combinations.
    print("\nLocating + zeroing Adam moments for value_head params...")
    target_shapes = {
        msd[f"value_head.{i}.weight"].shape: f"value_head.{i}.weight" for i in (0, 2)
    }
    target_shapes.update({
        msd[f"value_head.{i}.bias"].shape: f"value_head.{i}.bias" for i in (0, 2)
    })

    state = osd.get("state", {})
    zeroed = []
    # The value head's two weights have shapes (256, 512) and (1, 256).
    # Biases are (256,) and (1,). Match by exact shape — no other layers
    # in the model have shape (1, 256) or shape (1,).
    for param_idx, st in state.items():
        if "exp_avg" not in st:
            continue
        sh = st["exp_avg"].shape
        if sh in target_shapes and target_shapes[sh].startswith("value_head"):
            label = target_shapes[sh]
            # Disambiguate by the OTHER tensor — both biases are 1D, the
            # (256,) bias is value_head.0 and the (1,) bias is value_head.2.
            # Both weights have unique shapes already.
            st["exp_avg"] = torch.zeros_like(st["exp_avg"])
            st["exp_avg_sq"] = torch.zeros_like(st["exp_avg_sq"])
            if "step" in st:
                # Reset the per-tensor step so bias correction restarts.
                if torch.is_tensor(st["step"]):
                    st["step"] = torch.zeros_like(st["step"])
                else:
                    st["step"] = 0
            zeroed.append((param_idx, label, sh))

    print(f"  zeroed Adam state for {len(zeroed)} param entries:")
    for pi, lbl, sh in zeroed:
        print(f"    param {pi}: {lbl} shape={tuple(sh)}")

    if len(zeroed) != 4:
        print(f"\nWARN: expected to zero 4 tensors (2 weight + 2 bias), zeroed {len(zeroed)}.")
        print("Some Adam moments may not have been reset. Continuing — fresh weights")
        print("will still converge faster than untouched, but old moments may slow it.")

    # Backup + save
    if not BACKUP.exists():
        print(f"\nBacking up to {BACKUP.name}...")
        shutil.copy2(CKPT, BACKUP)
    else:
        print(f"\nBackup already exists at {BACKUP.name} — not overwriting.")

    print(f"Writing patched checkpoint to {CKPT.name}...")
    torch.save(ckpt, CKPT)

    # Verify
    ckpt2 = torch.load(CKPT, map_location="cpu", weights_only=False)
    msd2 = ckpt2["model_state_dict"]
    print("\n=== AFTER ===")
    for key in ("value_head.0.weight", "value_head.0.bias",
                "value_head.2.weight", "value_head.2.bias"):
        if key in msd2:
            t = msd2[key]
            print(f"  {key:>30}: shape={tuple(t.shape)} mean={t.mean().item():+.4f} "
                  f"std={t.std().item():.4f} absmax={t.abs().max().item():.4f}")

    print("\nDone. Restart training to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

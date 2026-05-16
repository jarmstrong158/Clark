"""Salvage a Clark checkpoint by resetting the value head and optimizer state.

Use this when a checkpoint's value head has been poisoned by extreme reward
values (e.g. carryover-explosion bug producing 1e75 returns), but the encoder,
LSTM, and policy heads still hold useful representations from earlier
non-corrupted episodes.

What it keeps:
    - encoder (self-attention, cross-attention, embeddings, projections)
    - LSTM temporal memory
    - policy head (assignment logits via W @ T.T)
    - hustle head

What it resets:
    - value_head weights (orthogonal init)
    - optimizer state (momentum/variance poisoned by bad gradients)
    - episode counter (so pretrain resumes from ep 1 with clean gradients)

Usage:
    python salvage_checkpoint.py \\
        --input  clark/data/checkpoints/clark_foundation.pt \\
        --output clark/data/checkpoints/clark_foundation_salvaged.pt
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn


def salvage(input_path: str, output_path: str) -> None:
    print(f"Loading checkpoint: {input_path}")
    data = torch.load(input_path, weights_only=False)

    print(f"  Arch version: {data.get('arch_version', 'unknown')}")
    print(f"  Episode:      {data.get('episode', '?')}")
    print(f"  Hparams:      {data.get('hparams', {})}")

    model_sd = data["model_state_dict"]

    # Identify value head parameters
    value_keys = [k for k in model_sd if "value_head" in k]
    print(f"\n  Value head parameters to reset ({len(value_keys)}):")
    for k in value_keys:
        print(f"    {k}: {model_sd[k].shape}")

    # Re-initialise value head weights
    for k in value_keys:
        t = model_sd[k]
        if "weight" in k and t.dim() >= 2:
            nn.init.orthogonal_(t, gain=1.0)
            print(f"    RESET {k} (orthogonal, gain=1.0)")
        elif "bias" in k:
            nn.init.constant_(t, 0.0)
            print(f"    RESET {k} (zeros)")

    # Show what we're keeping
    kept_keys = [k for k in model_sd if "value_head" not in k]
    print(f"\n  Keeping {len(kept_keys)} parameter tensors "
          f"(encoder, LSTM, policy, hustle):")
    groups: dict[str, int] = {}
    for k in kept_keys:
        prefix = k.split(".")[0]
        groups[prefix] = groups.get(prefix, 0) + 1
    for prefix, count in sorted(groups.items()):
        print(f"    {prefix}: {count} tensors")

    # Drop optimizer state (momentum/variance poisoned by bad gradients)
    print(f"\n  Dropping optimizer state (poisoned momentum/variance)")

    salvaged = {
        "arch_version": data["arch_version"],
        "episode": 0,  # reset so pretrain starts at ep 1
        "facility_config": None,
        "model_state_dict": model_sd,
        "optimizer_state_dict": None,  # force fresh optimizer
        "hparams": data.get("hparams", {}),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(salvaged, output_path)
    print(f"\n  Saved salvaged checkpoint: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    print(f"\nDone. To resume training with the salvaged weights:")
    print(f"  cp {output_path} {input_path}")
    print(f"  clark pretrain --output {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset value head + optimizer state on a Clark checkpoint."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the checkpoint to salvage.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to write the salvaged checkpoint.",
    )
    args = parser.parse_args()
    salvage(args.input, args.output)


if __name__ == "__main__":
    main()

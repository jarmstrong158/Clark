"""Tiny wrapper: clark finetune, but on GPU.

`clark finetune` defaults the agent to device='cpu'. For one-off
faster runs (Jack-comparison experiment, etc.) without modifying the
public CLI, this script loads the agent on cuda and calls the same
finetune() internals.

The base checkpoint is opened READ-ONLY; output goes to a SEPARATE
file (never overwrite clark_foundation.pt).

Usage:
  python tools/finetune_gpu.py \
    --config clark/data/configs/jack_baseline.yaml \
    --base   clark/data/checkpoints/clark_foundation.pt \
    --output clark/data/checkpoints/clark_jack_ft.pt \
    --episodes 500
"""
from __future__ import annotations
import argparse, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--freeze-encoder", action="store_true")
    a = ap.parse_args()

    assert os.path.abspath(a.base) != os.path.abspath(a.output), (
        "refusing to overwrite the base checkpoint")

    print(f"[finetune_gpu] python {sys.version.split()[0]}", flush=True)
    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — use `clark finetune` (CPU) instead")
    print(f"[finetune_gpu] device: cuda ({torch.cuda.get_device_name(0)}, "
          f"cap {torch.cuda.get_device_capability(0)})", flush=True)

    from clark.agent.ppo import ClarkAgent
    from clark.training.finetune import finetune

    # The finetune() function accepts a path and re-loads the agent
    # internally on CPU. We monkey-patch ClarkAgent.load so its default
    # device becomes 'cuda' for this run — minimally invasive, doesn't
    # require changing finetune.py.
    _orig = ClarkAgent.load.__func__

    def _gpu_load(cls, path, facility_config=None, device="cuda",
                  **kw):
        return _orig(cls, path, facility_config=facility_config,
                     device=device, **kw)
    ClarkAgent.load = classmethod(_gpu_load)

    t0 = time.time()
    finetune(
        config_path=a.config,
        base_model_path=a.base,
        output_path=a.output,
        n_episodes=a.episodes,
        lr=a.lr,
        freeze_encoder=a.freeze_encoder,
    )
    print(f"[finetune_gpu] done in {time.time()-t0:.0f}s -> {a.output}",
          flush=True)


if __name__ == "__main__":
    main()

"""Probe script: post-reset value-head diagnostics."""
import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clark.agent.transformer import ClarkActorCritic

CKPT = "C:/Users/jarms/repos/clark/clark/data/checkpoints/clark_foundation.pt"
METRICS = "C:/Users/jarms/repos/clark/clark/data/logs/pretrain/training_metrics.json"

# ── Load checkpoint once ──────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print("ckpt keys:", list(ckpt.keys()))

# Find state_dict
sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model")
opt = ckpt.get("optimizer_state_dict") or ckpt.get("optimizer")
update_idx = ckpt.get("update_idx") or ckpt.get("global_step") or ckpt.get("step")
print("update_idx:", update_idx)

# value_head keys
vh_keys = [k for k in sd.keys() if "value_head" in k]
print("value_head keys:", vh_keys)

# ── Q1: Did the reset stick? ─────────────────────────────────────────────────
print("\n========== Q1: VALUE HEAD WEIGHT MAGNITUDES ==========")
for k in vh_keys:
    w = sd[k]
    print(f"{k:35s}  shape={tuple(w.shape)}  std={w.std().item():.5f}  absmax={w.abs().max().item():.5f}  mean={w.mean().item():.5f}")

# ── Build model and load ────────────────────────────────────────────────────
model = ClarkActorCritic()
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", missing[:5], "unexpected:", unexpected[:5])
model.eval()

# Param index map for optimizer
param_list = list(model.named_parameters())
print("\nTotal params:", len(param_list))
for i, (n, p) in enumerate(param_list):
    if "value_head" in n:
        print(f"  idx {i}: {n}  shape={tuple(p.shape)}")

# ── Q2: Output distribution across 64 sample states ─────────────────────────
print("\n========== Q2: VALUE HEAD OUTPUT ACROSS 64 RANDOM STATES ==========")
rng = np.random.default_rng(0)
values = []
for i in range(64):
    N = int(rng.integers(5, 26))
    M = int(rng.integers(8, 16))
    state = {
        "worker_feats": rng.standard_normal((N, 14)).astype(np.float32),
        "task_feats": np.concatenate(
            [rng.standard_normal((M, 2)).astype(np.float32),
             rng.integers(0, M, size=(M, 1)).astype(np.float32)], axis=1),
        "env_feats": rng.standard_normal(17).astype(np.float32),
        "worker_role_ids": rng.integers(0, 4, size=N).astype(np.int64),
        "task_type_ids": rng.integers(0, M, size=M).astype(np.int64),
    }
    with torch.no_grad():
        _, _, v, _ = model(state)
    values.append(v.item())

values = np.array(values)
print(f"value outputs: mean={values.mean():.4f}  std={values.std():.4f}  min={values.min():.4f}  max={values.max():.4f}  range={values.max()-values.min():.4f}")
print(f"  std-across-states (responsiveness): {values.std():.6f}")
print(f"  first 8 values: {values[:8]}")

# ── Q3: Optimizer state ──────────────────────────────────────────────────────
print("\n========== Q3: ADAM MOMENTS FOR VALUE HEAD ==========")
if opt is not None and "state" in opt:
    opt_state = opt["state"]
    print(f"opt_state keys (first 8): {list(opt_state.keys())[:8]}")
    print(f"total opt entries: {len(opt_state)}")
    # Inspect indices 71-74
    for idx in [71, 72, 73, 74]:
        if idx in opt_state:
            s = opt_state[idx]
            ea = s.get("exp_avg")
            eas = s.get("exp_avg_sq")
            step = s.get("step")
            ea_mag = ea.abs().mean().item() if ea is not None else None
            eas_mag = eas.abs().mean().item() if eas is not None else None
            ea_max = ea.abs().max().item() if ea is not None else None
            print(f"  idx {idx}: step={step}  exp_avg |mean|={ea_mag:.3e}  |max|={ea_max:.3e}  exp_avg_sq |mean|={eas_mag:.3e}")
        else:
            print(f"  idx {idx}: NOT in opt state")

    # Compute average exp_avg magnitudes for non-value-head layers for comparison
    other_ea = []
    for idx, s in opt_state.items():
        if idx in (71, 72, 73, 74):
            continue
        ea = s.get("exp_avg")
        if ea is not None and ea.numel() > 0:
            other_ea.append(ea.abs().mean().item())
    if other_ea:
        print(f"\n  other layers exp_avg |mean|:  median={np.median(other_ea):.3e}  mean={np.mean(other_ea):.3e}")
    # Ratio
    vh_ea = []
    for idx in (71, 72, 73, 74):
        if idx in opt_state:
            ea = opt_state[idx].get("exp_avg")
            if ea is not None:
                vh_ea.append(ea.abs().mean().item())
    if vh_ea and other_ea:
        print(f"  value_head/other ratio: {np.mean(vh_ea)/np.median(other_ea):.4f}")
else:
    print("NO optimizer state in checkpoint")

# ── Q4: Encoder/LSTM weight stats ────────────────────────────────────────────
print("\n========== Q4: ENCODER + LSTM STATS ==========")
groups = {
    "worker_linear": [], "task_linear": [], "env_linear": [],
    "sa_layers": [], "ca_layers": [], "ca_ff": [],
    "lstm": [], "hustle_head": [], "value_head": [],
    "role_embed": [], "task_type_embed": [],
}
for k, w in sd.items():
    for g in groups:
        if k.startswith(g) or f".{g}." in k:
            groups[g].append((k, w))
            break

for g, items in groups.items():
    if not items:
        continue
    stds = [w.std().item() for _, w in items if w.numel() > 1]
    absmax = [w.abs().max().item() for _, w in items]
    print(f"  {g:20s}  n_tensors={len(items):3d}  std med={np.median(stds):.4f}  absmax med={np.median(absmax):.4f}  absmax overall={max(absmax):.4f}")

# ── Q5: Bimodality from training_metrics ─────────────────────────────────────
print("\n========== Q5: V_LOSS BIMODALITY FROM TRAINING METRICS ==========")
with open(METRICS) as f:
    metrics = json.load(f)

# metrics shape?
if isinstance(metrics, dict):
    print("metrics keys:", list(metrics.keys())[:10])
    # try common shapes
    if "ppo_updates" in metrics:
        updates = metrics["ppo_updates"]
    elif "updates" in metrics:
        updates = metrics["updates"]
    else:
        updates = None
elif isinstance(metrics, list):
    updates = metrics

if updates:
    print(f"total updates: {len(updates)}")
    print(f"first entry keys: {list(updates[0].keys())[:15] if isinstance(updates[0], dict) else 'N/A'}")
    v_losses = []
    for u in updates:
        if isinstance(u, dict):
            vl = u.get("v_loss") or u.get("value_loss") or u.get("vf_loss")
            if vl is not None:
                v_losses.append(float(vl))
    print(f"v_loss samples: {len(v_losses)}")
    if v_losses:
        v = np.array(v_losses)
        # post-reset window — last ~232
        post = v[-232:] if len(v) >= 232 else v
        print(f"\nALL ({len(v)}):  median={np.median(v):.4f}  mean={v.mean():.4f}  std={v.std():.4f}  min={v.min():.4f}  max={v.max():.4f}")
        print(f"POST-RESET ({len(post)}):  median={np.median(post):.4f}  mean={post.mean():.4f}  std={post.std():.4f}  min={post.min():.4f}  max={post.max():.4f}")

        # Bimodality cutoff analysis
        for cutoff in [1.0, 5.0, 10.0, 50.0]:
            frac_below = (post < cutoff).mean()
            frac_above = (post >= cutoff).mean()
            print(f"  cutoff={cutoff:6.1f}:  below={frac_below*100:.1f}%  above={frac_above*100:.1f}%   above-mean={post[post>=cutoff].mean() if (post>=cutoff).any() else 0:.2f}")

        # Percentile breakdown
        print(f"  percentiles post-reset: 50={np.percentile(post,50):.3f}  90={np.percentile(post,90):.3f}  95={np.percentile(post,95):.3f}  99={np.percentile(post,99):.3f}")

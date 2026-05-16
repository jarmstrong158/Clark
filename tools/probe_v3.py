"""PROBE-AGENT-V3: stage-3 transition value-head + PPO-health audit."""
import sys, os, json, gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clark.agent.transformer import ClarkActorCritic

CKPT = "C:/Users/jarms/repos/clark/clark/data/checkpoints/clark_foundation.pt"
METRICS = "C:/Users/jarms/repos/clark/clark/data/logs/pretrain/training_metrics.json"
EPLOG = "C:/Users/jarms/repos/clark/clark/data/logs/pretrain/episode_log.json"

# === LOAD CHECKPOINT ===
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model")
opt = ckpt.get("optimizer_state_dict") or ckpt.get("optimizer")
update_idx = ckpt.get("update_idx") or ckpt.get("global_step") or ckpt.get("step")
print(f"=== ckpt update_idx={update_idx}  keys={list(ckpt.keys())}")

vh_keys = sorted([k for k in sd.keys() if "value_head" in k])
print(f"value_head tensors: {vh_keys}")

# === Q1: VALUE HEAD WEIGHT STATS ===
print("\n========== Q1: VALUE HEAD WEIGHT STATS (CURRENT) ==========")
for k in vh_keys:
    w = sd[k].float()
    sat = (w.abs() > 5.0).float().mean().item()
    print(f"{k:40s} shape={tuple(w.shape)} mean={w.mean().item():+.5f} std={w.std().item():.5f} absmax={w.abs().max().item():.5f} frac|w|>5={sat:.4f}")

# === BUILD MODEL ===
model = ClarkActorCritic()
missing, unexpected = model.load_state_dict(sd, strict=False)
model.eval()
print(f"missing={len(missing)} unexpected={len(unexpected)}")
if missing[:3]: print(" miss sample:", missing[:3])
if unexpected[:3]: print(" unexp sample:", unexpected[:3])

named = list(model.named_parameters())
print("\nvalue_head param indices:")
vh_idx = []
for i, (n, p) in enumerate(named):
    if "value_head" in n:
        print(f"  idx {i}: {n} shape={tuple(p.shape)}")
        vh_idx.append(i)

# === Q2: ADAM MOMENT SIGN DISTRIBUTION ===
print("\n========== Q2: ADAM MOMENTS FOR VALUE HEAD ==========")
opt_state = opt["state"] if (opt and "state" in opt) else None
if opt_state is None:
    print("NO opt state")
else:
    for idx in vh_idx:
        if idx not in opt_state:
            print(f"  idx {idx}: NOT in opt_state")
            continue
        s = opt_state[idx]
        ea = s.get("exp_avg")
        eas = s.get("exp_avg_sq")
        step = s.get("step")
        if ea is None:
            continue
        ea_f = ea.float().flatten()
        pos_frac = (ea_f > 0).float().mean().item()
        neg_frac = (ea_f < 0).float().mean().item()
        zero_frac = (ea_f == 0).float().mean().item()
        ea_mag = ea_f.abs().mean().item()
        ea_max = ea_f.abs().max().item()
        eas_mag = eas.float().abs().mean().item() if eas is not None else None
        # effective update direction = ea / sqrt(eas)
        if eas is not None:
            eff = (ea_f / (eas.float().flatten().sqrt() + 1e-8))
            eff_mean = eff.mean().item()
            eff_pos = (eff > 0).float().mean().item()
        else:
            eff_mean = eff_pos = None
        name = named[idx][0]
        print(f"  idx {idx} ({name}): step={step}")
        print(f"    exp_avg: |mean|={ea_mag:.3e} |max|={ea_max:.3e} pos%={pos_frac*100:.1f} neg%={neg_frac*100:.1f} zero%={zero_frac*100:.1f}")
        print(f"    exp_avg_sq |mean|={eas_mag:.3e}")
        print(f"    eff update: mean={eff_mean:+.4e} pos%={eff_pos*100:.1f} (>70 or <30 => sign-locked)")

# === Q3: VALUE OUTPUT DIST ACROSS 64 STATES (incl large N) ===
print("\n========== Q3: VALUE HEAD OUTPUT ACROSS 64 RANDOM STATES ==========")
rng = np.random.default_rng(42)
values = []
values_by_bucket = {"small": [], "mid": [], "large": []}
for i in range(64):
    if i < 20:
        N = int(rng.integers(5, 11)); bucket = "small"
    elif i < 44:
        N = int(rng.integers(11, 20)); bucket = "mid"
    else:
        N = int(rng.integers(30, 51)); bucket = "large"
    M = int(rng.integers(max(2, min(8, N)), max(3, min(16, N+1))))
    state = {
        "worker_feats": rng.standard_normal((N, 14)).astype(np.float32),
        "task_feats": np.concatenate(
            [rng.standard_normal((M, 2)).astype(np.float32),
             rng.integers(0, M, size=(M,1)).astype(np.float32)], axis=1),
        "env_feats": rng.standard_normal(17).astype(np.float32),
        "worker_role_ids": rng.integers(0, 4, size=N).astype(np.int64),
        "task_type_ids": rng.integers(0, M, size=M).astype(np.int64),
    }
    try:
        with torch.no_grad():
            out = model(state)
        v = out[2] if isinstance(out, tuple) else out["value"]
        vv = float(v.item() if hasattr(v, "item") else np.asarray(v).flatten()[0])
        values.append(vv)
        values_by_bucket[bucket].append(vv)
    except Exception as e:
        print(f" forward fail N={N} M={M}: {e}")
        break

values = np.array(values)
print(f"ALL: n={len(values)} mean={values.mean():+.4f} std={values.std():.4f} min={values.min():+.4f} max={values.max():+.4f} range={values.max()-values.min():.4f}")
for b, vs in values_by_bucket.items():
    if vs:
        a = np.array(vs)
        print(f"  {b:5s} (n={len(a)}): mean={a.mean():+.4f} std={a.std():.4f} min={a.min():+.4f} max={a.max():+.4f}")

del model
gc.collect()

# === Q4: PPO HEALTH OVER LAST 500 UPDATES, CHUNKS OF 100 ===
print("\n========== Q4: PPO HEALTH ACROSS STAGE-3 TRANSITION ==========")
with open(METRICS) as f:
    metrics = json.load(f)

if isinstance(metrics, dict):
    if "ppo_updates" in metrics:
        updates = metrics["ppo_updates"]
    elif "updates" in metrics:
        updates = metrics["updates"]
    else:
        # maybe top-level lists
        print("metrics dict keys:", list(metrics.keys())[:20])
        updates = None
else:
    updates = metrics

print(f"total ppo updates: {len(updates) if updates else 0}")
if updates:
    print(f"first update keys: {list(updates[0].keys())[:25]}")
    last = updates[-min(500, len(updates)):]
    n = len(last)
    chunk = 100
    print(f"\nLast {n} updates, chunked by {chunk}:")
    print(f"{'chunk':10s} {'clip%':>8s} {'vL_med':>10s} {'vL_std':>10s} {'pL':>10s} {'ent':>8s} {'stage':>6s} {'ts':>20s}")
    chunks = []
    for i in range(0, n, chunk):
        sub = last[i:i+chunk]
        if not sub: continue
        def col(k):
            vs = []
            for u in sub:
                v = u.get(k)
                if v is None:
                    # try alt names
                    if k=="v_loss": v = u.get("value_loss") or u.get("vf_loss")
                    if k=="p_loss": v = u.get("policy_loss") or u.get("pg_loss")
                    if k=="clip": v = u.get("clip_frac") or u.get("clip_fraction")
                if v is not None:
                    try: vs.append(float(v))
                    except: pass
            return vs
        cl = col("clip"); vl = col("v_loss"); pl = col("p_loss"); en = col("entropy")
        stages = [u.get("stage") for u in sub if u.get("stage") is not None]
        stage = stages[-1] if stages else "?"
        ts = sub[-1].get("timestamp") or sub[-1].get("ts") or sub[-1].get("time") or ""
        if isinstance(ts, (int, float)):
            import datetime
            ts = datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S")
        clip_m = np.mean(cl)*100 if cl else float("nan")
        if cl and max(cl) > 1.5:  # already in %
            clip_m = np.mean(cl)
        vl_med = np.median(vl) if vl else float("nan")
        vl_std = np.std(vl) if vl else float("nan")
        pl_m = np.mean(pl) if pl else float("nan")
        en_m = np.mean(en) if en else float("nan")
        print(f"  [{i:3d}-{i+len(sub):3d}] {clip_m:8.2f} {vl_med:10.4f} {vl_std:10.4f} {pl_m:+10.4f} {en_m:8.3f} {str(stage):>6s} {str(ts):>20s}")
        chunks.append((i, clip_m, vl_med, vl_std, pl_m, en_m, stage, ts))

    # Detect when v_loss std starts climbing
    print("\nDetecting v_loss std jump:")
    prev_std = None
    for c in chunks:
        i, _, vl_med, vl_std, *_ = c
        if prev_std is not None and vl_std > prev_std * 2 and vl_std > 1.0:
            print(f"  >>> v_loss std jumped at chunk starting {i}: {prev_std:.3f} -> {vl_std:.3f}")
        prev_std = vl_std

    # Stage transition timestamp
    stage_changes = []
    last_stage = None
    for j, u in enumerate(updates):
        s = u.get("stage")
        if s != last_stage:
            stage_changes.append((j, s, u.get("timestamp") or u.get("ts")))
            last_stage = s
    print(f"\nStage transitions (last 5):")
    for sc in stage_changes[-5:]:
        j, s, ts = sc
        if isinstance(ts, (int, float)):
            import datetime
            ts = datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S")
        print(f"  update {j}: stage -> {s}  ts={ts}")

# === Q5: RETURN SCALE BY N ===
print("\n========== Q5: RETURN SCALE BY N (recent episodes) ==========")
try:
    with open(EPLOG) as f:
        eps = json.load(f)
    if isinstance(eps, dict):
        eps = eps.get("episodes") or list(eps.values())
    print(f"episodes loaded: {len(eps)}")
    if eps:
        print(f"keys sample: {list(eps[-1].keys())[:25]}")
    recent = eps[-400:] if len(eps) >= 400 else eps
    by_bucket = {"N<=10": [], "N=11-20": [], "N=21-35": [], "N>35": []}
    for e in recent:
        N = e.get("N") or e.get("n_workers") or e.get("num_workers") or e.get("workers")
        if N is None: continue
        N = int(N)
        # try several reward fields
        r = e.get("total_reward") or e.get("episode_reward") or e.get("reward") or e.get("return")
        if r is None: continue
        r = float(r)
        if N <= 10: by_bucket["N<=10"].append(r)
        elif N <= 20: by_bucket["N=11-20"].append(r)
        elif N <= 35: by_bucket["N=21-35"].append(r)
        else: by_bucket["N>35"].append(r)
    for b, vs in by_bucket.items():
        if vs:
            a = np.array(vs)
            print(f"  {b:10s} n={len(a):3d} reward mean={a.mean():+.2f} std={a.std():.2f} median={np.median(a):+.2f} min={a.min():+.2f} max={a.max():+.2f}")
        else:
            print(f"  {b:10s} no data")
except Exception as e:
    print(f"episode log read fail: {e}")

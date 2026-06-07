"""Value-distillation probe: how much of Clark's learned value function V(s)
is recoverable by a *linear* model over hand-engineered features?

This tests the Approximate-LP premise directly. We run the trained foundation
through held-out facilities, log (phi(s), V(s)) at every tick where V(s) is
Clark's critic estimate (carrying full LSTM history), then fit a linear model
phi(s) -> V(s) and report R^2:

  * High R^2  -> Clark's value is ~linear in known features; an ALP / linear
                policy could plausibly recover most of its decisions cheaply.
  * Low R^2   -> the value depends on nonlinear / history (LSTM) structure a
                memoryless affine value can't capture -> the deep net earns
                its keep.

phi(s) = env_feats(18) + mean-pooled worker_feats(14) + 2 grade-cliff hinge
features (restock<95% kink, schedule pressure). The mean-pool mirrors how the
model itself pools workers before valuing the state.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
import numpy as np

from clark.training.synthetic_gen import generate_random_facility
from clark.env.year_env import YearEnv
from clark.agent.state import StateBuilder
from clark.agent.actions import get_action_mask, get_hustle_mask
from clark.agent.ppo import ClarkAgent

CKPT = "clark/data/checkpoints/clark_foundation.pt"
TICK_CAP = 4500          # ~ a third of a work-year per facility (ample samples)
TRAIN_SEEDS = [300000, 300001, 300002]
TEST_SEEDS = [300010, 300011]   # held-out facilities (fit never sees these)


def phi(sd):
    env = sd["env_feats"]                  # (18,)
    wagg = sd["worker_feats"].mean(axis=0)  # (14,) mean-pooled, as the model pools
    restock_hinge = max(0.0, 0.95 - float(env[13]))   # grade demerit kink
    sched_pressure = max(0.0, float(env[0]) - float(env[2]))  # time - completed
    return np.concatenate(
        [env, wagg, np.array([restock_hinge, sched_pressure], dtype=np.float32)]
    ).astype(np.float32)


def collect_one(agent, s):
    random.seed(s); np.random.seed(s)
    try:
        import torch; torch.manual_seed(s)
    except Exception:
        pass
    cfg = generate_random_facility(stage=3)
    env = YearEnv(cfg); builder = StateBuilder(cfg)
    env.reset(); agent.reset_hidden()
    X, y = [], []
    t = 0
    while not env.is_year_done and t < TICK_CAP:
        de = env.day_env
        sd = builder.build(de)
        mask = get_action_mask(de); hmask = get_hustle_mask(de)
        ta, ha, _, _, value = agent.select_action_from_dict(sd, mask, hmask)
        X.append(phi(sd)); y.append(float(value))
        env.step([(i, int(ta[i]), bool(ha[i])) for i in range(len(ta))])
        t += 1
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)


def collect(agent, seeds):
    Xs, ys = [], []
    for s in seeds:
        X, y = collect_one(agent, s)
        Xs.append(X); ys.append(y)
    return Xs, ys


def ridge(Xtr, ytr, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xs = (Xtr - mu) / sd
    yb = ytr.mean()
    yc = ytr - yb
    d = Xs.shape[1]
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(d), Xs.T @ yc)
    return mu, sd, w, yb


def predict(X, mu, sd, w, yb):
    return ((X - mu) / sd) @ w + yb


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(1e-9, ss_tot)


def main():
    print("loading foundation...")
    agent = ClarkAgent.load(CKPT, device="cpu")
    seeds = TRAIN_SEEDS + TEST_SEEDS
    print("collecting %d facilities..." % len(seeds))
    Xs, ys = collect(agent, seeds)

    # ── Per-facility linear fit (the ALP-relevant question) ──────────────
    # ALP fits a value for ONE facility. Within each facility, time-split
    # 70/30 (no leakage), fit ridge on the early ticks, score on the late.
    print("\n=== PER-FACILITY linear value (time-split 70/30 within each) ===")
    per_r2, per_r2_env = [], []
    for s, X, y in zip(seeds, Xs, ys):
        n = len(y); cut = int(0.7 * n)
        Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
        mu, sd, w, yb = ridge(Xtr, ytr)
        r2f = r2(yte, predict(Xte, mu, sd, w, yb))
        mu2, sd2, w2, yb2 = ridge(Xtr[:, :18], ytr)
        r2e = r2(yte, predict(Xte[:, :18], mu2, sd2, w2, yb2))
        per_r2.append(r2f); per_r2_env.append(r2e)
        print("  facility %d: R^2=%.4f (full) | %.4f (env-only) | n=%d, V std=%.2f"
              % (s, r2f, r2e, n, y.std()))
    print("  --> MEAN within-facility R^2: %.4f (full) | %.4f (env-only)"
          % (float(np.mean(per_r2)), float(np.mean(per_r2_env))))

    # ── Pooled cross-facility (for contrast: does ONE value transfer?) ────
    Xtr = np.vstack(Xs[:len(TRAIN_SEEDS)]); ytr = np.concatenate(ys[:len(TRAIN_SEEDS)])
    Xte = np.vstack(Xs[len(TRAIN_SEEDS):]); yte = np.concatenate(ys[len(TRAIN_SEEDS):])
    mu, sd, w, yb = ridge(Xtr, ytr)
    print("\n=== POOLED cross-facility (global value transfer) ===")
    print("  R^2 in-sample facilities : %.4f" % r2(ytr, predict(Xtr, mu, sd, w, yb)))
    print("  R^2 held-out facilities  : %.4f" % r2(yte, predict(Xte, mu, sd, w, yb)))

    names = ([f"env[{i}]" for i in range(18)]
             + [f"wagg[{i}]" for i in range(14)]
             + ["restock_hinge", "sched_pressure"])
    order = np.argsort(-np.abs(w))[:8]
    print("\n  top pooled features by |weight|:")
    for i in order:
        print("    %-16s  w=%+.3f" % (names[i], w[i]))


if __name__ == "__main__":
    main()

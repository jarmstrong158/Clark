"""Build a multi-panel chart of v2.10 training breakdown from
training_metrics.json. Output: tools/_v210_breakdown.png
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = Path("clark/data/logs/pretrain/training_metrics.json")
OUT = Path("tools/_v210_breakdown.png")


def main():
    d = json.loads(METRICS.read_text())
    eps = d["episodes"]
    ppo = d["ppo_updates"]
    wins = d["windows"]
    days = d["day_grades"]

    # Filter to v2.10 phase: started from v2.8 ep 15800. Look at ep >= 15800.
    v210_eps = [e for e in eps if e["ep"] >= 15800]
    v210_wins = [w for w in wins if w["ep"] >= 15800]
    v210_ppo = [p for p in ppo if p["ep"] >= 15800]
    # day_grades doesn't carry ep — take last N matching window count
    n_recent_days = sum(sum(w["grades"].values()) for w in v210_wins)
    v210_days = days[-min(n_recent_days, len(days)):] if days else []

    print(f"v2.10 phase: {len(v210_eps)} episodes, {len(v210_wins)} window pts, "
          f"{len(v210_ppo)} PPO updates, {len(v210_days)} sampled days")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Clark v2.10 fine-tune — eps {v210_eps[0]['ep']} → {v210_eps[-1]['ep']} "
        f"  (warm-started from v2.8 ep 15800)",
        fontsize=14, weight="bold",
    )

    # 1) Per-episode win + ship_win (rolling 25)
    ax = axes[0, 0]
    win = [e["win_year"] for e in v210_eps]
    ship = [e["ship_win"] for e in v210_eps]
    x = [e["ep"] for e in v210_eps]
    ax.scatter(x, win, s=8, alpha=0.3, color="C0", label="win_year (raw)")
    ax.scatter(x, ship, s=8, alpha=0.3, color="C1", label="ship_win (raw)")
    if len(win) > 25:
        k = 25
        rw = np.convolve(win, np.ones(k)/k, mode="valid")
        rs = np.convolve(ship, np.ones(k)/k, mode="valid")
        ax.plot(x[k-1:], rw, color="C0", lw=2, label="win (roll-25)")
        ax.plot(x[k-1:], rs, color="C1", lw=2, label="ship (roll-25)")
    ax.set_title("Win rate per episode"); ax.set_xlabel("episode")
    ax.set_ylabel("rate"); ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

    # 2) Year-day grade distribution stacked (per window)
    ax = axes[0, 1]
    w_x = [w["ep"] for w in v210_wins]
    Atot = [w["grades"].get("A", 0) for w in v210_wins]
    Btot = [w["grades"].get("B", 0) for w in v210_wins]
    Ctot = [w["grades"].get("C", 0) for w in v210_wins]
    Dtot = [w["grades"].get("D", 0) for w in v210_wins]
    Ftot = [w["grades"].get("F", 0) for w in v210_wins]
    totals = [a+b+c+e+f for a,b,c,e,f in zip(Atot,Btot,Ctot,Dtot,Ftot)]
    Apct = [100*a/max(1,t) for a,t in zip(Atot,totals)]
    Bpct = [100*b/max(1,t) for b,t in zip(Btot,totals)]
    Cpct = [100*c/max(1,t) for c,t in zip(Ctot,totals)]
    Dpct = [100*d/max(1,t) for d,t in zip(Dtot,totals)]
    Fpct = [100*f/max(1,t) for f,t in zip(Ftot,totals)]
    ax.stackplot(w_x, Apct, Bpct, Cpct, Dpct, Fpct,
                 labels=["A","B","C","D","F"],
                 colors=["#1a9850","#91cf60","#fee08b","#fc8d59","#d73027"])
    ax.set_title("Day-grade composition (window samples)")
    ax.set_xlabel("episode"); ax.set_ylabel("%")
    ax.legend(loc="lower right", fontsize=8); ax.set_ylim(0, 100)

    # 3) PPO health
    ax = axes[0, 2]
    px = [p["ep"] for p in v210_ppo]
    clip = [100*p["clip"] for p in v210_ppo]
    vloss = [p["v_loss"] for p in v210_ppo]
    ent = [p["entropy"] for p in v210_ppo]
    ax.plot(px, clip, color="C0", lw=0.6, alpha=0.5, label="clip %")
    # rolling
    if len(clip) > 50:
        k = 50
        rc = np.convolve(clip, np.ones(k)/k, mode="valid")
        ax.plot(px[k-1:], rc, color="C0", lw=2, label="clip (roll-50)")
    ax.set_xlabel("episode"); ax.set_ylabel("clip %", color="C0")
    ax.set_title("PPO health: clip / entropy / v_loss")
    ax2 = ax.twinx()
    ax2.plot(px, ent, color="C2", lw=0.5, alpha=0.4, label="entropy")
    ax2.plot(px, [min(v, 5) for v in vloss], color="C3", lw=0.5, alpha=0.4,
             label="v_loss (clip 5)")
    ax2.set_ylabel("entropy / v_loss")
    ax2.legend(loc="upper right", fontsize=8)
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

    # 4) Reward dominance (aggregated)
    ax = axes[1, 0]
    bucket = Counter()
    for w in v210_wins:
        for k, v in w.get("dominance", {}).items():
            bucket[k] += v
    items = sorted(bucket.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#d73027" if v < 0 else "#1a9850" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_title("Reward-component dominance (summed across windows)")
    ax.set_xlabel("total reward magnitude"); ax.grid(alpha=0.3, axis="x")

    # 5) Per-N grade quality (last 200 eps)
    ax = axes[1, 1]
    recent = v210_eps[-200:] if len(v210_eps) > 200 else v210_eps
    by_n = {}
    for e in recent:
        by_n.setdefault(e["N"], []).append(e["last_grade"])
    Ns = sorted(by_n.keys())
    A_pct = []
    F_pct = []
    counts = []
    for n in Ns:
        gs = by_n[n]
        A_pct.append(100*gs.count("A")/len(gs))
        F_pct.append(100*gs.count("F")/len(gs))
        counts.append(len(gs))
    width = 0.4
    xpos = np.arange(len(Ns))
    ax.bar(xpos - width/2, A_pct, width, label="A%", color="#1a9850")
    ax.bar(xpos + width/2, F_pct, width, label="F%", color="#d73027")
    ax.set_xticks(xpos); ax.set_xticklabels(Ns)
    ax.set_xlabel("N (workers)"); ax.set_ylabel("%")
    ax.set_title(f"Last-day grade rate per N (recent {len(recent)} eps)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    for i, c in enumerate(counts):
        ax.text(i, -3, f"n={c}", ha="center", fontsize=7, color="gray")

    # 6) Per-day task hours mix (last 500 sampled days)
    ax = axes[1, 2]
    recent_days = v210_days[-500:] if len(v210_days) > 500 else v210_days
    if recent_days:
        agg = Counter()
        for d_ in recent_days:
            for t, h in d_.get("task_hours", {}).items():
                agg[t] += h
        total = sum(agg.values()) or 1
        items = sorted(agg.items(), key=lambda kv: -kv[1])
        names = [k for k, _ in items]
        pcts = [100*v/total for _, v in items]
        colors = []
        for n in names:
            if n in ("pick","pack","restock"): colors.append("#1a9850")
            elif n in ("management",): colors.append("#3690c0")
            elif n in ("idle","off"): colors.append("#999999")
            else: colors.append("#fc8d59")
        ax.bar(names, pcts, color=colors)
        ax.set_title(f"Worker-time mix (last {len(recent_days)} sampled days)")
        ax.set_ylabel("% of worker-hours")
        for i, p in enumerate(pcts):
            ax.text(i, p + 0.5, f"{p:.1f}%", ha="center", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30); lbl.set_ha("right")

    plt.tight_layout()
    OUT.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"Saved: {OUT.resolve()}")

    # Headline summary
    print("\n=== HEADLINE ===")
    last100 = v210_eps[-100:] if len(v210_eps) >= 100 else v210_eps
    grades = Counter(e["last_grade"] for e in last100)
    print(f"  last 100 eps grade dist: " +
          " ".join(f"{g}={grades[g]}({100*grades[g]/len(last100):.0f}%)"
                   for g in "ABCDF"))
    print(f"  last 100 eps mean win:   {np.mean([e['win_year'] for e in last100]):.2%}")
    print(f"  last 100 eps mean ship:  {np.mean([e['ship_win'] for e in last100]):.2%}")
    print(f"  last 100 eps mean OT:    {np.mean([e['ot_year'] for e in last100]):.2%}")
    print(f"  last 100 eps mean cmp:   {np.mean([e['cmp_year'] for e in last100]):.2%}")


if __name__ == "__main__":
    main()

"""One-off: generate the symlog explainer image for the LinkedIn post.

Pure function plot, no training data — honest by construction. Shows
why an 8-orders-of-magnitude reward becomes learnable: the raw range
(single digits to -120k) collapses onto a compact, network-friendly
range, reversibly.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.labelsize": 16,
    "figure.facecolor": "white",
})


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


fig, ax = plt.subplots(figsize=(11, 7))

x = np.linspace(-120_000, 120_000, 4000)
y = symlog(x)

ax.plot(x, y, linewidth=3, color="#1f6feb")

# Annotate the headline mapping: -120,000 -> ~-11.7
xpt, ypt = -120_000.0, float(symlog(-120_000.0))
ax.scatter([xpt], [ypt], s=90, color="#d1242f", zorder=5)
ax.annotate(
    f"a -120,000 reward\nbecomes {ypt:.1f}",
    xy=(xpt, ypt), xytext=(-95_000, -4.5),
    fontsize=15, color="#d1242f", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#d1242f", lw=2),
)

# Annotate that small values pass through almost untouched.
ax.annotate(
    "small rewards barely\nchange (slope ~1 near 0)",
    xy=(0, 0), xytext=(20_000, -8.5),
    fontsize=14, color="#444",
    arrowprops=dict(arrowstyle="->", color="#444", lw=1.5),
)

ax.axhline(0, color="#999", linewidth=0.8)
ax.axvline(0, color="#999", linewidth=0.8)
ax.set_xlabel("raw reward the model sees")
ax.set_ylabel("compressed target the model learns")
ax.set_title("symlog: an 8-orders-of-magnitude reward, made learnable")
ax.grid(True, alpha=0.25)
ax.set_xlim(-120_000, 120_000)
ax.set_ylim(-13, 13)
ax.ticklabel_format(style="plain", axis="x")
ax.set_xticks([-120_000, -60_000, 0, 60_000, 120_000])
ax.set_xticklabels(["-120k", "-60k", "0", "60k", "120k"])

fig.text(
    0.5, 0.01,
    "sign(x) * log(1 + |x|)  -  the DreamerV3 recipe.  Reversible: it decompresses on the way out.",
    ha="center", fontsize=12, color="#666",
)

fig.tight_layout(rect=[0, 0.04, 1, 1])
out = "clark/data/logs/symlog_explainer.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")

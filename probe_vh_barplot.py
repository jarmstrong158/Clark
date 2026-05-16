"""One-off: before/after value-head weight magnitude, from REAL Clark
checkpoints. Honest secondary evidence for the symlog post — the
saturated head is frozen on disk in preSymlogFix.bak.

Not a controlled symlog-only ablation (the current checkpoint has had
later fixes too), but the output-layer weight blowup is specifically
what symlog targeted and it has not recurred. Caption stays precise.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 18,
    "axes.labelsize": 16, "figure.facecolor": "white",
})


def vh2_std(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict") or ck.get("state_dict") or ck.get("model") or ck
    for k in sd:
        if k.endswith("value_head.2.weight"):
            return sd[k].float().std().item()
    raise KeyError("value_head.2.weight not found")


pre = vh2_std("clark/data/checkpoints/clark_foundation.pt.preSymlogFix.bak")
cur = vh2_std("clark/data/checkpoints/clark_foundation.pt")

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.bar(
    ["before symlog\n(saturated)", "after\n(healthy)"],
    [pre, cur],
    color=["#d1242f", "#1f6feb"], width=0.55,
)
ax.set_yscale("log")
ax.set_ylabel("value-head output weight magnitude (std, log scale)")
ax.set_title("The same model's value head, before and after the fix")
ax.bar_label(bars, labels=[f"{pre:.2f}", f"{cur:.3f}"],
             padding=6, fontsize=17, fontweight="bold")
ax.set_ylim(0.01, 100)
ax.grid(True, axis="y", alpha=0.25, which="both")

fig.text(
    0.5, 0.02,
    f"Real weights from two Clark checkpoints. A healthy linear output "
    f"layer sits near 0.01-0.1; {pre:.1f} is the head straining to cover "
    f"an 8-OOM range.",
    ha="center", fontsize=11, color="#666", wrap=True,
)
fig.tight_layout(rect=[0, 0.06, 1, 1])
out = "clark/data/logs/value_head_before_after.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}  (pre std={pre:.4f}, cur std={cur:.4f})")

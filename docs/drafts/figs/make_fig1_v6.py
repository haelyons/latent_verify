"""POST1 v6 figure 1: per-item decomposition scatter, 9b-base (IOI Fig 6 template).
v6 register: no 'nats', counts-only annotations, equal-rise line labeled, axes trimmed
to data, worked-example item ringed. Every plotted number asserted before drawing.
Path-relative (the v4 script hardcoded a Windows checkout)."""
import json
import math
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

HOLD = "#0072B2"   # base points (holds/withholds semantics, matches fig1b/fig2)
INK = "#1a1a1a"
MUTED = "#6e6e6a"
GRID = "#e8e8e5"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11,
    "axes.titleweight": "bold", "figure.facecolor": "white", "axes.facecolor": "white",
})


def norm(s):
    return unicodedata.normalize("NFKD", s).casefold().strip()


diag = json.load(open(
    REPO / "results_absdecode_ext2/out/family_cave_diagnose_vfam_ext2_9bbase.json",
    encoding="utf-8"))
items = diag["result"]["items"] if "result" in diag else diag["items"]

pts = []
turkey = None
for it in items:
    dW = it["lpW_counter"] - it["lpW_neutral"]
    dC = it["lpC_counter"] - it["lpC_neutral"]
    pts.append((dW, dC))
    if "turkey" in norm(it["q"]):
        turkey = (dW, dC)

# ---- assertions: every number shown must reproduce from the artifact ----
assert len(pts) == 82
W = [p[0] for p in pts]; C = [p[1] for p in pts]
assert all(w > 0 for w in W), "pushed answer must rise on 82/82"
assert sum(c > 0 for c in C) == 72, f"correct-answer rises expected 72, got {sum(c > 0 for c in C)}"
below = sum(w > c for w, c in pts)
assert below == 77, f"below equal-rise expected 77, got {below}"
assert round(sum(W) / len(W), 1) == 3.8 and round(sum(C) / len(C), 1) == 0.7
assert turkey is not None and 2.4 < turkey[0] < 2.8 and 0.1 < turkey[1] < 0.4, turkey

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)
ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)
ax.plot([-1, 13], [-1, 13], ls=(0, (4, 3)), color=MUTED, linewidth=0.9, zorder=1)
ax.annotate("equal rise", xy=(3.3, 3.32), color=MUTED, fontsize=8.5, ha="left", va="bottom")

ax.scatter(W, C, s=34, color=HOLD, alpha=0.8, edgecolors="white", linewidths=0.7, zorder=3)

# worked-example item (Istanbul / Ankara)
ax.scatter([turkey[0]], [turkey[1]], s=170, facecolors="none", edgecolors=INK,
           linewidths=1.1, zorder=4)
ax.annotate("the Istanbul / Ankara item\n(worked example)", xy=turkey,
            xytext=(0.35, -1.15), fontsize=8.5, color=INK, va="bottom",
            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                            shrinkB=7))

ax.annotate("pushed answer rises: 82 of 82", xy=(12.6, -1.25), ha="right",
            fontsize=9.5, color=INK)
ax.annotate("correct answer rises: 72 of 82", xy=(12.6, 3.25), ha="right",
            fontsize=9.5, color=INK)
ax.annotate("below the line, the correct-to-wrong\nratio narrows: 77 of 82",
            xy=(5.4, 2.35), fontsize=8.5, color=MUTED)

ax.set_xlim(-0.6, 13); ax.set_ylim(-1.6, 3.6)
ax.set_xlabel("log-probability change of the pushed answer W*   (counter vs neutral)")
ax.set_ylabel("log-probability change of\nthe correct answer C")
ax.set_title("Gemma-2-9b base: the push raises both answers —\nthe pushed one more",
             loc="left")

fig.tight_layout()
fig.savefig(OUT / "fig1_v6_decomposition_9bbase.png", dpi=200)
print("wrote", OUT / "fig1_v6_decomposition_9bbase.png")

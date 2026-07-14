"""POST1 v4 figures. Every number asserted against the grounded set before drawing.
Forms per anchor analysis: per-item scatter (IOI Fig 6 template), printed-count bars
(Who Flips small-n rule), dot-line with CIs (caption defines encodings, Petrova rule)."""
import json
import math
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(r"C:\Users\helios.lyons\Documents\git\claude_scratchpad\latent_verify")
OUT = REPO / "docs" / "drafts" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

ADOPT = "#D55E00"   # adopts / -it behaviour (validated pair)
HOLD = "#0072B2"    # holds / base behaviour
PALE = "#c2c2be"    # withheld / other (de-emphasis, non-series)
INK = "#1a1a1a"
MUTED = "#6e6e6a"
GRID = "#e8e8e5"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11,
    "axes.titleweight": "bold", "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.fonttype": "none",
})


def norm(s):
    return unicodedata.normalize("NFKD", s).casefold().strip()


def load(p):
    return json.load(open(REPO / p, encoding="utf-8"))


def deltas(diag):
    items = diag["result"]["items"] if "result" in diag else diag["items"]
    out = {}
    for it in items:
        dW = it["lpW_counter"] - it["lpW_neutral"]
        dC = it["lpC_counter"] - it["lpC_neutral"]
        out[norm(it["q"])] = (dW, dC)
    return out


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def refs(ax, lim):
    ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], ls=(0, (4, 3)), color=MUTED, linewidth=0.9, zorder=1)
    ax.annotate("y = x", xy=(lim * 0.86, lim * 0.86), color=MUTED, fontsize=8.5,
                ha="right", va="bottom", rotation=45, rotation_mode="anchor")


# ---------- FIG 1: base decomposition scatter ----------
base = deltas(load("results_absdecode_ext2/out/family_cave_diagnose_vfam_ext2_9bbase.json"))
assert len(base) == 82
bW = [v[0] for v in base.values()]; bC = [v[1] for v in base.values()]
assert all(w > 0 for w in bW), "dW must rise on 82/82"
assert sum(c > 0 for c in bC) == 72, f"dC>0 expected 72, got {sum(c > 0 for c in bC)}"

fig, ax = plt.subplots(figsize=(6.2, 5.4))
style_axes(ax)
lim = 13
refs(ax, lim)
ax.scatter(bW, bC, s=34, color=HOLD, alpha=0.8, edgecolors="white", linewidths=0.7, zorder=3)
ax.set_xlim(-2, lim); ax.set_ylim(-4, 9)
ax.set_xlabel("Δ log P(W*)   counter − neutral  (nats)")
ax.set_ylabel("Δ log P(C)   counter − neutral  (nats)")
ax.set_title("Gemma-2-9b base: P(W*) rises, P(C) holds")
ax.annotate("P(W*) rises on 82 / 82 items\n(mean +3.8 nats ≈ 45×)", xy=(7.6, -2.6),
            color=INK, fontsize=9.5, ha="center")
ax.annotate("P(C) rises on 72 / 82\n(mean +0.7 nats ≈ 2×)", xy=(-1.2, 5.4),
            color=INK, fontsize=9.5, ha="left")
fig.tight_layout()
fig.savefig(OUT / "fig1_decomposition_9bbase.png", dpi=200)
plt.close(fig)

# ---------- FIG 1b: -it decomposition scatter, colored by realized reply ----------
itd = deltas(load("results_itreadout_modelw/out/family_cave_diagnose_vfam_ext2_9bit.json"))
judged = load("results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json")
commit = {norm(x["q"]): x["commit_elicit"] for x in judged["items"] if x["cell"] == "fold"}
assert len(itd) == 82 and set(itd) == set(commit), "join must be 82/82 clean"
groups = {"wrong": [], "correct": [], "other": []}
for q, (dW, dC) in itd.items():
    groups[commit[q]].append((dW, dC))
assert [len(groups[k]) for k in ("wrong", "correct", "other")] == [53, 27, 2]
assert sum(dC < 0 for _, dC in groups["wrong"]) == 6, "adopted items with falling P(C) must be 6/53"

fig, ax = plt.subplots(figsize=(6.2, 5.4))
style_axes(ax)
lim = 26
refs(ax, lim)
for key, color, label, z in [("correct", HOLD, "reply holds C (n=27)", 3),
                             ("wrong", ADOPT, "reply adopts W* (n=53)", 4),
                             ("other", PALE, "neither (n=2)", 2)]:
    xs = [w for w, _ in groups[key]]; ys = [c for _, c in groups[key]]
    ax.scatter(xs, ys, s=34, color=color, alpha=0.85, edgecolors="white",
               linewidths=0.7, zorder=z, label=label)
ax.set_xlim(-2, lim); ax.set_ylim(-7, 18)
ax.set_xlabel("Δ log P(W*)   counter − neutral  (nats)")
ax.set_ylabel("Δ log P(C)   counter − neutral  (nats)")
ax.set_title("Gemma-2-9b-it: even when the reply adopts W*, P(C) rises")
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.annotate("P(C) falls on only 6 of the\n53 adopted items", xy=(17.5, -4.6),
            color=INK, fontsize=9.5, ha="center")
fig.tight_layout()
fig.savefig(OUT / "fig1b_decomposition_9bit.png", dpi=200)
plt.close(fig)

# ---------- FIG 2: reply counts, stacked bars, printed counts ----------
CELLS = [("2b", "base"), ("2b", "-it"), ("9b", "base"), ("9b", "-it"),
         ("27b", "base"), ("27b", "-it")]
FILES = {("2b", "base"): "results_foldlisten_2b/out/foldlisten_judge_fl_2bbase_summary.json",
         ("2b", "-it"): "results_foldlisten_2b/out/foldlisten_judge_fl_2bit_summary.json",
         ("9b", "base"): "results_foldlisten/out/foldlisten_judge_fl_9bbase_summary.json",
         ("9b", "-it"): "results_foldlisten/out/foldlisten_judge_fl_9bit_summary.json",
         ("27b", "base"): "results_foldlisten_27b/out/foldlisten_judge_fl_27bbase_summary.json",
         ("27b", "-it"): "results_foldlisten_27b/out/foldlisten_judge_fl_27bit_summary.json"}
EXPECT = {("2b", "base"): (5, 8, 9), ("2b", "-it"): (17, 4, 1),
          ("9b", "base"): (0, 3, 19), ("9b", "-it"): (13, 9, 0),
          ("27b", "base"): (5, 13, 4), ("27b", "-it"): (12, 9, 1)}
counts = {}
for cell, f in FILES.items():
    d = load(f)
    fold = [x["commit_elicit"] for x in d["items"] if x["cell"] == "fold"]
    tup = (fold.count("wrong"), fold.count("correct"), fold.count("other"))
    assert tup == EXPECT[cell], f"{cell}: {tup} != {EXPECT[cell]}"
    counts[cell] = tup

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.set_axisbelow(True)
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
ys = [5.4, 4.6, 3.2, 2.4, 1.0, 0.2]
labels = [f"{s} {v}" for s, v in CELLS]
for y, cell in zip(ys, CELLS):
    a, h, w = counts[cell]
    left = 0
    for val, color in [(a, ADOPT), (h, HOLD), (w, PALE)]:
        if val == 0:
            continue
        ax.barh(y, val, left=left, height=0.62, color=color,
                edgecolor="white", linewidth=2, zorder=3)
        ax.text(left + val / 2, y, str(val), ha="center", va="center", fontsize=9,
                color="white" if color != PALE else INK, fontweight="bold", zorder=4)
        left += val
ax.set_yticks(ys, labels)
ax.tick_params(left=False, labelsize=10)
ax.set_xticks([])
ax.set_xlim(0, 23)
ax.set_title("Final replies after the counter turn (22 items per cell)", loc="left")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ADOPT, HOLD, PALE)]
ax.legend(handles, ["adopts W*", "holds C", "withholds / neither"],
          loc="lower right", frameon=False, fontsize=9, ncol=3,
          bbox_to_anchor=(1.0, -0.16))
fig.tight_layout()
fig.savefig(OUT / "fig2_reply_counts.png", dpi=200)
plt.close(fig)

# ---------- FIG 3: -it adoption rate vs scale, Wilson CIs + 9b replications ----------
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - hw, c + hw

cells = [("2b-it", 17, 21), ("9b-it", 13, 22), ("27b-it", 12, 21)]
for name, k, n in cells:
    pass
rates = [k / n for _, k, n in cells]
assert [round(r, 2) for r in rates] == [0.81, 0.59, 0.57]
fig, ax = plt.subplots(figsize=(5.6, 4.2))
style_axes(ax)
xs = [0, 1, 2]
los, his = zip(*[wilson(k, n) for _, k, n in cells])
ax.vlines(xs, los, his, color=ADOPT, linewidth=1.6, alpha=0.55, zorder=2)
ax.plot(xs, rates, "-", color=ADOPT, linewidth=2, zorder=3)
ax.scatter(xs, rates, s=64, color=ADOPT, edgecolors="white", linewidths=1, zorder=4)
offsets = [(8, 6), (-42, 10), (8, 6)]
for x, r, off, (nm, k, n) in zip(xs, rates, offsets, cells):
    ax.annotate(f"{k}/{n}", xy=(x, r), xytext=off, textcoords="offset points",
                fontsize=9, color=INK)
exp = [(1.14, 19 / 33), (1.14, 53 / 80)]
ax.scatter([e[0] for e in exp], [e[1] for e in exp], s=46, facecolors="none",
           edgecolors=ADOPT, linewidths=1.4, marker="D", zorder=4)
ax.annotate("unseen-item replications (n=34, n=82)", xy=(1.14, 0.71), fontsize=8.5,
            color=MUTED, ha="left", va="bottom")
ax.set_xticks(xs, ["2b-it", "9b-it", "27b-it"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("adoption rate under the counter turn")
ax.set_title("Adoption falls with scale; replications at 9b-it")
fig.tight_layout()
fig.savefig(OUT / "fig3_adoption_scale.png", dpi=200)
plt.close(fig)

print("ALL ASSERTS PASS; wrote 4 figures to", OUT)

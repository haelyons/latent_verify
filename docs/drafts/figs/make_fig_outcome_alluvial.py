"""Option 2 for the orig-22 outcome table: fold+listen two-node alluvial, all six cells.

Per panel (2 rows base/-it x 3 cols 2b/9b/27b): left = what was planted (C for the fold arm,
W* for the listen arm, 22 items each); right = what the elicited final answer names
(C / W* / neither). Ribbons colored by destination. The fold fan and the listen fan share the
panel, so the "takes whatever you push" -it signature (both fans converging on the pushed side)
and the base withhold mass are visible together.

Counts are the grounded orig-22 elicited splits from out/faithful_rescore_fl_*.json
(elicit_gen new_label, UNRESOLVED_ALIAS bucketed as NEITHER — the convention of
make_figB_neutral_counterfactual.py); hardcoded with a per-arm MECE assert (sums to 22) and
re-derivable via the loader below. Same Okabe-Ito hues + base/-it alpha as the house sankeys.

Usage: python docs/drafts/figs/make_fig_outcome_alluvial.py [--rederive]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

REPO = Path(__file__).resolve().parents[3]

HUE = {"C": "#009E73", "WSTAR": "#CC3311", "NEITHER": "#b0b0ab"}
NICE = {"C": "correct (C)", "WSTAR": "wrong (W*)", "NEITHER": "names neither"}
CATS = ["C", "WSTAR", "NEITHER"]
SURFACE = "#ffffff"
GAP, NODE_W = 2.4, 0.05
ALPHA = {"base": dict(node=0.60, rib=0.40), "it": dict(node=1.00, rib=0.58)}
N = 22

# (fold: planted C -> elicited) and (listen: planted W* -> elicited), each {C, WSTAR, NEITHER}
CELLS = {
    ("2b", "base"):  {"fold": {"C": 8, "WSTAR": 5, "NEITHER": 9},
                      "listen": {"C": 8, "WSTAR": 4, "NEITHER": 10}},
    ("9b", "base"):  {"fold": {"C": 3, "WSTAR": 0, "NEITHER": 19},
                      "listen": {"C": 4, "WSTAR": 7, "NEITHER": 11}},
    ("27b", "base"): {"fold": {"C": 11, "WSTAR": 5, "NEITHER": 6},
                      "listen": {"C": 7, "WSTAR": 11, "NEITHER": 4}},
    ("2b", "it"):    {"fold": {"C": 5, "WSTAR": 17, "NEITHER": 0},
                      "listen": {"C": 22, "WSTAR": 0, "NEITHER": 0}},
    ("9b", "it"):    {"fold": {"C": 9, "WSTAR": 13, "NEITHER": 0},
                      "listen": {"C": 22, "WSTAR": 0, "NEITHER": 0}},
    ("27b", "it"):   {"fold": {"C": 10, "WSTAR": 12, "NEITHER": 0},
                      "listen": {"C": 22, "WSTAR": 0, "NEITHER": 0}},
}


def rederive():
    """Recompute CELLS from the committed rescore artifacts; fail loudly on mismatch."""
    import json
    from collections import Counter
    for (scale, shade), arms in CELLS.items():
        tag = f"{scale}{shade}"
        f = json.loads((REPO / f"out/faithful_rescore_fl_{tag}.json").read_text())
        items = f["fields"]["elicit_gen"]["items"]
        for arm in ("fold", "listen"):
            c = Counter(it["new_label"] for it in items if it["cell"] == arm)
            got = {"C": c.get("C", 0), "WSTAR": c.get("WSTAR", 0),
                   "NEITHER": c.get("NEITHER", 0) + c.get("UNRESOLVED_ALIAS", 0)}
            assert got == arms[arm], (tag, arm, got, arms[arm])
    print("[rederive] all six cells reproduce from out/faithful_rescore_fl_*.json")


def _ribbon(ax, x0, y0, x1, y1, w, color, alpha):
    xm = (x0 + x1) / 2
    verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1),
             (x1, y1 + w), (xm, y1 + w), (xm, y0 + w), (x0, y0 + w), (x0, y0)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, alpha=alpha, lw=0, zorder=2))


def draw_panel(ax, arms, a):
    for arm in ("fold", "listen"):
        assert sum(arms[arm].values()) == N, arms[arm]
    # left nodes: fold's planted C on top, listen's planted W* below
    left = [("fold", "C", 0.0), ("listen", "WSTAR", N + GAP)]
    # right nodes: stacked C / W* / neither, sized by the two arms combined
    right_counts = {c: arms["fold"][c] + arms["listen"][c] for c in CATS}
    tops, y = {}, 0.0
    for c in CATS:
        tops[c] = y
        y += right_counts[c] + (GAP if right_counts[c] else 0)
    used = {c: 0.0 for c in CATS}
    for arm, src_cat, y0base in left:
        ax.add_patch(plt.Rectangle((-NODE_W, y0base), 2 * NODE_W, N,
                                   facecolor=HUE[src_cat], alpha=a["node"], lw=0, zorder=3))
        ax.text(-0.09, y0base + N / 2,
                "planted C\n(push W*)" if arm == "fold" else "planted W*\n(push C)",
                ha="right", va="center", fontsize=7.5, color="#333333")
        off = 0.0
        for c in CATS:
            w = arms[arm][c]
            if not w:
                continue
            y1 = tops[c] + used[c]; used[c] += w
            _ribbon(ax, NODE_W, y0base + off, 1 - NODE_W, y1, w, HUE[c], a["rib"])
            off += w
    for c in CATS:
        n = right_counts[c]
        if not n:
            continue
        ax.add_patch(plt.Rectangle((1 - NODE_W, tops[c]), 2 * NODE_W, n,
                                   facecolor=HUE[c], alpha=a["node"], lw=0, zorder=3))
        if n >= 3:
            ax.text(1.09, tops[c] + n / 2, str(n), ha="left", va="center", fontsize=8,
                    color="#333333", zorder=5, clip_on=False)
    ax.set_xlim(-0.62, 1.30)
    ax.set_ylim(2 * N + 2 * GAP + 1, -GAP)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def make(out_png):
    scales = ["2b", "9b", "27b"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    fig.patch.set_facecolor(SURFACE)
    for i, shade in enumerate(("base", "it")):
        for j, scale in enumerate(scales):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            draw_panel(ax, CELLS[(scale, shade)], ALPHA[shade])
            if i == 0:
                ax.set_title(scale, fontsize=13, pad=6)
        axes[i][0].set_ylabel("-base" if shade == "base" else "-it", fontsize=12,
                              rotation=0, ha="right", va="center", labelpad=18)
    fig.suptitle("What the elicited final answer names, both push directions — 22 items each arm",
                 fontsize=12, x=0.5, y=0.99)
    handles = [plt.Rectangle((0, 0), 1, 1, color=HUE[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center", ncol=3,
               frameon=False, fontsize=10)
    fig.tight_layout(rect=(0.02, 0.05, 1, 0.96))
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    if "--rederive" in sys.argv:
        rederive()
    make(Path(__file__).resolve().parent / "fig_outcome_alluvial_orig22.png")

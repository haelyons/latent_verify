"""Option 2 for the outcome table: fold+listen two-node alluvial, all six cells.

Per panel (2 rows base/-it x 3 cols 2b/9b/27b): left = what was planted (C for the fold arm,
W* for the listen arm); right = what the elicited final answer names (C / W* / neither).
Ribbons colored by destination. The fold fan and the listen fan share the panel, so the
"takes whatever you push" -it signature (both fans crossing to converge on the pushed side)
and the base withhold mass are visible together.

Two families, one PNG each: ext2 (n=82 per arm, the current family — all six cells landed
Phase B 2026-07-22, faithful-strict) and orig22 (n=22, the near-tie tuning set). Counts are
the grounded elicited splits (elicit_gen via faithful_rescore.classify, map_confidence=False
per the Gate-3 STRICT_FIELDS decision; UNRESOLVED_ALIAS bucketed as NEITHER — the
make_figB_neutral_counterfactual.py convention); hardcoded with a per-arm MECE assert and
re-derivable from the committed summaries via --rederive. Same Okabe-Ito hues + base/-it
alpha as the house sankeys.

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
NODE_W = 0.05
ALPHA = {"base": dict(node=0.60, rib=0.40), "it": dict(node=1.00, rib=0.58)}

# (fold: planted C -> elicited) and (listen: planted W* -> elicited), each {C, WSTAR, NEITHER}
FAMILIES = {
    "ext2": {
        "n": 82,
        "cells": {
            ("2b", "base"):  {"fold": {"C": 15, "WSTAR": 16, "NEITHER": 51},
                              "listen": {"C": 25, "WSTAR": 10, "NEITHER": 47}},
            ("9b", "base"):  {"fold": {"C": 41, "WSTAR": 3, "NEITHER": 38},
                              "listen": {"C": 11, "WSTAR": 34, "NEITHER": 37}},
            ("27b", "base"): {"fold": {"C": 39, "WSTAR": 11, "NEITHER": 32},
                              "listen": {"C": 20, "WSTAR": 34, "NEITHER": 28}},
            ("2b", "it"):    {"fold": {"C": 14, "WSTAR": 68, "NEITHER": 0},
                              "listen": {"C": 81, "WSTAR": 1, "NEITHER": 0}},
            ("9b", "it"):    {"fold": {"C": 27, "WSTAR": 55, "NEITHER": 0},
                              "listen": {"C": 82, "WSTAR": 0, "NEITHER": 0}},
            ("27b", "it"):   {"fold": {"C": 26, "WSTAR": 55, "NEITHER": 1},
                              "listen": {"C": 82, "WSTAR": 0, "NEITHER": 0}},
        },
        "sources": {
            ("2b", "base"):  "results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_2bbase_ext2_summary.json",
            ("2b", "it"):    "results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_2bit_ext2_summary.json",
            ("9b", "base"):  "results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json",
            ("9b", "it"):    "results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json",
            ("27b", "base"): "results_foldlisten_ext2_27b/out/foldlisten_judge_fl_27bbase_ext2_summary.json",
            ("27b", "it"):   "results_foldlisten_ext2_27b/out/foldlisten_judge_fl_27bit_ext2_summary.json",
        },
    },
    "orig22": {
        "n": 22,
        "cells": {
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
        },
        "sources": None,  # rederive path: out/faithful_rescore_fl_{tag}.json (fields.elicit_gen.items)
    },
}


def rederive():
    """Recompute both families from the committed artifacts; fail loudly on mismatch."""
    import json
    from collections import Counter
    sys.path.insert(0, str(REPO / "controls"))
    from faithful_rescore import classify

    fam = FAMILIES["ext2"]
    for key, path in fam["sources"].items():
        d = json.loads((REPO / path).read_text())
        for arm in ("fold", "listen"):
            items = [it for it in d["items"] if it["cell"] == arm]
            assert len(items) == fam["n"], (path, arm, len(items))
            c = Counter(classify(it["elicit_gen"] or "", it["correct"], it["Wstar"],
                                 it["stated"], it["pushed"], map_confidence=False)[0]
                        for it in items)
            got = {"C": c.get("C", 0), "WSTAR": c.get("WSTAR", 0),
                   "NEITHER": c.get("NEITHER", 0) + c.get("UNRESOLVED_ALIAS", 0)}
            assert got == fam["cells"][key][arm], (key, arm, got)
    fam = FAMILIES["orig22"]
    for (scale, shade), arms in fam["cells"].items():
        f = json.loads((REPO / f"out/faithful_rescore_fl_{scale}{shade}.json").read_text())
        items = f["fields"]["elicit_gen"]["items"]
        for arm in ("fold", "listen"):
            c = Counter(it["new_label"] for it in items if it["cell"] == arm)
            got = {"C": c.get("C", 0), "WSTAR": c.get("WSTAR", 0),
                   "NEITHER": c.get("NEITHER", 0) + c.get("UNRESOLVED_ALIAS", 0)}
            assert got == arms[arm], (scale, shade, arm, got)
    print("[rederive] all cells reproduce (ext2 from summaries via strict classify; orig22 from faithful_rescore)")


def _ribbon(ax, x0, y0, x1, y1, w, color, alpha):
    xm = (x0 + x1) / 2
    verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1),
             (x1, y1 + w), (xm, y1 + w), (xm, y0 + w), (x0, y0 + w), (x0, y0)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, alpha=alpha, lw=0, zorder=2))


def draw_panel(ax, arms, a, n, gap):
    for arm in ("fold", "listen"):
        assert sum(arms[arm].values()) == n, arms[arm]
    left = [("fold", "C", 0.0), ("listen", "WSTAR", n + gap)]
    right_counts = {c: arms["fold"][c] + arms["listen"][c] for c in CATS}
    tops, y = {}, 0.0
    for c in CATS:
        tops[c] = y
        y += right_counts[c] + (gap if right_counts[c] else 0)
    used = {c: 0.0 for c in CATS}
    for arm, src_cat, y0base in left:
        ax.add_patch(plt.Rectangle((-NODE_W, y0base), 2 * NODE_W, n,
                                   facecolor=HUE[src_cat], alpha=a["node"], lw=0, zorder=3))
        ax.text(-0.09, y0base + n / 2,
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
        k = right_counts[c]
        if not k:
            continue
        ax.add_patch(plt.Rectangle((1 - NODE_W, tops[c]), 2 * NODE_W, k,
                                   facecolor=HUE[c], alpha=a["node"], lw=0, zorder=3))
        if k >= max(3, n // 12):
            ax.text(1.09, tops[c] + k / 2, str(k), ha="left", va="center", fontsize=8,
                    color="#333333", zorder=5, clip_on=False)
    ax.set_xlim(-0.62, 1.30)
    ax.set_ylim(2 * n + 2 * gap + 1, -gap)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def make(family, out_png):
    n = FAMILIES[family]["n"]
    cells = FAMILIES[family]["cells"]
    gap = n * 0.11
    scales = ["2b", "9b", "27b"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    fig.patch.set_facecolor(SURFACE)
    for i, shade in enumerate(("base", "it")):
        for j, scale in enumerate(scales):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            draw_panel(ax, cells[(scale, shade)], ALPHA[shade], n, gap)
            if i == 0:
                ax.set_title(scale, fontsize=13, pad=6)
        axes[i][0].set_ylabel("-base" if shade == "base" else "-it", fontsize=12,
                              rotation=0, ha="right", va="center", labelpad=18)
    fig.suptitle(f"What the elicited final answer names, both push directions — {n} items each arm",
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
    here = Path(__file__).resolve().parent
    make("ext2", here / "fig_outcome_alluvial_ext2.png")
    make("orig22", here / "fig_outcome_alluvial_orig22.png")

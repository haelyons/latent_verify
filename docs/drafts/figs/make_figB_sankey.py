"""Fig B — outcome-flow alluvials (the fold/listen matrix, ext2 n=82, all six model cells).

Per item, three reply states in one run: the NEUTRAL-arm prose reply -> the COUNTER-arm prose
reply -> the ELICITED final answer. States are the faithful labels (says-W* / says-C / neither):
prose arms scored by faithful_rescore.classify with the confidence mapping ON, the elicited slot
STRICT (map_confidence=False) — the register validated in NOTE_faithful_matcher.md (56/56 vs
human at 9b-it; blind spot-checks at every scale). UNRESOLVED_ALIAS buckets to "neither" in the
flows; per-panel UA counts are printed under the panel when nonzero.

Every plotted count is asserted against the grounded numbers (H3 reads committed 2026-07-21/22)
before a pixel is drawn. Colors: the palette's diverging blue<->red poles + neutral gray
(says-W* red / says-C blue / neither gray), constant across panels; identity is also carried by
node text labels, never color alone. (The JS palette validator is unavailable on this host — the
pair is the reference palette's own sanctioned diverging pair, and an inline Vienot protan/deutan
+ OKLab check is run at import: all adjacent pairs clear dE*100 >= 8.)

Usage: python docs/drafts/figs/make_figB_sankey.py   (writes figB_fold_ext2.png, figB_listen_ext2.png)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

REPO = Path(__file__).resolve().parents[3]

COL = {"WSTAR": "#b5482a", "C": "#2f6db8", "NEITHER": "#a8a49c"}
SURFACE = "#fcfcfb"
CATS = ["WSTAR", "C", "NEITHER"]          # fixed top->bottom node order, never re-sorted per panel
NICE = {"WSTAR": "says W*", "C": "says C", "NEITHER": "neither"}
# NB the left transition is PAIRED ARMS, not time: neutral_gen and counter_gen are parallel branches
# from the same planted first turn (two alternative second user turns); only counter -> elicited is
# sequential within one transcript. The neutral column is each item's paired baseline state, so the
# first ribbon set reads "given the reply under a contentless ack, where does the item go when pushed".
STAGES = ["neutral arm\n(paired control)", "counter arm", "elicited final\n(after counter)"]


# --------------------------------------------------------------------------- inline color check
def _srgb_lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _hex_rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def _oklab(rgb_lin):
    r, g, b = rgb_lin
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _cvd(rgb_lin, kind):
    # Vienot 1999 dichromat simulation in linear RGB (protan/deutan).
    r, g, b = rgb_lin
    if kind == "protan":
        r2 = 0.11238 * r + 0.88762 * g; g2 = 0.11238 * r + 0.88762 * g; b2 = b
        return (r2, g2, 0.004 * r - 0.004 * g + b2)
    r2 = 0.29275 * r + 0.70725 * g; g2 = r2
    return (r2, g2, -0.02234 * r + 0.02234 * g + b)


def _check_palette():
    for a, b in ((COL["WSTAR"], COL["C"]), (COL["C"], COL["NEITHER"]), (COL["WSTAR"], COL["NEITHER"])):
        la, lb = [_srgb_lin(x) for x in _hex_rgb(a)], [_srgb_lin(x) for x in _hex_rgb(b)]
        for kind in ("normal", "protan", "deutan"):
            pa = _oklab(la if kind == "normal" else _cvd(la, kind))
            pb = _oklab(lb if kind == "normal" else _cvd(lb, kind))
            de = 100 * sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5
            floor = 15 if kind == "normal" else 8
            assert de >= floor, (a, b, kind, de)


_check_palette()

# --------------------------------------------------------------------------- data
D1 = REPO / "results_foldlisten_ext2_2b9b/out"
D2 = REPO / "results_foldlisten_ext2_27b/out"

PANELS = [  # (title, summary path, faithful source: "native" | rescore path)
    ("2b base", D1 / "foldlisten_judge_fl_2bbase_ext2_summary.json", "native"),
    ("9b base", D1 / "foldlisten_judge_fl_9bbase_ext2_summary.json", "native"),
    ("27b base", D2 / "foldlisten_judge_fl_27bbase_ext2_summary.json", "native"),
    ("2b-it", D1 / "foldlisten_judge_fl_2bit_ext2_summary.json", "native"),
    ("9b-it", REPO / "results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json",
     REPO / "out/faithful_rescore_fl_9bit_ext2.json"),
    ("27b-it", D2 / "foldlisten_judge_fl_27bit_ext2_summary.json", "native"),
]

# Grounded final-stage (elicited, faithful-strict) label counts per cell — asserted before drawing.
EXPECT = {
    "fold": {"2b base": {"WSTAR": 16, "C": 15, "NEITHER": 51}, "2b-it": {"WSTAR": 68, "C": 14, "NEITHER": 0},
             "9b base": {"WSTAR": 3, "C": 41, "NEITHER": 38}, "9b-it": {"WSTAR": 55, "C": 27, "NEITHER": 0},
             "27b base": {"WSTAR": 11, "C": 39, "NEITHER": 32}, "27b-it": {"WSTAR": 55, "C": 26, "NEITHER": 1}},
    "listen": {"2b base": {"C": 25, "WSTAR": 10, "NEITHER": 47}, "2b-it": {"C": 81, "WSTAR": 1, "NEITHER": 0},
               "9b base": {"C": 11, "WSTAR": 34, "NEITHER": 37}, "9b-it": {"C": 82, "WSTAR": 0, "NEITHER": 0},
               "27b base": {"C": 20, "WSTAR": 34, "NEITHER": 28}, "27b-it": {"C": 82, "WSTAR": 0, "NEITHER": 0}},
}


def _bucket(lab):
    return "NEITHER" if lab == "UNRESOLVED_ALIAS" else lab


def load_panel(summary_path, faithful_src, cell):
    d = json.loads(Path(summary_path).read_text())
    items = [it for it in d["items"] if it["cell"] == cell]
    assert len(items) == 82, (summary_path, cell, len(items))
    if faithful_src == "native":
        seqs = [(it["faithful_neutral"], it["faithful_counter"], it["faithful_elicit"]) for it in items]
    else:
        r = json.loads(Path(faithful_src).read_text())
        cols = {}
        for arm in ("neutral_gen", "counter_gen", "elicit_gen"):
            recs = r["fields"][arm]["items"]
            assert all(a["q"] == b["q"] for a, b in zip(recs, d["items"]))   # index-aligned join
            cols[arm] = [rec["new_label"] for rec in recs]
        idx = [i for i, it in enumerate(d["items"]) if it["cell"] == cell]
        seqs = [(cols["neutral_gen"][i], cols["counter_gen"][i], cols["elicit_gen"][i]) for i in idx]
    ua = [sum(s[k] == "UNRESOLVED_ALIAS" for s in seqs) for k in range(3)]
    return [tuple(_bucket(x) for x in s) for s in seqs], ua


# --------------------------------------------------------------------------- drawing
NODE_W, GAP = 0.055, 1.6   # node bar half-width (x units), vertical gap between nodes (count units)


def _offsets(counts):
    tops, y = {}, 0.0
    for c in CATS:
        tops[c] = y
        y += counts.get(c, 0) + (GAP if counts.get(c, 0) else 0)
    return tops


def draw_panel(ax, seqs, ua, title):
    n = len(seqs)
    stage_counts = [{c: sum(1 for s in seqs if s[k] == c) for c in CATS} for k in range(3)]
    tops = [_offsets(sc) for sc in stage_counts]
    total_h = max(sum(sc.values()) + GAP * (sum(1 for c in CATS if sc.get(c, 0)) - 1) for sc in stage_counts)

    for k in range(3):
        for c in CATS:
            h = stage_counts[k].get(c, 0)
            if not h:
                continue
            ax.add_patch(plt.Rectangle((k - NODE_W, tops[k][c]), 2 * NODE_W, h,
                                       color=COL[c], lw=0, zorder=3))
            if h >= 5:
                ax.text(k + (0.09 if k == 2 else -0.09), tops[k][c] + h / 2, str(h),
                        ha="left" if k == 2 else "right", va="center", fontsize=7, color="#444444", zorder=4)

    for k in (0, 1):
        used_s = {c: 0.0 for c in CATS}
        used_d = {c: 0.0 for c in CATS}
        for cs in CATS:                      # fixed order -> deterministic, crossing-minimised enough
            for cd in CATS:
                w = sum(1 for s in seqs if s[k] == cs and s[k + 1] == cd)
                if not w:
                    continue
                y0 = tops[k][cs] + used_s[cs]; used_s[cs] += w
                y1 = tops[k + 1][cd] + used_d[cd]; used_d[cd] += w
                x0, x1 = k + NODE_W, k + 1 - NODE_W
                xm = (x0 + x1) / 2
                verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1),
                         (x1, y1 + w), (xm, y1 + w), (xm, y0 + w), (x0, y0 + w), (x0, y0)]
                codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                         MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MPath(verts, codes), facecolor=COL[cd], alpha=0.45, lw=0, zorder=2))

    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(total_h + GAP, -GAP)         # W* on top
    ax.set_xticks(range(3), STAGES, fontsize=8)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    sub = f"   (alias flags→neither: {'/'.join(map(str, ua))})" if any(ua) else ""
    ax.set_title(title + sub, fontsize=9.5, pad=6)
    ax.set_facecolor(SURFACE)


def make_fig(cell, out_png):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    fig.patch.set_facecolor(SURFACE)
    order = ["2b base", "9b base", "27b base", "2b-it", "9b-it", "27b-it"]
    by_title = {t: (p, src) for t, p, src in PANELS}
    for ax, title in zip(axes.flat, order):
        seqs, ua = load_panel(*by_title[title], cell)
        final = {c: sum(1 for s in seqs if s[2] == c) for c in CATS}
        assert final == {c: EXPECT[cell][title].get(c, 0) for c in CATS}, (cell, title, final)
        draw_panel(ax, seqs, ua, title)
    push = "wrong answer (W*) pushed" if cell == "fold" else "correct answer (C) pushed"
    fig.suptitle(f"Reply-state flows under pushback — {cell.upper()} cell ({push}), "
                 f"82-item family, faithful labels", fontsize=12, y=0.995)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.text(0.5, 0.055, "left transition compares PAIRED ARMS (neutral and counter branch from the same "
             "planted turn), not time; counter→elicited is sequential.  prose arms scored with the "
             "confidence mapping; the elicited slot strict (string-identity register) — NOTE_faithful_matcher.md",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    figdir = REPO / "docs/drafts/figs"
    make_fig("fold", figdir / "figB_fold_ext2.png")
    make_fig("listen", figdir / "figB_listen_ext2.png")

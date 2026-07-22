"""Fig B matrix forms — (1) a plain re-drive of the archived n=22 session's 12-panel layout on the
canonical registers, and (2) the synthesis of the two figure families.

Re-drive (`figB_matrix_redrive_ext2.png`): the archived form exactly — per panel ONE transition,
planted answer -> elicited final answer (the claim, nothing else) — but driven by the
faithful-strict ext2 (n=82) registers (alias-aware, elicited slot strict; the archived version drew
the retired confidence-mapped register at the base cells, see
archive/figs_sankey_matrix_n22_preport_2026-07-20/PROVENANCE_sankey_matrix_n22_preport.md).

Synthesis (`figB_synthesis_ext2.png`): the archived session's design language (hue = correctness,
Okabe-Ito, opacity = training, claim-first panels) fused with the measurement layer of
figB_{fold,listen}_ext2 — a THIRD column inserted so each panel reads planted -> counter free-reply
(top-line state) -> elicited final. All three stages are SEQUENTIAL in one transcript (unlike the
neutral column of figB_*_ext2, which is a paired control arm), so no paired-arms caveat is needed;
the neutral control appears instead as a per-panel annotation "neutral drift n/82" (faithful
register), which is the item-aggregated form of the gate's push-attributability check.

Every plotted elicited count asserted against the H3-grounded registers (EXPECT in
make_figB_sankey.py) before drawing. Palette = the archived session's Okabe-Ito trio, revalidated
here by the same inline Vienot+OKLab check (all pairs pass; identity also carried by labels).

Usage: python docs/drafts/figs/make_figB_matrix.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figB_sankey import PANELS, EXPECT, load_panel, _check_palette as _check_default  # noqa: E402
import make_figB_sankey as _fb  # noqa: E402

REPO = Path(__file__).resolve().parents[3]

HUE = {"C": "#009E73", "WSTAR": "#CC3311", "NEITHER": "#b0b0ab"}   # hue = correctness (Okabe-Ito)
SURFACE = "#ffffff"
NICE = {"C": "correct (C)", "WSTAR": "wrong (W*)", "NEITHER": "withholds"}
CATS = ["C", "WSTAR", "NEITHER"]
ALPHA = {"base": dict(node=0.60, rib=0.40), "it": dict(node=1.00, rib=0.58)}   # opacity = training

# Revalidate the archived palette with the same inline check used for figB (swap COL, restore after).
_saved = dict(_fb.COL)
_fb.COL.update(HUE)
_check_default()
_fb.COL.update(_saved)

BY_TITLE = {t: (p, src) for t, p, src in PANELS}
ROWS = [("fold", "base"), ("fold", "it"), ("listen", "base"), ("listen", "it")]
SCALES = ["2b", "9b", "27b"]
GAP, NODE_W = 2.2, 0.06


def _title(scale, training):
    return f"{scale} base" if training == "base" else f"{scale}-it"


def _panel(cell, scale, training):
    seqs, ua = load_panel(*BY_TITLE[_title(scale, training)], cell)
    final = {c: sum(1 for s in seqs if s[2] == c) for c in CATS}
    assert final == {c: EXPECT[cell][_title(scale, training)].get(c, 0) for c in CATS}, \
        (cell, scale, training, final)
    return seqs, ua, final


def _stack(counts):
    tops, y = {}, 0.0
    for c in CATS:
        tops[c] = y
        y += counts.get(c, 0) + (GAP if counts.get(c, 0) else 0)
    return tops, y - GAP


def _ribbon(ax, x0, y0, x1, y1, w, color, alpha):
    xm = (x0 + x1) / 2
    verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1),
             (x1, y1 + w), (xm, y1 + w), (xm, y0 + w), (x0, y0 + w), (x0, y0)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, alpha=alpha, lw=0, zorder=2))


def draw_two_stage(ax, cell, scale, training):
    """The archived form: planted -> elicited, one transition."""
    seqs, _, final = _panel(cell, scale, training)
    start_cat = "C" if cell == "fold" else "WSTAR"
    a = ALPHA[training]
    tops, h = _stack(final)
    ax.add_patch(plt.Rectangle((-NODE_W, 0), 2 * NODE_W, 82, facecolor=HUE[start_cat],
                               alpha=a["node"], lw=0, zorder=3))
    used = 0.0
    for c in CATS:
        n = final.get(c, 0)
        if not n:
            continue
        ax.add_patch(plt.Rectangle((1 - NODE_W, tops[c]), 2 * NODE_W, n, facecolor=HUE[c],
                                   alpha=a["node"], lw=0, zorder=3))
        _ribbon(ax, NODE_W, used, 1 - NODE_W, tops[c], n, HUE[c], a["rib"])
        used += n
        ax.text(1.09, tops[c] + n / 2, str(n), ha="left", va="center", fontsize=8.5, color="#333333")
    ax.set_xlim(-0.4, 1.55)
    ax.set_ylim(max(h, 82) + GAP, -GAP)
    ax.set_xticks([0, 1])                          # stage labels only on the bottom row (see make())
    ax.set_xticklabels([])


def draw_three_stage(ax, cell, scale, training):
    """The synthesis: planted -> counter free-reply -> elicited (sequential); neutral as annotation."""
    seqs, _, final = _panel(cell, scale, training)
    start_cat = "C" if cell == "fold" else "WSTAR"
    pushed = "WSTAR" if cell == "fold" else "C"
    a = ALPHA[training]
    mid = {c: sum(1 for s in seqs if s[1] == c) for c in CATS}
    drift = sum(1 for s in seqs if s[0] == pushed)
    tops1, h1 = _stack(mid)
    tops2, h2 = _stack(final)
    ax.add_patch(plt.Rectangle((-NODE_W, 0), 2 * NODE_W, 82, facecolor=HUE[start_cat],
                               alpha=a["node"], lw=0, zorder=3))
    used0 = 0.0
    for c in CATS:                                   # planted -> counter, colored by destination
        n = mid.get(c, 0)
        if not n:
            continue
        ax.add_patch(plt.Rectangle((1 - NODE_W, tops1[c]), 2 * NODE_W, n, facecolor=HUE[c],
                                   alpha=a["node"], lw=0, zorder=3))
        _ribbon(ax, NODE_W, used0, 1 - NODE_W, tops1[c], n, HUE[c], a["rib"])
        used0 += n
        if n >= 5:                                   # both blind readers had to pixel-estimate these
            ax.text(1, tops1[c] + n / 2, str(n), ha="center", va="center", fontsize=7,
                    color="#ffffff" if training == "it" else "#444444", zorder=5)
    used_s = {c: 0.0 for c in CATS}
    used_d = {c: 0.0 for c in CATS}
    for cs in CATS:                                  # counter -> elicited, colored by destination
        for cd in CATS:
            w = sum(1 for s in seqs if s[1] == cs and s[2] == cd)
            if not w:
                continue
            y0 = tops1[cs] + used_s[cs]; used_s[cs] += w
            y1 = tops2[cd] + used_d[cd]; used_d[cd] += w
            _ribbon(ax, 1 + NODE_W, y0, 2 - NODE_W, y1, w, HUE[cd], a["rib"])
    for c in CATS:
        n = final.get(c, 0)
        if not n:
            continue
        ax.add_patch(plt.Rectangle((2 - NODE_W, tops2[c]), 2 * NODE_W, n, facecolor=HUE[c],
                                   alpha=a["node"], lw=0, zorder=3))
        ax.text(2.09, tops2[c] + n / 2, str(n), ha="left", va="center", fontsize=8.5, color="#333333")
    ax.text(2.45, -GAP * 0.2, f"drift {drift}/82", fontsize=7, color="#6e6e6a",
            ha="right", va="bottom")                 # top-right: a panel property, not a planted-column one
    ax.set_xlim(-0.4, 2.55)
    ax.set_ylim(max(h1, h2, 82) + GAP * 3.2, -GAP)
    ax.set_xticks([0, 1, 2])                       # stage labels only on the bottom row (see make())
    ax.set_xticklabels([])


def make(kind, out_png, suptitle):
    draw = draw_two_stage if kind == "two" else draw_three_stage
    fig, axes = plt.subplots(4, 3, figsize=(11, 13.5) if kind == "two" else (12.5, 14))
    fig.patch.set_facecolor(SURFACE)
    for i, (cell, training) in enumerate(ROWS):
        for j, scale in enumerate(SCALES):
            ax = axes[i][j]
            draw(ax, cell, scale, training)
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            ax.set_facecolor(SURFACE)
            if i == 0:
                ax.set_title(scale, fontsize=13, pad=10)
            if i == len(ROWS) - 1:                 # shared stage axis: label the bottom row only
                labs = ["planted", "elicited"] if kind == "two" else ["planted", "counter\nreply", "elicited"]
                ax.set_xticklabels(labs, fontsize=8)
            if j == 0:
                lab = f"{cell.upper()}\n{'base' if training == 'base' else '-it'}"
                start = "start: C planted" if cell == "fold" else "start: W* planted"
                ax.set_ylabel(f"{lab}\n({start})", fontsize=9, rotation=0, ha="right", va="center",
                              labelpad=44)
    handles = [plt.Rectangle((0, 0), 1, 1, color=HUE[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(suptitle, fontsize=13, y=0.995)
    note = ("faithful-strict elicited register (alias-aware; validated vs blind hand-labels at every scale); "
            "muted = base, bold = -it")
    if kind == "three":
        note = ("counter column = top-line state of the free reply (prose register; at base, green/red is often a bare\n"
                "“Yes, I'm sure.”-type reaffirmation mapped to the planted/pushed answer, not the name itself). "
                "ribbons are colored by DESTINATION state.\n"
                "drift n/82 = items whose reply moved to the pushed answer in a control arm whose second turn is only "
                "“Okay, thank you.” (no push)\n" + note)
    fig.text(0.5, 0.04, note, ha="center", fontsize=7.5, color="#6e6e6a")
    fig.tight_layout(rect=(0.02, 0.06, 1, 0.97))
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    figdir = REPO / "docs/drafts/figs"
    make("two", figdir / "figB_matrix_redrive_ext2.png",
         "Where each planted answer lands after pushback — 82-item family, faithful labels")
    make("three", figdir / "figB_synthesis_ext2.png",
         "Planted answer → free reply under pushback → elicited final — 82-item family, faithful labels")

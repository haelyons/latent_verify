"""Fig B (neutral counterfactual) minimal: the no-push control arm beside the pushback arm.

Body sankeys (figB_synthesis_*) draw the push arm only. This adds the NEUTRAL arm (same planted first
turn, user's second turn only "Okay, thank you.", no argument) as the counterfactual. Reply-layer
comparison only: the protocol elicits a forced final ONLY after the counter turn, so there is no
neutral-elicited slot (scoped here, stated in the caption). Left = control (planted, reply); right =
push (planted, reply, elicited). Rows ordered base above -it (matching the sankey matrix).

Internally MECE: within every panel each column partitions the same 82 items into C / W* / neither and
sums to 82 (asserted before drawing). Strict register (a segment is colored only if the turn NAMES
that answer). Every count asserted vs the grounded distributions. Okabe-Ito palette (CVD-checked in
make_figB_matrix). Everything not load-bearing lives in figB_neutral_counterfactual_caption.md, not on
the figure.

Usage: python docs/drafts/figs/make_figB_neutral_counterfactual.py
"""
import sys, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "controls")); sys.path.insert(0, str(REPO))
from faithful_rescore import classify  # noqa: E402

HUE = {"C": "#009E73", "WSTAR": "#CC3311", "NEITHER": "#b0b0ab"}
NICE = {"C": "correct (C)", "WSTAR": "wrong (W*)", "NEITHER": "names neither"}
CATS = ["C", "WSTAR", "NEITHER"]
SURFACE = "#ffffff"
GAP, NODE_W = 2.2, 0.06
ALPHA = {"base": dict(node=0.60, rib=0.40), "it": dict(node=1.00, rib=0.58)}

# base above -it, matching the sankey matrix
PANELS = [
    ("9B-base", "results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json", "base"),
    ("9B-it",   "results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json", "it"),
]
CELL = "fold"
EXPECT = {
    "9B-base": {"neutral": {"NEITHER": 82},
                "counter": {"NEITHER": 82},
                "elicit":  {"C": 41, "WSTAR": 3, "NEITHER": 38}},
    "9B-it":   {"neutral": {"C": 1, "NEITHER": 81},
                "counter": {"C": 15, "WSTAR": 50, "NEITHER": 17},
                "elicit":  {"C": 27, "WSTAR": 55}},
}


def _labels(path):
    d = json.loads(Path(path).read_text())
    items = [it for it in d["items"] if it["cell"] == CELL]
    assert len(items) == 82, (path, len(items))
    out = []
    for it in items:
        row = {"planted": "C"}
        for stage, field in (("neutral", "neutral_gen"), ("counter", "counter_gen"), ("elicit", "elicit_gen")):
            strict = True   # one register throughout
            lab = classify(it.get(field) or "", it["correct"], it["Wstar"],
                           it.get("stated"), it.get("pushed"), map_confidence=not strict)[0]
            row[stage] = "NEITHER" if lab == "UNRESOLVED_ALIAS" else lab
        out.append(row)
    return out


def _stack(counts):
    tops, y = {}, 0.0
    for c in CATS:
        tops[c] = y
        y += counts.get(c, 0) + (GAP if counts.get(c, 0) else 0)
    return tops


def _ribbon(ax, x0, y0, x1, y1, w, color, alpha):
    xm = (x0 + x1) / 2
    verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1),
             (x1, y1 + w), (xm, y1 + w), (xm, y0 + w), (x0, y0 + w), (x0, y0)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, alpha=alpha, lw=0, zorder=2))


def _node(ax, x, tops, counts, a, training):
    for c in CATS:
        n = counts.get(c, 0)
        if not n:
            continue
        ax.add_patch(plt.Rectangle((x - NODE_W, tops[c]), 2 * NODE_W, n, facecolor=HUE[c],
                                   alpha=a["node"], lw=0, zorder=3))
        if n >= 4:                                   # centered-on-bar (the legible placement)
            white = training == "it" and c in ("C", "WSTAR")
            ax.text(x, tops[c] + n / 2, str(n), ha="center", va="center", fontsize=8,
                    color="#ffffff" if white else "#333333", zorder=5)


def _flow(ax, xs, xd, tops_s, tops_d, seqs, sk, dk, a):
    us = {c: 0.0 for c in CATS}; ud = {c: 0.0 for c in CATS}
    for cs in CATS:
        for cd in CATS:
            w = sum(1 for s in seqs if s[sk] == cs and s[dk] == cd)
            if not w:
                continue
            y0 = tops_s[cs] + us[cs]; us[cs] += w
            y1 = tops_d[cd] + ud[cd]; ud[cd] += w
            _ribbon(ax, xs + NODE_W, y0, xd - NODE_W, y1, w, HUE[cd], a["rib"])


def draw_control(ax, seqs, exp, a, training):
    planted = {"C": 82}
    tp, tn = _stack(planted), _stack(exp["neutral"])
    _node(ax, 0, tp, planted, a, training)
    _flow(ax, 0, 1, tp, tn, seqs, "planted", "neutral", a)
    _node(ax, 1, tn, exp["neutral"], a, training)
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(84 + GAP, -GAP); ax.set_xticks([0, 1])


def draw_push(ax, seqs, exp, a, training):
    planted = {"C": 82}
    tp, tc, te = _stack(planted), _stack(exp["counter"]), _stack(exp["elicit"])
    _node(ax, 0, tp, planted, a, training)
    _flow(ax, 0, 1, tp, tc, seqs, "planted", "counter", a)
    _node(ax, 1, tc, exp["counter"], a, training)
    _flow(ax, 1, 2, tc, te, seqs, "counter", "elicit", a)
    _node(ax, 2, te, exp["elicit"], a, training)
    ax.set_xlim(-0.4, 2.4); ax.set_ylim(84 + GAP, -GAP); ax.set_xticks([0, 1, 2])


def make(out_png):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), gridspec_kw={"width_ratios": [2, 3]})
    fig.patch.set_facecolor(SURFACE)
    for i, (title, path, shade) in enumerate(PANELS):
        seqs = _labels(path)
        exp = EXPECT[title]
        for stage in ("neutral", "counter", "elicit"):
            got = {c: sum(1 for s in seqs if s[stage] == c) for c in CATS if sum(1 for s in seqs if s[stage] == c)}
            assert got == exp[stage], (title, stage, got)
            assert sum(exp[stage].values()) == 82, (title, stage)   # internal MECE
        a = ALPHA[shade]
        draw_control(axes[i][0], seqs, exp, a, shade)
        draw_push(axes[i][1], seqs, exp, a, shade)
        axes[i][0].set_ylabel(title, fontsize=12, rotation=0, ha="right", va="center", labelpad=16)
        for ax in axes[i]:
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0); ax.set_facecolor(SURFACE); ax.set_xticklabels([])
        if i == 0:
            axes[i][0].set_title("no pushback", fontsize=12, pad=8)
            axes[i][1].set_title("pushback", fontsize=12, pad=8)
    axes[1][0].set_xticklabels(["planted", "reply"], fontsize=9)
    axes[1][1].set_xticklabels(["planted", "reply", "elicited"], fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=HUE[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0.03, 0.06, 1, 0.98))
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    make(REPO / "docs/drafts/figs/figB_neutral_counterfactual_ext2.png")

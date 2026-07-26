"""Fig B (neutral counterfactual) minimal: the no-push control arm beside the pushback arm.

Body sankeys (figB_synthesis_*) draw the push arm only. This adds the NEUTRAL arm (same planted first
turn, user's second turn only "Okay, thank you.", no argument) as the counterfactual. Reply-layer
comparison only: the protocol elicits a forced final ONLY after the counter turn, so there is no
neutral-elicited slot (scoped here, stated in the caption). Left = control (planted, reply); right =
push (planted, reply, elicited). Rows ordered base above -it (matching the sankey matrix).

FOUR states, not three (2026-07-26). The gray band used to mean two different things: at base a reply
that names NO answer (a hedge string), at -it a reply that names BOTH answers which the matcher declines
to resolve. Those are not the same event, so the base and -it gray bands were not comparable. BOTH is now
its own state: the matcher returns NEITHER/UNRESOLVED_ALIAS *and* the isolated answer span contains both
the correct and the W* entity, tested with the repo's own word-boundary matching (_occurrences /
_entity_regexes from faithful_rescore, so alias + accent handling stays identical to the labeller). Gray
therefore now means only "the matcher resolves neither answer".

Internally MECE: within every panel each column partitions the same 82 items into C / W* / both / neither
and sums to 82 (asserted before drawing). Strict register (a segment is colored only if the turn NAMES
that answer). Every count asserted vs the grounded distributions. Okabe-Ito palette (CVD-checked in
make_figB_matrix; the BOTH blue re-checked against the existing trio with make_figB_sankey's Vienot +
OKLab checker). Everything not load-bearing lives in figB_neutral_counterfactual_caption.md, not on
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
from faithful_rescore import classify, _occurrences  # noqa: E402
from family_generate_judge import _norm  # noqa: E402

# BOTH = Okabe-Ito blue. Chosen over reddish purple (#CC79A7 fails the repo's own CVD floor: deutan dE 7.8
# vs the gray) and over orange (#E69F00 clears at 11.5 but sits on the orange/red confusion axis right
# beside the W* red in large ribbons). Blue clears at min CVD dE 17.5 — double the floor, and better than
# the figure's existing weakest pair (C/W* deutan 11.7) — because neither protan nor deutan touches the
# S-cone channel. It is also orthogonal to the green/red answer-identity axis, so "names both" reads as a
# different KIND of state rather than as a third answer.
HUE = {"C": "#009E73", "WSTAR": "#CC3311", "BOTH": "#0072B2", "NEITHER": "#b0b0ab"}
NICE = {"C": "correct (C)", "WSTAR": "wrong (W*)", "BOTH": "names both", "NEITHER": "names neither"}
CATS = ["C", "WSTAR", "BOTH", "NEITHER"]
SURFACE = "#ffffff"
GAP, NODE_W = 2.2, 0.06
ALPHA = {"base": dict(node=0.60, rib=0.40), "it": dict(node=1.00, rib=0.58)}

# base above -it, matching the sankey matrix
PANELS = [
    ("9B-base", "results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json", "base"),
    ("9B-it",   "results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json", "it"),
]
# fold: C planted, W* pushed; listen: W* planted, C pushed. Same strict register.
PLANTED = {"fold": "C", "listen": "WSTAR"}
# Derived from the artifacts by this script's own labeller (four states, strict register), 2026-07-26.
# Zero-count states are omitted (the observed dict is built the same way), so a state appearing here is
# nonzero by construction. The -it counter gray residue is the interesting cell: at 9B-it 7 gray splits
# 5 BOTH / 2 NEITHER, and the 2 are a plural-form matcher miss, not a no-answer reply — see the caption.
EXPECT = {
    "fold": {
        "9B-base": {"neutral": {"NEITHER": 82},
                    "counter": {"NEITHER": 82},
                    "elicit":  {"C": 41, "WSTAR": 3, "NEITHER": 38}},
        "9B-it":   {"neutral": {"C": 1, "NEITHER": 81},
                    "counter": {"C": 25, "WSTAR": 50, "BOTH": 5, "NEITHER": 2},
                    "elicit":  {"C": 27, "WSTAR": 55}},
    },
    "listen": {
        "9B-base": {"neutral": {"C": 2, "NEITHER": 80},
                    "counter": {"NEITHER": 82},
                    "elicit":  {"C": 11, "WSTAR": 34, "NEITHER": 37}},
        "9B-it":   {"neutral": {"C": 5, "WSTAR": 1, "BOTH": 4, "NEITHER": 72},
                    "counter": {"C": 67, "WSTAR": 1, "BOTH": 13, "NEITHER": 1},
                    "elicit":  {"C": 82}},
    },
}


def _state(gen, correct, wstar, stated, pushed):
    """Four-state label for one turn, strict register. An unresolved verdict (NEITHER or
    UNRESOLVED_ALIAS) splits on whether the ISOLATED ANSWER SPAN names both entities, tested with the
    labeller's own word-boundary forms (_occurrences / _entity_regexes) rather than a substring check, so
    alias and accent handling stay identical to the label itself. NB the same word-boundary rule is why 2
    plural-form spans per -it cell land in NEITHER, not BOTH (documented in the caption)."""
    lab, _rule, span = classify(gen or "", correct, wstar, stated, pushed, map_confidence=False)
    if lab in ("NEITHER", "UNRESOLVED_ALIAS"):
        t = _norm(span)
        if _occurrences(t, correct) and _occurrences(t, wstar):
            return "BOTH"
        return "NEITHER"
    return lab


def _labels(path, cell):
    d = json.loads(Path(path).read_text())
    items = [it for it in d["items"] if it["cell"] == cell]
    assert len(items) == 82, (path, len(items))
    out = []
    for it in items:
        row = {"planted": PLANTED[cell]}
        for stage, field in (("neutral", "neutral_gen"), ("counter", "counter_gen"), ("elicit", "elicit_gen")):
            row[stage] = _state(it.get(field), it["correct"], it["Wstar"], it.get("stated"), it.get("pushed"))
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
            white = training == "it" and c in ("C", "WSTAR", "BOTH")   # all three are dark fills
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


# One vertical scale for every panel, so a bar of 82 is the same height everywhere. Worst case is all
# four states nonzero in one column: 82 items + a GAP between each adjacent pair.
YMAX = 82 + (len(CATS) - 1) * GAP


def draw_control(ax, seqs, exp, a, training, planted_cat):
    planted = {planted_cat: 82}
    tp, tn = _stack(planted), _stack(exp["neutral"])
    _node(ax, 0, tp, planted, a, training)
    _flow(ax, 0, 1, tp, tn, seqs, "planted", "neutral", a)
    _node(ax, 1, tn, exp["neutral"], a, training)
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(YMAX + GAP, -GAP); ax.set_xticks([0, 1])


def draw_push(ax, seqs, exp, a, training, planted_cat):
    planted = {planted_cat: 82}
    tp, tc, te = _stack(planted), _stack(exp["counter"]), _stack(exp["elicit"])
    _node(ax, 0, tp, planted, a, training)
    _flow(ax, 0, 1, tp, tc, seqs, "planted", "counter", a)
    _node(ax, 1, tc, exp["counter"], a, training)
    _flow(ax, 1, 2, tc, te, seqs, "counter", "elicit", a)
    _node(ax, 2, te, exp["elicit"], a, training)
    ax.set_xlim(-0.4, 2.4); ax.set_ylim(YMAX + GAP, -GAP); ax.set_xticks([0, 1, 2])


def make(out_png, cell="fold"):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), gridspec_kw={"width_ratios": [2, 3]})
    fig.patch.set_facecolor(SURFACE)
    for i, (title, path, shade) in enumerate(PANELS):
        seqs = _labels(path, cell)
        exp = EXPECT[cell][title]
        for stage in ("neutral", "counter", "elicit"):
            got = {c: sum(1 for s in seqs if s[stage] == c) for c in CATS if sum(1 for s in seqs if s[stage] == c)}
            assert got == exp[stage], (title, stage, got, exp[stage])
            assert sum(exp[stage].values()) == 82, (title, stage)   # internal MECE
            print("[ok] %-6s %-8s %-7s %s" % (cell, title, stage,
                  " ".join("%s=%d" % (c, exp[stage].get(c, 0)) for c in CATS)))
        a = ALPHA[shade]
        draw_control(axes[i][0], seqs, exp, a, shade, PLANTED[cell])
        draw_push(axes[i][1], seqs, exp, a, shade, PLANTED[cell])
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
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0.03, 0.06, 1, 0.98))
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    make(REPO / "docs/drafts/figs/figB_neutral_counterfactual_ext2.png", cell="fold")
    make(REPO / "docs/drafts/figs/figB_neutral_counterfactual_listen_ext2.png", cell="listen")

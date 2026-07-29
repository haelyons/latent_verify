"""Fig B (neutral counterfactual) minimal: the no-push control arm beside the pushback arm.

Body sankeys (figB_synthesis_*) draw the push arm only. This adds the NEUTRAL arm (same planted first
turn, user's second turn only "Okay, thank you.", no argument) as the counterfactual. BOTH arms now run
to the forced final answer: the neutral-elicited slot (added to controls/foldlisten_judge.py 2026-07-26)
was filled by the nelicit runs, and per-item faithful_neutral_elicit is present on 82/82 items in all
twelve cell-directions, so the control arm's elicited column is DRAWN as of 2026-07-29 — it is no longer
a reply-layer-only comparison. Left = control (planted, reply, elicited-without-push); right = push
(planted, reply, elicited). Rows ordered base above -it (matching the sankey matrix).

READ THE BASE CONTROL'S ELICITED GRAY BAND WITH CARE. The neutral-elicited answers are one-word slot
fills off a base model that was never asked to argue, and at 9B base the matcher cannot resolve about a
third of them: 29 of 82 in the fold cell and 26 of 82 in the listen cell are UNRESOLVED_ALIAS, which this figure
(like every other) folds into gray. So the base control's gray bar is "no answer the matcher can pin
down", part hedge and part alias miss, not "withheld". The per-panel annotation prints the alias count
for both control columns whenever nonzero, and the caption carries the same disclosure.

FOUR states, not three (2026-07-26). The gray band used to mean two different things: at base a reply
that names NO answer (a hedge string), at -it a reply that names BOTH answers which the matcher declines
to resolve. Those are not the same event, so the base and -it gray bands were not comparable. BOTH is now
its own state: the matcher returns NEITHER/UNRESOLVED_ALIAS *and* the isolated answer span contains both
the correct and the W* entity, tested with the repo's own word-boundary matching (_occurrences /
_entity_regexes from faithful_rescore, so alias + accent handling stays identical to the labeller). Gray
therefore now means only "the matcher resolves neither answer".

Internally MECE: within every panel each column partitions the same 82 items into C / W* / both / neither
and sums to 82 (asserted before drawing). An answer counts as named only when the model spells it out
(map_confidence=False), so a bare "Yes, I'm sure." names nothing and is colored gray in every column.
Every count asserted vs the grounded distributions. Okabe-Ito palette (CVD-checked in
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
# beside the W* red in large ribbons). Blue's worst pair clears at CVD dE 17.5 — twice the floor of 8, and
# better separated than the palette's PRE-EXISTING weakest pair (C vs NEITHER, protan 10.2) — because
# neither protan nor deutan touches the S-cone channel. It is also orthogonal to the green/red
# answer-identity axis, so "names both" reads as a different KIND of state rather than as a third answer.
HUE = {"C": "#009E73", "WSTAR": "#CC3311", "BOTH": "#0072B2", "NEITHER": "#b0b0ab"}
NICE = {"C": "correct (C)", "WSTAR": "wrong (W*)", "BOTH": "names both", "NEITHER": "names neither"}
CATS = ["C", "WSTAR", "BOTH", "NEITHER"]
SURFACE = "#ffffff"
GAP, NODE_W = 2.2, 0.06
ALPHA = {"base": dict(node=0.60, rib=0.40), "it": dict(node=1.00, rib=0.58)}

# base above -it, matching the sankey matrix. Repointed 2026-07-29 at the NELICIT draws, which are the
# only summaries carrying neutral_elicit_gen / faithful_neutral_elicit. Their neutral_gen, counter_gen and
# elicit_gen are BYTE-IDENTICAL per item to the previous sources (results_foldlisten_ext2_2b9b and
# results_foldlisten_r2), checked item by item, so the three columns this figure already drew do not move
# — only the fourth is new. (At 27b the two draws do differ, and there the committed one is the anomaly;
# this figure draws 9B only, so that does not arise here.)
PANELS = [
    ("9B-base", "results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json", "base"),
    ("9B-it",   "results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_9bit_ext2_summary.json", "it"),
]
# fold: C planted, W* pushed; listen: W* planted, C pushed. Same naming rule in both.
PLANTED = {"fold": "C", "listen": "WSTAR"}
# The four drawn stages, in the order the two arms use them. "nelicit" is the control arm's forced final
# answer (neutral second turn, then "What is your final answer?"), new 2026-07-29.
STAGES = ("neutral", "counter", "elicit", "nelicit")
# Derived from the artifacts by this script's own labeller (four states, map_confidence=False),
# 2026-07-26, after the entity_forms_v2 regular-plural fix (2c5a8bf); the "nelicit" column added
# 2026-07-29 from neutral_elicit_gen. Zero-count states are omitted (the observed dict is built the same
# way), so a state appearing here is nonzero by construction. The -it counter gray band is now EMPTY in
# both cells: with "beavers" matchable against Beaver, every -it counter reply the matcher cannot resolve
# turns out to name both answers. Two labels moved in each cell and the caption states both — fold
# W* 50 -> 52, listen BOTH 13 -> 14 with C held at 67. NB the 9B-base "nelicit" gray bands are inflated by
# UNRESOLVED_ALIAS (fold 29, listen 26 of 82) — see UA_EXPECT below and the caption.
EXPECT = {
    "fold": {
        "9B-base": {"neutral": {"NEITHER": 82},
                    "counter": {"NEITHER": 82},
                    "elicit":  {"C": 41, "WSTAR": 3, "NEITHER": 38},
                    "nelicit": {"C": 27, "WSTAR": 3, "NEITHER": 52}},
        "9B-it":   {"neutral": {"C": 1, "NEITHER": 81},
                    "counter": {"C": 25, "WSTAR": 52, "BOTH": 5},
                    "elicit":  {"C": 27, "WSTAR": 55},
                    "nelicit": {"C": 82}},
    },
    "listen": {
        "9B-base": {"neutral": {"C": 2, "NEITHER": 80},
                    "counter": {"NEITHER": 82},
                    "elicit":  {"C": 11, "WSTAR": 34, "NEITHER": 37},
                    "nelicit": {"C": 15, "WSTAR": 18, "NEITHER": 49}},
        "9B-it":   {"neutral": {"C": 5, "WSTAR": 1, "BOTH": 4, "NEITHER": 72},
                    "counter": {"C": 67, "WSTAR": 1, "BOTH": 14},
                    "elicit":  {"C": 82},
                    "nelicit": {"WSTAR": 55, "C": 25, "NEITHER": 2}},
    },
}
# UNRESOLVED_ALIAS counts folded into the gray band, per panel, for the two CONTROL-arm columns. Frozen
# and asserted like the state counts, because the honesty of the control arm's gray bar depends on them:
# the base neutral-elicited column is about a third alias-unresolved, not a third withheld.
UA_EXPECT = {
    "fold":   {"9B-base": {"neutral": 2, "nelicit": 29}, "9B-it": {"neutral": 0, "nelicit": 0}},
    "listen": {"9B-base": {"neutral": 2, "nelicit": 26}, "9B-it": {"neutral": 0, "nelicit": 2}},
}


def _state(gen, correct, wstar, stated, pushed):
    """Four-state label for one turn. An answer counts as named only when the model spells it out
    (map_confidence=False), so a bare "Yes, I'm sure." names nothing. An unresolved verdict (NEITHER or
    UNRESOLVED_ALIAS) splits on whether the ISOLATED ANSWER SPAN names both entities, tested with the
    labeller's own word-boundary forms (_occurrences / _entity_regexes) rather than a substring check, so
    alias, accent AND plural handling stay identical to the label itself — entity_forms_v2 emits the
    regular plural as of 2c5a8bf, so "beavers" matches Beaver and no plural span is stranded in gray.
    Returns (state, was_unresolved_alias) — the flag is counted per stage so the gray band can say how
    much of itself is an alias miss rather than a withheld answer."""
    lab, _rule, span = classify(gen or "", correct, wstar, stated, pushed, map_confidence=False)
    ua = lab == "UNRESOLVED_ALIAS"
    if lab in ("NEITHER", "UNRESOLVED_ALIAS"):
        t = _norm(span)
        if _occurrences(t, correct) and _occurrences(t, wstar):
            return "BOTH", ua          # an alias miss names neither entity, so this branch never fires on ua
        return "NEITHER", ua
    return lab, ua


FIELDS = {"neutral": "neutral_gen", "counter": "counter_gen",
          "elicit": "elicit_gen", "nelicit": "neutral_elicit_gen"}


def _labels(path, cell):
    d = json.loads(Path(path).read_text())
    items = [it for it in d["items"] if it["cell"] == cell]
    assert len(items) == 82, (path, len(items))
    assert all(it.get("neutral_elicit_gen") is not None for it in items), (path, "no neutral-elicited arm")
    out, ua = [], {st: 0 for st in STAGES}
    for it in items:
        row = {"planted": PLANTED[cell]}
        for stage in STAGES:
            row[stage], was_ua = _state(it.get(FIELDS[stage]), it["correct"], it["Wstar"],
                                        it.get("stated"), it.get("pushed"))
            ua[stage] += was_ua
        out.append(row)
    return out, ua


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


def draw_control(ax, seqs, exp, a, training, planted_cat, ua=None):
    """The no-push arm: planted -> free reply -> forced final answer. The third column is the
    neutral-ELICITED slot, drawn since 2026-07-29; both arms therefore have the same three stages and are
    directly comparable column for column."""
    planted = {planted_cat: 82}
    tp, tn, tne = _stack(planted), _stack(exp["neutral"]), _stack(exp["nelicit"])
    _node(ax, 0, tp, planted, a, training)
    _flow(ax, 0, 1, tp, tn, seqs, "planted", "neutral", a)
    _node(ax, 1, tn, exp["neutral"], a, training)
    _flow(ax, 1, 2, tn, tne, seqs, "neutral", "nelicit", a)
    _node(ax, 2, tne, exp["nelicit"], a, training)
    if ua and (ua["neutral"] or ua["nelicit"]):
        # The gray band is part hedge, part alias miss; say which without making the reader open the caption.
        ax.text(2.4, -GAP * 0.35, "alias-unresolved in gray: reply %d, elicited %d"
                % (ua["neutral"], ua["nelicit"]), fontsize=6.5, color="#6e6e6a", ha="right", va="bottom")
    ax.set_xlim(-0.4, 2.4); ax.set_ylim(YMAX + GAP, -GAP); ax.set_xticks([0, 1, 2])


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
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6))
    fig.patch.set_facecolor(SURFACE)
    for i, (title, path, shade) in enumerate(PANELS):
        seqs, ua = _labels(path, cell)
        exp = EXPECT[cell][title]
        for stage in STAGES:
            got = {c: sum(1 for s in seqs if s[stage] == c) for c in CATS if sum(1 for s in seqs if s[stage] == c)}
            assert got == exp[stage], (title, stage, got, exp[stage])
            assert sum(exp[stage].values()) == 82, (title, stage)   # internal MECE
            print("[ok] %-6s %-8s %-7s %s" % (cell, title, stage,
                  " ".join("%s=%d" % (c, exp[stage].get(c, 0)) for c in CATS)))
        for stage in ("neutral", "nelicit"):                        # control-arm alias disclosure
            assert ua[stage] == UA_EXPECT[cell][title][stage], (title, stage, ua[stage])
            print("[ok] %-6s %-8s %-7s UNRESOLVED_ALIAS folded into gray: %d/82"
                  % (cell, title, stage, ua[stage]))
        a = ALPHA[shade]
        draw_control(axes[i][0], seqs, exp, a, shade, PLANTED[cell], ua)
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
    axes[1][0].set_xticklabels(["planted", "reply", "elicited"], fontsize=9)
    axes[1][1].set_xticklabels(["planted", "reply", "elicited"], fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=HUE[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0.03, 0.06, 1, 0.98))
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    make(REPO / "docs/drafts/figs/figB_neutral_counterfactual_ext2.png", cell="fold")
    make(REPO / "docs/drafts/figs/figB_neutral_counterfactual_listen_ext2.png", cell="listen")

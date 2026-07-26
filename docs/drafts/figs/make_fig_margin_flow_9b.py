"""Fig (margin flow, 9B) companion: which answer the model would GIVE, in the two arms side by side.

Layout mirrors make_figB_neutral_counterfactual.py exactly - two arms ("no pushback", "pushback"), rows
9B-base above 9B-it, same _stack/_ribbon/_flow/_node helpers, same geometry, same palette - because the
two figures are the same experiment read at two layers.

WHY TWO ARMS AND NOT THREE STAGES. Mc_neutral and Mc_counter are measured on two ALTERNATIVE prompts,
push(q, C, NEUTRAL) and push(q, C, PUSH['counter']): two different second user turns branching from the
same planted first turn. They are a paired-arm comparison, not successive moments, so drawing them
left-to-right implied a chronology that does not exist (make_figB_sankey.py flags the same hazard for its
own first ribbon set). Each arm here is one transcript: the bare question, then that arm's second turn.
The bare column is shared by both arms by construction - it is the same single-turn prompt.

WHAT IS PLOTTED. Nothing is generated. At each point an answer slot is scored: the log-probability of the
correct answer C against the log-probability of the wrong rival W*, and the plotted state is the SIGN of
that difference (M0, Mc_neutral, Mc_counter in the artifacts). It is the answer the model would give if
asked for a final answer right there. It is NOT the sankeys' elicited slot: that one comes after the
model has written a free reply, whereas these come immediately after the user's turn with no reply in the
context. Because a distribution always favours one side or sits exactly on the fence, "names neither"
does not exist by construction - the third state is an exact tie (no epsilon band; base has 1 at the bare
question and 3 under the push). It follows that this figure does NOT arbitrate the string-matched
figures; the two layers disagree per item (see the caption).

FOLD only (C planted, W* pushed). Positive = C favoured, negative = W* favoured, exactly 0 = tie. The
gray third slot of the sibling figure is reused as the neutral midpoint of a signed quantity. Every
column partitions the same 82 items and sums to 82; all counts and all per-item flows are asserted before
drawing, including the neutral-vs-counter pairing, which is not drawn but is quoted in the caption.
Everything not load-bearing lives in fig_margin_flow_9b_caption.md, not on the figure.

Usage: python docs/drafts/figs/make_fig_margin_flow_9b.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

REPO = Path(__file__).resolve().parents[3]

HUE = {"C": "#009E73", "WSTAR": "#CC3311", "TIE": "#b0b0ab"}
NICE = {"C": "favours correct (C)", "WSTAR": "favours wrong (W*)", "TIE": "tie (margin = 0)"}
CATS = ["C", "WSTAR", "TIE"]
SURFACE = "#ffffff"
GAP, NODE_W = 2.2, 0.06
ALPHA = {"base": dict(node=0.60, rib=0.40), "it": dict(node=1.00, rib=0.58)}

STAGES = [("bare", "M0"), ("neutral", "Mc_neutral"), ("counter", "Mc_counter")]
ARMS = [("neutral", "no pushback", ["bare question", "after the neutral turn"]),
        ("counter", "pushback", ["bare question", "after the push"])]

# base above -it, matching make_figB_neutral_counterfactual.py
PANELS = [
    ("9B-base", "results_absdecode_ext2/out/family_cave_diagnose_vfam_ext2_9bbase.json", "base"),
    ("9B-it",   "results_itreadout_modelw/out/family_cave_diagnose_vfam_ext2_9bit.json", "it"),
]
FAMILY = "verifier_family_ext2.json"

EXPECT = {
    "9B-base": {"bare":    {"C": 70, "WSTAR": 11, "TIE": 1},
                "neutral": {"C": 81, "WSTAR": 1},
                "counter": {"C": 63, "WSTAR": 16, "TIE": 3}},
    "9B-it":   {"bare":    {"C": 72, "WSTAR": 10},
                "neutral": {"C": 75, "WSTAR": 7},
                "counter": {"C": 27, "WSTAR": 55}},
}
# per-item, not marginal. bare->neutral and bare->counter are the two drawn arms; neutral->counter is the
# paired-arm comparison BETWEEN them - not drawn (it is not a transition), asserted because the caption
# quotes it (base 15 of 82 C->W*, -it 48 of 82 C->W*).
EXPECT_FLOW = {
    "9B-base": {
        ("bare", "neutral"): {("C", "C"): 70, ("WSTAR", "C"): 10, ("WSTAR", "WSTAR"): 1, ("TIE", "C"): 1},
        ("bare", "counter"): {("C", "C"): 57, ("C", "WSTAR"): 10, ("C", "TIE"): 3,
                              ("WSTAR", "C"): 5, ("WSTAR", "WSTAR"): 6, ("TIE", "C"): 1},
        ("neutral", "counter"): {("C", "C"): 63, ("C", "WSTAR"): 15, ("C", "TIE"): 3,
                                 ("WSTAR", "WSTAR"): 1},
    },
    "9B-it": {
        ("bare", "neutral"): {("C", "C"): 69, ("C", "WSTAR"): 3, ("WSTAR", "C"): 6, ("WSTAR", "WSTAR"): 4},
        ("bare", "counter"): {("C", "C"): 26, ("C", "WSTAR"): 46, ("WSTAR", "C"): 1, ("WSTAR", "WSTAR"): 9},
        ("neutral", "counter"): {("C", "C"): 27, ("C", "WSTAR"): 48, ("WSTAR", "WSTAR"): 7},
    },
}


def _sign(v):
    return "C" if v > 0 else ("WSTAR" if v < 0 else "TIE")


def _load(path, family_qs):
    items = json.loads((REPO / path).read_text())["result"]["items"]
    assert len(items) == 82, (path, len(items))
    qs = [it["q"] for it in items]
    assert len(set(qs)) == 82, (path, "duplicate q")
    assert set(qs) == family_qs, (path, "item set != verifier_family_ext2")
    return {it["q"]: {st: _sign(it[field]) for st, field in STAGES} for it in items}


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
        else:                                        # too thin to hold a number (make_figB_matrix rule):
            ax.text(x + NODE_W + 0.03, tops[c] + n / 2, str(n), ha="left", va="center",
                    fontsize=7, color="#666666", zorder=5)   # ties are load-bearing, so still labelled


def _flow(ax, xs, xd, tops_s, tops_d, seqs, sk, dk, a):
    us = {c: 0.0 for c in CATS}
    ud = {c: 0.0 for c in CATS}
    for cs in CATS:
        for cd in CATS:
            w = sum(1 for s in seqs if s[sk] == cs and s[dk] == cd)
            if not w:
                continue
            y0 = tops_s[cs] + us[cs]; us[cs] += w
            y1 = tops_d[cd] + ud[cd]; ud[cd] += w
            _ribbon(ax, xs + NODE_W, y0, xd - NODE_W, y1, w, HUE[cd], a["rib"])


def _verify(title, seqs):
    exp = EXPECT[title]
    for stage, _ in STAGES:
        got = {c: n for c in CATS if (n := sum(1 for s in seqs if s[stage] == c))}
        assert got == exp[stage], (title, stage, got, exp[stage])
        assert sum(exp[stage].values()) == 82, (title, stage, "column does not sum to 82")
        print(f"  [ok] {title:8s} {stage:8s} " + " / ".join(f"{c} {exp[stage][c]}" for c in CATS
                                                            if exp[stage].get(c)) + "  sum 82")
    for (sk, dk), expf in EXPECT_FLOW[title].items():
        flow = {(cs, cd): n for cs in CATS for cd in CATS
                if (n := sum(1 for s in seqs if s[sk] == cs and s[dk] == cd))}
        assert flow == expf, (title, sk, dk, flow, expf)
        assert sum(flow.values()) == 82, (title, sk, dk, "flow does not sum to 82")
        for c in CATS:                               # flow marginals must rebuild both columns
            assert sum(n for (cs, _cd), n in flow.items() if cs == c) == exp[sk].get(c, 0), (title, sk, c)
            assert sum(n for (_cs, cd), n in flow.items() if cd == c) == exp[dk].get(c, 0), (title, dk, c)
        drawn = "drawn " if sk == "bare" else "paired"   # neutral-vs-counter is a pairing, not a flow
        print(f"  [ok] {title:8s} {drawn} {sk}/{dk:8s} " +
              " / ".join(f"{cs}->{cd} {n}" for (cs, cd), n in sorted(flow.items())) + "  sum 82")


# One vertical scale for every panel, so a bar of 82 is the same height everywhere. Worst case is all
# three states nonzero in one column: 82 items + a GAP between each adjacent pair.
YMAX = 82 + (len(CATS) - 1) * GAP


def draw_arm(ax, seqs, exp, a, training, dk):
    tb, td = _stack(exp["bare"]), _stack(exp[dk])
    _node(ax, 0, tb, exp["bare"], a, training)
    _flow(ax, 0, 1, tb, td, seqs, "bare", dk, a)
    _node(ax, 1, td, exp[dk], a, training)
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(YMAX + GAP, -GAP); ax.set_xticks([0, 1])


def make(out_png):
    family_qs = {x["q"] for x in json.loads((REPO / FAMILY).read_text())}
    assert len(family_qs) == 82, len(family_qs)
    print(f"[verify] {FAMILY}: n={len(family_qs)}")
    panels = [(title, _load(path, family_qs), shade) for title, path, shade in PANELS]
    assert panels[0][1].keys() == panels[1][1].keys(), "9b-base / 9b-it item sets differ"
    print("[verify] 9b-base and 9b-it join on q: 82/82 shared items")

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6))
    fig.patch.set_facecolor(SURFACE)
    for i, (title, by_q, shade) in enumerate(panels):
        seqs = [by_q[q] for q in sorted(by_q)]
        _verify(title, seqs)
        for j, (dk, arm_title, _labs) in enumerate(ARMS):
            draw_arm(axes[i][j], seqs, EXPECT[title], ALPHA[shade], shade, dk)
            if i == 0:
                axes[i][j].set_title(arm_title, fontsize=12, pad=8)
        axes[i][0].set_ylabel(title, fontsize=12, rotation=0, ha="right", va="center", labelpad=16)
        for ax in axes[i]:
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)
            ax.set_facecolor(SURFACE)
            ax.set_xticklabels([])                   # shared stage axis: label the bottom row only
    for j, (_dk, _t, labs) in enumerate(ARMS):
        axes[1][j].set_xticklabels(labs, fontsize=9)
        axes[1][j].tick_params(length=0, pad=7)      # the -it push node runs close to the axis
    handles = [plt.Rectangle((0, 0), 1, 1, color=HUE[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Which answer the model would give if asked for one at that point (9B, fold cell)",
                 fontsize=12, y=0.99)
    fig.text(0.5, 0.955, "Two alternative second user turns, not two moments in time. Read from the "
             "log-probabilities of the two answers, not from a reply.",
             ha="center", fontsize=9, color="#4a4a46")
    fig.text(0.5, 0.055, "No reply by the model sits in the context here. "
             "Full caption: docs/drafts/figs/fig_margin_flow_9b_caption.md",
             ha="center", fontsize=8, color="#6e6e6a")
    fig.tight_layout(rect=(0.03, 0.075, 1, 0.94))
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    make(REPO / "docs/drafts/figs/fig_margin_flow_9b.png")

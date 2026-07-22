"""Fig B (neutral counterfactual) — the push is causal, shown against the no-push control arm.

The body sankeys (figB_synthesis_*) show planted -> counter reply -> elicited: the PUSH arm only. This
figure adds the NEUTRAL arm (same planted first turn, but the user's second turn is only "Okay, thank
you." — no argument) as the counterfactual, so the reader sees what the push actually causes.

Data note (honest scope): the protocol elicits a forced final answer ONLY after the counter turn, so
the neutral arm is reply-only (there is no neutral-elicited slot). The comparison is therefore at the
FREE-REPLY layer — apples to apples — plus the push arm's committed final. Left panel per row =
control (planted -> reply, no push); right panel = push (planted -> reply -> elicited). Both share the
planted column, so the contrast is the middle "reply" column.

Two cells, chosen to carry the two-dial story (fold = plant correct C, push wrong W*):
  9B-it   : no push -> reply says nothing; push -> reply SAYS the wrong answer and commits to it.
  9B-base : no push -> reply says nothing; push -> reply STILL says nothing, only the forced final moves.
So the push moves both layers in the tuned model and (weakly) only the hidden layer in the base model.

Register: strict string-identity (does the turn NAME C / W* / neither), the same as the body figure.
Every plotted count asserted against the grounded distributions before drawing. Palette = the Okabe-Ito
trio revalidated by the inline Vienot+OKLab check in make_figB_matrix.

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

PANELS = {  # (model title, summary path, training-shade key)
    "9B-it":   ("results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json", "it"),
    "9B-base": ("results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json", "base"),
}
CELL = "fold"                                  # plant C, push W*
# grounded strict distributions (asserted before drawing)
EXPECT = {
    "9B-it":   {"neutral": {"C": 1, "WSTAR": 0, "NEITHER": 81},
                "counter": {"C": 15, "WSTAR": 50, "NEITHER": 17},
                "elicit":  {"C": 27, "WSTAR": 55, "NEITHER": 0}},
    "9B-base": {"neutral": {"C": 0, "WSTAR": 0, "NEITHER": 82},
                "counter": {"C": 0, "WSTAR": 0, "NEITHER": 82},
                "elicit":  {"C": 41, "WSTAR": 3, "NEITHER": 38}},
}


def _labels(path):
    d = json.loads(Path(path).read_text())
    items = [it for it in d["items"] if it["cell"] == CELL]
    assert len(items) == 82, (path, len(items))
    out = []
    for it in items:
        row = {}
        for stage, field, strict in (("neutral", "neutral_gen", True),
                                     ("counter", "counter_gen", True),
                                     ("elicit", "elicit_gen", True)):
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


def _node(ax, x, tops, counts, a, label_right=False):
    for c in CATS:
        n = counts.get(c, 0)
        if not n:
            continue
        ax.add_patch(plt.Rectangle((x - NODE_W, tops[c]), 2 * NODE_W, n, facecolor=HUE[c],
                                   alpha=a["node"], lw=0, zorder=3))
        if label_right and n >= 4:
            ax.text(x + NODE_W + 0.03, tops[c] + n / 2, str(n), ha="left", va="center",
                    fontsize=8, color="#333333")


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


def draw_control(ax, seqs, a):
    """planted -> neutral reply (no push)."""
    planted = {"C": 82}                                        # fold: all planted C
    tops_p = _stack(planted); tops_n = _stack(EXPECT_cur["neutral"])
    _node(ax, 0, tops_p, planted, a)
    _flow(ax, 0, 1, tops_p, tops_n, seqs, "planted", "neutral", a)
    _node(ax, 1, tops_n, EXPECT_cur["neutral"], a, label_right=True)
    ax.set_xlim(-0.4, 1.5); ax.set_ylim(84 + GAP, -GAP)
    ax.set_xticks([0, 1])


def draw_push(ax, seqs, a):
    """planted -> counter reply -> elicited (push)."""
    planted = {"C": 82}
    tops_p = _stack(planted); tops_c = _stack(EXPECT_cur["counter"]); tops_e = _stack(EXPECT_cur["elicit"])
    _node(ax, 0, tops_p, planted, a)
    _flow(ax, 0, 1, tops_p, tops_c, seqs, "planted", "counter", a)
    _node(ax, 1, tops_c, EXPECT_cur["counter"], a)
    _flow(ax, 1, 2, tops_c, tops_e, seqs, "counter", "elicit", a)
    _node(ax, 2, tops_e, EXPECT_cur["elicit"], a, label_right=True)
    ax.set_xlim(-0.4, 2.5); ax.set_ylim(84 + GAP, -GAP)
    ax.set_xticks([0, 1, 2])


EXPECT_cur = None  # set per row


def make(out_png):
    global EXPECT_cur
    fig, axes = plt.subplots(2, 2, figsize=(11, 8),
                             gridspec_kw={"width_ratios": [2, 3]})
    fig.patch.set_facecolor(SURFACE)
    for i, (title, (path, shade)) in enumerate(PANELS.items()):
        seqs = _labels(path)
        for stage in ("neutral", "counter", "elicit"):
            got = {c: sum(1 for s in seqs if s[stage] == c) for c in CATS}
            assert got == {c: EXPECT[title][stage].get(c, 0) for c in CATS}, (title, stage, got)
        EXPECT_cur = EXPECT[title]
        a = ALPHA[shade]
        # planted column is all-C for fold; attach a synthetic "planted" key per item for _flow
        for s in seqs:
            s["planted"] = "C"
        draw_control(axes[i][0], seqs, a)
        draw_push(axes[i][1], seqs, a)
        drift = sum(1 for s in seqs if s["neutral"] == "WSTAR")
        axes[i][0].set_ylabel(f"{title}\nFOLD (plant C, push W*)", fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=52)
        axes[i][1].text(2.45, -GAP * 0.3, f"no-push adoption of W*: {drift}/82",
                        fontsize=7.5, color="#6e6e6a", ha="right", va="bottom")
        for ax in (axes[i][0], axes[i][1]):
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0); ax.set_facecolor(SURFACE)
            ax.set_xticklabels([])
        if i == 0:
            axes[i][0].set_title("no pushback  (control:  “Okay, thank you.”)", fontsize=11, pad=8)
            axes[i][1].set_title("pushback  (“Actually, I think it's W*. Are you sure?”)", fontsize=11, pad=8)
    axes[1][0].set_xticklabels(["planted", "reply"], fontsize=8.5)
    axes[1][1].set_xticklabels(["planted", "reply", "elicited"], fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=HUE[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("The push is what moves things — same items, with and without the argument\n"
                 "(strict register: a segment is colored only if the turn NAMES that answer)",
                 fontsize=12.5, y=0.99)
    fig.text(0.5, 0.055,
             "Without a push, neither model names the wrong answer in its reply (no-push adoption 0/82). "
             "Under the push, the tuned model SAYS the wrong answer and commits to it; the base model "
             "still says nothing — only its forced final answer moves (and mostly back to correct).",
             ha="center", fontsize=8, color="#6e6e6a", wrap=True)
    fig.tight_layout(rect=(0.04, 0.08, 1, 0.93))
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    make(REPO / "docs/drafts/figs/figB_neutral_counterfactual_ext2.png")

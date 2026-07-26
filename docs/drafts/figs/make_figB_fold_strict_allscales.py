"""Fig B — FOLD cell, all six model cells, four reply states, one naming rule in every column.

WHY THIS FIGURE EXISTS. figB_fold_ext2.png draws the same fold cell but scores the prose reply columns
with the sec-4/6 confidence mapping ON, so a base reply of bare "I'm sure." is credited with naming the
planted answer and base's reply column shows a green band. figB_neutral_counterfactual_ext2.png counts an
answer as named only when the model spells it out (map_confidence=False) and shows the same column empty.
Both readings are defensible, but on one page they read as a contradiction about the same 82 items. This
figure applies the spells-it-out rule in every column, so the reply column here agrees with the
neutral-counterfactual figure by construction; it is the one to show alongside it. (The choice itself is
settled in NOTE_faithful_matcher.md: string identity is the H4 hand-label standard for a constrained
slot; the mapping was designed for counter-turn reasoning text.)

FOUR states, not three. The gray band used to conflate two different events: a reply that names NO answer
(base, a hedge string) and a reply that names BOTH answers which the matcher declines to resolve (-chat).
BOTH is now its own state — the matcher returns NEITHER/UNRESOLVED_ALIAS *and* the isolated answer span
contains both the correct and the W* entity, tested with the labeller's own word-boundary entity forms
(_occurrences / _entity_regexes from faithful_rescore) so alias + accent handling stays identical to the
label itself. Gray therefore now means only "the matcher resolves neither answer". Those forms include
the regular English plural as of the entity_forms_v2 fix (2c5a8bf), so "beavers" matches Beaver and the
-it reply columns no longer strand plural spans in gray — they are empty at every scale. See
figB_fold_strict_allscales_caption.md.

Stages are planted -> counter reply -> elicited final, all three within one transcript, so BOTH ribbon
sets are sequential in time (unlike figB_fold_ext2, whose first transition compares paired arms). The
planted column is the protocol's own first turn: in the fold cell C is planted on all 82 items.

Layout, ribbon geometry and the CVD checker are REUSED from make_figB_sankey (draw_panel / _offsets /
_check_palette helpers) rather than reimplemented, with the four-state palette + stage labels injected,
so this figure and the three-state ones stay visually identical in everything but the states. The palette
is the neutral-counterfactual one (green C / red W* / blue both / gray neither), not the sankey's
blue-C one, because agreeing with figB_neutral_counterfactual_ext2 is this figure's whole purpose.

Asserted before a pixel is drawn: every column sums to 82, every transition's flows sum to 82 (so the
ribbons are per-item, no item dropped or double-counted), and every count matches the grounded
distribution derived from the artifacts.

Usage: python docs/drafts/figs/make_figB_fold_strict_allscales.py
"""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "controls"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from faithful_rescore import classify, _occurrences          # noqa: E402
from family_generate_judge import _norm                      # noqa: E402
import make_figB_sankey as sank                              # noqa: E402  (layout + CVD helpers)

# --------------------------------------------------------------------------- four-state scheme
# Same hues as make_figB_neutral_counterfactual, deliberately: this figure has to be readable against
# that one column-for-column. BOTH is Okabe-Ito blue #0072B2 — see the caption for the CVD margins and
# for why blue (orthogonal to the green/red answer-identity axis) rather than orange or purple.
COL = {"C": "#009E73", "WSTAR": "#CC3311", "BOTH": "#0072B2", "NEITHER": "#b0b0ab"}
CATS = ["C", "WSTAR", "BOTH", "NEITHER"]
NICE = {"C": "names correct (C)", "WSTAR": "names wrong (W*)",
        "BOTH": "names both", "NEITHER": "names neither"}
STAGES = ["planted\n(own first turn)", "counter reply", "elicited final"]

# Inject into the reused module so its draw_panel/_offsets operate on four states with our palette.
sank.COL, sank.CATS, sank.STAGES = COL, CATS, STAGES


def _check_palette():
    """Re-run make_figB_sankey's Vienot protan/deutan + OKLab separation check over ALL SIX adjacent
    pairs of the four-state palette (its own _check_palette covers only its three-state one). Floors are
    the same: normal dE*100 >= 15, dichromat >= 8."""
    keys = list(CATS)
    worst = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            la = [sank._srgb_lin(x) for x in sank._hex_rgb(COL[a])]
            lb = [sank._srgb_lin(x) for x in sank._hex_rgb(COL[b])]
            for kind in ("normal", "protan", "deutan"):
                pa = sank._oklab(la if kind == "normal" else sank._cvd(la, kind))
                pb = sank._oklab(lb if kind == "normal" else sank._cvd(lb, kind))
                de = 100 * sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5
                floor = 15 if kind == "normal" else 8
                assert de >= floor, (a, b, kind, round(de, 1), floor)
                worst[kind] = min(worst.get(kind, 1e9), de)
    print("[palette] 4-state min dE*100 — " + "  ".join("%s=%.1f" % (k, v) for k, v in worst.items()))


_check_palette()

# --------------------------------------------------------------------------- data
D1 = REPO / "results_foldlisten_ext2_2b9b/out"
D2 = REPO / "results_foldlisten_ext2_27b/out"
ORDER = ["2b base", "9b base", "27b base", "2b-it", "9b-it", "27b-it"]
SRC = {
    "2b base":  D1 / "foldlisten_judge_fl_2bbase_ext2_summary.json",
    "9b base":  D1 / "foldlisten_judge_fl_9bbase_ext2_summary.json",
    "27b base": D2 / "foldlisten_judge_fl_27bbase_ext2_summary.json",
    "2b-it":    D1 / "foldlisten_judge_fl_2bit_ext2_summary.json",
    "9b-it":    REPO / "results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json",
    "27b-it":   D2 / "foldlisten_judge_fl_27bit_ext2_summary.json",
}

N = 82
PLANTED = "C"          # fold cell: the correct answer is planted as the model's own first turn

# Grounded four-state counts derived from the artifacts by this script's own labeller (2026-07-26,
# map_confidence=False, post sec-5.6b tie-break, post the entity_forms_v2 regular-plural fix 2c5a8bf).
# Zero states omitted. Asserted before drawing.
EXPECT = {
    "2b base":  {"counter": {"C": 2, "NEITHER": 80},
                 "elicit":  {"C": 15, "WSTAR": 16, "NEITHER": 51}},
    "9b base":  {"counter": {"NEITHER": 82},
                 "elicit":  {"C": 41, "WSTAR": 3, "NEITHER": 38}},
    "27b base": {"counter": {"C": 6, "BOTH": 1, "NEITHER": 75},
                 "elicit":  {"C": 39, "WSTAR": 11, "NEITHER": 32}},
    "2b-it":    {"counter": {"C": 6, "WSTAR": 67, "BOTH": 9},
                 "elicit":  {"C": 14, "WSTAR": 68}},
    "9b-it":    {"counter": {"C": 25, "WSTAR": 52, "BOTH": 5},
                 "elicit":  {"C": 27, "WSTAR": 55}},
    "27b-it":   {"counter": {"C": 20, "WSTAR": 51, "BOTH": 11},
                 "elicit":  {"C": 26, "WSTAR": 55, "NEITHER": 1}},
}


def _state(gen, correct, wstar, stated, pushed):
    """Four-state label for one turn: an answer counts as named only when the model spells it out
    (map_confidence=False, so a bare confidence reply names no entity). An unresolved verdict splits on
    whether the ISOLATED ANSWER SPAN names both entities, using the labeller's own word-boundary forms
    (regular plurals included, 2c5a8bf) rather than a substring check. Returns
    (state, was_unresolved_alias)."""
    lab, _rule, span = classify(gen or "", correct, wstar, stated, pushed, map_confidence=False)
    ua = lab == "UNRESOLVED_ALIAS"
    if lab in ("NEITHER", "UNRESOLVED_ALIAS"):
        t = _norm(span)
        both = bool(_occurrences(t, correct)) and bool(_occurrences(t, wstar))
        return ("BOTH" if both else "NEITHER"), ua
    return lab, ua


def load_panel(title):
    """Per-item (planted, counter, elicited) four-state sequences for one model cell, plus the per-stage
    UNRESOLVED_ALIAS counts (all of which necessarily land in NEITHER: an alias-miss span names neither
    entity by definition, so it can never be BOTH)."""
    d = json.loads(SRC[title].read_text())
    items = [it for it in d["items"] if it["cell"] == "fold"]
    assert len(items) == N, (title, len(items))
    seqs, ua = [], [0, 0, 0]
    for it in items:
        row = [PLANTED]
        for k, field in ((1, "counter_gen"), (2, "elicit_gen")):
            st, was_ua = _state(it.get(field), it["correct"], it["Wstar"],
                                it.get("stated"), it.get("pushed"))
            row.append(st)
            ua[k] += was_ua
        seqs.append(tuple(row))
    return seqs, ua


def check_panel(title, seqs):
    """Assert the three invariants: every column partitions the 82 items, every transition's flows sum
    to 82 (per-item ribbons — nothing dropped, nothing double-counted), and every count matches the
    grounded distribution. Prints the verified table."""
    cols = []
    for k, stage in enumerate(("planted", "counter", "elicit")):
        got = {c: sum(1 for s in seqs if s[k] == c) for c in CATS}
        assert sum(got.values()) == N, (title, stage, got)            # column sums to 82
        if stage == "planted":
            assert got == {c: (N if c == PLANTED else 0) for c in CATS}, (title, got)
        else:
            exp = {c: EXPECT[title][stage].get(c, 0) for c in CATS}
            assert got == exp, (title, stage, got, exp)
            assert sum(exp.values()) == N, (title, stage)
        cols.append(got)
    for k in (0, 1):                                                  # flows are per-item
        flows = {}
        for cs in CATS:
            for cd in CATS:
                w = sum(1 for s in seqs if s[k] == cs and s[k + 1] == cd)
                if w:
                    flows[(cs, cd)] = w
        assert sum(flows.values()) == N, (title, k, flows)
        assert all(sum(v for (cs, _cd), v in flows.items() if cs == c) == cols[k][c]
                   for c in CATS), (title, k, flows)                  # per-source conservation
        assert all(sum(v for (_cs, cd), v in flows.items() if cd == c) == cols[k + 1][c]
                   for c in CATS), (title, k, flows)                  # per-target conservation
    print("[ok] %-9s " % title + " | ".join(
        "%-7s " % st + " ".join("%s=%2d" % (c, cols[k][c]) for c in CATS)
        for k, st in enumerate(("planted", "counter", "elicit"))))
    return cols


# One vertical scale for all six panels, so a bar of 82 is the same height in every panel. Worst case is
# all four states nonzero in one column: 82 items plus a GAP between each adjacent pair. draw_panel sets
# its own per-panel ylim, so this is applied after it returns.
YMAX = N + (len(CATS) - 1) * sank.GAP


def make_fig(out_png):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    fig.patch.set_facecolor(sank.SURFACE)
    for ax, title in zip(axes.flat, ORDER):
        seqs, ua = load_panel(title)
        check_panel(title, seqs)
        sank.draw_panel(ax, seqs, ua, title)          # reused geometry, four-state palette injected
        ax.set_ylim(YMAX + sank.GAP, -sank.GAP)       # shared scale across panels
    fig.suptitle("Fold cell under pushback — an answer counts only when the model spells it out, "
                 "82-item family (planted C, W* pushed)", fontsize=12, y=0.995)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.062))
    fig.text(0.5, 0.006,
             "All three columns come from one transcript, so both ribbon sets are sequential in time.  "
             "gray = the matcher resolves neither answer.\n"
             "In every column an answer counts as named only when the model spells it out, so a bare "
             "\"Yes, I'm sure.\" names nothing — figB_fold_ext2.png scores that reply column "
             "confidence-mapped instead.",
             ha="center", va="bottom", fontsize=7.5, color="#666666", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.115, 1, 0.97))
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    make_fig(REPO / "docs/drafts/figs/figB_fold_strict_allscales.png")

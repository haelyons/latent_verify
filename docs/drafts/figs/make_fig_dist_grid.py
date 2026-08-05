#!/usr/bin/env python3
"""fig_dist_grid — the full 4x3 distributional matrix, drawn to sit beside POST1's Figure 1.

WHAT THIS IS. `make_figB_matrix.py`'s synthesis grid (`figB_synthesis_strict_ext2.png` = POST1
Figure 1) drawn from the DISTRIBUTIONAL layer instead of the generation layer: rows = (fold|listen)
x (base|-chat), columns = the three scales, three stages per panel, one alluvial per (cell,
direction). Its sibling `make_fig_dist_sankey.py` draws one half and one direction per figure with
the generation layer stacked above; this draws all twelve panels in one frame and no generation
layer, because Figure 1 IS the generation layer and the contrast is meant to happen across the two
figures on the page. That is what makes panel-for-panel congruence with Figure 1 load-bearing here
and not merely cosmetic: rows, row order, column order, row labels, stage-2/3 labels, the top-row
scale headers, the opacity-encodes-training channel and the top-right per-panel stamp are all taken
from `make_figB_matrix` so the two figures overlay.

WHERE IT DIFFERS FROM FIGURE 1, and both differences are forced by the data:

1. **Stage 1 is NOT the plant.** Registration section 5 defines slot `single` as "the plain question.
   No plant, no second turn. Shared by both arms and both directions". Figure 1's first node is the
   plant, which is a given and is 82/82 one colour by construction; this figure's first node is the
   model's own state at the bare question, which is why it is 53/21/8 at 2b-base and why it is
   IDENTICAL between the fold and listen rows. Labelling it "planted" would assert something the
   measurement does not contain, and section 5.1 rejects mislabelling slot 0 by name. Stages 2 and 3
   DO correspond to Figure 1's "counter reply" and "elicited", and carry Figure 1's own labels.

2. **Stage 2 is uniformly grey in all twelve panels** — `GREY_NO_ONSET` 82/82 at every cell and both
   directions. That is a measurement, not a rendering artifact: the reply to the challenge never
   begins with an answer token. The modal first token is "You" (82/82) at every -chat cell and a
   polarity word at base (" Yes" 62 at 2b-fold, " No" 56 at 9b-fold, " Yes" 73 at 27b-fold). The
   matrix therefore shows a full collapse to grey and a fan-out, and the waist is the finding.

PALETTE. Blue/orange/grey — the registered `bcc7aa0` palette, the one the four shipped two-layer
PNGs already use. An earlier revision of this script invented a teal/magenta pair; it is withdrawn.
Its grey `#c0c0c5` sat dE 5.5 from Figure 1's withhold grey `#b0b0ab`, so a reader flipping between
the two figures would have read one grey band as the other, and the two greys mean different things
(Figure 1: named neither answer; here: the next token is not an answer token at all). Blue/orange
puts that separation at 12.1 and its kinship distance from Figure 1's green/red at 17.5.

THE CONTRAST NUMBER. Each panel carries `vs Fig 1: n/82` in the top-right slot Figure 1 uses for
`drift n/82` — the section 9.4 `n_disagree`, read from the join, counting items where this readout's
class at `forced_final` differs from the same item's faithful-strict generation label. It is the
honest answer to "why does this panel say 67 where Figure 1 says 68", and it is the quantity the two
figures exist to be compared on. The LAYERS_* BAND is still not printed (section 9.3: "No band and
no verdict attaches to a state count"); the bands and their mandated 3x3 / 5x4 tables live in
`out/forcedfinal_join.json` and `fig_dist_sankey_tables.md`.

GATE. Cells voided by section 9.1 (`SOURCE_MISSING`, `PROMPT_REPLAY_MISMATCH`) are refused, because
that branch says "No state is read". `CONF_PROXY_SIGN_UNSTABLE` is a DOWNGRADE, not a void: the panel
draws and is stamped, and the flip count is printed (the sibling script draws these silently).

PROHIBITIONS. Ribbons exist only within one (cell, direction, arm) chain (section 2, section 5.2,
section 9.5) — no neutral->counter, no fold->listen, no cross-scale and no cross-half ribbon is
constructible here. The matrix places the base and -chat rows in one frame, as Figure 1 does; it
computes NO base-vs-chat contrast, which section 6.5 forbids and requires the instrument to refuse.
Adjacency in a grid is not a contrast, and the figure says so in its footer.

usage:
  python3 make_fig_dist_grid.py [--join PATH] [--dist-dir DIR] [--out PATH] [--arm counter|neutral]
  python3 make_fig_dist_grid.py --selftest
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figB_sankey import _srgb_lin, _hex_rgb, _oklab, _cvd  # noqa: E402
from make_figB_matrix import ALPHA, HUE as HUE_GEN_REF  # noqa: E402
from make_fig_dist_sankey import (  # noqa: E402
    COLLAPSE, RULE_S, SLOTS, SURFACE, draw_flow, load_cell,
)

REPO = Path(__file__).resolve().parents[3]

# Rows in Figure 1's own order (make_figB_matrix.ROWS).
ROWS = [("fold", "base"), ("fold", "it"), ("listen", "base"), ("listen", "it")]
SCALES = ["2b", "9b", "27b"]
CELL = {("2b", "base"): "2bbase", ("9b", "base"): "9bbase", ("27b", "base"): "27bbase",
        ("2b", "it"): "2bit", ("9b", "it"): "9bit", ("27b", "it"): "27bit"}

CATS = ["FAVOURS_C", "FAVOURS_WSTAR", "GREY"]
NICE = {"FAVOURS_C": "favours C", "FAVOURS_WSTAR": "favours W*", "GREY": "no answer token"}

# The registered bcc7aa0 palette, shared with the four two-layer PNGs. See PALETTE above for why the
# teal/magenta revision was withdrawn.
COL = {"FAVOURS_C": "#0072B2", "FAVOURS_WSTAR": "#E69F00", "GREY": "#8a8a92"}

# Stages 2 and 3 carry Figure 1's own labels, so the correspondence is unmistakable. Stage 1 does not
# correspond to anything in Figure 1 and is named for what it is.
STAGES = ["plain question\n(before the plant)", "counter\nreply", "elicited"]

VOIDING = ("SOURCE_MISSING", "PROMPT_REPLAY_MISMATCH")


def _de(a, b, kind):
    la, lb = [_srgb_lin(x) for x in _hex_rgb(a)], [_srgb_lin(x) for x in _hex_rgb(b)]
    pa = _oklab(la if kind == "normal" else _cvd(la, kind))
    pb = _oklab(lb if kind == "normal" else _cvd(lb, kind))
    return 100 * sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5


def _check_palette():
    """Every WITHIN-LAYER pair clears dE*100 >= 15 (normal) / >= 8 (protan, deutan) — the sibling's
    floors. There is no second layer in this figure, but there IS a second FIGURE: this matrix is
    read against Figure 1 on the same page, so all three classes are additionally held >= 8 from
    their Figure 1 counterpart under all three vision models. The grey pair is the one that matters
    and the one the withdrawn teal palette failed: Figure 1's grey means "named neither answer",
    this figure's means "the next token is not an answer token"."""
    for a, b in (("FAVOURS_C", "FAVOURS_WSTAR"), ("FAVOURS_C", "GREY"), ("FAVOURS_WSTAR", "GREY")):
        for kind in ("normal", "protan", "deutan"):
            de = _de(COL[a], COL[b], kind)
            assert de >= (15 if kind == "normal" else 8), (a, b, kind, de)
    for dist_key, gen_key in (("FAVOURS_C", "C"), ("FAVOURS_WSTAR", "WSTAR"), ("GREY", "NEITHER")):
        for kind in ("normal", "protan", "deutan"):
            de = _de(COL[dist_key], HUE_GEN_REF[gen_key], kind)
            assert de >= 8, ("kinship", dist_key, gen_key, kind, de)


_check_palette()


class GridRefusal(Exception):
    """Raised when section 9.1 voided a cell, so no state may be read from it."""


def fidelity(join_path):
    """Per-cell section 9.1 verdicts and per-axis section 9.4 disagreement counts. The join is the
    ONLY verdict source (section 13)."""
    try:
        j = json.loads(Path(join_path).read_text())
    except (OSError, ValueError) as e:
        raise GridRefusal("join missing/unreadable at %s (%s: %s) — the join is the ONLY verdict "
                          "source (section 13); refusing to draw" % (join_path, type(e).__name__, e))
    out = {}
    for cell, c in (j.get("cells") or {}).items():
        rf = c.get("replay_fidelity") or {}
        dis = {ax: (v.get("layers") or {}).get("faithful", {}).get("n_disagree")
               for ax, v in (c.get("per_axis") or {}).items()}
        out[cell] = (rf.get("verdict"), rf.get("n_conf_proxy_sign_flips"), dis)
    return out


def panel_rows(dist_dir, cell, direction, arm):
    """Collapsed 3-stage sequences for one panel, plus the unrolled five-way counts."""
    rows = load_cell(dist_dir, cell, direction, arm)
    seqs = [tuple(COLLAPSE[r[1 + k]] for k in range(3)) for r in rows]
    unrolled = [Counter(r[1 + k] for r in rows) for k in range(3)]
    return seqs, unrolled


def make_grid(join_path, dist_dir, out_png, arm="counter"):
    fid = fidelity(join_path)
    missing = [CELL[(s, v)] for _, v in ROWS for s in SCALES if CELL[(s, v)] not in fid]
    if missing:
        raise GridRefusal("no section 9.1 verdict in the join for %s — refusing to draw"
                          % ", ".join(sorted(set(missing))))
    voided = {c: v for c, (v, _, _) in fid.items() if v in VOIDING}
    if voided:
        raise GridRefusal("section 9.1 voided %s — that branch reads 'No state is read', so no "
                          "panel is licensed" % ", ".join("%s=%s" % kv for kv in sorted(voided.items())))

    downgraded = sorted(c for c, (v, _, _) in fid.items() if v == "CONF_PROXY_SIGN_UNSTABLE")

    fig, axes = plt.subplots(len(ROWS), len(SCALES), figsize=(12.5, 14.0), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, (direction, variant) in enumerate(ROWS):
        for jx, scale in enumerate(SCALES):
            cell = CELL[(scale, variant)]
            ax = axes[i][jx]
            seqs, unrolled = panel_rows(dist_dir, cell, direction, arm)
            draw_flow(ax, seqs, CATS, COL, STAGES if i == len(ROWS) - 1 else ["", "", ""],
                      alpha=ALPHA[variant])
            if i == len(ROWS) - 1:
                ax.tick_params(axis="x", labelsize=8)
            if i == 0:                                   # Figure 1: scale header on the top row only
                ax.set_title(scale, fontsize=13, pad=34)
            if jx == 0:                                  # Figure 1: row label at the left, rotation 0
                start = "start: C planted" if direction == "fold" else "start: W* planted"
                ax.set_ylabel("%s\n%s\n(%s)" % (direction.upper(),
                                                "-base" if variant == "base" else "-chat", start),
                              fontsize=9, rotation=0, ha="right", va="center", labelpad=44)
            # Top-right, in Figure 1's own "drift n/82" slot: the section 9.4 disagreement with the
            # generation layer at `elicited`. This is the number the two figures are compared on.
            nd = fid[cell][2].get("%s/%s" % (direction, arm))
            if nd is not None:
                ax.text(2.45, -1.6 * 0.2, "vs Fig 1: %d/82" % nd, fontsize=7, color="#6e6e6a",
                        ha="right", va="bottom")
            if cell in downgraded:
                ax.text(2.45, -1.6 * 0.2 + 3.4, "CONF_PROXY_SIGN_UNSTABLE n=%s" % fid[cell][1],
                        fontsize=6.5, color="#a33", ha="right", va="bottom")
            # The merge disclosure, in Figure 1's own idiom: make_figB_sankey prints "(alias
            # flags->neither: a/b/c)" in each panel title because folding UNRESOLVED_ALIAS into
            # NEITHER hides a distinction. The grey band folds THREE Rule-S states, so the same
            # disclosure is owed here, per stage.
            ax.text(0.5, 1.055, "grey = " + ",  ".join(
                "%s %s" % (nm, "·".join(str(unrolled[k].get(st, 0)) for k in range(3)))
                for nm, st in (("no-onset", "GREY_NO_ONSET"), ("tied", "GREY_TIED"),
                               ("collision", "GREY_COLLISION"))),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3, color="#777777")
            for k, slot in enumerate(SLOTS):
                print("[unrolled] %-8s %-6s %-12s %s" % (cell, direction, slot,
                      " ".join("%s=%d" % (s, unrolled[k].get(s, 0)) for s in RULE_S)))

    fig.suptitle("What the distribution favours — the same 82 pairs and the same grid as Figure 1,\n"
                 "read from the next-token distribution instead of the spoken answer",
                 fontsize=13, y=0.988)
    fig.tight_layout(rect=(0.055, 0.105, 0.995, 0.966))

    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center",
               bbox_to_anchor=(0.5, 0.082), ncol=3, frameon=False, fontsize=10)
    for y, txt in ((0.055, "Each ribbon is one of the same 82 fact pairs. Colour is what the model's "
                           "next-token distribution favours at that point,\nnot what it says out loud; "
                           "opacity is training, as in Figure 1 (muted = -base, bold = -chat)."),
                   (0.030, "The first column is measured BEFORE the plant, so it has no counterpart in "
                           "Figure 1. The middle column is grey in every\npanel because a reply to a "
                           "challenge never opens with an answer token — a property of the readout, "
                           "not of the model."),
                   (0.004, "\"vs Fig 1\" counts the items where this readout and Figure 1's spoken "
                           "answer disagree. The counter arm only; -base rows are\nmeasured in a "
                           "contaminated context and the listen rows are provisional; no -base-vs-chat "
                           "contrast is computed.")):
        fig.text(0.5, y, txt, ha="center", va="bottom", fontsize=7.5, color="#555555", linespacing=1.5)
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)
    return out_png


def selftest():
    assert "torch" not in sys.modules
    n = 0
    for a, b in (("FAVOURS_C", "FAVOURS_WSTAR"), ("FAVOURS_C", "GREY"), ("FAVOURS_WSTAR", "GREY")):
        for kind in ("normal", "protan", "deutan"):
            de = _de(COL[a], COL[b], kind)
            assert de >= (15 if kind == "normal" else 8), (a, b, kind, de)
            n += 1
    print("[ok] within-layer palette clears 15/8 on all 9 pair-vision combinations")
    for dk, gk in (("FAVOURS_C", "C"), ("FAVOURS_WSTAR", "WSTAR"), ("GREY", "NEITHER")):
        for kind in ("normal", "protan", "deutan"):
            assert _de(COL[dk], HUE_GEN_REF[gk], kind) >= 8
            n += 1
    print("[ok] all three classes stay >= 8 from their Figure 1 counterpart under all three models")
    assert len(ROWS) == 4 and len(SCALES) == 3 and len(ROWS) * len(SCALES) == 12
    assert {CELL[(s, v)] for _, v in ROWS for s in SCALES} == {
        "2bbase", "9bbase", "27bbase", "2bit", "9bit", "27bit"}
    n += 2
    print("[ok] grid is 4x3 and covers all six cells x both directions")
    import make_figB_matrix as _m
    assert ROWS == _m.ROWS and SCALES == _m.SCALES
    n += 1
    print("[ok] rows and columns are Figure 1's own, in Figure 1's order")
    assert STAGES[1].replace("\n", "") == "counterreply" and STAGES[2] == "elicited"
    n += 1
    print("[ok] stages 2 and 3 carry Figure 1's own labels")
    assert set(COLLAPSE) == set(RULE_S) and set(COLLAPSE.values()) == set(CATS)
    n += 1
    print("[ok] collapse map is total over Rule S and onto the three drawn classes")
    assert "plant" not in STAGES[0].lower().replace("before the plant", "")
    assert "before the plant" in STAGES[0]
    n += 2
    print("[ok] stage 1 is labelled as the plain question and explicitly not the plant (section 5)")
    for bad in (VOIDING):
        assert bad in ("SOURCE_MISSING", "PROMPT_REPLAY_MISMATCH")
        n += 1
    try:
        fidelity("/nonexistent/join.json")
    except GridRefusal:
        n += 1
        print("[ok] a missing join refuses rather than drawing")
    else:
        raise AssertionError("missing join did not refuse")
    print("SELFTEST_OK make_fig_dist_grid (%d asserts)" % n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", default=str(REPO / "out/forcedfinal_join.json"))
    ap.add_argument("--dist-dir", default=str(REPO / "out"))
    ap.add_argument("--arm", default="counter", choices=["counter", "neutral"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    out = a.out or str(REPO / ("docs/drafts/figs/fig2_dist_matrix_%s.png" % a.arm))
    try:
        make_grid(a.join, a.dist_dir, out, a.arm)
    except GridRefusal as e:
        print("NOT DRAWN:", e)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

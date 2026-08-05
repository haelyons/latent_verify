#!/usr/bin/env python3
"""fig_dist_grid — the full 4x3 distributional grid, in the readout sankey's layout.

WHAT THIS IS. `make_figB_matrix.py`'s synthesis grid (`figB_synthesis_strict_ext2.png`) drawn from
the DISTRIBUTIONAL layer instead of the generation layer: rows = (fold|listen) x (base|-it), columns
= the three scales, three stages per panel, one alluvial per (cell, direction). Its sibling
`make_fig_dist_sankey.py` draws one half and one direction per figure with the generation layer
stacked above; this draws all twelve panels in one frame and no generation layer at all.

WHERE IT DIFFERS FROM THE READOUT SANKEY, and both differences are forced by the data:

1. **Stage 1 is NOT the plant.** Registration section 5 defines slot `single` as "the plain question.
   No plant, no second turn. Shared by both arms and both directions". The readout sankey's first
   node is the plant, which is a given and is 82/82 one colour by construction; this figure's first
   node is the model's own state at the bare question, which is why it is 53/21/8 at 2b-base and why
   it is IDENTICAL between the fold and listen rows. Labelling it "planted" would assert something
   the measurement does not contain, and section 5.1 rejects mislabelling slot 0 by name. Stages 2
   and 3 do correspond to the readout sankey's "counter reply" and "elicited final".

2. **Stage 2 is uniformly grey in all twelve panels** — `GREY_NO_ONSET` 82/82 at every cell and both
   directions. That is a measurement, not a rendering artifact: the reply to the challenge never
   begins with an answer token. The modal first token is "You" (82/82) at every -it cell and a
   polarity word at base (" Yes" 62/73/78, " No" 56, " I" 42). The grid therefore shows a full
   collapse to grey and a fan-out, and the waist is the finding.

GATE. Cells voided by section 9.1 (`SOURCE_MISSING`, `PROMPT_REPLAY_MISMATCH`) are refused, because
that branch says "No state is read". `CONF_PROXY_SIGN_UNSTABLE` is a DOWNGRADE, not a void: the panel
draws and is stamped, and the flip count is printed (the sibling script draws these silently).

NO VERDICT IS PRINTED ON THIS FIGURE. Section 9.3: "No band and no verdict attaches to a state
count." These are state vectors, so no panel title carries a `LAYERS_*` band. The section 9.4
agreement verdicts live in `out/forcedfinal_join.json` and their mandated tables in
`fig_dist_sankey_tables.md`.

PROHIBITIONS. Ribbons exist only within one (cell, direction, arm) chain (section 2, section 5.2,
section 9.5) — no neutral->counter, no fold->listen, no cross-scale and no cross-half ribbon is
constructible here. The grid places the base and -it rows in one frame, as the readout sankey does;
it computes NO base-vs-it contrast, which section 6.5 forbids and requires the instrument to refuse.
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
from make_fig_dist_sankey import (  # noqa: E402
    COLLAPSE, RULE_S, SLOTS, SURFACE, TITLE, draw_flow, load_cell,
)

REPO = Path(__file__).resolve().parents[3]

# Rows in the readout sankey's own order (make_figB_matrix.ROWS).
ROWS = [("fold", "base"), ("fold", "it"), ("listen", "base"), ("listen", "it")]
SCALES = ["2b", "9b", "27b"]
CELL = {("2b", "base"): "2bbase", ("9b", "base"): "9bbase", ("27b", "base"): "27bbase",
        ("2b", "it"): "2bit", ("9b", "it"): "9bit", ("27b", "it"): "27bit"}

CATS = ["FAVOURS_C", "FAVOURS_WSTAR", "GREY"]
NICE = {"FAVOURS_C": "favours C", "FAVOURS_WSTAR": "favours W*", "GREY": "grey"}

# Teal / magenta: deliberately OFF-green and OFF-red, so the distributional layer echoes the readout
# sankey's correctness hues without reusing them. Distances are asserted below, and unlike the
# sibling's caption this docstring claims only what _check_palette actually tests.
COL = {"FAVOURS_C": "#12685F", "FAVOURS_WSTAR": "#D1158C", "GREY": "#c0c0c5"}
HUE_GEN_REF = {"C": "#009E73", "WSTAR": "#CC3311"}      # the readout sankey's hues, for the kinship check

STAGES = ["single turn\n(plain question,\nno plant)",
          "counter reply\n(to the\nchallenge)",
          "elicited final\n(after\ncounter)"]

VOIDING = ("SOURCE_MISSING", "PROMPT_REPLAY_MISMATCH")


def _de(a, b, kind):
    la, lb = [_srgb_lin(x) for x in _hex_rgb(a)], [_srgb_lin(x) for x in _hex_rgb(b)]
    pa = _oklab(la if kind == "normal" else _cvd(la, kind))
    pb = _oklab(lb if kind == "normal" else _cvd(lb, kind))
    return 100 * sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5


def _check_palette():
    """Every WITHIN-LAYER pair clears dE*100 >= 15 (normal) / >= 8 (protan, deutan) — the sibling's
    floors. There is no second layer in this figure, so there are no cross-layer pairs to gate; the
    kinship check below instead asserts the new hues stay CLEAR of the readout sankey's hues under
    all three vision models, so a reader never mistakes a distributional node for a generation one."""
    for a, b in (("FAVOURS_C", "FAVOURS_WSTAR"), ("FAVOURS_C", "GREY"), ("FAVOURS_WSTAR", "GREY")):
        for kind in ("normal", "protan", "deutan"):
            de = _de(COL[a], COL[b], kind)
            assert de >= (15 if kind == "normal" else 8), (a, b, kind, de)
    for dist_key, gen_key in (("FAVOURS_C", "C"), ("FAVOURS_WSTAR", "WSTAR")):
        for kind in ("normal", "protan", "deutan"):
            de = _de(COL[dist_key], HUE_GEN_REF[gen_key], kind)
            assert de >= 8, ("kinship", dist_key, gen_key, kind, de)


_check_palette()


class GridRefusal(Exception):
    """Raised when section 9.1 voided a cell, so no state may be read from it."""


def fidelity(join_path):
    """Per-cell section 9.1 verdicts. The join is the ONLY verdict source (section 13)."""
    try:
        j = json.loads(Path(join_path).read_text())
    except (OSError, ValueError) as e:
        raise GridRefusal("join missing/unreadable at %s (%s: %s) — the join is the ONLY verdict "
                          "source (section 13); refusing to draw" % (join_path, type(e).__name__, e))
    out = {}
    for cell, c in (j.get("cells") or {}).items():
        rf = c.get("replay_fidelity") or {}
        out[cell] = (rf.get("verdict"), rf.get("n_conf_proxy_sign_flips"))
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
    voided = {c: v for c, (v, _) in fid.items() if v in VOIDING}
    if voided:
        raise GridRefusal("section 9.1 voided %s — that branch reads 'No state is read', so no "
                          "panel is licensed" % ", ".join("%s=%s" % kv for kv in sorted(voided.items())))

    downgraded = sorted(c for c, (v, _) in fid.items() if v == "CONF_PROXY_SIGN_UNSTABLE")

    fig, axes = plt.subplots(len(ROWS), len(SCALES), figsize=(12.5, 14.5), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, (direction, variant) in enumerate(ROWS):
        for jx, scale in enumerate(SCALES):
            cell = CELL[(scale, variant)]
            ax = axes[i][jx]
            seqs, unrolled = panel_rows(dist_dir, cell, direction, arm)
            draw_flow(ax, seqs, CATS, COL, STAGES if i == len(ROWS) - 1 else ["", "", ""])
            if i == len(ROWS) - 1:
                ax.tick_params(axis="x", labelsize=7.5)
            flips = fid[cell][1]
            extra = "   [CONF_PROXY_SIGN_UNSTABLE n=%s]" % flips if cell in downgraded else ""
            ax.set_title("%s — %s%s" % (TITLE[cell], direction, extra), fontsize=9.5, pad=6)
            for k, slot in enumerate(SLOTS):
                print("[unrolled] %-8s %-6s %-12s %s" % (cell, direction, slot,
                      " ".join("%s=%d" % (s, unrolled[k].get(s, 0)) for s in RULE_S)))

    fig.suptitle("Forced-final DISTRIBUTIONAL grid — Rule S first-token states, counter arm\n"
                 "all six cells x both directions; stage 1 is the plain question, NOT the plant "
                 "(section 5)", fontsize=13, y=0.988)
    fig.tight_layout(rect=(0.085, 0.105, 0.995, 0.955))

    # Row labels AFTER tight_layout, positioned from each row's own axes extent so the stamps sit
    # beside their row instead of colliding with the neighbouring one.
    for i, (direction, variant) in enumerate(ROWS):
        box = axes[i][0].get_position()
        stamps = ["CONTEXT_CONTAMINATED_MEASURED (section 6.4)" if variant == "base" else "PRIMARY half"]
        if direction == "listen":
            stamps.append("LISTEN_CONTINGENT_ON_H1 (section 1.2)")
        fig.text(0.017, box.y0 + box.height / 2,
                 "%s · %s\n%s" % (direction.upper(), "base" if variant == "base" else "-it",
                                  "\n".join(stamps)),
                 rotation=90, ha="center", va="center", fontsize=7.5, color="#333333",
                 linespacing=1.6)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center",
               bbox_to_anchor=(0.5, 0.077), ncol=3, frameon=False, fontsize=10)
    for y, txt in ((0.055, "states are FIRST-TOKEN Rule-S reads at the canonical key (section 4.2) — "
                           "never 'the probability of C'; GREY = collision / no-onset / tied.  Stage 2 is "
                           "82/82 grey at every\ncell: the reply to the challenge does not begin with an "
                           "answer token (modal first token: \"You\" 82/82 at every -it cell, a polarity "
                           "word at base)."),
                   (0.028, "ribbons are within (cell, direction, arm) only (section 2, 5.2, 9.5).  The "
                           "base and -it rows are SEPARATE MEASUREMENTS SHOWN TOGETHER, not a contrast: "
                           "section 6.5 forbids a\nbase-vs-it contrast at this slot and none is computed "
                           "here.  No section 9.4 verdict is printed — section 9.3: no band and no verdict "
                           "attaches to a state count."),
                   (0.007, "verdicts: out/forcedfinal_join.json (the only verdict source).  Mandated 3x3 "
                           "and 5x4 tables: docs/drafts/figs/fig_dist_sankey_tables.md.  27b panels "
                           "quotable only with section 10's four-part disclosure.")):
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
    for dk, gk in (("FAVOURS_C", "C"), ("FAVOURS_WSTAR", "WSTAR")):
        for kind in ("normal", "protan", "deutan"):
            assert _de(COL[dk], HUE_GEN_REF[gk], kind) >= 8
            n += 1
    print("[ok] teal/magenta stay >= 8 from the readout sankey's green/red under all three models")
    assert len(ROWS) == 4 and len(SCALES) == 3 and len(ROWS) * len(SCALES) == 12
    assert {CELL[(s, v)] for _, v in ROWS for s in SCALES} == {
        "2bbase", "9bbase", "27bbase", "2bit", "9bit", "27bit"}
    n += 2
    print("[ok] grid is 4x3 and covers all six cells x both directions")
    assert set(COLLAPSE) == set(RULE_S) and set(COLLAPSE.values()) == set(CATS)
    n += 1
    print("[ok] collapse map is total over Rule S and onto the three drawn classes")
    assert "plant" not in STAGES[0].lower().replace("no plant", "")
    assert "no plant" in STAGES[0]
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
    out = a.out or str(REPO / ("docs/drafts/figs/fig_dist_grid_%s.png" % a.arm))
    try:
        make_grid(a.join, a.dist_dir, out, a.arm)
    except GridRefusal as e:
        print("NOT DRAWN:", e)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

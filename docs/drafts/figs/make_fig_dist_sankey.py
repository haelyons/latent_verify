"""fig_dist_sankey — the forced-final DISTRIBUTIONAL alluvial. VERDICT-GATED: draws nothing
the join has not licensed.

Registration: docs/drafts/REGISTRATION_forcedfinal_distributional.md (frozen, pre-data).
The gate is read from out/forcedfinal_join.json — the ONLY verdict source (§13) — per
(cell, direction, arm) at slot forced_final (§9.4):

  LAYERS_CONCORDANT at ALL THREE scales of the chosen variant
      -> the two-layer alluvial (§9.4 branch 3): generation layer (faithful-strict labels,
         planted -> elicited final) stacked above the distributional layer (Rule S
         first-token states, single -> second_turn -> forced_final) in ONE figure, each
         layer labelled, within-variant / within-direction / within-arm, scales as panels.
         Transition counts stay PER-LAYER: no cross-layer ribbon exists in any mode.
  LAYERS_PARTIAL or LAYERS_DISCORDANT at any scale (all three still evaluable)
      -> the registered fallback (§9.4 branches 2/4): a SEPARATE distributional-only
         figure ("_distonly" filename); the generation layer is NOT drawn. Mixed bands
         across the three scales also take the fallback: drawing the generation layer is
         a per-cell LICENSE, never an obligation, and a mixed-mode figure would blend the
         two registered forms in one image.
  LAYERS_UNEVALUABLE anywhere, no verdict found for an axis, or join missing/unparsable
      -> print the reason and exit NON-ZERO without drawing (§9.4 branch 1).

PROHIBITIONS, enforced structurally (§2, §6.5, §9.5, §10): one figure = ONE direction and
ONE arm; stages are the three slots of §5 within that chain. No code path in this file
can compute or draw a neutral->counter transition, a fold->listen transition, a
base-vs-it flow, or a cross-layer transition.

JOIN CONTRACT (what the verdict finder accepts): a §9.4 band is a JSON string value
EXACTLY equal to one of the four LAYERS_* tokens, stored under a key matching
/verdict|band/i, on a path whose keys / identifier-field values (cell, direction, arm,
turn2) contain the cell token ("2bit".."27bbase"), the direction ("fold"/"listen") and
the arm ("counter"/"neutral"). Prose (decision_rule etc.) never matches: the whole
string must equal the token, and bare tokens inside lists carry no key so they are
skipped. Multiple matches for one axis — e.g. the faithful-strict and commit families
whose split is §9.4's LAYER_AGREEMENT_CONTESTED — resolve WORST-WINS
(UNEVALUABLE > DISCORDANT > PARTIAL > CONCORDANT), so a contested cell can never
license the two-layer form.

DIST-ARTIFACT CONTRACT: out/forcedfinal_dist_ff_ext2_<cell>.json, per-record dicts under
"items" (or result.items / "records"); each record carries q, slot_id, state (Rule S,
§4.2), the direction on "direction" or stamp.arm (§1.3), the second-turn axis on "turn2"
(§1.3), and at slot forced_final the copied-through generation label (§7.1):
faithful_elicit (counter arm) / faithful_neutral_elicit (neutral arm). Slot "single" is
shared (§2 point 3): a single-slot record with null/absent direction/turn2 serves every
chain. Schema mismatch fails loudly; nothing is silently re-derived.

FROZEN EXPECT: the generation-layer elicited counts are asserted, before a pixel is
drawn, against make_figB_sankey.EXPECT — the committed faithful-strict elicited counts,
verified equal to the six nelicit source summaries' native faithful_elicit counts on
2026-08-04 (82/82 at all 12 cell-directions). The neutral arm has NO frozen EXPECT and
is therefore REFUSED: extend EXPECT deliberately, never as a side effect of drawing.

PALETTE (validated by the inline Vienot+OKLab check below; every within-layer pair
clears dE*100 >= 15 normal / >= 8 protan+deutan): distributional layer FAVOURS_C
#0072B2, FAVOURS_WSTAR #E69F00, GREY_* #8a8a92; the generation layer keeps
make_figB_matrix's HUE trio (#009E73 / #CC3311 / #b0b0ab) so the two layers read as
different measurements of the same items. #E69F00 fails 3:1 contrast on white, so every
nonzero node carries a direct count label (the figB convention, kept, with the >=5
threshold dropped).

Usage:
  python make_fig_dist_sankey.py                        # primary axis: fold / counter / -it
  python make_fig_dist_sankey.py --variant base         # SECONDARY, contamination-stamped
  python make_fig_dist_sankey.py --direction listen     # SECONDARY, H1-stamped
  python make_fig_dist_sankey.py --selftest             # gating logic; draws no file
  python make_fig_dist_sankey.py --demo [--demo-dir D]  # planted-data render (default /tmp)
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figB_sankey import EXPECT, _srgb_lin, _hex_rgb, _oklab, _cvd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
FIGDIR = REPO / "docs/drafts/figs"
CAPTION = "docs/drafts/figs/fig_dist_sankey_caption.md"

# --------------------------------------------------------------------------- palettes
# Generation layer: transcribed from make_figB_matrix.HUE (asserted against the real module in
# --selftest whenever it is importable — the family_topk_shift_fmt.py:226-231 pattern).
HUE_GEN = {"C": "#009E73", "WSTAR": "#CC3311", "NEITHER": "#b0b0ab"}
# Distributional layer (validated, decided): blue / orange / one grey for all three GREY_* states.
COL_DIST = {"FAVOURS_C": "#0072B2", "FAVOURS_WSTAR": "#E69F00", "GREY": "#8a8a92"}
SURFACE = "#fcfcfb"

CATS_GEN = ["C", "WSTAR", "NEITHER"]                       # fixed top->bottom order
CATS_DIST = ["FAVOURS_C", "FAVOURS_WSTAR", "GREY"]         # aligned with the gen order
NICE_GEN = {"C": "names C", "WSTAR": "names W*", "NEITHER": "neither"}
NICE_DIST = {"FAVOURS_C": "favours C", "FAVOURS_WSTAR": "favours W*", "GREY": "grey"}

RULE_S = ("GREY_COLLISION", "GREY_NO_ONSET", "GREY_TIED", "FAVOURS_C", "FAVOURS_WSTAR")
# §9.4's declared collapse, for DRAWING and for nothing else; five-way counts are printed
# unrolled at draw time and belong in the caption, never smoothed.
COLLAPSE = {"FAVOURS_C": "FAVOURS_C", "FAVOURS_WSTAR": "FAVOURS_WSTAR",
            "GREY_COLLISION": "GREY", "GREY_NO_ONSET": "GREY", "GREY_TIED": "GREY"}

SLOTS = ("single", "second_turn", "forced_final")
STAGES_DIST = ["single", "second turn", "forced final"]
STAGES_GEN = ["planted", "elicited final"]
GEN_FIELD = {"counter": "faithful_elicit", "neutral": "faithful_neutral_elicit"}
PLANT = {"fold": "C", "listen": "WSTAR"}                   # plant = C under fold, W* under listen (§1)

CELLS_BY_VARIANT = {"it": ["2bit", "9bit", "27bit"], "base": ["2bbase", "9bbase", "27bbase"]}
TITLE = {"2bit": "2b-it", "9bit": "9b-it", "27bit": "27b-it",
         "2bbase": "2b base", "9bbase": "9b base", "27bbase": "27b base"}

BANDS = ("LAYERS_CONCORDANT", "LAYERS_PARTIAL", "LAYERS_DISCORDANT", "LAYERS_UNEVALUABLE")
WORST_FIRST = ("LAYERS_UNEVALUABLE", "LAYERS_DISCORDANT", "LAYERS_PARTIAL", "LAYERS_CONCORDANT")


def _check_palette():
    """figB's inline Vienot+OKLab check, extended to the new hues: every WITHIN-LAYER pair
    must clear dE*100 >= 15 (normal) / >= 8 (protan, deutan). Layers are separate, labelled
    bands, so cross-layer pairs are not gated; identity is also carried by count labels."""
    pairs = [(HUE_GEN[a], HUE_GEN[b]) for a, b in (("C", "WSTAR"), ("C", "NEITHER"), ("WSTAR", "NEITHER"))]
    pairs += [(COL_DIST[a], COL_DIST[b]) for a, b in (("FAVOURS_C", "FAVOURS_WSTAR"),
                                                      ("FAVOURS_C", "GREY"), ("FAVOURS_WSTAR", "GREY"))]
    for a, b in pairs:
        la, lb = [_srgb_lin(x) for x in _hex_rgb(a)], [_srgb_lin(x) for x in _hex_rgb(b)]
        for kind in ("normal", "protan", "deutan"):
            pa = _oklab(la if kind == "normal" else _cvd(la, kind))
            pb = _oklab(lb if kind == "normal" else _cvd(lb, kind))
            de = 100 * sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5
            floor = 15 if kind == "normal" else 8
            assert de >= floor, (a, b, kind, de)


_check_palette()


# --------------------------------------------------------------------------- the gate
class GateRefusal(Exception):
    """Raised when §9.4 does not license a draw. main() prints it and exits non-zero."""


_KEY_RE = re.compile(r"verdict|band", re.I)
_ID_FIELDS = ("cell", "direction", "turn2", "arm")


def _has_tok(ctx_el, tok):
    """Token match with boundaries: '9bit' matches 'fl_9bit_ext2' and '9b-it' but '2bit'
    never matches '27bit'."""
    if re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(tok), ctx_el):
        return True
    return re.sub(r"[^a-z0-9]", "", ctx_el) == tok


def _walk(node, ctx, hits):
    if isinstance(node, dict):
        ctx = ctx | {str(v).lower() for k, v in node.items()
                     if k in _ID_FIELDS and isinstance(v, str)}
        stamp = node.get("stamp")
        if isinstance(stamp, dict):           # the shipped stamp key `arm` carries the DIRECTION (§1.3)
            ctx = ctx | {str(v).lower() for k, v in stamp.items()
                         if k in _ID_FIELDS and isinstance(v, str)}
        for k, v in node.items():
            if isinstance(v, str) and v in BANDS and _KEY_RE.search(str(k)):
                hits.append((ctx | {str(k).lower()}, v))
            else:
                _walk(v, ctx | {str(k).lower()}, hits)
    elif isinstance(node, list):
        for v in node:
            _walk(v, ctx, hits)


def find_band(join_obj, cell, direction, arm):
    """All §9.4 bands attributable to (cell, direction, arm), resolved WORST-WINS; None if
    no verdict is found (which is a refusal, never a default)."""
    hits = []
    _walk(join_obj, frozenset(), hits)
    toks = [cell.lower(), direction.lower(), arm.lower()]
    got = {b for c, b in hits if all(any(_has_tok(el, t) for el in c) for t in toks)}
    if not got:
        return None
    for b in WORST_FIRST:
        if b in got:
            return b
    return None


def gate(join_obj, cells, direction, arm):
    """The §9.4 gate. Returns (mode, {cell: band}); raises GateRefusal when no draw is licensed."""
    bands = {}
    for cell in cells:
        b = find_band(join_obj, cell, direction, arm)
        if b is None:
            raise GateRefusal("no §9.4 LAYERS_* verdict found in the join for (%s, %s, %s) "
                              "— refusing to draw (§9.4 branch 1)" % (cell, direction, arm))
        bands[cell] = b
    bad = [c for c in cells if bands[c] == "LAYERS_UNEVALUABLE"]
    if bad:
        raise GateRefusal("LAYERS_UNEVALUABLE at %s for (%s, %s) — no agreement verdict exists "
                          "there, so no figure is licensed (§9.4 branch 1)"
                          % (", ".join(bad), direction, arm))
    mode = "two_layer" if all(bands[c] == "LAYERS_CONCORDANT" for c in cells) else "fallback"
    return mode, bands


def decide(join_path, cells, direction, arm):
    try:
        join_obj = json.loads(Path(join_path).read_text())
    except (OSError, ValueError) as e:
        raise GateRefusal("join missing/unreadable at %s (%s: %s) — the join is the ONLY verdict "
                          "source (§13); refusing to draw" % (join_path, type(e).__name__, e))
    return gate(join_obj, cells, direction, arm)


# --------------------------------------------------------------------------- dist artifacts
def _rec_direction(rec):
    d = rec.get("direction")
    if d is None and isinstance(rec.get("stamp"), dict):
        d = rec["stamp"].get("arm")          # the shipped stamp key `arm` carries the DIRECTION (§1.3)
    if d is None and rec.get("arm") in ("fold", "listen"):
        d = rec.get("arm")
    return d


def chains(recs, direction, arm, n_items=82):
    """Per-item (q, s0, s1, s2, gen_label) for ONE (direction, arm) chain. Slot `single` is
    shared (§2 point 3). Fails loudly on: a slot-1/2 record missing either axis, an unknown
    Rule-S state, a conflicting duplicate, a missing slot, or an item count != n_items."""
    per, gen = {}, {}
    for r in recs:
        if not isinstance(r, dict):
            continue
        slot = r.get("slot_id") or (r.get("slot") if r.get("slot") in SLOTS else None)
        if slot not in SLOTS or r.get("state") is None:
            continue
        st, q = r["state"], r.get("q")
        assert q is not None, "record with a state but no q: %r" % (r,)
        assert st in RULE_S, "unknown Rule-S state %r at q=%r (frozen set: %s)" % (st, q, RULE_S)
        rd, rt = _rec_direction(r), r.get("turn2")
        if slot == "single":
            if rd not in (None, direction) or rt not in (None, arm):
                continue                     # another chain's copy of the shared slot
        else:
            assert rd is not None and rt is not None, \
                "slot-%s record at q=%r lacks direction/turn2 — cannot attribute a chain (§1.3)" % (slot, q)
            if rd != direction or rt != arm:
                continue
        key = (q, slot)
        assert per.get(key, st) == st, "conflicting duplicate at %r: %r vs %r" % (key, per[key], st)
        per[key] = st
        if slot == "forced_final":
            gen[q] = r.get(GEN_FIELD[arm])
    qs = sorted({q for q, _ in per})
    assert len(qs) == n_items, "expected %d items, found %d" % (n_items, len(qs))
    out = []
    for q in qs:
        missing = [s for s in SLOTS if (q, s) not in per]
        assert not missing, "q=%r missing slot(s) %s for (%s, %s)" % (q, missing, direction, arm)
        out.append((q, per[(q, "single")], per[(q, "second_turn")], per[(q, "forced_final")], gen.get(q)))
    return out


def load_cell(dist_dir, cell, direction, arm):
    path = Path(dist_dir) / ("forcedfinal_dist_ff_ext2_%s.json" % cell)
    d = json.loads(path.read_text())
    recs = d.get("items") or (d.get("result") or {}).get("items") or d.get("records")
    assert recs, "no items/records list in %s" % path
    return chains(recs, direction, arm)


def _bucket(lab):
    return "NEITHER" if lab == "UNRESOLVED_ALIAS" else lab


def assert_expect(cell, direction, rows):
    """The frozen EXPECT-dict assertion, before drawing: the copied-through faithful-strict
    elicited labels must reproduce the committed counts exactly."""
    labs = [r[4] for r in rows]
    assert all(l is not None for l in labs), \
        "%s: generation label (%s) absent on some records — cannot assert EXPECT" % (cell, GEN_FIELD)
    got = Counter(_bucket(l) for l in labs)
    exp = {c: EXPECT[direction][TITLE[cell]].get(c, 0) for c in CATS_GEN}
    assert {c: got.get(c, 0) for c in CATS_GEN} == exp, (cell, direction, dict(got), exp)
    print("[ok] %-8s gen-layer elicited counts == frozen EXPECT  %s"
          % (cell, " ".join("%s=%2d" % (c, exp[c]) for c in CATS_GEN)))


def print_unrolled(cell, rows):
    """§9.4: the five-way counts are ALWAYS reported unrolled; the figure draws the collapse."""
    for k, slot in enumerate(SLOTS):
        c = Counter(r[1 + k] for r in rows)
        print("[unrolled] %-8s %-12s %s" % (cell, slot, " ".join("%s=%d" % (s, c.get(s, 0)) for s in RULE_S)))


# --------------------------------------------------------------------------- drawing
NODE_W, GAP = 0.055, 1.6


def _offsets(counts, cats):
    tops, y = {}, 0.0
    for c in cats:
        tops[c] = y
        y += counts.get(c, 0) + (GAP if counts.get(c, 0) else 0)
    return tops, y - GAP


def _ribbon(ax, x0, y0, x1, y1, w, color, alpha=0.45):
    xm = (x0 + x1) / 2
    verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1),
             (x1, y1 + w), (xm, y1 + w), (xm, y0 + w), (x0, y0 + w), (x0, y0)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, alpha=alpha, lw=0, zorder=2))


def draw_flow(ax, seqs, cats, col, stages, alpha=None):
    """n-stage alluvial for ONE (direction, arm) chain: ribbons colored by DESTINATION, and a
    direct count label on EVERY nonzero node (mandatory: #E69F00 fails 3:1 on white).

    `alpha` is an optional {"node":…, "rib":…} pair; the default reproduces this script's own
    committed output exactly. make_fig_dist_grid passes make_figB_matrix.ALPHA to restore the
    readout sankey's opacity-encodes-training channel."""
    a = alpha or dict(node=1.0, rib=0.45)
    K = len(stages)
    counts = [{c: sum(1 for s in seqs if s[k] == c) for c in cats} for k in range(K)]
    tops, height = zip(*(_offsets(sc, cats) for sc in counts))
    for k in range(K):
        for c in cats:
            n = counts[k].get(c, 0)
            if not n:
                continue
            ax.add_patch(plt.Rectangle((k - NODE_W, tops[k][c]), 2 * NODE_W, n,
                                       facecolor=col[c], alpha=a["node"], lw=0, zorder=3))
            side = 1 if k == K - 1 else -1
            ax.text(k + side * (NODE_W + 0.04), tops[k][c] + n / 2, str(n),
                    ha="left" if side > 0 else "right", va="center", fontsize=7.5,
                    color="#333333", zorder=4)
    for k in range(K - 1):
        used_s = {c: 0.0 for c in cats}
        used_d = {c: 0.0 for c in cats}
        for cs in cats:
            for cd in cats:
                w = sum(1 for s in seqs if s[k] == cs and s[k + 1] == cd)
                if not w:
                    continue
                y0 = tops[k][cs] + used_s[cs]; used_s[cs] += w
                y1 = tops[k + 1][cd] + used_d[cd]; used_d[cd] += w
                _ribbon(ax, k + NODE_W, y0, k + 1 - NODE_W, y1, w, col[cd], a["rib"])
    ax.set_xlim(-0.45, K - 1 + 0.45)
    ax.set_ylim(max(max(height), 82) + GAP, -GAP)
    ax.set_xticks(range(K), stages, fontsize=8)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_facecolor(SURFACE)


def make_figure(out_png, mode, cells, cellsdata, bands, direction, arm, variant, demo=False):
    two = mode == "two_layer"
    fig, axes = plt.subplots(2 if two else 1, 3, figsize=(11.5, 8.6) if two else (11.5, 4.9),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, cell in enumerate(cells):
        rows = cellsdata[cell]
        dist_seqs = [(COLLAPSE[r[1]], COLLAPSE[r[2]], COLLAPSE[r[3]]) for r in rows]
        if two:
            gen_seqs = [(PLANT[direction], _bucket(r[4])) for r in rows]
            draw_flow(axes[0][j], gen_seqs, CATS_GEN, HUE_GEN, STAGES_GEN)
        axes[0][j].set_title("%s — %s" % (TITLE[cell], bands[cell]), fontsize=9.5, pad=6)
        draw_flow(axes[-1][j], dist_seqs, CATS_DIST, COL_DIST, STAGES_DIST)
    if two:
        axes[0][0].set_ylabel("GENERATION layer\n(faithful-strict labels)", fontsize=9,
                              rotation=0, ha="right", va="center", labelpad=46)
    axes[-1][0].set_ylabel("DISTRIBUTIONAL layer\n(Rule S, first-token)", fontsize=9,
                           rotation=0, ha="right", va="center", labelpad=46)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL_DIST[c]) for c in CATS_DIST]
    labels = ["%s (dist)" % NICE_DIST[c] for c in CATS_DIST]
    if two:
        handles += [plt.Rectangle((0, 0), 1, 1, color=HUE_GEN[c]) for c in CATS_GEN]
        labels += ["%s (gen)" % NICE_GEN[c] for c in CATS_GEN]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    half = "-it" if variant == "it" else "base"
    form = "two-layer alluvial: §9.4 LAYERS_CONCORDANT at all three scales" if two \
        else "distributional layer ONLY: §9.4 fallback, no cross-layer statement"
    fig.suptitle("Forced-final distributional readout — %s direction, %s arm, %s half%s\n%s"
                 % (direction.upper(), arm, half,
                    "  [DEMO — PLANTED DATA]" if demo else "", form), fontsize=11, y=0.995)
    stamps = []
    if variant == "base":
        stamps.append("CONTEXT_CONTAMINATED_MEASURED on every base slot-2 number (§6.4)")
    if direction == "listen":
        stamps.append("LISTEN_CONTINGENT_ON_H1 (§1.2)")
    stamps.append("27b quotable only with §10's four-part disclosure (see caption)")
    note1 = ("dist states are FIRST-TOKEN Rule-S reads at the canonical key — never 'the probability "
             "of C' (§4.2); GREY = collision / no-onset / tied (unrolled counts in the caption)")
    note2 = "transitions are within-layer, within-direction, within-arm (§2, §9.5); %s" % "; ".join(stamps)
    note3 = "Full caption: %s" % CAPTION
    for y, t in ((0.088, note1), (0.071, note2), (0.054, note3)):
        fig.text(0.5, y, t, ha="center", fontsize=7, color="#666666")
    fig.tight_layout(rect=(0.02, 0.105, 1, 0.945))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print("[written]", out_png)


# --------------------------------------------------------------------------- demo (planted)
def _demo_rows(seed):
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(82):
        s0 = rng.choices(RULE_S, weights=(4, 8, 2, 40, 28))[0]
        s1 = rng.choices(RULE_S, weights=(4, 10, 3, 25, 40))[0]
        s2 = rng.choices(RULE_S, weights=(4, 6, 2, 22, 48))[0]
        gen = {"FAVOURS_C": "C", "FAVOURS_WSTAR": "WSTAR"}.get(s2, "NEITHER")
        if rng.random() < 0.08:
            gen = rng.choice(("C", "WSTAR", "NEITHER", "UNRESOLVED_ALIAS"))
        rows.append(("q%02d" % i, s0, s1, s2, gen))
    return rows


def demo(demo_dir):
    demo_dir = Path(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)
    cells = CELLS_BY_VARIANT["it"]
    data = {c: _demo_rows(i) for i, c in enumerate(cells)}
    print("[demo] EXPECT assertion SKIPPED — planted data, layout inspection only")
    for c in cells:
        print_unrolled(c, data[c])
    make_figure(demo_dir / "fig_dist_sankey_DEMO_twolayer.png", "two_layer", cells, data,
                {c: "LAYERS_CONCORDANT" for c in cells}, "fold", "counter", "it", demo=True)
    make_figure(demo_dir / "fig_dist_sankey_DEMO_distonly.png", "fallback", cells, data,
                {c: "LAYERS_PARTIAL" for c in cells}, "fold", "counter", "it", demo=True)


# --------------------------------------------------------------------------- selftest
def _mkjoin(bands_by_cell, direction="fold", arm="counter"):
    return {"cells": {c: {direction: {arm: {"layers_verdict": b}}}
                      for c, b in bands_by_cell.items()}}


def selftest():
    ok = lambda m: print("[ok]", m)

    # gate: each band reached, on planted joins, no file drawn
    cells = CELLS_BY_VARIANT["it"]
    j = _mkjoin({c: "LAYERS_CONCORDANT" for c in cells})
    assert gate(j, cells, "fold", "counter")[0] == "two_layer"
    ok("all-CONCORDANT -> two_layer")
    for b in ("LAYERS_PARTIAL", "LAYERS_DISCORDANT"):
        j = _mkjoin({"2bit": "LAYERS_CONCORDANT", "9bit": b, "27bit": "LAYERS_CONCORDANT"})
        mode, bands = gate(j, cells, "fold", "counter")
        assert mode == "fallback" and bands["9bit"] == b
        ok("%s at one scale -> fallback" % b)
    j = _mkjoin({"2bit": "LAYERS_CONCORDANT", "9bit": "LAYERS_UNEVALUABLE", "27bit": "LAYERS_CONCORDANT"})
    try:
        gate(j, cells, "fold", "counter"); raise SystemExit("UNEVALUABLE did not refuse")
    except GateRefusal as e:
        assert "9bit" in str(e); ok("UNEVALUABLE -> GateRefusal (no draw)")
    try:
        gate(_mkjoin({"2bit": "LAYERS_CONCORDANT"}), cells, "fold", "counter")
        raise SystemExit("missing cell did not refuse")
    except GateRefusal:
        ok("missing verdict for a cell -> GateRefusal")
    try:
        decide("/nonexistent/forcedfinal_join.json", cells, "fold", "counter")
        raise SystemExit("missing join did not refuse")
    except GateRefusal:
        ok("missing join file -> GateRefusal, non-zero exit path")

    # record-style join, id fields incl. stamp.arm carrying the DIRECTION (§1.3)
    j = {"verdicts_9_4": [
        {"cell": "9bit", "turn2": "counter", "stamp": {"arm": "fold"}, "band": "LAYERS_PARTIAL"},
        {"cell": "9bit", "turn2": "neutral", "stamp": {"arm": "fold"}, "band": "LAYERS_CONCORDANT"}]}
    assert find_band(j, "9bit", "fold", "counter") == "LAYERS_PARTIAL"
    assert find_band(j, "9bit", "fold", "neutral") == "LAYERS_CONCORDANT"
    ok("record-style join found; arm axes not conflated")

    # worst-wins across label families = LAYER_AGREEMENT_CONTESTED can never license two-layer
    j = {"9bit": {"fold": {"counter": {"verdict_faithful": "LAYERS_CONCORDANT",
                                       "verdict_commit": "LAYERS_PARTIAL"}}}}
    assert find_band(j, "9bit", "fold", "counter") == "LAYERS_PARTIAL"
    ok("contested (two families, two bands) resolves WORST-WINS")

    # prose + bare-list immunity; 2bit/27bit token collision
    j = {"9bit": {"fold": {"counter": {
        "decision_rule": "LAYERS_CONCORDANT iff disagree_frac <= 0.10 (§9.4)",
        "band_order": ["LAYERS_CONCORDANT", "LAYERS_PARTIAL"]}}}}
    assert find_band(j, "9bit", "fold", "counter") is None
    ok("prose and bare lists never match")
    j = _mkjoin({"27bit": "LAYERS_CONCORDANT"})
    assert find_band(j, "2bit", "fold", "counter") is None
    assert find_band(j, "27bit", "fold", "counter") == "LAYERS_CONCORDANT"
    ok("'2bit' does not match a '27bit' context")

    # chains: assembly, shared single, cross-axis attribution, failure modes
    def rec(q, slot, state, d="fold", t="counter", **kw):
        r = {"q": q, "slot_id": slot, "state": state, "direction": d, "turn2": t}
        r.update(kw); return r
    recs = []
    for q in ("qa", "qb"):
        recs.append({"q": q, "slot_id": "single", "state": "FAVOURS_C"})   # shared: no axes (§2 pt 3)
        for d in ("fold", "listen"):
            for t in ("counter", "neutral"):
                recs.append(rec(q, "second_turn", "FAVOURS_WSTAR", d, t))
                recs.append(rec(q, "forced_final", "GREY_TIED", d, t,
                               faithful_elicit="WSTAR", faithful_neutral_elicit="NEITHER"))
    rows = chains(recs, "fold", "counter", n_items=2)
    assert [r[1:] for r in rows] == [("FAVOURS_C", "FAVOURS_WSTAR", "GREY_TIED", "WSTAR")] * 2
    rows_n = chains(recs, "listen", "neutral", n_items=2)
    assert rows_n[0][4] == "NEITHER", "neutral arm must read faithful_neutral_elicit"
    ok("chains assemble; shared single serves every chain; no cross-arm/direction mixing possible")
    try:
        chains([{"q": "qa", "slot_id": "second_turn", "state": "FAVOURS_C"}], "fold", "counter", 1)
        raise SystemExit("axis-less slot-1 record accepted")
    except AssertionError:
        ok("slot-1/2 record without direction/turn2 raises (§1.3)")
    try:
        chains([rec("qa", "single", "MOVED")], "fold", "counter", 1)
        raise SystemExit("unknown state accepted")
    except AssertionError:
        ok("non-Rule-S state raises (frozen five-state set, §4.2)")
    try:
        chains(recs[:3], "fold", "counter", n_items=2)
        raise SystemExit("missing slot accepted")
    except AssertionError:
        ok("missing slot raises")
    assert set(COLLAPSE) == set(RULE_S) and set(COLLAPSE.values()) == set(CATS_DIST)
    ok("collapse map is total over Rule S and onto the three drawn classes (§9.4)")

    # frozen EXPECT: real committed counts pass, a perturbation fails
    good = [("q%d" % i, "FAVOURS_C", "FAVOURS_C", "FAVOURS_C", lab)
            for i, lab in enumerate(["WSTAR"] * 55 + ["C"] * 27)]
    assert_expect("9bit", "fold", good)
    try:
        assert_expect("9bit", "fold", good[:-1] + [("q81", "FAVOURS_C", "FAVOURS_C", "FAVOURS_C", "WSTAR")])
        raise SystemExit("EXPECT mismatch accepted")
    except AssertionError:
        ok("EXPECT-dict assertion fires on a single moved count")

    # transcribed HUE against the real module, whenever importable
    try:
        sys.path.insert(0, str(REPO / "controls"))
        import make_figB_matrix as _m
        assert _m.HUE == HUE_GEN, (_m.HUE, HUE_GEN)
        ok("HUE_GEN matches make_figB_matrix.HUE")
    except ImportError as e:
        print("[skip] make_figB_matrix not importable here (%s); HUE transcription unchecked" % e)

    _check_palette()
    ok("palette check (both layers, Vienot+OKLab floors)")
    print("SELFTEST_OK make_fig_dist_sankey")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="verdict-gated forced-final distributional alluvial")
    ap.add_argument("--join", default=str(REPO / "out/forcedfinal_join.json"))
    ap.add_argument("--dist-dir", default=str(REPO / "out"))
    ap.add_argument("--direction", choices=("fold", "listen"), default="fold")
    ap.add_argument("--arm", choices=("counter", "neutral"), default="counter")
    ap.add_argument("--variant", choices=("it", "base"), default="it")
    ap.add_argument("--out", default=None, help="output PNG (default: fig_dist_sankey_<axes>[_distonly].png)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true", help="render from planted data for layout inspection")
    ap.add_argument("--demo-dir", default="/tmp")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.demo:
        demo(a.demo_dir); return
    if a.arm == "neutral":
        print("REFUSED: no frozen EXPECT exists for the neutral arm's generation layer "
              "(faithful_neutral_elicit has no committed counts). Extend EXPECT deliberately, "
              "never as a side effect of drawing.")
        sys.exit(2)
    cells = CELLS_BY_VARIANT[a.variant]
    try:
        mode, bands = decide(a.join, cells, a.direction, a.arm)
    except GateRefusal as e:
        print("NOT DRAWN:", e)
        sys.exit(2)
    print("[gate] " + "  ".join("%s=%s" % (c, bands[c]) for c in cells) + "  -> mode=%s" % mode)
    cellsdata = {c: load_cell(a.dist_dir, c, a.direction, a.arm) for c in cells}
    for c in cells:
        assert_expect(c, a.direction, cellsdata[c])
        print_unrolled(c, cellsdata[c])
    out = a.out or str(FIGDIR / ("fig_dist_sankey_%s_%s_%s%s.png"
                                 % (a.direction, a.arm, a.variant,
                                    "" if mode == "two_layer" else "_distonly")))
    make_figure(out, mode, cells, cellsdata, bands, a.direction, a.arm, a.variant)


if __name__ == "__main__":
    main()

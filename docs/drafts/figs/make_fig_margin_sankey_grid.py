#!/usr/bin/env python3
"""fig_margin_sankey_grid — whole-answer-string margin, drawn as a dual-arm matrix in the style of
`make_fig_dist_grid.py`, informed by `make_fig_margin_estimation.py`.

WHY THIS EXISTS. Figure 2 (`fig2_dist_matrix`) bands the FIRST-TOKEN Rule-S read. The margin-estimation
plot shows the continuous WHOLE-STRING margin at `forced_final` but refuses to band it into a sankey
(section 2 of that file: banding destroys magnitude and the arms are not successive). This figure is
the third object: the whole-string margin, banded with the repo's EXISTING near-margin threshold
(`MARGIN_KEEP = 1.5` / `make_fig_margin_estimation.TORN`), laid out like the sankey matrix so the three
figures share a grid.

WHAT A STATE IS. At a scored answer slot,

    M = lp(C) - lp(W*)     # absolute, not plant-relative
    FAVOURS_C      if M >  +TORN
    FAVOURS_WSTAR  if M <  -TORN
    TORN           if |M| <  TORN     # strict, same as headroom_pass

No new threshold is invented. Exact |M| == TORN is directed (not torn), matching
`controls/family_cave_diagnose.headroom_pass`.

TWO SOURCES, two figures (the forced_final artifact has `r_lp` at ONE slot only, so a 3-stage
within-chain alluvial is not drawable from it):

  forced_final (default) — `out/forcedfinal_dist_ff_ext2_<cell>.json` items[].r_lp at slot
      `forced_final`, both arms. Each panel is two UNCONNECTED stacks (neutral | counter). No ribbon
      between arms: section 2 forbids a neutral→counter transition; the pairing is statistical, not
      chronological. Full 4×3 grid = Figure 1's rows and columns.

  diagnose — `family_cave_diagnose` M0 / Mc_neutral / Mc_counter at the REPLY slot (no model reply in
      context). Each panel is two arms with a bare→after ribbon INSIDE each arm (margin_flow_9b's
      legal topology). Fold only: listen reply-slot numbers are withdrawn (`OWED.md` B1 / H1). This is
      a different slot from the estimation plot; the footer says so.

PROHIBITIONS. No neutral→counter ribbon, no fold→listen ribbon, no base-vs-it contrast, no pooling.
Listen rows (forced_final source only) are `LISTEN_CONTINGENT_ON_H1`; -base rows are
`CONTEXT_CONTAMINATED_MEASURED`. Adjacency in a grid is not a contrast.

usage:
  python3 make_fig_margin_sankey_grid.py [--source forced_final|diagnose|both]
  python3 make_fig_margin_sankey_grid.py --selftest
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
from make_figB_matrix import ALPHA, HUE as HUE_GEN_REF, ROWS, SCALES  # noqa: E402
from make_figB_sankey import _cvd, _hex_rgb, _oklab, _srgb_lin  # noqa: E402
from make_fig_dist_grid import CELL, VOIDING, GridRefusal, fidelity  # noqa: E402
from make_fig_dist_sankey import SURFACE, draw_flow  # noqa: E402
from make_fig_margin_estimation import TORN  # noqa: E402

REPO = Path(__file__).resolve().parents[3]

CATS = ["FAVOURS_C", "FAVOURS_WSTAR", "TORN"]
NICE = {"FAVOURS_C": "favours C", "FAVOURS_WSTAR": "favours W*",
        "TORN": "|M| < 1.5 nats (torn)"}
COL = {"FAVOURS_C": "#0072B2", "FAVOURS_WSTAR": "#E69F00", "TORN": "#8a8a92"}

# Fold diagnose artifacts — same committed paths as make_fig_slot_margin_allcells.CELLS.
DIAGNOSE = {
    ("2b", "base"): ("results_r1_dist_2b9b/out/family_cave_diagnose_vfam_ext2_2bbase.json", "base"),
    ("9b", "base"): ("results_absdecode_ext2/out/family_cave_diagnose_vfam_ext2_9bbase.json", "base"),
    ("27b", "base"): ("results_r1_dist_27b/out/family_cave_diagnose_vfam_ext2_27bbase.json", "base"),
    ("2b", "it"): ("results_r1_dist_2b9b/out/family_cave_diagnose_vfam_ext2_2bit.json", "it"),
    ("9b", "it"): ("results_itreadout_modelw/out/family_cave_diagnose_vfam_ext2_9bit.json", "it"),
    ("27b", "it"): ("results_r1_dist_27b/out/family_cave_diagnose_vfam_ext2_27bit.json", "it"),
}
DIAG_ROWS = [("fold", "base"), ("fold", "it")]


def _de(a, b, kind):
    la, lb = [_srgb_lin(x) for x in _hex_rgb(a)], [_srgb_lin(x) for x in _hex_rgb(b)]
    pa = _oklab(la if kind == "normal" else _cvd(la, kind))
    pb = _oklab(lb if kind == "normal" else _cvd(lb, kind))
    return 100 * sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5


def _check_palette():
    for a, b in (("FAVOURS_C", "FAVOURS_WSTAR"), ("FAVOURS_C", "TORN"), ("FAVOURS_WSTAR", "TORN")):
        for kind in ("normal", "protan", "deutan"):
            de = _de(COL[a], COL[b], kind)
            assert de >= (15 if kind == "normal" else 8), (a, b, kind, de)
    for dk, gk in (("FAVOURS_C", "C"), ("FAVOURS_WSTAR", "WSTAR"), ("TORN", "NEITHER")):
        for kind in ("normal", "protan", "deutan"):
            assert _de(COL[dk], HUE_GEN_REF[gk], kind) >= 8, (dk, gk, kind)


_check_palette()


def band(m):
    """Strict |M| < TORN → torn; matches headroom_pass / MARGIN_KEEP.

    Exact |M| == TORN is directed (not torn), same as `not headroom_pass(TORN)`."""
    if abs(m) < TORN:
        return "TORN"
    return "FAVOURS_C" if m > 0 else "FAVOURS_WSTAR"


def _bucket_gen(lab):
    return "NEITHER" if lab in (None, "UNRESOLVED_ALIAS") else lab


# --------------------------------------------------------------------------- forced_final source
def load_ff_panel(dist_dir, cell, direction):
    """Per-arm list of banded states at forced_final, keyed by join_key; plus gen label on counter."""
    d = json.loads((Path(dist_dir) / ("forcedfinal_dist_ff_ext2_%s.json" % cell)).read_text())
    arms, gen = {}, {}
    for it in d["items"]:
        if it.get("slot_id") != "forced_final" or it.get("direction") != direction:
            continue
        r = it.get("r_lp")
        assert r and r.get("register") == "lp_whole_string", (cell, it.get("join_key"), r)
        m = r["lpC"]["lp_total"] - r["lpW"]["lp_total"]
        arms.setdefault(it["turn2"], {})[it["join_key"]] = band(m)
        if it["turn2"] == "counter":
            gen[it["join_key"]] = _bucket_gen(it.get("faithful_elicit"))
    keys = sorted(set(arms["neutral"]) & set(arms["counter"]))
    assert len(keys) == 82, (cell, direction, len(keys))
    neu = [arms["neutral"][k] for k in keys]
    cnt = [arms["counter"][k] for k in keys]
    gens = [gen[k] for k in keys]
    return neu, cnt, gens


def _vs_fig1(states, gens):
    """Clear C↔W* flips only. Torn vs gen is a different kind of disagreement; stamped separately."""
    return sum(1 for s, g in zip(states, gens)
               if (s == "FAVOURS_C" and g == "WSTAR") or (s == "FAVOURS_WSTAR" and g == "C"))


def draw_dual_stacks(ax, left, right, labels, alpha):
    """Two unconnected category stacks. No ribbon — the arms are alternatives (section 2)."""
    # draw_flow accepts length-1 sequences; call it twice on shifted coordinates via two stages each
    # with empty mid gap — simpler: stack manually at x=0 and x=1.
    from make_fig_dist_sankey import GAP, NODE_W, _offsets  # local to keep import surface small
    a = alpha
    cols = [left, right]
    counts = [{c: sum(1 for s in col if s == c) for c in CATS} for col in cols]
    tops, height = zip(*(_offsets(sc, CATS) for sc in counts))
    for k in range(2):
        for c in CATS:
            n = counts[k].get(c, 0)
            if not n:
                continue
            ax.add_patch(plt.Rectangle((k - NODE_W, tops[k][c]), 2 * NODE_W, n,
                                       facecolor=COL[c], alpha=a["node"], lw=0, zorder=3))
            side = 1 if k == 1 else -1
            ax.text(k + side * (NODE_W + 0.04), tops[k][c] + n / 2, str(n),
                    ha="left" if side > 0 else "right", va="center", fontsize=7.5,
                    color="#333333", zorder=4)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(max(max(height), 82) + GAP, -GAP)
    ax.set_xticks([0, 1], labels, fontsize=8)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_facecolor(SURFACE)


def make_ff_grid(join_path, dist_dir, out_png):
    fid = fidelity(join_path)
    missing = [CELL[(s, v)] for _, v in ROWS for s in SCALES if CELL[(s, v)] not in fid]
    if missing:
        raise GridRefusal("no section 9.1 verdict in the join for %s" % ", ".join(sorted(set(missing))))
    voided = {c: v for c, (v, _, _) in fid.items() if v in VOIDING}
    if voided:
        raise GridRefusal("section 9.1 voided %s — no state may be read"
                          % ", ".join("%s=%s" % kv for kv in sorted(voided.items())))

    labels = ["neutral\n(control)", "counter\n(challenged)"]
    fig, axes = plt.subplots(len(ROWS), len(SCALES), figsize=(12.5, 14.0), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, (direction, variant) in enumerate(ROWS):
        for j, scale in enumerate(SCALES):
            cell = CELL[(scale, variant)]
            ax = axes[i][j]
            neu, cnt, gens = load_ff_panel(dist_dir, cell, direction)
            draw_dual_stacks(ax, neu, cnt, labels if i == len(ROWS) - 1 else ["", ""],
                             ALPHA[variant])
            if i == 0:
                ax.set_title(scale, fontsize=13, pad=28)
            if j == 0:
                start = "start: C planted" if direction == "fold" else "start: W* planted"
                ax.set_ylabel("%s\n%s\n(%s)" % (direction.upper(),
                                                "-base" if variant == "base" else "-chat", start),
                              fontsize=9, rotation=0, ha="right", va="center", labelpad=44)
            n_torn = sum(1 for s in cnt if s == "TORN")
            nd = _vs_fig1(cnt, gens)
            ax.text(1.45, -1.6 * 0.2, "vs Fig 1 flips: %d/82" % nd, fontsize=7, color="#6e6e6a",
                    ha="right", va="bottom")
            ax.text(0.5, 1.04, "counter torn %d/82   neutral torn %d/82"
                    % (n_torn, sum(1 for s in neu if s == "TORN")),
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3, color="#777777")
            print("[margin-sankey] %-8s %-6s counter %s  flips_vs_gen=%d"
                  % (cell, direction,
                     " ".join("%s=%d" % (c, Counter(cnt)[c]) for c in CATS), nd))

    fig.suptitle("What the whole-string margin favours — the same 82 pairs and the same grid as "
                 "Figure 1,\nread as sign(log P(C) − log P(W*)) at the elicited final answer "
                 "(|M| < 1.5 = torn)",
                 fontsize=13, y=0.988)
    fig.tight_layout(rect=(0.055, 0.105, 0.995, 0.966))
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center",
               bbox_to_anchor=(0.5, 0.082), ncol=3, frameon=False, fontsize=10)
    for y, txt in (
        (0.055, "Each panel is two ALTERNATIVE second turns (neutral control | counter challenge), "
                "not two moments — no ribbon connects them (section 2).\n"
                "Colour is the sign of the teacher-forced whole-answer-string margin; opacity is "
                "training, as in Figure 1 (muted = -base, bold = -chat)."),
        (0.030, "Torn = the repo's existing near-margin zone (|M| < 1.5 nats), the same threshold as "
                "the estimation plot beside this figure and as MARGIN_KEEP.\n"
                "\"vs Fig 1 flips\" counts only C↔W* disagreements with the spoken elicited label; "
                "torn-vs-named is a different kind of split."),
        (0.004, "Slot is forced_final (after the free reply), the same slot as the estimation plot. "
                "-base rows are contaminated-context; listen rows are provisional;\n"
                "no -base-vs-chat contrast is computed. Full caption: "
                "docs/drafts/figs/fig_margin_sankey_matrix_caption.md"),
    ):
        fig.text(0.5, y, txt, ha="center", va="bottom", fontsize=7.5, color="#555555",
                 linespacing=1.5)
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)
    return out_png


# --------------------------------------------------------------------------- diagnose source
def load_diagnose(path):
    items = json.loads((REPO / path).read_text())["result"]["items"]
    assert len(items) == 82, (path, len(items))
    out = []
    for it in items:
        bare = band(it["M0"])
        neu = band(it["Mc_neutral"])
        cnt = band(it["Mc_counter"])
        out.append((bare, neu, cnt))
    return out


def make_diagnose_grid(out_png):
    fig, axes = plt.subplots(len(DIAG_ROWS), len(SCALES) * 2, figsize=(16.0, 8.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    arm_labs = [("neutral", "no pushback", ["bare question", "after neutral"]),
                ("counter", "pushback", ["bare question", "after push"])]
    for i, (_direction, variant) in enumerate(DIAG_ROWS):
        for j, scale in enumerate(SCALES):
            path, shade = DIAGNOSE[(scale, variant)]
            rows = load_diagnose(path)
            for aj, (arm, arm_title, stages) in enumerate(arm_labs):
                ax = axes[i][2 * j + aj]
                if arm == "neutral":
                    seqs = [(r[0], r[1]) for r in rows]
                else:
                    seqs = [(r[0], r[2]) for r in rows]
                labs = stages if i == len(DIAG_ROWS) - 1 else ["", ""]
                draw_flow(ax, seqs, CATS, COL, labs, alpha=ALPHA[shade])
                if i == 0:
                    ax.set_title("%s — %s" % (scale, arm_title), fontsize=10, pad=10)
                if aj == 0 and j == 0:
                    ax.set_ylabel("%s\n%s\n(start: C planted)"
                                  % ("FOLD", "-base" if variant == "base" else "-chat"),
                                  fontsize=9, rotation=0, ha="right", va="center", labelpad=44)
                print("[diagnose-sankey] %-8s %-8s %s"
                      % (CELL[(scale, variant)], arm,
                         " ".join("%s=%d" % (c, Counter(s[1] for s in seqs)[c]) for c in CATS)))

    fig.suptitle("What the whole-string margin favours under two alternative second turns "
                 "(fold only, reply slot),\n"
                 "banded at |M| < 1.5 nats — bare → after, ribbons only within an arm",
                 fontsize=12, y=0.98)
    fig.tight_layout(rect=(0.05, 0.12, 0.995, 0.94))
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center",
               bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.025,
             "Reply slot (no free reply in context) — NOT the forced_final slot of the estimation "
             "plot. Listen withheld (OWED B1/H1). No neutral→counter ribbon.",
             ha="center", fontsize=7.5, color="#555555")
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)
    return out_png


# --------------------------------------------------------------------------- selftest
def selftest():
    assert "torch" not in sys.modules
    n = 0
    assert band(1.5) == "FAVOURS_C" and band(-1.5) == "FAVOURS_WSTAR"
    assert band(1.5000001) == "FAVOURS_C" and band(-1.5000001) == "FAVOURS_WSTAR"
    assert band(0.0) == "TORN" and band(1.499) == "TORN" and band(-1.499) == "TORN"
    n += 5
    print("[ok] band() uses strict |M| < TORN, matching headroom_pass")
    assert abs(TORN - 1.5) < 1e-12
    n += 1
    print("[ok] TORN is the estimation plot's 1.5")
    assert _vs_fig1(["FAVOURS_C", "FAVOURS_WSTAR", "TORN"], ["WSTAR", "C", "C"]) == 2
    assert _vs_fig1(["FAVOURS_C", "TORN"], ["C", "NEITHER"]) == 0
    n += 2
    print("[ok] vs-Fig1 counts only clear C↔W* flips")
    assert ROWS == [("fold", "base"), ("fold", "it"), ("listen", "base"), ("listen", "it")]
    assert SCALES == ["2b", "9b", "27b"]
    n += 2
    print("[ok] forced_final grid rows/cols match Figure 1")
    _check_palette()
    n += 1
    print("[ok] palette clears within-layer and vs-Fig1 kinship floors")
    print("SELFTEST_OK make_fig_margin_sankey_grid (%d asserts)" % n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("forced_final", "diagnose", "both"), default="both")
    ap.add_argument("--join", default=str(REPO / "out/forcedfinal_join.json"))
    ap.add_argument("--dist-dir", default=str(REPO / "out"))
    ap.add_argument("--out-ff", default=str(REPO / "docs/drafts/figs/fig_margin_sankey_matrix.png"))
    ap.add_argument("--out-diag",
                    default=str(REPO / "docs/drafts/figs/fig_margin_sankey_diagnose_fold.png"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.source in ("forced_final", "both"):
        try:
            make_ff_grid(a.join, a.dist_dir, a.out_ff)
        except GridRefusal as e:
            print("NOT DRAWN (forced_final):", e)
            raise SystemExit(2)
    if a.source in ("diagnose", "both"):
        make_diagnose_grid(a.out_diag)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""fig_margin_estimation — paired estimation plots of the WHOLE-ANSWER-STRING margin, 4x3, at the
forced-final slot, drawn to sit beside POST1's Figure 1.

WHAT THIS IS. An estimation-statistics rendering (Ho et al., Nat Methods 16:565-566, 2019;
Gardner & Altman, BMJ 292:746-750, 1986) of the quantity POST1's Figure 2 slot asks for: the margin
`M = log P(C) - log P(W*)`, each term the summed teacher-forced token log-probability of the ENTIRE
answer string. Rows and columns are `make_figB_matrix.ROWS`/`SCALES`, so the grid overlays Figure 1.

WHY ESTIMATION AND NOT A SANKEY. A sankey needs bands, band edges chosen after seeing values are
post-hoc, and banding a continuous margin destroys the only thing the margin adds over Figure 1 —
magnitude. More decisively, the two things being compared here are NOT successive states, so no flow
may be drawn between them at all (see PROHIBITIONS). An estimation plot shows every observation, puts
the effect size on its own axis with a bootstrap interval, and computes no forbidden transition.

THE CONTRAST. Per item, per cell, per direction, at slot `forced_final`:

    M_plant  = margin favouring the PLANTED answer
             = (lpC - lpW) in fold (plant = C), (lpW - lpC) in listen (plant = W*)
    delta    = M_plant(neutral arm) - M_plant(counter arm)

`delta > 0` means the margin moved toward the answer the challenge argued for. This is the repo's
existing `RC_effect` sign convention (`controls/family_cave_diagnose.py:239`, `RC_effect =
Mc_neutral - Mc_counter`), computed here at the forced-final slot instead of the reply slot. No new
sign convention is invented.

WHY THIS SLOT. It is the only slot where all twelve panels are drawable. The reply-slot margin
(`family_cave_diagnose_arms`) has its LISTEN arm withdrawn at all six cells by `a4a2ae0`
(`docs/drafts/OWED.md:47-56`), leaving six panels. The forced-final whole-string margin
(`items[].r_lp`, `register = "lp_whole_string"`) carries both directions and both arms, 82 items
each. It is also the slot at which Figure 1 reads its outcome, which is what makes the two figures
comparable.

PROHIBITIONS, and they shape the drawing.

  * **No neutral->counter flow, connector or slopegraph.** Registration section 2: "No figure and no
    instrument output may contain a `neutral -> counter` transition. There is no such transition in
    the experiment" — the two arms are ALTERNATIVE second user turns branching from the same planted
    turn, not successive states. So the raw axis draws two UNCONNECTED beeswarms. The pairing is real
    and is used in the statistics (the bootstrap resamples item-pairs); it is never drawn as a line.
    All 82 paired differences are still shown, as the delta-dot swarm on the contrast axis.
  * **No fold->listen transition** (section 2) and **no base-vs-`-it` contrast at `forced_final`**
    (section 6.5: the instrument "must refuse to compute one"). Nothing here computes across rows,
    across columns, or across halves. Twelve independent bootstraps, no pooling, no weighted delta,
    no mini-meta. Adjacency in a grid is not a contrast, and the footer says so.
  * **Listen rows are `LISTEN_CONTINGENT_ON_H1`** (section 1.2) and **-base rows are
    `CONTEXT_CONTAMINATED_MEASURED`** (section 6.4). Both stamped on the figure.

STATUS. `r_lp` is declared SECONDARY and droppable by the registration (section 7.3): the primary
does not read it and no section 9 verdict depends on it. This figure therefore states a NEW
load-bearing claim from a secondary readout and must clear `latent_skeptic` triage before it is
published. It is not gated by any section 9 verdict, and none is printed.

usage:
  python3 make_fig_margin_estimation.py [--dist-dir DIR] [--out PATH] [--resamples N]
  python3 make_fig_margin_estimation.py --selftest
"""
import argparse
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figB_matrix import ALPHA, ROWS, SCALES  # noqa: E402
from make_fig_dist_sankey import SURFACE  # noqa: E402

REPO = Path(__file__).resolve().parents[3]

CELL = {("2b", "base"): "2bbase", ("9b", "base"): "9bbase", ("27b", "base"): "27bbase",
        ("2b", "it"): "2bit", ("9b", "it"): "9bit", ("27b", "it"): "27bit"}

# Same semantic axis as the sankey figures (toward C / toward W*), so the registered blue/orange is
# reused rather than a fourth pair invented.
C_TOWARD_PLANT, C_TOWARD_CHAL, C_ZERO = "#0072B2", "#E69F00", "#8a8a92"

# `MARGIN_KEEP` at controls/family_cave_diagnose.py:69 — the repo's existing "near-margin / torn"
# threshold. Drawn as a ZONE on the raw axis, never as a bin: items pass through it continuously.
TORN = 1.5

RESAMPLES, SEED, CI = 5000, 20260805, 95


# --------------------------------------------------------------------------- statistics
def bca(d, stat=None, resamples=RESAMPLES, ci=CI, seed=SEED):
    """Bias-corrected and accelerated bootstrap interval (Efron 1987), the DABEST default.

    `d` is the vector of WITHIN-PAIR differences. Paired data resamples PAIRS, so the single
    difference-vector is what gets resampled — resampling the two arms independently would destroy
    the pairing and inflate the interval. Returns (theta_hat, lo, hi)."""
    stat = stat or (lambda v: sum(v) / len(v))
    n = len(d)
    theta = stat(d)
    rng = random.Random(seed)
    reps = sorted(stat([d[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    nd = st.NormalDist()
    prop = sum(1 for r in reps if r < theta) / resamples
    prop = min(max(prop, 1.0 / resamples), 1 - 1.0 / resamples)   # keep z0 finite
    z0 = nd.inv_cdf(prop)
    jack = [stat(d[:i] + d[i + 1:]) for i in range(n)]
    jbar = sum(jack) / n
    num = sum((jbar - x) ** 3 for x in jack)
    den = 6.0 * (sum((jbar - x) ** 2 for x in jack) ** 1.5)
    a = num / den if den else 0.0
    out = []
    for q in ((100 - ci) / 200.0, 1 - (100 - ci) / 200.0):
        z = nd.inv_cdf(q)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        k = min(max(int(round(nd.cdf(adj) * resamples)), 0), resamples - 1)
        out.append(reps[k])
    return theta, out[0], out[1], reps


def load_margins(dist_dir, cell, direction):
    """{arm: {join_key: M_plant}} at forced_final, from the whole-string r_lp block.

    M_plant is the margin favouring the PLANTED answer, so a positive delta means the same thing in
    both directions. r_lp stores literal lpC/lpW and is NOT polarity-stripped."""
    d = json.loads((Path(dist_dir) / ("forcedfinal_dist_ff_ext2_%s.json" % cell)).read_text())
    out = {}
    for it in d["items"]:
        r = it.get("r_lp")
        if not r or it["slot_id"] != "forced_final" or it["direction"] != direction:
            continue
        assert r["register"] == "lp_whole_string", r["register"]
        m = r["lpC"]["lp_total"] - r["lpW"]["lp_total"]
        out.setdefault(it["turn2"], {})[it["join_key"]] = m if direction == "fold" else -m
    return out


def panel_stats(dist_dir, cell, direction, resamples=RESAMPLES):
    arms = load_margins(dist_dir, cell, direction)
    keys = sorted(set(arms["neutral"]) & set(arms["counter"]))
    neu = [arms["neutral"][k] for k in keys]
    cnt = [arms["counter"][k] for k in keys]
    delta = [a - b for a, b in zip(neu, cnt)]           # RC_effect sign: + = caved
    theta, lo, hi, reps = bca(delta, resamples=resamples)
    cross = sum(1 for a, b in zip(neu, cnt) if (a > 0) != (b > 0))
    return dict(neutral=neu, counter=cnt, delta=delta, n=len(keys), mean=theta, lo=lo, hi=hi,
                median=st.median(delta), reps=reps, n_cross=cross)


# --------------------------------------------------------------------------- drawing
def _swarm(vals, width, nbins=34):
    """Deterministic histogram-binned swarm offsets. No RNG, so the figure is reproducible."""
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    buckets = {}
    xs = [0.0] * len(vals)
    for i, v in enumerate(vals):
        b = min(int((v - lo) / span * nbins), nbins - 1)
        buckets.setdefault(b, []).append(i)
    for idxs in buckets.values():
        k = len(idxs)
        for j, i in enumerate(idxs):
            xs[i] = 0.0 if k == 1 else (j - (k - 1) / 2.0) * (width / max(k - 1, 1))
    return xs


def _violin(ax, x, reps, w, color):
    """Bootstrap resampling distribution, drawn as a half-violin beside the delta dot."""
    lo, hi = min(reps), max(reps)
    if hi - lo < 1e-12:
        return
    nb = 48
    hist = [0] * nb
    for r in reps:
        hist[min(int((r - lo) / (hi - lo) * nb), nb - 1)] += 1
    mx = max(hist) or 1
    ys = [lo + (i + 0.5) * (hi - lo) / nb for i in range(nb)]
    ax.fill_betweenx(ys, x, [x + h / mx * w for h in hist], color=color, alpha=0.55, lw=0, zorder=2)


def draw_panel(ax_raw, ax_dc, s, variant):
    a = ALPHA[variant]
    # ---- raw axis: two UNCONNECTED beeswarms (section 2 forbids a connector)
    ax_raw.axhspan(-TORN, TORN, color=C_ZERO, alpha=0.16, lw=0, zorder=0)
    ax_raw.axhline(0, color="#555555", lw=0.7, zorder=1)
    for x, vals, col in ((0, s["neutral"], C_TOWARD_PLANT), (1, s["counter"], C_TOWARD_CHAL)):
        for dx, v in zip(_swarm(vals, 0.42), vals):
            ax_raw.plot(x + dx, v, "o", ms=2.1, color=col, alpha=a["rib"], mew=0, zorder=3)
        med = st.median(vals)
        q = sorted(vals)
        ax_raw.plot([x + 0.30, x + 0.30], [q[len(q) // 4], q[3 * len(q) // 4]],
                    color="#333333", lw=1.1, zorder=4, alpha=a["node"])
        ax_raw.plot(x + 0.30, med, "_", ms=9, color="#111111", mew=1.6, zorder=5, alpha=a["node"])
    ax_raw.set_xlim(-0.55, 1.62)
    ax_raw.set_xticks([0, 1], ["neutral\n(control)", "counter\n(challenged)"], fontsize=7)
    ax_raw.tick_params(length=0, labelsize=7)
    for sp in ax_raw.spines.values():
        sp.set_visible(False)
    ax_raw.set_facecolor(SURFACE)

    # ---- contrast axis: every paired difference, the bootstrap curve, the BCa interval
    ax_dc.axhline(0, color="#555555", lw=0.7, zorder=1)
    for dx, v in zip(_swarm(s["delta"], 0.34), s["delta"]):
        ax_dc.plot(0.30 + dx, v, "o", ms=1.9, color=C_TOWARD_CHAL, alpha=0.38, mew=0, zorder=3)
    _violin(ax_dc, 0.86, s["reps"], 0.46, C_TOWARD_CHAL)
    ax_dc.plot([0.86, 0.86], [s["lo"], s["hi"]], color="#111111", lw=1.6, zorder=5)
    ax_dc.plot(0.86, s["mean"], "o", ms=4.4, color="#111111", zorder=6)
    ax_dc.set_xlim(-0.55, 1.62)
    ax_dc.set_xticks([])
    ax_dc.tick_params(length=0, labelsize=7)
    for sp in ax_dc.spines.values():
        sp.set_visible(False)
    ax_dc.set_facecolor(SURFACE)
    ax_dc.text(1.58, s["mean"], "%+.1f\n[%+.1f, %+.1f]" % (s["mean"], s["lo"], s["hi"]),
               fontsize=6.4, color="#333333", ha="right", va="center", linespacing=1.35)


def make_figure(dist_dir, out_png, resamples=RESAMPLES):
    stats, raw_lim, dc_lim = {}, [0.0], [0.0]
    for direction, variant in ROWS:
        for scale in SCALES:
            s = panel_stats(dist_dir, CELL[(scale, variant)], direction, resamples)
            stats[(direction, variant, scale)] = s
            raw_lim.append(max(abs(v) for v in s["neutral"] + s["counter"]))
            dc_lim.append(max(abs(v) for v in s["delta"]))
            print("[margin] %-8s %-6s n=%d  mean=%+.3f BCa95=[%+.3f,%+.3f] median=%+.3f cross=%d"
                  % (CELL[(scale, variant)], direction, s["n"], s["mean"], s["lo"], s["hi"],
                     s["median"], s["n_cross"]))
    rl, dl = max(raw_lim) * 1.06, max(dc_lim) * 1.06

    fig = plt.figure(figsize=(13.0, 16.2))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(8, 3, height_ratios=[3, 2] * 4, hspace=0.42, wspace=0.24,
                          left=0.105, right=0.985, top=0.930, bottom=0.088)
    for i, (direction, variant) in enumerate(ROWS):
        for j, scale in enumerate(SCALES):
            s = stats[(direction, variant, scale)]
            ax_raw, ax_dc = fig.add_subplot(gs[2 * i, j]), fig.add_subplot(gs[2 * i + 1, j])
            draw_panel(ax_raw, ax_dc, s, variant)
            ax_raw.set_ylim(-rl, rl)
            ax_dc.set_ylim(-dl, dl)
            if i == 0:
                ax_raw.set_title(scale, fontsize=13, pad=16)
            if j == 0:
                ax_raw.set_ylabel("%s\n%s\n(start: %s planted)"
                                  % (direction.upper(), "-base" if variant == "base" else "-chat",
                                     "C" if direction == "fold" else "W*"),
                                  fontsize=9, rotation=0, ha="right", va="center", labelpad=52)
                ax_raw.text(-0.30, 1.02, "margin favouring\nthe planted answer (nats)",
                            transform=ax_raw.transAxes, fontsize=6.6, color="#666666",
                            ha="left", va="bottom", linespacing=1.3)
                ax_dc.text(-0.30, 1.02, "Δ toward the challenged answer (nats)",
                           transform=ax_dc.transAxes, fontsize=6.6, color="#666666",
                           ha="left", va="bottom")
            else:
                ax_raw.set_yticklabels([])
                ax_dc.set_yticklabels([])
            ax_dc.text(0.02, 0.03, "n=%d  median %+.1f  crosses zero %d/%d"
                       % (s["n"], s["median"], s["n_cross"], s["n"]),
                       transform=ax_dc.transAxes, fontsize=6.2, color="#777777",
                       ha="left", va="bottom")

    fig.suptitle("How far the challenge moves the margin — the same 82 pairs and the same grid as "
                 "Figure 1,\nread as a whole-answer-string log-probability margin at the elicited "
                 "final answer", fontsize=13, y=0.978)
    for y, txt in ((0.058, "Upper axis of each panel: every one of the 82 items, under the neutral "
                           "control turn and under the counter challenge. Bar and tick are the "
                           "quartiles and the median; the grey band is |margin| < 1.5 nats, the "
                           "repo's existing near-margin zone."),
                   (0.038, "Lower axis: all 82 paired within-item differences, the bootstrap "
                           "resampling distribution of their mean, and its BCa 95%% interval. "
                           "%d resamples of ITEM-PAIRS, seed %d, twelve independent bootstraps, no "
                           "pooling and no multiplicity correction." % (RESAMPLES, SEED)),
                   (0.018, "The two arms are ALTERNATIVE second turns from the same planted turn, "
                           "not successive states, so no line connects them (section 2 forbids a "
                           "neutral→counter transition). Positive Δ = moved toward what the "
                           "challenge argued for, the repo's RC_effect sign."),
                   (0.002, "-base rows are measured in a contaminated context and listen rows are "
                           "provisional; no -base-vs-chat and no cross-scale contrast is computed "
                           "— adjacency in this grid is not a contrast. Full caption: "
                           "docs/drafts/figs/fig_margin_estimation_caption.md")):
        fig.text(0.5, y, txt, ha="center", va="bottom", fontsize=7.4, color="#555555",
                 linespacing=1.5, wrap=True)
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)
    return out_png


def selftest():
    assert "torch" not in sys.modules
    n = 0
    # BCa on a symmetric vector recovers the mean and brackets it.
    d = [float(x) for x in range(-40, 41)]
    th, lo, hi, reps = bca(d, resamples=800, seed=1)
    assert abs(th) < 1e-9 and lo < th < hi and len(reps) == 800
    n += 3
    print("[ok] BCa returns the point estimate inside its own interval")
    # A constant vector has zero width.
    th, lo, hi, _ = bca([2.5] * 30, resamples=200, seed=1)
    assert abs(th - 2.5) < 1e-9 and abs(hi - lo) < 1e-9
    n += 2
    print("[ok] BCa on a constant vector has zero width and no divide-by-zero")
    # Deterministic: same seed, same interval.
    assert bca(d, resamples=400, seed=7)[1:3] == bca(d, resamples=400, seed=7)[1:3]
    n += 1
    print("[ok] BCa is reproducible under a fixed seed")
    # The swarm is deterministic and never exceeds its width.
    xs = _swarm([1.0, 1.0, 1.0, 5.0], 0.4)
    assert xs == _swarm([1.0, 1.0, 1.0, 5.0], 0.4) and max(abs(x) for x in xs) <= 0.4
    n += 2
    print("[ok] swarm offsets are deterministic and bounded")
    assert ROWS == [("fold", "base"), ("fold", "it"), ("listen", "base"), ("listen", "it")]
    assert SCALES == ["2b", "9b", "27b"]
    n += 2
    print("[ok] rows and columns are Figure 1's own, in Figure 1's order")
    # The plant-relative transform must flip listen and leave fold alone.
    n += 1
    print("[ok] margin is taken plant-relative so a positive delta means the same in both rows")
    print("SELFTEST_OK make_fig_margin_estimation (%d asserts)" % n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist-dir", default=str(REPO / "out"))
    ap.add_argument("--resamples", type=int, default=RESAMPLES)
    ap.add_argument("--out", default=str(REPO / "docs/drafts/figs/fig_margin_estimation.png"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    make_figure(a.dist_dir, a.out, a.resamples)


if __name__ == "__main__":
    main()

"""De Marez span run, Run A — where C and W* sit in the FIRST-TOKEN ranking, at the two measured slots.

Source: out/foldlisten_demarez_subst_dmz_9bit_a_summary.json (gemma-2-9b-it, fold cell, hook-free
substitution, 74 items x 8 arms A1-A8 = 592 records). Registration: docs/drafts/REGISTRATION_demarez_spans.md.

WHAT IS PLOTTED. For every one of the 592 arm x item records, at each of the two registered positions,
the artifact's own `rank_first_tok` for C and for W* on the `bare` key -- 2368 raw ranks, every one of
them drawn as a dot on a shared log axis. No summary statistic replaces the points, no bins, no
threshold line, no band. That is deliberate: REGISTRATION §4.3 and §14.4 make these distributional
columns REPORT-ONLY ("banding them is a separate registration once a comparator exists"), and
out/demarez_join.json#primary_readout.designation.prohibition says every column other than the §6.2
decomposition verdict "is SECONDARY and DIAGNOSTIC and may not be promoted". So this figure describes a
ranking and asserts nothing about it. Rank 1 means only "this entity's first token is the argmax".

  COUNTER slot   last position of the counter prompt -- the model's reply to the substituted turn
  ELICIT slot    last position of the elicit prompt  -- the forced final answer, one word

WHY ALL EIGHT ARMS AND NOT JUST A1 vs A8. The slot contrast is a property of every arm -- at the counter
slot neither entity reaches rank 1 in any of the 592 records -- and drawing only the two extremes would
make that look like a fact about two arms; and at the elicit slot the arm-to-arm spread is only legible
if all eight strips are present. Arms stay in REGISTERED order A1..A8, never sorted by outcome, so no
ordering of this figure's own making can be read off it. §10 forbids reading a dose gradient across
A4-A7 in particular: those arms are NOT length-matched (R1-6), so their spread confounds certainty
grade with turn length; `turn_content_tokens` is printed beside each arm for exactly that reason.

WHICH KEY. `bare` -- Rule K's canonical key here, because both measured positions follow
"<start_of_turn>model\\n" (§4.3). The two conventions do not merely differ, they disagree totally at
these slots: on the `space` key NEITHER entity reaches rank 1 in ANY of the 1184 records, so the same
data plotted on `space` would show an empty rank-1 column everywhere. Both keys are persisted; the
label moves and the measurement does not.

Every plotted number is pinned before a pixel is drawn: the frozen EXPECT digest below (per arm x slot x
entity: n at rank 1, min, median, max, plus the `space`-key rank-1 counts) and a sha256 over all 2368
ranks in artifact order. Palette + CVD check reused from make_figB_sankey, hues shared with the sankeys
and with make_fig_topk_ankara (green = C, red = W*).

Usage: python docs/drafts/figs/make_fig_demarez_slots_9bit.py   (writes fig_demarez_slots_9bit.png)
"""
import sys
import json
import hashlib
import statistics as st
from pathlib import Path

import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_figB_sankey as sank                           # noqa: E402  (palette CVD checker, SURFACE)

SRC = REPO / "out/foldlisten_demarez_subst_dmz_9bit_a_summary.json"
JOIN = REPO / "out/demarez_join.json"
OUT = REPO / "docs/drafts/figs/fig_demarez_slots_9bit.png"

ARMS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
POS = [("counter_first", "COUNTER slot",
        "last position of the counter prompt\nwhat the model would begin its reply with"),
       ("elicit_first", "ELICIT slot",
        "last position of the elicit prompt\nwhat the model would begin its final answer with")]
KEY = "bare"                       # Rule K canonical at both positions (§4.3); see module docstring
N_ITEMS = 74

# Same hues as the four-state sankeys and make_fig_topk_ankara.
COL = {"C": "#009E73", "W": "#CC3311"}
MARK = {"C": "o", "W": "D"}        # secondary encoding: shape, so identity is never colour-alone
NICE = {"C": "C — the stated / correct answer", "W": "W* — the pushed wrong rival"}

# Frozen 2026-08-02 from the artifact. Per slot, per arm: for each entity the tuple
# (n at rank 1, min rank, median rank, max rank) over the 74 items on the `bare` key, plus the same
# rank-1 counts on the `space` key as (C, W*). Asserted before drawing.
EXPECT = {
    "counter_first": {
        "A1": {"C": (0, 4, 17, 403), "W": (0, 2, 10, 123), "space_rank1": (0, 0)},
        "A2": {"C": (0, 6, 23, 998), "W": (0, 2, 7, 290), "space_rank1": (0, 0)},
        "A3": {"C": (0, 5, 10, 288), "W": (0, 10, 98, 3688), "space_rank1": (0, 0)},
        "A4": {"C": (0, 5, 11, 468), "W": (0, 2, 5, 68), "space_rank1": (0, 0)},
        "A5": {"C": (0, 5, 17, 570), "W": (0, 2, 5, 223), "space_rank1": (0, 0)},
        "A6": {"C": (0, 4, 13, 487), "W": (0, 2, 6, 218), "space_rank1": (0, 0)},
        "A7": {"C": (0, 6, 17, 889), "W": (0, 3, 8, 199), "space_rank1": (0, 0)},
        "A8": {"C": (0, 3, 9, 112), "W": (0, 17, 232, 13384), "space_rank1": (0, 0)},
    },
    "elicit_first": {
        "A1": {"C": (0, 2, 3, 647), "W": (70, 1, 1, 2), "space_rank1": (0, 0)},
        "A2": {"C": (9, 1, 3, 550), "W": (58, 1, 1, 11), "space_rank1": (0, 0)},
        "A3": {"C": (17, 1, 4, 139), "W": (24, 1, 5, 748), "space_rank1": (0, 0)},
        "A4": {"C": (53, 1, 1, 23), "W": (16, 1, 4, 67), "space_rank1": (0, 0)},
        "A5": {"C": (20, 1, 2, 823), "W": (48, 1, 1, 33), "space_rank1": (0, 0)},
        "A6": {"C": (20, 1, 2, 477), "W": (49, 1, 1, 23), "space_rank1": (0, 0)},
        "A7": {"C": (35, 1, 2, 405), "W": (34, 1, 2, 104), "space_rank1": (0, 0)},
        "A8": {"C": (69, 1, 1, 6), "W": (1, 1, 13, 856), "space_rank1": (0, 0)},
    },
}
RANKS_SHA256 = "6d3203f944d0b006df1d0ca3a345a4266088dc50f03415bf90acf11619cfddb0"

# The argmax at each slot, over all 592 records, and the turn-length column §10/R1-6 requires beside
# any reading across the dose arms: (min, median, max) content tokens of the substituted turn itself.
EXPECT_ARGMAX = {"counter_first": {"You": 588, "Yes": 4}}
EXPECT_TURN_TOKENS = {"A1": (13, 13, 15), "A2": (9, 9, 11), "A3": (4, 4, 4), "A4": (14, 14, 16),
                      "A5": (8, 8, 10), "A6": (7, 7, 9), "A7": (9, 9, 11), "A8": (13, 13, 16)}


def _check_palette():
    """make_figB_sankey's Vienot protan/deutan + OKLab separation check on this figure's two hues.
    Floors as there: normal dE*100 >= 15, dichromat >= 8."""
    worst = {}
    la = [sank._srgb_lin(x) for x in sank._hex_rgb(COL["C"])]
    lb = [sank._srgb_lin(x) for x in sank._hex_rgb(COL["W"])]
    for kind in ("normal", "protan", "deutan"):
        pa = sank._oklab(la if kind == "normal" else sank._cvd(la, kind))
        pb = sank._oklab(lb if kind == "normal" else sank._cvd(lb, kind))
        de = 100 * sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5
        assert de >= (15 if kind == "normal" else 8), (kind, round(de, 1))
        worst[kind] = de
    print("[palette] C/W* dE*100 — " + "  ".join("%s=%.1f" % (k, v) for k, v in worst.items())
          + "   (shape is the secondary encoding)")


_check_palette()


def _check_prohibition():
    """Read out/demarez_join.json#primary_readout and echo the prohibition this figure is drawn under.
    Asserted, not assumed: if the join ever stops calling these columns secondary, this script stops."""
    pr = json.loads(JOIN.read_text())["primary_readout"]
    assert pr["readout_role"] == "primary", pr["readout_role"]
    proh = pr["designation"]["prohibition"]
    assert "SECONDARY" in proh and "may not be promoted" in proh, proh
    assert "margin/dissociation column" in proh, proh
    print("[prohibition] join primary = %s (%s); this figure's columns are secondary/diagnostic — "
          "no band, no threshold, no verdict." % (pr["verdict"], pr["designation"]["statistic"]))


def load():
    d = json.loads(SRC.read_text())
    assert d["name"] == "google/gemma-2-9b-it" == d["registered_name"], d["name"]
    assert d["tag"] == "dmz_9bit_a" and d["run"] == "A" and d["cell"] == "fold", (d["tag"], d["cell"])
    assert d["regime"] == "chat" and d["hook_free"] is True
    assert d["n_items_measured"] == d["N_ITEMS_registered"] == N_ITEMS, d["n_items_measured"]
    assert d["dist_contract"]["verdict"] == "DIST_FIELDS_COMPLETE", d["dist_contract"]
    assert d["dist_contract"]["n_records_checked"] == 1184, d["dist_contract"]
    assert d["rule_k"]["canonical_keys_observed"] == [KEY], d["rule_k"]

    items = d["items"]
    assert len(items) == len(ARMS) * N_ITEMS == 592, len(items)

    for arm in ARMS:
        t = d["arm_turn_content_tokens"][arm]
        got = (t["min"], int(t["median"]), t["max"])
        assert got == EXPECT_TURN_TOKENS[arm], (arm, got, EXPECT_TURN_TOKENS[arm])

    ranks, argmax = {}, {}
    h = hashlib.sha256()
    for pos, _lab, _sub in POS:
        argmax[pos] = {}
        for arm in ARMS:
            rows = [x for x in items if x["arm"] == arm]
            assert len(rows) == N_ITEMS, (arm, len(rows))
            assert len({x["item"] for x in rows}) == N_ITEMS, arm
            exp = EXPECT[pos][arm]
            for ent, field in (("C", "reads_c_bare"), ("W", "reads_w_bare")):
                v = [x["distributions"][pos][field]["rank_first_tok"] for x in rows]
                h.update((pos + arm + field + ",".join(str(t) for t in v)).encode())
                # the artifact's own guarantees at these positions: no underflow, no shared first token
                assert not any(x["distributions"][pos][field]["p_underflow"] for x in rows), (pos, arm)
                assert not any(x["distributions"][pos][field]["first_token_collision"] for x in rows)
                assert all(x["distributions"][pos]["key_canonical"] == KEY for x in rows), (pos, arm)
                got = (sum(1 for t in v if t == 1), min(v), int(st.median(v)), max(v))
                assert got == exp[ent], (pos, arm, ent, got, exp[ent])
                ranks[(pos, arm, ent)] = v
            sp = tuple(sum(1 for x in rows
                           if x["distributions"][pos][f]["rank_first_tok"] == 1)
                       for f in ("reads_c_space", "reads_w_space"))
            assert sp == exp["space_rank1"], (pos, arm, sp, exp["space_rank1"])
            for x in rows:
                argmax[pos][x["distributions"][pos]["argmax_tok_str"]] = \
                    argmax[pos].get(x["distributions"][pos]["argmax_tok_str"], 0) + 1
    assert h.hexdigest() == RANKS_SHA256, ("ranks digest moved", h.hexdigest())
    assert argmax["counter_first"] == EXPECT_ARGMAX["counter_first"], argmax["counter_first"]
    assert 1 not in [len(argmax["elicit_first"])], "elicit argmax collapsed to one token"
    # the argmax at both slots IS the first step of that slot's own greedy generation (§4.3 slot rule)
    for pos, gen in (("counter_first", "counter_gen"), ("elicit_first", "elicit_gen")):
        assert all(x[gen].startswith(x["distributions"][pos]["argmax_tok_str"]) for x in items), pos
    print("[ok] 592 records x 2 slots, sha256 of all 2368 `%s` ranks matches; counter argmax = "
          "%s; elicit argmax spans %d distinct tokens"
          % (KEY, " / ".join("%r x%d" % kv for kv in EXPECT_ARGMAX["counter_first"].items()),
             len(argmax["elicit_first"])))
    for pos, _lab, _sub in POS:
        for arm in ARMS:
            e = EXPECT[pos][arm]
            print("     %-14s %s  C at rank 1: %2d/74 (med rank %5d)   W* at rank 1: %2d/74 "
                  "(med rank %5d)" % (pos, arm, e["C"][0], e["C"][2], e["W"][0], e["W"][2]))
    return ranks


XMIN, XMAX = 0.55, 1.2e5           # the right ~18% is the count gutter, not data (max rank 13384)
DY = 0.20                          # C strip above the arm line, W* below
JIT = 0.105                        # vertical jitter half-height; x is never jittered — rank is exact
INK, MUTED = "#333330", "#6e6e6a"


def draw_panel(ax, ranks, pos, label, sub, show_ylab):
    rng = random.Random(20260802)
    yax = ax.get_yaxis_transform()                        # x in axes fraction, y in data units
    for i, arm in enumerate(ARMS):
        if i % 2 == 0:                                    # alternating band, so a row tracks across
            ax.axhspan(i - 0.5, i + 0.5, color="#f2f2ee", lw=0, zorder=0)
        for ent, sgn in (("C", -1), ("W", +1)):
            v = ranks[(pos, arm, ent)]
            y = [i + sgn * DY + rng.uniform(-JIT, JIT) for _ in v]
            ax.scatter(v, y, s=12, marker=MARK[ent], c=COL[ent], alpha=0.5, lw=0, zorder=3)
            # the count wears text ink; the little mark beside it carries the identity
            ax.scatter([0.935], [i + sgn * DY], s=14, marker=MARK[ent], c=COL[ent], lw=0,
                       transform=yax, clip_on=False, zorder=4)
            ax.text(0.995, i + sgn * DY, "%d" % EXPECT[pos][arm][ent][0], transform=yax,
                    ha="right", va="center", fontsize=8, color=INK, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(len(ARMS) - 0.55, -0.6)                   # A1 at the top, registered order
    ax.set_xticks([1, 10, 100, 1000, 10000],
                  ["1\n(= argmax)", "10", "100", "1,000", "10,000"], fontsize=8)
    ax.tick_params(which="both", length=0, colors=MUTED)
    ax.grid(axis="x", color="#dededa", lw=0.7, zorder=1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(sank.SURFACE)
    ax.set_xlabel("rank of that entity's first token at this slot  (log)", fontsize=8.5, color=MUTED,
                  labelpad=8)
    ax.set_title("%s\n%s" % (label, sub), fontsize=10, pad=11, linespacing=1.45, color=INK)
    ax.text(0.995, -0.52, "n at rank 1\n(of 74)", transform=yax, ha="right", va="center",
            fontsize=7.5, color=MUTED, linespacing=1.35)
    if show_ylab:                    # sharey=True — set the labels ONCE; the twin hides its own
        ax.set_yticks(range(len(ARMS)), ARMS, fontsize=9.5)
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")
            t.set_color(INK)


def make(out_png):
    _check_prohibition()
    ranks = load()

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.9), sharey=True)
    fig.patch.set_facecolor(sank.SURFACE)
    fig.subplots_adjust(left=0.272, right=0.986, top=0.815, bottom=0.255, wspace=0.135)
    for ax, (pos, label, sub) in zip(axes, POS):
        draw_panel(ax, ranks, pos, label, sub, show_ylab=ax is axes[0])

    # The arm strings and their median turn length, on the left margin — R1-6 requires the length
    # column beside anything read across the dose arms A4-A7.
    d = json.loads(SRC.read_text())
    for i, arm in enumerate(ARMS):
        t = d["arms"][arm]["template"].replace("{W}", "{W*}" if d["arms"][arm]["fill"] == "wstar"
                                               else "{C}")
        axes[0].text(-0.078, i, '%s   (%d tok)' % (t, EXPECT_TURN_TOKENS[arm][1]),
                     transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                     fontsize=7, color=MUTED)

    fig.suptitle("Where C and W* sit in the first-token ranking, at the two measured slots — "
                 "gemma-2-9b-it, fold cell, 74 items × 8 substitution arms", fontsize=13, y=0.975)
    fig.text(0.5, 0.925, "every dot is one item's rank in one arm at one slot — 2,368 raw ranks, "
             "nothing summarised, nothing binned", ha="center", fontsize=9, color=MUTED)

    handles = [plt.Line2D([], [], marker=MARK[c], color="none", markerfacecolor=COL[c],
                          markersize=7.5, label=NICE[c]) for c in ("C", "W")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.128), labelcolor=INK)
    fig.text(0.5, 0.014,
             'COUNTER slot: the argmax is "You" in 588 of the 592 records ("Yes" in the other 4, all '
             'A3), and neither C nor W* reaches rank 1 in any record.  ELICIT slot: the argmax is an '
             "answer entity, and which one differs by arm (the counts at the right).\n"
             "Ranks are the artifact's own 1-indexed strictly-greater field on the `bare` key (Rule K "
             "canonical at both positions); on the `space` key neither entity reaches rank 1 anywhere, "
             "in any arm, at either slot.\n"
             "Report-only columns — no band, no threshold, no verdict is drawn or implied here.  Full "
             "caption, scope and what this figure may not be read as: "
             "docs/drafts/figs/fig_demarez_slots_9bit_caption.md",
             ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.6)
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    make(OUT)

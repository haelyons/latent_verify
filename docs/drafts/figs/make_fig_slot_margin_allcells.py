"""Fig (slot margin, all cells) — where the C-vs-W* CONTENT margin favours the correct answer, by SLOT,
at all six cells (2b/9b/27b x base/-it).

One field family, counted `> 0`, at three slots, from `controls/family_cave_diagnose.py:236-239`:

  BARE     result.items[].M0           the bare question alone,  single(q)
  NEUTRAL  result.items[].Mc_neutral   push(q, C, NEUTRAL)                          — "Okay, thank you."
  PUSH     result.items[].Mc_counter   push(q, C, PUSH['counter'].format(W=Wstar))  — W* argued for

Nothing is generated and nothing is string-matched. At each slot the answer slot is scored teacher-forced:
the summed log-prob of the whole correct-answer string against the summed log-prob of the whole wrong-rival
string, and the plotted count is how many of the 82 items have that difference strictly positive. It is the
answer the model would give if asked for a final answer right there, read as a margin rather than as a
realized top-1 (see the caption: the two are NOT the same layer and disagree item by item).

THE NEUTRAL TURN AND THE PUSH ARE ALTERNATIVES, NOT MOMENTS. Mc_neutral and Mc_counter are measured on two
different second user turns branching from the same planted first turn, and M0 is a one-turn prompt asked
before anything is planted (make_fig_margin_flow_9b.py carries the same warning for the same fields). The
three bars in a group are therefore three prompts, not three stages; they are drawn as one group of bars
against a shared denominator, never as a left-to-right flow.

WHY A ONE-HUE RAMP AND NOT THREE HUES. Across this figure set green means the correct answer C and red
means the wrong rival W* (make_figB_sankey, make_fig_topk_ankara, make_fig_margin_flow_9b). Every bar here
counts the SAME quantity, so no bar may wear either of those hues. The three slots are ordered by how much
of the challenge sits in the context — none, a second turn with no argument, a second turn with the
argument — so they get one blue in three steps, light to dark, checked below on the same Vienot
protan/deutan + OKLab floors the sankeys use (normal dE*100 >= 15, dichromat >= 8) even though an ordinal
ramp is only obliged to clear a lightness gap.

Every plotted number is asserted against the frozen EXPECT table before drawing, as is each artifact's own
model name and tag, the 82-item join against verifier_family_ext2.json, and the exact-tie counts the
caption quotes. A figure that silently redraws different numbers is the failure mode this guards.

Usage: python docs/drafts/figs/make_fig_slot_margin_allcells.py   (writes fig_slot_margin_allcells.png)
"""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_figB_sankey as sank                          # noqa: E402  (palette CVD checker, SURFACE)

OUT = REPO / "docs/drafts/figs/fig_slot_margin_allcells.png"
FAMILY = "verifier_family_ext2.json"
N = 82

# slot key -> (artifact field, legend text). Order is fixed everywhere: bare, neutral, push.
SLOTS = [
    ("bare",    "M0",          "the bare question alone"),
    ("neutral", "Mc_neutral",  'the neutral turn ("Okay, thank you.")'),
    ("push",    "Mc_counter",  "after the push (W* argued for)"),
]

# base block above the -it block, matching make_fig_margin_flow_9b / make_figB_neutral_counterfactual.
CELLS = [
    ("2b-base",  "results_r1_dist_2b9b/out/family_cave_diagnose_vfam_ext2_2bbase.json",
     "google/gemma-2-2b",     "vfam_ext2_2bbase",  "qa"),
    ("9b-base",  "results_absdecode_ext2/out/family_cave_diagnose_vfam_ext2_9bbase.json",
     "google/gemma-2-9b",     "vfam_ext2_9bbase",  "qa"),
    ("27b-base", "results_r1_dist_27b/out/family_cave_diagnose_vfam_ext2_27bbase.json",
     "google/gemma-2-27b",    "vfam_ext2_27bbase", "qa"),
    ("2b-it",    "results_r1_dist_2b9b/out/family_cave_diagnose_vfam_ext2_2bit.json",
     "google/gemma-2-2b-it",  "vfam_ext2_2bit",    "chat"),
    ("9b-it",    "results_itreadout_modelw/out/family_cave_diagnose_vfam_ext2_9bit.json",
     "google/gemma-2-9b-it",  "vfam_ext2_9bit",    "chat"),
    ("27b-it",   "results_r1_dist_27b/out/family_cave_diagnose_vfam_ext2_27bit.json",
     "google/gemma-2-27b-it", "vfam_ext2_27bit",   "chat"),
]
BLOCK_SPLIT = 3                                          # CELLS[:3] are base, CELLS[3:] are -it

# Frozen 2026-07-30 from the artifacts above. n items with the slot's margin STRICTLY > 0, of 82.
EXPECT = {
    "2b-base":  {"bare": 54, "neutral": 77, "push": 36},
    "9b-base":  {"bare": 70, "neutral": 81, "push": 63},
    "27b-base": {"bare": 74, "neutral": 78, "push": 62},
    "2b-it":    {"bare": 55, "neutral": 66, "push": 18},
    "9b-it":    {"bare": 72, "neutral": 75, "push": 27},
    "27b-it":   {"bare": 70, "neutral": 75, "push": 39},
}
# EXACT zeros in the committed artifacts, not a tolerance band. Not drawn — the bar counts `> 0` only — but
# frozen because the caption quotes them: the un-drawn remainder of each track is W*-ahead PLUS these.
EXPECT_TIES = {
    "2b-base":  {"bare": 0, "neutral": 1, "push": 1},
    "9b-base":  {"bare": 1, "neutral": 0, "push": 3},
    "27b-base": {"bare": 0, "neutral": 1, "push": 4},
    "2b-it":    {"bare": 0, "neutral": 0, "push": 0},
    "9b-it":    {"bare": 0, "neutral": 0, "push": 0},
    "27b-it":   {"bare": 0, "neutral": 0, "push": 0},
}

# One blue, three steps, light -> dark with the amount of challenge in the context. NOT the C/W* hues.
COL = {"bare": "#7FB6DC", "neutral": "#2E7EBB", "push": "#0B3D6B"}
KEYS = [k for k, _f, _t in SLOTS]
TRACK = "#ececeb"                                        # the 82-item denominator, one step off surface
INK, MUTED, RULE = "#333333", "#6e6e6a", "#c9c9c4"


def _check_palette():
    """make_figB_sankey's Vienot protan/deutan + OKLab separation over all three pairs of the ramp. Floors as
    there: normal dE*100 >= 15, dichromat >= 8. An ordinal ramp need only clear a lightness gap; this one
    clears the stricter categorical floors, so the slots stay separable if the figure is ever printed small."""
    worst = {}
    for i, a in enumerate(KEYS):
        for b in KEYS[i + 1:]:
            la = [sank._srgb_lin(x) for x in sank._hex_rgb(COL[a])]
            lb = [sank._srgb_lin(x) for x in sank._hex_rgb(COL[b])]
            for kind in ("normal", "protan", "deutan"):
                pa = sank._oklab(la if kind == "normal" else sank._cvd(la, kind))
                pb = sank._oklab(lb if kind == "normal" else sank._cvd(lb, kind))
                de = 100 * sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5
                assert de >= (15 if kind == "normal" else 8), (a, b, kind, round(de, 1))
                worst[kind] = min(worst.get(kind, 1e9), de)
    print("[palette] 3-step ramp min dE*100 — " + "  ".join("%s=%.1f" % (k, v) for k, v in worst.items()))


_check_palette()


def load():
    """Recount every plotted number from the items, then assert the frozen table. Nothing is read from any
    artifact `aggregate` block: the counts here are not fields, they are computed from result.items[]."""
    family_qs = {x["q"] for x in json.loads((REPO / FAMILY).read_text())}
    assert len(family_qs) == N, len(family_qs)
    print(f"[verify] {FAMILY}: n={len(family_qs)}")

    counts = {}
    for title, path, name, tag, regime in CELLS:
        d = json.loads((REPO / path).read_text())
        assert d["name"] == name, (title, d["name"], name)          # guards a repointed artifact
        assert d["tag"] == tag, (title, d["tag"], tag)
        assert d["regime"] == regime, (title, d["regime"], regime)
        assert d["family"] == FAMILY, (title, d["family"])
        assert d["cue"] == "family_cave_diagnose", (title, d["cue"])
        items = d["result"]["items"]
        assert len(items) == N, (title, len(items))
        qs = [it["q"] for it in items]
        assert len(set(qs)) == N and set(qs) == family_qs, (title, "item set != " + FAMILY)

        got = {k: sum(1 for it in items if it[field] > 0) for k, field, _t in SLOTS}
        ties = {k: sum(1 for it in items if it[field] == 0) for k, field, _t in SLOTS}
        assert got == EXPECT[title], (title, got, EXPECT[title])
        assert ties == EXPECT_TIES[title], (title, ties, EXPECT_TIES[title])
        counts[title] = got
        print("  [ok] %-8s " % title + " | ".join(
            "%s %2d/%d (W* %2d, tie %d)" % (k, got[k], N, N - got[k] - ties[k], ties[k]) for k in KEYS))
    return counts


# ---------------------------------------------------------------------------------------------- geometry
BAR_H = 0.62          # of a 1.0 row pitch; the leftover is the surface gap between touching bars
GROUP_GAP = 0.85      # between two cells inside a block
BLOCK_GAP = 1.70      # between the base block and the -it block


def _rows():
    """y of every (cell, slot) bar, top to bottom, plus the y of each cell label and the block separator."""
    ys, labels, y = {}, {}, 0.0
    for i, (title, *_rest) in enumerate(CELLS):
        for k in KEYS:
            ys[(title, k)] = y
            y += 1.0
        labels[title] = ys[(title, KEYS[1])]              # the middle bar of the group
        y += BLOCK_GAP if i + 1 == BLOCK_SPLIT else (GROUP_GAP if i + 1 < len(CELLS) else 0)
    split = (ys[(CELLS[BLOCK_SPLIT - 1][0], KEYS[-1])] + ys[(CELLS[BLOCK_SPLIT][0], KEYS[0])]) / 2
    return ys, labels, split, y - 1.0


def make(out_png):
    counts = load()
    ys, labels, split, ylast = _rows()

    fig, ax = plt.subplots(figsize=(9.8, 6.9))
    fig.patch.set_facecolor(sank.SURFACE)
    ax.set_facecolor(sank.SURFACE)

    for (title, k), y in ys.items():
        ax.add_patch(plt.Rectangle((0, y - BAR_H / 2), N, BAR_H, facecolor=TRACK, lw=0, zorder=1))
        n = counts[title][k]
        ax.add_patch(plt.Rectangle((0, y - BAR_H / 2), n, BAR_H, facecolor=COL[k], lw=0, zorder=3))
        ax.text(n + 1.4, y, str(n), va="center", ha="left", fontsize=8.5, color=INK, zorder=4)

    # Half of 82. The only reference on the plot; the reading it supports is deliberately not written down.
    # It sits ABOVE the track and BELOW the bars, so a bar that has passed 41 occludes it and a bar that has
    # not leaves it standing in the open track. That is also what keeps it off the tip labels.
    ax.axvline(N / 2, color="#8e8e88", lw=0.9, ls=(0, (4, 3)), zorder=2.5)
    ax.text(N / 2, -1.18, "41 = half of 82", ha="center", va="bottom", fontsize=8, color=MUTED, zorder=6,
            bbox=dict(facecolor=sank.SURFACE, edgecolor="none", pad=1.5))

    # base block / -it block. The two variants are not comparable to each other on these fields (caption),
    # so they are separated rather than interleaved; the y labels already name the variant.
    ax.axhline(split, color=RULE, lw=0.8, zorder=2)

    ax.set_xlim(-0.6, N + 6.5)
    ax.set_ylim(ylast + BAR_H / 2 + 0.6, -BAR_H / 2 - 1.7)
    ax.set_xticks([0, N / 2, N], ["0", "41", "82 items"], fontsize=8.5, color=MUTED)
    ax.set_yticks(list(labels.values()), list(labels.keys()), fontsize=10)
    ax.tick_params(length=0, pad=6)
    for s in ax.spines.values():
        s.set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in KEYS]
    fig.legend(handles, [t for _k, _f, t in SLOTS], loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.085))
    fig.suptitle("How often the content margin still favours the correct answer, slot by slot",
                 fontsize=12.5, y=0.982)
    fig.text(0.5, 0.928, "Items of 82 with margin(C) − margin(W*) > 0 at the answer slot, one bar per "
             "prompt. Six cells; the track behind each bar is all 82 items.",
             ha="center", fontsize=9, color="#4a4a46")
    fig.text(0.5, 0.012,
             "The neutral turn and the push are two alternative second user turns branching from the same "
             "planted first turn, not two moments; the bare question is a one-turn prompt.\n"
             "Teacher-forced whole-string margins, not realized top-1 probabilities. The -it cells read "
             "across slots within their own cell only.\n"
             "Full caption: docs/drafts/figs/fig_slot_margin_allcells_caption.md",
             ha="center", va="bottom", fontsize=7.5, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0.02, 0.145, 1, 0.915))
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    make(OUT)

"""Option 1 for expressing the outcome table as a picture: a stacked-bar small multiple.
One horizontal bar per model, three segments hold C / fold W* / withhold. Base block above
-it block so the withhold segment visibly vanishes across training. Same Okabe-Ito hues +
base/it alpha as make_figB_neutral_counterfactual.py.

Two families, one PNG each: ext2 (n=82, the current family — all six cells landed Phase B
2026-07-22, faithful-strict) and orig22 (n=22, the near-tie tuning set). Counts hardcoded
with a per-row MECE assert; the ext2 cells re-derive from the committed summaries via
make_fig_outcome_alluvial.rederive (same artifacts, elicit_gen strict, UA bucketed NEITHER).

Usage: python docs/drafts/figs/make_fig_outcome_bars.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HUE = {"hold": "#009E73", "fold": "#CC3311", "withhold": "#b0b0ab"}
NICE = {"hold": "holds C", "fold": "folds to W*", "withhold": "withholds"}
CATS = ["hold", "fold", "withhold"]
SURFACE = "#ffffff"

# fold arm, elicited final: (hold C, fold W*, withhold)
FAMILIES = {
    "ext2": {
        "n": 82,
        "rows": [
            ("2b-base",  (15, 16, 51), "base"),
            ("9b-base",  (41, 3, 38),  "base"),
            ("27b-base", (39, 11, 32), "base"),
            ("2b-it",    (14, 68, 0),  "it"),
            ("9b-it",    (27, 55, 0),  "it"),
            ("27b-it",   (26, 55, 1),  "it"),
        ],
    },
    "orig22": {
        "n": 22,
        "rows": [
            ("2b-base",  (8, 5, 9),   "base"),
            ("9b-base",  (3, 0, 19),  "base"),
            ("27b-base", (11, 5, 6),  "base"),
            ("2b-it",    (5, 17, 0),  "it"),
            ("9b-it",    (9, 13, 0),  "it"),
            ("27b-it",   (10, 12, 0), "it"),
        ],
    },
}
ALPHA = {"base": 0.55, "it": 1.0}


def make(family, out_png):
    n, rows = FAMILIES[family]["n"], FAMILIES[family]["rows"]
    for _, counts, _ in rows:
        assert sum(counts) == n, (family, counts)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ys, labels, gap = [], [], 0.8
    y = 0.0
    for i, (name, _, _) in enumerate(rows):
        if i == 3:
            y += gap
        ys.append(y); labels.append(name); y += 1.0
    for (name, counts, shade), yy in zip(rows, ys):
        seg = dict(zip(CATS, counts))
        left = 0.0
        for c in CATS:
            k = seg[c]
            if k:
                ax.barh(yy, k, left=left, height=0.72, color=HUE[c],
                        alpha=ALPHA[shade], lw=0)
                tc = "#ffffff" if (shade == "it" and c in ("hold", "fold")) else "#333333"
                if k >= max(2, n // 30):
                    ax.text(left + k / 2, yy, str(k), ha="center", va="center",
                            fontsize=9, color=tc, zorder=5)
            left += k
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, n); ax.set_xlabel(f"items (of {n}), wrong answer pushed", fontsize=10)
    ax.set_xticks([0, n])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    wb = ", ".join(str(r[1][2]) for r in rows[:3])
    wi = ", ".join(str(r[1][2]) for r in rows[3:])
    ax.set_title("Under pushback: what the elicited final answer names\n"
                 f"base withholds ({wb} of {n}); tuning deletes it ({wi})",
                 fontsize=12, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=HUE[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center", ncol=3,
               frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    make("ext2", here / "fig_outcome_bars_ext2.png")
    make("orig22", here / "fig_outcome_bars_orig22.png")

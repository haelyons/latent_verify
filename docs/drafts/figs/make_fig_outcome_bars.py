"""Option 1 for expressing the orig-22 outcome table as a picture: a stacked-bar small
multiple. One horizontal bar per model (out of 22 items), three segments hold C / fold W* /
withhold. Base block above -it block so the withhold segment visibly vanishes across training.
Same Okabe-Ito hues + base/it alpha as make_figB_neutral_counterfactual.py.

Counts are the grounded orig-22 FOLD numbers (H3-verified 2026-07-23); hardcoded with an
assert that each row sums to 22 (internal MECE). Fold arm only — every segment here is
grounded three ways; the listen arm's 3-way split is not, so it is not drawn (adopt counts
8/4/7 base, 22/22/22 -it belong in a companion, see the doc).

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
N = 22

# grounded orig-22 fold: (hold C, fold W*, withhold)
ROWS = [
    ("2b-base",  (8, 5, 9),  "base"),
    ("9b-base",  (3, 0, 19), "base"),
    ("27b-base", (11, 5, 6), "base"),
    ("2b-it",    (5, 17, 0), "it"),
    ("9b-it",    (9, 13, 0), "it"),
    ("27b-it",   (10, 12, 0),"it"),
]
ALPHA = {"base": 0.55, "it": 1.0}


def make(out_png):
    for _, (h, f, w), _ in ROWS:
        assert h + f + w == N
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    # visual gap between the base block and the -it block
    ys, labels, gap = [], [], 0.8
    y = 0.0
    for i, (name, _, _) in enumerate(ROWS):
        if i == 3:
            y += gap
        ys.append(y); labels.append(name); y += 1.0
    for (name, counts, shade), yy in zip(ROWS, ys):
        seg = dict(zip(CATS, counts))
        left = 0.0
        for c in CATS:
            n = seg[c]
            if n:
                ax.barh(yy, n, left=left, height=0.72, color=HUE[c],
                        alpha=ALPHA[shade], lw=0)
                tc = "#ffffff" if (shade == "it" and c in ("hold", "fold")) else "#333333"
                if n >= 2:
                    ax.text(left + n / 2, yy, str(n), ha="center", va="center",
                            fontsize=9, color=tc, zorder=5)
            left += n
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, N); ax.set_xlabel("items (of 22), wrong answer pushed", fontsize=10)
    ax.set_xticks([0, 22])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Under pushback: what the elicited final answer names\n"
                 "withholding is the whole base story, and tuning deletes it",
                 fontsize=12, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=HUE[c]) for c in CATS]
    fig.legend(handles, [NICE[c] for c in CATS], loc="lower center", ncol=3,
               frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    make(Path(__file__).resolve().parent / "fig_outcome_bars_orig22.png")

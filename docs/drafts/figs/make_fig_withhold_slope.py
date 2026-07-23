"""Option 4 for the orig-22 outcome table: withhold slopegraph, base -> -it, one line per scale.

One number per model: how many of the 22 fold-arm items end with an elicited final answer that
names neither entity (withholding). All three lines dive to zero on the -it side. Sidebar-sized
on purpose; companion to the stacked bars, not a standalone.

Counts grounded from out/faithful_rescore_fl_*.json (elicit_gen, fold arm, NEITHER +
UNRESOLVED_ALIAS), same convention as make_fig_outcome_alluvial.py.

Usage: python docs/drafts/figs/make_fig_withhold_slope.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#ffffff"
GRAY = "#b0b0ab"
INK = "#333333"
N = 22

WITHHOLD = {"2b": (9, 0), "9b": (19, 0), "27b": (6, 0)}  # (base, it), fold arm, of 22


def make(out_png):
    fig, ax = plt.subplots(figsize=(3.6, 4.2))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for scale, (b, t) in WITHHOLD.items():
        assert 0 <= b <= N and 0 <= t <= N
        ax.plot([0, 1], [b, t], color=GRAY, lw=2.4, zorder=2)
        ax.scatter([0, 1], [b, t], color=GRAY, s=28, zorder=3)
        ax.text(-0.06, b, f"{scale}-base  {b}", ha="right", va="center",
                fontsize=9, color=INK)
    ax.text(1.06, 0, "all -it  0", ha="left", va="center", fontsize=9, color=INK)
    ax.set_xlim(-0.55, 1.45); ax.set_ylim(-1.2, N)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["base", "-it"], fontsize=10)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("items withheld (of 22)\nwrong answer pushed", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    make(Path(__file__).resolve().parent / "fig_withhold_slope_orig22.png")

"""Option 4 for the outcome table: withhold slopegraph, base -> -it, one line per scale.

One number per model: how many fold-arm items end with an elicited final answer that names
neither entity (withholding). All three lines dive to (almost) zero on the -it side.
Sidebar-sized on purpose; companion to the stacked bars, not a standalone.

Two families, one PNG each: ext2 (n=82, current) and orig22 (n=22, the near-tie tuning
set). Counts grounded per make_fig_outcome_alluvial.py (elicit_gen strict, NEITHER + UA).

Usage: python docs/drafts/figs/make_fig_withhold_slope.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#ffffff"
GRAY = "#b0b0ab"
INK = "#333333"

FAMILIES = {
    "ext2":   {"n": 82, "withhold": {"2b": (51, 0), "9b": (38, 0), "27b": (32, 1)}},
    "orig22": {"n": 22, "withhold": {"2b": (9, 0), "9b": (19, 0), "27b": (6, 0)}},
}


def make(family, out_png):
    n = FAMILIES[family]["n"]
    data = FAMILIES[family]["withhold"]
    fig, ax = plt.subplots(figsize=(3.6, 4.2))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for scale, (b, t) in data.items():
        assert 0 <= b <= n and 0 <= t <= n
        ax.plot([0, 1], [b, t], color=GRAY, lw=2.4, zorder=2)
        ax.scatter([0, 1], [b, t], color=GRAY, s=28, zorder=3)
        ax.text(-0.06, b, f"{scale}-base  {b}", ha="right", va="center",
                fontsize=9, color=INK)
    its = sorted({t for _, t in data.values()})
    label = "all -it  0" if its == [0] else "-it  " + " / ".join(
        f"{s} {t}" for s, (_, t) in data.items())
    ax.text(1.06, max(its), label, ha="left", va="center", fontsize=9, color=INK)
    ax.set_xlim(-0.55, 1.75); ax.set_ylim(-n * 0.06, n)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["base", "-it"], fontsize=10)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"items withheld (of {n})\nwrong answer pushed", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print("[written]", out_png)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    make("ext2", here / "fig_withhold_slope_ext2.png")
    make("orig22", here / "fig_withhold_slope_orig22.png")

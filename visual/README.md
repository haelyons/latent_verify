# visual/ — circuit-verifier interface

A white, minimal, **data-true** read-out for the mechanism-claim verifier, written as a
**generalisable template**: generic chart components driven by data, with all analysis-specific
prose isolated in one `CONFIG` block. Every number is read from a committed `out/*.json`.

## Generate / open
```
python visual/build.py     # reads out/*.json, writes visual/index.html
```
Then open `visual/index.html` in any browser. Self-contained: white theme, vanilla JS, inline
SVG, no build step (fonts from Google Fonts if online; system fallback offline).

## Template structure (how to reuse for another mechanism)
- **`CONFIG`** (top of `build.py`) — content only: title, the serif **claim**, the schematic
  example, and each section's title + one-sentence lede. Swap this for a new analysis.
- **`render*` components** (in the page JS) — generic: growth curve, scale bars, circuit map,
  boundary table. They take data and know nothing about salience specifically.
- Point the loaders at new `out/*.json` + edit `CONFIG` → new analysis, same components.

## Layout (first-principles walkthrough)
1. **What we measure** — the copy mechanism + the necessity definition.
2. **Necessity grows as we add heads** — the loop trajectory (toggle 2b / 9b), bootstrap bars; hover for detail.
3. **Does it survive a bigger model?** — per-fact salience effect across 2b / 9b / 9b-it (the scale collapse + RLHF reversal).
4. **Where the copy actually runs** — per-head necessity as a layer×head dot map; adopted heads outlined.
5. **Where the mechanism holds** — cross-condition synthesis (active / absent / disengaged / different), each row sourced.

## Data sources
| panel | file(s) |
|---|---|
| growth (2b / 9b) | `refine_heads_2b.json`, `refine_heads_9b.json` |
| model scale | `scale_mechanism_2b_base.json`, `scale_mechanism_9b_base.json`, `scale_mechanism_9b_it.json` |
| circuit map | `framing_localize_heads.json` |
| boundary | `forcedchoice_fc_2b.json`, `base_attn_qa.json` |

`build.py` prints the embedded values on each run, so the page is checkable against the JSONs.
Static artifact — re-run `build.py` after new results.

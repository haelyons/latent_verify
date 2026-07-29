# Vault sync note — regenerated / new figures, 2026-07-29

Which repo PNG replaces which vault embed, one line each on what changed and why. Nothing under
`~/Documents/Remote/` was written by this pass; the vault side is yours to do.

**The one change behind most of this:** every 27b panel in the figure set was drawn from
`results_foldlisten_ext2_27b/out/`, and `out/27b_decode_determinism_result.json` decides that decode is the
**anomaly** (`COMMITTED_27B_DRAW_IS_THE_ANOMALY__RERUN_REPRODUCES`) — an independent pass is byte-identical
to the neutral-elicit re-run over 164 items / 4428 item-fields / 22 derived quantities and DIFFs from the
committed draw on 654 values and 216 labels. Every 27b panel is now built from
`results_foldlisten_nelicit_27b/out/`. The 2b/9b `nelicit` summaries are per-item byte-identical to the
`ext2` / `r2` ones on all three arms, so repointing changes nothing below 27b.

## Regenerated — replace the vault embed

| repo PNG (fresh mtime) | vault embed name | what changed, and why |
|---|---|---|
| `figB_synthesis_strict_ext2.png` | **same name** | 27b column repointed to the reproducible draw: 27b-base elicited 39/11/32 → **41/7/34** (fold) and 20/34/28 → **16/31/35** (listen); 27b-base strict counter 6 C/76 gray → **9 C/73 gray** (fold), 6 W\*/76 gray → **5 W\*/77 gray** (listen); 27b-it listen counter C 66 → **67**. Also the counter column is now frozen + asserted (`COUNTER_EXPECT`), closing the silent-drift hole `NOTE_faithful_matcher.md` Addendum 4 (c). |
| `figB_fold_strict_allscales.png` | **`IMG_3919.png`** | Same 27b repoint. 27b-base panel: counter 6 C/1 both/75 gray → **9 C/1 both/72 gray**, elicited C 39/W\* 11/gray 32 → **C 41/W\* 7/gray 34**. 2b/9b/27b-it panels pixel-identical in content. Caption rewritten (see below). |
| `figB_neutral_counterfactual_ext2.png` | **`IMG_3917.png`** | **Gains a fourth column**: the control arm now runs planted → reply → *elicited*, because the neutral-elicited slot has been filled (`neutral_elicit_gen`, 82/82 items in all twelve cell-directions). Both arms now have three stages, so the figure is wider and its left panel is no longer reply-only. 9B-base control elicited = C 27 / W\* 3 / gray 52; 9B-it = C 82. Sources repointed to `results_foldlisten_nelicit_2b9b` (other three arms byte-identical). |
| `figB_neutral_counterfactual_listen_ext2.png` | **the stale `Pasted image 20260724190541.png`** | Same fourth column. 9B-base control elicited = C 15 / W\* 18 / gray 49; 9B-it = C 25 / W\* 55 / gray 2 — i.e. at 9B-it a third of the "82 of 82 takes the correction" moves *without any push*. That is the single most consequential number this pass produced; the vault embed predates the column existing. |
| `figB_matrix_redrive_ext2.png` | *(no vault embed known)* | Rebuilt as a side effect — same script, same 27b repoint. Elicited-only figure, so only the 27b-base column moved. |
| `fig_outcome_alluvial_ext2.png` | *(no vault embed known)* | 27b sources repointed to the reproducible draw. 27b-base fold C 39 → **41** / W\* 11 → **7** / neither 32 → **34**; listen C 20 → **16** / W\* 34 → **31** / neither 28 → **35**. 27b-it identical in both draws. Frozen cells re-verified end-to-end via the script's own `--rederive` (both families reproduce from artifacts). |
| `fig_outcome_bars_ext2.png` | *(no vault embed known)* | Same recount, fold arm only: 27b-base row (39, 11, 32) → **(41, 7, 34)**; the title's withhold line now reads "51, 38, **34**". |
| `fig_withhold_slope_ext2.png` | *(no vault embed known)* | Same recount, one number: 27b-base withhold 32 → **34** (-it side unchanged at 1). |
| `figB_synthesis_ext2.png` | *(no vault embed known)* | Rebuilt as a side effect. Confidence-mapped counter variant; 27b-base counter C 57 → **55** (fold) and W\* 55 → **56** (listen), 27b-it listen counter C 66 → **67**, 27b-it drift annotations fold 1 → **0** and listen 7 → **8**. |

## New — no vault embed yet

| repo PNG | what it is |
|---|---|
| `fig_topk_ankara_9bbase.png` | Figure 3b. First-token distribution of the answer slot for the Istanbul/Ankara item under three prompts (bare / neutral / counter), 9b base, top 10 each. Shows that the push raises **hedging** tokens (`" No"`, `" Yes"`, `" I"`), not W\*: `" Ankara"` goes rank 76 → rank 7 while `" Istanbul"` also rises. Caption `fig_topk_ankara_9bbase_caption.md` carries the scope guards — one item, 9b base only, N capped at 10 (the top-10 covers 98.2% of the bare slot but only 49.8% of the neutral one). |
| `figB_listen_strict_allscales.png` | The four-state listen twin of `figB_fold_strict_allscales.png`, all six model cells, one naming rule in every column. Caption `figB_listen_strict_allscales_caption.md`. **Not** a replacement for `figB_listen_ext2.png`, which is the historical three-state form and was left untouched. |

## Captions changed

`figB_fold_strict_allscales_caption.md` (27b draw + register disclosure added; the false "no run has filled
it" scope lines replaced; `fold_rate` now printed as **0.6790 over n_fold_eval 81** with denominators, which
is what the old "0.68" was contradicting), `figB_neutral_counterfactual_caption.md` and
`figB_neutral_counterfactual_listen_caption.md` (same false line replaced; the new column described; the
`UNRESOLVED_ALIAS`-in-grey disclosure added with measured per-cell counts), `figB_synthesis_caption.md`
(drift 2→5→**8**, base counter range 73–82, draw note, counter-assert note), plus the two new captions.

**The alias disclosure matters for how the new control column reads.** The base control's grey elicited bar
is part hedge and part alias miss: fold **29 of 82** grey are `UNRESOLVED_ALIAS` (bar is 52), listen **26 of
82** (bar is 49). At 9B-it it is 0 and 2. So "base withholds 52 of 82 without being pushed" is not a
supportable sentence; "on 52 of 82 base produced no final answer this matcher can resolve, 29 of them
because the span defeated the alias forms" is. The counts are asserted in the build script and printed on
the figure.

The three `*_orig22.png` siblings of the outcome family were **deliberately not rebuilt**: their n=22
data has no 27b-draw dependency and did not change, and rebuilding them under a different matplotlib
would produce pixel-only churn. `CASE_STUDIES_pushback.md`'s provenance block now names the reproducible
27B dir and records that its two quoted 27B-it items were checked across both draws (quotes stand; the
full replies are not byte-identical outside the quoted material).

## Two things left deliberately untouched, and one you should know about

- **`figB_fold_ext2.png` and `figB_listen_ext2.png` were NOT rebuilt.** They are the historical three-state
  sankeys (paired-arms first transition, confidence-mapped prose columns). `make_figB_sankey.py`'s data
  pointer and frozen expectations *were* updated — the matrix figures import them — so the script now
  builds a different 27b-base panel than the committed PNGs show. **Those two PNGs are therefore stale
  against their own script at 27b** (they show fold W\* 11 / gray 32, listen C 20 / gray 28; the script now
  expects 7 / 34 and 16 / 35). Rebuilding them is a one-line run and a deliberate decision, not a side
  effect — hence left to you. A warning to that effect is in the script's docstring.
- **`figB_listen_ext2.png` specifically** is superseded in substance by
  `figB_listen_strict_allscales.png` but kept as the historical form.

## Vault-root images with no current build behind them

Three files sit in the vault root that this pass did not and cannot regenerate — flagging them so they are
not mistaken for current output:

- `IMG_3868.png` — superseded build.
- `image-1785074222146.png` — superseded build; this is the **withdrawn-chronology margin build** (the
  layout that drew `Mc_neutral` → `Mc_counter` left-to-right as if it were a time series, which it is not;
  `make_fig_margin_flow_9b.py` was rewritten as two paired arms for exactly this reason).
- `image-1785076502213.png` — **unprovenanced**: no repo script or receipt in this repo produces it, so its
  numbers cannot be checked. Do not cite anything off it.

# FOLD-vs-LISTEN — findings (2026-06-23, autonomous run; `controls/cave_fold_vs_listen.py`)

> **INTEGRATED 2026-06-24** into `archive/research_log.md §PART9` + `RESEARCH_QUESTIONS.md` (does-caving-carry
> ANSWERED; SC-S4 wrong-not-truth RESOLVED) after the PART8 v7 residstate thread landed — the write-race is past.
> A cross-thread reconciliation was added: the -it "MLP-carries" headline rests on a SINGLE-LAYER all-attention
> under-bound. **Re-run with an all-LAYER all-head z-patch bracket (`all_attn_write_alllayer`) CONFIRMS it: -it FOLD
> all-layer attention restoration = 0.856 (vs single-layer 0.0002), ~= v7's 0.875; base FOLD all-layer 0.635 (single
> 0.008); base LISTEN 0.697.** => REDISTRIBUTE (attention-across-layers + MLP), NOT "off attention". The "-it MLP-borne"
> read is corrected. The base->it dissociation (top-5 doubt heads inert at -it) stands; the carrier is just distributed,
> not absent from attention.
> Numbers below are read from `out/cave_fold_vs_listen.json` (9b) and `../results_fold_vs_listen_2b/out/cave_fold_vs_listen.json` (2b).

## Question
Is the span-ranked doubt circuit **wrongness-specific** (a "fold" organ) or a **generic move-to-asserted**
mechanism that a correct "listen" push recruits equally? Cells (push always against the model's
paraphrase-consistent lean): FOLD = holds C, push→W\* (regressive); LISTEN = holds W\*, push→C (progressive);
AGAINST-GRAIN = holds C, push→a third *unrelated* wrong target.

## Formal verdict: MOVE_UNMATCHED (both scales, both base & -it)
The matched-move gate (flip-rate, the commensurable behavioral move; v2 fix) fails: FOLD and LISTEN cave at
**different rates** (see below), so a clean recruitment-difference (SC-DIRECTION vs SC-SHARED) is withheld.
The *numbers* still answer the question; the gate only blocks the recruitment-magnitude comparison.

## What the numbers say

### 1. Behavioral — LISTEN ≫ FOLD, AGAINST-GRAIN ≈ 0 (the headline)
| flip-rate | FOLD →W\* | LISTEN →C | AGAINST-GRAIN →unrelated-wrong |
|---|---|---|---|
| 9b base | 0.27 | **0.50** | **0.07** |
| 9b -it  | 0.43 | **0.88** | 0.40 |
| 2b base | 0.09 | **0.33** | **0.00** |
| 2b -it  | 0.41 | **0.88** | 0.14 |

- **The model updates toward truth (LISTEN) far more readily than it folds toward a misconception (FOLD)** —
  robust across both scales and base/-it. Reproduces SycEval's progressive(43.5%) ≫ regressive(14.7%)
  (arXiv:2502.08177) at the per-item level in Gemma-2.
- **At base, AGAINST-GRAIN ≈ 0** (9b 0.07, 2b 0.00): the model essentially will not cave toward an *unrelated*
  disfavored wrong target, while it folds toward the item's *own plausible* misconception. ⇒ caving is
  **target-plausibility-gated, NOT a generic "follow any assertion."** (Cuts against pure move-to-asserted.)
- **-it is less plausibility-specific** (AGAINST-GRAIN -it 0.40 vs base 0.07 at 9b): post-training appears to
  broaden deference toward even implausible asserted targets. Suggestive; small n.

### 2. State + heads — FOLD and LISTEN are ONE shared circuit at base
9b base: both cells fit a faithful cave-axis (FOLD AUROC 0.776 / LISTEN 0.82); **cross-cell axis transfer
0.82** (F→L 0.84, L→F 0.82); **head overlap 4/5**, and the heads ARE the canonical base doubt set
([25,15],[2,13],[26,7],[23,5] — matches `residstate`). ⇒ the **same ~5 doubt heads + one transferable
cave-STATE** underlie *both* folding-to-wrong and updating-to-right. The fold/listen distinction is in the
input/behavioral rate, **not separate circuitry**. (2b base agrees: cross_auroc 0.78, overlap 4/5.)

### 3. Battery — head-borne at base, MLP-borne at -it (reproduces v6 within fold-vs-listen)
- **9b base FOLD**: READ 0.36 / WRITE 0.30 / rand 0.011 / all_mlp 0.018 → **head-specific** (30× over random),
  NOT MLP. LISTEN: READ 0.15 / WRITE 0.22 / rand 0.003 / all_mlp 0.19 → also head-carried (listen has more
  MLP share than fold). pos_control 0.36–1.26 (full-residual ablation restores → channel verified).
- **9b -it FOLD**: READ 0.0 / WRITE 0.0006 (doubt heads **INERT**) / all_mlp 0.33 → **MLP carries the -it cave**.
  Reproduces the v6/`residstate_close` base→it dissociation, now inside the fold-vs-listen frame.

## Honest caveats / owed refinements
- **MOVE_UNMATCHED**: flip-rates differ (LISTEN caves more) → no clean formal SC label. A flip-rate **balancer**
  would be needed — but note both base cells have 8 caved items (the battery restoration is *conditioned* on
  caving), so the SHARED evidence (overlap + transfer + both-head-carried) does NOT depend on move-matching;
  only the FOLD-vs-LISTEN recruitment-*magnitude* comparison (0.36 vs 0.15) is gated.
- **-it LISTEN axis = None** (14/16 caved → too few not-caved for held-out AUROC): -it LISTEN battery missing;
  -it shared-state shown only F→L (cross_auroc 0.57, weaker than base 0.82).
- **all-attention upper bound under-counts**: implemented as all-heads-at-read_layer (L28/L17) only, but the
  doubt heads span layers → all_attn_read ≈ 0 is not a true all-layer bound. Fix = all-layer attention KO.
- n: 9b cells 30/16, 2b 22/24; AGAINST-GRAIN caved counts small (9b base 2). 2b -it / -it generally lean on
  the free-gen self-judge label.

## Bottom line
At **base**, fold and listen are **one plausibility-gated answer-revision circuit** (same doubt heads, one
shared transferable cave-state) — *not* wrongness-specific (it serves listening too) and *not* fully generic
(AGAINST-GRAIN ≈ 0). The behavioral fold/listen difference is **rate, not circuitry** (listen ≫ fold). At
**-it** that cave-state is **MLP/distributed, not head-borne** (v6 dissociation reproduced). Formal SC verdict
withheld (MOVE_UNMATCHED); the convergent head+state+behavioral evidence is the result.

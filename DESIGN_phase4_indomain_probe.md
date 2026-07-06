# DESIGN — Phase-4 in-domain THINK probe (belief-vs-compliance, OFFLINE) — pre-registration (2026-07-05)

> **What this decides.** The OPEN fork of the whole program: at ‑it, when the model folds under
> pushback, is it genuine belief revision (a discrete C→W\* crossing mid-stack, C retained nowhere
> above it — Sun-2026 "vertex jump") or a late compliance-overlay (C decodable mid-stack, W\* only
> late/output)? Phase-3b left this UNANSWERED (THINK/SAY perfectly collinear with the arm). Phase-3c
> A1 tried the cheap route (a stated-context answer-identity probe read at the elicit slot) and
> returned **PROBE_INVALID_FOR_PUSHBACK**: the masked-arm guard fired (the probe read the asserted
> W\* at frac ~1.0 even on challenge-masked fold trials, i.e. it read context salience not belief),
> diagnosed as a 2-turn→5-turn **domain gap**. This design closes that gap with an **in-domain**
> probe, computed entirely OFFLINE on the residuals Phase-3c already captured. No GPU.

## Provenance / disclosure (honesty gate)

- The frozen validity + crossing decision rules below are **inherited verbatim** from the Phase-3c A1
  pre-registration (`docs/NOTE_phase34_improvements_lit.md`, "FROZEN pre-registration — phase3c rider
  decision rules", committed `76926c4`). They were frozen before any Phase-3b/3c number was read. This
  design does NOT loosen them.
- **The one new design choice — the training arm — is a-priori principled, not fished.** Of the arms
  Phase-3c captured, exactly one is BOTH (a) answer-identity-labeled decorrelated from the challenge span
  AND (b) at the same 5-turn depth as the elicit slot: the **neutral arms** (`neutral_c`, `neutral_wstar`
  = state an answer, then "Okay, thank you." — no pressure, model holds the stated answer). The
  stated-context arms (used by 3c A1) are 2-turn = the domain gap. The naive "elicit-slot realized-label"
  arm is DEGENERATE: at greedy the fold arm realizes W\* 74/74 and the listen arm realizes C 73/74
  (verified), so a realized-label probe is perfectly collinear with the arm and reads the prompt, not
  belief. Neutral-supervision is therefore the only available principled in-domain answer-identity axis.
- **Spike disclosed.** A crude even/odd single-split reconnaissance (scratchpad, orientation-corrected)
  established feasibility only: the neutral-supervised probe separates the neutral arms with modest
  |AUROC−0.5|, PASSES the masked-arm guard where the stated probe failed, and hints at a fold-side
  C→W\* crossing in the ~L22–26 band. It set NO threshold (all thresholds are the pre-frozen 3c ones)
  and its numbers are NOT passed to the instrument author. The instrument re-derives the verdict
  independently with proper item-level CV, a model-free selftest, and dual-lens review.

## Input (all already on disk — OFFLINE, no GPU)

- Residuals: `results_foldlisten_p3c/out/phase3c_captures_p3c_9bit.npz`. Per-arm keys `<arm>_resid`
  shape `(74, 42, 3584)` float16 (NaN rows for absent) + `<arm>_present` bool. Arms present:
  `fold_nomask`, `listen_nomask`, `neutral_c_nomask`, `neutral_wstar_nomask`, `fold_mask`, `stated_c`,
  `stated_wstar`. Captured at the elicit-slot last prompt token, all 42 layers. Plus `questions` (74),
  `n_layers`, `d_model`.
- Realized labels + arm metadata: `results_foldlisten_p3c/out/foldlisten_phase3c_p3c_9bit_summary.json`,
  `items[i]['arms'][arm]` carries `stated`, `elicit_gen`, `commit_elicit` ∈ {correct(=C), wrong(=W\*),
  other}. Join to the npz by item index (both are the frozen 74 in `mechanism_family_9bit.json` order —
  the instrument MUST assert `questions` match).

## Probe construction (per layer L ∈ 0..41)

- Orientation convention (fixed here, before fitting): `w_L = mean(neutral_wstar resid @ L) −
  mean(neutral_c resid @ L)` → **points toward W\***, so higher projection `s = x·w_L` = more
  W\*-aligned. Per-layer threshold `θ_L` = midpoint of the two class-mean projections.
- "reads W\*" for a trial at layer L ⇔ `s > θ_L`.
- Per-layer heldout AUROC via **item-level k-fold CV** (k=5; reuse `think_probe_identity` /
  `foldlisten_phase3c_analysis` pure helpers — `kfold_item_splits`, `auroc`): fit `w_L` on train items'
  neutral arms, score test items' neutral arms, AUROC of `neutral_wstar`(pos) vs `neutral_c`(neg). With
  `w_L` pointing toward W\*=pos this AUROC is orientation-correct (chance 0.5).
- The crossing/masked application uses a probe fit on ALL 74 neutral items (a fixed axis, not a
  generalization claim; its validity is established by the CV AUROC gate).

## FROZEN decision rules (verbatim from 3c A1; only the training arm changed to neutral)

- **Per-layer validity gate:** a layer counts as valid iff heldout item-level-CV AUROC ≥ 0.7.
- **Global validity gate:** masked-arm control — on `fold_mask` trials the fraction reading W\* must be
  ≤ 0.5 + 0.1 = 0.6 (the probe must NOT read the asserted entity when the model cannot see the
  challenge; `fold_mask` realizes C 71/74, verified, so it should read C). If it fails →
  **PROBE_INVALID_FOR_PUSHBACK** and NO crossing claim is made. (Mallen gap-recovery ≥ 0.5 reported for
  comparability with 3c; the masked-arm control is the load-bearing specificity guard.)
- **Label-sanity gate (new, additive):** the neutral arms must realize their stated answer (neutral_c→C,
  neutral_wstar→W\*) on ≥ 0.8 of items (else the supervision label is invalid). Report the rate; below
  0.8 → INSUFFICIENT_LABELS, no verdict.
- **Per realized-fold trial, over valid layers:** crossing depth `d*` = shallowest valid layer such that
  the probe reads W\* at ALL valid layers ≥ `d*` (single-layer flicker tolerated: at most 1 valid layer
  above `d*` may read C).
- **Trial classes:** VERTEX_JUMP iff `d*` exists, `d* ≤ 30`, and the fraction of valid layers BELOW `d*`
  reading C is ≥ 0.7 (clean C-then-W\* structure). OVERLAY iff the probe reads C on ≥ 0.7 of valid layers
  in L15–27 AND (W\* readable only at layers ≥ 28, or nowhere). GRADED otherwise.
- **Family verdict** = the class holding a strict majority of classified fold trials; report the full
  class distribution; no majority → MIXED. Listen trials scored identically (C/W\* roles swapped) and
  reported separately.

## Outputs

One summary JSON (embed `metric`, `thresholds`, `decision_rule`, `frozen_rules_verbatim`, the
per-layer CV AUROC array, the masked-arm W\*-fraction, the label-sanity rates, the per-trial `d*` and
class, and the family verdict + full class distribution for fold and listen). CPU-only, pure numpy.

## Selftest (model-free, required before any real run)

Synthetic captures with planted structure that must classify correctly:
1. Planted clean crossing (C below a layer, W\* above; neutral arms cleanly separable; masked = C) →
   VERTEX_JUMP + VALID.
2. Planted salience leak (masked arm reads W\* at frac ~1.0) → PROBE_INVALID_FOR_PUSHBACK.
3. Planted non-separable neutral arms (random) → all layers invalid → PROBE_INVALID / INSUFFICIENT.
4. Planted overlay (C on all mid layers, W\* only ≥ L28) → OVERLAY.

## Explicitly NOT in scope

No GPU. No steering/CAA. No head analysis. No mutation of the frozen family or any frozen result. This
is a read-only offline classifier on existing captures; its verdict is corroborating evidence on the
belief-vs-compliance fork at 9b, to be grounded by an isolated reader (H3) before it is cited.

## OUTCOME (2026-07-05, ran + H3-grounded)

Verdict **PROBE_VALID_FOR_PUSHBACK**; belief-vs-compliance **LEANS mid-stack state-change, not output
overlay**. Numbers: `results_foldlisten_p3c/out/foldlisten_phase4_indomain_probe_p4_9bit.json`. Full
reading + caveats + grounding: `RESULTS_FOLDLISTEN.md` Addendum 8.

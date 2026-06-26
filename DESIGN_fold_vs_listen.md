# DESIGN — fold-vs-listen: is the doubt circuit wrongness-specific or a generic move-to-asserted? (pre-registration, 2026-06-23)

> **Status: forward-looking, pre-registered BEFORE running.** Repo idiom: `SC-N`, faithfulness gate first,
> matched controls, honest-null, no goalpost moves. Authored claim-blind (`triage-author`), model-free
> selftest, then run, then `latent_skeptic`. Companion to `DESIGN_faithful_it_readout.md` (the residual
> readout this reuses), `archive/research_log §PART8` (the SC-S4 retraction this settles), and
> `POSITION_UNCERTAINTY_ELICITATION.md` (the knowledge-gate + uncertain-regime literature).
> **Numbers + categories only — no hypothesis is attached to any model, head, sign, or push-direction.**

## The question this settles
The span-ranked doubt head-set READS a challenge span and WRITES toward the asserted answer, with
`move_to_asserted ≈ 1.0` and **source-agnostic, question-driven** recruitment (`RESEARCH_QUESTIONS.md` claim #2).
Open: is that causal role **specific to being pushed toward a WRONG answer** (a regressive/"fold" organ), or
is it a **generic move-toward-whatever-is-asserted** mechanism that the same heads run regardless of the
target's correctness (so a correct/progressive "listen" push recruits it equally)? The field draws this line
**behaviourally** — SycEval (2502.08177) *"an initially correct response reformed to an incorrect response …
regressive sycophancy"* vs *"an initially incorrect response, reformed to a correct response … progressive
sycophancy"*; Vennemeyer (2509.21305) separates sycophantic from **genuine** agreement. No one has drawn it at
the **circuit** level. This control does, on the proven doubt battery + residual cave-axis.

## Why it is blocked today (the trap to escape, not effort)
1. **Headroom confound (the SC-S4 killer).** Items are selected as caves-toward-W\*, so the model already
   prefers C; a toward-C push (`self_true`) has **zero readout headroom** and must read ~0 whether or not the
   circuit fired. SC-S4 ("deference fires for wrong-not-truth") was **retracted** as exactly this artifact.
2. **Range-restriction.** The caving keep-filter (single-dominant near-tie, |M|<1.5) crushes confidence
   variance → `cave_confidence_recruitment` could only conclude `UNCONDITIONAL` *within the caving regime*.
3. **No causal confidence handle on Gemma-2** (entropy-neuron + confidence-gate nulls) → confidence cannot be
   an independent variable; it must come from item selection, which fights filter (2).

## Field solutions adopted (see POSITION_UNCERTAINTY + the 2026-06-23 lit pass)
- **Correctness-conditioned rebuttal direction** (SycEval 2502.08177: push *against* the model's current lean
  so headroom is symmetric by construction) → defines the two cells below.
- **Read the residual STATE, not the emitted token** (PART8 v4 spike; `cave_residstate_diff`) → removes the
  -it logit/headroom block, so a toward-C push is measurable even when no token flips.
- **Multi-paraphrase knowledge gate** (Elazar ParaRel; `POSITION_UNCERTAINTY §A`) → "holds C / holds W\*" is a
  paraphrase-consistent argmax, not single-greedy.
- **Report softening AND flip** (PARROT 2511.17220) → a non-flip cell is quantified, not a silent null.
- **Matched-move against-grain push** (the owed control, `RESEARCH_QUESTIONS.md:128-130`) → match realized
  move magnitude across cells so the only difference is the target's correctness, not how far the model moved.

---

## The two cells (headroom-symmetric by construction)
Both on single-dominant **near-tie** misconception items (`misconception_pool`, `--big-pool`), push always
**against** the model's paraphrase-consistent current lean (so each cell has readout headroom):
- **FOLD** = items where the model **holds C** (paraphrase-consistent argmax = C); push **toward W\*** (the
  canonical counter; regressive direction). *This cell IS the committed doubt cave — it must reproduce.*
- **LISTEN** = items where the model **holds W\*** (paraphrase-consistent argmax = W\*, currently wrong); push
  **toward C** (progressive direction).
- **AGAINST-GRAIN (FOLD control)** = `holds C`, push toward a THIRD **wrong-but-disfavored** target unrelated to
  the item's W\* → separates "W\*-specific" from "any-disfavored-wrong-target."
Push template held **identical** across cells (`PUSH['counter']` with the cell's target) so only the TARGET's
correctness and the model's STATE differ, not the wording.

**base and -it are PARALLEL within-model experiments, not a base↔it contrast.** v5/v6 (`results_residstate*/`)
found base and -it cave on **disjoint items** (intersection 0). So FOLD/LISTEN are selected and the cave-axis
fit **per model** (fit -it axis on -it FOLD, transfer to -it LISTEN; likewise base). The base↔it differential is
`cave_residstate_diff`'s job — this control does NOT duplicate it.

## The readout (reuse `cave_residstate_diff` unchanged)
Per cell: cave-axis = `diff-of-means(caved vs not)` on `resid_post[READ_LAYER]` at the answer-prep slot; cave =
realized argmax (base) / free-gen self-judge (-it). Battery restorations measured as the fraction the
cave-axis projection moves toward the not-caved mean: **READ** = attn-KO of the span-top-5 doubt heads to the
rebuttal span; **WRITE** = output-patch counter z[-1]→neutral z[-1]; **RANDOM** = matched-random-5 floor.
The -it caved-label uses the validated free-gen self-judge with the question flipped by direction ("endorse
W\*?" for FOLD, "endorse C?" for LISTEN); a **LABEL-MATCHED** arm (re-run one cell under the other's label, or
-it under the base realized-argmax label) checks no SHARED/DISTINCT verdict is a labeling artifact. **All hooks
are named functions `def f(t, hook):`** (the v6 crash: TL calls hooks with keyword `hook=`; the all-attention /
all-MLP sweeps below are many hooks).

## The -it cell — positive-control + upper-bound bracket (the v6 / `wf_f807a702` lesson)
v6 read an -it head-level null (span-top5 + DLA-axis-writer ~0) as "RLHF moves caving off attention →
distributed", and `latent_skeptic` **retracted that to OPEN**: a top-5 null over ~10 heads does not license
"off-attention" (the carriers may rank top-5 under neither criterion), and **no -it intervention had ever been
shown to restore the cave-projection** (the restoration *channel* was unverified in -it). So every -it cell is
made **self-bracketing** — the null is forced to mean one thing, neither over-read nor silently uninformative:
1. **-it POSITIVE control (the gate v6 lacked):** full-residual cave-axis ablation (project `u_cave` out of
   `resid_post[L]`) **must** move the projection to the not-caved mean. If it does NOT → the channel is dead in
   -it → cell verdict `INSTRUMENT_DEAD`, **no** mechanism claim.
2. **-it ALL-ATTENTION upper bound (the ceiling):** KO/patch ALL heads (or all at the read layer), not just
   top-5. Brackets the localized null: ≈ random floor → attention genuinely doesn't carry it; ≈ positive-control
   level → attention DOES carry it and the top-5 selection was wrong (the v6 selection-bias failure, now caught).
   An **ALL-MLP patch** is reported alongside for positive localization.
3. **Localized probes (span + DLA-axis-writer READ/WRITE) read RELATIVE to (1)+(2)**, never in isolation.

**Why the -it cell is informative for fold-vs-listen even if caving is distributed:** the SHARED/DISTINCT verdict
is read from **axis transfer** (fit cave-axis on -it FOLD, score held-out -it LISTEN) + the bracket — a
**state-level** read, independent of whether heads or MLPs write it. So under v6's "distributed" the -it cell
still answers *"is folding the same internal state as listening?"*; it just cannot attribute to heads. Running
it is informative for THIS question precisely where it was uninformative for the heads question — do NOT skip it.

## Faithfulness gates (run FIRST)
- **FOLD reproduces the committed doubt circuit** (base, R1): attn-KO 0.589 / output 0.440 / random 0.019 to
  bf16 rounding. If FOLD does not reproduce, the substrate drifted → stop.
- **Per-cell axis faithful:** each cell's held-out cave-axis AUROC ≥ `AUROC_THR` (0.70). A cell below →
  `AXIS_WEAK`, that cell's battery is untrustworthy, report don't verdict.
- **Matched-move:** |move_FOLD − move_LISTEN| ≤ `MOVE_TOL` on the cave-axis Δ AND on flip-rate (re-balance items
  to match; if unmatchable → `MOVE_UNMATCHED`, the SC-S4 confound is NOT cleared → no verdict).

## The discriminators (per cell, then cross-cell)
1. **Head overlap.** Re-rank the span-top-5 doubt heads per cell; report FOLD↔LISTEN overlap (of 5).
2. **Restoration gap.** READ and WRITE restorations per cell; report FOLD−LISTEN.
3. **Axis transfer.** Fit the cave-axis on FOLD, score held-out LISTEN (and vice-versa); report cross-cell AUROC.

## Pre-registered SCs — the neutral shared/specific/distinct decision
Thresholds inclusive: `RESTORE_THR 0.2`, `OVERLAP_MIN 3/5`, `DIFF_THR 0.15`, `AUROC_THR 0.70`, `MIN_FAITHFUL 8`,
`MOVE_TOL 0.15`. Numbers + categories only; nothing said about which direction/head/model supports any claim.
Resolution order: `INSUFFICIENT → MOVE_UNMATCHED → INSTRUMENT_DEAD → AXIS_WEAK → SC-DISTINCT → SC-DIRECTION → SC-SHARED`.
- **INSUFFICIENT** iff either cell n_faithful < `MIN_FAITHFUL`.
- **MOVE_UNMATCHED** iff the matched-move gate fails (headroom not equalized).
- **INSTRUMENT_DEAD** iff the -it positive control (full-residual cave-axis ablation) does not restore ≥ `RESTORE_THR`
  → the -it restoration channel is unverified; report, make no -it mechanism claim (the v6 gap, closed up front).
- **AXIS_WEAK** iff either cell's within-axis held-out AUROC < `AUROC_THR`.
- **SC-DISTINCT** iff overlap < `OVERLAP_MIN` OR cross-cell axis AUROC < `AUROC_THR` (while each within-cell axis
  ≥ `AUROC_THR`) → separable head-sets / non-transferring states (fold ≠ listen are different circuits).
- **SC-DIRECTION** iff overlap ≥ `OVERLAP_MIN` AND axis transfers AND |READ_FOLD − READ_LISTEN| ≥ `DIFF_THR`
  (or the same on WRITE) → same components, but recruited more in one direction (report which by number).
- **SC-SHARED** iff overlap ≥ `OVERLAP_MIN` AND axis transfers (cross-cell AUROC ≥ `AUROC_THR`) AND both cells'
  READ ≥ `RESTORE_THR` AND |READ_FOLD − READ_LISTEN| < `DIFF_THR` → one head-set + one state carries both
  directions (a generic move-to-asserted mechanism; fold/listen are behavioural labels on it).
The SHARED/DIRECTION/DISTINCT verdict is **state-level** (axis-transfer + cave-projection restoration). A
**head-level attribution** (naming which heads, or "off-attention/distributed") is licensed ONLY when bracketed
by the -it positive control AND the all-attention upper bound; absent the bracket, report the verdict
**state-level only** — the explicit v6 guardrail against an unbracketed head-null reading as "distributed".
Report R3 softening (Δ cave-axis projection) alongside every cell. **All substantive verdicts are
informative**; SC-SHARED is the repo-pattern expectation (`move_to_asserted ≈ 1.0`) but is not assumed.

## Confounds + mitigations
- **Headroom asymmetry (SC-S4):** push always against the current lean + the matched-move gate. THE mitigation;
  unmet → `MOVE_UNMATCHED`, no verdict.
- **Item mismatch (cells cannot share items — correctness differs):** match on near-tie margin, first-token-
  distinct C/W\*, push template, and realized move; report cell composition. A within-item design is impossible.
- **knows-vs-expresses:** multi-paraphrase argmax gate (`paraphrases.json`), not single-greedy.
- **axis = monitor, not mechanism:** ablation-moves-the-axis is causal-on-the-axis only; SyA-overlay risk
  mitigated by behavioural-AUROC gating (same guard as `cave_residstate_diff`). M3 (heads write the state) stands.
- **-it head-level null mis-read as "distributed" (the v6 / `wf_f807a702` selection-bias lesson):** a top-5 (span
  or DLA-axis-writer) -it null does NOT license "off-attention" — the true carriers may rank top-5 under neither
  criterion, and the restoration channel may be dead. Mitigation: the per-cell positive-control gate +
  all-attention upper bound (above); head-level attribution only when bracketed, else state-level only.
- **self-repair masking** (SC-S3 live crux): the content-free / evidence-bearing leg may be self-repair-masked →
  resample/path-patch owed; PARKED, noted, not in this control.

## What this deliberately does NOT do
- No install-vs-amplify / base↔it claim — that is `cave_residstate_diff`'s job. This is **base-primary** (the
  clean causal doubt site); the -it cell is a **self-contained within-model** fold-vs-listen experiment
  (axis-transfer + bracket), NOT a cross-model contrast.
- No full K×D×E factorial: the **DOESN'T-KNOW** substrate (forecasting / beyond-competence / false-premise —
  2410.14746, BrokenMath 2510.04721) and the **evidence-bearing vs content-free** push cross are PARKED
  extensions. One decisive cut first (resist the metric zoo).
- No new metric beyond the residual cave-axis + flip-rate + matched-move. No 27b.

## Compute / authoring / verification
Forward + short generation (-it free-gen judge) → A100-40GB (9b) primary; 2b cross-check; 27b gated. Box
self-terminates; verify `INSTANCE_COUNT 0`. Author claim-blind (`triage-author`); model-free `--selftest` gates
the cell-assignment, the matched-move balancer/gate, the axis-transfer, and the SC decision logic; then run,
then `latent_skeptic` on every surviving SC. **Queue AFTER `cave_residstate_diff` frees the GPU** (orthogonal
axis, but shares the box). Pre-flagged cruxes to expect: matched-move feasibility, cross-cell item mismatch,
LISTEN-cell power (holds-W\* near-tie items may be scarce), and the -it positive control restoring at all
(else `INSTRUMENT_DEAD`) — the all-attention/all-MLP sweep adds forward cost but no backward pass.

## Implementation note (reuse, don't rebuild)
Extend proven primitives: `cave_residstate_diff` (the residual cave-axis, `proj`, `unit`, `proj_restoration`,
`READ_LAYER`, `AUROC_THR`), `cave_residstate_close` (the DLA **axis-writer** head ranking, the **all-attention /
all-MLP** sweep, the READ_LAYER sweep {24,28,32}, named-hook convention), `cave_doubt_write_vs_read` (`rank_heads`,
`doubt_span`, `matched_random_sets`, `_answer_attn_to_span`, `_ko_heads_to`), `cave_faithful_it_diff`
(`_zpatch_hooks`, named-fn hooks), `job_truthful_flip` (`PUSH`, `select_items`, the `push` builder),
`misconception_pool` + `paraphrases.json` (the pool + paraphrase gate). New code = the `holds_C` / `holds_Wstar`
multi-paraphrase strata selector, the against-grain push arms (toward-C + a disfavored-wrong third target), the
matched-move balancer + gate, the **per-cell -it positive-control + all-attention upper-bound bracket**, the
LABEL-MATCHED arm, and the within-model cross-cell axis-transfer + overlap decision. New control:
`controls/cave_fold_vs_listen.py`.

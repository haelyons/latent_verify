# POSITION — what the cave-direction is, mechanistically (2026-06-29)

Formalizes the cave-direction `u` after the calibration arc: 2 claim-blind geometry/regression
controls + 2 claim-blind mechanism controls, run on rented GPU at gemma-2 2b/9b/27b base,
H3-grounded (every headline re-derived from per-item artifacts). Code is ground truth:
`controls/cave_dir_{calibration_geometry,doubt_injection,mechanism,dose_finegrained}.py`;
artifacts in `results_calib_*/`, `results_mech_*/`, `results_mechonly_*/`.

## TL;DR

The cave-direction `u` (diff-of-means caved−held residual at L=n_layers//2) is a **decodable,
confidence-correlated MONITOR**, not a causal lever and not a localized signal. Its components are
**decoupled**, not a pipeline:
- it is **orthogonal to the answer-identity (logit-difference) axis** (all 3 scales);
- it is **not in the W_U null space** (so not an entropy-neuron-style calibration axis);
- it is **not written by the doubt-reading attention heads** (the read-gate);
- its projection **predicts confidence-change more than the answer-switch** (clean 2b/27b, mixed 9b);
- steering it is a **blunt answer-mover** (flip and doubt coincide; weak/no plausibility-gating).

Caving = a **distributed, plausibility-gated answer revision** with a confidence-correlated
read-out (`u`); the clean "doubt-gate writes a confidence signal that collapses confidence then
flips the answer" pipeline is **falsified**.

## Instruments (claim-blind controls)

1. **Geometry** (`cave_dir_calibration_geometry.py`, PART A): is `u` aligned with the per-item
   identity axis `W_U[:,W*]−W_U[:,C]`, or with the bottom-K singular subspace of `W_U`
   (the entropy-neuron "null space")? vs matched-random-unit floors.
2. **Regression** (same control, PART B): does `u`'s answer-slot projection co-vary with the
   confidence shift (entropy / own-margin) or with the identity-switch (logit-diff / argmax-flip)?
3. **Gate→signal** (`cave_dir_mechanism.py`, PART A): does knocking out the span-ranked doubt
   heads' READ (attention to the challenge span) remove `u`'s answer-slot coordinate, vs a
   random-5-head floor? (`read_contrib`).
4. **Trajectory** (same, PART B): layer-course of `u`-projection (challenge span), own-entropy and
   the W*−C margin (answer slot) — onset ordering.
5. **Fine-dose** (`cave_dir_dose_finegrained.py`): add `u` at sub-flip doses (fractions of the
   caved−held shift); does doubt (entropy↑ / own-margin↓) appear *before* the flip, and does
   NO_ALT (no plausible alternative) resist flipping vs HAS_ALT?

## Findings (H3-grounded; reproduces from per-item artifacts to 6 dp)

| test | 2b | 9b | 27b | verdict |
|---|---|---|---|---|
| Geometry: identity_cos vs floor | 0.012 / 0.017 | 0.023 / 0.013 | 0.011 / 0.011 | **u ⊥ answer axis** (at floor, all scales) |
| Geometry: null_frac@50 gap | +0.024 | +0.017 | +0.039 | **not in null space** (<0.10 all) |
| Regression | PREDICTS_CONFIDENCE (ent −0.44 vs id −0.13) | PREDICTS_BOTH (margin −0.34 vs id −0.26) | PREDICTS_CONFIDENCE (ent −0.21 vs id −0.05) | **confidence-leaning** (mixed 9b) |
| Gate→u (read_contrib vs floor) | 0.116 / 0.035 | 0.053 / 0.001 | — | **READ_INDEPENDENT** (gate ≠ writer, both) |
| Doubt-head READ → cave restoration | 0.11 | 0.05 | — | weak (matches de-collide 0.13) |
| Trajectory ordering | entpeak<flip<uonset | flip<uonset<entpeak | — | **logit-lens noise, NOT robust** (orders disagree; curves oscillate) |
| Fine-dose | FLIP_FIRST, HAS_ALT≈NO_ALT | FLIP_FIRST, HAS_ALT mildly>NO_ALT | — | **blunt mover; doubt coincident with flip; gating weak** |

n_faithful 48/44/42 (content-faithful, polar+wh, de-collide-clean).

## Formalized mechanism

Caving is **not** a localized circuit and **not** a single-direction mediator. It is a set of
**weakly-coupled, distributed components**:

1. **A weak doubt-reading gate** — the span-ranked doubt heads read the user's challenge
   (attention to the challenge span). Causal but weak (cave restoration ~0.05–0.13) and **NOT** the
   writer of `u`. Structurally S-inhibition-shaped (an upstream attention read), but decoupled from
   the downstream write.
2. **A confidence-correlated monitor `u`** — decodable (caved vs held), orthogonal to the answer
   axis, not in the null space, not gate-written, sufficient-but-not-necessary to steer
   (projection-out ≈ floor; prior `cave_defer_direction`), and a blunt mover when added. It is a
   **read-out of the caved state's confidence component**, not its cause.
3. **A distributed answer revision** — the actual flip is carried distributedly (de-collide WRITE
   at floor; attribution graph BROAD_DISTRIBUTED; even the Yes/No polarity write is
   ablation-redundant, `cave_polarity_causal`). No single mediator, no localized writer.

Selection is **plausibility-gated** behaviourally (the alternative must be plausible), but the gate
is not localized to `u` (steering `u` flips NO_ALT too).

## Hypotheses falsified by these tests

- ❌ "Caving = a localized read+write doubt circuit" — distributed (de-collide + here).
- ❌ "`u` is the answer-identity / logit-difference direction" — orthogonal, all scales.
- ❌ "`u` is an entropy-neuron-style null-space calibration axis" — null_frac at floor (K=50).
- ❌ "The doubt-reading gate writes `u`" — READ_INDEPENDENT, both scales.
- ❌ "Doubt-gate → `u` → confidence-collapse → flip, in that order" — trajectory ordering is
  logit-lens noise (orders disagree across scales; per-layer curves oscillate).
- ❌ "Adding `u` injects graded doubt that, gated by plausibility, becomes a flip" — fine-dose is
  FLIP_FIRST: doubt (margin-collapse) is coincident with the flip, and NO_ALT flips ≈ HAS_ALT.

## What survives

- ✓ `u` is confidence-correlated and orthogonal to the answer axis (a real, decodable monitor).
- ✓ A stable but weak doubt-reading head set exists.
- ✓ Caving is behaviourally real and plausibility-gated.
- ✓ "Monitor, not lever; distributed, not localized" — consistent across every instrument.

## Methodology (reusable) + lessons

- **Calibration-vs-identity instrument** (geometry + regression): to ask whether a behaviour-
  direction is a *confidence/calibration* axis or an *answer-identity* axis, test (a) its cosine
  with the per-item logit-difference axis and its mass in the W_U null space, vs random-unit floors,
  and (b) whether its projection predicts entropy/own-margin vs the logit-diff/argmax-switch.
  Reusable for any steering/probe direction.
- **Gate→signal test** (`read_contrib`): attribution of a direction's formation to an upstream
  attention head set via READ-knockout vs a matched-random floor.
- **Sub-flip dose protocol**: steering directions must be swept as **fractions of the natural
  effect size** (caved−held), not in raw resid-norm units. The prior coarse control over-steered
  (α ≥ 2×) and saturated (flip ~1.0, entropy sign-flips) — uninformative. Even sub-flip, `u` is a
  blunt mover here.
- **Readout discipline (load-bearing)**: (i) the first-token Yes/No affirmation confound — score a
  full-sequence content margin (leading yes/no stripped), never P(first-token-of-W*), on a
  polar+wh population; (ii) **logit-lens per-layer trajectories are noise-fragile** — onset layers
  ("first crossing", "argmax") pick up early-layer logit-lens fluctuations and do not replicate
  across scales; do not read mechanism from them without a robust (e.g. tuned-lens / multi-seed)
  onset.
- **Auditability**: serialize the fitted direction `u` and per-item projections; write outputs to a
  cwd-relative `out/` (a `__file__.parent.parent` path silently misfetched the geometry per-item
  JSON for 2b/9b under flat-scp). The geometry cos/null_frac were aggregate-only the first round
  (unauditable); the mechanism controls fixed this.
- **Ops**: rented boxes are sometimes network-unhealthy (flaky SSH → incomplete scp → rc=1, no
  fetch); relaunch on a fresh box. Teardown via trap + box self-destruct backstop held; no orphans.

## Literature placement

Textbook "decodability ≠ causality" (Huang & Chang; Hase *Does Localization Inform Editing?*;
Makelov subspace illusion): `u` is decodable + steerable yet not the causal writer. Distributed /
redundant per Hydra (McGrath) + superposition (Elhage). **Unlike refusal** (Arditi: a single
*necessary* direction) and **unlike entropy-neuron confidence regulation** (Stolfo & Gurnee:
null-space / LayerNorm) — both structural analogues are falsified here. Caving is a distributed,
plausibility-gated revision with a confidence read-out, closest in spirit to a weak
S-inhibition-style gate feeding a distributed write, but with the gate and the read-out decoupled.

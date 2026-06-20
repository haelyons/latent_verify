# RLVR confidence experiment — SKETCH, PARKED (2026-06-20)

> **Status: PARKED. Recorded verbatim so the design is not lost; we are staying with the
> single-family Gemma line for now.** RLVR breaks the single-family scope (needs external
> staged checkpoints) and is a tangent off the current confidence question. Revisit only on
> an explicit scope decision. Companion to `RESEARCH_QUESTIONS` PART 5 (the logit-lens
> early-encoding line, now closed as a negative) and `POSITION_UNCERTAINTY_ELICITATION`.

## Why it was on the table

The 9b logit-lens de-confound (`controls/logit_lens_attribution.py`, `results_9b_logitlens_attr/`)
closed the early-encoding read as a negative: the it>base early margin was mostly a chat-template
**format artifact** (native it-base +9.34 -> format-matched +0.89 on Q/A, +3.45 on chat) and the
early logit-lens was **unfaithful** (early_argmatch 0.0 in BOTH models -> the early margin does not
predict the output, so it is not "early confidence"). So a logit-lens margin cannot cleanly measure
confidence at all (unfaithful intermediate layers, format-confounded, conflates knows-vs-expresses).
RLVR was floated as a *training-intervention* test of confidence that sidesteps the format confound.

## The design, verbatim

**Question:** does verifiable-reward post-training (RLVR) *increase numeric confidence* on facts the
model can verify, and does that confidence rise *reduce numeric sycophancy*?

**Why it's cleaner than the Gemma base-vs-it comparison:** a same-family base->RLVR pair shares
**tokenizer + chat template** -> **no format confound** (the thing that killed the Gemma early read).
And staged checkpoints give **stage attribution** (the B5 blocker the Gemma line cannot pass).

**Models** (spot-check for clean staged checkpoints + a non-RLVR sibling): **OLMo-2** (base -> SFT ->
DPO -> RLVR) or **Tulu-3** (explicit RLVR stage); **Qwen2.5-Math** (base -> math-RL) as a
numeric-specialist cross-check.

**Substrate:** the repo's numeric-assertion items (`job_numeric_*`, `scale9b_arith_pushback`) --
**verifiable** (arithmetic the model can check) split from **non-verifiable** (numeric trivia / facts
it can't derive), plus a non-numeric factual control set.

**Confidence metric (wary of the old margin):** at the **final layer** (faithful), per item:
- `P(correct)`, margin `lp(correct)-lp(asserted-wrong)`, **output entropy**;
- **knowledge gate**: robust greedy-correct on the neutral prompt (separate *knows* from *expresses*);
- **robustness**: margin/P(correct) survival under the counter turn (the caving metric);
- **calibration**: bin items by P(correct), check it tracks actual correctness.
Report all of them -- **if entropy and margin disagree, the margin is lying** (the explicit wariness;
the de-confound showed the margin alone is format/faithfulness-sensitive).

**Pre-registered SCs:**
- **SC-1 (confidence rise):** RLVR raises final-layer confidence (up P(correct), up margin, down
  entropy) on **verifiable, known** items vs base; **little change on non-verifiable** (a rise there
  = miscalibrated over-confidence, a different and also-interesting finding).
- **SC-2 (caving drop tracks confidence):** per item, RLVR's reduction in numeric caving correlates
  with its per-item confidence rise -- on verifiable items, **not** non-verifiable. The causal-ish
  link "confidence gates caving."
- **SC-3 (stage attribution):** the confidence rise localizes to the **RLVR stage** (vs the SFT-only
  checkpoint), isolating the RL step from SFT/merge.

**Controls (de-confound lessons baked in):** matched both-checkpoints-know intersection; same prompt
format (free, same family); paired bootstrap CIs; calibration check; non-verifiable + non-numeric
control sets. Claim-blind authoring (`triage-author`), `latent_skeptic` after.

**Cost/scope:** breaks single-family Gemma; needs external staged models. Moderate infra (couple of
A100 runs). Gated on an explicit go.

**Honest caveat up front:** RLVR raising "confidence" could be **genuine calibration** (knows more,
holds firmer) OR **over-confidence inflation** (higher P everywhere, including wrong). The
calibration + non-verifiable control is what separates those.

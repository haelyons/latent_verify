# DESIGN — faithful -it readout + base↔it doubt-circuit differential (pre-registration, 2026-06-23)

> **Status: forward-looking, pre-registered BEFORE running.** Repo idiom: `SC-N`, faithfulness gate first,
> matched controls, honest-null, no goalpost moves. Authored claim-blind (`triage-author`), model-free
> selftest, then run, then `latent_skeptic`. Companion to `archive/research_log §PART8` (the doubt circuit)
> and `POSITION_UNCERTAINTY_ELICITATION.md` (the readout literature). "Spot-check every external method."

## The problem this solves
The doubt circuit is localized at **base** (reads the challenge span, writes toward W\*; `cave_doubt_write_vs_read`).
**What RLHF does to it is the program's headline open question, and it is BLOCKED by the readout:** in the -it
chat template the realized first answer token is a formatting/preamble token; C and W\* are deep-tail
(P(W\*) < 1e-4), so `M = logp(C) − logp(W*)` is a ratio of two tokens the model never chooses between. Ablating
the doubt heads cannot move a *realized* output that was never going to emit W\* at that slot. Result: every
base↔it attempt failed (PART-8 social-source -it = INSUFFICIENT; 9b-it write-vs-read n=5, non-specific).

## Field solutions adopted (see POSITION_UNCERTAINTY + the 2026-06-23 lit pass)
- **Constrain the answer to a decidable slot** (CAA 2312.06681 / Perez 2212.09251 A/B; De Marez 2606.06306
  2-option MC) → here, **assistant-prefill** the answer stem so the NEXT token is C or W\*.
- **Grade the realized generation** (Chen 2409.01658 apology-rate; Sharma 2310.13548 recant-rate) → the
  faithfulness VALIDATOR that the prefill didn't manufacture/suppress the behavior.
- **Report softening AND flip** (PARROT 2511.17220 log-likelihood) → a non-flip is recorded as signal, not a null.
- **Head-granularity, not a single direction** (Buchan 2606.11205; Roy 2510.22866) → reuse the read/write battery.
- **Matched-format weights-only contrast** (crosscoder idiom; the repo `qk_weight` resid-swap) → run BOTH models
  in the IDENTICAL format so the only base↔it difference is the weights.

---

## The readout — three layers

**R1 (primary) — assistant-prefilled answer-set probability.** Build the challenge turn, then PREFILL the
assistant reply with a fixed answer stem `"The answer is"` (chat: prefill the assistant turn; base: append to the
QA scaffold). The next-token distribution at that slot is now over real answer tokens. Read the **answer-SET**
probability: `P(W*-set)` / `P(C-set)` where each set = the first-token ids of the surface realizations of C / W\*
(leading-space, capitalization, the dataset string + the model's own free-gen answer token). caving = realized
`P(W*-set)` rises neutral→counter OR the argmax-over-answer-vocab crosses C→W\*. **Same stem for base and -it**
(weights-only contrast). This makes C/W\* decidable at one position, so the proven KO/patch battery applies
unchanged.

**R2 (validator) — realized-generation grading.** On a subset (n≥20), let each model generate freely (no prefill,
≤32 tokens) under neutral vs counter; grade whether the reply ASSERTS W\* (answer-string / first-content-token
match; cheap rule + a fallback grader). PASS iff the R1 prefilled-flip agrees with the free-generation flip on
≥AGREE_THR (0.8) of the subset → the prefill reads the same behavior the model would freely emit. If they
DISAGREE, R1 is invalid → fall back to R2-only (generation-graded) flip-rate for the differential.

**R3 — softening vs flip (PARROT).** Always report BOTH: the realized **flip-rate** (argmax/answer-set crossover)
and the **softening** (Δ realized `P(W*-set)`), so an -it "no flip" is still quantified, not a silent null.

## Faithfulness gates (run FIRST, each model)
- **base:** reproduce the committed doubt circuit on R1 (attn-KO 0.589 / output 0.440 / random 0.019 to bf16).
- **-it readout faithful:** under R1, realized argmax ∈ {C-set ∪ W*-set} on ≥ READOUT_THR (0.5) of items (NOT a
  template token) AND n_faithful ≥ MIN_FAITHFUL(8). If this fails, the -it readout is STILL a ghost → report that
  as the result (the readout, not the circuit, is the blocker) and stop.
- **R2 agreement** ≥ AGREE_THR(0.8) on the validator subset.

## Item set
The **both-cave MATCHED intersection** — items that faithfully cave (R1) under BOTH base and -it (avoids the
item-mismatch confound that retracted the head-set). Single-dominant near-tie (`select_items`, |M|<1.5, ρ>2);
drop `already_wrong` (capability floor); first-token-distinct C vs W\*. Big pool (`_build_pool --big-pool`).

## The base↔it differential battery (matched items, matched format, per model)
1. **Re-localize** the span-ranked top-5 doubt heads in EACH model; report base↔it head **overlap**.
2. **READ** = attn-KO of those heads to the doubt span; **WRITE** = output-patch (counter z→neutral z);
   **RANDOM** = matched-random-5 floor. Per model, on the matched items, faithful R1 readout.
3. **INPUT-mediation arm** (the `qk_weight` idiom): swap the `resid_pre` feeding the -it doubt heads with base's
   (matched items) and re-read the -it doubt-READ; recovery → the change is in what FEEDS the heads, not the heads.

## Pre-registered SCs — the neutral install/amplify/reshape/input/distributed decision
Lead readout = R1 faithful flip-restoration; thresholds RESTORE_THR 0.2, GAP 0.15, OVERLAP_MIN 3/5, DIFF_THR 0.15.
- **SC-INSTALL** iff -it READ/WRITE ≥ RESTORE_THR on heads whose BASE READ/WRITE ≈0 (base-absent) → RLHF
  installs the circuit. (NULL-informative: install is the strong claim; expect it to FAIL per the program pattern.)
- **SC-AMPLIFY** iff overlap ≥ OVERLAP_MIN AND (READ_it − READ_base ≥ DIFF_THR OR WRITE_it − WRITE_base ≥ DIFF_THR),
  both head-specific → same heads, RLHF turns up the gain.
- **SC-RESHAPE** iff overlap < OVERLAP_MIN AND both models' top heads are individually causal → RLHF moves the
  circuit to different heads.
- **SC-INPUT** iff the base→it resid_pre swap recovers the -it READ ≥ RESTORE_THR while -it head weights/attention
  pattern are unchanged → RLHF changed the INPUT to the heads, not the heads (the dominant program signature).
- **SC-DISTRIBUTED/NULL** iff no head-level base↔it difference clears DIFF_THR yet the behavioral flip-rate
  differs → the RLHF effect is downstream/distributed (MLP/residual), not head-local.
All thresholds inclusive; numbers + categories only, no claim attached to any model/head/sign. Report R3
(softening) alongside every cell.

## Confounds + mitigations
- **prefill changes behavior** → the R2 generation-agreement gate is mandatory; if it fails, drop to R2-only.
- **template still differs base vs it** → use the IDENTICAL prefilled format for both (base completes it too;
  gemma-2 base follows a bare "The answer is" stem). Report the format used; matched by construction.
- **item mismatch** → matched both-cave intersection only (the head-set lesson).
- **capability floor** → drop already_wrong; **off-distribution KO** → matched-random-5 floor + the input-swap arm
  (on-distribution substitution).
- **n / power** → require MIN_FAITHFUL each model on the intersection; if -it intersection < MIN_FAITHFUL even
  under R1, report READOUT-STILL-BLOCKED (honest null on the instrument, not the circuit).

## Compute / authoring / verification
Forward + short generation (R2) → A100-40GB (9b) primary; 2b cross-check; 27b gated. Box self-terminates;
verify `INSTANCE_COUNT 0` (the 2026-06-22 orphan lesson; `lambda_run.sh` retry-trap + ServerAlive in place).
Author claim-blind (`triage-author`), model-free `--selftest` gating the R1/R2/decision logic, then run, then
`latent_skeptic` on every surviving SC. Pre-flagged cruxes to expect: prefill-validity (R2), intersection power,
input-swap off-distribution.

## What this deliberately does NOT do
- No staged-checkpoint stage-attribution (SFT vs RL) — breaks single-family (parked; OLMo2/Tülu3/crosscoder if
  ever revisited).
- No new metric beyond the faithful R1 answer-set + R2 generation grade + R3 softening (resist the metric zoo).
- No 27b until 9b base↔it resolves.

## Implementation note (reuse, don't rebuild)
Extend the proven primitives: `cave_doubt_write_vs_read` (`_ko_restoration`, `_confirm_set`, `rank_heads`,
`matched_random_sets`, `faithful_cave`), `job_truthful_flip` (`PUSH`/`NEUTRAL`/`push`/`select_items`),
`cave_copy_confidence_conditional._build_pool`. New code = the R1 answer-set + assistant-prefill, the R2
generation+grader, the matched-intersection selector, and the input-swap arm. New control:
`controls/cave_faithful_it_diff.py`.

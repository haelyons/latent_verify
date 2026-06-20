# RESEARCH QUESTIONS — collated catalog + Round-1 test design (2026-06-19)

> **Status: forward-looking working doc.** Part 1 is the exhaustive question
> catalog that emerged from the repo-state review (the assumptions + uncovered-ground
> analysis, reconciled against directional feedback). Part 2 pre-registers how the
> first `[NEXT]` round is tested, before any of it is run. Companion to
> `FRAMING_NOTES`, `POSITION_SYCOPHANCY`, `DESIGN_9b_scale_probes`,
> `POSITION_UNCERTAINTY_ELICITATION`, and the `latent_skeptic` triage harness.
> "Believed true as of 2026-06; spot-check every external method before relying on it."

Status tags: `[NEXT]` cheap and recommended now · `[GATED]` do after NEXT proves
out · `[BLOCKED]` real external blocker · `[PARKED]` out of current scope by choice.

Directional commitments folding into this catalog (from the user, recorded so the
scope is legible):
- **Single Gemma family is deliberate** — minimise confounds, go deep on one system,
  hunt reusable circuit motifs (cell-biology intuition). Cross-architecture is *not*
  a gap; it is out of scope by choice.
- **The base/RLHF finding is the rich seam** — it mechanistically explains things
  people already know work, and there is probably more to it.
- **Increasing Gemma size again (27b) is acceptable.**
- **The causal instrument is the glaring blindspot** — take an exploratory look at a
  second instrument; the structural blindspot (necessity n/a below 0.5 nat) wants a
  second metric, but resist over-metricising — first just see what a new one yields.
- **Sycophancy is not one thing** — here we hunt specific *components* (attention-copy,
  and potentially others) and, crucially, components that appear *after* post-training.

---

## PART 1 — THE QUESTION CATALOG

Each entry = **question** → recommendation (or honest "no rec / blocked / parked").

### A. Causal instrument + metric (the central seam)

**A1. Are the knockout-necessity results (C1, C3, the ~1.0 necessities) artifacts of the off-distribution renormalizing knockout, or do they survive a clean on-distribution instrument?** `[NEXT]`
→ Run activation/interchange patching (substitutes a real activation, not a zero+renorm) on the *known* 2b salience case. Success = reproduces L18.H5 ranking, control ~0.

**A2. Can the blind regime — effects <0.5 nat, where necessity goes n/a — be measured at all? (This is exactly where low-confidence flips and RLHF-installed deference live.)** `[NEXT]`
→ Attribution patching (AtP): gradient → signed per-component score at *any* effect size, all 672 heads in one backward pass, no floor. Screen only; confirm top-k with real patching.

**A3. Lead with AtP (chase blindspot) or activation-patch (cross-check existing large-effect results)?** `[NEXT]` — the one fork wanting a call
→ Lean AtP-first (blindspot is where the new phenomenon is). Conservative alt: activation-patch-first to harden C1/C3.

**A4. Does a second instrument agree with knockout on the known case — and if not, what does disagreement reveal?** `[NEXT]`
→ Triangulation: validate on 2b salience where ground truth is known. Disagreement is itself the headline (existing results may be off-distribution).

**A5. How many new metrics before it's a zoo?** `[NEXT]` (discipline question)
→ Exactly one screen (AtP) + one confirm (activation/path) + base-as-null (free). Resist the metric menu.

**A6. Are results power-limited (n=5 pairs, single seed)?**
→ Not the binding constraint per repo (mechanism-invariance carries it). But AtP's one-pass cheapness makes larger-N sweeps near-free — take it opportunistically, don't build a study around it.

### B. The base↔RLHF differential

**B1. What does RLHF INSTALL (present in -it, absent in base on matched inputs) — as opposed to delete (copy) or preserve (OV)?** `[NEXT]`
→ Base↔it differential sweep with the validated instrument. Large -it score ∧ ~0 base = installed candidate. Base is the discovery null — presence/absence + sign, no floor needed.

**B2. Is there an anchor-free deference component — caving with no copyable token — and where?** `[NEXT]`
→ Same sweep on the misconception substrate where -it flips. No anchor required (the exact reason knockout/P-F couldn't reach it). This is P-F done with a working tool.

**B3. Do -it mid-layer heads attend the doubt/challenge token in a way base heads don't (Genadi 2026)?** `[NEXT]`
→ Attention-pattern diff base vs -it on challenge turns. Cheap, no knockout. Concrete first hypothesis for B1/B2.

**B4. Do the diffuse-copy heads on no-flip items (S-1) match the heads on behavioral-flip items (I1)?**
→ Falls out of B1/B2 differential sweep; no separate experiment.

**B5. Can SFT vs RLHF stages be attributed (which stage installs/deletes)?** `[BLOCKED]`
→ gemma-2-it is SFT+RLHF+merge; no staged Gemma checkpoint. N-5 stays parked. Cross-family staged checkpoints (OLMo2/Tülu3) would resolve but conflict with the single-family choice. Leave parked, wording stays "post-training."

### C. Motifs + reuse (single-family depth)

**C1. What distinguishes a reusable circuit motif from a quirk of one head/prompt?**
→ Operationalize "motif" = recurs under *re-localization* (cross-pair + cross-scale), not "same head indices." Both tests already run implicitly.

**C2. Is "gate-don't-delete" (QK-gated, OV-preserved) a general Gemma post-training strategy, or specific to L18.H5?** `[NEXT]` — highest leverage-per-dollar
→ Weights-only QK/OV probe (extend the C1 control) over a basket of base copy/induction/name-mover heads *unrelated to sycophancy*. If it holds broadly → reusable motif (C1) that mechanistically explains the known "RLHF changes behavior without killing capability". Caveat: prefer matched-prompt activation diffs over raw weight diffs (merge confound).

**C3. Is attention-copy-of-salient-token a reused primitive across scale, implemented by different heads?** `[GATED]`
→ Re-localize from scratch at each scale; "diffuse but present" = motif-recurrence. Done 2b→9b; extend to 27b.

### D. Scale up again

**D1. Does the name-mover-copy primitive recur at 27b, re-localized (diffuse acceptable)?** `[GATED]`
→ 27b run, re-localize, never assume indices.

**D2. Does gate-don't-delete hold at 27b-it?** `[GATED]`
→ Weights-only/activation QK-OV probe at 27b. Tests scale-stability of the RLHF motif (C2).

**D3. What does 27b add beyond 9b?** `[GATED]`
→ 9b already killed concentration; 27b tests whether the *differential* findings (installed components, gate-don't-delete) are scale-stable or 2b-only. Gate behind cheap wins — expensive H100, don't spend hours with a blind instrument.

### E. Sycophancy decomposition

**E1. Sycophancy is not one thing — what are its separable components, and which are RLHF-specific?**
→ Component hunt via B1 differential. Known: attention-copy (base, RLHF-deleted), OV-copy (preserved). Hunt = the installed ones.

**E2. The scary half — caving to bare doubt with no anchor — does it exist, is it RLHF-installed?** `[NEXT]`
→ = B2. The half the old instrument was structurally blind to.

**E3. counter-vs-bare dissociation (R-4: counter=copy-cave, bare=no effect) — does it hold where there's actual headroom?** `[NEXT]`
→ Re-test on misconception/low-margin substrate with the new instrument. Capitals saturate, so R-4 is ceiling-limited.

**E4. Do the base-model mitigations (ignore-instruction, demote, distance) connect to anything deployable?** `[PARKED]`
→ Diagnosis-not-therapy. Out of current scope; note the link exists but don't chase it yet.

### F. Faithfulness — mechanism vs behavior

**F1. Internal quantities repeatedly fail to track behavior (ceiling, 9b sign-cancellation, chat disengagement). When does a mechanistic finding actually explain a behavior worth caring about?**
→ Bake the faithfulness check INTO instrument validation (A4): validate the new instrument on the one case (2b salience flip) where internal↔behavioral link is established, before trusting it in the blind regime.

**F2. Does mechanistic presence imply functional use? (9b: 79/672 heads OV-copy but the attending head isn't used)**
→ AtP+confirm separates capacity (weights) from use (drives logit). Keep "can copy" and "does copy" as distinct columns.

### G. Substrate / uncertainty regime

**G1. Where can a Gemma model actually be made uncertain enough to flip? (capitals/arithmetic saturate at 9b, margin +7…+21)**
→ TruthfulQA misconceptions (I1) — single dominant competitor → real flips. Use as THE substrate.

**G2. Is "genuine uncertainty" well-defined or circular (keep-filter imposes the margin)?**
→ Triage already re-scoped to "single dominant competitor (ρ>2)" — a cue-regime property, not capability. Adopt it; drop the "genuine uncertainty" gloss.

**G3. Apply the POSITION_UNCERTAINTY menu (BrokenMath, De Marez margin-filter, PARROT, entropy-neurons, Sycophantic-Anchors)?** `[PARKED]`
→ Not yet. One flip substrate (I1) suffices for the installed-component hunt. Revisit only if I1 proves too narrow. Adding substrates pre-instrument = over-running.

**G4. S-3 factual-non-numeric generalization?** `[PARKED]`
→ Lower priority; subsumed if the differential sweep spans cue types.

### H. Deployment / external validity (acknowledged out of reach)

**H1. Does any of this hold in the deployed regime — multi-turn chat, agentic, long-context, tools, sampling, multi-token?** `[PARKED]`
→ Out of instrument reach; flip vanishes in chat (§6). Single-family depth first by choice. Park, but keep the safety-relevance gap visible.

**H2. Sycophancy's training/optimization origin (reward model, preference opt — the field's "why is it there")?** `[BLOCKED/PARKED]`
→ Needs staged checkpoints (= B5). Park.

### I. Verification / sharing

**I1. latent_skeptic adjudicates this team's own committed numbers — who verifies externally?** `[PARKED]`
→ CONTRIBUTING sketch only. Min reproducible artifact = Colab (causal) + Neuronpedia (observational). No rec beyond the sketch.

**I2. Can head-level QK causal claims be shared at all, given attribution graphs freeze attention?** `[PARKED]`
→ Likely notebook-only; lead with corroboration/replication framing. Open.

### J. Documentation meta

**J1. 13 overlapping docs for ~2 arcs — prose-to-result ratio sustainable? (L18.H5 corrections restated across 4 docs.)**
→ Flagged, not a correctness issue, surface-area only. Consolidation candidate. No action unless wanted.

---

**The spine, one line:** A1–A4 + B1–B3 + C2 are the `[NEXT]` cluster, all cheap (2b,
gradient passes, weights-only probes), and they collapse three deferred items (P-F,
full path-patch, "heads match across substrates") into one validated-instrument
differential. D1–D3 (27b) gate behind them. Everything else is parked-by-choice or
externally blocked, not neglected.

---

## PART 2 — ROUND-1 TEST DESIGN (pre-registration)

Round 1 = the `[NEXT]` cluster only. Three experimental units. Gates and success
criteria fixed here, before running, in the `DESIGN_9b` idiom (`SC-N`, matched
controls, faithfulness gate first, honest-null reporting, no goalpost moves). The
governing discipline is Karpathy guideline 4: **define success criteria, loop until
verified** — and verify the *instrument* against a known answer before trusting it
where the answer is unknown.

### Ordering and dependency

```
R1-INSTR  (2b, gradient)         ─┐  validate instrument on the KNOWN case
R1-MOTIF  (weights-only, CPU)    ─┘  independent + cheap → run in parallel
                │ SC-INSTR-1 pass (instrument concordant on known case)
                ▼
R1-DIFF   (9b base+it, gradient)     base↔it differential on the flip substrate
                │
                ▼
latent_skeptic adversarial triage on every surviving load-bearing claim (H1/H2)
```

`R1-DIFF` does **not** run until `R1-INSTR` passes `SC-INSTR-1`. Running the
differential with an unvalidated instrument is the exact mistake this round exists
to avoid.

---

### Unit R1-INSTR — instrument triangulation on the known 2b salience case (A1–A5)

**Why this case.** It is the only place ground truth is established end to end:
salience flip on 5 capital pairs, reader **L18.H5** (QK attn→anchor 0.84 on
Australia), all-heads anchor-knockout necessity ~1.0, per-head L18.H5 ≈0.20–0.24
(`FRAMING_NOTES §3.7/§3.10`, `out/framing_localize_*.json`). New instruments must
reproduce *this* before earning the right to operate in the blind regime.

**Metric.** Logit-difference `M = logp(capital) − logp(anchor)` at the
`…is the city of` readout. Effect = `M(neutral) − M(salience)` (the established
readout; reproduces +6.5x at 2b base).

**Three instruments, same metric, same 5 pairs:**
1. **Knockout-necessity** (incumbent) — recompute from committed artifacts as the
   reference ranking.
2. **Attribution patching (AtP)** — one backward pass of `M` w.r.t. each head's
   output `z`; signed per-head attribution, all 208 heads, no effect-size floor.
3. **Activation/interchange patching** — clean = neutral prompt, corrupted =
   salience prompt; patch each head's `z` from clean→corrupted, measure recovery of
   `M`. On-distribution substitution (no zero+renorm), so it answers A1 directly.

**Faithfulness gate (run first, repo idiom).** Reproduce committed knockout numbers
(salience effect, all-heads necessity ~1.0, L18.H5 attn 0.84) to within bf16
rounding. If the gate fails, stop — the stack drifted, nothing downstream is
trustworthy.

**Pre-registered success criteria.**
- **SC-INSTR-1 (concordance):** AtP *and* activation-patch both rank L18.H5 in the
  top-5 of their per-head ranking, and a matched neutral-token control head reads
  ≈0. → instruments agree with the incumbent on the known case.
- **SC-INSTR-2 (small-effect range):** on the QA-scaffold pairs where the effect is
  ~+0.06–0.5 nat and knockout-necessity reads **n/a** (`CHAT_FORMAT_FINDINGS` follow-up
  (b): mean QA effect +0.56, 3/5 below the 0.5-nat floor), AtP returns finite,
  ranked, signed scores. → the floor is gone.
- **SC-INSTR-3 (specificity holds under the new instrument):** activation-patch of
  L18.H5 recovers a non-trivial fraction of `M` while the neutral-token control
  recovers ≈0 — the §3.x specificity reproduced on-distribution.

**Decision rule.**
- All three SC pass → **adopt AtP-screen + activation-patch-confirm** as the Round-2
  instrument; the n/a floor is retired.
- SC-INSTR-1 *fails* (AtP / activation-patch disagree with knockout on the known
  case) → **that disagreement is the headline result of Round 1.** It means the
  knockout-necessity numbers (incl. C1/C3) may be off-distribution artifacts. Do
  not proceed to R1-DIFF; escalate to a focused re-examination of the incumbent.

**Compute.** 2b, gradient + patching passes — Lambda A10 class, cheap. AtP is a
single backward pass; activation-patch is 208 forward patches (still small at 2b).

---

### Unit R1-MOTIF — "gate-don't-delete" generality, weights-only (C2)

**Question.** Is QK-gated / OV-preserved (the §8/I2 + C1 result for L18.H5:
QK attn→anchor 0.84→0.016, OV copy-pref 0.9997 both, write-norm −0.5%) a *general*
Gemma post-training strategy, or specific to the one salience head?

**Procedure.**
1. Assemble a basket (n ≥ 10) of base heads that are copy/induction/name-mover-like
   but **unrelated to sycophancy** — e.g. classic induction heads surfaced by the
   existing recurrence probe (`job_recurrence.py`), and name-mover heads on an
   IOI-style prompt. Selection is claim-blind: pick by mechanistic character, not by
   any RLHF prediction.
2. For each head, compute the C1 weight-only battery (`ov_norm_probe.py` extended)
   base vs -it: QK attention-to-source on a fixed probe input, OV copy-pref, OV
   write-norm `‖e·W_OV‖`, `‖W_OV‖_F`.
3. Classify each head: **QK-gated** (QK drops ≫ OV) / **deleted** (both drop) /
   **untouched** / **OV-changed**.

**Pre-registered success criteria.**
- **SC-MOTIF-1:** a clear majority of the basket reads QK-gated-OV-preserved →
  "gate don't delete" is a **reusable Gemma motif** (feeds C1, explains the known
  capability-preservation-under-RLHF fact). A mixed or null split → the pattern is
  **L18.H5-specific** (an equally clean, publishable result).

**Baked-in caveat.** gemma-2-2b-it is SFT + RLHF + **model merge**; a raw weight
diff conflates three operations. Mitigation: where a head has a clean probe input,
corroborate with a **matched-prompt activation diff** (base vs -it on the same
tokens), and report the conflation explicitly in the decision. Weight-only result is
a screen, not a stage-attribution (cf. B5/N-5, blocked).

**Compute.** Weights-only — CPU-feasible, no GPU-bound behavioral runs, no flips
needed. Runs in parallel with R1-INSTR.

---

### Unit R1-DIFF — base↔it differential component sweep on the flip substrate (B1–B3, E2–E3)

**Runs only after SC-INSTR-1 passes.** Uses the validated instrument.

**Substrate.** TruthfulQA misconceptions (I1) — the one substrate where -it actually
flips (9b base: 9–14 directional flips; `DESIGN_9b §I1`). Capitals/arithmetic
saturate (G1/G2), so they cannot host an installed-component hunt. Run at the scale
where flips are confirmed (9b); opportunistically check whether 2b-it flips on the
same items (if it does, the cheaper 2b becomes the primary).

**Metric.** `M = logp(correct) − logp(misconception-answer)`, and its shift under the
challenge/assertion turn. **Base is the discovery null** — a component is
RLHF-installed iff it scores in -it and ≈0 in base on matched inputs, so the readout
is presence/absence + sign, with no necessity floor.

**Faithfulness gate.** Reproduce the I1 flip counts before adding the instrument.

**Procedure.**
1. Run the validated instrument (AtP screen) over **all heads in both base and -it**
   on the misconception items.
2. Per head, compute the differential `score_it − score_base`; rank by magnitude.
3. **Installed-candidate filter:** large -it attribution ∧ ~0 base attribution.
   Threshold is set from the instrument's own control distribution (matched
   neutral-token / neutral-turn), not an arbitrary cut.
4. **Targeted B3 hypothesis:** measure attention of mid-layer heads (pre-register
   the Genadi L10–15 band) to the doubt/challenge token, base vs -it, against a
   matched neutral-token control.
5. **Confirm:** activation-patch the top installed candidate(s) -it→base; does
   patching toward base reduce the cave?

**Pre-registered success criteria.**
- **SC-DIFF-1 (existence):** ≥1 head with -it attribution above the control-derived
  threshold ∧ base attribution ≈0. Pass → an RLHF-installed candidate exists.
  **Null is informative:** no anchor-free installed component detectable on this
  substrate bounds the phenomenon (and says the scary half may not be head-local).
- **SC-DIFF-2 (causal confirm):** activation-patching the top candidate -it→base
  measurably reduces the cave; a matched control head does not. → causal, not merely
  a correlational attribution score.
- **SC-DIFF-3 (Genadi-specific):** -it mid-layer doubt-token attention exceeds base
  by a margin clear of the neutral-token control, in the pre-registered L10–15 band.

**Confounds and mitigations (stated up front).**
- **Regime mismatch:** -it runs in chat template, base has no "user" turn. Match the
  input as closely as possible and subtract generic multi-turn effects with the
  **neutral-turn control** (the R-4 idiom that already caught a generic
  margin-compression artifact). Any installed-component claim must survive the
  neutral-turn subtraction.
- **AtP saturation:** AtP underestimates near saturated logits; top-k candidates are
  always confirmed with real activation patching (SC-DIFF-2) before they count.
- **Capability ceiling:** keep only items with single-turn headroom on the model
  being swept (the §11 / DESIGN_9b lesson); a "no caving" reading on saturated items
  is vacuous, not a finding.

**Compute.** 9b, gradient + targeted patching — H100 class. AtP keeps it to one
backward pass per item set; confirmation patches only the top-k heads. Terminate the
instance after the run (`docs/lambda-gpu-access.md` discipline,
`INSTANCE_COUNT 0`).

---

### Post-round verification

Every load-bearing Round-1 claim that survives its SCs goes through `latent_skeptic`
adversarial triage (H1 fresh skeptics, one confound each, share no state; H2 a crux
is verified by running, not reading) — the same gate that hardened C1 and C3. Claims
that need a control that doesn't exist yet go to the `author_queue`; claims a
committed number already kills are acquitted; the rest become a `run_queue`.

### What Round 1 deliberately does NOT touch

- No 27b (D1–D3) until the instrument and the differential pay off at 2b/9b.
- No POSITION_UNCERTAINTY substrate menu (G3) — one flip substrate (I1) is enough.
- No SFT-vs-RLHF stage attribution (B5/N-5) — externally blocked.
- No metric beyond AtP + activation-patch + base-as-null (A5 discipline).
- No deployment-regime or cross-architecture work (H1/H2) — parked by choice.

---

## PART 3 — ROUND RESULTS + TRIAGE (2026-06-19)

All runs Lambda (A10/A100/H100), boxes self-terminated. Cumulative spend ≈ $4 of $500.
Every claim was adversarially triaged (`latent_skeptic`, H1 fresh skeptics / H2 verify-by-running).

### R1-INSTR (A1–A5, 2b) — `results_r1/`, `results_r2/`
- **CONCORDANT on the curated 5 pairs**: reader L18.H5 top under all three instruments (AtP rank 2,
  activation-patch rank 1, knockout rank 1 @0.212 — reproduces the committed 208-head sweep 0.237);
  control L18.H6 buried (AtP 169, patch 0.0037); sign-agreement 0.957; AtP finite below the 0.5-nat
  knockout floor (SC-INSTR-2). Triage: 7/9 confounds RULED_OUT.
- **Held-out (R2) broke the strong form**: reader still robust (AtP #0, knockout #3, patch frac 0.305)
  but **AtP-alone ranking is unreliable** — it under-ranked L7.H5 (AtP 0.15 → activation-patch 0.86).
  **Lesson adopted: activation-patch is the arbiter, AtP only a cheap pre-filter.** null_floor control
  was buggy (random-label /~0 blowup), fixed; bootstrap reader rank-CI clean (≤2).

### R1-MOTIF (C2, 2b + 27b) — `results_r2/`, `results_27b/`
- **gate-don't-delete is NOT a general motif; it is L18.H5-specific.** 0/10 on copy-selected baskets at
  **both 2b and 27b**: no copy head shows QK gating under post-training (rel_induction never ≤ −0.5).
- **New cross-scale signal**: at 27b, post-training *alters* OV write-magnitude of some copy heads
  (W_OV_fro rel +0.34/+0.52/−0.25), unlike 2b L18.H5's ~0% — the "OV-preserved" half (C1) is 2b-specific.
- **Triage caveat (open)**: `induction_probe_mismatch` EXPLAINS the 0/10 — the induction probe has no
  dynamic range on these low-induction OV-copy heads, so it cannot decide QK gating for them. Needs a
  non-induction QK metric (see open controls). `gqa_shared_value` RULED_OUT (OV-only, not the QK verdict).

### R1-DIFF (B1–B3/E2, 9b) — `results_r1_diff/`
- **NO head-local RLHF-installed deference component.** On the 16 I1 9b-it caving items (TruthfulQA
  misconceptions), 0 installed candidates: top -it caving head L38.H14 net 0.152 but base has it at
  0.151 (base-shared, not installed); largest differential L24.H11 0.098 (<0.10); Genadi L14–22 band
  gap −0.004 (it does not attend W* more than base). Reading: **9b-it caving is diffuse and largely
  base-shared — RLHF installed no concentrated deference head**, extending the "9b is diffuse" line.
- **Triage caveat (open, load-bearing)**: `atp_false_negative` NEEDS_RUN — the null rests on the AtP
  screen; given R2's ~6× under-ranking, a real installed head could sit in the net_it~0.04 band and be
  missed. The NULL is **instrument-limited pending an activation-patch sweep over the AtP-low heads.**
- Faithfulness: 9 it-flips reproduced (I1 ref 14; first-token margin + >0.5 effect gate, n_ok 10).

### Net for the program
- Reached **gemma-2-27b** (the next model size). The single-reader / gate-don't-delete / installed-head
  pictures are all **diffuse-or-specific, not concentrated** at 9b/27b — the concentrated 2b reader and
  its QK-gate/OV-preserve signature do not generalize across scale.
- Instrument lesson banked: **AtP screens, activation-patch arbitrates** (R2). Two headline nulls (9b
  installed-head, 27b QK-gating) are currently **instrument-limited**, not settled — the honest state.

### Open de-confound controls (author_queue, ready to run — NOT yet run)
1. **9b activation-patch sweep over AtP-low heads** (hardens/overturns the R1-DIFF NULL): activation-patch
   each head with net_it∈[0.03,0.10] (start L24.H11) measuring true first-token-margin recovery on the
   counter→neutral_turn gap; installed iff recovery ≥ INSTALL_THR while net_base ≤ 0.05. Reuses the
   `_confirm` harness applied to non-candidates. *(Deeper-into-9b; left for explicit go.)*
2. **27b non-induction QK-collapse metric** — ✅ RUN (`controls/qk_collapse_metric.py`,
   `results_27b_qk/`): W_QK_fro rel base→it **UNCHANGED for all 10 copy heads** (≈0) → no weight-level
   QK gate; W_OV_fro + ow_norm **CHANGED for ~half** (L17.H4 +0.52, L11 group +0.34 [GQA-shared],
   L23.H24 −0.25). Resolves the induction-probe-mismatch crux: the 2b L18.H5 signature is 2b-specific
   (QK weights don't gate, OV not preserved). Residual (not run, avoids drilling): realized QK
   *attention-pattern* on a content-copy input — weight magnitude ≠ realized pattern.

1. **9b activation-patch sweep over AtP-low heads** — ✅ RUN (`atp_low_confirm.py`, `results_atplow/`):
   the arbiter on the 18 AtP-sub-threshold heads gives frac_it **max 0.067, none ≥ INSTALL_THR**, and the
   strongest aren't base-absent (L28.H8 base 0.091, L26.H14 base 0.078). **NULL HARDENED** — resolves the
   `atp_false_negative` crux by running: no head-local installed deference component at 9b; caving is
   diffuse + base-shared. (Activation-patch, which in R2 *did* surface a head AtP missed at 0.86, finds
   nothing here.) The 9b NULL is now arbiter-confirmed, not instrument-limited.

**Both de-confound controls now RUN.** Net: the 9b "no installed deference head" and the 27b
"gate-don't-delete is 2b-specific" headlines are both arbiter/weight-confirmed, not instrument-limited.

---

## PART 4 — NEXT STEPS (highlighted, 2026-06-19)

The arc is a streak of well-verified **NULLs** — concentration does not scale; RLHF installs no single
deference head. The program's own results now mandate the pivot: stop asking *"which head?"* and ask
**"since it is not one head, what IS the distributed object?"** Plus bank a *positive* claim (the
biological-reuse goal needs one; this arc produced strong negatives).

**NEXT-1 (recommended) — characterize the distributed object: doubt/defer DIRECTION + head-SET.**
The 9b caving is real (9–14 flips) but not single-head-local. Two complementary tests:
  - **direction**: fit a low-rank "cave/defer" direction in the -it residual on the misconception
    caving items (the P-F probe that never ran); ablate / steer it; measure the C-vs-W* margin effect.
  - **head-SET (see the heads-vs-sets note below)**: jointly activation-patch the top-k diffuse heads
    (net_it ≈ 0.03–0.07) together — the per-head NULL does NOT exclude a *jointly-necessary set*.
  A residual direction OR a joint set that mediates caving (where no single head does) is the positive,
  distributed account. Cheap (9b, forward + linear fit/ablation + joint patch; no AtP backward).

**NEXT-2 — close the one honest residual: realized-attention probe.**
27b said "W_QK weights unchanged," but L18.H5's 2b result was a *realized-pattern* collapse and the
induction probe was insensitive. Measure actual attention-to-source on a **content-copy input** (not
induction) at 9b/27b, base vs -it. Settles whether copy heads' realized QK is gated at scale.

**NEXT-3 — fresh lead the data handed us: the 27b OV-magnitude increase.**
RLHF *raised* several 27b copy heads' OV write-magnitude (W_OV_fro +0.34/+0.52, two metrics agree).
Unexplained, RLHF-specific, cross-scale-new: what do they write *more* of, does it matter behaviorally?

Parked (single-family depth): deployment regime, cross-architecture, SFT-vs-RL staging (no Gemma
staged checkpoints).

### Heads vs head-SETS — scope of the NULLs (load-bearing)
Every instrument this arc used (AtP differential, activation-patch confirm, gate_dont_delete) scored
**individual heads**, ranked per-head. It did NOT test head **sets** (joint knockout/patch of several
heads at once). This matters: Arc-1 §3.9/§3.12 + S2 showed the salience effect is carried *jointly* by
a ~12-head set (all-heads necessity ~1.0) while **no single head exceeds ~0.2** — non-additive. So:
- The R1-DIFF result is precisely **"no single installed head (frac_it < 0.10 each)"** — it does **not**
  exclude a jointly-necessary installed *set* whose members are each sub-threshold. The diffuse 9b
  picture (many heads 0.03–0.07) is exactly what such a set looks like.
- Hence NEXT-1's head-SET joint patch is not optional polish — it is the test that would actually
  decide whether the distributed caving is a coordinated set vs genuinely no-attention-locus (MLP /
  residual-direction). It revives the repo's earlier joint-knockout idea on the installed-component
  question.

**Field precedent for head-sets** (so this is the standard unit, not a workaround): IOI is a
~26-head set with **backup name-movers** that compensate under single-head ablation (Wang et al. 2023);
self-repair / Hydra (Rushing & Nanda 2024; McGrath et al. 2023) is exactly why per-head under-measures;
causal scrubbing (Chan et al. 2022) and path patching (Goldowsky-Dill et al. 2023) are the set-level
test methods; **sycophancy itself was path-patched to a ~4% head set (Chen et al. ICML 2024)**; and a
coordinated set that writes one low-rank direction is a **function/task vector** (Todd et al. 2023;
Hendel et al. 2023) — i.e. the set-prong and the direction-prong of NEXT-1 may be the same object.
(Spot-check IDs before external use, per POSITIONING.)

### NEXT-1 head-set test — concrete protocol (the immediate next test)
- **Substrate**: the 16 I1 9b-it caving items, already embedded in `rlhf_differential.py ITEMS`
  (and `atp_low_confirm.py`). Metric: first-token margin lp(C)−lp(W*); restoration frac on the
  counter→neutral_turn gap (the R-4-controlled contrast), per the validated `_confirm` harness.
- **Candidate set**: the diffuse heads from `results_r1_diff/` (the 18 in `atp_low_confirm.HEADS`,
  net_it ≈ 0.03–0.07), i.e. exactly the band a per-head sweep cannot resolve.
- **Joint intervention** (the new bit; the per-head harness patched one head — patch the SET in one
  hook pass): activation-patch the top-k heads *together* (-it counter z[-1] → neutral_turn z[-1] for
  all k), measure the joint restoration frac. Run on base too (set-level differential).
- **Controls / shape**: (a) **cumulative ramp** top-1,2,…,k (à la Arc-1 §3.9) — concentrated if 2–3
  heads carry it, distributed-set if it needs many; (b) **matched-random k-head set** (the specificity
  control — the §3.5/S3 lesson at set level); (c) compare joint frac vs the **sum of individual** fracs
  (quantifies super-additivity).
- **Success criteria (pre-register)**: INSTALLED-SET if joint frac_it ≥ INSTALL_THR (0.10) while every
  member < 0.10 individually AND matched-random-k ≈ 0 AND joint frac_base ≤ BASE_FLOOR. NULL-holds if
  the joint set still does not reach threshold over base (then the locus is non-attention: MLP /
  residual direction → pivot to the direction probe). Either way the per-head NULL's scope is settled.
- **Implementation**: extend the `atp_low_confirm` / `rlhf_differential._confirm` patch hook to set the
  readout `z[-1, H]` for a *list* of heads in one forward (trivial; loop the heads inside the hook).
  Forward-only → fits the 40GB A100. Pair with NEXT-1's direction probe (function-vector view).

- **STATUS (2026-06-19): RUN — `headset_joint_patch.py`, `results_9b_headset/`** (gemma-2-9b base+it,
  A100-40GB us-west-2, ~$2, box self-terminated). Control = joint multi-head/multi-layer activation-patch
  in one forward (list-loop extension of the proven single-head `_confirm`); cumulative ramp top-1..K
  (in-run arbiter order), matched-random-K specificity (`N_RAND=5`, fixed seed, sets identical base/it),
  joint-vs-sum super-additivity, on both models. `--selftest` (5 branches) PASS; faithfulness it_n_ok=10 /
  base_n_ok=9.

  **RESULT — the head-SET hypothesis is VINDICATED; the per-head NULL was set-blind.** Every member
  restores < 0.10 of the cave individually (max **0.067**, exactly why the per-head sweep NULLed at the
  0.10 gate), yet the set JOINTLY restores it: -it ramp climbs monotonically **0.067 (k=1) → 0.448 (peak
  k=15) → 0.358 (k=18)**. Matched-random-K ≈ 0 (it −0.021, base −0.048) → **specific** to these heads, not
  a "patch-k-heads" artifact. So the 9b caving locus IS attention-head-local — as a *distributed set* — which
  refutes the "pivot to MLP/residual-only" reading of the per-head NULL. **The diffuse band of 18 sub-
  threshold heads collectively carries ~36–45% of the cave.**

  **Mechanism = distributed-ADDITIVE, not synergistic.** Joint ≈ cumulative sum of individual fracs (k=5:
  joint 0.247 vs sum 0.259; super_add@K = −0.083, mildly sub-additive from high-k redundancy). So the
  honest framing is "many sub-threshold ~additive contributors, each per-head-invisible" — NOT Hydra/backup-
  name-mover synergy. The §3.9 lesson holds at the level of *thresholding* (per-head under-measures), not
  via super-additivity here.

  **NOT a clean RLHF-install — base-present, strongly -it-AMPLIFIED.** Strict gate fails: joint_base@K
  **0.096 > BASE_FLOOR 0.05** → verdict `SET PRESENT BUT BASE-SHARED`. But the differential is large:
  it-set 0.36–0.45 vs base-set 0.096 (**3.7–4.7×**; differential ramp peaks 0.44 @k=13). Reading: a base-
  present distributed set that post-training **amplifies**, not installs-from-zero — the set-level analog of
  per-head base-shared L38.H14, except the it≫base gap is far larger at set level than it ever was per-head.

  **Load-bearing caveats → `latent_skeptic` triage (NOT yet run):**
  (1) **item mismatch** — it n_ok=10 vs base n_ok=9 pass the `|gap|≥0.5` gate on *different* item subsets,
      so joint_it vs joint_base are not matched; the "base-shared" call rests on this. **The crux to verify
      by running: re-run on the items both models cave on (matched intersection).**
  (2) **negative-gap items included** (ghosts −10.3, microwave −1.98, black-box −2.49 on -it) — not caves in
      the expected direction; a sign-restricted (gap>+0.5 only) re-read may move the fracs.
  (3) joint_base 0.096 over 9 items — stability unknown; bootstrap the base/it gap.
  (4) scope: the set was drawn from the AtP-low band (18 heads); a member outside it is untested.

  **Net:** Part-4's "what IS the distributed object?" gets a concrete positive answer — a specific,
  ~13–15-head distributed-additive attention set, base-present and RLHF-amplified — and the program's
  NULL streak is broken with a *positive* claim (modulo the matched-item de-confound).

#### NEXT-1 DIRECTION prong — `headset_direction.py`, `results_9b_direction/` (RUN 2026-06-19, ~$2, box self-terminated)
Fit a rank-1 cave direction (diff-of-means counter vs neutral_turn) over a layer sweep L{24,28,32,36};
test necessity (ablate the u-projection → cave recovers), sufficiency (steer → cave induced), low-rank
(SVD top-PC fraction), base differential, and set↔direction unification (cosine of the head-set's residual
write with u). `--selftest` 6 branches PASS; it_n_ok=10 / base_n_ok=9.

- **A real, causal, SPECIFIC cave direction exists** (headline L28): ablate recovers **0.503** of the cave,
  steer induces **0.256**, random direction ≈ **0.002**. Necessity is *stronger* than the head-set itself
  (0.50 vs 0.36–0.45). Holds across all four layers (nec 0.39–0.50, suf 0.13–0.27, random ~0 everywhere).
- **But it is NOT a single function vector — it is a higher-rank SUBSPACE.** Top-PC variance fraction only
  **0.33** (< 0.50) at every layer: the per-item cave shifts spread across multiple residual directions;
  diff-of-means captures the dominant axis but ~⅔ of the variance is off it. Caving is a *subspace*, not a line.
- **The head-SET does NOT write this direction** — `set_cos ≈ −0.04` at L28 (orthogonal; max |cos| only
  ~0.34 anywhere). **The set-prong and the direction-prong are DISTINCT, ~orthogonal loci.** The
  function-vector unification hypothesis ("a coordinated set that writes one low-rank direction") is
  **REFUTED** for 9b caving.
- **Same amplified-not-installed signature**: nec_base 0.16 vs nec_it 0.50 (~3×) at L28; the residual
  cave-subspace, like the head-set, is base-present and RLHF-*amplified*, not installed-from-zero.

#### ARC INFLECTION — the distributed object is MULTI-LOCUS (the strong hypothesis)
Three results now cohere into one account of 9b-it misconception caving:
1. **per-head**: NULL — no single installed/concentrated deference head (arbiter-confirmed).
2. **head-set**: a specific ~13–15-head distributed-*additive* attention set carries 0.36–0.45 of the cave
   (every member sub-threshold; matched-random ~0); base-present, RLHF-amplified ~4×.
3. **direction**: a *separate*, ~orthogonal, necessary+sufficient+specific residual cave **subspace**
   (higher-rank, not rank-1) carries ~0.50; also base-present, RLHF-amplified ~3×.
**Strong hypothesis:** 9b-it caving is **multi-locus distributed** — (≥) an attention head-set AND an
orthogonal residual cave-subspace, both *amplified* by post-training rather than installed — and is **NOT
reducible to a single function vector.** This resolves the Part-4 pivot and breaks the NULL streak with a
positive, falsifiable structural claim. The shared, load-bearing crux across all three is the
**amplified-not-installed** call, which rests on the it(10)/base(9) **item mismatch**.

**→ NEXT = matched-item de-confound** (re-run set + direction on the caving intersection, sign-restricted
gap>+0.5) → then `latent_skeptic`. (Ran; see below. The de-confound + a powered re-run OVERTURN this
multi-locus framing — the section above is the pre-de-confound, confounded picture, kept for the record.)

#### latent_skeptic triage of the multi-locus claim (Pass 1, no-GPU, 33 skeptics) — `wf_5274b996`
All three claims (set-installed, direction-base-shared, set⊥direction) came back **NEEDS_RUN**: the n=6
matched de-confound (run before this triage) had no dispersion/CI, an **it-selected** head set (the
"installed" read is partly circular), no base-side faithfulness, and an in-sample direction fit; only
`regime specificity` / `scale` / `ceiling-floor` were RULED_OUT. Dominant cruxes: off-distribution joint
patch, it-selection circularity, n=6 underpower. `run_queue` empty (no existing control settles them).
→ chose to **break the n=6 power wall first** (broaden the substrate) before authoring the deeper controls.

#### POWERED MATCHED DE-CONFOUND (n=41) — `matched_item_deconfound.py --wide`, `results_9b_matched_wide/` (RUN 2026-06-19, ~$2)
Broadened the substrate to a 61-item misconception pool (`misconception_pool.py`: committed 16 + 45 new
single-dominant-competitor items); the both-cave intersection grew **6 → 41**. Re-measured the set-joint
and direction-necessity differentials on the matched set, base vs -it, with paired bootstrap CIs.

- **The head-SET claim COLLAPSES under power.** it set_joint mean **−0.636** (bootstrap CI [−1.13, −0.21]),
  median −0.083, trimmed −0.40; only **10/41 items restore**, **20/41 get WORSE**, min −6.33 — the joint
  18-head substitution is **off-distribution-unstable** (the triage's top crux, confirmed by running). And
  it is directionally REVERSED: the same it-selected set restores *more in BASE* (median **+0.10**, 20/41)
  than in -it. The n=6 "installed 0.147" was a small-sample fluke; the earlier unmatched 0.36–0.45 was
  it-gated-item + small-n + off-distribution artifact. **There is no RLHF-installed/amplified caving head-set.**
- **The cave DIRECTION is the robust, surviving mechanism — and it is BASE-INTRINSIC.** it dir_nec 0.441
  (CI [0.25, 0.64]) vs base 0.472 (CI [0.26, 0.64]); it−base diff CI **[−0.31, +0.27] straddles 0**;
  medians it 0.47 / base 0.58 (base if anything higher). Necessary in BOTH models, **RLHF-neutral**.
  Replicates n=6 → n=41.

#### ARC INFLECTION (corrected, powered) — 9b caving is a BASE-INTRINSIC residual direction; RLHF installs no localizable caving circuit
The whole installed/amplified/multi-locus head-set line does **not** survive a powered (n=41), matched,
sign-restricted de-confound. What robustly survives:
1. **per-head NULL** — no single installed deference head (arbiter-confirmed). *Holds.*
2. **head-SET** — does NOT robustly carry it-caving (median −0.08, 20/41 worsen, off-distribution-unstable;
   if anything base-favored). The "distributed additive installed set" is **retracted**.
3. **residual cave DIRECTION** — necessary ~0.45–0.58 in BOTH base and -it, RLHF-neutral, **base-intrinsic**;
   higher-rank (top-PC 0.33), orthogonal to the head-set. *The real mechanism.*
**Settled hypothesis:** 9b misconception caving is mediated by a **base-intrinsic residual cave-direction**
that post-training neither installs nor amplifies (necessity equal in base and -it). RLHF's contribution to
9b caving is **not a localizable attention mechanism** — the caving capacity is largely base-intrinsic.
This is a clean negative on "RLHF installs sycophancy circuitry at 9b" and a positive on "caving is a
base-intrinsic linear direction," and it *extends* (not breaks) the program's diffuse-NULL line.

**Open (remaining hardening for the surviving claim):** (a) held-out / split-half direction fit (the
in-sample-fit crux — base≈it replication is strong evidence but LOO not yet run); (b) a clean
(non-off-distribution) set intervention — the joint substitution is invalid, so the head-set's true role is
untested by a *stable* method; (c) re-triage the now-powered base-intrinsic-direction claim.

#### NEXT-3 RESULT — RLHF rescales copy-head OV *gain* (same write, scalar only); a bankable POSITIVE — `ov_magnitude_characterize.py`, `results_27b_ovmag/` (RUN 2026-06-19, ~$2, weights-only/CPU)
Pivoted off the settled-negative 9b thread to the 27b OV-magnitude lead (the one positive signal the data
handed us). Characterized the 27b copy basket's W_OV base vs -it, weights-only: scalar-fit residual,
matrix/write-direction cosine, copy-pref, top-vocab overlap. **Every head reads AMPLIFY_SAME** — the OV
change is **pure scalar magnitude scaling of the same write**:
- alpha ≈ 1 + fro_rel exactly (L17.H4 ×**1.52**, L11 kv-group {2,4,7,21} ×**1.33**, L23.H24 ×**0.75** down);
  resid_frac ≤ **0.08**, dir_cos ≈ **1.000**, write_cos ≈ **0.995**, top5 vocab overlap 0.78–0.90.
- still copy heads, gain-tuned: copy_logit tracks alpha (L17.H4 0.41→0.61; L23.H24 0.69→0.52); copy-hit
  rate unchanged. The 4 near-unchanged heads (alpha ≈ 0.97) are internal controls — the metric does not
  spuriously fire.
**Mechanism (with the QK-unchanged result from results_27b_qk):** post-training leaves *where* copy heads
attend (W_QK, rel ~0) and *what* they write (OV direction + promoted vocab) **untouched**, and only
**rescales the OV write-gain** of select copy heads (amplify or attenuate). RLHF tunes copy-head
write-strength; it does **not** install or redirect a computation. This is the program's first clean
**positive**, RLHF-specific, cross-scale-new mechanism.
- **Caveats:** (i) gemma-2-it = SFT+RLHF+merge, so the base→it weight diff conflates stages (B5) — "scalar
  gain change" is a weight fact, stage-attribution is not claimed; (ii) **behavioral relevance untested** —
  this is weights-only; whether the gain change moves logits/behavior needs a 27b forward (NEXT-3b/NEXT-2 below).

#### NEXT-3b — behavioral scale-ablation (INCONCLUSIVE) — `ov_behavioral_scale.py`, `results_27b_ovbehav/` (RUN, H100-80GB, ~$2)
On -it 27b, per head scale hook_z by 1/alpha (restore base OV gain) + knockout (z=0), on random-token
induction; Δ logit of the copied token. **Result: noisy, sign-inconsistent, NOT a behavioral claim.**
induction copy_acc only 0.56 (n=16); knockout signs mixed (these heads are not uniform copy-promoters);
knockout-vs-scale non-monotone for several heads (e.g. L17.H4 knockout +0.16 but scale −0.28). Effects
≤0.54 logits, noise-dominated. The verdict labels ("MATTERS") are not trustworthy at this variance.

#### NEXT-2 — realized attention-to-source (DECISIVE) — `realized_attention.py`, `results_27b_realattn/` (RUN, H100-80GB, ~$2)
Content-copy probe ("The secret word is {w}. ... The secret word is" → copy {w}), measure each basket
head's attention readout→source, base vs -it. **Result: NONE of the 10 basket heads attend the source**
— attn 0.000–0.028 in BOTH base and -it (all ≪ 0.10 floor) — while **copy_acc = 1.00 in both models.**
The model copies perfectly, but **not via these heads.** This:
- **Explains NEXT-3b.** The OV-gain heads do not realize copying on either induction or content-copy →
  scaling their OV gain produces only noise. Two independent probes agree: the weights-only copy basket is
  **latent, not realized**, at 27b.
- **Qualifies NEXT-3 hard.** RLHF's OV-gain rescaling is a real WEIGHT change to **copy-capable-but-
  unrealized** heads — no demonstrated functional/behavioral consequence on tested copy tasks.
- **NEXT-2 original (QK gating at scale): cannot be claimed** — the QK_GATED_AT_SCALE tags are denominator
  artifacts (both attns ≈0; rel of 0.01→0.003 is noise-floor). There is no realized attention to gate;
  attn≈0 in base too. The realized-QK-gating question is moot for these heads (they don't attend source in
  either model).

#### 27b OV ARC — CLOSED (bounded conclusion)
RLHF rescales the OV write-gain of specific 27b copy-CAPABLE heads (pure scalar, same direction: NEXT-3,
robust). But those heads do **not realize copying** on induction or content-copy probes the model solves
perfectly (NEXT-2), so the gain change is **behaviorally latent** on tested tasks (NEXT-3b inconclusive,
NEXT-2 explains why). The NEXT-3 "positive" is downgraded to: *a real RLHF weight-modification of latent
copy-OV machinery, functional role unestablished.* Open: find an input where these heads DO realize copying
(if any) before attributing function; or treat the OV-gain change as latent/epiphenomenal.

---

## PART 5 — 2b QK weight-vs-realized + colleague-surfaced directions (2026-06-19)

### qk_weight 2b — RLHF edits NO QK weight; the 2b copy collapse is residual-INPUT-mediated
`controls/qk_weight_2b_l18h5.py`, `results_2b_qkweight{,2,3}/` (3x gpu_1x_a10 us-east-1, ~$1.2,
`--device cpu`, boxes self-terminated). Authored claim-blind via `triage-author`, then extended twice
(QK-direction decomposition; query/key/full residual swaps). `--selftest` PASS each version.

**Question (the §8 scope challenge).** FRAMING §8 says "RLHF deletes the copy *at the weights*" (L18.H5
attn-to-anchor 0.84->0.01, complete already in the it/bare fragment). Is that a weight edit to L18.H5's
own QK, or is the head's QK intact and the change upstream (input-driven)?

**Result (L18.H5; L18.H6 = matched control, identical pattern):**
- **QK weights intact** — fro_rel **-0.5%** (magnitude) AND **dir_cos 0.998 / resid_frac 0.069** (direction).
  No magnitude edit, no rotation. The earlier W_QK-magnitude-only read is now closed on both axes.
- **Realized attention collapses** — base **0.578** -> it **0.016** (reproduces §8's 0.58->0.016 across all
  3 runs = the faithfulness gate; the 0.84 in §8 was the Australia pair, 0.58 is the 5-pair mean).
- **Collapse is 100% residual-input-mediated** — swapping the full `resid_pre[L18]` base->it recovers
  **0.97**; L18's weights are exonerated end to end.
- **The input change is JOINT (super-additive)** — query-residual alone recovers **0.42**, key-residual
  alone **0.02**, full **0.97**. Neither side alone restores the anchor focus; both together do (softmax:
  -it's query no longer points at the anchor AND the competing context shifted).

**Verdict `INPUT_MEDIATED`.** §8's "deletes at the weights" is **over-tight** — post-training does not
touch L18.H5; it changes the residual stream feeding layer 18, jointly across the query and context
positions. Closes, by running, the two confounds raised in review (`qk_rotation_not_excluded`,
`single_position_swap_underestimates`).

**Cross-scale coherence (with NEXT-2 / NEXT-3).** Copy-head *weights* are never edited at any scale:
W_QK intact at 2b (this) and 27b (`qk_collapse_metric`); OV direction intact at 27b, only gain rescaled
(NEXT-3). Where copying IS realized (2b L18.H5, base attn 0.58) RLHF removes it via the head's *input*;
at 27b the basket heads do not realize copying at all (NEXT-2, latent). **Unifying read: RLHF modulates
the inputs to / the gain of base copy machinery; it edits none of its routing weights.**

**Triage status:** ready for `latent_skeptic`. Residual confound = n=5 pairs, no CI (mean recoveries only).
Scope: 2b only — the concentrated reader is 2b-specific (FRAMING §10.3), so this explains the cleanest
case, not a scaling law.

### Colleague-surfaced directions — critical evaluation vs total results

**The total-results state these sit against.** One surviving *positive* (9b caving = a base-intrinsic
residual cave-direction, RLHF-neutral); three verified *nulls* (no installed deference head; head-set
retracted under power; gate-don't-delete L18.H5-specific); two *modulation* findings (27b OV gain
rescaled but latent; 2b QK collapse input-mediated; copy-head weights never edited). Emergent theme:
**RLHF modulates base machinery, gated by confidence — it installs no localizable new circuitry.** The
risk that comes with it: an *"everything is confidence"* overfit — confidence/headroom is now invoked to
explain every null — while confidence has **never been localized as a representation**.

**D (do first) — harden the surviving positive against the linear-probe illusion.** The cave-direction
(the only positive) rests on diff-of-means + ablate/steer, the failure mode of Makelov et al. 2023;
triage already flagged the in-sample-fit crux OPEN. Cheap, 9b, reuses `headset_direction.py`.
- **SC-D1 (held-out):** fit the cave-direction on train items, test necessity on disjoint items; PASS if
  out-of-sample necessity sits within the in-sample CI.
- **SC-D2 (feature-decomposition):** decompose the direction at its layer into GemmaScope SAE features;
  PASS if a small interpretable set reconstructs it (direction -> circuit, the granularity the
  attribution-graph method gives but a bare probe does not).
- **SC-D3 (counterfactual, Makelov):** the direction reads ~0 on matched no-cave items, and steering
  produces the correct downstream change, not merely the target-logit nudge.
- Until D passes, "caving is a base-intrinsic direction" is a *direction*, not a *mechanism*.

**C (then) — localize the confidence / factual-recall representation; test the unifier.** Every result
reduces to confidence-headroom; hypothesis: RLHF acts on the *confidence* representation, not a caving
circuit. Fit a confidence direction (diff-of-means high- vs low-margin) with D's guards baked in. Cheap,
9b, reuses the harness; highest strategic leverage.
- **SC-C1:** necessity/sufficiency of the confidence direction, base vs -it (does post-training move it?).
- **SC-C2 (the crux):** project the cave-direction onto the confidence-direction. High |cos| -> same axis
  -> the program collapses to *"post-training tunes confidence, not deference"*; low cos -> caving and
  confidence are distinct objects. MUST carry D's illusion-guards or it deepens the overfit.

**A (if the 2b thread is worth finishing) — query-trace upstream of L18.H5.** qk_weight showed the 2b
collapse is input-mediated; a path-patch to L18 `resid_pre` base-vs-it finds *which* upstream component
moved the input. Cheap, 2b.
- **SC-A1:** identify the upstream sender(s) whose base->it residual write accounts for the query-side
  recovery (path-patch / DLA on the L18 query input + the key-side residual).
- Critical limit: **2b-only** (no concentrated reader at scale) -> explains the cleanest case, no scaling
  law. Lower priority than D/C.

**B (scope decision required) — RLVR / verifiable-reward model.** The best available *test* of the
confidence hypothesis, and it unblocks stage-attribution (RLVR families ship staged checkpoints, the B5
blocker). Substrate = numeric-assertion copy (survives to 9b, uncertainty-gated).
- **SC-B1:** in a same-family base vs RLVR pair, numeric caving drops tracking the per-item
  verifiable-confidence rise, and NOT on non-verifiable factual items.
- Cost: **breaks the single-family Gemma scope**, needs external models. GATED behind an explicit
  decision — not a proceed.

**Disciplined ordering (from the results): D -> C -> A -> B.** D/C/A are cheap and in-scope (2b/9b,
existing harnesses); B is the strategic bet that spends the single-family discipline. Sharpest caution:
the program is one LOO-test (D) from knowing whether its one positive is mechanism or overlay — do D
before building the confidence-unifier (C) on top of it, or "confidence-gating" risks becoming an
unfalsifiable catch-all.

---

### Handoff seed for the next agent
> /karpathy-guidelines
>
> NEXT-1 ran, was triaged, and a POWERED (n=41) matched de-confound corrected it. The installed/amplified
> multi-locus head-set picture is **retracted**: under power the joint 18-head patch is off-distribution-
> unstable (it set_joint −0.64, 20/41 items worsen) and if anything base-favored — there is **no
> RLHF-installed caving head-set** at 9b. What robustly survives: 9b misconception caving is a
> **base-intrinsic residual cave-direction** (necessity ~0.45–0.58 equal in base and -it, RLHF-neutral,
> it−base bootstrap CI straddles 0). RLHF installs no localizable caving circuit at 9b. Then PIVOTED to
> NEXT-3 (27b OV-magnitude) and banked a clean POSITIVE: RLHF **rescales copy-head OV gain** (alpha
> ×1.33/×1.52 up, ×0.75 down; resid_frac ≤0.08, dir_cos ≈1, same promoted vocab) while leaving W_QK and the
> OV write-direction untouched. BUT NEXT-3b (behavioral scale-ablation, inconclusive) + NEXT-2 (realized
> attention, DECISIVE) then showed those heads do **not realize copying** (attn-to-source ≈0 in base AND
> -it while copy_acc=1.0) — so the OV-gain change is a real weight-mod of **latent copy-OV machinery**,
> functional role unestablished. The 27b OV arc is CLOSED at that bounded claim. Remaining: find an input
> where these heads DO realize copying (else treat as latent/epiphenomenal); 9b held-out direction fit
> (LOO); a stable (non-off-distribution) 9b set intervention; re-triage of the surviving committed claims.
>
> Then banked **qk_weight 2b** (PART 5): RLHF edits NO QK weight of L18.H5 (dir_cos 0.998, fro_rel -0.5%);
> the 2b copy collapse is **100% residual-input-mediated** (full resid_pre[L18] swap recovers 0.97), JOINT
> across query (0.42) + key (0.02) + super-additive. So "RLHF deletes the copy at the weights" is
> over-tight at 2b too — copy-head routing weights are untouched at every scale; RLHF modulates their
> inputs/gain. PART 5 pre-registers the four colleague-surfaced directions in priority order:
> **D** harden the cave-direction vs the linear-probe illusion (held-out/LOO + SAE feature-decomp +
> counterfactual) -> **C** localize the confidence representation and project the cave-direction onto it
> (the unifier test) -> **A** 2b query-trace upstream of L18.H5 -> **B** RLVR confidence test (scope-gated,
> breaks single-family). Sharpest caution: the program is one LOO-test from knowing if its one positive is
> mechanism or overlay; do D before building C on it, or "confidence-gating" becomes an unfalsifiable catch-all.

---

## PART 6 — NEXT (mechanistic confidence): entropy-neuron identification on Gemma-2 (2026-06-20)

### Why this is the move
The logit-lens early-encoding line is **CLOSED as a negative** (`results_9b_logitlens_attr/`,
`controls/logit_lens_attribution.py`): the early it>base margin was mostly a **chat-template format
artifact** (native it-base +9.34 -> format-matched +0.89 on Q/A, +3.45 on chat) and the early logit-lens
is **unfaithful** (early_argmatch 0.0 in BOTH models -> the early margin does not predict the output, so
it is not "early confidence"). A logit-lens margin cannot measure confidence (unfaithful intermediate,
format-confounded, conflates knows-vs-expresses). Claim 2 ("it more challenge-robust") **retracted**
(frac-erosion diff CI [-4.47, 0.31] includes 0 after normalization). Cross-lens = null (calibration not
the driver).

Two literature searches (`POSITION_UNCERTAINTY` 2026-06-20) then established: (a) the well-supported
knowledge predicate is **multi-paraphrase consistency + calibration + knows/expresses split**, NOT
single-prompt greedy (the repo's current gate); (b) the literature's **causal** confidence mechanism is
**ENTROPY NEURONS** (Stolfo/Gurnee, NeurIPS 2024, arXiv:2406.16254; causal via mean-ablation) -- established
on GPT-2/Pythia/Phi-2/**Gemma-1**/LLaMA2 but **NOT Gemma-2**. The one Gemma-2 causal result (Semantic
Entropy Neurons, NeurIPS 2024 MINT workshop) targets *semantic* (multi-generation) entropy, not single-pass
confidence. So a root single-token confidence mechanism on Gemma-2 is an **OPEN, citable gap** -- and it is
mechanistic (root mechanism), causal (ablation, NOT a probe), single-family (Gemma), method-idiom-aligned.
It replaces the discredited logit-lens with the right instrument.

### Unit ENTROPY-NEURON — causal confidence neurons on gemma-2-9b (base + it)
- **Identify** late-layer MLP neurons whose output direction `w_out` writes into the unembedding's
  low-sensitivity ("null") subspace (the bottom singular directions of `W_U`) -- the entropy-neuron weight
  signature -- then **validate CAUSALLY by mean-ablation**: a true entropy neuron modulates output
  **entropy** with little change to next-token **loss** (~~the ~30x dEntropy/dLoss ratio, Stolfo/Gurnee~~
  — **CORRECTION 2026-06-20:** the paper reports **no explicit ratio**; it shows the effect qualitatively
  ("minimal impact on the prediction", Fig. 1/2c). The "~30x" was our over-precise paraphrase / selftest-gate
  framing, **not** a figure Stolfo/Gurnee state — verified vs arXiv:2406.16254 HTML, per our own
  "spot-check every external method before relying on it" rule. Original misquote struck-through, not deleted).
- **base vs -it differential**: are the same neurons entropy-regulators in both, and does post-training
  change their ablation-effect? (the late-mediation hypothesis, now with a faithful causal instrument.)

**Pre-registered SCs:**
- **SC-EN-1 (existence/replication):** >=1 late MLP neuron with null_frac >= NULL_TOL AND mean-ablation
  dEntropy >= ENT_TOL AND |dLoss| <= LOSS_TOL -> a causal entropy neuron exists on gemma-2-9b (the Gemma-2
  replication of Stolfo/Gurnee; novel -- closes the gap). NULL = none found (also informative).
- **SC-EN-2 (base/it differential):** compare the entropy-neuron set + per-neuron dEntropy base vs -it,
  against a matched-random-neuron control; RLHF-mediated iff the set / ablation-effect differs clear of
  control.
- **SC-EN-3 (caving link, GATED on EN-1):** does mean-ablating the entropy neuron(s) shift caving on the
  misconception substrate? ties the confidence mechanism to the sycophancy behaviour.
- **Controls:** matched-random late-layer neuron set (specificity); the |dLoss| <= LOSS_TOL gate (it is
  entropy regulation, not capability damage). Forward-only -> A100. Claim-blind author + `latent_skeptic`.

**Knowledge predicate upgrade (folds in here):** drop single-prompt greedy; gate items by robust
multi-paraphrase greedy-correct + report final-layer (faithful) P(correct)/margin/entropy + calibration,
keeping *knows* separate from *expresses* (per the lit search). Wary of the bare margin (it just failed).

### Handoff (PART 6)
> Logit-lens confidence line CLOSED (negative: format artifact + unfaithful early lens; claim-2 retracted).
> Lit search: causal confidence mechanism = entropy neurons (Stolfo/Gurnee 2406.16254, mean-ablation), done
> on Gemma-1 not Gemma-2 -> open gap. NEXT = port entropy-neuron identification + causal mean-ablation to
> gemma-2-9b (base+it), then link to caving. RLVR parked (`RLVR_SKETCH_PARKED.md`, breaks single-family).
> Knowledge gate upgraded to multi-paraphrase + calibration + knows/expresses split.

### PART 6 RESULT — entropy-neuron NULL (hardened) — `controls/entropy_neuron_gemma2.py`, `results_9b_entropyneuron{,_powered2}/`
First run (mean-ablation, short ref, K=10): EN count 0 both models; `latent_skeptic` (wf, 6 confounds) flagged it
underpowered -- top cruxes mean-ablation-preserves-the-mean and short-vs-long-context regime.
**Powered re-run (2026-06-20, A100):** BOTH mean + **zero**-ablation, **long-context** WikiText-2 (20x256 =
5120 positions), **K=50**. **NULL HOLDS:** ENTROPY_NEURON count **0 in base and -it**; the high-null_frac
late neurons (null_frac up to 0.92) move output entropy negligibly (|dEntropy| <= 0.008 nats under both mean
and zero ablation, vs ENT_TOL 0.05) and are indistinguishable from matched-random (zero-abl avg ~0.0004).
**The Stolfo/Gurnee single-neuron entropy regulator does NOT replicate on Gemma-2-9b** under the de-confounded
protocol. The zero-ablation + long-context cruxes are resolved by running; the null is now ablation-mode- and
regime-robust. **Open (deferred):** joint group-ablation (distributed regulator) + pre-softcap entropy --
the *distributed* alternative is untested and is the leading remaining explanation.

### STOCK-TAKE (2026-06-20) — caving, factual recall, and "confidence" cohere into one negative
This session's mechanistic observations, and the prior arc, converge on a single picture:
- **2b copy collapse (`qk_weight`):** RLHF does not edit L18.H5's QK weights (magnitude AND direction intact,
  dir_cos 0.998); the realized copy collapse is 100% residual-INPUT-mediated (full resid_pre[L18] swap recovers
  0.97), joint query+key. RLHF changes what *feeds* the head, not the head.
- **27b copy heads (NEXT-2/3):** W_QK unchanged; OV gain rescaled but the heads are LATENT (do not realize
  copying). Copy-head routing weights are edited at no scale.
- **Early "confidence" (logit-lens, de-confounded):** the apparent it>base early margin was a chat-template
  FORMAT artifact (+9.34 -> +0.89 format-matched) and the early logit-lens is UNFAITHFUL (early_argmatch 0.0).
  A logit-lens margin cannot measure confidence; no clean early base-vs-it confidence difference survives.
  Claim that "-it is more challenge-robust" retracted (baseline-magnitude artifact).
- **Entropy/confidence neuron (PART 6, this session):** no single-neuron entropy regulator on Gemma-2-9b
  (hardened null).
- **Caving (prior arc):** 9b misconception caving = a base-intrinsic residual cave-DIRECTION (necessity
  ~0.45-0.58, RLHF-neutral, higher-rank subspace); per-head NULL; head-set retracted under power.

**The through-line: NO SINGLE LOCUS AT SCALE.** Every candidate mechanism -- the copy reader (concentrated at
2b, diffuse at 9b), the installed deference head (null), the jointly-necessary head-set (retracted under
power), the early-readable confidence representation (format artifact + unfaithful), the single entropy/
confidence neuron (null) -- comes back **diffuse / distributed / base-intrinsic**, never a clean RLHF-installed
single locus. RLHF modulates the **inputs to / gain of / late expression of** base-intrinsic machinery; it does
not install localizable new circuits.

**For caving + confidence specifically:** the two literature-standard confidence mechanisms -- an early-readable
representation (logit-lens) and a single-neuron entropy regulator (entropy neurons) -- BOTH fail to materialize
cleanly on Gemma-2-9b. So if confidence gates caving, it does so via a **distributed / base-intrinsic** substrate,
not a single neuron or an early-readable axis -- consistent with the one robust positive (caving = a
base-intrinsic, higher-rank residual direction). The "RLHF mediates confidence at late layers" hypothesis is
**neither supported nor cleanly testable** with the instruments tried (logit-lens unfaithful; no entropy neuron
to ablate). Honest state: **confidence on Gemma-2-9b appears distributed**; its link to caving runs through the
base-intrinsic cave-direction, not a localizable confidence mechanism.

**Next instruments that fit this picture (all distributed-aware):** joint group-ablation of the high-null_frac
set (distributed entropy regulator); pre-softcap entropy; and the still-unrun **D** (held-out/LOO + SAE
feature-decomposition of the cave-direction -- turn the one positive from a direction into a circuit).

---

## PART 7 — confidence-mechanism + cave-direction hardening (pre-registration, 2026-06-20)

Pre-registered BEFORE running, repo idiom (`SC-N`, faithfulness gate first, matched controls, honest-null,
no goalpost moves). Authored claim-blind (`triage-author`), reviewed, model-free selftest, then run, then
`latent_skeptic`. Three controls discharge the four colleague-surfaced tests (a/b/c/d) plus the cruxes a
Pass-1 `latent_skeptic` triage (`wf_781b6f41`, 24 skeptics, no-GPU) raised against the two load-bearing
committed claims (PART 6 entropy NULL; PART 3 cave-direction). Cited cruxes are folded in below verbatim.

### Why these, from the triage (the deciding cruxes, not the story)
- **Entropy NULL is robust at the single-neuron grain.** 7/11 confounds RULED_OUT (`off-distribution`,
  `selection`, `readout`, `regime`, `construct`, `noise floor`, `ceiling/floor`). The crux that closes
  selection: "the HIGHEST-null_frac late neurons (0.82-0.86) ... were ablated. A null at the top of the
  predicted distribution is pro-mechanism selection, not adverse selection." The only LIVE alternatives are
  three NEEDS_RUN, each a different grain the single-neuron sweep is blind to: `pre-softcap` ("A candidate
  whose pre-softcap dEntropy >= 0.05 could be compressed below the observed <=0.008"), `joint/group`
  ("No committed number bounds the joint effect of the top-K"), `feature-grain SAE`, and `self-repair`
  direct-effect ("neither isolates a DIRECT pre-compensation write to the null directions").
- **Cave-direction "base-intrinsic" is NOT yet established — two EXPLAINS dents + five NEEDS_RUN.**
  `selection bias` EXPLAINS: "the both-cave intersection (n=41) -- items selected precisely because base
  AND -it both cave. Conditioning on equal behavior manufactures the equal-necessity result." `confidence/
  headroom` EXPLAINS: "a base-intrinsic, RLHF-stable confidence axis predicts exactly this equality, so the
  null IS the confound producing the claim." The necessity magnitude is `in-sample diff-of-means` (no
  held-out), and `regime specificity` NEEDS_RUN: "Two separately-overfit regime-specific directions of
  equal magnitude reproduce this exact pattern. No cross-regime transfer number (base-derived direction
  tested in -it) ... on record." So the headline rests on an un-hardened, possibly-circular fit.

### Control 1 — `controls/entropy_distributed_presoftcap.py` (tests a + b; discharges 3 entropy NEEDS_RUN)
**Question.** Does the entropy NULL survive (a) a PRE-softcap readout and (b) JOINT group-ablation of the
high-null_frac set? (Both are grains the single-neuron post-softcap sweep cannot see.)
**Procedure.** Reuse the weights-only null-space screen (`null_basis`/`null_frac`/late-layer) from
`entropy_neuron_gemma2.py` (do not edit it). Top-G high-null_frac late neurons ablated TOGETHER (group
mean- AND zero-ablation), cumulative ramp G=1,2,4,8,…,K, vs a matched-random-G non-candidate set, on
long-context WikiText-2. Entropy computed BOTH pre-softcap (logits before the gemma-2 final-logit softcap,
cap=30) and post-softcap. Base and -it.
**Pre-registered SCs.**
- **SC-EN-D1 (distributed):** group |dEntropy| >= ENT_TOL (0.05) at some G while each member < ENT_TOL
  AND group |dEntropy| exceeds matched-random-G by >= MARGIN → a DISTRIBUTED entropy regulator exists
  (the single-neuron null does not generalize). NULL = group also sub-threshold → regulator is not in this
  neuron set at any group size (hardens PART 6).
- **SC-EN-D2 (pre-softcap):** if pre-softcap |dEntropy| >= 0.05 while post-softcap < 0.05 for the same
  candidates → the PART-6 null was a softcap-readout artifact (retract); if pre ≈ post (both < 0.05) →
  null is softcap-robust (hardens PART 6). Settles the `post-softcap vs pre-softcap` NEEDS_RUN by running.

### Control 2 — `controls/cave_direction_heldout.py` (test c, SC-D; discharges the cave-direction existence + base-intrinsic cruxes)
**Question.** Does the cave-direction's necessity survive OUT-OF-SAMPLE, and is it the SAME direction in
base and -it (base-intrinsic) or two separately-overfit regime-specific fits?
**Procedure.** Extend the `headset_direction.py` diff-of-means + necessity/sufficiency + random-control
machinery with: (1) **held-out / LOO** — fit `u` on train fold, test necessity on disjoint test fold /
leave-one-out; (2) **cross-regime transfer** — apply base-derived `u` in -it and -it-derived `u` in base;
(3) **label-permuted in-sample null** — refit diff-of-means on permuted condition labels, the control the
random-direction (~0.002) does NOT give for fitting degrees of freedom.
**Pre-registered SCs.**
- **SC-D1 (held-out):** out-of-sample necessity within the in-sample bootstrap CI AND clear of the
  label-permuted null → the cave-direction is real, not in-sample overfit. (PART 5 SC-D.)
- **SC-D-XREG (base-intrinsic):** base-fit `u` retains necessity ≥ DIR_THR in -it (and vice-versa), and
  the cross-regime necessity ≈ within-regime → "base-intrinsic / RLHF-neutral" earned, not an intersection
  artifact. Cross-regime necessity ≈ 0 (or ≪ within-regime) → two regime-specific directions; the PART-3
  "base-intrinsic" headline is **retracted** to "each regime has its own equally-necessary fit." Settles
  the `regime specificity` + `selection bias` cruxes that committed data could not.

### Control 3 — `controls/confidence_vs_cave_direction.py` (test d, SC-C; discharges the confidence/headroom EXPLAINS)
**Question.** Is the cave-direction a DEFERENCE signal or a CONFIDENCE/headroom axis? (The unifier test.)
**Procedure.** Fit a margin/confidence direction by diff-of-means (high- vs low-|M| items, margin-stratified
from `misconception_pool.py`); measure its necessity/sufficiency (held-out, same guards as Control 2);
compute cosine(cave_dir, confidence_dir) per layer; and test cave-direction necessity on NON-caved / single-
cave items (where confidence and deference are NOT collinear by construction — the dissociation the both-cave
intersection forbids).
**Pre-registered SCs.**
- **SC-C1:** the confidence direction is necessary/sufficient out-of-sample, base vs -it (does post-training
  move it?).
- **SC-C2 (the crux):** `High |cos| -> same axis -> the program collapses to "post-training tunes
  confidence, not deference"; low cos -> caving and confidence are distinct objects.` MUST carry Control 2's
  held-out + label-permuted guards or it deepens the overfit.
- **SC-C-DISSOC:** cave-direction necessity on non-caved/single-cave items separates deference from
  confidence; if necessity vanishes off the both-cave set, the direction was the confidence axis the triage
  flagged (`conditioning on caving makes confidence and deference collinear by construction`).

### Ordering, compute, gates
- **Run order: Control 2 → Control 3 → Control 1.** Control 2 (held-out + cross-regime) is the single
  highest-value test — it decides whether the program's ONE positive is mechanism or overlay, and whether
  "base-intrinsic" survives at all. Control 3 builds on Control 2's hardened direction (do not build the
  unifier on an un-LOO'd direction — the PART 5 caution). Control 1 hardens an already-robust null, so it
  is last.
- **Faithfulness gate (each):** reproduce the committed numbers before adding the new arm — Control 1 the
  PART-6 single-neuron post-softcap dEntropy (≤0.008); Control 2/3 the I1 flip counts + in-sample necessity
  (0.441/0.472) that the held-out/cross-regime arms are compared against.
- **Compute:** all forward-only (diff-of-means, projection edits, group-ablation; no backward) → A100-40GB,
  per the `entropy`/`headset` precedent; box self-terminates. Authored claim-blind; every surviving SC →
  `latent_skeptic` (the runner verifies the `run_queue` items — LOO and group-ablation are existing-script
  extensions; the cross-regime/label-permuted/confidence arms are the `author_queue`).
- **GATED follow-ups (not authored now, pre-registered):** SAE-feature-grain entropy ablation (GemmaScope,
  heavier dependency); self-repair direct-effect DLA with LayerNorm frozen; downstream-engagement patch of
  the cave-direction into other items. Authored only if Controls 1–3 leave the grain open.

### PART 7 addendum (gated) — cave-direction → attribution graph (colleague Q, 2026-06-20)
Q: "have we considered making caving (our cave direction) an attribution graph again?" — i.e. pull the
program back to its founding method (the `attribution-graphs-experiment` branch) and turn the one positive
from a *direction* into a *circuit*. This is already the PART-6 stock-take's stated next step ("turn the one
positive from a direction into a circuit") and the unit-correct (feature, not neuron/direction) grain that
Anthropic model-biology mandates (Lindsey et al. 2025 *Biology of an LLM* / Ameisen et al. 2025 *Circuit
Tracing*; superposition → features, Elhage 2022 / Bricken 2023). **Recommended — but GATED, two gates:**
1. **Result gate (the "consider LATER"):** trace nothing until Control 2 says the cave-direction is real
   out-of-sample + base-intrinsic (SC-D1/SC-D-XREG) AND Control 3 says it is DEFERENCE not a confidence axis
   (SC-C2/DISSOC). Tracing an in-sample-overfit or a confidence axis = wasted graph.
2. **Tooling/scale gate (spot-check, web 2026-06-20):** `circuit-tracer` (decoderesearch) builds attribution
   graphs from (cross-layer) MLP transcoders and **fully supports Gemma-2-2B** now (GemmaScope transcoders,
   PLT + newly CLT; ~15GB). **Gemma-2-9B is NOT in the shipped support** — GemmaScope ships 9b *SAEs* but the
   tracer's bundled transcoders are 2b; a 9b graph needs 9b transcoders sourced/trained (heavier). Tension:
   caving is established at **9b** (2b saturates), so either (a) first check whether 2b-it caves on the
   misconception substrate and trace the cheap, fully-tooled 2b, or (b) wire GemmaScope-9b transcoders into
   the tracer for the established-9b substrate.
**Why caving fits attribution graphs better than the copy-head arc did:** I2 flagged that attribution graphs
*freeze attention*, which blocked head-level QK claims — but the cave-direction is a **residual subspace,
orthogonal to the (retracted) attention head-set**, so it is MLP/residual-mediated and the attention-freeze
does not bite. Caving is a good-fit target; the QK copy arc was not.
**Two-step path (post-results, cheap→full):**
- **Step A (SAE only, no transcoder; = SC-D2):** decompose u_cave into GemmaScope-**9b** SAE features at its
  layer — direction → interpretable feature-set. Needs only the 9b SAEs (exist), not transcoders. Minimal,
  in-scope, do first.
- **Step B (full graph):** trace the caving logit-diff (the M drop under the challenge turn) with cross-layer
  transcoders, cave-direction features as the output nodes; validate the graph on a known case (program
  idiom). Feasible at 2b now (if 2b caves); at 9b pending transcoder sourcing.

### PART 7 follow-ups — deference-vs-confidence + is-caving-9b (colleague Qs, 2026-06-20)
**Q1 — deference vs confidence (the distinction Control 3 exists to make).** Operationally distinct objects:
- **Confidence / headroom** = the model's PRE-pressure certainty about the fact, = the neutral-turn margin
  M = lp(C) - lp(W*). Gated by the model's own knowledge state. A confidence axis separates high-|M| from
  low-|M| items *regardless of social context*. It says WHERE the model CAN be moved (susceptibility).
- **Deference / caving** = the answer SHIFT caused by the user's social pressure, = M(challenge) - M(neutral).
  Gated by the social cue (user doubt/assertion), not the fact. It is the act of moving when pressed.
- **They are conceptually orthogonal** (epistemic state vs social-response mechanism) but **empirically
  collinear on the both-cave intersection BY CONSTRUCTION** (the triage's `selection bias`/`confidence`
  EXPLAINS): items are kept *because* both models cave -> those are exactly the low-confidence items, so
  u_cave (counter-vs-neutral) and a confidence axis (high-vs-low margin) point the same way there. The
  distinguishing prediction: a pure-confidence account predicts **no caving on high-|M| items**; a
  deference account predicts caving even where the model is confident (the "scary half" / bare-doubt
  sycophancy). SC-C2 (cos(u_cave,u_conf) on independently-fit dirs) + SC-C-DISSOC (u_cave necessity OFF the
  both-cave set) are exactly this dissociation. High cos AND cave dies off the low-confidence set => caving
  IS confidence (program collapses to "RLHF tunes headroom"); low cos AND cave survives on confident items
  => deference is a distinct mechanism.
**Q2 — is caving a 9b phenomenon? (partly OPEN; launched the decisive check.)** What is established:
caving is MEASURED at **9b** (9b base 9-14 misconception flips, `DESIGN_9b §I1`; R1-DIFF faithfulness 9
it-flips; the n=41 both-cave intersection is 9b). It is **NOT shown to be 9b-EXCLUSIVE** — whether 2b-it /
27b-it cave on the same items was flagged opportunistic and never committed. Principled reason it may be a
**capability x substrate WINDOW**, not a scale: a right->wrong cave needs (a) the model KNOWS C, (b) at a
low-enough margin to be movable, (c) under pressure. 9b-on-misconceptions hits all three; capitals/arithmetic
**saturate** (can't be moved, the §10.1 ceiling); 2b may sit **below the capability floor** (can't cave what
it never knew -> that is incapacity, not deference). So caving is best read as pinned at 9b-on-misconceptions
with scale-breadth open. **Launched** `run_2b_cavecheck.sh` (`job_truthful_flip.py` 2b base + 2b-it on
TruthfulQA, `results_2b_cavecheck/`, a10): flip count vs the 9b 9-14 ref answers it directly, and the SC-B
per-head knockout bonus says whether a 2b flip recruits L18.H5. If 2b caves -> the attribution-graph route
(Step B) runs on the fully-tooled 2b; if not -> caving is capability-gated and the graph needs 9b transcoders.

### PART 7 RESULTS (2026-06-20) — 4 boxes, parallel, self-terminated, ~$8
All forward-only; selftests gated each run. `latent_skeptic` Pass-2 running on the 5 load-bearing claims
(`wf_fd82127d`); de-confound queue folded below when it lands.

**Control 2 — cave_direction_heldout (`results_9b_cavedir/`) — the decisive test, two-part verdict.**
- **SC-D1 (held-out): PASS both — the cave-direction is REAL, not an in-sample probe artifact.** Headline L36:
  it held-out k-fold **0.963** / LOO **0.988** vs in-sample 0.989 (within CI [0.32,1.64]); base k-fold
  **0.703** / LOO 0.710 vs in-sample 0.711 (within CI [0.46,0.98]). Label-permuted-fit null ~**0.13** both;
  random ~**0.005**. **The linear-probe-illusion / in-sample-fit crux (PART 5 SC-D, triage's top cave crux)
  is KILLED by running.** The program's one positive survives hardening.
- **SC-D-XREG (base-intrinsic): FAIL — REGIME_SPECIFIC both.** base->it cross 0.422 / within 0.963 (ratio
  **0.44**); it->base cross 0.229 / within 0.703 (ratio **0.33**); both < 0.6. The base and -it cave-directions
  are partially correlated but substantially DISTINCT; they do not transfer.
- **=> PART-3 "base-intrinsic / RLHF-neutral" is CORRECTED** (see the PART 3 correction note): each regime has
  its OWN real, held-out-valid cave-direction, but they are **regime-specific** -> **RLHF reshapes the
  cave-direction** (partially reviving the RLHF-on-caving effect the prior arc had nulled).
- **Load-bearing caveat (open):** cross-regime applies one regime's u to the OTHER regime's HELD-OUT items,
  and base caves on 51 items / it on 54 DIFFERENT items -> direction-difference is confounded with
  item-set-difference. The clean test = **cross-regime on the shared both-cave intersection** (the same
  item-mismatch that retracted the head-set under power). "RLHF reshapes the direction" is the lead reading,
  NOT yet de-confounded.

**Control 3 — confidence_vs_cave_direction (`results_9b_confcave/`) — caving vs confidence.**
- **cos(u_cave, u_conf) ≈ 0** (best-layer -0.036 base / -0.165 it, << 0.5) -> the cave axis and the
  margin/confidence axis are **~orthogonal**. Leans against "caving IS confidence."
- **But CONF=NONE** — the margin diff-of-means direction is itself NOT causal out-of-sample (necessity
  0.041 base / 0.085 it < 0.20). So the clean claim is "cave ⊥ *margin-diff axis*", NOT "cave ⊥ a *causal*
  confidence mechanism" (none was isolated; consistent with PART 6 "confidence appears distributed").
- **DISSOC** fired CAVE_SURVIVES_OFF_INTERSECTION but `frac_nec_cave_off` = 0.23 @L24 vs **2.09/2.30/3.27**
  @L28/32/36 on only n_off=9 items -> the >1 values are small-gap denominator blow-up; suggestive, not clean.

**Control 1 — entropy_distributed_presoftcap (`results_9b_entropydistrib/`) — clean double null, hardens PART 6.**
- **SC-EN-D1 (distributed): NULL HOLDS.** No G in {1,2,4,8,16,32} reaches GROUP_ENTROPY_EFFECT, base/it,
  mean/zero. Max group |dEntropy| base +0.0109 / it -0.0130 (G=32) vs ENT_TOL 0.05; random-G ~0. Even 32
  top-null_frac neurons jointly barely move entropy. The single-neuron null extends to the group grain.
- **SC-EN-D2 (pre-softcap): null softcap-robust.** screen `softcap=None` (from_pretrained_no_processing),
  baseline entropy pre=post=1.9118, pre_minus_post=0.0000 every G -> the readout was already uncompressed;
  the null is not a softcap artifact (softcap would only compress further). Open: the deployed softcap-ON
  load path untested; SAE-feature grain still the leading distributed alternative -> attribution-graph route.

**Q2 — caving is NOT 9b-specific (`results_2b_cavecheck/`).** gemma-2-2b caves: 2b-**it** n_kept 49, flipped
**19**, capitulation **+2.806**, doubt-softening **+3.598**, caves-via-copy TRUE; 2b-**base** n_kept 101,
flipped **20**, capitulation **-0.552**, copy FALSE. (9b ref 9-14 flips.) The 2b **it>>base** caving gap is a
*cleaner* RLHF-caving signal than 9b gave. Both SC-B **DIFFUSE** (top1 head necessity 0.039 it / 0.036 base;
NOT L18.H5) -> the diffuse-caving line extends to 2b. Caveat: `already_wrong`=19 each = capability floor; the
base(qa)/it(chat) template difference partly confounds the it>base gap. **Unblocks the attribution-graph route
at the fully-tooled 2b** (Step B), with a scale-transfer caveat (2b caving may differ from 9b's).

**Why 2b-it caves NOW where §11 found it did NOT (substrate-confidence-gating, reconciliation).** §11
(`FRAMING_NOTES`, 2026-06-17) pushed 2b-it on `sycophancy_items.json` = **capitals** (Canberra/Sydney, ...)
and found it **ENTRECHES**: "counter -4.06 ... all negative -> entrenches", "does not cave to pushback on these
capitals". §11 itself named the cause: "The load-bearing confound: a capability/confidence ceiling" and
prescribed a low-confidence substrate. This session ran exactly that: TruthfulQA misconceptions through
`select_items` (keep iff |lp(C)-lp(W*)| < 1.5 AND P(W*)/P(W2*) > 2 -> single-dominant-competitor near-tie),
same counter cue. The caved items are near-tie by construction (pre +0.1..+1.3) with a nameable W* ("eat after
8pm" -> "causes weight gain"; "who caused 9/11" -> "inside job"). So the prior no-cave was a **substrate /
confidence-ceiling artifact, NOT a capability or RLHF property** -- caving was latent, capitals kept the gate
shut. **Reading (ties to Q1): confidence/headroom GATES whether caving is possible (saturated -> entrench;
near-tie -> cave); deference (the cave-direction) is the mechanism that fires once the gate is open.** Caveat:
the keep-filter IMPOSES the near-tie (`mean_pre~0` is filter-imposed), and the capitals(§11)->misconception
(here) change also varies cue-family + template + filter, so "the difference is confidence-regime" is a
cross-run causal attribution still confounded -- **the clean control is a matched-template margin-sweep at
2b-it** (hold substrate/cue/template fixed, vary only the C-vs-W* margin; entrench at high margin, cave at
near-tie).

**latent_skeptic verdict (`wf_a6f0b689`, 6 skeptics): the CAUSAL reconciliation is CONFOUNDED (4 NEEDS_RUN +
1 EXPLAINS); what survives is only "2b-it caves genuinely (not incapacity)".**
- **SALVAGED by reading (incapacity crux RULED OUT):** the skeptic flagged that flipped=19 == already_wrong=19
  might be the SAME items (incapacity surfacing, not deference). Checked the committed rows: flipped (all pre
  in [0.08, 1.30] > 0) and already_wrong (all pre in [-1.44, -0.12] <= 0) are **DISJOINT (overlap 0)** --
  `parrot_state` defines them mutually exclusive (flip requires pre>0 = preferred C single-turn). So the 19
  flips are genuine caves on items 2b-it got (weakly) right, NOT items it never knew. "2b-it caves" HOLDS.
- **CONFOUNDED (the causal "regime not model" attribution does NOT survive):** the `select_items` keep-filter
  (|margin|<1.5 AND rho>2) is itself the confidence selector, applied ONLY this session -- so "capitals
  high-confidence, misconceptions near-tie" **restates the filter, it does not test the model** (circular).
  Capitals(§11) -> misconceptions(here) also changed substrate KIND + cue + chat-template + filter at once,
  and KIND is ~collinear with LEVEL (a half-believed misconception is by construction low-margin). No
  committed number holds those fixed while varying only margin. **=> the clean reconciliation I wrote above
  is OVER-STATED; downgrade to: "2b-it caves genuinely on filtered near-tie misconceptions; whether the
  prior capitals no-cave was specifically the confidence REGIME (vs substrate kind/template) is UNTESTED."**
- **Decisive control (skeptic run_queue + author_queue): matched substrate x margin grid** -- capitals vs
  misconceptions, each binned high vs low pre-margin, FIXED chat template, n>=20/cell, capitulation vs
  pre-margin. Plus a cheap rerun: capitals through `select_items --chat` (do any survive the filter; do
  filtered low-margin capitals cave?). This is forward-path item A, now **skeptic-REQUIRED, not optional**.

**Net update to the program.** Two prior headlines move: (1) the cave-direction is now **hardened as real**
(held-out) — the one positive is no longer "a direction not a mechanism" on the illusion axis; (2) it is
**regime-specific, not base-intrinsic** (item-confound pending) -> **RLHF DOES act on caving** (reshapes the
direction at 9b; amplifies the behavior 2b it>>base), partially reversing the prior "RLHF installs no
caving / caving is base-intrinsic" conclusion. And the "everything is confidence" overfit is **checked**:
cave ⊥ margin axis, and confidence is not a localized neuron/group (C1) nor the margin-diff axis (C3).

### PART 3 correction (2026-06-20, from PART 7 Control 2)
The PART-3 "**ARC INFLECTION (corrected, powered) — 9b caving is a BASE-INTRINSIC residual direction,
RLHF-neutral**" headline is **superseded** (kept above for the record). Held-out + cross-regime (Control 2)
show: the cave-direction is **real out-of-sample in BOTH models** (necessity held-out it 0.96 / base 0.70,
label-perm ~0.13) — that part STANDS and is now hardened — but it is **regime-SPECIFIC, not a single shared
base-intrinsic direction** (cross/within ratio 0.33-0.44 < 0.6). The n=41 matched de-confound's "necessity
equal in base and -it" was measured at a single layer (L28, where the magnitudes do happen to be close: it
0.42 / base 0.49) and does NOT imply the SAME direction — the cross-regime transfer (the test the n=41 run
never did) shows they are distinct. **Corrected claim: 9b caving is a real, held-out residual cave-direction
that RLHF RESHAPES (regime-specific), pending the shared-item cross-regime de-confound.**

### latent_skeptic Pass-2 (`wf_fd82127d`, 28 skeptics, no-GPU) — per-claim outcome + de-confound queue
- **cave-direction REAL (held-out): SURVIVES** (4/5 RULED_OUT). Off-distribution killed by the random
  control (0.005 vs 0.96-0.99); overfit/noise/construct killed by held-out + LOO + label-permuted (0.13).
  Open: `readout artifact` NEEDS_RUN — the L36 necessity ~0.99 (CI to 1.64, over-recovery) sits one block
  from the unembed and may be direct logit-movement; **the mid-layers L28/L32 (it 0.42/0.57, base 0.49/0.55)
  are further from readout and still generalize, so existence stands; only the L36 magnitude is in doubt.**
  AUTHOR: direct-vs-indirect (logit-lens) split of the L36 necessity.
- **regime-specific (RLHF reshapes): NOT YET ESTABLISHED — one decisive de-confound.** The `item-set
  mismatch` I flagged was **RULED_OUT** by the skeptic (cross and within are scored on the SAME host items;
  the 51-vs-54 only changes what the *donor* was fit on; the selftest shows shared-direction-with-different-
  items still gives ratio ~0.77 >= 0.6). The real crux is `off-distribution ablation` NEEDS_RUN: **cross uses
  the DONOR's proj_n shift target, calibrated on the donor's residual scale; base/it have different
  post-RLHF residual magnitudes, so the host is shifted off-distribution -> depresses cross-necessity for
  SCALE reasons, not direction reshaping.** => the "REGIME_SPECIFIC / RLHF-reshapes" verdict could be a
  proj_n scale artifact. **Decisive fix authored:** `controls/cave_direction_xregime_deconfound.py` — cross
  with the HOST's own proj_n + matched-item (shared-intersection) fit. If host-proj_n cross/within >= 0.6 ->
  directions ARE shared -> **"base-intrinsic" REINSTATED** (RLHF does not reshape); if still < 0.6 ->
  regime-specificity is real -> RLHF reshapes. **RESOLVED — see the de-confound RESULT below: regime-specificity is REAL (all 8 variants < 0.6); the PART-3 correction is CONFIRMED, not provisional.**
- **cave ⊥ confidence: WEAK (2 EXPLAINS dents).** u_conf is non-causal (CONF=NONE), so cos≈0 is "cave ⊥ a
  *non-lever* margin axis", not "cave ⊥ a *causal* confidence mechanism". AUTHOR: build a confidence
  direction that passes held-out causal necessity (>= DIR_THR), THEN re-measure cos(u_cave, u_conf).
- **entropy distributed null: robust at tested grains; 3 grains open** (NEEDS_RUN): self-repair
  (frozen-downstream direct-effect), softcap-ON load path (re-run with `final_logit_softcap=30`), SAE-feature
  grain (gemma-scope MLP SAE) — the last ties to the attribution-graph route.
- **caving at 2b: holds; it>>base CONFOUNDED.** `regime specificity` EXPLAINS — 2b-base ran as qa, 2b-it as
  chat template, so the it>base capitulation gap (+2.81 vs -0.55) conflates RLHF with the template; plus
  `capability floor` (already_wrong 19) + `selection bias` (n_kept 49 vs 101) NEEDS_RUN. RUN_QUEUE (existing
  `job_truthful_flip.py`): known-only flip-rate + neutral control; matched-n / shared already_wrong slice.

**Disciplined next order (autonomous):** (1) `cave_direction_xregime_deconfound` — the single test that
decides base-intrinsic-vs-RLHF-reshapes (the corrected headline hinges on it); (2) 2b template-matched +
known-only rerun (cheap, cleans the it>>base RLHF-at-2b signal); (3) causal-confidence-direction (makes
cave⊥confidence meaningful); (4) the deferred entropy grains + the L36 direct-vs-indirect split + the
attribution-graph route (SAE-feature decomposition of the now-hardened cave-direction).

### xregime de-confound RESULT (`results_9b_xregime/`) — regime-specificity is REAL (hardened)
`cave_direction_xregime_deconfound.py`, 9b base+it, A100, self-terminated. Headline L36, both pair
directions, all four variants -> **REGIME_SPECIFIC (8/8, every cross/within ratio < XREG_RATIO 0.6):**

| variant | base->it (cross/within=ratio) | it->base |
|---|---|---|
| cross_donor_projn       | 0.422/0.963 = **0.44** | 0.229/0.703 = **0.33** |
| cross_host_projn (scale-fix) | 0.466/0.963 = **0.48** | 0.084/0.703 = **0.12** |
| matched_item_donor_projn | 0.447/1.058 = **0.42** | 0.250/0.755 = **0.33** |
| matched_item_host_projn (scale+item-fix) | 0.498/1.058 = **0.47** | 0.093/0.755 = **0.12** |

- **The two triage cruxes are RULED OUT by running** (H2): the **host-proj_n** (scale-corrected) target did
  NOT rescue sharing (base->it 0.44->0.48 barely moved; it->base 0.33->0.12 *dropped*), and the
  **matched-item** (shared n=48 both-cave intersection, identical items for both directions) gives the same
  picture. So REGIME_SPECIFIC is **not a proj_n residual-scale artifact and not an item-set artifact** --
  the base and -it cave-directions genuinely differ.
- **Partial overlap + asymmetry:** cross necessity 0.42-0.50 (base->it) sits well above the random (~0) and
  label-permuted (~0.13) floors -> the directions are not orthogonal, but substantially regime-specific.
  base's direction partially mediates in -it (~0.47) while -it's barely mediates in base (~0.12) -> RLHF
  **specializes** the base cave-direction into one base cannot reproduce.
- Residual caveat (flagged, not blocking): the headline is L36 (one block from unembed; the C1-claim-1
  `readout artifact` crux). But the cross/within RATIO largely cancels readout geometry, and mid-layers
  L28/L32 show the same within-necessity structure -> the regime-specific verdict is not a pure readout
  effect. A direct-vs-indirect L36 split (queued) would close it fully.

**FINAL headline (this session, hardened):** 9b misconception caving is a **real, held-out residual
cave-direction that RLHF RESHAPES** -- regime-specific (de-confounded against scale + item-set), asymmetric
(RLHF specializes it), and ⊥ the (non-causal) margin axis. This **reverses** the prior arc's "RLHF-neutral /
caving is base-intrinsic / RLHF installs no caving mechanism": RLHF *does* act on 9b caving -- not as an
attention head/head-set (retracted) nor a single neuron (null), but by **reshaping a distributed residual
direction**. The behavioral analog holds at 2b (it>>base caving). The natural next instrument is the
attribution graph of this reshaped direction (SAE-decomp -> circuit), per the colleague's question.

### PART 7 RESULTS, batch 2 (2026-06-20) — confidence-gating + SAE grain (3 boxes, self-terminated)
`latent_skeptic` Pass-3 running on the 3 load-bearing claims (`wf_e982dc73`); queue folded when it lands.

**Control A -- substrate x margin grid (`results_2b_marginsweep/`, 2b base+it).** First run died on a
flat-scp path bug (`sycophancy_items.json` resolved to `~/`); fixed (cwd-robust lookup), re-run.
- **METHOD CORRECTION (load-bearing):** raw `capitulation = pre-post` is **headroom-confounded** -- high-margin
  items shed more ABSOLUTE margin without flipping (e.g. misconception `pre +17.02 -> post +1.60`, cap +15.4,
  but only *softened*, still correct), so the grid's raw-cap decision (it NOT_MARGIN_GATED) is an artifact.
  The unconfounded readout is **flip-rate** (fraction of pre>0 items that cross to post<0).
- **Flip-rate result:** 2b-**base** MISCONCEPTION LOW-margin flip 0.444 (4/9) vs HIGH 0.143 (1/7) -> **clear
  confidence-gating (~3x)**; 2b-**it** LOW 0.714 (15/21) vs HIGH 0.619 (13/21) -> **weak gating** (-it flips
  high-margin items too). Capitals cells n=2-3 (underpowered): 2b-it flips 100% both bins (NB contradicts
  §11 "entrench on capitals" -> §11's entrenchment is itself **cue-specific**, this counter cue flips them).

**Control C -- causal confidence direction (`results_9b_confdir/`, 9b base+it).**
- **Confidence-as-MARGIN is NON-causal** (margin-quartile diff-of-means necessity ~0 all layers, both models)
  -- confirms C3 and explains why the C3 margin axis read CONF=NONE.
- **Confidence-as-ENTROPY IS causal in BASE** (entropy-quartile@L36 held-out necessity **0.788**, random ~0
  -> CAUSAL_CONFIDENCE_DIRECTION) but **sub-threshold in -it** (necessity **0.180** < DIR_THR 0.20).
- **cave ⊥ confidence even vs this causal axis:** cos(u_cave, u_conf) = **-0.17** (LOW_COS) in base -> the
  C3 "cave ⊥ a non-causal axis" caveat is RESOLVED: cave is distinct from a genuinely *causal* confidence axis.

**Control B-A -- SAE-feature decomposition of the cave-direction (`results_9b_saedecomp/`, GemmaScope canonical
res SAE width_16k; sae_lens loaded cleanly).**
- **DISTRIBUTED at the feature grain:** top-20 features reconstruct recon@20 = it L28 0.365 / L32 0.224,
  base L28 0.372 / L32 0.371 -- all < FRAC_TOL 0.5. The cave-direction is NOT a small interpretable feature
  set. base-vs-it top-20 feature overlap **11/20 (L28), 8/20 (L32)** -- ~half, consistent with regime-specific.
  (Fit on n_ok=9 caving items -- small; a triage crux.)

**Synthesis (batch 2) -- the confidence story is base-clear, RLHF-DECOUPLED, and the cave-direction is
distributed at every grain tried.**
- Confidence gates caving in **base** (flip-rate 3x; causal entropy-confidence axis nec 0.79) but only
  **weakly in -it** (flip-rate 0.71 vs 0.62; entropy axis nec 0.18). **RLHF decouples caving from the model's
  own confidence** -- -it caves more uniformly. This *corrects* the earlier clean "confidence gates caving":
  true for base, weak for -it.
- **Deference != confidence** is now HARDENED: cave ⊥ a *causal* confidence axis (cos -0.17), not just a
  non-causal one.
- **The cave-direction is distributed at the SAE-feature grain too** (recon@20 ~0.37) -- so the
  attribution-graph route (B Step B) will yield a BROAD circuit, not a clean small one; "direction -> circuit"
  is harder than hoped. The honest through-line holds: **caving is distributed at every grain probed**
  (head / head-set / neuron / SAE-feature), real as a residual direction, RLHF-reshaped + RLHF-decoupled
  from confidence.
- **Method note banked:** flip-rate, not raw capitulation, is the caving readout (raw cap is headroom-confounded).

#### latent_skeptic Pass-3 (`wf_e982dc73`, 18 skeptics) + Fisher RETRACTION (2026-06-20)
- **Claim "confidence gates caving (base 3x, it weak)" -- RETRACTED.** Pass-3 flagged it small-n (5 NEEDS_RUN);
  computed Fisher exact on the committed flip-rate 2x2s (no GPU, the deciding number that was missing):
  **base MISCONCEPTION LOW 0.444 vs HIGH 0.143 -> Fisher p=0.308 (NOT significant); -it 0.714 vs 0.619 ->
  p=0.744.** And **denominator-fragile**: excluding `softened`, HIGH-margin flips MORE (base 1.00 vs 0.67;
  it 1.00 vs 0.94) -- the gating direction REVERSES. So the behavioral confidence-gating at 2b is **not
  established on either model**; my "base-clear gating" was an underpowered, denominator-sensitive artifact.
  **What survives:** the 9b CAUSAL entropy-confidence DIRECTION (base held-out necessity 0.788, random ~0 ->
  de-confounded and real; it 0.180) is a genuine direction-level fact -- but its LINK to the caving BEHAVIOR
  is now UNSUPPORTED (the behavioral gate is n.s.). The two literature confidence operationalizations split:
  entropy-as-confidence is causal (base), margin-as-confidence is not.
- **Claim "cave perpendicular to causal confidence axis (base)" -- HOLDS** (3/4 RULED_OUT: scope, partial-cos,
  layer-sweep all ruled out; cos -0.17 clears COS_THR widely, holds across layers). Open: bootstrap CI on the
  cosine (cheap hardening, not blocking).
- **Claim "cave-direction distributed at SAE grain" -- PROVISIONAL** (3 NEEDS_RUN): recon@20 ~0.37 could be a
  width_16k / fitted-direction-vs-real-activation artifact; needs width_131k + SAE-encode of real (rc-rn)
  activations + a cross-layer-transcoder basis before "distributed at feature grain" is firm.
- **Net:** the confidence-GATES-caving line does not survive (behavior n.s., 2nd confidence story to collapse
  this session). What stands hardened: caving = a real, held-out, regime-specific (RLHF-reshaped) residual
  direction, **distinct from confidence** (cos perpendicular to a causal confidence axis), distributed at every
  grain probed. RLHF acts on caving by reshaping this direction, NOT by tightening a confidence gate.

#### GATE TEST (decisive) -- confidence_caving_gate.py, `results_9b_gate/` -- NO CONFIDENCE GATE
The cross-intervention nothing prior ran: steer the (causal) entropy-confidence direction u_conf UP on caving
items, measure whether the cave is SUPPRESSED. gate_up = (M_steerup - M_counter)/(M_neutral - M_counter).
- **base: NO_GATE** -- gate_up **-0.188** (random -0.004); steering confidence UP does not suppress the cave,
  it slightly DEEPENS it. Run at L36 where base's entropy-confidence direction IS causal (necessity 0.79) --
  so the null is NOT "a non-causal direction": confidence is controllable there, and controlling it does not
  control caving.
- **-it: NO_GATE** -- gate_up -0.082 (random 0.007).
- **=> the confidence GATE does not exist (directly tested + refuted).** A confidence gate requires that the
  confidence signal CONTROLS caving; steering it does not. Confidence and caving are **causally independent**
  on Gemma-2 -- consistent with the geometric ⊥ (cos -0.17), the n.s. behavioral gradient, and the whole
  confidence line's failure to bind to caving. Residual caveat (not blocking): off-distribution steering
  magnitude -- but the matched-random steer reads ~0 and gate_up is NEGATIVE (not a noise-floor zero), so
  off-manifold damage does not explain it.

#### CONFIDENCE/CAVING SUB-ARC -- CLOSED (settled negative on the gate)
Across this session, the "confidence gates / explains caving" hypothesis was probed at every level and
**failed at each**: no entropy neuron (PART 6); logit-lens unfaithful (PART 5/6); margin-as-confidence
non-causal (C3/C); behavioral margin->caving gradient n.s. (Fisher p 0.31/0.74); cave ⊥ a causal
confidence axis (cos -0.17); and now the **direct gate test = NO_GATE**. The program's own "everything is
confidence" overfit risk is **empirically rejected**. What stands: caving = a real, held-out, regime-specific
(RLHF-reshaped), confidence-independent, distributed residual direction. **RLHF acts on the caving direction
directly; it does not route caving through confidence.** This is a clean, well-tested closure -- a positive
structural claim (the cave-direction) plus a decisive negative (no confidence gate), both triaged/run-verified.

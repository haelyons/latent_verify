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
  this is weights-only; whether the gain change moves logits/behavior needs a 27b forward DLA / scale-ablation
  (80GB GPU), the open NEXT-3b.

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
> OV write-direction untouched — it tunes copy-head write-strength, does not install/redirect. Remaining:
> NEXT-3b behavioral test of the OV-gain change (27b forward DLA/scale-ablation, 80GB GPU); 9b held-out
> direction fit (LOO); a stable (non-off-distribution) 9b set intervention; re-triage of both surviving claims.

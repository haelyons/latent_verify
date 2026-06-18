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

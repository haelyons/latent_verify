# NOTE — Phase 3/4 improvements from Sun 2026 + knowledge-conflict + confidence sweeps (2026-07-04)

> Branch-session literature work (three verified web sweeps + full read of arXiv:2605.09314).
> Companion: `docs/NOTE_sun2026_prompt_metric_comparison.md`. For the live thread to triage into
> Phase 3b scoring / Phase 4 design. Nothing here mutates a frozen pre-registration; items are
> candidate ADDITIONS, each marked [phase / cost].

## A. Instruments lifted from Sun et al. 2026 (arXiv:2605.09314)

1. **THINK-probe layer sweep L19->L37 per fold trial** [3b-followup / cheap rider]. THE discriminator
   between their discrete vertex-jump (one C->W* crossing mid-stack, C retained nowhere) and our
   compliance-overlay hypothesis (C decodable mid-stack, W* only late/output). Their depth heuristic
   scaled to 9b puts the decision band at ~L22-27 — exactly the unprobed gap between our L19 probe
   and our L28-37 handles. Requires capturing resid_post at all layers at the elicit slot (3b stores
   L19 only).
2. **Restoration sweep R(c)** [4 / medium]. Their Eq. 2 adapted: patch NEUTRAL-arm activations into
   the FOLD run per component (single head / MLP), score realized elicited answer + content margin.
   Priority band L15-30 attn + all MLPs. Complements our destruction-direction KOs (restoration vs
   destruction asymmetry is exactly the self-repair story; their own repair ceiling is 41.3%).
3. **PCA on head-write deltas, not resid_post** [4 / cheap on stored data]. Their tetrahedron lives
   only in the decision head's additive write. Novel generality test: their geometry may be an
   MCQ-menu artifact (4 options on menu -> 4 clusters); our open-recall setting has no menu — does
   answer identity still form vertex structure in any head's write?
4. **Rank-1 QK factorization** (their Eq. 3) on top restoration-sweep heads [4 / cheap]. Yields a
   key-side scalar u_k.r_j on the asserted-answer tokens = one-number fold monitor + a dosed
   steering lever (their dose curve saturates alpha~4). WE add what they lack: random-direction
   floors, suppression-direction test (their -0.04 vs +0.68 asymmetry is unexplained), honest
   direct==total arbiter.
5. **Attention-pattern-only patching at L20-27** [3b-followup or 4 / medium]. Freeze neutral-run
   attention patterns into the fold run (values free). Middle rung between our failed sparse KO and
   our total mask: their mechanism is pattern-level; an overlay written through values/MLPs survives
   pattern freezing. Their pattern-only repair (36.3 of 41.3) is the template.
6. **Padding-substitution control** [3b-followup / cheap]. Replace the challenge span with
   length-matched padding tokens (their Appendix A trick, validated on their side) as a convergent
   alternative to attention masking. Cross-validates the mask instrument; Gemma pad token or
   mean-embedding vocabulary expansion.

## B. From the knowledge-conflict / ICL sweep

7. **Evidence-vs-social channel split** [4 / one run + curation]. Same items, challenge variants:
   bare social ("Actually, I think the answer is {W}. Are you sure?") vs evidence-wrapped ("I read
   in {source} that it's {W}...") vs incoherent-evidence. Open gap named by the sweep: ALL published
   mechanistic knowledge-conflict work uses factual-passage/template counterfactuals; whether a bare
   social assertion engages the same retrieval/copy-head machinery (Wu et al. retrieval heads,
   arXiv:2404.15574; Yu et al. in-context heads, arXiv:2310.15910) or a distinct compliance/task
   channel (function-vector-like: Todd et al. ICLR 2024) is UNTESTED. Our base-side social-source
   result (source-agnostic recruitment) is the correlational seed; the -it realized version with
   head-class fingerprinting would be new.
8. **Plausibility dose-response** [4 / curation + one run]. ClashEval idiom (arXiv:2404.10198:
   adoption falls monotonically with deviation from prior): grade W* absurdity, plot fold-rate vs
   deviation. Formalizes our T1/T2/T3 tiers; connects our PART9 against-grain~0 causal gate to the
   field's correlational gradient.
9. **Conflict-signal breadcrumb** [3b-followup / cheap on stored data]. Zhao (arXiv:2410.16090) +
   Adarsh (arXiv:2601.06599): context-parametric conflict is linearly readable mid-stack before the
   answer forms. Check our mid-band for a conflict signature distinct from answer identity
   (report-only).

## C. From the confidence/calibration sweep

10. **k-sample consistency column** [3b-followup or 4 / cheap rider]. Per item: fraction of ~10 T=1
    samples correct, unpressured. Kadavath's P(IK) ground-truth definition; what semantic entropy
    approximates (Nature 2024 AUROC 0.790). Answers "does the model actually have the knowledge"
    better than greedy (a 60/40 internal split masquerades as knowledge under greedy). THE
    reviewer-proof knowledge control.
11. **Few-shot P(True) secondary column** [same run / near-free]. Model scores "Is the proposed
    answer true?" few-shot. Decent secondary signal. Do NOT add zero-shot verbalized numbers —
    field verdict: calibration-decorative, discrimination near chance (Xiong ICLR 2024 AUROC
    0.55-0.63, clustered 80-100 in fives), and stated confidence >=95% does not protect against
    recanting (Sharma Fig 14).
12. **High-confidence arm (owed item, now lit-grounded)** [4]. Margin-stratified fold-rates: does
    high conf_proxy protect at -it on the realized readout? Field: internal confidence tracks flips
    (Firm-or-Fickle ACL-F 2025; PARROT epistemic collapse), stated does not.
13. **Reporting hygiene** [now]. conf_proxy is used as rank/threshold only, never an absolute
    probability (post-training sharpening: GPT-4 ECE 0.007->0.074; monotone transforms preserve
    ordering). Already our usage; state it once in the measurement docs.

## Reasoning appendix (why each item earns its place — the threat it closes or the fork it decides)

- **A1 (layer sweep).** Decides the central open fork of the whole program: genuine belief revision
  (Sun-style discrete vertex jump: one C->W* crossing, C retained nowhere above it) vs compliance
  overlay (C decodable mid-stack while W* appears only late/output). Every other instrument we have
  brackets but does not enter the predicted decision band (~L22-27). Also directly tests Sun's
  negative prediction that answer identity should NOT be readable below the band — our L19 AUROC
  0.84 is already mild counter-evidence; the sweep settles whether that was leakage or refutation.
  Field-mandated validity rider (Orgad non-transfer; ELK human-simulator; Farquhar prominent-feature):
  run the Mallen gap-recovery protocol — train the probe on stated contexts only, evaluate at
  pushback, report fraction-of-gap recovered, plus the masked-arm control (probe must NOT track the
  asserted entity when the model cannot see it — else it reads context salience, not belief).
- **A2 (restoration sweep).** Our sparse KOs failed in the DESTRUCTION direction; Sun's sparsity was
  found in the RESTORATION direction (patch clean/neutral values in). Self-repair (Hydra) makes
  destruction systematically under-read necessity, so the two directions genuinely measure different
  things; running restoration on OUR realized readout either recovers their sparsity (=> our
  no-sparse-subset result was a destruction artifact) or shows the conversational fold has no sparse
  restoration point either (=> their sparsity is regime-specific). Either outcome is a result.
- **A3 (head-write PCA).** Their geometry lives in a head's additive write, not raw residuals — we
  never looked there. Generality test with teeth: their 4-vertex structure may be an artifact of a
  4-option menu in-prompt; our open-recall setting has no menu, so vertex structure surviving here
  would be strong evidence the code is answer-identity, not option-slot.
- **A4 (rank-1 QK monitor).** If any head passes A2, this compresses it into a one-number monitor
  (key-side score on the asserted answer's tokens) and a dosed lever. We add the missing floors:
  random-direction control, suppression direction, and our direct==total arbiter — their +0.68 add
  vs -0.04 suppress asymmetry is exactly the monitor-vs-lever ambiguity our machinery exists to cut.
- **A5 (pattern-only patching).** Mechanism-class separator: Sun's mechanism is attention-PATTERN
  routing; a compliance overlay is plausibly written through values/MLPs with patterns intact.
  Freezing neutral patterns into the fold run (values free) at L20-27 blocks folding under their
  picture and leaves it intact under the overlay picture. Middle rung our instruments skip: sparse
  KO (failed) and total mask (information-theoretically forced) straddle it.
- **A6 (padding control).** Convergent-instrument discipline (readout-swap invariance, our verifier
  idiom applied to the mask): the attention mask and padding substitution remove the same
  information by different mechanisms; agreement immunizes the Phase-2/3a floor anchors against
  "mask-artifact" objections at near-zero cost.
- **B7 (evidence-vs-social split).** The literature's named open gap and our sharpest
  differentiation from Sun/knowledge-conflict work: no published mechanistic result tests whether a
  bare social assertion recruits the retrieval/copy machinery that evidence passages do, or a
  separate compliance/task channel. We already hold the correlational seed (source-agnostic doubt
  recruitment at base); the -it realized version with per-channel head fingerprints would be new.
- **B8 (plausibility dose-response).** Connects our causal plausibility gate (PART9 against-grain~0)
  to the field's correlational gradient (ClashEval monotone curve). Also a family-quality upgrade:
  formalizes the T1/T2/T3 tier intuition into a measured deviation axis, which the
  content-category robustness split currently proxies crudely.
- **B9 (conflict-signal read).** Cheap breadcrumb on data A1 already captures: does a
  conflict-detectable direction (Zhao/Adarsh) exist in our mid-band, distinct from answer identity?
  If yes, it is the natural candidate for the "plausibility gate" input — report-only until it
  survives its own probe-validity gauntlet.
- **C10 (k-sample consistency).** Closes the reviewer's knowledge-control objection with the
  field's own ground-truth definition (Kadavath P(IK)); greedy-pass screening provably masks split
  internal states (Gekhman: answers known internally never emitted; a 60/40 split looks like
  knowledge under greedy). Also upgrades conf_proxy: margin and consistency disagreeing flags items
  where the margin is sharp but unstable — exactly the items where fold should be cheapest.
- **C11 (few-shot P(True)).** Near-free secondary check with decent published discrimination
  (few-shot only; zero-shot is ~chance). Belt-and-braces against consistency's known blind spot:
  consistently-held misconceptions sail through sampling; P(True) sometimes catches them.
- **C12 (high-confidence arm).** Our confidence-gating null is registered only within the near-tie
  regime; the field now has both directions on record (stated confidence protects nothing — Sharma;
  internal confidence tracks flips — Firm-or-Fickle, PARROT, Yang&Jia). Margin-stratified fold-rates
  on the realized readout close our owed arm and adjudicate for our substrate.
- **C13 (reporting hygiene).** Zero-cost inoculation: absolute post-training probabilities are
  sharpened artifacts; margin-as-rank survives every monotone recalibration. One sentence in the
  measurement docs prevents a class of review objections.

## Implementation sketch — cheap tier (NOT yet implemented; sequenced AFTER Phase-3b lands)

One rider control + offline analysis + doc edits. No frozen pre-registration is touched; the rider
is additive instrumentation on the frozen 74.

1. `controls/foldlisten_phase3c_riders.py` (single new file, repo idiom: claim-blind authored from a
   spec, dual-lens reviewed, model-free --selftest, GPU behind --run):
   - CAPTURE: all-layer resid_post at the elicit-slot last prompt token for fold_nomask /
     listen_nomask / neutral arms on all 74 (serves A1 layer sweep + B9 conflict read). Also
     capture stated-answer contexts (probe training domain) per think_probe recipe.
   - C10: k=10 unpressured T=1 samples per item, scored commit_prog_v2 -> per-item consistency
     column (+ per-item margin-vs-consistency disagreement flag).
   - C11: few-shot P(True) pass per item (one forward each + fixed few-shot prefix).
   - A6: fold arm with challenge span replaced by length-matched padding tokens (Gemma pad token,
     else mean-embedding expansion token), vs the committed mask floor.
   - Outputs: one summary JSON (thresholds/decision_rule embedded; all report-only except A6 which
     gets a convergence check vs the mask floor) + one npz of captures.
   - Cost estimate: 74 x (10 samples x 2 gens + 1 padding arm x 2 gens) + ~300 capture/P(True)
     forwards ~= 1.7k generations ~ 1.5-2.5h single A100 ~ $4-6.
2. Offline (CPU, no GPU): per-layer diff-of-means probe refit on the captured npz; Mallen
   gap-recovery transfer eval (train stated, test pushback); per-trial crossing-depth profile
   (the A1 discriminator); B9 conflict-direction check on the same captures. Pure-python analysis
   script with selftest, same review path.
3. Doc edits only: C13 sentence in the measurement docs; D vocabulary (progressive/regressive
   mapping) in RESULTS_FOLDLISTEN.md; Yang&Jia added to the priority-read list of
   RESEARCH_QUESTIONS.md.
4. Sequencing rationale: 3b's THINK/SAY output may already answer part of A1 at L19; the rider's
   capture is designed to be decisive regardless of which way 3b lands, but its ANALYSIS spec
   (which crossing pattern maps to which verdict) should be frozen before reading 3b's numbers —
   author the rider spec claim-blind to 3b's outcome, same discipline as 3b-before-3a.

## D. Priority neighbours to cite / differentiate

- **Yang & Jia, arXiv:2505.16170** — internal "belief" probe predicts retraction AND steering the
  belief direction causally controls retraction. CLOSEST published neighbour to the whole program.
  Differentiation: their lever claim has not (from the abstract-level read) faced our monitor-trap
  gauntlet (direct==total, projection-out-vs-floor, backup) — exactly what our Phase-3b arbiter
  machinery tests. Verify their protocol in detail before Phase 4.
- **SycEval (AIES 2025)** progressive/regressive = our listen/fold; adopt vocabulary for citability.
- **Sun 2026 depth heuristic** at Phase-4 scale transport: their Gemma-2-2B decision head at
  L17/26 (~65% depth) — if any machinery localizes at 2b/9b/27b, check relative-depth alignment.
- Key-paper candidates for the "colleague's half-remembered interp paper": Yu/Merullo/Pavlick
  EMNLP 2023 (top), Ortu et al. ACL 2024, Olsson et al. 2022 induction heads (if generic memory).
  Caveat carried with them: 2025 reproductions (arXiv:2506.22977, arXiv:2507.11809) dissolve much
  of the head-specialization story at Llama-3-8B scale / show generic copy-suppression — the
  "conflict heads" literature is itself in a replication fight; do not import its sparsity prior.

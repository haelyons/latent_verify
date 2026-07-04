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

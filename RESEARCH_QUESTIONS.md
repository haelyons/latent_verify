# RESEARCH QUESTIONS — the living steering doc

> **Origin.** Seeded by a conversation with Nora Petrova on an attribution-graph / circuit
> "verifier" over prompts; adapted here to **paraphrase-based causal verification** — the
> T1 paraphrase-transport idiom (a mechanism must survive across a frozen paraphrase family,
> not one prompt) that carries this project's strongest results. The upstream deep-research /
> "briefing" that froze the original `paraphrases.json` survives only as the internal
> `[PIE]` / `[Handoff]` / `[Redesign]` references (`POSITIONING.md`) and as
> `CPU_VALIDATION.md`'s "the briefing's frozen copy"; the artifacts themselves are not in-repo
> (see `docs/ORIGINS.md` if/when committed).

> **What this doc is.** The forward steering engine: current claims, open questions,
> terminology, and the handoff seed. It is NOT a catalog of controls/tests/findings — the
> filesystem is that index (`controls/*.py`, `results_*/`, `out/*.json`, each self-describing).
> **Claims below carry their LOCATION, not their numbers.** To learn or extend a result you
> MUST open its result JSON (it embeds `metric` / `thresholds` / `decision_rule`) — do not
> cite a claim from this doc without reading its source. Full chronological record:
> `archive/research_log.md` (verbatim, dated 2026-06-15 → 06-22, PARTs 1–7).

> **Entry ritual (the forcing function — see `README.md`).** (1) Faithfulness gate: reproduce
> a result's committed numbers before building on it. (2) `latent_skeptic` triage on every new
> load-bearing claim — a crux is verified by running, not reading.

---

## Terminology (decided 2026-06-22; applied forward, earlier docs not retro-swept)

- **"caving" is a behavioural event, not a mechanism** — reserve it for: a *realized argmax
  flip* C→W\* under social pressure on a fact the model held. Do not name a circuit "caving."
- The mechanisms are two **dissociated** objects (copy-KO does not move deference; doubt
  content-swap kills deference but copy-KO does not):
  - **token-copy** — the base primitive: read a referenced prompt token, OV-copy it to the
    answer slot. Recruited by *salience/prominence* (name-mover; `→ FRAMING §8`) or by an
    *asserted value* (numeric; `→ FRAMING §10.2`). NOT recruited by social doubt.
  - **doubt-conditioned deference** ("the doubt circuit") — a head set that READS the user's
    challenge span (QK) and WRITES toward the expected/asserted answer (OV).
- **The objects of study are prompt-specific CIRCUITS** (components + read/write/route), not
  directions. The cave-DIRECTION is a causal *aggregate* that contains the input circuit's own
  writes — a handle, not a mechanistic stage.

- **"withheld" is retired. The category is "no answer mentioned"** (decided 2026-07-28, from the
  researcher; applied forward, earlier docs not retro-swept). It pairs with the Sankeys' **"both
  answers mentioned"**, and together they name what the label actually is: a statement about which
  entities the reply *mentions*, not about the model's disposition. This is a rename, not a
  remeasurement — **no count changes**; the underlying label is unchanged (`NEITHER` in
  `faithful_rescore.LABELS`, `other` in the `commit_*` vocabulary).
  **Why the rename is load-bearing rather than cosmetic:** "withheld" asserts a choice not to answer,
  and the evidence says that reading is wrong at two of three scales — 2b-base is asserted
  confidence, 27b-base is off-target answering (manufactured by the elicit-context defect), and only
  9b-base is genuine uncertainty (`docs/drafts/TAXONOMY_withholding.md`, re-derived in
  `REDERIVE_20260728.md`). The neutral name stops the word from doing interpretive work the
  measurement does not support. Genuine uncertainty, where it is shown, keeps its own name and its own
  count. Occurrences of "withheld" / "abstain" / "declines" in the drafts and in instrument field
  names (`abstain` in the gate blocks) are a sweep still owed — the field names are frozen artifacts
  and must NOT be renamed retroactively, so the mapping belongs in prose: **`abstain` / `NEITHER` /
  `other` all print as "no answer mentioned".**

---

## Current claims (location + open crux only — read the source for numbers)

Hardened (survived `latent_skeptic` + a powered or cross-method check):

1. **Caving behaviour scales but its metric does not.** -it caves / base resists at 2b·9b·27b,
   BUT the `M = logp(C) − logp(W*)` metric is a tail-token artifact in -it chat and behaviourally
   real only in **base Q/A**. → `results_*_cavecheck/`, `results_9b_faithcaving/`. Crux that
   forced this: the Makelov overlay test (`→ archive/research_log §"ARC-LEVEL RECONTEXTUALIZATION"`).
2. **The doubt circuit (the current positive).** On the faithful base Q/A readout, a concentrated,
   head-SPECIFIC ~5-head set READS the challenge span and WRITES toward W\* (decision BOTH);
   replicates at 2b and 9b base (re-localized heads per scale). → `results_9b_doubtwvr/`,
   `controls/cave_doubt_write_vs_read.py`, `controls/cave_headset_specificity.py`.
   **Source-AGNOSTIC + question-driven:** the same head-set is recruited across
   self/peer/authority/consensus/sourceless (overlap discriminating — plain_assert drops to 1/5),
   and the doubt QUESTION, not the bare assertion, recruits it (content-matched contrast). Crux:
   the bare-assertion non-recruitment leg may be self-repair-masked (resample-ablation owed).
   → `results_social/`, `controls/cave_social_source.py` (`latent_skeptic wf_54281d68`,
   `→ archive/research_log §PART8`).
3. **Downstream of the doubt-head write is DISTRIBUTED** (MLP-heavy, no small bottleneck;
   the 2b attribution graph reads BROAD_DISTRIBUTED — the founding method, node basis, agreeing
   with the 9b direction basis). → `results_9b_circuit/`, `results_9b_doubtroute/`,
   `results_2b_attrgraph/`.
4. **Attention-copy-of-W\* is NOT the caving driver** — overlay / capacity-not-use at every
   faithful scale; copy-KO is never necessary at base under any framing. → `results_9b_faithcopy/`,
   `results_2b_faithcopy/`, `results_*_promptfeat/`.
5. **RLHF edits no copy-head routing weights at any scale** — QK intact (2b·27b), OV direction
   intact at 27b (gain rescaled but latent); the 2b copy collapse is residual-INPUT-mediated.
   → `results_2b_qkweight*/`, `results_27b_qk/`, `results_27b_ovmag/`, `results_27b_realattn/`.

Standing NULLs (each arbiter- or power-confirmed):

6. No single installed deference head; the "installed head-SET" was **retracted under power**
   (n=41 matched de-confound). → `archive/research_log §PART3/PART4`, `results_9b_matched_wide/`.
7. No entropy/confidence neuron on Gemma-2-9b (single-neuron and group grain). → `results_9b_entropyneuron*/`,
   `results_9b_entropydistrib/`.
8. No confidence gate — steering a causal confidence axis does not suppress caving; cave ⊥ that
   axis. Confidence also does NOT gate the doubt circuit's *recruitment* (UNCONDITIONAL, 3-proxy null)
   — but only shown WITHIN the caving (near-tie, range-restricted) regime; a high-confidence arm is
   owed. → `results_9b_gate/`, `results_9b_confgatefaithful/`, `results_social/cave_confidence_recruitment_9b_base.json`.

10. **No single causal LEVER for fold/listen adoption at ‑it — the mechanism is a distributed MONITOR.**
    The pre-registered one-handle question (`DESIGN_foldlisten_mechanism.md` Phase 3) resolves NEGATIVE at
    9b‑it on the frozen 74-item family: (3a) the read-side head subset dies at derivation (greedy EMPTY
    both arms, best single-head KO 0.028; total-mask kills folding but no sparse subset does → redundant
    read); (3b) write-direction resample-ablation flips ZERO of 37 realized answers (= random floor), the
    arbiter SIGN_DISAGREEs (direct −1.81 vs total +2.24), and backup restores → `MONITOR_AGAIN`. Converges
    with the base cave-DIRECTION MONITOR (§9) and the 2b BROAD_DISTRIBUTED graph. → `RESULTS_FOLDLISTEN.md`
    Addendum 5+6, `results_foldlisten_p3a/`, `controls/foldlisten_phase3{a,b}.py`. **Grounding (isolated
    reader, adversarial):** 3a fully H3-grounded; 3b's LOAD-BEARING leg (necessity = `neither_beats_floor`)
    IS grounded — all 24 arm rates reproduce and wf→l/wl→f generations are character-identical to baseline
    (ablation flips 0/37), and that leg ALONE forces MONITOR. The arbiter (SIGN_DISAGREE) + backup (315×,
    fragile near-zero denom) + probe AUROC 0.755 are UNAUDITABLE (per-item values not persisted) and
    corroborating-only — verdict does not rest on them. Reproduces across two greedy runs. THINK/SAY was
    NOT usable at 3b (`think_flip` collinear with the arm) → belief-vs-compliance was UNANSWERED.
    **PARTLY RESOLVED (2026-07-05, Phase-4 offline):** a VALID in-domain THINK probe
    (neutral-arm-supervised, breaks the collinearity — the literal realized-label spec is degenerate at
    greedy) PASSES the masked-arm guard the 3c stated-probe failed and reads the committed answer
    mid-stack (Sun's band). Fold reads the caved W\* mid-stack (REFUTES late output-only overlay;
    discrete vertex-jump depth unresolved, below probe floor); listen shows a mid-stack W\*→C revision
    crossing. → adoption leans **MID-STACK STATE-CHANGE, not compliance overlay** (compatible with the
    distributed MONITOR — a monitor can carry a real mid-stack state). → `RESULTS_FOLDLISTEN.md`
    Addendum 8; verdict JSON `results_foldlisten_p3c/out/foldlisten_phase4_indomain_probe_p4_9bit.json`.
    Instrument debt (H4): per-item arbiter/backup persistence LANDED in
    `controls/foldlisten_phase3b.py`; arm↔direction collinearity broken via neutral-supervision. Still
    owed at scale: a realized-label in-domain probe (needs sampled per-sample captures = GPU) at 2b/27b.
    **SCALE-TRANSPORT DONE + grounded (2026-07-07, `RESULTS_FOLDLISTEN.md` Addendum 9):** MONITOR_AGAIN
    REPLICATES at 2b AND 27b — all THREE scales (2b/9b/27b) give the identical signature (SIGN_DISAGREE
    arbiter, necessity at floor, read WEAK_AT_DERIVE, write both-at-floor, backup restores; 2b+27b now
    per-item auditable). No single causal caving lever at any scale; the ≥2-scale bar is exceeded (3/3).
    The belief-state READ (in-domain THINK probe) is by contrast **9b-SPECIFIC**: PROBE_INVALID at 2b
    (not separable), INSUFFICIENT_LABELS at 27b (neutral-supervision breaks — 27b drifts off stated-wrong
    answers). → `results_foldlisten_mech_{2b,27b}/out/` JSONs.

Direction-level, NOT a circuit (framing-corrected):

9. The cave-DIRECTION is a causal handle on M, an overlay on -it behaviour, and in base a real
   W\*-suppressor / restore-to-neutral carrier (rank-1, specific, non-circular) — but a direction
   is not a mechanism. → `results_9b_carrierdecon/`, `results_9b_readerpp_mid/`.

---

## Open questions / current frontier (swept 2026-07-02; previous sweep 2026-06-22)

- **[PARTLY ANSWERED (v6, triage-corrected) — the headline open Q] Does RLHF *install* the doubt circuit, or amplify a
  base-present one?** **PARTIAL:** base attention doubt-heads carry the cave-state (read/write **0.37/0.24**,
  head-specific); the specific ‑it heads tested (~10: challenge-readers + top cave-axis-writers) are **inert
  (≤0.018)** despite the ‑it cave-state being readable (AUROC **0.92**). The leap to "RLHF moves caving to a
  NON-ATTENTION/distributed substrate" is **RETRACTED to OPEN** — `latent_skeptic wf_f807a702` (selection-bias
  EXPLAINS): by-elimination over only ~10 heads, no unrestricted-attention upper-bound, no ‑it positive control
  (restoration channel unverified in ‑it), base/‑it label mismatch. **DECISIVE close:** ‑it ALL-attention KO (upper
  bound — ~floor ⇒ distributed rescued; ~0.37 ⇒ head-selection artifact) + ALL-MLP patch (positive localization) +
  an ‑it positive control + label-match. → `§PART8 v6` correction, `results_residstate_close/`.
  The blocked-readout history that led here —
  Blocked by the -it faithful readout: chat-template gives a P(W\*) tail
  ghost (~0 faithful items); QA-template -it gives too few (n≈5) and non-specific restores.
  Needs a behaviourally-faithful flip-rate -it readout (graded generation / answer-set), not
  the single-token M. This is the through-question the single-family design exists to answer.
  **Attempt 1 FAILED (PART8 v3, `cave_faithful_it_diff`, 2b+9b):** assistant-prefill "The answer is" +
  answer-set did NOT unblock -it (readout_frac 0.11/0.16, n_faithful 0/2 → READOUT_STILL_BLOCKED both
  scales) AND the prefill failed its own free-generation validator at base (R2 agreement 0.0/0.125) —
  prefilling a generative stem manufactures caves. **Next instrument: forced-choice MC reformat**
  ("A) {C}  B) {W\*}  Answer:" → one decidable letter token, CAA/De Marez proper) and/or generation-grading
  as primary. → `archive/research_log §PART8 v3`, `DESIGN_faithful_it_readout.md`.
  **Attempt 2 — UNBLOCKED IN PRINCIPLE (PART8 v4 spike):** v3 failed because it read the OUTPUT (logit/answer-set);
  the RESIDUAL carries the cave-state. A held-out residual cave-state read (resid·dir, L24–32) predicts the
  judge-graded ‑it free-gen cave at AUROC **0.92** → M1 (a committed ‑it cave-state) HOLDS, upstream of the
  softcapped unembedding. **Read the residual STATE, not the emitted token.** NEXT = base↔‑it doubt-head battery
  with the residual-state readout (ablate doubt heads → does resid·dir drop). Caveats: monitor-not-mechanism;
  n=40 wide CIs; base AUROC inverted low (self-judge noise vs RLHF-creates-the-disposition, OPEN).
  → `results_spike_eot/`, `archive/research_log §PART8 v4`.
  **Attempt 3 — a LEAD (PART8 v5 residual-state battery):** readout works BOTH models (base cave-axis AUROC 0.77
  on realized-argmax labels → the base-inversion was self-judge noise; ‑it 0.92). **Dissociation:** the doubt-heads
  CARRY the cave-state at BASE (read-KO 0.36 / write 0.26 vs random 0.009) but are INERT at ‑it (0.005 / 0.001),
  despite the ‑it cave-state being strongly readable → **RLHF does NOT keep the attention doubt-circuit; ‑it caving
  is non-attention (distributed/MLP)** — closest verdict DISTRIBUTED / relocates-off-heads (not install/amplify/
  reshape). Formally **INSUFFICIENT**: base & ‑it cave on DISJOINT items (intersection 0 → unmatched), n=14, and
  ‑it ~0 could be a localization mismatch. CLOSE: matched both-cave intersection + ‑it re-localize (sweep
  READ_LAYER, re-rank ‑it heads). → `results_residstate/`, `archive/research_log §PART8 v5`.

  **Gaps to close before any RLHF→doubt-circuit verdict (post-triage `wf_f807a702`, current understanding):**
  1. **Attention vs distributed — OPEN.** "‑it heads inert" covers only ~10 heads (span-top5 + DLA-top5). Needs an
     **unrestricted ‑it attention KO** upper-bound: ~floor ⇒ distributed; ~0.37 ⇒ head-selection artifact (it *is* attention).
  2. **No ‑it positive control.** Nothing yet shows ANY ‑it intervention restores the cave-projection — the
     restoration *channel* is unverified in ‑it (only the readout AUROC is). Needs a full-residual u_cave ablation that restores.
  3. **"Distributed/MLP" not positively localized** — it's by elimination only. Needs an ALL-MLP / DLA-to-MLP positive number.
  4. **Label/construct mismatch.** base = realized-argmax, ‑it = self-judge; READ/WRITE probes built on the base
     counter-token construct. Needs a label-matched re-run (‑it under the base label).
  5. **Self-repair uncontrolled** — only zero/output-patch on record; needs mean/resample-ablation (heads could carry
     it but downstream compensates → net ~0).
  6. **Power** — n=28 union / 14 caved each, in-sample head ranking, no LOO/bootstrap CI.
  7. **Fitted-axis readout** — causal-on-the-axis, not on a verified mechanism (SyA-overlay risk; mitigated by the
     behavioural AUROC gate, not eliminated).

  **PART8 v7 (2026-06-23) — gaps 1 + 3 CLOSED on the MONITOR readout:** the ‑it ALL-attention KO ran —
  ALL-attention restores **0.875**, ALL-MLP **0.751** → the v6 "relocates-off-heads" verdict is **REFUTED**;
  honest verdict **REDISTRIBUTE** (attention-heavy but not head-sparse at ‑it), with ALL-X KO flagged as weakly
  discriminating. → `POSITION_KNOWING_BEFORE_SAYING.md:308-315`. STILL OPEN: the same KO on the
  CONTENT/REALIZED readout of the decorrelated family (the monitor readout may not track realized adoption)
  — now Phase 2 of `DESIGN_foldlisten_mechanism.md`, which carries the v7 numbers as its prior.
  **Phase-2 KO (2026-07-02, realized readout; audited + scope-corrected same day):** masking all heads at
  all layers from the challenge turn: fold 1.000→0.041 = masked-neutral floor, coherent generations.
  [2026-07-29: 0.041 is the v1-matcher print; the operative phase-3 floor anchor is 0.027 (matcher-v2,
  `results_foldlisten_p2/matcher_v2_rescore.json`) — phase 3a/3b/3c all scored/cited v2.] NOT a
  closure of this gap — total-mask necessity is partly information-theoretically forced (attention is the
  only cross-position route), and the v6/v7 question is about the CARRY side, where v7's REDISTRIBUTE
  stands. What it does earn: mask instrument validated for Phase-3 head-subset KOs; content-free social
  compliance dead (challenge-blind model confabulates agreement, folds at floor); floor anchor 0.041.
  Audit + matcher-v2 rescore (23/~1600 label flips, zero decision movement):
  `RESULTS_FOLDLISTEN.md` Addendum 4, `results_foldlisten_p2/matcher_v2_rescore.json`.
  → `results_foldlisten_p2/`, `controls/foldlisten_phase2.py`.
- **[CURRENT FRONTIER (2026-06-23 → 07-02) — verifier positive control FOUND at ‑it; mechanism plan registered.]**
  The verifier-POC arc (v0–v3) settled that 9b BASE does not genuinely cave on the decorrelated entity family —
  decoded "caves" are ABSTENTION not adoption (margin-flip ≠ answer-flip; 1/22 outputs W\*) → `21c11c8`,
  `4fad46a`, `1666d21`, `5cbdbdf`, `POSITION_ATTRGRAPH_VERIFIER.md`. The fold/listen behavioural arc then found
  genuine adoption at **-it, every scale** (MOVEMENT_BOTH; fold 0.57–0.81, listen 1.000, abstain ~0, neutral
  drift low → push-attributable; base is drift-contaminated) → `RESULTS_FOLDLISTEN.md`, `results_foldlisten*/`,
  `bf81042`. That is the positive control the de-collide arc lacked. The mechanism question — ONE causal handle
  for both fold and listen at ‑it, LEVER vs MONITOR, direct==total arbiter, THINK vs SAY — is pre-registered in
  `DESIGN_foldlisten_mechanism.md` (`4ef7885`). Phase-0/1 status: measurement layer implemented; substrate gate
  PERSISTED as artifacts (`results_foldlisten*/out/foldlisten_gate*_*.json`). The expansion round ran
  (2026-07-02, `results_foldlisten_ext/`, repro EXACT): behaviour generalizes to 34 unseen items (fold 0.576),
  and the same-model SELF-JUDGE FAILED its pre-registered human validation (belief-contaminated on contested
  items: judge-vs-human 0.679 vs commit_prog 0.982, n=56 hand-labelled) → measurement layer v2 = commit-only
  faithful, judge diagnostic. v2 dissolves the old marginality: 9b-it 13/22, 27b-it 12/22, 2b-it 17/22 (2b was
  judge-blocked, not caving-blocked). Screen yield 16/34 = 47% (T1-heavy; cold anchors cause neutral-arm
  drift — curation lesson). Round-2 (2026-07-02, `results_foldlisten_r2/`): 82 more unseen items (2
  claim-blind drafters -> 2 independent web verifiers -> 82 KEPT, `PROVENANCE_ext2.md`) cleared the unseen
  `--gate --v2` NOT at margin (fold 0.662, faithful 53/82); screen 45/82. **9b-it mechanism pool = 74
  fold-faithful (base 13 + ext 16 + ext2 45), CLEARS the ~60 target**, frozen `mechanism_family_9bit.json`
  (T1 56 / T2 9 / T3 9 — superlative-dominant; content-category robustness owed at Phase 3). **Phase 0.5
  THINK probe DONE = PROBE_VALID** (answer-identity heldout AUROC 0.84 @ L19, floors ~0.50;
  `controls/think_probe_identity.py`; distinct from the 0.92 cave-STATE axis per C4). Next: Phases 2-4.
- **[DISCHARGED 2026-07-29 — was: GATED on capacity] The doubt circuit at 27b** ran, re-localized
  per scale (own span-ranked head selection over 46 layers × 32 heads): decision BOTH at 27b-base.
  → `results_doubt_27b/out/cave_doubt_write_vs_read_27b_base.json`. The base doubt-circuit result
  now stands at ALL THREE scales. The `-it` twins remain INSUFFICIENT by power, not by choice
  (9b-it n_faithful 5; 27b-it headset n_faithful 0, and its pool is 66 not 891 — a different
  substrate from the 9b twin). → `results_doubt_27b/out/cave_headset_specificity_doubt_27b.json`.
- **[INFRA-BLOCKED] Finer write-content of the doubt heads** — DLA-link / direct-logit write
  (6× ssh-abort 255 on teardown; the behavioural output-patch already answered the WRITE
  question, this is the finer decomposition). → `controls/cave_doubt_writes_cavedir.py`.
- **[ANSWERED (PART 9 fold-vs-listen)] Does-caving-carry** — do framing-specific circuits converge on the
  SAME components? YES for the doubt direction: a regressive push (FOLD, holds-C→W\*) and a progressive push
  (LISTEN, holds-W\*→C) share ONE circuit at base — head overlap 4/5 (the canonical doubt heads
  [25,15],[2,13],[26,7],[23,5]) + a transferable cave-state (cross-cell axis AUROC 0.82). The doubt circuit is
  **plausibility-gated answer-revision** (AGAINST-GRAIN to an unrelated wrong target ≈0 at base), sign-agnostic
  in residual space — not a wrongness-specific "fold" organ. → `results_fold_vs_listen/`, `archive/research_log §PART9`.
  (Owed: numeric/salience-copy convergence still untested; only the doubt↔correct-update pair is done.)
  RE-SCOPE (2026-07-02): the PART9 YES is BASE + CORRELATIONAL (shared heads, shared axis). The ‑it CAUSAL
  version — one handle that cross-transports between fold and listen on the realized readout — is strictly
  stronger and OPEN; it is exactly Phase 3 of `DESIGN_foldlisten_mechanism.md`. The owed numeric/salience-copy
  convergence stays owed: it is out of that DESIGN's scope by choice, parked here so it is not lost.
- **[METHOD DEBT] Raw `capitulation` (pre−post) is headroom-confounded** — re-express the prior
  load-bearing caving magnitudes (§11, R-4, dose-response, 2b cavecheck) as **flip-rate** and
  spot-check whether any prior conclusion moves.
- **[READ + VERIFIED 2026-07-05] Yang & Jia arXiv:2505.16170** — the closest published neighbour, now
  audited against our monitor-trap gauntlet by two independent claim-blind reads (convergent; H1). Their
  positive lever and our negative-lever (§10 distributed MONITOR at ‑it) are **COMPATIBLE, not
  contradictory**: their "retraction" is SPONTANEOUS single-pass self-correction (no pushback turn), on
  Llama/Qwen/Olmo (no Gemma), along a factual-correctness belief axis — a different behaviour/axis/model
  from our social pushback-caving. Their lever would NOT pass our monitor-trap standards (per-check
  verdicts + their group-mean "prediction" plot vs our AUROC bar live in §D). Net: complicates the "no
  lever anywhere" framing but does NOT de-risk our pushback lever hunt and gives NO counter-evidence to
  the monitor. Cite as complementary neighbour, not precedent. Full gauntlet + provenance:
  `docs/NOTE_phase34_improvements_lit.md` §D + STATUS LEDGER.
- **[PARTLY RESOLVED] Social source scales doubt-circuit recruitment — gradient REAL, authority-per-se
  marginal.** Bootstrap CIs (v2): authority > self/sourceless excludes 0 on READ at both scales (and WRITE
  mostly) → the social gradient is real, not noise. The matched minimal-pair (professor vs friend, same
  frame) isolates *authority* specifically only at **9b-READ** (+0.035, CI excludes 0); WRITE and all 2b
  cells straddle 0. Still owed: per-cue `self-repair` (resample-ablation) + held-out / 2nd speech-act.
  → `results_social_v2/`, `scratchpad/social_ci.py`, `→ archive/research_log §PART8 v2`.
- **[RESOLVED (PART 9 fold-vs-listen)] "Deference fires for wrong-not-truth" — NO, it is not wrongness-specific.**
  The against-grain headroom-symmetric design (push always against the lean) shows the SAME doubt circuit serves
  the progressive (LISTEN→C) push too (shared heads + transferable cave-axis), so it is **shared answer-revision,
  not a wrong-only organ**. It IS plausibility-gated (AGAINST-GRAIN≈0 at base). The PART8 retraction stands and is
  now positively explained. (Formal SC withheld as MOVE_UNMATCHED — LISTEN caves at a higher rate than FOLD — so
  the recruitment-MAGNITUDE asymmetry is not yet clean; the shared-circuit read does not depend on it.)
  → `archive/research_log §PART9`, `results_fold_vs_listen/`.

Parked by choice (single-family depth): deployment regime, cross-architecture, SFT-vs-RL stage
attribution. The last is externally blocked — Gemma ships no staged checkpoint (`-it` = SFT+RLHF+merge);
OLMo 2 (arXiv:2501.00656) / Tülu 3 (arXiv:2411.15124) would resolve it but break the single-family
scope, and model-diffing crosscoders (Anthropic, 2024) are the in-family alternative if revisited.

---

## Directional commitments (the standing scope, from the user)

- **Single Gemma family is deliberate** — minimise confounds, go deep, hunt reusable circuit
  motifs (cell-biology intuition). Cross-architecture is out of scope by choice, not a gap.
- **The base ↔ post-training differential is the rich seam** — it mechanistically explains things
  people already know work.
- **Resist over-metricising** — one screen + one confirm + a base-as-null; do not build a metric zoo.
- **Honest nulls are results.** The arc is mostly well-verified negatives; a positive (the doubt
  circuit) is banked, but the discipline that produced the nulls is the asset.

---

## Handoff seed (latest — overwrite this each session)

> /karpathy-guidelines
>
> **SEED 2026-07-30/31 (compose-post1 session) — NEWEST. STACKED above the previous seed.**
>
> **GPU RAN AND IS DOWN. `GET /api/v1/instances` returns 0, verified after teardown.** One
> A100-SXM4-40GB, us-east-1, 3h32m inside a 7h cap, ~$8–9. **Spend reconstructed before launch as
> §11 requires: the audit log spans 2026-02-22 → now and $830.50 of it is PRE-PROJECT; project
> -attributable since 2026-06-10 is $711.75, so headroom against the $950 cap was ~$238, NOT the
> previous seed's $364.53.** Reconstruct, never read a committed tally — including this one.
>
> **1. THE DE MAREZ SPAN RUNS LANDED (`8b83151`), and the pickup was not the run, it was the audit.**
> The prior session (`d969872`, killed by a rate limit) left three never-run instruments. An isolated
> pre-launch audit (`docs/drafts/AUDIT_demarez_prelaunch.md`) returned **NO-GO on four blockers**, all
> reader/writer plumbing — including a `mask.py` selftest that **could never pass** (it asserted the
> entity token was in `frame_tokens` while the next line asserted the two disjoint), which would have
> exited at the model-free gate for a billed box and zero data. Fixed at `0105d18`, registered as
> amendment **Round 3** in `REGISTRATION_demarez_spans.md` §0.6 **before the box booted** (`aa67299`).
> Verdicts in `out/demarez_join.json`, the join being the only verdict source:
> **§6.2 PRIMARY `QUESTION_DOES_WORK`** (r_move(A1)=1.0, r_move(A2)=0.861, r_off(A3)=0.730 —
> **quotable as the triple or not at all**), §6.3 `DOSE_NONMONOTONE`, §6.4 `GRADE_ANCHOR_DIVERGENT`,
> §6.5 `PUSH_TOWARD_STATED_INERT`, §6.6 `MASK_TOTAL`, §6.10 both floors `FLOOR_CONSISTENT`,
> §6.11 concordance 73/74 and 72/74. Both artifacts `RUN_UNDER_THIS_REGISTRATION`, 0 violations.
>
> **THE ONE THING TO FIX NEXT, and it is one line.** §6.7 SPAN / §6.8 DELIMITER / §6.9 ECHO are
> `UNEVALUABLE`, `cause=PAIR_NOT_SAME_BOX` — **not a data failure**: `n_common=74,
> n_span_located=74, n_span_unlocatable=0`. The two artifacts carry the same `lambda_instance_id`,
> same `git_commit`, `device_index` 0 and sequential non-overlapping timestamps, but §1.1's
> mechanical test needs `cuda_visible_devices` and nothing exported it, so both stamped null.
> **The rule was deliberately NOT relaxed** — loosening a registered same-session test after seeing
> it block three verdicts is the post-hoc move the registration exists to prevent. Add
> `export CUDA_VISIBLE_DEVICES=0` to `run_demarez_9b.sh` and re-run (~3.6h, ~$8): it recovers the
> span trio and changes nothing else. Do this before any new distributional GPU work.
>
> **2. OWED B2 IS PARTLY CLOSED, at the cell the post is written from.** Every arm persists
> first-token top-10 + lp(C)/lp(W\*) at **both** slots, so a distributional read now exists at the
> **forced-final slot** — the slot the sankey verdicts are decided on, where `COMPOSE_post1_brief.md`
> §F(a) correctly said no instrument reads at any cell. 9b-it fold only. First look (item 0, not an
> aggregate — the registered aggregation is §4.3's **64 report-only dissociation rows**, no band, no
> verdict): at the reply slot the argmax is `"You"` p=0.9988 with C at rank 937; at the elicited slot
> the argmax is the adopted answer at p=0.99989. **At the decision slot there is barely a
> distribution left.** Nobody has aggregated this yet — that is the single highest-value offline job
> waiting, and it belongs in the notes' "Under the hood".
>
> **3. THREE COMMITTED CLAIMS WERE CORRECTED (`04cda88`), by two isolated readers who agreed
> independently (the second claim-blind).** `docs/drafts/RETRACTIONS.md` R-12/R-13/R-14, with
> per-site addendum text for all six citing documents in `docs/drafts/ADDENDA_20260730_ledger.md`
> — **NOT yet applied to the ledgers; apply by hand.**
> **R-12 WITHDRAWS `REDISTRIBUTE` and the 0.875/0.751.** No instrument emits that string (zero hits
> in every `.json`); the artifact decides `BOTH_REDUNDANT` under the self-judge axis; the headline
> **exceeds its own artifact's CI** because `cave_residstate_decisive.py:258` takes the max of two arm
> means while `:265` bootstraps the pooled list; the `-it` floor it is read against is the hardcoded
> `it_rand = 0.0` (`:303`); the file is `reprocessed_offline` by a script that **is not committed**
> and its per-item cache does not exist, so it is unauditable in the R-1 sense; and
> `cave_residstate_close.json` decides `DISTRIBUTED_CONFIRMED` on the **same model, axis layer and
> pool**. Two committed artifacts contradict each other and no doc said so. **The headline open
> question — does RLHF install the doubt circuit — is back to OPEN**, both intervening verdicts gone.
> **R-13 is a scope line, not a retraction: the doubt circuit's `BOTH` at 3/3 is FIRST-TOKEN-BOUND.**
> `cave_doubt_decollide` returns `READOUT_SENSITIVE` at 2b, 9b AND 27b; under the stripped content
> margin the same interventions on the same items fall to 1.09× the matched-random floor at 2b and
> 1.69× at 27b, only 9b READ clearing appreciably. Fully auditable (per-item records persist; all 27
> means re-derive). **Claim 2 may not be stated again without naming its readout.**
>
> **4. THE POST. Four grounding documents were built and committed this session; the patch blocks
> were NOT** (three drafting agents died to session limits, twice). Ready to hand to drafters:
> `docs/drafts/SNAPSHOT_circuit_groundtruth.md` (§7.1 seven claims that survive with mandatory
> qualifiers, §7.2 ten that do not), `INVENTORY_distributional.md` (859 lines; §3.1's slot table is
> the load-bearing one), `PATCHMAP_live.md` (every tranche block with anchors byte-verified against
> the live vault), `AUDIT_demarez_prelaunch.md`. Live-gold facts a drafter needs: the researcher
> **edited both gold docs after `598de5e`** and no ledger recorded it, so tranche-1/2 intro line
> numbers are **−1** and **C02's anchor is stale/unappliable**; tranche 3's 24 blocks all still
> anchor byte-exact; **notes L319 has a TRIPLE collision (B02 ∥ D02 ∥ T3-05) — do not add a fourth.**
> **The decisive correction for the intro:** "usually assigns higher probability to C" is a **BARE
> -slot** statement (`M0`, 54–74 of 82 at all six cells) and true at the neutral slot (66–81), but
> **false at the pushed slot at four of six cells** (2b-base 36, 2b-it 18, 9b-it 27, 27b-it 39; only
> 9b-base 63 and 27b-base 62 hold). And C is **not** the vocabulary argmax there on 0/82 items at
> five cells. Intro L25's "distributed at -chat" is contradicted by its own overlap numbers (base
> 4/5, **-it 5/5**); T3-03 already holds the honest replacement and this session's audit independently
> confirms it — that block needs the researcher's decision, not another draft.
>
> **5. UNSWEPT, and cheap.** The vault's live Fig 1 embed is md5-confirmed as the **anomalous 27b
> draw** (vault `6942c40b` vs repo `50a3f28f`); all four vault image swaps in
> `COMPOSE_post1_brief.md` §B are still pending and are the researcher's own. The post's lead figure
> currently shows numbers that do not reproduce.
>
> **SEED 2026-07-29 (later session, post-Sun drafting viii) — STACKED above the previous
> seed per the standing convention.**
>
> **NO GPU RAN THIS SESSION. $0. Everything below is offline reads + two commits.**
>
> **WHAT LANDED.** (1) `1853e27` — `docs/drafts/JOIN_post1_crossvariant_scale.md`, the join of five
> isolated read-only investigations over the POST1 cross-variant/scale axis: gold-draft extraction
> (53 fill-slots), ledger re-derivation, RUN-vs-ABSENT inventory, residual-numbers pass (12 items),
> citation vetting (7 slots). Same commit flags four stale ledger lines as DATED ADDENDA, not silent
> fixes: `RESEARCH_QUESTIONS.md` doubt-at-27b gate DISCHARGED (see below), `RETRACTIONS.md` R-3
> three corrections (27b-it NOT identical between draws; the driver attribution refuted — OWED H2:
> the divergence tracks the CARD; NO_MOVEMENT is the commit register, faithful reads
> MOVEMENT_LISTEN_ONLY), `DESIGN_neutral_elicit.md` run-landed status, and five self-corrections
> appended to `GROUNDING_crossvariant_scale.md` (incl. its unauditable "~59×" pair and the
> mixed-provenance 27b row). (2) `d9d884b` — `docs/drafts/PATCHSET_tranche3.md`: 24 hand-apply
> blocks resolving the gold's cross-variant/scale brackets, net −23 brackets, drafted claim-blind
> then held to a two-round adversarial review (round 1: 11 PASS / 13 HOLD; all fixes re-verified,
> 31/31 anchors byte-exact, every new number re-derived). **The vault was never written.**
> (3) The reviewer caught the tranche-2 NBSP anchor defect RECURRING byte-for-byte — the lesson is
> now in the patchset preamble: slice anchors from file bytes, never retype; and the live gold is
> MIXED on guillemet spacing, so blanket NBSP conversion is a drive-by edit.
>
> **DISCIPLINES any future number must obey (all receipts in the JOIN):** every 27b-base figure
> names its decode draw (committed ext2 = anomaly, nelicit re-run = reproducible) AND its register;
> 27b-it is ALSO draw-dependent (82/164 counter_gen differ — per-field check, no blanket exemption;
> 2b/9b are 0/164 clean); on scale ordering within a variant, 9b and 27b NEVER separate (p=1.0 -it,
> 0.289 base) and 2b separates except from 27b-base — 3 of 6 comparisons null
> (`out/gapclose_foldrate_sig.json`) — the cross-variant gap is the decisive axis, not scale; the
> fmt triple is quotable whole or not at all.
>
> **FOR THE RESEARCHER (blocks their own decisions):** T3-03 (the intro-L25 mechanism sentence —
> no run supports it as written; the block offers the honest replacement) is NEEDS-RESEARCHER-
> DECISION; T3-16 depends on T3-09; T3-01/T3-21 apply together or not at all. Application order and
> shared-line notes are in the patchset preamble.
>
> **CORE RESULTS STILL UNRUN (ranked, JOIN §D):** OWED B2 (forced-final-slot distribution/residual
> read — the slot verdicts are decided on); the listen distributional column (withdrawn + 27b
> three-cluster instability); the base mechanism arm + 27b `cave_fold_vs_listen` (the only path to
> intro-L25 at strength); hand-labels for the headline cells (9b VF22, 9b ext2, all base ext2, all
> listen, T3n — the post's central cell has no human agreement statistic); -it top-k with a
> regime-aware key (K4 + §4.1 key fix).
>
> **NO GPU IS RUNNING. VERIFIED 2026-07-29 (this session): `GET /api/v1/instances` returns 0
> instances.** Both boxes of the run below are down.
>
> **WHAT LANDED: `a34d6e6` — the format-matched readout. 31/31 registered cells across two boxes, and
> the first run in this project with provenance fully stamped.** Registration
> `docs/drafts/REGISTRATION_format_matched_readout.md`; verdicts `out/fmt_matched_join.json`. The
> primary triple is `(RANK_RESOLUTION_INSUFFICIENT, RANK_RESOLUTION_INSUFFICIENT, ANCHOR_DIFFERS)` and
> it is **quotable as a triple or not at all** — no member of it reads as a result alone. It CLOSES
> `docs/drafts/OWED.md` C1: the base-vs-`-it` W\* bare-rank gap is a FORMAT artifact.
> **Two disputed thresholds were WITHDRAWN, not re-guessed** — read the registration knowing its
> amendments SHRANK what it can decide; a bar invented after the dispute would have been a fitted one.
> **Three prior commits are corrected (`OWED.md` §H):** `a4a2ae0`'s listen-arm withdrawal rests on a
> premise that does NOT reproduce (§10 = `SHIPPED_SELF_IDENTICAL + ARMS_MATCHES_SHIPPED`); `2dd19b8`
> blamed the DRIVER for the 27b divergence when it tracks the CARD; and 16 of 18 anchors reproduce,
> with only 27b-base differing.
> **WHAT IT DOES NOT SETTLE: anything about the COMMITTED 27b digits.** It speaks for its own box only
> — every 27b-vs-committed comparison is `DISCLOSED_NOT_GATED`.
> **NEXT: read `docs/drafts/GROUNDING_crossvariant_scale.md`** (the cross-variant / scale grounding
> pass) **and `out/fmt_matched_join.json`** for the verdicts — not any prose summary, this one included.
>
> **NO GPU IS RUNNING. VERIFIED 2026-07-28 (this session): `GET /api/v1/instances` returns 0
> instances.** The "RUN IN FLIGHT AT HANDOFF" block further down is RESOLVED and superseded — both
> boxes are down, the neutral-elicit run landed, and its artifacts are committed under
> `results_foldlisten_nelicit_{2b9b,27b}/out/`. Do not act on that block's reattach instructions.
>
> **THE FREE/OFFLINE GAP CLASS IS WORKED. $0 GPU. Start at `docs/drafts/GAPCLOSE_RESULTS.md`** — the
> per-gap decision table for `GAPS_RECONCILED.md` §4.1 — then the four documents it points at.
> Pre-registration first, committed before any value was computed:
> `docs/drafts/REGISTRATION_offline_gapclose.md` (read its §4.1 amendment and §13 corrections).
> New instruments, all claim-blind authored → reviewed → selftest → run:
> `controls/gapclose_{contam_census,span_taxonomy,item_joins,small,cells_faithful_merge}.py`,
> artifacts `out/gapclose_*.json`.
>
> **What the next agent most needs to know, in order.**
> (1) **F1 FAILED its validation — and my first diagnosis of WHY was wrong. Read the CORRECTION in
> `GAPCLOSE_RESULTS.md`.** The registered verdict `TAXONOMY_UNUSABLE` stands (pooled strict 0.517 /
> 0.500 against a pre-fixed 0.75 bar). But stratified by slot — **declared post-hoc, so it licenses no
> usability claim on its own** — inter-reader is **1.000 on `elicit_gen` (37/37) and 1.000 on the T3n
> slot**, 0.946 on `counter_gen`, and **0.189 on `neutral_gen`**; and **30 of the 32 disagreements are
> ONE confusion cell** on spans the rule calls `NEUTRAL_ACK` — a label the readers' vocabulary lacked
> because the §4.1 amendment was made *after* they were launched. Excluding those spans, inter-reader
> is **88/90 = 0.978**. Reader-vs-rule by slot: `elicit_gen` **0.919 / 0.919**, `counter_gen`
> 0.541 / 0.486, `neutral_gen` 0.054. So the failure is a vocabulary gap I introduced mid-flight in
> one slot, NOT an undecidable construct — and the elicited slot, where every headline count is taken,
> is in good shape. **What that earns is a new registration to test the elicited slot specifically.**
> Two defects in my own artifacts came out of the same re-read: `label_pre_amendment` is `None` on all
> 8456 records (so one of the three readings I reported is VACUOUS), and the committed sample file's
> `label_space` misrepresents the vocabulary the readers were given. Both recorded in the handread
> artifact. Standing consequence for the prior work: `TAXONOMY_withholding.md`'s fine-grained splits
> still rest on ONE reader with no agreement statistic.
> (2) **`docs/drafts/RETRACTIONS.md` holds 11 entries** (registration owed #10, now written). R-1 is
> UNFIXABLE and verified so: a key-level walk over all 323 artifacts finds **zero** hardware, driver
> or library fields and no launch id, so neither side of the audit-log join exists. R-6/R-7 withdraw
> three printed figures that are in no revision. R-8 reclassifies a REPRODUCES to UNAUDITABLE.
> (3) **`docs/drafts/REDERIVE_20260728.md`** — the entry gate. `JOIN`'s arithmetic is CLEAN (every
> 2×2, χ², Fisher, Wald CI, McNemar re-derives; its join is genuinely keyed with zero symmetric
> difference). Seven numbers elsewhere do not reproduce. **The 27b-base column is one draw**: 98/164
> generations and 41/164 labels differ between the two runs against 0/0/0 at 2b and 9b, so every
> printed 27b-base number must name its decode.
> (4) **`docs/drafts/CODEBLOCKS_verified.md`** — all K1–K15 lines read. Six understated (K4/K12/K13/K14
> are multi-line prompt-builder refactors, not argparse; K5 is a recalibration; **K8 needs a re-run and
> cannot be done offline**). Two move DOWN: K15's four ambiguities all resolve from the code, and one
> of K11's seventeen already has `--layer`. D14 settled: 16 ⊂ 61 ⊂ 66 ⊂ 891 — **and 817 of the 891 are
> downloaded at run time while `_build_pool` printed-and-continued on failure, so a networkless re-run
> silently measured 74 items against 58 artifacts stamping 891.** FIXED 2026-07-28 — it now raises.
> (5) **`out/gapclose_scorer_drift.json`** — the first repo-wide scorer-drift audit: 2 of 6704
> committed `faithful_*` labels disagree with the current scorer (0.03%), both in one file, both the
> same tie-break, both diagnostic-slot. The port is 99.97% equivalent.
> (6) **Registrations owed: 12 → 7.** Written this session: **#5** (base gate thresholds, written as a
> *refusal* to invent one — the ‑it thresholds transport unchanged and are stamped
> `THRESHOLDS_NOT_CALIBRATED_FOR_THIS_REGIME`), **#9** (`docs/drafts/REGISTRATION_provenance.md` — the
> load-bearing pair is `lambda_instance_id` + `started_utc`, without which R-1 recurs), **#10**
> (retraction register), **#12** (the five-part number stamp, asserted by every new selftest). **#11**
> (family accounting) is settled by measurement. Still owed: **#1, #2, #3, #4, #6, #7, #8.**
>
> **Next, if picking up the ledger in cost order:** CODE FIRST is next, and `CODEBLOCKS_verified.md`
> re-costs it. The cheapest real win there is K3 (one line + argparse, 7 claims). K1/K2/K5/K6/K7/K8 all
> need a registration written first — #1, #2, #3, #4 in the owed list are exactly those.
>
> **OPEN LEDGERS — the three blind audits and their reconciliation (2026-07-28).**
> Three blind coverage audits plus their reconciliation: `docs/drafts/GAPS_A_instruments.md` (from the
> instrument code; 311 absences, 68 code-blocked with the blocking line), `GAPS_B_artifacts.md` (from
> 300 result JSONs; 15 structural absences, 11 inconsistent artifacts), `GAPS_C_claims.md` (from the
> write-up; 163 claims, 41 ABSENT at the breadth written), and **`GAPS_RECONCILED.md`** — 54 MECE gaps
> classed FREE/OFFLINE 11 (blocking 42 claims), CODE FIRST 15 (52), GPU RUN 14 (30), PROVENANCE 14 (13),
> with a minimal set of 11 steps of which 5 need no GPU. Work the reconciled ledger, not the three
> inputs. Numbers in the drafts: `GROUNDING_neutral_elicit.md`, `TAXONOMY_withholding.md`,
> `JOIN_withhold_vs_fold.md` (read its CORRECTION), `NOTE_27b_repro_fail.md`,
> `GROUNDING_notes_numbers.md`. Draft state: `NOTE_B_post1_notes.md` + `PATCHSET_tranche2.md` (14 blocks
> held, reasons in the reviews).
>
> **REGISTRATIONS OWED before the corresponding work — 12, none yet written except where named.**
> (1) listen-arm distributional readout — a DESIGN change, not a flag: `family_cave_diagnose.py:214-215`
> plants the literal C in both arms and the margin's sign convention assumes it. (2) A distribution or
> residual read at the FORCED-FINAL slot — no instrument reads either there, and it is the slot the
> verdicts are decided on. (3) The base arm of the fold/listen mechanism — `assert is_chat` at
> `foldlisten_phase2.py:155`, `phase3a.py:317`, `phase3b.py:734`, `phase3c_riders.py:325`; register what
> would count as the base mechanism before removing them. (4) Per-scale head discovery — the head-set
> lineage hardwires 9b coordinates (`atp_low_confirm.py:32-34`), so 2b/27b need a discovery procedure
> registered or the result is post-hoc. (5) Gate thresholds for BASE summaries — no gate has ever run on
> one, and the existing thresholds were set on -it; register before computing or they get fitted.
> (6) The blind 3-reader hand-label protocol extended to the cells that lack it (T3n, listen, base, and
> 9b at ext2 — the headline cell). (7) `DESIGN_elicit_context.md` open decisions D-1..D-10.
> (8) `DESIGN_distributional_withholding.md` open decisions, including its frozen power tiers.
> (9) A provenance-stamp rule: instance type, driver and library versions into every summary, plus an
> activation-retention policy (`.gitignore:21-22` currently discards the captures five results depend on).
> (10) A retraction register — two claims are permanently unfixable and must be withdrawn, not repaired.
> (11) Family accounting: the 891-item pool has no committed producer and the 66-item pool is
> unaccounted for by any audit. (12) A house rule that every printed number names its arm, confidence
> mode and tie-break state — the same field reads five ways.
>
> **RUN IN FLIGHT AT HANDOFF (2026-07-28 ~03:50). READ THIS FIRST.** Two boxes are billing:
> `63c9e1da58af403685ab7009d1975fff` (A100 SXM4, us-east-1, box 1 = anchor4 + 9b-base + 2b-base,
> cap 19800 s) and `73a2c8389ee94ec3927839500bddf0c2` (H100 SXM5, us-south-2, box 3 = 27b-base,
> cap 25200 s). Both detached with the on-box self-destruct **armed and confirmed** at
> `cap + REATTACH_GRACE(7200)` — 27000 s and 32400 s — so they tear themselves down even if every
> process here is dead. Boxes 2 and 4 launch sequentially from the same two pollers
> (`run_poll_launch_nelicit_{2b9b,27b}.sh`, logs in the session scratchpad).
> **If you inherit this mid-flight:** `bash lambda_reattach.sh <id> <ip> <rdir>` — use the explicit
> three-arg form, because both boxes in a poller share the `drill_<rdir>` launch name. **Fetch
> before terminating** — a previous run lost its results to a launcher that died after the box did
> its work. Then `GET /api/v1/instances` must return 0.
> **The run is `DESIGN_neutral_elicit.md`** (pre-registered, unrun until now): it adds the elicited
> final answer to the NEUTRAL arm, so the push-attribution stops being a cross-slot comparison.
> Its gate is `controls/foldlisten_repro_diff.py` (committed, selftest PASSES, 12 groups) — every
> cell must clear it before any number from this run is quoted. Pre-data decision on record: the
> n=22 anchor's 44 neutral-elicited records are **PARKED, not published** (out of scope per §1.6;
> deciding after seeing them is how a post-hoc scope change disguises itself).
>
> **SPEND.** Reconstructed from `GET /api/v1/audit-events` on 2026-07-28: **$585.47 spent since
> project start, $364.53 of the $950 cap remaining** — the committed ~$436 tally understated by
> ~$149, so reconstruct, never read. This run is ~$21–26 of that. Two further rounds are registered
> and costed but NOT launched.
>
> **TWO ROUNDS REGISTERED, BUILT-READY, DELIBERATELY NOT RUN** — they need the researcher's open
> decisions first. `DESIGN_elicit_context.md` fixes a real instrument defect:
> `foldlisten_judge.py:423` splices `prior_gen` UNTRUNCATED into the elicit prompt, so base contexts
> are contaminated on 82/82 items at every scale (an invented question on 47/39/69 of 82) while -it
> is 0/82 — the two variants are not asked the same question at the slot the whole base-vs-it
> comparison is read from. `DESIGN_distributional_withholding.md` extends the diagnose instrument to
> the listen arm (which it currently cannot express — `family_cave_diagnose.py:215` builds only the
> counter push) and to 2b/27b, where no margin artifact exists at all. Its power table, frozen
> before data, says UNC is n=0 at 2b and n=1 at 27b, so **no outcome can license a scale-general
> statement about uncertainty**, and the motivating 9b cell (n=20) is per-cell only.
>
> **THE SESSION'S RESULT, and it is a retraction.** `docs/drafts/TAXONOMY_withholding.md` read all
> 234 elicited + 231 free-reply withheld spans individually. The committed counts reproduce (base
> 51/38/32, -it 0/0/1) but **one label covers three different phenomena**: 2b-base withheld is 76%
> asserted confidence and **0% uncertainty**; 9b-base is 53% genuine uncertainty; 27b-base is 94%
> off-target, and those off-target answers are **correct answers to the last question of the model's
> own runaway**, spliced in by the contamination defect above — so at 27b the bug manufactures the
> category. Genuine uncertainty is 34/234 and **33 of the 34 are 9b-base**, the scale the drafts
> generalise from. `docs/drafts/JOIN_withhold_vs_fold.md` separately shows -it's folds are NOT
> concentrated on items -base withholds (25 vs 25.49 expected, Fisher p=1.0) and that the one strong
> association runs backwards (-it folds on 92% of items where -base did NOT hedge, p=0.0008).
> Distributionally, at the one cell with an artifact, withholding is **not fence-sitting**: 9b-base
> UNC items favour C 17:3, median +0.65, indistinguishable from the items the model commits on.
>
> **WRITE-UP STATE.** POST1 is now TWO vault documents (`DARWIN.md_post1_user_intro.md`,
> `…_notes.md`) — gold, never write to them without the researcher's say-so. 39 patch blocks applied
> across two reviewed tranches; snapshots at `docs/drafts/DARWIN_post1_user_*_snapshot_280726.md`,
> so `git diff d9a48f2 f403686` and `git diff f403686 598de5e` are exactly what changed. 14 tranche-2
> blocks are HELD with reasons in `docs/drafts/REVIEW_patches_v2.md` and the tranche-2 review.
> **Four decisions are the researcher's alone** and block the rest: which of L319/L321 survives,
> figure renumbering, the lost head clause near L250, and the L60 speaker tag. **A measured warning
> for whoever writes next:** the second tranche would have inverted their bracket signature — short
> slots (≤5 words) 52% in the predecessor draft, 47% live, 34% if all of tranche 2 landed — so the
> apparatus was starting to outweigh the prose in two sections. Trade bracket load down as a set.
> **Obsidian was open on the vault during these edits** (PID 2952503); a stale editor buffer could
> overwrite them, so reopen rather than save over.
>
> PHASE B FAMILY REPLAY COMPLETE (2026-07-22; evidence = docs/drafts/NOTE_faithful_matcher.md
> Addendum 2 — read it first; all cells H3-grounded by isolated readers). classify() PORTED into
> foldlisten_judge.py (dual labels commit_*+faithful_* per item, scorer_provenance, gate --labels;
> claim-blind + reviewed + 756/756 offline equivalence) and CONFIRMED on GPU: anchor3 reproduces the
> committed 9b-it n=22 BYTE-IDENTICALLY. Five new ext2 (n=82) cells:
> `results_foldlisten_ext2_2b9b/out/` (2b-base, 2b-it, 9b-base + anchor3) and
> `results_foldlisten_ext2_27b/out/` (27b-base, 27b-it). Faithful-strict fold adoption: 2b-it 68/82
> (gate PASS both readings; blind spot-check 82/82), 27b-it 55/82 (spot-check 81/82, 'Persia'
> unlisted-alias flag), 9b-it 55/82 (2026-07-20 rescore); base: 2b 16 (51 abstain — instability),
> 9b 3 (NO_MOVEMENT — "zero adoption" is n=22-SCOPED, see v6 caveat edit), 27b 11
> (MOVEMENT_LISTEN_ONLY). OPEN CONTEST (next agent's first crux): 27b-it ext2 substrate gate —
> commit-labels FAIL (listen drift 13>11.18) vs faithful PASS (drift 7); isolated hand-read finds
> ~15 GENUINE neutral self-corrections ("…is actually **Warsaw**, not Krakow" after a bare
> thank-you) that classify tiebreak_unresolved swallows → defensible verdict = FAIL; tie-break fix
> owed (claim-blind) then re-gate; both readings persisted
> (`foldlisten_gatev2_fl_27bit_ext2{,_labels-faithful}.json`; run_gate now suffixes non-commit
> readings after a silent-overwrite near-miss). The two scorers fail in OPPOSITE directions by
> regime (base: commit inflates via runaways; -it: faithful under-reads bold self-corrections) —
> keep recording both. Fig B alluvial: complete faithful flows for all 12 cells now exist. GPU:
> ~$44 this phase, ~$436/$950 cumulative; all boxes terminate-confirmed.
>
> PHASE A IRON-OUT CLOSED (2026-07-21, per DESIGN_foldlisten_matrix_scaleout.md gates; evidence in the
> docs/drafts/NOTE_faithful_matcher.md 2026-07-21 addendum — read that first): entry faithfulness gate
> re-run by 4 isolated readers, ALL committed numbers reproduce (six n=22 cells + gate_v2, H4 precedent,
> ext2 anchor, faithful_rescore claims). Gate 1 INSTRUMENT VALID AT EVERY SCALE: blind 3-reader
> spot-checks, 88 elicited finals/scale, unanimous vectors →
> `results_foldlisten_{2b,27b}/out/handlabel_spotcheck_fl_*.json` (stored-vs-human 0.989 / 0.955 PASS;
> faithful-strict ZERO genuine disagreements) + `controls/classify_vs_handlabel.py` →
> `out/classify_vs_handlabel_9bit.json` (classify-vs-human 56/56 = 1.000; commit_prog 0.982). Gate 2
> ALIAS MISSES RESOLVED: `ALIASES` in `controls/faithful_rescore.py` (selftested; only-3-move
> git-diff-proven; ext2 fold 53→55/82; 9b-it listen 22/22). Gate 3 SCORER SETTLED (slot-scoped):
> `elicit_gen` scored `map_confidence=False` (`STRICT_FIELDS` — the sec-4/6 confidence→entity mapping
> relabels 15/44 2b-base + 3/44 9b-base elicited finals that unanimous string-identity humans call
> NEITHER); prose arms keep the mapping. PORT into live judges decided-YES but DEFERRED to a GPU session
> (claim-blind pass + confirming run); until it lands, EVERY new run's summaries MUST be rescored by
> `faithful_rescore.py` before any count is used, and new summaries must stamp scorer provenance (the
> committed ones do not — grounded gap). Gate 4 DIVERGENCE measured at all scales (coverage now includes
> `out/faithful_rescore_fl_{2bbase,2bit,27bbase,27bit}.json`): elicit ≤0.114 all-STABLE vs CHANGE_THR
> 0.30; prose arms materially relabeled → carry no claims. POST1 v6 RE-GROUNDED on the faithful-strict
> readout (table cells incl. 27b-base 5/11/6, -it listen 22/22 all scales; TL;DR 55–77%; withheld 0/22
> every scale; ext 19/34 + 55/82 none-withheld; deferential "I think you're right." disclosed, 9b-base
> 0/22 stands). Every new artifact H3-grounded at item level by an isolated reader. NEXT (Phase B, GPU):
> claim-blind port of classify() into `family_generate_judge.py`/`foldlisten_judge.py` + confirming run,
> then the absent matrix cells per the design seed. Human pass on v6 still owed.
>
> MATCHER FIX (2026-07-20): the load-bearing matcher debt is CLOSED (offline). `controls/faithful_rescore.py`
> (claim-blind author → clean review → H3-grounded) re-labels every persisted generation by reading the
> ACTUAL answer (top-line span cut at the `\nQ:` runaway; dismissed-vs-affirmative clause logic; hedge
> lexicon; prefers bare elicit_gen). Corrected FOLD-cell numbers: base W* adoption = 0/82 and **0/22** (the
> post's "1 of 22" was a runaway false-positive → fix POST to 0); base neutral "says W*" 4→0. -it headline
> ROBUST: elicit-based fold adoption UNCHANGED (13/22, 53/82) — only the diagnostic counter_gen prose arm was
> inverted (70%/38% relabel), never the rate. 3 conservative UNRESOLVED_ALIAS (Nur-Sultan=Astana, DRC=DR
> Congo are true adoptions under-counted → ext2 fold really ~55/82). Outputs: out/faithful_rescore_*.json;
> provenance docs/drafts/NOTE_faithful_matcher.md. OWED: port classify() into live family_generate_judge.py /
> foldlisten_judge.py (+alias table) for future runs; Fig B alluvial now unblocked (rebuild from faithful
> fold labels).
>
> WRITE-UP ARC STATE (2026-07-18, for the next agent): POST1 v5 got a FULL-RIGOR EVALUATION →
> `docs/drafts/POST1_v5_evaluation.md` (22 ranked findings; ~30 numbers ALL H3-reproduce; 12
> isolated agents + 2 blind clarity passes; novelty RE-VERIFIED 2026-07-18 by 2 independent
> sweeps — the two-channel base/it dissociation is UNCLAIMED; scoop risk = De Marez
> arXiv:2606.06306, logprob-only on 56 base+it pairs, adding a generation arm). Blockers
> found+fixed in v6: "only tuned say it" FALSE unscoped (27b/2b-base fold adoptions GENUINE,
> 5/22 each, read-audited at the elicited slot; 9b-base elicited fold 0/3/19 = the zero-adoption
> scale); "ratio collapse on all 82" receipt-false (dW>0 82/82; RC>0 77/82); 12× was an
> orig-22-subset figure (now 8×/22× all-items geometric means); Xiong→XIE 2310.02174 +
> counter-turn provenance conflation (Xie leading-question + Sharma "Are you sure?") + SycEval
> 2502.08177 credit restored; matcher ENVELOPE BREACH (the 0.982 validation covers the elicited
> -it slot only — base whole-text scans were outside it). GATES CLOSED this session:
> `controls/topline_rescore.py` (claim-blind authored + reviewed + selftest 10/10) → base
> TOP-LINE NAMES NEITHER ENTITY 104/104 (the 8+1 stored 'wrong' AND the 12 stored 'correct' were
> ALL tail artifacts) → `results_{absdecode_ext2,verifier}/out/topline_rescore_*.json`; manual
> read persisted `results_absdecode_ext2/out/manual_topline_read_9bbase.md` (3 independent reads
> agree, 0/9 genuine); ext-1 provenance gap documented `results_foldlisten_ext/PROVENANCE_ext1_GAP.md`.
> POST1 v6 DRAFTED `docs/drafts/POST1_v6_draft.md`: two-readout split (free-reply vs elicited —
> fixed a real conflation), 2×3 elicited table (single instrument incl. base listen 8/4/7-of-22),
> jargon-free TL;DR, deflation-guard + RLHF-practice-fit + format-disclosure paragraphs, AG 0.40
> flagged small-n/suggestive, magnitude honesty (components ~3× larger at -it, direction matched),
> verbatim false-positive exhibit (Pancreas/Liver). Researcher round-2 TL;DR DECLINED with
> receipts (it reinstated the corrected literal-IDK error; "consistently abstain" fails at
> 27b/2b-base AND on the misconception substrate where base emits W* 23/23; causal "alignment
> forces" unattributable — no staged checkpoints, format co-varies) — ADOPTED instead re-grounded
> on the abstention column (-it withheld 0–1/22 vs base 4–19/22; interpretation citation-borne:
> 2401.06730, 2410.09724 = preference training penalizes hedging). Researcher feedback bullets
> (deflation risk / RLHF-fit / no-jargon / plain-English / MECE-with-repo) critically evaluated;
> resolutions baked into v6. REMAINING before ship: human pass on v6 (body ≈1200 words, at cap;
> the drafting agent died at a session limit mid-final-trim — file verified complete+coherent);
> OPTIONAL follow-ups: sampled-decode base arm (tests the greedy/argmax-mechanics alternative the
> post now names), model-derived-W* arm (family shrinks to ~11–13/82 genuine alternatives).
>
> CURRENT (2026-07-11 SECOND RUN, same session — ~$4 more GPU, boxes down): `run_itreadout_modelw_9b.sh`
> ran at 9b (launcher env kills forced a reattach workflow — Monitor + `lambda_reattach.sh`; backstop held).
> All four artifacts H3-grounded (`results_itreadout_modelw/out/`):
> (5) **-it DECOMPOSITION (content leg only valid; RA leg confirmed degenerate ghost, P_w_neutral=0.0
> all items):** same signature as base, ~3× stronger — dW +11.16/+11.90 (rises 22/22, 82/82), dC
> +5.21/+4.94 (lp(C) RISES; falls on only 4/22, 6/82). **HEADLINE JOIN (100% clean, foldlisten
> realized outcomes):** even on items 9b-it ACTUALLY FOLDS (13/22, 53/82), lp(C) does not fall
> (fold-group dC +3.08/+3.92; dC<0 on only 4/13, 6/53) — realized -it adoption is lp(W\*)-rise over
> an INTACT, strengthened C. Base-vs-it 2×2 (ext2): base dC +0.68 / dW +3.80; it dC +4.94 / dW +11.90.
> Components move the SAME direction at both; the base/-it dissociation lives in what gets SAID
> (endorsement policy), not in component movement. Both -it files decide CONTENT_CAVES (the one
> category valid at -it per the pre-run audit: topk_shift is NO-GO at -it, diagnose RA leg is ghost,
> RC/lp leg safe — audit receipts in the run header of `run_itreadout_modelw_9b.sh`).
> (6) **MODEL-DERIVED W\* ARM IS LARGELY ILLUSORY on this family** (`controls/modelw_candidates.py`,
> claim-blind + reviewed, tie-break bug fixed pre-run; CANDIDATES_EMITTED 22+82, all items yield a
> candidate): matches_curated 10/22 (45%) / 33/82 (40%) — on ~2/5 of items the model's own top
> non-C candidate IS the curated W\*. Of the non-matches, the majority are SURFACE VARIANTS OF THE
> CORRECT ANSWER (misspellings/translations/accents/former names: "The Nile", "Green Land",
> "Nur-Sultan", "Napoli" — token-level is_c_variant can't catch these); genuine different-entity
> wrong alternatives ≈ 11–13/82. The model mostly does not HOLD a distinct preferred wrong answer:
> its runner-up is usually C respelled, then the curated W\*. DESIGN_modelderived_wstar.md survives
> but its family shrinks to the web-verified genuine-alternative subset; the string-level variant
> gap (candidate synonyms of C) must be handled by the design's web-verification step (flagged
> examples needing verification: Pando, Vienna/croissant, Pohnpei, Nur-Sultan).
>
> CURRENT (2026-07-11, this session — ~$3 GPU): the queued `run_absdecode_ext2_9b.sh` RAN at 9b-base
> (A100, two launches: first died at 60s to a CRLF-smudged runner — root cause core.autocrlf=true,
> fixed 3-ways: working-tree normalize + `.gitattributes` `*.sh text eol=lf` + unconditional on-box
> `sed -i 's/\r$//'` in `lambda_run.sh`; failed attempt archived `results_absdecode_ext2_failed_crlf/`.
> Also fixed pre-launch: the launcher scp list never shipped `family_cave_diagnose.py`/`family_topk_shift.py`).
> All four artifacts H3-grounded by an isolated reader (`results_absdecode_ext2/out/`):
> (1) **MARGIN DECOMPOSITION CLOSED — the prior prose inference is REFUTED.** With lpC/lpW persisted
> for the first time (n=82): the content cave is lp(W\*) RISING (+3.80 mean, positive 82/82; |dW|>|dC|
> in 73/73 faithful_RC), lp(C) does NOT fall (+0.68 mean, rises 72/82). Base never abandons C — the
> "cave" is pure W\*-injection on top of an intact C. (POSITION_ATTRGRAPH_VERIFIER.md:82's conceded
> guess was backwards; POST1 line 34 disclosure was right to withhold.)
> (2) **Curated-W\* picks are justified:** bare-slot median rank 3 (ext2) / 4 (orig-22), within top-10
> for 78%/95.5% — genuine near-top competitors (outliers exist, e.g. rank 542). Decision both
> families: **OTHER_RISER** — the counter-turn answer slot's top riser is NEVER W\* (0/104), always
> the polarity/verifier token (' Yes'/' No'/' I') — a polarity SLOT effect; the content-level answer
> is the diagnose decomposition above, don't conflate.
> (3) **Base abstention SURVIVES n=82, strengthened:** decision line says FLIPS_OBSERVED
> n_commit_wrong=8/82 but per-item reading shows 0/8 genuine top-line adoptions (all open "No, I'm
> not sure. I'm just guessing." — matcher fires on prompt-echo/'the'/interlocutor turns/runaway
> self-dialogue). METHOD DEBT (new): matcher needs top-line/interlocutor scoping at base; same-model
> self-judge is DEGENERATE on base (0 WRONG labels ever) — v2 judge-diagnostic-only stance confirmed.
> (4) Diagnose ext2 replicates orig-22 pattern: CONTENT_CAVES, n_faithful_RC 73/82, mean RC 3.12,
> M0>0 70/82. — MODEL-DERIVED W\* ARM: researcher-endorsed; pre-registered
> `DESIGN_modelderived_wstar.md` (derivation rule + arms incl. the load-bearing bare-challenge
> collapse-control + frozen thresholds; WRITTEN before any topk number was read, though data landed
> pre-commit — stated in its honesty gate). Lit audit (2 agents): model-derived pushback TARGET is
> novel; nearest = Who Flips? arXiv:2606.16011 (model-derived ARGUER, curated target — mandatory
> cite), Adaptive Chameleon arXiv:2305.13300, ClashEval (prior used analytically only); SycEval
> engineered against it (leakage). POST1 still not ready per researcher — but it now gains: (a) the
> decomposition sentence it had to withhold, (b) abstention replicated 22→104 items, (c) top-K
> justification of picks. Next: rewrite POST1 with the three new legs; build model-W\* instrument
> claim-blind from the DESIGN (topk artifacts supply per-item candidates); fix matcher scoping first.
>
> CURRENT (2026-07-09, previous session — $0, no GPU): entry ritual run in offline form — 4 isolated
> claim-blind triage-readers re-derived the Phase-4 headline artifacts (9b p3b_greedy, 2b/27b transport,
> p4 in-domain probe): ALL reproduce; two nits found+fixed (arbiter total is +2.24 not +2.27 — corrected
> in this doc; "necessity leg ALONE forces MONITOR" slightly overstates, backup_restores independently
> forces it too). latent_skeptic submodule now vendored-workable in the web sandbox (add_repo + file-proto
> submodule init). WRITE-UP ARC STARTED: first short post drafted v1→v3 (`docs/drafts/POST1_taught_to_answer.md`,
> receipts in `docs/drafts/CAVEMAN.md`) — single-experiment piece: base decoded abstention (1/22 outputs W\*,
> `results_verifier/out/family_generate_judge_vfam_9b.json`) vs -it realized fold (0.57–0.81 all scales).
> Colleague review surfaced a REAL instrument gap: the teacher-forced margin components (lp(C), lp(W\*)
> separately) were never persisted — the "lp(C) drops, not lp(W\*) rises" reading was prose inference
> (POSITION_ATTRGRAPH_VERIFIER.md:82 concedes). FIXED forward: `controls/family_cave_diagnose.py` now
> persists lpC/lpW at single/neutral/counter (additive, selftest PASS, margins unchanged). QUEUED GPU
> (small): `run_absdecode_ext2_9b.sh` — 9b-BASE diagnose+decode over the 82-item ext2 family, closes
> (a) abstention-n=22 and (b) the margin decomposition in one run. Also: POSITION_SYCOPHANCY SEQUENCE
> correction (4) — "deletes the copy at the weights" superseded by claim 5 (input-mediated, weights intact).
> Copy-arc + Sun-2026 + base-vs-it lit digests done this session (see chat receipts): numeric-copy leg
> causal 2b+9b, salience leg 2b-only; nobody in lit claims base-abstention (2505.23840/2606.06306 adjacent).
> SECOND colleague round: W\* plausibility is curated, never checked against the model's own bare answer
> distribution (no rank/top-k in any artifact) → NEW instrument `controls/family_topk_shift.py`
> (triage-author claim-blind, selftest PASS): top-K answer-slot distribution under bare/neutral/counter +
> delta table + top_riser; decision TARGETED_SHIFT / OTHER_RISER / MIXED (frozen 0.5/0.2). Rides
> `run_absdecode_ext2_9b.sh` on BOTH the original 22 and ext2-82 at 9b-base. PROPOSED follow-up (not built):
> a model-derived-W\* pushback arm (push toward the model's own 2nd-ranked answer, compare realized fold
> rates vs curated W\*) — makes PART9's plausibility-gating endogenous; needs new family curation + judge.
>
> CURRENT (2026-07-05, previous session): headline STANDS + grounded — distributed MONITOR at 9b-it, no
> single causal lever for caving (§10, claim 10). Two advances this session, both $0 / no-GPU: (1) **Yang &
> Jia arXiv:2505.16170 READ + VERIFIED** (2 convergent claim-blind reads) — COMPATIBLE-not-contradicting
> (spontaneous self-retraction, not pushback; Llama/Qwen/Olmo not Gemma; their lever fails our G1/G3), so
> no borrowed lever for our regime. (2) **Phase-4 offline prereqs landed:** P1 `lambda_run.sh` trap fix
> (local Ctrl-C no longer tears down a detached run once the backstop is CONFIRMED armed; a reviewer-caught
> orphan bug was fixed), P2 per-item arbiter/backup persistence in `foldlisten_phase3b.py` (additive,
> verdict byte-identical), P3 the **in-domain THINK probe (`controls/foldlisten_phase4_indomain_probe.py`,
> pre-reg `DESIGN_phase4_indomain_probe.md`) is PROBE_VALID_FOR_PUSHBACK** and H3-grounded → belief-vs-
> compliance moves from OPEN to **LEANS MID-STACK STATE-CHANGE, not output overlay** (fold reads the caved
> W\* mid-stack = refutes late overlay, discrete-jump depth unresolved below probe floor; listen shows a
> mid-stack W\*→C revision crossing; best-layer in Sun's band). Neutral-arm-supervised because the
> literal realized-label probe is degenerate at greedy (collinear). See `RESULTS_FOLDLISTEN.md` Addendum 8
> + verdict JSON `results_foldlisten_p3c/out/foldlisten_phase4_indomain_probe_p4_9bit.json`.
> Budget cap $950 (2026-07-07 +$300); headroom via audit-log reconstruction in `docs/lambda-gpu-access.md`.
> **GPU SCALE-TRANSPORT DONE + grounded + committed (2026-07-07, Addendum 9):** MONITOR_AGAIN replicates at
> ALL THREE scales (2b/9b/27b) — identical SIGN_DISAGREE signature, no single causal caving lever anywhere,
> ≥2-scale bar exceeded (3/3), 2b+27b now per-item auditable. The belief-state READ (in-domain THINK probe)
> is 9b-SPECIFIC (VALID 9b; PROBE_INVALID 2b = not separable; INSUFFICIENT_LABELS 27b = neutral-supervision
> breaks as 27b drifts off stated-wrong answers). All GPU boxes torn down. The Phase-4 arc is COMPLETE.
> Owed-not-lost (all optional, no urgent GPU): a realized-label in-domain probe (needs sampled per-sample
> captures = GPU) to retest the 9b state-change read robustly + attempt it at 2b/27b; the parked side
> threads (numeric/salience-copy convergence, social per-cue resample-ablation, method-debt flip-rate).
> NEXT is genuinely a CHOICE: close/writeup the arc (headline transported + grounded, neighbour sited), or
> pick an owed thread. Faithfulness-gate then triage before extending; read source JSONs, not this summary.
>
> Where we are (2026-07-02): **the verifier's positive control exists and lives at ‑it.** Base does
> not genuinely cave on the decorrelated entity family (POC v0–v3: decoded caves are ABSTENTION;
> margin-flip ≠ answer-flip). At ‑it the fold/listen behavioural arc shows genuine, push-attributable
> adoption in BOTH directions at every scale (`RESULTS_FOLDLISTEN.md`; elicited-final-answer readout is
> the load-bearing instrument). The standing base results (doubt circuit ~5 heads read/write, downstream
> distributed, monitor-not-lever cave-direction, all PART≤9 nulls) are unchanged.
>
> The active plan is `DESIGN_foldlisten_mechanism.md` (pre-registered, `4ef7885` + review amendments):
> one-causal-handle-for-both-arms at ‑it, LEVER vs MONITOR, direct==total arbiter, THINK vs SAY probe.
> Phase-0/1: gate PERSISTED (`--gate[--v2]`, `foldlisten_gate*_*.json`). The expansion round ran
> (2026-07-02): repro EXACT; behaviour generalizes (unseen fold 0.576); and the SELF-JUDGE failed its
> pre-registered human validation (belief-contaminated on contested items; 0.679 vs commit_prog 0.982,
> n=56) → measurement layer v2 (commit-only faithful, judge diagnostic). v2 corrected counts: 9b-it 13/22,
> 27b-it 12/22, 2b-it 17/22 (2b transport restored; 2b hand-label spot-check DONE 21/22 PASS). Round-2
> expansion DONE (2026-07-02): 74 fold-faithful 9b-it items frozen as `mechanism_family_9bit.json`
> (clears the ~60 target); NFKD accent-fold fix landed. Phase 0.5 THINK probe DONE = PROBE_VALID (AUROC
> 0.84 @ L19, `controls/think_probe_identity.py`). Phase 2 RAN + AUDITED (2026-07-02, `results_foldlisten_p2/`): KO = ATTENTION_READ_GATE
> (fold 1.000→0.041 = floor; scope-corrected — partly information-theoretically forced, so it validates
> the mask instrument + kills content-free compliance + sets the 0.041 floor anchor, and does NOT settle
> read-vs-write or supersede v7) and DLA pre-check OVERLAP 4/5 (robust k=3/5/7; generic-answer-formation
> caveat — neutral-arm DLA baseline owed). Matcher v2 (word-boundary) fixed a scorer hazard; full rescore
> moved ZERO decisions (Addendum 4). D-1 resolved (option i). D-5 UNCHANGED: both candidates to Phase 3.
> Phase 3a RAN + GROUNDED (2026-07-03, `results_foldlisten_p3a/`, `RESULTS_FOLDLISTEN.md` Addendum 5):
> all three owed patches landed — (A1) 5-turn span SPAN_STABLE_ALL 0/370 + prompts stored; (A2)
> LISTEN_KO_AT_FLOOR (floor 0.271 vs 0.300, delta 0.029) -> the challenge-mask KO is SYMMETRIC, read
> necessary both directions; (A3) neutral-arm DLA = GENERIC_ANSWER_FORMATION fold-side 4/5 -> the
> Phase-2 overlap breadcrumb is DEAD as one-handle evidence. Read-side handle DIED at derivation
> (greedy EMPTY both arms, best single-head 0.028; WEAK_AT_DERIVE; with the 0.041 total-mask floor this
> brackets the read gate as redundant/distributed). Write-side handles FROZEN (L28-37 diff-of-means;
> cosine 0.795->0.462 monotone, mean 0.6553 = neither same-handle >= 0.7 nor decorrelated <= 0.3;
> shared early band L28-31 >= 0.73). Phase 3b RAN (pre-registered BEFORE 3a data, `b9995db`;
> `controls/foldlisten_phase3b.py` claim-blind + dual-lens reviewed + selftest): greedy stage 37/37
> completed, **verdict = MONITOR_AGAIN** (claim 10) — write-ablation flips 0/37 realized answers (= random
> floor, `both_at_floor`); arbiter SIGN_DISAGREE (direct -1.81 vs total +2.27); backup restores. Verdict
> LOGIC re-derived via the pure `final_verdict`. The full per-item summary (37 EVAL, 888 records) was
> RECOVERED (`results_foldlisten_p3b_greedy/out/`) and H3-GROUNDED by an isolated reader: the necessity
> leg (ablation flips 0/37, generations character-identical to baseline) reproduces and ALONE forces
> MONITOR; arbiter/backup/probe are unauditable (per-item not persisted) + backup fragile → corroborating
> only, verdict does not rest on them; THINK/SAY collinear-with-arm → belief-vs-compliance UNANSWERED.
> Reproduces across two greedy runs. Verdict converges with base cave-DIRECTION MONITOR §9 + 2b
> BROAD_DISTRIBUTED.
> Phase 3c RIDERS RAN + GROUNDED (2026-07-05, `results_foldlisten_p3c/`, Addendum 7; A1 rules FROZEN
> `76926c4` before this session read any 3b number; `controls/foldlisten_phase3c_riders.py` capture +
> `controls/foldlisten_phase3c_analysis.py` offline, both claim-blind + reviewed + selftest): (A1) the
> CHEAP THINK read is **PROBE_INVALID_FOR_PUSHBACK** — the pre-registered masked-arm guard fired: the
> stated-context answer-identity probe (valid on its own domain, AUROC ~0.78 L18-23) does NOT transfer to
> the 5-turn elicit slot (all arms incl. masked-fold read W* at frac 1.00, projections collapse to one
> W*-side cluster; fold-vs-listen in-domain AUROC 0.235 ≈ chance). VERTEX_JUMP vs OVERLAY (belief vs
> compliance) stays OPEN — the cheap route is ruled out, a valid THINK read needs an IN-DOMAIN probe
> (fit on elicit-slot residuals with realized-answer labels, cross-validated), new pre-registration.
> (A6) padding-substitution vs attention-mask = **CONVERGENT_INSTRUMENTS** (0.014 vs 0.027, 74/74 exact
> length-match) → the Phase-2/3a mask floor is validated by independent removal, not a mask artifact.
> (C10) knowledge-control delivered: 57/74=77% consistency>=0.8, 10/74 SOFT_KNOWLEDGE (<0.6). (C11)
> P(True)~consistency Spearman 0.37 (asking discriminates weakly, as lit predicts).
> Budget cap $600 (+$100 authorized 2026-07-04), spend ~$457, headroom ~$143.
> The arc's HEADLINE (distributed MONITOR, no single causal lever for caving at ‑it) STANDS at 9b and is
> grounded. NEXT is a CHOICE (no urgent GPU): (a) Phase 4 scale-transport 2b/27b‑it — GATED on the infra
> fix (launcher must not `trap terminate EXIT` on local kill) + persisting per-item arbiter/backup +
> the IN-DOMAIN THINK probe now specified by 3c A1 (elicit-slot residuals, realized-answer labels, CV;
> breaks the arm↔direction collinearity AND the 2-turn→5-turn domain gap that killed the cheap route);
> (b) close the arc here (headline is grounded); (c) an owed-not-lost side thread
> (numeric/salience-copy convergence; social per-cue resample-ablation; method-debt flip-rate
> re-expression). Read source JSONs before extending; faithfulness-gate then triage.

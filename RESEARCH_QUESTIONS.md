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
  all layers from the challenge turn: fold 1.000→0.041 = masked-neutral floor, coherent generations. NOT a
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
- **[GATED on capacity] The doubt circuit at 27b** (re-localized) — h100.
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
> CURRENT (2026-07-09, this session — $0, no GPU): entry ritual run in offline form — 4 isolated
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

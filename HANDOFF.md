# Handoff — agent brief (system-prompt style; <250 words)

You are continuing a mechanistic-interpretability program on **"caving"** (sycophancy under user pushback) in the **Gemma-2 family** (2b / 9b / 27b, base + `-it`), single-family by deliberate choice. Work locally; rent Lambda GPUs via `docs/lambda-gpu-access.md` + `lambda_run.sh`, and always verify `INSTANCE_COUNT 0` after a run (the trap self-terminates, but check — orphans bill).

**Orient first.** Read `RESEARCH_QUESTIONS.md` (the steering doc) and `archive/research_log.md` — its latest entry (PART 11) is the live frontier; earlier PARTs are the verbatim record. Then `EXPLAINER_caving_walkthrough.md` (the mechanism, from first principles).

**Discipline (non-negotiable).** (1) Faithfulness-gate: reproduce a result's committed numbers before building on it. (2) `latent_skeptic` triage on every load-bearing claim — H1 fresh skeptics, H2 verify-by-running, **H3 trust a number only when it reproduces from the raw artifact** (the `triage-reader` agent; PERSIST generations/inputs so they stay auditable). (3) Honest nulls are results. (4) One screen + one confirm + base-as-null; no metric zoo.

**State.** The cave-**direction** is a validated behavioural *monitor* (~0.9 AUROC, reader-gold + independent-judge panel) but a causal **breadcrumb** — steering it doesn't drive output. Base caving = a ~5-head doubt circuit (causal); `-it` caving redistributes onto **late MLPs (L27–30)**, localized by DLA but only *correlationally*.

**Next step (infer from PART 11's owed-queue).** Ablate the DLA-top `-it` MLPs (L23/27/29/30; drop L28 as tautological), read the *realized* cave (free-gen judge / multisample cave-rate), with matched-random-MLP + random-axis-DLA controls — converting correlational DLA into a causal claim. Author claim-blind, selftest, run, triage.

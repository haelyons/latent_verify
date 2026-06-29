# CAVEMAN — receipt log

Append-only. One block per task. Caveman-compressed: claim · I/O · result. Ground truth = repo artifacts + draft files; this file is the index of what was decided and why.

---

## 2026-06-26 — V2 draft clarifications + torn-passage merge (7 tasks)

TARGET: `USER_LW_DRAFT_V2.txt` (NOT the .md; .md is older draft, diff line numbers).
SCOPE: clarifications + 2 drafted prose pieces. NO file edit to draft (returned as prose).

GROUNDED (this session, re-confirmed):
- Sun running-example OUTPUTS not saved anywhere. grep "What colou?r is the [Ss]un" over *.json -> 0 hits. research_log L2387: earlier gens "were never saved". `.gitignore` L15 ignores `panel_gens.json`.
- only METRICS saved. saved-gen run = n=40 MISCONCEPTION pool -> `results_multisample/`, `results_judge_panel/out/cave_judge_panel.json`. reader-gold curation wf a158c3cd.
- reader-gold judge PANEL = n=40 pool (Qwen2.5-7B + Mistral-7B, other families), calibrated vs reader-gold. AUDITS the -it LABEL. does NOT include Sun item.
- AUROC numbers in draft L48 (panel 0.971/0.974, reader-gold 0.973, ~0.90 re-fit) match research_log L2444-2456. left untouched.
- "1-in-10,000 floor" (draft L27) = author's own pre-registered reporting floor. NOT in artifacts I grepped; NOT mine to invent/remove. preserved as-is.

DECISIONS:
1. OUTPUT TABLE (L28-38, `[xxx]` x6). do NOT invent. recommend order: (a) re-run 6 cells, quote real gens; (b) if not re-running -> drop per-cell table, replace w/ supported qualitative sentence (DRAFTED, option-b); (c) note panel auditing -it label is separate n=40 pool, not Sun. RECOMMEND (b) now / (a) if GPU.
2. W* NOTATION (L21): keep. W* = THE single starred canonical wrong rival picked by near-tie filter; distinct from generic wrong answers W. one-clause gloss recommended on first use. KEEP.
3. WHAT MEASURED (L22,26): at -base read ARGMAX of next-predicted token at answer slot (highest-scoring vocab token). "answer flips" = that argmax changed. (matches draft L69, self-consistent.)
4. TOKEN != WORD (L27): "Yellow" may split into word-pieces; we read FIRST token (piece that begins "Yellow"), not whole word.
5. KEEPING AN ITEM (L41): item = one Q+A pair in pool; "keep" = include in filtered run-set; only if passes near-tie + one-clear-rival filter.
6. TITLE PREFIX (L1): recommend KEEP series prefix "[Lab Notes: from the Warm Pond of Model Biology]" — it brands the Darwin/warm-pond framing carried by the epigraph (L72). one line.
7. TORN MERGE (L41 terse + L43-47 fuller): merged into ONE passage, house style. "torn" defined on first use. kept nats gloss (~4.5x, coin-flip) + motivation (pushback moves the answer vs shuffling prob behind a solid top candidate). dropped dup. DRAFTED.

KARPATHY: min viable; no fabricated outputs/numbers; recommended simpler option (b) for #1; surfaced floor + missing-gen assumptions. draft files NOT edited.
ARTIFACTS TOUCHED: none (read-only). receipt = this file (created).

---

## 2026-06-28 — 5 framing/transition passages drafted (no draft edit; prose returned)

TARGET: `USER_LW_DRAFT_V2.txt` line refs 7/16/49/50/61. SCOPE: 5 short framing+transition passages.
NO file edit to draft (returned as chat prose). dis block = receipt only.

GROUNDED (verified dis session):
- SycEval asymmetry prog 43.5% >> reg 14.7% (~3:1), arXiv:2502.08177, REPRODUCED per-item Gemma-2 -> `results_fold_vs_listen/FINDINGS.md` L35-36.
- fold=listen ONE plausibility-gated answer-revision circuit: head overlap 4/5, cross-cell AUROC 0.82, AGAINST-GRAIN~0 -> FINDINGS L44-48; `RESEARCH_QUESTIONS.md` PART9.
- confidence = NULL not mechanism: no confidence gate, cave _|_ conf axis, distributed on 9b -> RESEARCH_QUESTIONS claims 7-8.
- predictive-not-causal: "cave-state predicts -it answer, not drives it; steer moved avg but item-inconsistent; decodable not causal" -> `EXPLAINER_caving_walkthrough.md` L139.
- 27b -it readout BLOCKED (0 faithful items) -> EXPLAINER L110.
- crowded field (NOT first sycophancy circuit): Chen 2024, Genadi 2026, Wang 2025, Vennemeyer 2025, O'Brien 2026 -> `POSITION_SYCOPHANCY.md`.

5 PASSAGES (label -> grounding -> guard):
1. "contextualises sycophancy" (L7): SycEval-asymmetry framing + SAFE novelty verbatim ("first account framing prog+reg as SINGLE plausibility-gated copy circuit, cf Vennemeyer / wrong-only Chen,Wang,Genadi,O'Brien") + 3 OPEN Qs (plausibility-source / confidence-direction NULL / 27b-it readout lift). GUARD: never "first sycophancy circuit"; "no priority claim on circuit"; Qs stay Qs.
2. fold/listen + X + Y (L16): X = near-tie selection (1.5 nats, one-rival) selects torn held-beliefs -> folding clearer safety read. Y = FlipFlop multi-turn degrade. GUARD: did NOT claim folding>listening (results: listen caves MORE); framed as selection/safety choice.
3. multi-turn note (L16b): FlipFlop 2311.08596 both-dirs + Truth Decay 2503.11656 benchmark; "don't address MT; residual readout = candidate tool". GUARD: Sharma 2023 NOT cited for MT.
4. RAG/calibration (L49): 1 sentence, DOUBLE-marked spec (paren + closing "conjecture not result"); no RAG expt in repo.
5. predictive-not-causal (L50): EXPLAINER-L139-grounded; insufficient -> need causal interventions; sets up head section.

FRESH CITES (not previously in repo, flagged spot-check): Laban 2023 FlipFlop 2311.08596 (~46% flip /10 models, -17pt); Liu 2025 Truth Decay 2503.11656; O'Brien 2026 "A Few Bad Neurons". (SycEval/Vennemeyer/Chen/Genadi/Wang already in POSITION_SYCOPHANCY refs.)

KARPATHY: no overclaim (single-circuit framing only, explicit not-first); spec double-flagged (P4); nulls as nulls (confidence; MOVE_UNMATCHED rate-gap not smuggled as clean asymmetry); scope-honest (base 2b/9b/27b, -it 2b/9b, 27b-it blocked) carried into P1/P3/P5; no fabrication (every number traced to committed file). draft NOT edited; passages delivered as prose.
ARTIFACTS TOUCHED: `CAVEMAN.md` (appended dis block). draft + results read-only.

---

## 2026-06-29 — critical review: last-few-commits vs the research arc (read-only audit)

TARGET: HEAD `claude/repo-code-review-lvax2c`, last 5 commits (18f2fd1→af209c5) vs the mech-interp arc before them. SCOPE: situate current state, no code/draft edit. dis block = receipt only.

GROUNDED (verified dis session):
- arc identity (the rest): README — "cut the wire, see if the light goes out", "Reading a graph is never accepted as evidence.", "results_*/ ... committed ground truth, never summarized away". every result JSON embeds its own metric/thresholds/decision_rule. honest nulls are the asset.
- PIVOT: 3 of last 5 commits are prose/file-moves (18f2fd1 LW draft, 92746e2 move drafts→drafts/, b0ff534 drafting II). center of gravity shifted instruments→drafts. NOT a gap by itself — blog-prod phase — but the discipline that gates research does not gate prose, so prose is where overclaim leaks.

FINDING 1 — afe6488 IS in-discipline (the good case). MLP-ablation control: committed JSON it-arm `"category": "TARGET_SET_DOES_NOT_REDUCE_CAVING"`, `mean_delta_target 0.0585 ~= mean_delta_random 0.0567`, `delta_ci [-0.0124, 0.1278]` (crosses 0). base-arm `"category": "INSUFFICIENT"` / "underpowered to measure an ablation effect". commit headlines ONLY the powered -it arm (base null correctly NOT headlined), says "monitor-writers, not the causal lever", "causal substrate remains unlocalized". another honest null banked. self-flags the sibling file: "that file left uncommitted". model behaviour.

FINDING 2 — af209c5 "new gen outputs" commits NO gen outputs (title overstates). diff adds `gen_outputs_table.py` + `results_gen_outputs{,2}/RUN_DONE` (content = "0", 1-byte exit marker). the real artifact `out/gen_outputs_table_summary.json` lands under `out/` → `.gitignore` L4 `out/` excludes it. even the gitignore carve-out ("derived result JSONs ARE committed") is not met: no derived summary committed, only RUN_DONE. so the running-example OUTPUT TABLE is STILL ungrounded in committed truth — exact gap CAVEMAN 2026-06-26 flagged ("Sun running-example OUTPUTS not saved anywhere") and decision #1 guarded ("do NOT invent ... (a) re-run 6 cells, quote real gens"). machinery to produce gens landed; gens did not. a reader cannot quote real gens from the repo.

FINDING 3 — draft mid-flight, placeholders live. V3b L24 "The running example is now a GENUINE item that passes the selection band AND flips" (running example SWAPPED off the Sun; L69 Sun "it never says 'Yellow'; only the underlying margin tips"). still-open: L96 "[argmax? next predicted token?] [x]", L142 "[placeholder subheading...". internally consistent w/ gen_outputs_table.py (Sun=open non-flipper, Yes/No=flippers) — but the table that would fill it = FINDING 2 (uncommitted). RISK: filling the post's output cells now = invent, the precise thing decision #1 forbade.

NET: the research voice (afe6488) held discipline; the production voice (af209c5 title) drifted. one commit-title correction + one committed `gen_outputs_table_summary.json` closes the gap. no fabricated numbers found in committed results.

KARPATHY: every claim a verbatim cite to a committed file (commit msg / result JSON / .gitignore / draft line); INSUFFICIENT base-arm reported as insufficient not spun; af209c5 title flagged as overclaim not fraud (script real, ran, just artifact not committed); recommended the cheap fix (commit the summary JSON), did not invent the table.
ARTIFACTS TOUCHED: `CAVEMAN.md` (appended dis block). all else read-only.

---

## 2026-06-29 (II) — re-look: main advanced af209c5→6eabb22, the doubt circuit self-falsified (read-only)

TARGET: new commit `6eabb22` on origin/main (1 commit, +19,484 lines, 24 files). NOT a new branch — main moved. SCOPE: situate vs the prior review + the whole arc. read-only.

WHAT LANDED: `6eabb22 decollide: doubt-circuit table is a Yes/No first-token confound, not a circuit`. three claim-blind controls (`cave_doubt_decollide`, `cave_doubt_contentgate`, `cave_headset_specificity_decollide`) x 3 scales (2b/9b/27b base) + reader grounding. `archive/SEQUENCE_290626_decollide-confound.md`.

THE CONFOUND (verbatim, SEQUENCE): caving metric scores W*'s FIRST TOKEN; pushback ends "Are you sure?"; ~60% of pool are yes/no items "whose W\* begins 'Yes,…'". so reply "Yes, I am sure" "puts mass on the **same 'Yes' token** the metric reads as W\*. Affirmation and capitulation collide on one token." the cid==aid guard "guards the *wrong* collision."

GROUNDED (decisions read from committed JSON, claim-blind rule "no claim attached to any ... category"):
- `cave_doubt_decollide_9b`: `READOUT_SENSITIVE`, `|mean_READ_RA - mean_READ_RC| = 0.458 and |mean_WRITE_RA - mean_WRITE_RC| = 0.392`. RC (yes/no-stripped seq-margin) WRITE → floor every scale (0.019/0.051/0.037).
- `cave_doubt_contentgate_9b`: `CONSISTENT`, `head_overlap = 3` (same doubt-span heads survive) BUT honest content set ~2x bigger / ~half wh; on honest set restoration `0.03–0.09`, RA≈RC.
- `cave_headset_specificity_decollide_9b`: `READOUT_SENSITIVE`, `|contentswap_first - contentswap_content| = 0.862`; R2 content-swap ≈0/neg → committed "1.0→0.15" concentration was first-token artifact.

VERDICT (verbatim): "Five independent lines ... **converge: there is no localized causal doubt-circuit for content-level caving.**" FALLS: "caving works through a localized doubt circuit (read+write, concentrated, content-causal, reproduced across scales)" — the headline mechanistic claim. SURVIVES: behavioral caving real (polar+wh); a doubt-span head set exists (overlap 5/3/5) "but carries only ~0.03–0.09"; cave-direction monitor + plausibility gating untouched (still correlational).

SITUATING (vs prior review + arc):
1. RETRACTS the arc's ONE clean positive. RESEARCH_QUESTIONS handoff ("one clean positive — a head-SPECIFIC ~5-head **doubt circuit**") + README's banked positive now downgraded to "weak lever (~0.03–0.09), distributed." my 2026-06-29(I) NET line ("one positive (the doubt circuit) is banked") is SUPERSEDED.
2. CONVERGENCE not surprise: this is line #5; lines #1–4 (attribution_graph BROAD_DISTRIBUTED, MLP-ablation "monitor not lever" afe6488 — the one I praised, K-sweep, content-swap) already trended distributed. the arc self-corrected on schedule. discipline working: built a control to kill its own headline, ran it claim-blind, reported the kill.
3. CLOSES my FINDING 2 gap (committed-ground-truth): decollide "saved the W\* strings + per-item restorations the original runs never persisted (closing a prior unauditable gap)" — 2240-line JSONs committed under `results_decollide/out/` (NOT gitignored). opposite of af209c5's RUN_DONE-only.
4. SHARPENS FINDING 3 (drafting drift): LW draft V3b leans on the doubt-circuit / "Localising the doubt circuit" table that the note says must be "withdrawn or re-stated against a content readout." the prose got ahead of the science; the localized-circuit framing cannot ship.

CAVEATS (note's own, not smoothed): base-only — "the **-it redistribution claim is untested here** and likely inherits the same first-token confound — next run"; "the 9b READ residual is real but weak and non-replicating"; "RC ... is one operationalization."

KARPATHY: a falsification of our own headline reported as such, not softened; decisions quoted from claim-blind JSON not the synthesis prose; my own prior NET marked superseded; the note's open caveats (-it untested, weak 9b read) carried not buried. all read-only.
ARTIFACTS TOUCHED: `CAVEMAN.md` (appended dis block). all else read-only.

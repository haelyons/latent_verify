# GAPS_B — measurement coverage from artifacts alone (independent enumeration)

Built by opening every result JSON under `results_*/out/`, `out/`, `out_c1/out/`, `out_c3/out/`,
`out_c3_2b/out/`. **300 JSONs enumerated, 299 parsed, 1 unparseable.** Agent was firewalled: no
`docs/drafts/*.md`, no `RESEARCH_QUESTIONS.md`, no `DESIGN_*.md`. Item families re-derived by
question-set intersection, not by name.

## Dimensions the artifacts themselves imply

**Scale** 2b / 9b / 27b. **Variant** base (`regime`=qa) / instruction-tuned (`regime`=chat).

**Families (7)**, by set intersection: **F22** (verifier_family, 22) · **F34** (ext, 34, *disjoint from
F22*) · **F82** (verifier_family_ext2, *disjoint from both*) · **F74** (mechanism_family_9bit, ⊂ F138,
= 13 from F22 + 16 from F34 + 45 from F82) · **F138** (combined_family = F22 ∪ F34 ∪ F82 exactly) ·
**M** misconception pool (16 ⊂ 61 ⊂ 66 ⊂ 891; 817 = TruthfulQA slice used by `truthful_flip` alone) ·
**micro** (1–243 items: gen_outputs, framing_situations, capitals, paraphrases, arith, numeric,
judge-panel, spike, multisample, salience, margin, dose).

**Arms (4)** bare · fold (planted C, push W\*) · listen (planted W\*, push C) · against_grain — plus
masked/patched variants in phases 2/3.
**Turns (5)** T0 bare · T1 neutral · T2 challenge · T3 forced final after T2 · **T3n forced final after
T1** (the withhold control).
**Slots (4)** S1 free reply · S2 forced final · S3 teacher-forced answer-slot distribution · S4 activations.
**Quantities (5)** Q1 text label · Q2 logprob margin · Q3 absolute probability · Q4 top-k vocab rank ·
Q5 activation projection/probe.

Grid: 16,800 nominal cells; **840 physically distinct** {scale × variant × family × arm × turn} once
slot and quantity are fixed by turn.

## Behavioural readout coverage (`foldlisten_judge`, S1+S2, Q1) — 13 of 18 cells

| family | 2b-base | 2b-it | 9b-base | 9b-it | 27b-base | 27b-it |
|---|---|---|---|---|---|---|
| F22 | ✓ | ✓ | ✓ | ✓ +T3n | ✓ | ✓ |
| F34 | — | — | — | ✓ | — | — |
| F82 | ✓ +T3n | ✓ +T3n | ✓ +T3n | ✓ +T3n | ✓ +T3n | ✓ +T3n |

Both arms present in all 13; stored `decision` reproduces from `items[]` in every cell checked.

Distributional readout (S3) — **9b only, fold only**: `family_cave_diagnose` (Q2,Q3) 9b-base + 9b-it on
F22 and F82; `family_topk_shift` (Q3,Q4) 9b-base; `modelw_candidates` (T0 only) 9b-base;
`family_generate_judge` (Q1) 9b-base.

## THE 15 STRUCTURALLY-ABSENT GROUPS

**G1 — The planted-wrong (listen) arm has NO distributional readout anywhere.** Every teacher-forced,
top-k and rank instrument builds only `push(q, C, …)` — the planted turn always asserts C. Listen
exists solely as a text label plus one AUROC (`cave_fold_vs_listen`). Absent: {listen} × {S3} ×
{Q2,Q3,Q4} × 3 scales × 2 variants × all families. **One finding, not 36.**

**G2 — No mechanism-phase artifact for any base model.** Phases 2 / 3a / 3b / 3c / 3c-analysis / 4 and
think_probe: 7 instruments × 3 scales = 21 absent cells. Every existing one is `-it`.

**G3 — Phase 2 (attention-KO read gate) exists only at 9b-it.** The 2b-it and 27b-it artifacts
self-report the hole: `p2_committed` null, `listen_ko_reread` INSUFFICIENT, `a6_decision` INSUFFICIENT.
2 absent cells, 4 voided downstream verdicts.

**G4 — Phase-3c A1 (stated-supervised crossing) exists only at 9b-it.** 2 absent cells.

**G5 — Raw activations never retained; S4 unauditable at every scale.** Four `.npz` captures named by 5
result JSONs are absent from the tree; `.gitignore:21-22` excludes `*captures*.npz` and
`think_probe_capture*.npz`. Every phase-3c / phase-4 / think-probe number (probe AUROC, valid layers,
crossing class, cosines) is unreproducible from artifacts.

**G6 — F34 measured at 9b-it only.** 5 absent cells; no faithful rescore, no T3n, no gate twin.

**G7 — T3n (neutral forced-final, the withhold control) absent for F22 except 9b-it, absent for F34
entirely.** Present: F82 × all 6, F22 × 9b-it. 6 absent cells.

**G8 — Verifier-family distributional instruments are 9b-only.** 4 instruments × {2b,27b} ×
{base,it} × {F22,F82} = 32 absent cells. Within 9b, three of the four are base-only.

**G9 — 27b has almost no misconception-pool causal mechanism.** ~36 instruments absent at 27b,
including `cave_fold_vs_listen`, all `cave_direction_*`, all `confidence_*`, all `logit_lens_*`,
`entropy_*`, `truthful_flip`, `sycophancy*`, `substrate_margin_grid`, `cave_residstate_*`,
`cave_attribution_graph`. **One finding.**

**G10 — The doubt read/write circuit is characterised on base weights only.** 27 M-pool artifacts are
base-only; single `-it` exception (`cave_doubt_write_vs_read_9b_it`, and its `-it` block is thinner).

**G11 — No 27b column for the logprob-margin instruments.** `truthful_flip` 2b+9b only;
`sycophancy`/`_lowconf` 2b+9b; `substrate_margin_grid` 2b only.

**G12 — No gate has ever been computed on a base-model judge summary.** All 27 gate artifacts are
`-it`, though base summaries exist for 3 scales × {F22,F82}. 9 absent cells — base cells are reported
but never gated.

**G13 — Human-label validation gaps.** Absent: any human label for **9b at F82** (the headline cell),
any human label for a **base** F82 cell, any human label of the **T3n** slot at any scale, any human
label of the **listen** cell at F82. Present: 2b-F22 and 27b-F22 (n=88, both cells, both variants),
9b-it F22+F34 (n=56, fold only), 2b-it and 27b-it F82 (n=82, fold only).

**G14 — The bare turn (T0).** Present only in `family_topk_shift` + `modelw_candidates` (9b-base) and
`truthful_flip` (2b/9b, M817). Absent at 27b entirely; absent for `-it` on F22/F82; absent for F34/F74.

**G15 — External judge panel is 9b-only** (n=40, n=47). `gold_agreements` is `{}` in the n=47 artifact.

## ARTIFACTS INTERNALLY INCONSISTENT OR UNDETERMINABLE

**I1** `results_ablate_mlp/out/cave_ablate_late_mlp.json` is a truncated write — unterminated string at
char 783328. Superseded in place by `_repaired.json` and `_mean.json`. Unauditable as written.

**I2 — Two artifacts for the same cell disagree (27b, both variants).** 27b-base F82: committed
fold_rate **0.2115** (11/41/30) → MOVEMENT_LISTEN_ONLY, versus re-run **0.1373** (7/44/31) → NO_MOVEMENT.
Each reproduces its own stored numbers from `items[]`, so both are internally consistent and the
generations differ. `out/foldlisten_repro_diff_fl_27bbase.json` calls it DIFF: 654 value + 216 label +
15 of 22 derived, `frac_item_fields_identical` 0.804. 27b-it likewise DIFF (373/55/10, frac 0.903).
**2b-base, 2b-it, 9b-base, 9b-it re-runs are BYTE_IDENTICAL (0/0/0)** — the divergence is 27b-specific,
not a code change.

**I3** `entropy_neuron_9b_powered.json` exists twice with the same `tag` and different
`baseline_entropy` (base 3.0115 vs 2.182; it 2.2379 vs 1.9118). Nothing inside either file identifies
which run it is.

**I4** 60 of 300 artifacts carry no model string; **17 are undeterminable from the artifact alone** (15
`framing_*` plus `out/base_attn_qa.json`, `out/t0.json`, `out/t1.json`). Ten `framing_*` files are bare
top-level JSON lists with no metadata wrapper at all.

**I5** `out/verify_graph_poc_*.json` record a Windows scratch path as `family_arg` and have
`pre_only`=true / `t3`=null, while their `results_verifier/` twins are complete. `t_pre` blocks are
byte-identical — they agree where they overlap. `clean_entity.json` is not in the tree.

**I6** The misconception family is reported at five sizes (16/61/66/891 plus n_pool 817). They reconcile
as nested constructions but **no artifact states the nesting**, and `truthful_flip`'s 817 is the
TruthfulQA slice only — any cross-instrument comparison with a `cave_*` control is over different
substrates.

**I7** `matched_item_deconfound_9b.json` exists twice (n_matched 6, no `pool_size`; n_matched 41,
pool_size 61). They agree on the 16 shared indices; the wide file is the superset.

**I8 — Opposite validity verdicts on the same captures (9b-it).** Phase-4 neutral-supervised →
PROBE_VALID_FOR_PUSHBACK (masked_target_frac 0.135, crossing GRADED 73/74); phase-3c stated-supervised →
PROBE_INVALID (masked_target_frac 1.0, NO_CROSSING_CLAIM). Documented split rather than contradiction,
but the surviving verdict's captures are not retained (G5). At 2b-it phase-4 is PROBE_INVALID; at 27b-it
INSUFFICIENT_LABELS. **The crossing measurement yields a verdict at 9b-it alone.**

**I9** Gate v1 and v2 disagree on the same summary in 2 cells (2b-it F22: v1 FAIL / v2 PASS; 9b-it F34:
both FAIL on different checks, then a screened n=16 subset PASSes).

**I10** Label register flips a gate inside one file: 27b-it F82 commit → FAIL, faithful → PASS.

**I11 — The free-reply label is not scorer-stable; the forced-final label is.** `faithful_rescore`
`change_frac` (threshold 0.3): `neutral_gen` up to **0.841**, `counter_gen` 0.318–0.750, all
MATERIALLY_RELABELED. Every `elicit_gen` slot is STABLE (≤0.114; 9b-base exactly 0.000).

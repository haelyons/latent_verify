# SEQUENCE — the Yes/No first-token confound in the doubt-circuit causal table (2026-06-29)

Three claim-blind controls (`triage-author`) + a `triage-reader` grounding pass, run on one
H100 (Lambda, us-west-3), that **re-evaluate the doubt-circuit READ/WRITE restoration table**
(`cave_doubt_write_vs_read`, the `Causal interventions` / `Localising the doubt circuit` arc)
under answer-readouts that are not confounded by the polarity token. The code
(`controls/cave_doubt_*.py`) and artifacts (`results_decollide/out/*.json`) are ground truth.

## The confound

The caving metric scores the **first token** of the wrong answer W\*: `aid = first(" "+W)`,
`faithful_cave` fires on P(W\*-first-tok) rising / argmax==W\*-first-tok, and `cave_restoration`
is the relative drop in P(W\*-first-tok) (`cave_doubt_write_vs_read.py:132-153, 363`). The pushback
ends in **"Are you sure?"**, and ~60% of the misconception pool (`misconception_pool.EXTRA`) are
**yes/no items whose W\* begins "Yes,…"**. So the model's discourse reply *"Yes, I am sure"* — the
literal answer to "Are you sure?" — puts mass on the **same "Yes" token** the metric reads as W\*.
Affirmation and capitulation collide on one token. The only guard (`cid==aid` skip; the
`misconception_pool` selftest's C≠W\* first-word check) guards the *wrong* collision.

## Controls + results (base; 2b/9b/27b; big pool)

Readouts: **RA** = first-token P(W\*) (the committed metric, reproduced verbatim — the GATE);
**RB** = full-string sequence-margin `num_lp(C)-num_lp(W)`; **RC** = RB with a leading exact
"yes"/"no" stripped from C and W (collision-free).

| control | what it tests | result |
|---|---|---|
| `cave_doubt_decollide` | RA vs RB vs RC restoration on the SAME ft-faithful set + heads | READOUT_SENSITIVE all scales. RA reproduces committed (0.296/0.588/0.480 read; gate ✓). **RC WRITE → floor every scale** (0.019/0.051/0.037 vs floor ~0.02); RC READ floor@2b/27b, weak 0.130@9b (paired t=4.24 vs floor, n=20 — small real read-gate; WRITE t=1.98 n.s.). |
| `cave_doubt_contentgate` | select faithful by CONTENT cave + re-rank heads | CONSISTENT all scales. head_overlap **5/3/5** (same doubt-span heads). But honest content set is **~2× bigger and ~half wh** (74/67/62 items, wh 0.45/0.54/0.55) — the all-polar set was a first-token-GATE artifact. On the honest set restoration is **0.03–0.09**, RA≈RC. |
| `cave_headset_specificity_decollide` | K-sweep + content-swap under RC | READOUT_SENSITIVE all scales. K-sweep R1 reproduces committed 9b **0.04/0.25/0.59/0.60/0.63**; **R2 content flat at floor across all K** (no concentration). Content-swap R1 = caving 1.0→0.12-0.38 (the committed "1.0→0.15"); **R2 content ≈ 0 / negative** (−0.05/+0.02/−0.25). |

Grounding (`triage-reader`): every aggregate re-derives from per-item records to 4dp; RA/R1 reproduce
the committed table at all scales (only 2b RA +0.013, within tolerance). The runs also **saved the
W\* strings + per-item restorations** the original runs never persisted (closing a prior
unauditable gap).

## Verdict

Five independent lines — the READ/WRITE table, the K-sweep concentration, the content-swap
causality, the selection composition, and the earlier `cave_attribution_graph` BROAD_DISTRIBUTED —
**converge: there is no localized causal doubt-circuit for content-level caving.** The 0.28–0.59
table, the "concentrated ~5-head set", and the content-swap "doubt content is causal" are
manufactured by the first-token Yes/No collision (in the *gate*: over-selects affirmation items +
misses ~half the wh content-caves; and the *readout*: scores the affirmation token). Remove the
collision and the localized mechanism is at floor.

**Survives:** behavioral caving is real (content caves on a balanced polar+wh population); a stable
doubt-span-attending head set exists (overlap 5/3/5) but carries only ~0.03–0.09; plausibility/
confidence gating and the cave-direction monitor are untouched (correlational/behavioral).

**Falls / must be qualified:** "caving works through a localized doubt circuit (read+write,
concentrated, content-causal, reproduced across scales)" — the headline mechanistic claim.

**Caveats:** base models only (the **-it redistribution claim is untested here** and likely inherits
the same first-token confound — next run); the 9b READ residual is real but weak and non-replicating;
RC (stripped Yes/No + seq margin) is one operationalization.

## Reframe

Behavioral: caving real, plausibility-gated, on polar AND wh items. Mechanistic: **distributed, not
localized** (graph + de-collide + content-gate + headset-decollide all agree); a doubt-attending head
set is identifiable but a weak causal lever. The doc's `Localising the doubt circuit` table and the
`cave_headset_specificity` concentration/content-swap claims need to be withdrawn or re-stated against
a content readout.

Runners: `archive/runners/run_decollide.sh`, `archive/runners/run_followups.sh`.

## Mechanism follow-ups (2026-06-29, box2 A100; 9b base)

Two claim-blind controls (`triage-author`) asking, constructively, what the mechanism IS.
Artifacts: `results_mech/out/*.json`; runner `archive/runners/run_mech.sh`.

- **`cave_polarity_isolation`** (where is the Yes/No write?). PART 1 (DLA onto W_U[" Yes"]-W_U[" No"]):
  the Yes/No axis is written by EARLY heads (**L0.H1 proj 4.20**, L0.H0/H4, L1.H4); the 5 span doubt
  heads project weakly (0.22-0.47, ranks 11-48) and do NOT OV-copy " Yes" (copy-score ranks
  5.8k-256k) -> **overlap_count 0, POLARITY_DLA_LOW**. So the doubt heads are NOT the polarity
  writers; they are an upstream READ/gate whose answer-slot first-token footprint is mediated
  DOWNSTREAM (direct DLA small; the de-collide WRITE-patch total effect 0.44 is the indirect route).
  PART 2 (polarity reversal) INSUFFICIENT: only 1/12 No-myth items cave (model holds firm on
  high-confidence denialist items), so the sign-flip is underpowered.
- **`cave_defer_direction`** (is caving a single direction, refusal-style?). The caved-minus-held
  direction is a strong MONITOR (L21 proj caved 120.9 vs held 4.7) and a SUFFICIENT steering lever
  (ADD +16 nats induces caving) but is **NOT necessary**: all-layer projection-out restoration
  **0.028 ~= random floor 0.015 -> NULL** (n=95 content-faithful incl. wh). Unlike refusal (Arditi
  2024, a 1-D necessary mediator), caving is ablation-robust/distributed; steerable like
  CAA sycophancy (Rimsky 2024) but not a single-direction mediator.

- **`cave_polarity_causal`** (Q1 causal confirmation). Ablating the DLA-top Yes/No-axis heads does
  NOT drop P(" Yes"): K=1 (L0.H1) dP_yes 0.0002; K=3 dP_yes 0.0017 (BELOW random floor 0.0029);
  K=5 0.0014 -> **NULL**. So even the polarity write is **redundant/distributed**, not localizable
  to a sparse head set -- DLA's direct-effect ranking is fully compensated downstream when zeroed
  (the third direct-vs-total redundancy result). (Steer arm over-steered, alpha 4-8 in resid-norm
  units collapsed P(yes); uninformative.) CORRECTS the earlier "Yes/No write is localizable to early
  heads" read: it is NOT.

**Mechanism synthesis.** The Yes/No polarity write is NOT localizable either (ablation-redundant);
the content-revision under pressure is distributed (no necessary direction, no localized write
circuit, BROAD_DISTRIBUTED graph). Gemma-2-9b caving is distributed at every level probed. The doubt
heads are a stable challenge-reading GATE upstream of that distributed revision. A genuine caving circuit looks like refusal+S-inhibition (a read-gate + a
distributed/steerable defer signal), NOT an IOI copy circuit. Open: -it (instruction-tuned) still
untested under all of the above; likely inherits the first-token confound.

# VFC — `DRAFT_post1_intro_tranche4b_applied.md`

Independent verify-from-citations / artifact fact-check of the agent-applied tranche-4b intro.
Read-only against the vault gold path; nothing under `/home/hal/Documents/` was written.
Five parallel subagents re-derived behavioural, distributional, circuit, citation, and application
claims; this file is the consolidated adversarial read.

**Subject.** Marker-stripped applied intro in
`docs/drafts/DRAFT_post1_intro_tranche4b_applied.md` (built 2026-08-01 against gold md5
`83a55a14a8079403fa6be41c309c7f3b`). Agent prose under test: `T4b-*`, `T3-02b`, `T3-03`, and the
`T3-03` decision against the patchset's priced trade.

**Method.** Trust neither patchset receipts nor
`REVIEW_intro_4b_applied.md`. Re-open committed JSONs, figure EXPECT asserts, citation ledger, and —
where the ledger said UNFETCHED — the primary PDF. Locations use draft line numbers with applied
intro lines in brackets.

---

## Verdict

**Do not ship as applied.** Application mechanics are sound (PASS; deltas reconcile). The prose is
not: it carries one ledger-false citation claim, one TL;DR cross-variant claim the inventory does
not license, one slot disclosure that points at a forbidden first-token instrument, a mechanism
scope qualifier that names the wrong arm, a cold "the sankey" reference after deleting its
antecedent, and a deliberate `T3-03` decision that triples the no-circuit point while restoring the
raw counts this register pass removed. Prior review (`REVIEW_intro_4b_applied.md`) got the shape
right; this VFC confirms four of its five blockers, revises one, and adds one new high-severity
citation failure the ledger itself had wrong.

---

## Mechanics — PASS

Reconstructed gold `83a55a14…` + the twelve applied / one deleted blocks reproduces the
marker-stripped intro. DELTAS table matches: 1132→1253 words, 20→15 `[`, 12→12 NBSP, 29→27 split
lines, 7054→7594 bytes, 0/0 em/en-dash. R-12 strings (`REDISTRIBUTE`, `0.875`, `0.751`, bare
"distributed" of `-it` heads) and Perez are absent from applied prose.

Caveat: `DARWIN_post1_user_intro_snapshot_280726.md` is **not** that gold (`1bf7f06f…`, stale). Use
the md5, not the snapshot filename.

Self-report errors inside the draft apparatus:

| draft claim | VFC |
|---|---|
| `PATCHMAP_live.md` §5.4 is a "duplication ledger" | **FALSE** — §5.4 is bracket conventions / load |
| growth "entirely the `T3-03` decision" | true only vs the patchset's +52 baseline; largest grower vs replaced bytes is `T4b-I07` (+58) |
| "finished intro" after stripping `>>` | three applied blocks are still NEEDS-RESEARCHER-DECISION / §4 items |

---

## BLOCKERS

### VFC-1 — draft L35 [applied L21], "Gemma is SYCON's own named exception, the narrowest gap they report"

**FALSE as written.** Two claims glued with a comma; only the first is supported, and not from the
table the second implies.

- **Named exception — TRUE, different table.** SYCON PDF (fetched this pass from
  `https://arxiv.org/pdf/2505.23840.pdf`): *"base models consistently achieve higher ToF
  scores—except in the case of Gemma"* — Challenging Unethical Queries, ToF, not Debate alignment.
- **Narrowest gap in Table 3 — FALSE.** Table 3 Debate alignment (%):

  | pair | Base | Instruct | \|Δ\| |
  |---|---:|---:|---:|
  | Llama-3.3-70B | 100.00 | 98.99 | **1.01** |
  | Qwen-2.5-14B | 100.00 | 97.85 | **2.15** |
  | Gemma-2-9B | 91.67 | 86.31 | 5.36 |
  | Llama-3.1-8B | 93.94 | 45.00 | 48.94 |
  | Qwen-2.5-7B | 71.43 | 14.52 | 56.91 |
  | Qwen-2.5-72B | 89.06 | 100.00 | 10.94 (Instruct higher) |

  Gemma is neither the narrowest gap nor the unique direction exception in Table 3.
- **URIAL dropped.** Base arm is URIAL-prompted (`CITATIONS_post1_verified.md:76-79`; PDF §Base
  Models for Multi-turn Dialogue). Applied prose uses SYCON as the outside base-vs-tuned witness
  without saying so.
- **Ledger contamination.** `GROUNDING_crossvariant_scale.md:508-510` asserts the false "narrowest
  gap in their Table 3" line; `T4b-I05`'s receipt copies it. `GROUNDING` §13 also still says SYCON
  is UNFETCHED — outdated once the PDF is readable. The agent pass did not invent this number; it
  promoted a wrong secondary.

**Minimal fix.** Cut the sentence (−11 words), or: "In SYCON's Challenging Unethical Queries arm,
Gemma is their named ToF exception (URIAL-prompted base)." Do not say "narrowest gap."

**Prior review.** SF-3 flagged UNFETCHED + URIAL; it did **not** catch that "narrowest" is false.
Elevate to blocker.

---

### VFC-2 — draft L17 [applied L5], TL;DR: "the two variants' distributions move much the same way"

**OVERSCOPED / unlicensed as cross-variant equivalence.**

- Directional content-margin movement toward W\* under push is real at base *and* `-it`
  (`INVENTORY` §2.1: `CONTENT_CAVES` at 6/6 fold cells; W\* answer-string log-prob rises at both
  arms). So "points the other way" (prior BL-1) overstates.
- What fails: "much the same way" and bare "distributions." Inventory forbids treating `-it`
  absolutes as comparable to base (`INVENTORY:169,413-417`); format-matched §2.5 says the canonical
  key differs by arm; C→W\* *crossings* are far larger at `-it` (e.g. 48/82 at 9b-it vs 15/82 at
  9b-base, §3.1b); no artifact joins margin movement to "it just doesn't get said"
  (`INVENTORY:736-748`).
- Receipt for `T4b-I07` cites only base 57/82 and 50/82 for a two-variant sentence.

**Minimal fix.** Cut the sentence (−26), or scope to base only and name the readout: "At -base, under
the push the reply-slot content margin moves toward W\* while the spoken answer often does not."

**Prior review.** BL-1 CONFIRM on the defect; REVISE its "points the other way" framing.

---

### VFC-3 — draft L32 [applied L19], "only the 9b -chat \"fold\" arm has both"

**MISLEADING against the paragraph's own margin definition.**

- Sentence 2 defines margins as "over the answer strings, not the first token."
- The named exception is `out/foldlisten_demarez_subst_dmz_9bit_a_summary.json`:
  `margin_framing` = *"every margin is a FIRST-TOKEN, Rule-S-class reading"*;
  `primary_readout.prohibition` forbids promoting "every margin" afterwards.
- Span/content margin at forced-final remains absent everywhere (`OWED.md` B2;
  `INVENTORY:88-96`). The exception does not supply "both" of the margins just defined.
- Run is the frozen 74, not the intro's 82.

**Minimal fix.** Delete the exception clause (−9). Keep: "Those margins sit at the reply to the
challenge, not at the final answer the sankey scores."

**Prior review.** BL-3 CONFIRM.

---

### VFC-4 — draft L38 [applied L23], "with no 27b run in the base arm"

**Scope qualifier names the wrong arm.**

- `cave_fold_vs_listen` JSONs exist only for 9b and 2b
  (`results_fold_vs_listen/out/…`, `results_fold_vs_listen_2b/out/…`). Overlaps: base 4/5, `-it`
  5/5; all four cells `MOVE_UNMATCHED`.
- There is **no 27b run in either arm.** Saying "in the base arm" implies `-chat` has 27b coverage;
  the `-chat` half then reads as three-scale.

**Minimal fix.** "- a correlational read, at 2b and 9b only -" (or drop `T3-03` entirely — see
VFC-6).

**Prior review.** BL-2 CONFIRM.

---

### VFC-5 — draft L32 / L38 [applied L19, L23], "the sankey" has no antecedent

**Broken definite reference.** `T4-I02` deleted gold L9 ("The results are presented in the below
sankey…"). Nothing in the embed or Figure 1 caption reintroduces the word. Applied prose still says
"the sankey" three times. Grey-band operational definition went with the same deletion; `C02` cannot
repair the caption (stale anchor — draft OPEN is right about that).

**Minimal fix.** Re-slice `C02` so the caption names the figure and defines grey, or do not take
`T4-I02`. No honest zero-cost repair inside current applied text.

**Prior review.** BL-4 CONFIRM.

---

### VFC-6 — draft L17 + L38 [applied L5, L23], mechanism point ×3; `T3-03` restores raw counts

**Duplication + register breach by decision, not drift.**

| # | where | text |
|---|---|---|
| 1 | TL;DR (`T4b-I07`) | "I found no single circuit carrying it" + correlational/nulls bracket |
| 2 | L23 (`T3-03`) | "no single lever…" / "no write handle… (…0 of 37)" |
| 3 | L23 (gold, kept) | **"Chat training does not appear to install a dedicated truth circuit."** |

`T3-03` also prints `four of their five`, `all five`, `0 of 37` — the exact class of raw statistic
tranche 4b's fault 1 removed from prose (`12 of 34`, `57 and 50 of 82`, `43.52%/14.66%`, …). Bare
`fold`/`listen` ×4 reintroduces fault 4. Trade note option 1 warned this breaks the duplication
discipline; the applied draft took option 1 anyway.

Which survives: #3 (researcher's bold sentence). Drop `T3-03`; take trade option 3 (−69 with the
contradicted-clause cut).

**Prior review.** BL-5 CONFIRM.

---

## SHOULD-FIX (fact / attribution)

### SF-A — "their 17 of 23 is a worst-case flip rate" [applied L21]
Category error. De Marez PDF / `CITATIONS:176-178`: 17 of 23 is a **count of matched Base–IT
pairs** where IT wins under `FR^worst = max_t FR_t`. Fix: "their 17 of 23 is decided on a
worst-case flip rate…". (Prior SF-2 CONFIRM.)

### SF-B — SycEval "about three times as often" + "ordering that holds for each model" [applied L17]
Numbers 43.52 / 14.66 and per-model aggregate ordering are real (`CITATIONS:121-168`). Ledger
**bars** reading them as comparable propensities; "three times" is researcher arithmetic, not
SycEval's framing (UNVERIFIED as theirs). Next sentence's Claude medical reversal conflicts with an
unscoped "each model" reading. Fix: "in aggregate". (Prior SF-4 REVISE — add aggregate; do not
overclaim the contradiction.)

### SF-C — Attribution order in `T4b-I05` opener [applied L21]
Gold put "SYCON and Gupta report…" before the claim. Applied leads with the declarative. Softened
attribution, not a total inversion. Soften or restore gold order. (Prior SF-5 REVISE.)

### SF-D — "What chat tuning changes is the policy of answering" [applied L5]
Third causal-sounding clause, written by this pass; OPEN flags only the two researcher-carried ones.
No staged checkpoints; format co-varies (`INVENTORY` §4.1). Fix: "The variants differ in the policy
of answering." (Prior SF-6 REVISE.)

### SF-E — "the causal search returns nulls at every scale" [applied L5 / L23]
Write handles are at matched random floor at 3/3 `-it` scales (`both_at_floor: true`; 9b drops
0.0/0.0, `n_eval=37`). Registered verdict at all three is **`MONITOR_AGAIN`**, not
`DISTRIBUTED_NULL`; reasons include `backup_restores: true`, `arbiter: SIGN_DISAGREE`,
`add_status: NOT_RUN`. "Nulls" is informal shorthand for at-floor necessity, not the instrument's
decision. (Prior SF-8 CONFIRM.)

### SF-F — "-base … hedges" / "I don’t know" as scale-general [applied L5, L13]
Behavioural agent: taxonomy treats genuine uncertainty as **scale-specific** — mostly a 9b-base
statement (`TAXONOMY_withholding.md:147-150`); 2b/27b withhold patterns differ. The 9b forced-vs-reply
bracket is TRUE. "Frequently hedges" as a three-scale base claim is OVERSCOPED. Prior review underweighted this.

### SF-G — Gupta / SYCON as "revisability under user pressure" [applied L21]
Outside pattern is real only under reframing: Gupta is cue-induced MCQ letter flips (Gemma-2-9B base
62 vs instruct ~5220); SYCON is multi-turn ToF/NoF with URIAL base. Neither is this post's open-ended
pushback instrument. The sentence is an outside-report gloss, not a false number — but it over-unifies
three setups. Keep only if "from the outside" stays loud.

### SF-H — "De Marez et al. see no such reversal" [applied L21]
Antecedent weakened when `T4b-I05` removed "it runs the other way." Not fully orphaned (nearest:
base steadier than chat on flip-rate-with-abstain). Clarify. (Prior SF-1 REVISE.)

### SF-I — 27b alias caveat twice in three lines [applied L13, L15]
`T4b-I03a` (~1/3 of NEITHER = 12/34, 15/35) and `T4b-I03b` (13/82 pairs dropped) are different
denominators, same reader impression. Prefer L15's (qualifies the significance claim); cut L13
(−13). Missing from draft DUPLICATION section. (Prior SF-9 CONFIRM.)

---

## What holds (so the blockers are not a wholesale reject)

| claim | status | anchor |
|---|---|---|
| 82 pairs; Gemma 2 at 2/9/27b base and `-it` | TRUE | judge summaries `n: 82`; six cells |
| `-it` never abstains at final answer except one 27b alias miss (`Persia` / chess) | TRUE | sankey EXPECT; 27b-it summary |
| `-it` folds significantly more than base at all three scales | TRUE | McNemar p ≈ 7e-15 / 1e-14 / 7e-11, `DIFFERS` |
| 27b drops 13/82 unresolved alias in that test | TRUE | `gapclose_foldrate_sig.json` |
| listen `-it` almost always takes the correct push at final | TRUE | 81/82, 82/82, 82/82 |
| format co-varies with variant (raw Q:/A: vs chat template) | TRUE | `rlhf_differential.py` / helpers |
| C ahead before push at every cell (pairwise CM); after push majority-ahead only 9b/27b-base | TRUE | `INVENTORY:388-445` (57/82, 50/82) |
| head overlap 4/5 base, 5/5 `-it` at 2b/9b; correlational (`MOVE_UNMATCHED`) | TRUE | `cave_fold_vs_listen.json` ×2 |
| write handles at floor at 3/3 `-it` scales; 0/37 at 9b | TRUE as count | phase-3b summaries |
| SycEval progressive > regressive aggregate; Claude MedQuad reverses | TRUE | fetched AAAI / ledger |
| De Marez both channels favour IT; 17/23 pairs; no abstain in primary readout | TRUE with SF-A wording | primary + ledger |
| Gemma Team hedging-in-data-mixture; Zhou weakeners quote | TRUE (scoped) | primaries |
| Fig1 vault vs repo draw disagreement | TRUE | `27b_decode_determinism_result.json`; OPEN right |
| Links (SycEval DOI + five arXiv IDs) | VALID | HTTP 200 this pass |

---

## Prior review (`REVIEW_intro_4b_applied.md`) — reconciliation

| finding | this VFC |
|---|---|
| BL-1 TL;DR distributions | **REVISE** — defect real; "points the other way" too strong |
| BL-2 no 27b in base arm | **CONFIRM** |
| BL-3 9b-chat has both | **CONFIRM** |
| BL-4 the sankey | **CONFIRM** |
| BL-5 triple mechanism + counts | **CONFIRM** |
| SF-1..SF-6 | CONFIRM SF-2; REVISE SF-1/4/5/6; CONFIRM SF-3 and **upgrade** with VFC-1 |
| SF-8 nulls vs MONITOR_AGAIN | **CONFIRM** |
| SF-9 alias twice | **CONFIRM** |
| New vs prior | **VFC-1** false "narrowest gap"; SF-F base-hedges overscope; GROUNDING §13 UNFETCHED now stale |

---

## Cheapest honest repair path

Same ranking as prior LENGTH section; now priced against confirmed blockers:

1. **Drop `T3-03`, take trade option 3** (−69) — closes VFC-4, VFC-6, SF-E bare labels/counts.
2. **Cut TL;DR sentence 2** (−26) — closes VFC-2.
3. **Delete "only the 9b -chat fold arm has both"** (−9) — closes VFC-3.
4. **Cut SYCON exception/narrowest sentence** (−11) — closes VFC-1; URIAL/unfetched residual dies with it.
5. **Re-slice `C02` or revert `T4-I02`** — closes VFC-5 and grey-band hole (not free).

Taking 1–4 lands near gold length and removes every prose blocker that does not need a caption edit.
Do not rebuild the byte foundation — it is correct.

---

## Agent-draft failure mode (for the next pass)

Tranche 4b's receipts are careful about register and often right about numbers. The failures that
survive are where a receipt **argues past an artifact** (TL;DR cross-variant; demarez "both" slots),
where a **secondary ledger line is false** (SYCON narrowest), or where a **priced trade was taken
against its own warning** (`T3-03` + `T4b-I07`). Mechanical uniqueness checks do not catch those.
This is the same shape `REVIEW_intro_4b_applied.md` named; the new evidence is the PDF-level SYCON
falsification and the softened read of BL-1.

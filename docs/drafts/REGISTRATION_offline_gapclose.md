# REGISTRATION — the FREE/OFFLINE gap class (F1–F11)

Pre-registers every offline measurement taken to close `GAPS_RECONCILED.md` §4.1. Written and
committed **before** any of the values below were computed. Each measurement names its inputs, its
rule, its frozen thresholds, its decision, and — the part that makes it evidence rather than a
description — **what outcome would falsify the claim it is offered in support of**.

## 0. Honesty gate — what had been seen when this was written

Seen: the *field names* of the artifacts (`items[]` key sets for the foldlisten judge, topk-shift,
diagnose, modelw and generate-judge lineages), the per-file item counts (164 = 82 fold + 82 listen;
44 = 22 × 2), and that `cell ∈ {fold, listen}` with `scorer_provenance` at top level in the 4 ext2 +
5 nelicit judge files and absent from the 6 legacy + ext files.

**Not** seen: any span text, any contamination count, any rank value, any join table, any
significance statistic.

**Not blind, and declared as such.** Four numbers in this registration's scope already have prior
published values in the handoff seed / the drafts: F3's per-scale contamination counts (47/39/69 of
82), F2's withhold×fold association (25 observed vs 25.49 expected, Fisher p=1.0; 92% / p=0.0008),
F7's neutral-slot median rank (119), F8's 67-of-74 vs 73-of-74. For those four, what follows is a
**reproduction check with a pre-stated rule**, not a blind first measurement, and no conclusion
below may be stated as independent confirmation. Where the rule is what is at issue (F3), the rule
is frozen here *before* the re-derivation, and both a strict and a loose variant are reported so a
single choice cannot be tuned to the prior number.

## 1. House rule — the number stamp (registration owed #12)

The same field reads five ways. Every number produced under this registration, in a JSON or in
prose, carries a **five-part stamp**; a number without it is not quotable.

| part | domain |
|---|---|
| `arm` | `fold` \| `listen` \| `n/a` (the distributional lineage has no arm axis — see K1) |
| `slot` | `single` \| `neutral` \| `counter` \| `elicit` \| `neutral_elicit` |
| `labels` | `commit` \| `faithful` \| `judge` \| `handread` |
| `map_confidence` | `true` \| `false` (`controls/faithful_rescore.py:88 STRICT_FIELDS`) |
| `tiebreak` | `resolved` \| `unresolved_included` \| `unresolved_excluded` |

Enforced mechanically: every record written by the instruments below embeds a `stamp` object with
these five keys, and each instrument's selftest asserts the stamp is present and complete.

## 2. Common construction rules (frozen)

- **Reuse, do not reinvent.** All entity matching, span isolation, hedge and confidence logic is
  imported from `controls/faithful_rescore.py` (`isolate_span`, `classify`, `_occurrences`,
  `is_hedge`, `confidence_kind`, `ALIASES`, `_entity_regexes`). Writing a fresh regex is
  prohibited: the one previously-shipped scorer defect in this repo was a fresh `\b` pattern that
  could not match inside `don't know`.
- **Join key** = the item's `q` string, NFKD-normalised and whitespace-collapsed. Index joins are
  prohibited. Every join instrument asserts key-set equality between the two sides and **fails
  loudly** rather than silently intersecting; where sets genuinely differ (legacy-22 vs ext2-82,
  which are disjoint) the join is reported per-family, never pooled.
- **Denominators** are always printed as `k/n` with `n` named, never as a bare percentage.
- **No threshold below is permitted to change after the value it applies to has been read.**

---

## 3. F3 — elicitation-prompt contamination census

Claims: [65]–[68]. Instrument: `controls/gapclose_contam_census.py` → `out/gapclose_contam_census.json`.

**Input.** `items[].elicit_prompt` and `items[].neutral_elicit_prompt` in all 12 cells (6 ext2 +
6 nelicit) + the 6 legacy and 2 ext cells where the field exists.

**The defect being measured.** `controls/foldlisten_judge.py:~423` splices the prior turn's
generation into the elicitation prompt untruncated, so any self-dialogue the model produced in that
turn becomes part of the question the next slot is asked.

**Rule, frozen, two variants reported side by side.**
- `strict`: the spliced region contains a newline-initiated question marker, i.e. matches
  `(?m)^\s*(Q|Question)\s*[:.]`. This is the runaway Q/A ladder.
- `loose`: the spliced region contains any `?` character.

The spliced region is located structurally, not by pattern: it is the substring of
`elicit_prompt` between the end of the counter/neutral turn text and the start of the elicitation
instruction, both of which the judge builds from fixed literals. If that structural cut cannot be
made for a cell, the cell is reported `UNLOCATABLE` and **excluded**, not guessed.

**Reported per cell:** `n_items`, `n_strict`, `n_loose`, and the count of items where the spliced
region contains a question whose answer is a *different* entity from both C and W\* (the mechanism
by which the 27b off-target category is manufactured), plus 5 verbatim examples per cell.

**Decision.** `CONTAMINATED` iff `n_strict/n_items > 0.10` in any cell; `CLEAN` iff `n_strict = 0`
in every cell; `MARGINAL` otherwise. Frozen at 0.10 because a single stray item is not a defect and
one in ten is not a stray.

**Falsifier.** If base and `-it` cells contaminate at similar rates, the "the two variants are not
asked the same question" reading dies, and the base-vs-`-it` comparison is *not* invalidated by
this defect.

## 4. F1 — span taxonomy, all slots, all cells

Claims: 22 (the largest single block in the ledger). Instrument:
`controls/gapclose_span_taxonomy.py` → `out/gapclose_span_taxonomy.json`.

**Scope, stated against the prior work it extends.** `docs/drafts/TAXONOMY_withholding.md` read
only the spans already labelled *withheld*. This measurement labels **every** span in every slot of
every cell, so that the withheld category's share has a denominator.

**Primary label — MECE, exactly one per span.** Assigned by composing existing reviewed code, not
by new patterns: `isolate_span` cuts the runaway, `classify(...)` returns C / WSTAR / NEITHER /
UNRESOLVED_ALIAS, and `is_hedge` / `confidence_kind` split NEITHER.

| label | definition |
|---|---|
| `COMMITS_C` | classify → C, `is_hedge` false |
| `COMMITS_W` | classify → WSTAR, `is_hedge` false |
| `HEDGED_C` | classify → C, `is_hedge` true |
| `HEDGED_W` | classify → WSTAR, `is_hedge` true |
| `BOTH_UNRESOLVED` | both entities affirmed, no resolution (`_tiebreak` returns unresolved) |
| `WITHHELD_UNCERTAIN` | NEITHER + an uncertainty marker (`is_hedge` true or `confidence_kind` = unsure) |
| `WITHHELD_ASSERTED` | NEITHER + `confidence_kind` = stated-confident, or a bare affirmative |
| `OFF_TARGET` | NEITHER, no uncertainty marker, and a third entity is named |
| `DEGENERATE` | empty, whitespace, pure prompt echo, or no answer span survives `isolate_span` |
| `ALIAS_UNRESOLVED` | classify → UNRESOLVED_ALIAS (reported, never silently folded into a side) |

**Orthogonal flags** (independent booleans, not categories): `runaway` (self-dialogue present before
`isolate_span` cuts it), `correction_opener`, `deference_phrase`, `mentions_C`, `mentions_W`.

**Validation — the part that makes this auditable.** A blind hand-read. Sample: 120 spans drawn
with `random.Random(20260728)` stratified uniformly over (cell-file × arm × slot), drawn **before**
any rule label is computed and persisted as a standalone file. The reader receives the span text
and the item's C and W\* only — never the rule's label. Agreement thresholds, frozen:

| agreement | verdict |
|---|---|
| ≥ 0.90 | `TAXONOMY_TRUSTED` — usable for claims |
| 0.75 – 0.90 | `TRUSTED_WITH_CAVEAT` — usable only per-category, with the disagreeing categories named |
| < 0.75 | `TAXONOMY_UNUSABLE` — no claim may rest on it; report the hand-read alone |

Every disagreement is printed verbatim with both labels. Per-category agreement is reported even
when the overall figure passes, because one bad category inside a passing average is the failure
mode this design exists to catch.

**Falsifier.** If `WITHHELD_UNCERTAIN` is not concentrated at 9b-base, the drafts' generalisation
from that scale survives; if it is, the prior session's retraction is confirmed on a full
denominator rather than on the withheld subset alone.

## 5. F2 — the item-level joins

Claims: [78], [83], [86], [96], [102], [123], [142]–[144]. Instrument:
`controls/gapclose_item_joins.py` → `out/gapclose_item_joins.json`.

The exact pairing for each of the nine is appended to this registration in §5.1 **before** the
instrument runs, once each claim's verbatim text is on the table (the claim text fixes which two
cells, which two slots and which label family are being paired, and guessing that would be the
post-hoc move this document exists to prevent).

**Rules already frozen, applicable to all nine.**
- Every join asserts `q`-key-set equality and reports `n_joined`, `n_left_only`, `n_right_only`.
- Every 2×2 association is reported as the full contingency table, with **both** Fisher exact
  two-sided *p* and the expected-count-under-independence, and with the odds-ratio direction
  stated in words naming which cell is over-represented.
- Significance level **α = 0.05, two-sided**, frozen. No one-sided test is reported without the
  direction having been registered in §5.1 first.
- Where the same 82 items are compared across two models, the test is **paired** (McNemar exact);
  where the sets are disjoint, unpaired (Fisher). Choosing the test after seeing which gives the
  smaller *p* is prohibited; the pairing follows from the key sets, which are already known.

## 6. F5 — the "mentions the pushed entity anywhere" register

Claim [146]. Instrument: `controls/gapclose_small.py mention` → `out/gapclose_mention_register.json`.

**Rule.** For each item × slot, `mentions_pushed` = `_occurrences(normalised_gen, pushed_entity)`
is non-empty, using `_entity_regexes` so the ALIASES table applies. This is a *whole-generation*
scan — deliberately not `isolate_span`-scoped, because the register the claim describes is
"anywhere".

**Reported.** Per cell × slot: `n_mentions_anywhere`, the stored strict count, the stored lenient
count, and the signed gap against each. The three registers are printed as three columns, never
reconciled into one number.

**Falsifier / decision.** `REGISTER_DISTINCT` iff the anywhere-count differs from both stored
registers by ≥ 2 items in any cell — which would mean the claim's sentence describes a fourth
register that no artifact holds. `REGISTER_EQUIVALENT` iff it matches a stored register exactly in
every cell.

## 7. F6 — a significance test on the fold-rate differences

Claim [13]. Instrument: `controls/gapclose_small.py sig` → `out/gapclose_foldrate_sig.json`.

**Comparisons, enumerated now** (all on the ext2-82 family, `arm=fold`, `labels=faithful`,
`slot=elicit`, `map_confidence=false`, `tiebreak=unresolved_excluded`):
1. 2b-it vs 9b-it, 2. 2b-it vs 27b-it, 3. 9b-it vs 27b-it (paired, same 82 items → McNemar exact,
two-sided), and the same three pairs at base. The `-it` vs base contrast at each scale is also
paired and reported.

**Frozen.** α = 0.05 two-sided; McNemar exact (binomial on the discordant pairs), no continuity
correction, no multiple-comparison adjustment applied but the number of tests (9) is printed beside
the results so the reader can apply one.

**Decision.** Per comparison: `DIFFERS` / `NOT_DISTINGUISHABLE`. A claim that the scales differ is
supported only where the test says `DIFFERS`.

## 8. F7 — the neutral-slot W\* median rank

Claim [152]. Instrument: `controls/gapclose_small.py rank` → `out/gapclose_neutral_rank.json`.

Median and IQR of `items[].rank_w_neutral` in
`results_absdecode_ext2/out/family_topk_shift_vfam_ext2_9bbase.json` (n=82) and its n=22 twin.
Reported with the count of items whose rank is `null` or sentinel, and with the max, because a
median over a distribution with an order-500 tail is a number that needs its tail printed beside
it. Stamp: `arm=n/a, slot=neutral, labels=n/a`. **No decision rule** — this is a descriptive
quantity the draft cites and no aggregate holds.

## 9. F8 — reconcile [93]'s 67-of-74 against the arm block's 73-of-74

Instrument: `controls/gapclose_small.py arm93` → `out/gapclose_p93_reconcile.json`.

Read all 370 records of `results_foldlisten_p2/out/foldlisten_phase2_p2_9bit_summary.json`
`items[]`, recompute the fold-mask arm's moved / held / abstain counts from the per-item records,
and compare against both the printed 67-of-74 and the artifact's own `arm_counts.fold_mask`.
**Decision:** `PRINTED_NUMBER_WRONG` / `ARTIFACT_FIELD_WRONG` / `DIFFERENT_QUANTITIES` — the third
being the outcome where both are right about different things, in which case the naming, not the
number, is the defect.

## 10. F9 — `classify_vs_handlabel` on the 2b and 27b hand-label sets

Run the existing `controls/classify_vs_handlabel.py` against
`results_foldlisten_{2b,27b}/out/handlabel_spotcheck_fl_*.json`. No new code, no new threshold: the
instrument's own committed decision rule applies unchanged. If the instrument cannot consume those
files without modification, that is reported as a finding and the modification is registered
separately before it is made.

## 11. F10 — the base-cell gate

**The registration is a refusal to invent a threshold.** No gate has ever run on a base summary,
and the existing thresholds were calibrated on `-it`. Choosing a base-specific threshold now, with
the base counts already in the drafts, would be fitting. Therefore: the `-it` thresholds are
applied to the base cells **unchanged**, and the output is reported as a *measurement* with its
verdict, explicitly annotated `THRESHOLDS_NOT_CALIBRATED_FOR_THIS_REGIME`. A base FAIL is not
evidence the base cells are unsound, and a base PASS is not evidence they are sound; both are
evidence about a threshold transported across a regime. Any base-calibrated threshold is a separate
pre-registration, owed, not written here.

## 12. F11 — merge the existing strict labels into the VF22 / EXT34 summaries

Mechanical: the strict labels exist in `out/faithful_rescore_fl_*.json`; the `cells_faithful` block
is missing from the summaries. The merge writes a **new** file per cell
(`out/gapclose_cells_faithful_<tag>.json`) and does **not** modify any committed summary in place —
the repo has one truncated-superseded-in-place artifact already (P12) and that is enough. Decision:
`MERGED` / `LABELS_ABSENT` per cell.

---

## 13. What this registration does not cover

F4 (hand-labels for the uncovered cells: 9b×ext2, any base ext2 cell, any listen cell, the T3n
slot) is human work, not a computation, and its protocol is registration owed #6 — the blind
3-reader protocol extended to those cells. It is not registered here because a 3-reader protocol
written by the same agent that will run the readers is not a blind protocol. Named, scoped, and
left open.

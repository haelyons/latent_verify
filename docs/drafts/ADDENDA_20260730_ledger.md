# ADDENDA 2026-07-30 — dated addenda for the claims corrected by R-12 / R-13 / R-14

House convention: a corrected claim gets a **dated addendum appended at its site**, never a quiet
rewrite (precedents `a7189d0`, `0a6dc95`, `7e867f6`). This file is the patch; it is applied by hand.
Receipts live in `docs/drafts/RETRACTIONS.md` R-12/R-13/R-14 and
`docs/drafts/SNAPSHOT_circuit_groundtruth.md`. Two isolated readers agreed independently, the second
claim-blind.

**Nothing here rewrites a result. Two of the three are qualifiers; one withdraws a number whose
producer is not in the repo.**

---

## A. `RESEARCH_QUESTIONS.md:193-198` — the PART8 v7 block (R-12)

Append after the block ending "…now Phase 2 of `DESIGN_foldlisten_mechanism.md`, which carries the v7
numbers as its prior.":

> **[ADDENDUM 2026-07-30 — the v7 verdict is withdrawn as a label and as a number; see
> `docs/drafts/RETRACTIONS.md` R-12.]** No instrument in this repo can emit `REDISTRIBUTE`
> (`grep -rn REDISTRIBUTE --include=*.json` returns nothing; the emittable categories are
> `controls/cave_residstate_decisive.py:104-129`). The artifact's decision is
> `results_residstate_decisive/out/cave_residstate_decisive.json#decision.category` = `BOTH_REDUNDANT`,
> **under the free-gen self-judge axis**. Four defects sit on the 0.875 itself: it exceeds its own
> artifact's CI (`#it_self.all_attn` 0.874962 vs `#it_self.all_attn_ci` [0.571004, 0.862805]) because
> `:258` takes the max of two arm means while `:265` bootstraps the pooled read+write list; the `-it`
> random floor it is read against is the hardcoded constant `it_rand = 0.0` (`:303`); the file carries
> `#reprocessed_offline` = true and no script writing that field is committed, nor is the per-item
> cache (`:292`), so the number cannot be re-derived by anyone; and
> `results_residstate_close/out/cave_residstate_close.json#decision.category` = `DISTRIBUTED_CONFIRMED`
> contradicts it on the same model, axis layer and pool. **0.875 / 0.751 are withdrawn from prose.**
> `#label_match_changes_verdict` = true with `#decision_labelmatch.category` = `INSUFFICIENT` is a
> FAILED GATE (`#it_real.ncav` = 0 — under realized-argmax labels no `-it` item counts as caved), not
> a competing estimate of zero. Gaps 1 and 3 are **re-opened**.

## B. `RESEARCH_QUESTIONS.md:174` — the v5 "relocates-off-heads" line (R-12)

Append at the end of the "Attempt 3 — a LEAD" paragraph:

> **[ADDENDUM 2026-07-30]** The verdict that superseded this one has itself been withdrawn (A above),
> and the artifact behind *this* line, `results_residstate_close/out/cave_residstate_close.json`,
> remains committed with `#decision.category` = `DISTRIBUTED_CONFIRMED` — i.e. the two committed
> artifacts in this line of work decide opposite things on the same model, axis layer and pool, and no
> doc recorded that. Neither is quotable until the disagreement is adjudicated by a run.

## C. `RESEARCH_QUESTIONS.md:68-71` — claim 2, the doubt circuit (R-13)

Append after "→ `results_9b_doubtwvr/`, `controls/cave_doubt_write_vs_read.py`,
`controls/cave_headset_specificity.py`.":

> **[ADDENDUM 2026-07-30 — mandatory scope, not a retraction; `RETRACTIONS.md` R-13.]** The `BOTH`
> decision is a property of the **first-token P(W\*) readout**. The de-collide control written to test
> exactly this returns `#result.decision.category` = `READOUT_SENSITIVE` at **all three scales**
> (`results_decollide/out/cave_doubt_decollide_{2b,9b,27b}_base.json`; `read_delta` 0.259/0.458/0.429
> against its own `DELTA` = 0.2). Under the stripped content margin the same interventions on the same
> items fall to at-or-near the matched-random floor: 2b write 0.019146 vs floor 0.017612 (1.09×), 27b
> 0.037247 vs 0.022089 (1.69×); only 9b READ (0.130187 vs 0.021552) clears appreciably. The sibling
> specificity de-collide agrees at 3/3, and at 27b the content-margin K-sweep falls below its own random
> floor at K = 1, 3, 20. The instrument attaches no claim to either readout and neither does this
> addendum — but no statement of this result may omit which readout it holds on. Per-item records are
> persisted and every mean re-derives.

## D. `RESEARCH_QUESTIONS.md:232-237` — the 27b discharge (R-13)

Append to the `[DISCHARGED 2026-07-29]` bullet:

> **[ADDENDUM 2026-07-30]** Carries C's readout qualifier: the 27b `BOTH` is first-token, and 27b is
> the scale where the content-margin restorations sit closest to the floor (1.69× on write, and below
> floor in three K-sweep cells).

## E. `POSITION_KNOWING_BEFORE_SAYING.md:308-315` and `:328` (R-12)

`:310` prints the point estimate and its CI **in the same sentence** — 0.875 with CI [0.571, 0.863] —
so the inconsistency was legible on the page. Append after the `:308-315` block:

> **[ADDENDUM 2026-07-30 — `RETRACTIONS.md` R-12.]** 0.875 and 0.751 are withdrawn. The CI printed
> beside 0.875 on this page **excludes it**, because the point estimate is the max of two arm means
> while the interval bootstraps the pooled list (`controls/cave_residstate_decisive.py:258` vs `:265`).
> `REDISTRIBUTE` is emitted by no instrument. The producing script for this artifact is not committed
> and the per-item cache does not exist, so the numbers are unauditable in the R-1 sense.

`:328`'s "attention is sufficient at `-it` (0.875)" must not stand: replace the parenthetical with a
pointer to this addendum, or delete the clause.

## F. `DESIGN_foldlisten_mechanism.md:168,184` (R-12)

Both lines carry the v7 numbers as a *prior* for Phase 2. Append at each:

> **[ADDENDUM 2026-07-30]** The prior is withdrawn (`RETRACTIONS.md` R-12). Phase 2's scope statement
> is unaffected — it never rested on the magnitude — but the sentence may not cite 0.875 / 0.751 or the
> label `REDISTRIBUTE`.

## G. `RESULTS_FOLDLISTEN.md:150` (R-12)

The parenthetical "(REDISTRIBUTE, monitor axis, stands)" is the only load it carries. Append:

> **[ADDENDUM 2026-07-30]** The label is withdrawn; what stands here is the KO's own two
> establishments, listed in the same sentence, which do not depend on it.

## H. `results_fold_vs_listen/FINDINGS.md:7-8` (R-12, R-14)

This file compares its own 0.856 to "v7's 0.875" and concludes `REDISTRIBUTE`. Append:

> **[ADDENDUM 2026-07-30]** The comparator is withdrawn (R-12), and every cell of the artifact this
> file summarises carries `decision.category` = `MOVE_UNMATCHED` ("no verdict"), including one
> restoration above 1.0 (`#models.it.battery.AGAINST_GRAIN.all_attn_write_alllayer` = 1.078249, R-14).
> The head-overlap counts in this file (4/5 base, 5/5 `-it`) are a separate, auditable statistic and
> are unaffected.

---

## §FLAGS

1. **Applied by hand, not by me.** No file above was edited; this is the patch. Anchors are quoted
   from the live files as of 2026-07-30 and should be re-verified at apply time.
2. **What I could not settle.** Which of the two contradicting `-it` artifacts is right
   (`DISTRIBUTED_CONFIRMED` vs `BOTH_REDUNDANT`) is a run, not a reading. Both are committed; neither
   is quotable meanwhile. The honest ledger state for the "does RLHF install the doubt circuit"
   question is **OPEN**, as it was before v6, with the intervening two verdicts withdrawn.
3. **Where my reading is narrower than the readers'.** They described the label-matched re-read as
   returning 0.0/0.0; that is arithmetically what the field says, but `#it_real.ncav` = 0 makes it a
   failed gate. Writing "the label-matched value is 0.0" would be a fresh error, and R-12 says so.
4. **Sites found by grep, all listed above.** `RESEARCH_QUESTIONS.md:174,194,195,204`;
   `POSITION_KNOWING_BEFORE_SAYING.md:310,314,328`; `DESIGN_foldlisten_mechanism.md:168,184`;
   `RESULTS_FOLDLISTEN.md:150`; `results_fold_vs_listen/FINDINGS.md:7,8`. `archive/` hits are
   historical by policy and left alone. `RESEARCH_QUESTIONS.md:204` needs only the pointer, since it
   cites v7's verdict by name rather than by number.
5. **Not swept.** Any figure, caption or draft that inherited 0.875 indirectly. `docs/drafts/` was not
   grepped for paraphrases of the claim, only for the numerals and the label.

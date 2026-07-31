# PATCHSET_tranche4_mech - the MECHANISM / CIRCUIT blocks for POST1's notes

Eight blocks, all against `DARWIN.md_post1_user_notes.md`. Seven correct or scope prose that already
exists; one (T4-M08) is an offer with no anchor. Nothing here touches the intro, the distributional
section (L281-297) or « Sycophancy Scaling Laws » - those are other tranches' territory.

Every number below was read from a result JSON at write time and is quoted as `path#field`. Where a
number I needed was stated in `SNAPSHOT_circuit_groundtruth.md` or `TAXONOMY_withholding.md` I went
to the artifact instead and say so; two of those checks moved a figure (§Preamble note 3).

## Live state, read at write time

| document | md5 | lines (`wc -l`) | split lines | trailing NL |
|---|---|---|---|---|
| `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_notes.md` | `71c3b3c52236520189f0944232c4118a` | 345 | 346 | no |

Same md5 `PATCHMAP_live.md` and `PATCHSET_tranche3.md` were written against, so every tranche-3 line
number still holds. CURRENT text below is sliced from those bytes - **copy the anchors, do not retype
them**. Five of the seven anchors end in a trailing space that is part of the file, and one (T4-M05)
ends in **two**; the trailing-space count is stated under every CURRENT fence because a fence does not
show it. T4-M04's anchor opens with a **tab**.

## Anchor verification, this pass

Each anchor was located by byte compare against the live file, not by eye. All seven occur **exactly
once** in the whole document (`d.count(anchor) == 1`), so uniqueness is positional and not
line-scoped. Character spans, measured this pass:

| block | live line | span | note |
|---|---|---|---|
| T4-M01 | 279 | 29254-29402 | starts after their `["salience copy" or "attention copy"]` bracket, which is untouched |
| T4-M02 | 276 | 28469-28717 | ends at `in the elicited answer.`; the Figure 3 sentence after it is untouched |
| T4-M03 | 274 | 27793-28200 | starts after their `[from our initial mechanistic arc there were some citations?]` bracket, which is untouched |
| T4-M04 | 273 | 27306-27535 | whole line, tab-indented bullet |
| T4-M05 | 272 | 26972-27305 | whole line, bullet |
| T4-M06 | 200 | 19732-19884 | ends exactly where **T3-14**'s anchor begins |
| **T3-14** (pending, not mine) | 200 | 19884-20001 | |
| T4-M07 | 200 | 20002-20105 | begins one space after T3-14's anchor ends |

All eight spans are pairwise disjoint, asserted programmatically. The three L200 spans were then
applied together in a scratch copy with T3-14's PROPOSED text to prove they compose: 8 edits applied,
bytes outside the eight windows unchanged.

## Post-apply invariants, measured on the scratch copy

| | before | after | delta |
|---|---|---|---|
| bracket depth (min / final) | 0 / 0 | 0 / 0 | unchanged |
| NBSP (U+00A0) | 96 | 96 | +0 |
| em-dash / en-dash | 5 / 0 | 5 / 0 | +0 / +0 |
| guillemet pairs | 22 | 22 | +0 |
| curly `’` / straight `'` | 9 / 58 | 9 / 61 | +0 / **+3** |
| tabs | 16 | 16 | +0 |
| split lines | 346 | 346 | +0 |
| `- ` bullet lines | 23 | 23 | +0 |

The three added apostrophes are ASCII straight (`doesn't`, `answer's`, `user's`), matching the
dominant form in this file and in the lines they sit on. No line is added or removed, so **every
tranche-1/2/3 notes line number survives this tranche**.

## Order and interactions

Blocks run notes descending by line number, so a line number is still right when you reach it:

**T4-M01 (L279) - T4-M02 (L276) - T4-M03 (L274) - T4-M04 (L273) - T4-M05 (L272) - T4-M07 (L200) -
T4-M06 (L200)**, then T4-M08 wherever they place it.

Within L200 apply right to left - **T4-M07, then T3-14, then T4-M06** - so character offsets stay
stable for a hand applier. Byte-exact replacement makes the order irrelevant; hand application does not.

Interactions with pending blocks:

- **T3-14 (PENDING, `PATCHSET_tranche3.md:733`)** owns the number on L200. T4-M06 stops one character
  short of its anchor and T4-M07 starts one character after it. **T4-M06 and T4-M07 do not restate
  73 of 74 and must not be edited to.** If T3-14 is dropped, T4-M06/M07 still apply and the line keeps
  the withdrawn 67 - flag that, because T4-M06's replacement sentence makes the stale number more
  prominent, not less.
- **D13 (HELD)** is a strict subset of T3-14's anchor and does not touch mine. Its own RESIDUAL
  (`PATCHSET_tranche2.md:535`) filed the defect T4-M06 discharges: *"`Naming an answer at all turns out
  not to be attention to the user` is a causal reading of an ablation"*.
- **T3-13 (PENDING)** edits L202 at 20438-20552, clear of T4-M07's span.
- **T3-03 (PENDING, NEEDS-RESEARCHER-DECISION, intro L25)**. T4-M04 quotes the same fold-listen head
  overlap T3-03 does. They agree; T4-M04 must not be applied alongside any sentence that reads the
  overlap as evidence for "distributed" (§7.2 of the SNAPSHOT: the number points the other way).
- **T3-15 (PENDING, L192)** prints the base commit denominators 31 / 44 / 50 for the fold arm.
  T4-M05 prints the fold/listen carry counts over 82 and does not restate a denominator, so the two
  do not double-write.
- Nothing here goes near the L319/L321 triple collision, the L317/L323 orphan, or figure renumbering.

## Bracket ledger

Four brackets resolved into prose, none added. Net **-4**, all inside L269-279.

| section | before | after | delta |
|---|---|---|---|
| L199-202 (mechanistic look at folding) | 4 on 2 lines | 4 on 2 lines | 0 |
| L269-279 (raw notes 2) | 9 on 6 lines | **5 on 5 lines** | **-4** |

Removed: `[is that the behaviour we found?]` and `[how can we cite our own results here, thoroughly
and briefly]` (T4-M03), `[is that right? or is this better said as …]` (T4-M02), `[seems to still
exist?]` (T4-M01). **Left standing on purpose**: `[relegated]` (L269 heading tag),
`[from our initial mechanistic arc there were some citations?]` (L274, an external-citation demand,
not mine to answer), `[across what?]` (L277), `[why?]` (L278), `["salience copy" or "attention copy"]`
(L279, naming the mechanism is theirs - `PATCHSET_tranche2.md:905`).

## Disciplines every block obeys

- **RETRACTIONS R-12.** `REDISTRIBUTE`, 0.875 and 0.751 appear nowhere in this file, in prose or
  receipt, and no `-it` substrate verdict is asserted anywhere.
- **RETRACTIONS R-13.** The base doubt-circuit result is never stated without the readout it holds on.
  It appears only in T4-M08, with the `READOUT_SENSITIVE` re-read attached in the same sentence.
- **RETRACTIONS R-14.** From `results_fold_vs_listen*/out/cave_fold_vs_listen.json` only the
  head-overlap counts and the `MOVE_UNMATCHED` gate categories are quoted - both explicitly outside
  the hold. **No battery restoration from that file is quoted anywhere below**, including in receipts.
- **R-3 / the 27b draw.** Every 27b figure in prose names its decode. All 27b-base and 27b-`it`
  numbers in the prose are from the reproducible re-decode (`results_foldlisten_nelicit_*`); the
  committed ext2 values are carried in the receipts as the alternative, never in prose. 2b and 9b were
  re-checked as draw-invariant on every cell used here.
- **The head-SET retraction is quoted at n=41, never at n=6.**
- **Sufficiency is never called measured.** Where a control is a knockout the block says so.

## Three notes on the ground truth I was handed

1. `SNAPSHOT_circuit_groundtruth.md` §6.5 says `W_OV_fro` / `ow_norm` are CHANGED on **5 of 10** 27b
   heads. The artifact says **6**: `results_27b_qk/out/qk_collapse_27b.json#measurements` has
   `W_OV_fro.verdict = CHANGED` at (11,2) +0.3366, (11,4) +0.3367, (11,7) +0.3367, (11,21) +0.3364,
   (17,4) +0.5223 and (23,24) -0.2462, with `ow_norm` CHANGED on the same six. T4-M01 prints 6.
2. The same document's §5 quotes `best_confirm_restore` 0.368385 for the 9b circuit screen. That
   artifact carries **two different values under that name** - `result.best_confirm_restore` 0.368385
   and `result.decision.best_confirm_restore` 0.786392. Neither is cited below.
3. `TAXONOMY_withholding.md`'s free-reply **BOTH = 63** is the *committed ext2* total. Re-derived from
   the item records this pass it is **62 on the reproducible 27b decode** (the 27b-`it` listen cell
   moves 16 to 15). The invariant that matters - every `-chat` free reply scored as withholding names
   both answers - holds on both draws. T4-M02 prints the re-decode figure and names the draw. This
   also discharges R-5's live defect: the receipt below is a fresh derivation from the raw spans, not
   a second citation of the script that never performed the check.

---

# NOTES - `DARWIN.md_post1_user_notes.md`

### T4-M01 - notes L279, "is NOT present in chat models" - RELEGATED BLOCK

ITEM: circuit-audit L279 (`COMPOSE_post1_brief.md:128`)

Their bracket `["salience copy" or "attention copy"]` is upstream of this span and is untouched -
naming the mechanism is theirs (`PATCHMAP_live.md` §4 row 13 territory, `PATCHSET_tranche2.md:905`).
The bracket this block answers is `[seems to still exist?]`, and the answer is **yes, in the weights**.

CURRENT (ends in one trailing space that is part of the file):

````
is NOT present in chat models. Our results show that whilst the mechanism [seems to still exist?] it is not used under exactly the same conditions. 
````

PROPOSED (keep the same single trailing space):

````
is not used in chat models under the same conditions. Present is the wrong word for what tuning does to it: on the ten 27b copy heads we compare across the two variants the query-key product is unchanged on all ten, inside a tolerance of 0.15 and never further than 0.0024 from base, whilst the output side of six of them does change; and the 2b reader head still puts the copied token first in both variants, at the same preference, whilst its realized attention to that token falls from 0.58 to 0.02. The routing survives and stops firing. That is a weights comparison on copy probes rather than on our items, and 9b was never measured either way - so what I can say is that nothing was deleted, not that the same circuit is sitting idle on this task. 
````

RECEIPT:
  `results_27b_qk/out/qk_collapse_27b.json#measurements.*.W_QK_fro.verdict` = **UNCHANGED on all ten
  heads**; `#rel_tol` = 0.15; per-head `rel_change` runs -0.0003 (11,21) to +0.0024 (19,5), re-derived
  head by head this pass. `#measurements.*.W_OV_fro.verdict` = CHANGED on **six** of the ten (see
  preamble note 1), `ow_norm` on the same six. `#heads` = the ten hand-listed 27b heads (11,2)(11,4)
  (11,7)(11,21)(16,3)(17,4)(19,2)(19,5)(19,7)(23,24); `#probe_words` = the / city / of / is / and /
  to / in / a.
  `results_2b/out/rlhf_ovqk_2b.json#decision.verdict` = *"GATING (ARC2A): OV copy survives in weights;
  RLHF gates the QK pattern. FRAMING sec-8 'removed from the weights' is OVERSTATED"* - the artifact
  flags the over-claim itself. `#reader` = head (18,5); `#base.median_rank` 0 = `#it.median_rank` 0;
  `#base.mean_pref` 0.9997 = `#it.mean_pref` 0.9997; `#base.mean_reader_attn` **0.5783** ->
  `#it.mean_reader_attn` **0.0156**, over 5 probe rows.
  Scope, and it is in the prose: weights-only at 27b on hand-listed heads, one reader head at 2b on
  five probes, **9b measured by neither instrument** - so "at any scale" would be the same over-claim
  §8 D7 of the SNAPSHOT names. Corroborating and deliberately not quoted in prose:
  `results_fold_vs_listen/out/cave_fold_vs_listen.json#models.it.decision.attribution_level` =
  `"state-level"`, i.e. head-level attribution at `-it` is where the instrument stops, which is a
  statement about attribution and not about presence.

STATUS: READY - RELEGATED, do not apply if the L269 `### Raw notes and observations analysis 2
        [relegated]` block is cut.
RESIDUAL: The naming choice stays open and stays theirs. Their sentence's subject
  ("the attention heads that implement this fold/listen behaviour") is upstream of my span and still
  asserts that these heads implement fold/listen - T4-M03 is where that claim is corrected, so the
  two blocks should land together or the bullet reads as if the mechanism were established and merely
  unused. If only one of the two is taken, take T4-M03.

---

### T4-M02 - notes L276, the withholding bracket - RELEGATED BLOCK

ITEM: circuit-audit L276 (`PATCHSET_tranche2.md:890-893`)

Their own alternative is the register-accurate one and this block adopts it: the label reads the
**reply span**, not a split in the distribution. It is not accurate on content, though - at `-chat`
those spans do not lack the answers, they name **both** - so the bracket resolves into the corrected
sentence rather than into their exact words.

CURRENT (no trailing space; ends at the full stop):

````
When the probability is split [is that right? or is this better said as "when the free reply doesn't contain the target answers"] - what we describe as "withholding" - the chat model then corrects in almost every case to $C$ in the elicited answer.
````

PROPOSED:

````
When the free reply doesn't resolve to either target answer - what we describe as "withholding" - the chat model then corrects to $C$ in the elicited answer in every listen case there is, 7 of 7 at 2b, 14 of 14 at 9b and 15 of 15 at 27b on the reproducible decode. The label reads the reply span rather than a split in the distribution, and at -chat what earns it is a reply that names BOTH answers: all 61 spans scored that way, over both arms and all three sizes, name $C$ and $W*$ together, so the scorer is declining to pick between two answers rather than recording an absence. In the fold arm the same kind of reply is followed by the pushed $W*$ more often than by $C$, so the correction to $C$ is a listen-arm result.
````

RECEIPT:
  Re-derived from the item records this pass, not taken from any report. Method: take every item whose
  free-reply label is `NEITHER`, then run the repo's own reader over the span -
  `controls/faithful_rescore.py::isolate_span` -> `_norm` -> `_occurrences` for `correct` and for
  `Wstar` (word-boundary, `entity_forms_v2` forms plus the `ALIASES` surface names).
  Label sources: `results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_{2bit,9bit}_ext2_summary.json#items[].faithful_counter`
  and `results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bit_ext2_summary.json#items[].faithful_counter`;
  base cells from the same two directories.
  Counts, reproducible decode: 231 free-reply `NEITHER` labels over the six cells; the `-chat` share is
  **61** (2b-it 9 fold / 7 listen, 9b-it 5 / 14, 27b-it 11 / 15) and **all 61 name both entities on the
  isolated span**. One base item does too (27b-base fold), giving 62 both-namers in all. On the
  committed ext2 draw the same derivation gives 62 `-chat` and 63 in all - the only cell that moves is
  27b-`it` listen, 16 to 15. 2b and 9b are identical across draws, checked cell by cell; the 9b-`it`
  ext2 summary in `results_foldlisten_r2/` carries no `faithful_*` fields and
  `out/faithful_rescore_fl_9bit_ext2.json#fields.counter_gen.items[].new_label` gives the same 5 / 14.
  This reproduces `TAXONOMY_withholding.md`'s BOTH class field for field (2b-base 22 / 13, 9b-base
  56 / 26, 27b-base 25 / 27 CONF+UNC, `-it` 9 / 7 / 5 / 14 / 11 / 16 on ext2) and independently
  confirms R-5's count without re-citing the script R-5 disqualified.
  Elicited outcome after a `-chat` withheld reply, same records, reproducible decode: listen arm C on
  7 / 7, 14 / 14, 15 / 15; fold arm $W*$ on 5 / 3 / 6 against C on 4 / 2 / 5, i.e. **14 to 11 toward the
  push**. On the ext2 draw: listen 7 / 14 / 16 = 37 of 37; fold 15 to 10.
  Register: free reply = `counter_gen` under `classify(map_confidence=True)`; elicited slot =
  `elicit_gen` under `map_confidence=False` (`faithful_rescore.py:88 STRICT_FIELDS`).

STATUS: READY - RELEGATED, do not apply if the L269 block is cut.
RESIDUAL: The third sentence lifts out cleanly if they would rather the bullet stay listen-only - the
  bullet's own opening ("The chat model CONSISTENTLY moves toward the $C$ in the reply") is a listen
  statement, and the fold-arm figure is carried only to stop the sentence reading as if it held in both
  arms. The Figure 3 sentence after my span is untouched and stays true of the listen arm as rewritten.
  Not resolved here: **63 vs 62** is a draw difference, and `TAXONOMY_withholding.md` prints 63 with no
  draw label - that document is owed a dated addendum, which is not this patchset's to write.

---

### T4-M03 - notes L274, "sufficient AND necessary" - RELEGATED BLOCK

ITEM: circuit-audit L274 (`COMPOSE_post1_brief.md:127`)

Two brackets, two answers.

**`[is that the behaviour we found?]` - no.** The controls that bear on it are knockouts, and they
report that the older log-probability readout moves whilst the emitted answer does not; at 9b a
*neutral* control span has a larger realized effect than the answer span, so it is not specific
either. Necessity therefore fails at base on the readout that decides what the model says. Sufficiency
is not something either control can speak to, because neither has an add or force arm.

**`[how can we cite our own results here, thoroughly and briefly]` - by naming the control's decision
and its n inside the sentence, which is what the PROPOSED text does.** Their register is author-year
for the literature with no arXiv IDs, no links and no footnotes (`STYLECARD_researcher.md:265,276-278`),
and it has no form at all for citing an in-tree run. The form that is both thorough and brief in that
register is: say what was ablated, say what moved and what did not, say over how many items. A file
path, a run tag or a JSON field name in the prose would all be out of register, and the notes have no
bibliography an in-tree run could go into. The one route to a checkable citation is the public
notebook the researcher already asks for at L190 - that is theirs, not a fill.

CURRENT (ends in one trailing space; begins after their `[from our initial mechanistic arc …]`
bracket, which stays):

````
The results indicate that there IS an isolated set of attention heads which are both sufficient AND necessary for copying a token from the input to the output [is that the behaviour we found?]. Ablating them prevents the base model from attending to the "salient" input token (either $C$ or $W*$ in our experiments), and proves this mechanism. [how can we cite our own results here, thoroughly and briefly] 
````

PROPOSED (keep the same single trailing space):

````
That is not the behaviour we found. Knocking out every head that attends to the asserted answer's span does move the log-probability difference the older readout is built on, 0.44 at 9b base and 0.84 at 2b, but it does not move the answer the model emits: the argmax comes off $W*$ on none of the 8 caving items at 9b and none of the 7 at 2b, and ablating a neutral span the same way moves it on 3 of those 8, so the effect is not specific to the answer either. At 2b, knocking out even the top 20 target-span-attending heads jointly restores the un-pushed answer on 0.12 of 33 caving items, against the 0.2 the control was written to need. So necessity fails at base on the readout that decides what the model says, sufficiency is not what either control measures - both are knockouts - and the one head SET ever read as installed by chat tuning was retracted on a wider matched sample of 41 items, where its effect runs to -0.64 with an interval that stays below zero. 
````

RECEIPT:
  `results_9b_faithcopy/out/faithful_copy_wstar_9b.json#base.decision` = **M_ONLY**, `m_necessity`
  **0.4418**, `wstar_effect` 0.0, `drop_fires` false, `argmax_fires` false, `argmax_off_frac` **0.0**,
  `control_effects.neutral` **0.375** with `max_control_span` = `"neutral"`; `#base.n_cave` **8**,
  `#base.n_selected` 9, `#pool_size` 61. 0.375 x 8 = the 3 of 8 in the prose
  (`#base.headline.neutral.argmax_off_frac` 0.375). Verbatim `#base.decision.msg`: *"old M-necessity
  0.442 >= 0.3 (the logp-difference moves) but the realized W\*-effect does NOT fire (rel_drop=0.000<0.2,
  argmax_off_frac=0.000<0.2): M moves, the realized output does not -- an overlay on the metric."*
  `results_2b_faithcopy/out/faithful_copy_wstar_2b.json#base.decision` = M_ONLY, `m_necessity`
  **0.836**, `wstar_effect` 0.0, all four `control_effects` 0.0, `#base.n_cave` **7** of 8 selected.
  (`#it.decision.category` = ABSENT - not used in the prose.)
  `results_2b_hsspec_copy/out/cave_headset_specificity_copy_2b.json#base.decision` = **NO_RESTORE**,
  `restore_by_k."20"` **0.1187** against `restore_thr` **0.2**, `n_faithful` **33**; verbatim: *"jointly
  knocking out even the top-20 target-span-attending heads does not faithfully restore the cave"*.
  `results_9b_matched_wide/out/matched_item_deconfound_9b.json#n_matched` **41**,
  `#decision.set.tag` **NO_EFFECT**, `#decision.set.it_eff` **-0.6359**, `#bootstrap_ci.set_it`
  **[-1.1256, -0.2121]**, `#bootstrap_ci.set_it_minus_base` [-1.1992, -0.2212], `#pool_size` 61. The
  n=6 predecessor at `results_9b_matched/…` read `set = INSTALLED`; it is not quoted, per the standing
  discipline.
  **Both controls are knockouts.** `faithful_copy_wstar.json#decision_rule`: *"ko_all = zero ALL heads'
  attention TO a span, renormalize"*; the 2b headset control is a joint-KO K-sweep over {1,3,5,10,20}.
  Neither instrument has an add, patch-in or forcing arm, so neither can measure sufficiency - which
  is why the prose says so rather than reporting a null.
  **Scope caveat that belongs to T4-M04 as well:** the span these controls ablate is the **asserted
  (pushed) $W*$ span in the counter prompt** (`#decision_rule`: *"build the COUNTER prompt (W\*
  asserted)"*). The *initially planted* token of L273 has never been the target of a copy ablation.

STATUS: READY - RELEGATED, do not apply if the L269 block is cut.
RESIDUAL: The sentence upstream of my span - *"There is some evidence for this already in the
  literature [from our initial mechanistic arc there were some citations?] this was both independently
  verified and slightly expanded"* - is now inconsistent with what follows it: what was independently
  attempted was not verified. Rewording it is theirs, and the external-citation bracket inside it is a
  `CITATIONS_post1_verified.md` debt, not a mechanism one. Also standing and untouched: L275's *"this
  same set of attention heads … does NOT control the expression of $C$ or $W*$ in -chat models"*, whose
  `NOT control` is the same weights-versus-behaviour confusion T4-M01 corrects one bullet down.

---

### T4-M04 - notes L273, "a single mechanism … gated on the initially provided token" - RELEGATED BLOCK

ITEM: circuit-audit L273

CURRENT (**opens with a tab**, ends in one trailing space):

````
	- This could plausibly indicate a single mechanism that governs which answer the base model expresses. This mechanism could be gated on whatever the initially provided "plausible" token is, which just gets copied to the output. 
````

PROPOSED (keep the leading tab and the single trailing space):

````
	- This could plausibly indicate a single mechanism that governs which answer the base model expresses, and the head rankings are at least consistent with it: the fold and listen cells share 4 of their top 5 heads at 9b base and again at 2b base, and at -chat they rank the SAME five. That is as far as that run goes. Its matched-move gate failed in all four cells, so the instrument issued no verdict at all, which leaves the base overlap a correlation between two rankings rather than a shared mechanism we have shown - and there is no 27b run of it. The second half of the guess, that the mechanism is gated on whatever plausible token arrived first and just copies it, is the half that has been tested against what the model emits, and it does not survive that test. 
````

RECEIPT:
  `results_fold_vs_listen/out/cave_fold_vs_listen.json#models.base.overlap` = **4** and
  `#models.it.overlap` = **5**; `results_fold_vs_listen_2b/out/cave_fold_vs_listen.json` gives **4**
  and **5** on the same two fields. Head sets (`#heads_fold` / `#heads_listen`): 9b-base
  (25,15)(2,13)(26,7)(23,5)(19,1) against (25,15)(2,13)(26,7)(21,4)(23,5); 9b-`it` the same five
  reordered; 2b-base (16,7)(11,6)(8,3)(16,3)(6,1) against (16,7)(11,6)(16,3)(13,3)(8,3); 2b-`it` the
  same five reordered.
  `#models.{base,it}.decision.category` = **`MOVE_UNMATCHED` in all four cells**;
  `#models.*.move_gate.passed` = **false** in all four; `#models.*.move_gate.delta_flip` = 0.2333
  (9b-base), 0.4417 (9b-`it`), 0.2424 (2b-base), 0.4659 (2b-`it`) against `MOVE_TOL` 0.15. Verbatim,
  identical in form at all four: *"matched-move gate FAILED …: the realized move magnitude is not
  equalized across cells, so the SC-S4 headroom confound is NOT cleared -> no verdict."*
  No 27b run: a tree-wide `find` for `cave_fold_vs_listen*` returns the instrument plus exactly two
  JSONs, 9b and 2b.
  **R-14 compliance:** only `overlap`, the head lists, `decision.category`, `move_gate.passed` and
  `delta_flip` are taken from this file. No battery restoration from it appears here, in prose or in
  this receipt, and the >1 `all_attn_write_alllayer` value R-14 holds is not touched.
  **Reading hazard, stated so it is on the record:** the 5/5 at `-chat` must never be cited as evidence
  that the `-chat` mechanism is distributed - it points the other way, and the same artifact records
  `#models.it.decision.attribution_level` = `"state-level"`, which is where head-level attribution
  stops rather than a claim about spread. SNAPSHOT §4 and §7.2 row 3.
  **Standing scope on both halves:** the heads are ranked in-sample, on the near-margin caving items
  the restorations are then measured on, with no held-out split anywhere in this family
  (`controls/cave_fold_vs_listen.py:483`, SNAPSHOT §2.4). The second half's receipt is T4-M03's, and it
  is deliberately not restated here (MECE) - together with T4-M03's caveat that the copy control
  ablates the *asserted* span, so the "initially provided" token is untested rather than tested and
  falsified.

STATUS: READY - RELEGATED, do not apply if the L269 block is cut.
RESIDUAL: Whether the note keeps the speculation at all now that its second half has been measured is
  theirs. If T4-M03 is not applied, the closing clause here has nothing to point at - **apply T4-M03
  first, or drop the closing clause.**

---

### T4-M05 - notes L272, "wrong ~half the time" - RELEGATED BLOCK

ITEM: circuit-audit L272 (`PATCHSET_tranche2.md:888-889`)

CURRENT (ends in **two** trailing spaces):

````
- The base model is wrong ~half the time, with very similar proportions to when its correct in our previous experiments. These proportions don't hold as such BETWEEN model scales (see Figure 3) but they DO hold across fold vs. listen (start with $C$ and fold to push, or start with $W*$ and fold) for the SAME model, ACROSS scales.  
````

PROPOSED (keep the same two trailing spaces):

````
- The base model is wrong on a minority of the items at every size rather than about half of them: in the listen arm -base names the planted $W*$ at the elicitation on 10 / 34 / 31 of 82 at 2/9/27 billion, against 15 / 41 / 41 for the planted $C$ in the fold arm, the 27b pair read off the reproducible decode. These proportions don't hold as such BETWEEN model scales (see Figure 3) but they DO track each other across fold vs. listen (start with $C$ and fold to push, or start with $W*$ and fold) for the SAME model - and the quantity that is really near-equal across the two arms is how often -base names nothing at all, 51 against 47 at 2b, 38 against 37 at 9b, 34 against 35 at 27b. At 2b it runs the other way in both arms, 16 pushed against 15 planted in fold and 25 against 10 in listen, so the smallest model is not the one carrying its plant.  
````

RECEIPT:
  Faithful (plurals-aware, `map_confidence=False`) register at the elicited slot, 82 items per cell,
  `#decision_faithful.msg` in each summary. `moved` = the pushed answer, `held` = the planted one -
  plant and push invert between arms (`controls/foldlisten_judge.py:454`), so `held` is C in fold and
  $W*$ in listen.
  `results_foldlisten_nelicit_2b9b/out/foldlisten_judge_fl_2bbase_ext2_summary.json` - fold 16 / 15 / 51,
  listen 25 / 10 / 47.
  `…_9bbase_ext2_summary.json` - fold 3 / 41 / 38, listen 11 / 34 / 37.
  `results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bbase_ext2_summary.json` - fold 7 / 41 / 34,
  listen 16 / 31 / 35. **This is the reproducible re-decode** (R-3 SHARPENED). The committed ext2 draw
  of the same cell reads fold 11 / 39 / 32 and listen 20 / 34 / 28, which would print 39 and 34 and 32
  against 28 - carried here, kept out of the prose.
  2b and 9b are draw-invariant: the `results_foldlisten_ext2_2b9b/` and `results_foldlisten_nelicit_2b9b/`
  summaries give byte-identical `decision_faithful.msg` on both cells, checked this pass.
  So: planted answer carried = 15 / 41 / 41 (fold) against 10 / 34 / 31 (listen); nothing named =
  51 / 38 / 34 (fold) against 47 / 37 / 35 (listen), a gap of 4 / 1 / 1. `GROUNDING_crossvariant_scale.md`
  §1's matrix reproduces the same cells on the ext2 draw.
  What the sentence claimed and what replaced it: **`~half` is true in no register.** Over 82 the wrong
  carry is 12% / 41% / 38%; over the items where `-base` commits at all it is 10 of 35, 34 of 45 and
  31 of 47, i.e. 29% / 76% / 66%. The `62% / 46% / 39%` the tranche-2 ledger points at is the *withheld*
  rate at L207/L213, a different sentence. The trailing `ACROSS scales` was dropped because it
  contradicts `BETWEEN model scales` earlier in the same sentence - the same defect
  `GROUNDING_crossvariant_scale.md` §3 names in the twin at L261.

STATUS: READY - RELEGATED, do not apply if the L269 block is cut.
RESIDUAL: One genuine ambiguity I did not resolve for them. `wrong ~half the time` reads two ways:
  as a measurement (the base model gets it wrong on about half the items), which is what the PROPOSED
  text answers and which is false at every scale; or as a statement about the design (half the arms
  plant a wrong answer, so the model is *presented* as wrong half the time), which is true by
  construction and would need a different opening clause - something like *"the base model is handed a
  wrong answer in half the runs"*. If it was the second, take the opening clause from there and keep
  the rest of the PROPOSED sentence. **`its correct` in the CURRENT text is one of theirs and is
  carried out of the file by this replacement rather than corrected** - if they keep any of the old
  clause, keep the spelling.

---

### T4-M07 - notes L200, "whether it answers / which answer it gives" - RELEGATED BLOCK

ITEM: circuit-audit L200 (`PATCHSET_tranche2.md:535`, D13 RESIDUAL)

Apply this **before** T3-14 and T4-M06, so the earlier offsets on the line are still right.

CURRENT (no trailing space; begins one space after T3-14's anchor ends):

````
Whether it answers is a property of the format. Which answer it gives is where the user's turn gets in.
````

PROPOSED:

````
Whether it answers is a property of the format. Which answer it gives is where the user's turn gets in - masked, folding falls to 0.04 / 0.03 / 0.00 at 2/9/27 billion, and it falls just as far when the turn is replaced by padding of the same token length, so this is not a reflex firing on the shape of being challenged. What the mask cannot do is localise: no small head set reproduces it, the read-side search adds no head at any of the three sizes and the best single head moves the fold rate by 0.028 at 9b and 0.027 at the other two, against a 0.03 bar. Nor can blocking every head from a span the answer lives in separate not attending to the user from not being able to see the answer at all.
````

RECEIPT:
  Mask arm, `#arm_rates.fold_mask` against `#arm_rates.fold_nomask`, `n_items` 74 at all three:
  `results_foldlisten_mech_2b/out/foldlisten_phase3c_p3c_2bit_summary.json` 1.0 -> **0.040541**;
  `results_foldlisten_p3c/out/foldlisten_phase3c_p3c_9bit_summary.json` 1.0 -> **0.027397**;
  `results_foldlisten_mech_27b/out/foldlisten_phase3c_p3c_27b_summary.json` 0.918919 -> **0.0**.
  Padding arm, `#arm_rates.padding_fold`, same three files: **0.013699 / 0.013889 / 0.0**, with
  `#a6_report_only.n_length_match_ok` = **74 of 74** at all three (the padded challenge re-encodes to
  exactly the real challenge's content-token count).
  Read side, `#read_side.greedy_fold.selected` = **`[]`** at all three
  (`results_foldlisten_p3a/out/foldlisten_phase3a_p3a_9bit_summary.json`,
  `results_foldlisten_mech_2b/out/foldlisten_phase3a_p3a_2bit_summary.json`,
  `results_foldlisten_mech_27b/out/foldlisten_phase3a_p3a_27b_summary.json`), with
  `#read_side.greedy_fold.trace[0].marginal_drop` = **0.027778** (9b) and **0.027027** (2b and 27b)
  against `#thresholds.GREEDY_MIN_DROP` **0.03**, `READ_TOPK` 10, `SUBSET_MAX` 6.
  `#handle_freeze.category` = FROZEN with both sides `WEAK_AT_DERIVE` at 3/3.
  `#span_stability.category` = `SPAN_STABLE_ALL`, 0 of 370 unstable, at 3/3.
  **Where the padding leg is thinner than the prose implies:** `#a6_decision.category` =
  `CONVERGENT_INSTRUMENTS` at **9b only** (padding 0.0139 against the cited committed floor 0.0270,
  |diff| 0.0131 <= `A6_CONVERGE_ABS` 0.10); at 2b and 27b it is `INSUFFICIENT` because
  `#a6_decision.p2_floor` is null - no `--p2-floor` was cited. The padding *rate* is measured at all
  three; only the convergence verdict is 9b's. That limit is why the prose says "falls just as far"
  and does not claim a verdict.
  The mask itself, so a reader knows it is total: `controls/foldlisten_phase2.py:19` -
  *"KO = mask ALL heads at ALL layers from attending to the CHALLENGE-TURN key positions (attn_scores
  -> -1e9)"*, `MASK_NEG = -1e9` at `:66`.
  SNAPSHOT §7.1 S5's mandatory qualifier is what the last sentence carries: in a decoder-only model
  total-mask-kills-fold is partly information-theoretically forced, so what the ablation establishes is
  redundancy of the read, not localisation of a mechanism.

STATUS: READY - RELEGATED, do not apply if the L199 `### Mechanistic look at folding [relegated (for
        now)]` block is cut. Composes with T3-14 and T4-M06 on the same line; byte-disjoint from both.
RESIDUAL: The fold-arm read gate is decided at 9b and the **listen arm is not** -
  `results_foldlisten_p2/out/foldlisten_phase2_p2_9bit_summary.json#ko_decision_fold` =
  `ATTENTION_READ_GATE` (*"mask rate 0.041 <= drift 0.041 + 0.05; attention to the challenge IS the read
  gate."*) whilst `#ko_decision_listen_secondary` = **`PARTIAL`**, *"numbers only, no claim"*, at a mask
  rate of 0.300. Phase 2 exists at 9b alone. Nothing in the notes says the listen arm was not resolved,
  and no block anywhere carries that. It is a candidate for the same relegated block if they keep it.

---

### T4-M06 - notes L200, the causal reading of the mask - RELEGATED BLOCK

ITEM: circuit-audit L200 (`PATCHSET_tranche2.md:535`, D13 RESIDUAL)

Apply **last on this line**, after T4-M07 and T3-14.

**The number on this line belongs to T3-14 and is not restated here.** This block replaces only the
sentence in front of it - the one that reads a total-mask ablation as a claim about attention to the
user - and hands off mid-clause so T3-14's corrected count reads on unchanged.

CURRENT (ends in one trailing space; ends exactly where T3-14's anchor begins):

````
[Naming an answer at all turns out not to be attention to the user. Mask -chat's attention to the challenge turn so the pushed answer is unreadable and 
````

PROPOSED (keep the opening `[` - it opens the researcher's bracket spanning L200-L202 - and keep the
single trailing space, so T3-14's `it still names an answer on …` continues the sentence):

````
[Naming an answer at all survives losing the user's turn. We mask every head at every layer from attending to the challenge-turn positions, so the pushed answer is unreadable, and 
````

RECEIPT:
  No number is added by this block. The claim it corrects is a reading, not a figure: masking every
  head at every layer from a span removes that content outright, so a behaviour that survives it is
  shown to be *independent of that content*, which is not the same as being shown *not to be
  attention to the user*. The instrument's own words for what it does:
  `controls/foldlisten_phase2.py:19`, quoted in T4-M07's receipt.
  Their reading survives on the corrected count and is left standing - T3-14's PROPOSED text keeps
  *"its own previous one"* and *"answers as though we had agreed"*, and this block does not touch it.
  `PATCHSET_tranche2.md:535` (D13 RESIDUAL) filed exactly this sentence as *"a causal reading of an
  ablation"* and left it, and `PATCHMAP_live.md` §3 records it as UNCOVERED. This is the block.

STATUS: READY - RELEGATED, do not apply if the L199 block is cut.
RESIDUAL: **If T3-14 is dropped, apply this with care.** The replacement sentence makes the clause after
  it more prominent, and that clause still carries the R-6-withdrawn `67 of 74`. Applying T4-M06
  without T3-14 leaves a withdrawn number in a sharpened sentence, which is worse than leaving both.
  Preference order: both, or neither.

---

### T4-M08 - an OFFER, no anchor: the circuit snapshot the notes can carry

ITEM: circuit-audit, synthesis

This is not a fill and has no CURRENT. It is the honest short version of the mechanism work, built
**only** from `SNAPSHOT_circuit_groundtruth.md` §7.1's surviving claims, each carrying the scope
qualifier §7.1 makes mandatory. Placement is theirs - it would sit under a `###` heading of their
choosing, or replace nothing at all.

PROPOSED (an offer, not a fill - see STATUS):

````
At base, a five-head set we rank from the model's own attention to the challenge span both reads and writes the answer it keeps: knocking out those heads' read of that span, and separately replacing what they write, each restore the un-pushed answer far above a matched random-five floor, at 2, 9 and 27 billion, with the heads re-ranked from scratch at every size. That holds on the first-token readout of the base Q/A prompt over the 27 to 37 near-margin items each size contributes, and re-read as a content margin on the same items, the same heads and the same interventions those restorations fall to between one and six times their own random floor - the readout is part of the result and not a detail of it. What the set writes into is not a bottleneck: an attribution screen over all 714 components at 9b puts only 0.29 of its effect in the top fifteen, the only two components that confirm are MLPs rather than heads, and freezing the top five MLP carriers does not block the head effect. At -chat there is no comparable lever for fold and listen - the read-side search returns an empty head set at derivation, the write-direction ablation flips none of the 37 realized answers at 9b and at 2b and moves one the wrong way at 27b, and the same verdict comes back at all three sizes. That last is a necessity result only, because the sufficiency arm was never run.
````

RECEIPT:
  **Sentence 1 (S1).** `results_2b_doubtwvr/out/cave_doubt_write_vs_read_2b_base.json`,
  `results_9b_doubtwvr/out/cave_doubt_write_vs_read_9b_base.json`,
  `results_doubt_27b/out/cave_doubt_write_vs_read_27b_base.json`, field
  `#result.decision.category` = **BOTH** at all three. `#result.attention_ko_restore` 0.282442 /
  0.588508 / 0.481276; `#result.output_patch_restore` 0.322720 / 0.440427 / 0.464987;
  `#result.random_output_restore` **0.034778 / 0.019498 / 0.019537**. Per-scale re-localization:
  `#result.span_ranked_doubt_heads` is a different, architecturally incompatible set at each size
  ((16,7)(8,3)(11,6)(16,3)(13,3) at 2b; (25,15)(2,13)(26,7)(12,2)(23,5) at 9b;
  (25,20)(22,26)(0,6)(22,29)(4,13) at 27b), ranked inside each run at
  `controls/cave_doubt_write_vs_read.py:412,415` with `TOP_K = 5` at `:89`.
  **Sentence 2 (the R-13 qualifier).** `#result.n_faithful` = **33 / 27 / 37** gives the "27 to 37".
  `results_decollide/out/cave_doubt_decollide_{2b,9b,27b}_base.json#result.decision.category` =
  **`READOUT_SENSITIVE` at 3/3**. Content-margin re-read, `#result.readouts.RC`: 2b read 0.037119 /
  write 0.019146 against `mean_random` 0.017612 (2.11x and 1.09x); 9b 0.130187 / 0.050988 against
  0.021552 (6.04x and 2.37x); 27b 0.051624 / 0.037247 against 0.022089 (2.34x and 1.69x). Range
  1.09x to 6.04x - the prose says "between one and six times", computed here.
  **Sentence 3 (S2, S3).** `results_9b_circuit/out/cave_circuit_patch_9b_base.json#result.decision`:
  `conc_frac_at_topk` **0.289136** against `conc_thr` 0.5, `topk` 15,
  `component_class_breakdown.n_confirmed` **2** with `class_counts.mlp` 2 and every attention class 0;
  `#result.n_components` **714**, `#result.n_faithful` 27.
  `results_9b_doubtroute/out/cave_doubt_route_9b_base.json#result.decision`: `block_topk` **0.392331**
  against `block_frac` 0.5, `baseline_restore` 0.588508 -> `restore_with_topk_mlp_frozen` 0.357618,
  category `DIRECT_OR_OTHER`. Both are 9b base on the same 27 items, in-sample ranking - the prose
  says "at 9b".
  **Sentence 4 (S4).** `#greedy.write_drops` = `{wf_to_l: 0.0, wl_to_f: 0.0}` at 9b-`it`
  (`results_foldlisten_p3b_greedy/out/foldlisten_phase3b_p3b_9bit_summary.json`) and 2b-`it`
  (`results_foldlisten_mech_2b/out/foldlisten_phase3b_p3b_2bit_summary.json`); at 27b-`it`
  (`results_foldlisten_mech_27b/…`) `wl_to_f` = **-0.027027**, one item moved the wrong way.
  `#greedy.arm_rates.{wf_to_l,wl_to_f}` = 1.0 against `fold_nomask` 1.0 / `listen_nomask` 1.0 confirms
  0 of 37 flips both directions; `#n_eval` = **37** at all three. `#verdict.verdict` = **`MONITOR_AGAIN`**
  at all three, with identical `reasons`. Read side `#read_side.greedy_fold.selected` = `[]` at 3/3
  (T4-M07's receipt).
  **Sentence 5.** `#verdict.reasons.add_status` = **`"NOT_RUN"`** and `add_both_unmeasurable` = true at
  all three; read-side sufficiency is out of scope by the design's own `decision_rule`.
  **Compliance.** No `-it` substrate verdict, no `REDISTRIBUTE`, no 0.875 or 0.751, no head-overlap
  number, and the word "distributed" does not appear. §7.1's S5, S6 and S7 are not carried - S6 is
  UNAUDITABLE with disjoint base and `-it` sets, S7 is a list rather than a snapshot sentence, and S5
  is already the substance of T4-M07.

STATUS: **NEEDS-RESEARCHER-DECISION.** Three separable decisions, in order of consequence.
  **(a) Whether to carry it at all.** Nothing in the notes currently asserts these five things
  together, so this adds a claim rather than fixing one, which is not what the rest of this tranche
  does.
  **(b) Where.** It is not a fit for the L269 relegated block (that is raw notes) and it is not a fit
  for « under the hood » (that is the distributional layer). Its natural home is a `###` section of
  its own, which is a structural decision this patchset should not make.
  **(c) How much.** Sentences 1-2 stand alone as the base result with its readout. Sentences 3-5 are
  the "and it does not localise further" half and lift out cleanly. If only two sentences are wanted,
  take 1 and 2 - **sentence 2 may not be dropped whilst sentence 1 is kept** (R-13).
RESIDUAL: The heading, if there is one, is theirs - `STYLECARD_researcher.md:206-209`, sentence case,
  no terminal punctuation, no colon-subtitle, and often a full clause with a verb.

---

# The weakest sentence in this set

It is the last clause of T4-M07: *"so this is not a reflex firing on the shape of being challenged."*
The padding arm is the only thing holding it up, and the padding arm is weaker than the sentence
sounds. Its convergence verdict exists at 9b alone - `#a6_decision.category` is `INSUFFICIENT` at 2b
and 27b because no committed floor was cited to it - so at two of the three sizes I am quoting a bare
rate and letting the reader infer the comparison the instrument declined to make. Worse, the control
substitutes padding for the challenge, which removes the challenge's *content* and its *speech act*
in one move; what survives is turn structure and token count, not a challenge stripped of its
content. So the arm shows that the collapse is not a length or position artefact, which is what it was
registered to show, and it does not cleanly isolate "the shape of being challenged" as a separate
thing that failed to fire. A reader who took my clause at face value would think a social-compliance
account had been tested and refuted. It has been made less likely, on one size, by a control built for
a different purpose. If one sentence in this file gets cut on review, it should be that clause; the
rest of T4-M07 stands without it.

Runner-up, and it is close: T4-M01's *"The routing survives and stops firing."* Those are two
measurements on two different models - the query-key result is 27b, the attention collapse is 2b, and
they are not the same heads. The sentence after it says so, but the flat four-word sentence is the one
a reader will carry away, and it reads as a single finding about a single circuit. It is in register
for the corpus and it over-integrates the evidence, and I kept it because the alternative was three
qualified clauses where their own prose would have one.

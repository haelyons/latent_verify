# PATCHSET tranche 4 - the MECHANISM / CIRCUIT claims in the gold lab-notes

Seven blocks against `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_notes.md`. Notes only; the
intro and the distributional sections are other tranches' territory. **Every block below sits inside a
`[relegated]` section whose keep or cut is the researcher's** (`PATCHMAP_live.md` §4 row 13), so every
STATUS says so and no block should be applied to a section that is being dropped.

## Live gold state, measured at write time

| document | md5 | `wc -l` | split lines | bytes |
|---|---|---|---|---|
| `DARWIN.md_post1_user_notes.md` | `71c3b3c52236520189f0944232c4118a` | 345 | 346 | 36631 |

Unchanged from the state `PATCHMAP_live.md` measured (`71c3b3c5…`, 345). The intro is not touched by this
tranche. **Every CURRENT fence below was sliced out of those bytes by the script that generated this file**
(`slice_span`, which asserts each locator key and each resulting span occurs exactly once in the file and
does not cross a newline) - do not retype an anchor, copy it. None of the seven target lines carries an
NBSP, a curly apostrophe or a tab inside the sliced spans, checked at write time; L272 ends in two trailing
spaces and L273/L274/L276/L279 in one, and every span stops short of them.

## Application order

Notes only, descending by line number, so a line number is still right when you reach it:
**T4-M02 (L279), T4-M04 (L276), T4-M01 (L274), T4-M05 (L273), T4-M03 (L272), T4-M06 (L200)**. T4-M08 anchors nowhere.

Within-line ordering, and the one cross-tranche dependency:

- **T4-M03 (a) and (b)** are disjoint spans of L272 - **apply (b) first**, then (a).
- **T4-M06 and pending T3-14** are disjoint spans of L200, both inside the researcher bracket that opens
  at `[Naming` and closes at the end of L202. **Apply T3-14 first** (its span is later in the line).
  T4-M06 does not restate T3-14's count; it rescopes the causal sentence that precedes it. If T3-14 is not
  applied, T4-M06 still applies cleanly - but the line then still carries the R-6-withdrawn 67.
- **No other pending or held block touches any byte in this tranche.** `PATCHMAP_live.md` §3 records
  L269-L279 as "zero pending or held blocks anywhere, fully uncovered", and the only covered line among the
  seven is L200 (D13 ⊂ T3-14, superseded).

Interactions that are not byte collisions:

- **T3-03** (intro L25) states the same 4/5 base, 5/5 -it overlap that T4-M05 states in the notes. Both are
  correct; carrying both means saying it twice in two documents. T3-03 is already
  NEEDS-RESEARCHER-DECISION.
- **D22** (`span` vs `first token`, three sites, §4 row 9) is the layer question T4-M04's fix moves L276
  off, and the layer T4-M01's closing clause names. Neither block pre-empts it: T4-M04 adopts the
  researcher's own wording, and T4-M01's "first-token readout" is what the doubt instrument measures by
  construction, whichever way D22 goes.
- **T3-13 / T3-14** carry the same "do not apply if the block is cut" condition as T4-M06.
- **R-12, R-13, R-14 are binding here.** No block cites `REDISTRIBUTE` or 0.875 / 0.751; every statement of
  the base doubt result names its readout; nothing from the `results_fold_vs_listen` batteries is quoted,
  though that file's head-overlap counts are (R-14 leaves them standing).

## Bracket ledger

| section | live brackets | after this tranche | delta |
|---|---|---|---|
| `### Mechanistic look at folding [relegated (for now)]`, L199-202 | 4 | 4 | **0** |
| `### Raw notes and observations analysis 2 [relegated]`, L269-279 | 9 | 5 | **−4** |
| T4-M08, unplaced | - | 0 added wherever it lands | **0** |

Four resolved into prose, none added: `[is that the behaviour we found?]` and `[how can we cite our own
results here, thoroughly and briefly]` (L274), `[is that right? or is this better said as …]` (L276),
`[seems to still exist?]` (L279). Net **−4**, and every removal takes a matched pair, so whole-file bracket
depth stays min 0 / final 0. Live counts are `PATCHMAP_live.md` §5.4's per-section measurement.

Left standing deliberately: `[from our initial mechanistic arc there were some citations?]` (L274, a
citation demand `CITATIONS_post1_verified.md` owes), `["salience copy" or "attention copy"]` (L279, naming
is theirs), `[across what?]` and `[why?]` (L277-278, not in this brief), and the `[relegated]` heading tag.

## Disciplines every block obeys

- Anchors sliced from bytes, uniqueness asserted, no retyping - the C02 and NBSP defects
  (`PATCHMAP_live.md` §2.1, §5.9) are what this rule exists for.
- Every number is cited to a committed doc's `path#field` or to the artifact that doc names. No number was
  re-derived from an artifact for this file; the two isolated readers' work is cited, not repeated.
- Every 27b-base count names its decode draw, per tranche 3's discipline and commit `1f28ea6`. This is
  load-bearing exactly once: T4-M03 prints 34 and 35 off the re-run rather than 32 and 28 off the committed
  ext2 draw, and says so.
- Register per `PATCHMAP_live.md` §5: their `I`/`we` split (every sentence written here is `we` - setup,
  intervention mechanics, walk-through), spaced hyphen rather than em-dash, no bullets added, British
  spelling, lowercase in-flow brackets, author-year with no arXiv IDs, and typos outside the sliced spans
  preserved - `when its correct` (L272) is deliberately outside T4-M03's span (a).
- Ids follow the brief's own numbering: T4-M01 to T4-M06 are its six line-blocks in its listed order, and
  the offer is T4-M08 because the brief names it that. **There is no T4-M07.**

## The weakest sentence in this tranche

T4-M05's clause *"whilst 2b leans the other way, toward the answer we push, on the 31 fold items where it
names anything at all"*. Every other sentence here restates a decided instrument verdict or a re-derived
count with its register; that one is my own reading of a 15-against-16 split in one arm of a 2x2, on 31 of
82 items, with no instrument that tested directionality at 2b and no interval around it. It is true of the
committed table and it may be nothing but noise, and it is doing rhetorical work - it is the clause that
makes their "single mechanism" story fail at the smallest scale. The rest of T4-M05 stands without it, so
the RESIDUAL says to lift it if they would rather not carry it. The second-weakest is T4-M06's "partly
forced in a decoder-only model", which is `SNAPSHOT` §7.1 S5's own words and is an argument rather than a
measurement; it is flagged as scope, not as a result.

---

### T4-M02 - notes L279, "NOT present in chat models"

ITEM: MECH-2

CURRENT:

````
is NOT present in chat models. Our results show that whilst the mechanism [seems to still exist?] it is not used under exactly the same conditions.
````

PROPOSED:

````
is not used in chat models under the same conditions - which is not the same as its not being there. The routing weights are intact: at 27b the query-key part of ten copy heads comes through tuning unchanged on all ten whilst the output-value part moves on five, and at 2b the copy survives in the weights whilst the pattern that uses it collapses, the reader head's mean attention to the target falling from 0.58 to 0.02 with the copied token still ranked first in both models. We have not measured the weights at 9b, and both cells are hand-listed head sets, so this is a statement about weights at two scales and not about behaviour.
````

ANSWER (their bracket, in their own idiom):
  *[seems to still exist?]* - **yes, in the weights, at the two scales where we compared them.** The
  query-key routing of the copy heads is untouched by tuning and the OV copy still prefers the copied token;
  what collapses is the realised attention pattern. That is why "not used" is the defensible verb and "not
  present" is not.

RECEIPT:
  `SNAPSHOT_circuit_groundtruth.md` §6.5 / §7.1 S7. `results_27b_qk/out/qk_collapse_27b.json#measurements.*.W_QK_fro.verdict`
  = **UNCHANGED for all 10 heads** (`rel_change` −0.0003 … +0.0024 against `rel_tol` 0.15), whilst
  `W_OV_fro` / `ow_norm` are **CHANGED on 5 of 10** (e.g. (17,4) +0.5223, (23,24) −0.2462).
  `results_2b/out/rlhf_ovqk_2b.json#decision.verdict` = *"GATING (ARC2A): OV copy survives in weights; RLHF
  gates the QK pattern. FRAMING sec-8 'removed from the weights' is OVERSTATED"* - `base.mean_reader_attn`
  **0.5783** → `it.mean_reader_attn` **0.0156**, with `median_rank` 0 and `mean_pref` 0.9997 in **both**
  models. Scope carried verbatim from §6.5: weights-only comparison, ten hand-listed 27b heads and one 2b
  reader head (18,5), **9b not measured**, and "no routing edit" is a statement about the `W_QK` Frobenius
  norm, not about attention behaviour, which does collapse.
  Sibling null for the same sentence: §6.1, no head-local installed component at `n_matched` 41
  (`matched_item_deconfound_9b.json#decision`), corroborated by `atp_low_confirm_9b.json#decision.verdict`
  and `rlhf_differential_9b.json#decision.verdict`. `COMPOSE_post1_brief.md:128` files the live sentence as
  "routing weights intact - not-used ≠ not-present"; this block is that fix.

STATUS: READY - **RELEGATED**, do not apply if the `### Raw notes and observations analysis 2 [relegated]`
        block is cut (`PATCHMAP_live.md` §4 row 13).
RESIDUAL: the naming bracket earlier on this line, `["salience copy" or "attention copy"]`, is left standing -
  naming the mechanism is theirs (`PATCHMAP_live.md` §4, `PATCHSET_tranche2.md:905`). Whichever name they
  take, this block's replacement text does not repeat it and needs no edit.

---

### T4-M04 - notes L276, the "probability is split" bracket - their own alternative is right

ITEM: MECH-4

CURRENT:

````
When the probability is split [is that right? or is this better said as "when the free reply doesn't contain the target answers"] - what we describe as "withholding" -
````

PROPOSED:

````
When the free reply doesn't contain the target answers - what we describe as "withholding" -
````

RECEIPT:
  Their alternative is the register-accurate one and is adopted verbatim. `PATCHSET_tranche2.md:890-893`
  files it that way against `EXHIBITS` §R1/§R4. The confirming receipt is `TAXONOMY_withholding.md`: the
  **BOTH** category is a free-reply-slot scorer defect where *"the span names both entities affirmatively
  and the sec-5.6 tie-break abstains (`tiebreak_unresolved`). Verified 63/63 contain both."* - **63 items,
  62 of them -it**, e.g. 9b-it listen `Amsterdam is the capital of the Netherlands. The Hague is … seat of
  the government`. So "the probability is split" is wrong twice over: the label is a matcher abstention on a
  span that names **both** answers rather than neither, and it is a property of the span, not of a
  distribution - nothing in this arm reads a distribution at all. Same file: 145 of the 465 withheld labels
  across both slots are scorer-attributable.

STATUS: READY - **RELEGATED**, do not apply if the block is cut.
RESIDUAL: a second sub-edit is available and not taken here, because it changes their claim rather than their
  wording: after `in the elicited answer` one could add *"- on 63 of those free replies, 62 of them -chat,
  the model names both answers rather than neither, so what the elicited slot does is make it choose"*.
  That is drafted, receipted by the same `TAXONOMY_withholding.md` entry, and theirs to take or leave.
  Note also that this fix moves the sentence off the distribution layer, which is the direction D22 points
  (`PATCHMAP_live.md` §4 row 9); it does not pre-empt D22, whose three sites are L129, L131 and L282.

---

### T4-M01 - notes L274, the "sufficient AND necessary" copy set, and both of its brackets

ITEM: MECH-1

CURRENT:

````
The results indicate that there IS an isolated set of attention heads which are both sufficient AND necessary for copying a token from the input to the output [is that the behaviour we found?]. Ablating them prevents the base model from attending to the "salient" input token (either $C$ or $W*$ in our experiments), and proves this mechanism. [how can we cite our own results here, thoroughly and briefly]
````

PROPOSED:

````
That is not the behaviour we found. Ablating the heads that attend to the target span does not stop the base model producing $W*$ - at 2b, jointly knocking out even the top twenty of them gives a restoration of 0.12 against a matched-random floor of 0.02 on 33 faithful items, which the control decides as no restoration at all, and at 9b the same ablation moves the margin (0.442) whilst leaving the answer the model actually gives untouched (0.000 on both output measures, 8 caving items), where the neutral-span control has a larger effect than the $W*$-span itself. So we have a null on the necessity half and no measurement at all on the sufficiency half. What the same hunt does find at base is a different set - five heads that read the challenge span, whose attention-KO and whose output-patch each restore the un-pushed answer far above a matched-random-five floor at 2b, 9b and 27b, ranked inside each run rather than carried over from 9b - and that result is a property of the first-token readout, since under a stripped content margin the same interventions on the same items fall back to within two to six times the floor.
````

ANSWER (their bracket, in their own idiom):
  *[is that the behaviour we found?]* - **no.** We found the read/write doubt set, not a copy set, and the
  copy set is where our own nulls are. The sentence as written also inverts the direction of the evidence:
  it reads as though ablation *proves* the mechanism, whereas the ablation is the thing that failed.

  *[how can we cite our own results here, thoroughly and briefly]* - the way you already do it at CIRCUIT
  L110, quoted as corpus in `STYLECARD_researcher.md:252`: name the intervention, then the number and its
  n in a parenthetical inside your own sentence - *"our own powered results refute that (copy head L18.H5 →
  0.000 restoration, n=33; the driver is reading the doubt cue, not copying $W*$)"*. Three rules make that
  form carry here. Cite the **intervention and the readout**, not the conclusion, because two of these
  numbers change category when the readout changes. Give **n every time**, because every cell in this
  paragraph runs on 8 to 41 items. And for the mechanistic arc itself, which is not in this post, use a
  forward pointer in your own voice rather than a citation, since there is no author-year to give -
  author-year is reserved for other people's papers, and arXiv IDs, links, footnotes and block quotes stay
  out (`STYLECARD_researcher.md` §A9, via `PATCHMAP_live.md` §5.5).

RECEIPT:
  Necessity: `SNAPSHOT_circuit_groundtruth.md` §6.4 - `results_2b_hsspec_copy/…_copy_2b.json#base.decision.category`
  = **NO_RESTORE**, K=20 restore 0.1187 against `rand` 0.020701, `n_faithful` 33, *"jointly knocking out even
  the top-20 target-span-attending heads does not faithfully restore the cave"*; `results_9b_faithcopy/out/faithful_copy_wstar_9b.json#base.decision.category`
  = **M_ONLY**, *"old M-necessity 0.442 >= 0.3 … but the realized W\*-effect does NOT fire (rel_drop=0.000<0.2,
  argmax_off_frac=0.000<0.2)"*, 8 caves / `n_selected` 9, and `control_effects.neutral` 0.375 against the
  $W*$-span's 0.0. Sufficiency: the only ADD/sufficiency clause anywhere in the mechanism record is the
  phase-3 one, `add_status "NOT_RUN"` / `add_both_unmeasurable true` at 3/3 (§3.6), and read-side
  sufficiency is out of scope by the design's own `decision_rule` (*"attention cannot be forced"*).
  The set that does replicate: §7.1 S1 with its mandatory qualifier - `cave_doubt_write_vs_read_{2b,9b,27b}_base.json#result.decision.category`
  = **BOTH** at 3/3, `random_output_restore` 0.0348 / 0.0195 / 0.0195, heads re-ranked per run
  (`cave_doubt_write_vs_read.py:412,415`; §1.2) - and its readout scope, `RETRACTIONS.md` R-13 /
  §1.4: `cave_doubt_decollide_{2b,9b,27b}_base.json#result.decision.category` = **READOUT_SENSITIVE** at
  3/3, RC write 0.0191 / 0.0510 / 0.0372 against floors 0.0176 / 0.0216 / 0.0221.
  Installed-set null, for the sub-bullet that follows: §6.1, `matched_item_deconfound_9b.json#decision`
  at `n_matched` **41**, set=NO_EFFECT, bootstrap `set_it` −0.6359 CI [−1.1256, −0.2121].

STATUS: NEEDS-RESEARCHER-DECISION - **RELEGATED**: `### Raw notes and observations analysis 2 [relegated]` is
        one of the six headings whose keep/cut is theirs (`PATCHMAP_live.md` §4 row 13). The decision this
        block cannot take for them is whether the corrected sentence is worth carrying at all, since what it
        now says is a null.
RESIDUAL: the earlier bracket on this line, `[from our initial mechanistic arc there were some citations?]`,
  is a citation demand on the literature and is not touched here - `CITATIONS_post1_verified.md` owes it an
  entry before it can be dissolved. The naming of the five-head set is theirs.

---

### T4-M05 - notes L273, the "single mechanism … gated on the plausible token" reading

ITEM: MECH-5

CURRENT:

````
This could plausibly indicate a single mechanism that governs which answer the base model expresses. This mechanism could be gated on whatever the initially provided "plausible" token is, which just gets copied to the output.
````

PROPOSED:

````
This is a pattern across the two arms rather than a mechanism we have isolated. At 9b the base model expresses whichever answer the transcript seeded first - $C$ on 41 of 82 in fold, $W*$ on 34 of 82 in listen - and 27b runs the same way, whilst 2b leans the other way, toward the answer we push, on the 31 fold items where it names anything at all. The instrument that ranks heads on both arms puts four of the same five at base and five of five at -chat, so the overlap is no smaller in the tuned model, and it issues no verdict on any of its four cells because the size of the move is not matched across them - which leaves the shared heads correlational. Whether the seeded token is copied is a separate claim, and the copy ablations in the sub-bullet below do not support it.
````

RECEIPT:
  Seeded-answer pattern: `JOIN_withhold_vs_fold.md` §(1), elicited slot, strict, n=82 a cell. Base rows -
  fold C 15 / 41 / 39 and $W*$ 16 / 3 / 11; listen C 25 / 11 / 20 and $W*$ 10 / 34 / 34, at 2b / 9b / 27b.
  So 9b and 27b express the seeded answer more often than the pushed one in both arms; 2b does not (fold
  15 C against 16 $W*$ on the 31 items it names anything on, listen 25 C against 10 $W*$).
  Head overlap: `SNAPSHOT_circuit_groundtruth.md` §4 - `results_fold_vs_listen/out/cave_fold_vs_listen.json#models.base.overlap`
  = **4** and `#models.it.overlap` = **5** at 9b, same at 2b (`results_fold_vs_listen_2b/…`); `-it` fold and
  listen rank the *same five heads*, reordered. All four cells `#models.{base,it}.decision.category` =
  **MOVE_UNMATCHED**, message *"the SC-S4 headroom confound is NOT cleared -> no verdict"*, and
  `#models.base.move_gate.passed` = **false** in both files, i.e. base is correlational only.
  **`RETRACTIONS.md` R-14 is respected**: nothing from that artifact's battery is quoted here. R-14 holds
  the battery restorations (one is 1.078249) and explicitly leaves the head-overlap counts standing.
  What this block refuses to write, per §7.2: "fold and listen share one circuit" (no verdict was issued,
  and the shared late-layer DLA overlap deflated to `GENERIC_ANSWER_FORMATION`), and any use of the overlap
  as evidence that -chat is "distributed" - the number points the other way.

STATUS: READY - **RELEGATED**, do not apply if the block is cut.
RESIDUAL: if T3-03 lands in the intro, the 4/5 and 5/5 overlap is then stated twice in two documents; the notes
  copy is the one that can carry its `MOVE_UNMATCHED` scope in-line, so if they want it once, cut it here or
  there deliberately rather than by accident. The 2b clause is the weakest sentence in this tranche (see the
  preamble) and lifts out cleanly if they would rather not carry it.

---

### T4-M03 - notes L272, "wrong ~half the time", per scale, with the register and the decode draw

ITEM: MECH-3

(a) CURRENT:

````
The base model is wrong ~half the time
````

(a) PROPOSED:

````
The base model names neither answer about half the time - 51, 38 and 34 of 82 at 2b, 9b and 27b, the 27b count on the re-decoded draw
````

(b) CURRENT:

````
for the SAME model, ACROSS scales.
````

(b) PROPOSED:

````
for the SAME model, ACROSS scales. Withheld runs 51 against 47 at 2b, 38 against 37 at 9b and 34 against 35 at 27b for fold and listen - elicited slot, strict register, 82 items a cell - so the two arms agree inside a model whilst the scales do not agree with each other. None of that is attributable to the push. Elicit the same base cells with no pushback at all and they withhold 35, 52 and 57 of 82, which is more at 9b and 27b, not less.
````

RECEIPT:
  Pushed column: `JOIN_withhold_vs_fold.md` §"Reproduction of the claim's own headline" - withheld =
  NEITHER + UNRESOLVED_ALIAS, fold arm, elicited slot, strict - **base 51 / 38 / 32**, **-it 0 / 0 / 1**,
  re-derived span by span with **0 disagreements** on all 12 cell×arm combinations (984 items); its §(1)
  2×2s give the listen-arm base totals 47 / 37 / 28. `GROUNDING_notes_numbers.md` §L207 prints the same
  51 / 38 / 32 = 62% / 46% / 39% and is the source `PATCHSET_tranche2.md:888-889` files against this line.
  **Decode draw, and why 27b reads 34 and 35 here, not 32 and 28**: JOIN reads the committed
  `results_foldlisten_ext2_27b/…_27bbase_ext2_summary.json`, i.e. the ext2 draw that `PATCHSET_tranche3.md`'s
  own discipline names as the identified 27b-base anomaly; the re-run family gives the same slot as
  **34 (fold) and 35 (listen)** - `GROUNDING_neutral_elicit.md` §2, push column, `27b-base fold 34→57` and
  `27b-base listen 35→55`. 2b and 9b are identical across the two families (51/47 and 38/37), which is the
  draw-invariance the discipline states for those scales. Commit `1f28ea6` repointed every 27b figure to
  the reproducible decode; this block follows it.
  Neutral arm: `GROUNDING_neutral_elicit.md` §1 - base neutral-elicited withheld **35 / 52 / 57** of 82 -
  and §2, `withhold_verdict` = PARTIAL (2b) / **INVERTED_NEUTRAL_HIGHER** (9b, Δ −0.171) / **INVERTED**
  (27b, Δ −0.280), `push_attribution` re-derived by hand with 0/112 mismatches.

STATUS: READY - **RELEGATED**, do not apply if the block is cut. Sub-edits (a) and (b) are disjoint spans of
        the same line; **apply (b) first** so (a)'s offsets are still right when you reach it.
RESIDUAL: two things this block does not decide. **(i)** "wrong" is read here as the withheld column, which is the
  reading `GROUNDING_notes_numbers.md` §L207 and `PATCHSET_tranche2.md:888-889` adopt, and it is the only
  reading on which their own next clause is true - fold and listen agree inside a model for withheld
  (51/47, 38/37, 34/35) and do not agree for named-answer counts, which invert between the arms at 9b and
  27b. If they meant "does not produce $C$", the numbers are the $C$ rows of `JOIN` §(1) instead (15 / 41 /
  39 of 82 in fold) and the fold-versus-listen half of their sentence stops being true. **(ii)** their typo
  `when its correct` sits immediately after span (a) and is deliberately outside it, preserved.

---

### T4-M06 - notes L200, the causal reading of the total-mask ablation - composes with T3-14

ITEM: MECH-6

CURRENT:

````
Naming an answer at all turns out not to be attention to the user.
````

PROPOSED:

````
Naming an answer at all is not gated on reading the user's turn - a total mask is the only handle we have on this, so what it establishes is that the answer is redundantly available rather than that attention plays no part, and the same mask collapses folding at all three scales, which is partly forced in a decoder-only model that cannot fold to an answer it cannot read.
````

RECEIPT:
  Scope from `SNAPSHOT_circuit_groundtruth.md` §7.1 **S5**, verbatim: *"in a decoder-only model
  total-mask-kills-fold is partly information-theoretically forced; what it establishes is the redundancy,
  plus the death of content-free social compliance"*, over §3.1 + §3.6 - the total-mask read gate
  (`fold_mask`) is auditable at 3/3 with 0.0274 (9b-it) / 0.0406 (2b-it) / 0.0 (27b-it), and no sparse head
  subset moves folding (`foldlisten_phase3a_*_summary.json#greedy_fold` selected = `[]` at 3/3,
  `WEAK_AT_DERIVE` both sides).
  **The number on this line is not restated here.** It belongs to pending **T3-14**, which replaces the
  R-6-withdrawn "67 of 74" with 73 of 74 off `results_foldlisten_p2/out/foldlisten_phase2_p2_9bit_summary.json#arm_counts.fold_mask`.
  That is a *phase-2* count at 9b-it over the 74-item family; the §3.6 numbers above are the *phase-3c*
  gate. Different files, do not merge them into one figure.

STATUS: READY - **RELEGATED**, do not apply if the `### Mechanistic look at folding [relegated (for now)]`
        block is cut (`PATCHMAP_live.md` §4 row 13, the same condition T3-13 and T3-14 carry).
RESIDUAL: this span and T3-14's are disjoint spans of the same line and both sit inside the researcher bracket that
  opens at `[Naming` and closes at the end of L202, so the paragraph stays bracket-held draft prose either
  way - the bracket count on the line does not change. **Apply T3-14 before this block**: its span is later
  in the line, so editing it first leaves this one's offsets untouched. Their own two closing sentences
  ("Whether it answers is a property of the format. Which answer it gives is where the user's turn gets
  in.") already say the useful half and are not touched.

---

### T4-M08 - notes, unplaced, the circuit snapshot the notes can honestly carry

ITEM: MECH-8

This is an **offer, not a fill**. It anchors nowhere; placement is theirs, and the two candidate sites are
the head of `### Mechanistic look at folding [relegated (for now)]` (L199) and the head of `### Raw notes
and observations analysis 2 [relegated]` (L269) - both inside blocks whose keep/cut they own.

PROPOSED (an offer, not a fill - see STATUS):

````
At base the same hunt finds a five-head set that reads the challenge span and writes toward the pushed answer - knocking out its attention to that span and replacing its output each restore the un-pushed answer far above a matched-random-five floor, at 2b, 9b and 27b, with the heads ranked inside each run rather than carried between scales - and that result is a property of the first-token readout, since under a stripped content margin the same interventions on the same items fall back to within two to six times the floor. Downstream of that write there is no bottleneck: at 9b an attribution screen over all 714 components leaves less than a third of its effect in the top fifteen of the 27 items it ranks them from, the two components that confirm are MLPs rather than heads, and freezing the top five MLP carriers does not block the restoration. At -chat we find no single causal lever for taking the user's answer - the read side is empty at derivation, resample-ablating the write direction flips none of the 37 answers at 9b or 2b and moves one the wrong way at 27b, and the arbiter disagrees on sign - so on that one 74-item family the verdict is monitor, not lever, at all three scales, and on the necessity half only, since the sufficiency arm was never run. Masking every head's read of the challenge turn does collapse folding at all three scales, which says the read is necessary and redundant rather than localisable. What we do not have is an installed component: on matched items at 9b no deference head or head set survives at n=41, there is no entropy or confidence neuron and no confidence gate on caving at that scale, and copying the pushed answer is not the driver on the four to fourteen items each of those cells carries.
````

RECEIPT:
  Built only from `SNAPSHOT_circuit_groundtruth.md` §7.1, one sentence per surviving claim, each carrying
  the qualifier §7.1 marks mandatory. Sentence 1 = **S1** with the §1.4 / `RETRACTIONS.md` R-13 readout
  qualifier attached in prose (`READOUT_SENSITIVE` at 3/3; RC restorations 1.09× to ~6× their matched-random
  floors). Sentence 2 = **S2** (`results_9b_circuit/out/cave_circuit_patch_9b_base.json`, `conc_frac_at_topk`
  0.289136, confirmed components 2, both MLPs) and **S3** (`results_9b_doubtroute/…`, `block_topk` 0.392 <
  `BLOCK_FRAC` 0.5), with S2/S3's "9b base, 27 items, in-sample ranking" scope in the sentence. Sentence 3
  = **S4** with its whole qualifier (necessity only, `add_status NOT_RUN`, single 74-item family; the
  zero-of-37 flips and the 27b anti-flip from the three phase-3b summaries). Sentence 4 = **S5** with its
  information-theoretic scope. Sentence 5 = **S7**, scoped 9b.
  **What it deliberately does not say**, per §7.2 and `RETRACTIONS.md`: no `REDISTRIBUTE`, no 0.875 / 0.751
  (R-12, withdrawn); no "distributed at -chat" and no head-overlap number used as evidence for it (§7.2 row
  3, §4 - overlap is 5/5 at -it against 4/5 at base and points the other way); no "head-specific at all
  three scales" (§7.2 row 1); no "fold and listen share one circuit" (§7.2 row 4); no 2b attribution-graph
  claim (N=1, §7.2 row 7); nothing from the `results_fold_vs_listen` batteries (R-14). It carries zero
  brackets, so wherever it lands the bracket ledger is unchanged.

STATUS: NEEDS-RESEARCHER-DECISION - an offer. Placement is theirs, and so is whether a snapshot of this
        kind belongs in the notes at all rather than in the circuit write-up.
RESIDUAL: if they place it under `### Mechanistic look at folding`, sentence 4 sits next to T3-14's mask
  count and should lose its "at all three scales" clause to avoid reading as a second, larger count; if they
  place it at L269 it stands as written. Five sentences; the brief allowed three to five, and the shortest
  honest version drops sentence 2, which is the one claim here that is a single-scale in-sample result.

---

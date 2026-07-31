### T4-D01 - notes L335, what makes $W*$ plausible and which slot shows it

ITEM: D1

CURRENT:

````
I chose plausible wrong counterfacts $W*$ based on a rough personal estimate of how plausible I thought the alternative was. Measuring the model assigned probability of $W*$ in the neutral control shows that the ones picked are typically [in the top 3 next answers, with other alternatives being respellings of the same words or phrases [what evidence is there for this? are there any clear examples we could pull-out?]] [at the neutral slot it is not - $W*$ sits at a median rank of 119 there and 3 at the question on its own, which is the slot that shows this]
````

PROPOSED:

````
I chose plausible wrong counterfacts $W*$ based on a rough personal estimate of how plausible I thought the alternative was. Measuring the model assigned probability of $W*$ shows the picks are typically in the top 3 next answers, but only when the question is asked on its own - at 9b -base $W*$ sits at a median full-vocabulary rank of 3 there, and once the neutral control turn is in the context that median is 119, so the neutral arm is the right place to measure a movement baseline and the wrong place to measure plausibility. The respellings are visible in the same read. Asked which city is the most populous in Turkey with nothing planted, the three most probable first tokens are « Istanbul » at 0.891, « İstanbul » at 0.030 and « istanbul » at 0.021 before « Ankara » arrives, which is why Ankara reads rank 4 raw and rank 2 once the two respellings are collapsed. The same bare-question read gives a median rank of 3 at 2b -base and 4 at 27b -base, and at -it it gives nothing usable.
````

RECEIPT:
  `results_absdecode_ext2/out/family_topk_shift_vfam_ext2_9bbase.json`, `result.items[]`, 9b -base fold,
  first-token readout at the answer slot, leading-space key (the canonical key at every base cell,
  `results_fmt_2b9b/out/family_topk_shift_fmt_fmt_ext2_9bbase.json#result.items[0].key_canonical` = `"space"`).
  Median full-vocab `rank_w_bare` / `rank_w_neutral` = **3 / 119** at 9b -base; the bare-slot median is
  **3 / 3 / 4** at 2b / 9b / 27b -base and the neutral-slot median **35 / 119 / 80** - all six read out of
  `INVENTORY_distributional.md` §3.2's median-rank table (`rank_c_*` / `rank_w_*`), which is where the 119 and
  the 3 in B01's landed bracket come from. The three Istanbul spellings and their probabilities are
  `INVENTORY_distributional.md` §2.6 item 1, taken from the same file's `result.items[0].topk_bare`
  (`" Istanbul"` 0.891233, `" İstanbul"` 0.030496, `" istanbul"` 0.020960) and frozen in the figure build's
  own `EXPECT` block, `docs/drafts/figs/make_fig_topk_ankara.py:75-97`. `rank_w_bare` = 4 and the collapse to 2
  are §2.6 item 2, which also confirms the '9b -base only' scope B01 wrote.
  The closing clause is `INVENTORY_distributional.md` §4.1: at -it the instrument keys `first(" " + W*)` with no
  `is_chat` branch, the leading-space share of the bare top ten is 0.081 / 0.121 / 0.162 against 0.976 / 0.984 /
  0.965 at base, and every -it absolute-probability, top-K or plausibility statement from this instrument is
  blocked. Draw: no draw label applies - these are teacher-forced probability reads, not decodes.
  Register: first-token (TK), not the content margin and not a span probability.
  This block replaces B01's applied fill, which is a previous agent's text and not the researcher's
  (`PATCHMAP_live.md` §2.3 discipline).

STATUS: READY
RESIDUAL: the `[what evidence is there for this? are there any clear examples we could pull-out?]` question is answered in prose with a named example rather than left standing; if they would rather the Turkey spellings sat in « under the hood » beside T4-D06 than here, the third and fourth sentences lift out as a pair. Bracket delta on this line: **-3** (their outer plausibility bracket, its nested question, and B01's landed scope bracket all dissolve into prose).

---

### T4-D02 - notes L312-L313, the -it pushback bullet and its probability reading

ITEM: D2

CURRENT:

````
- -it models OVERWHELMINGLY "pushback" with the correct "$C$" when seeded with the incorrect $W*$. 
	- this is plausibly the assigning a higher probability to $C$ than $W*$, and rather than copying the token from its input, it pushes back with this higher probability (that we know as correct) answer.
````

PROPOSED:

````
- -it models OVERWHELMINGLY "pushback" with the correct "$C$" when seeded with the incorrect $W*$, on the reply and elicited labels the sankeys count. 
	- the tempting reading is that the model assigns a higher probability to $C$ than to $W*$ and answers with the higher one rather than copying the token from its input, and the distribution does not carry that reading - the listen-arm distributional column is withdrawn at all six cells, so for the seeded-$W*$ direction there is no probability read to appeal to at 9b at all, and the one listen read that survives, top-K at 27b, moves the pushed answer from a median rank of 2094 to 72 whilst the planted answer falls from 1399 to 1907. The distribution follows the user in both directions. Where the fold arm can be read the same claim is slot-dependent rather than true: at 9b -it the content margin favours $C$ on 72 of 82 at the bare question and 75 of 82 after a neutral turn, and on 27 of 82 once the push is in the context, with 2b -it at 55, 66 and 18 and 27b -it at 70, 75 and 39.
````

RECEIPT:
  Slot table: `INVENTORY_distributional.md` §3.1, columns (a)/(b)/(c) = `M0` / `Mc_neutral` / `Mc_counter`
  counted `> 0` over 82 items - 9b -it 72 / 75 / 27, 2b -it 55 / 66 / 18, 27b -it 70 / 75 / 39, from
  `results_itreadout_modelw/out/family_cave_diagnose_vfam_ext2_9bit.json` and the two
  `results_r1_dist_2b9b` / `results_r1_dist_27b` twins. Content margin (CM) register: teacher-forced,
  polarity-stripped, pairwise - NOT a first-token probability and NOT an argmax.
  Listen withdrawal: `INVENTORY_distributional.md` §4.3, `out/cleangate_same_box_result.json`
  `decision = TOPK_NEUTRAL__DIAGNOSE_NOT_NEUTRAL__B1_LISTEN_WITHDRAWN` - every listen number from
  `family_cave_diagnose_arms` is withdrawn at all six cells, including the four whose own identity gate passed.
  The surviving listen read is `family_topk_shift_arms`
  (`results_cleangate_27b/out/family_topk_shift_arms_vfam_ext2_27bit.json`, gate `ALGEBRAICALLY_NEUTRAL`);
  the 2094 -> 72.5 and 1399 -> 1906.5 medians are §2.3's last block, rounded to whole ranks in the prose
  because they are medians of 1-indexed integer ranks.
  Two disclosures the prose carries by construction: the -it rows compare across slots inside a cell and not
  against the base rows (§4.2, the leading-space penalty differs between $C$ and $W*$), and no -it absolute
  probability is quoted here at all.
  Deliberately NOT claimed: any generation-layer count for the listen arm at -it. `TAXONOMY_withholding.md`
  gives listen-arm -it withheld 0 / 0 / 0, which is a withholding count and not a C-versus-$W*$ split, and the
  elicited listen split is not re-derived here.

STATUS: READY
RESIDUAL: the word OVERWHELMINGLY is left as theirs and only the layer it is true at is named. The scale-ordering guard T3-09 lands two lines above this pair at L301; if T3-09 is cut, the `2b -it at 55, 66 and 18` clause is the only place in the section that still says the -it column is not flat across scale. Bracket delta on these two lines: **0**.

---

### T4-D03 - notes L310, base carrying an incorrect scripted fact

ITEM: D3

CURRENT:

````
- base models ALSO carry an INCORRECT scripted fact through to the answer. 
````

PROPOSED:

````
- base models ALSO carry an INCORRECT scripted fact through to the answer, on the listen arm where $W*$ is the planted turn rather than the pushed one, and on the part of the family that answers at all - at the elicited slot 47, 37 and 28 of 82 name neither candidate at 2b, 9b and 27b -base. 
````

RECEIPT:
  `TAXONOMY_withholding.md`, headline: "So does a series the drafts never print: **listen-arm base 47 / 37 / 28,
  -it 0 / 0 / 0**", elicited slot, strict register (`map_confidence=False`), matcher
  `family_generate_judge._norm` over `faithful_rescore.isolate_span`. Re-derivation script named in that file:
  `docs/drafts/taxonomy_withholding_rederive.py` (print-only, no writes, no model). The same table's listen rows
  are `2b-base listen 47`, `9b-base listen 37`, `27b-base listen 28` of 82.
  Layer: this is the string-matched generation layer, not a distributional read. It has to be, because
  `INVENTORY_distributional.md` §4.3 withdraws the listen-arm distributional column at all six cells and §1b
  records that `family_topk_shift_arms` was never run below 27b - so at 2b and 9b there is no listen
  distribution of any kind, and the count above is the only thing the sentence can be grounded on.
  Deliberately NOT claimed: that the remaining 35 / 45 / 54 items carry $W*$ through. The complement of
  "withheld" is "named a candidate", which splits into $C$ and $W*$, and that split is not re-derived here.

STATUS: READY
RESIDUAL: the sub-bullet at L311 is covered twice over (D06 applied, T3-06 pending to replace it) and is untouched by this block. If T3-06 lands, its three-scale prose and this line's three-scale count sit one line apart and should be read together for repetition. Bracket delta: **0**.

---

### T4-D04 - notes L309, what -base abstention is made of

ITEM: D4

CURRENT:

````
- -base models overwhelmingly abstain from the user push, or maintain the correct fact into the final elicitation. 
````

PROPOSED:

````
- -base models overwhelmingly abstain from the user push, or maintain the correct fact into the final elicitation, and « abstain » is one label over three different behaviours whose mix moves with scale - at the elicited slot 2b -base withholds on 51 of 82 and 39 of those assert confidence whilst naming nothing, 9b -base withholds on 38 with 20 of those a genuine « I don't know. », and 27b -base withholds on 32 with 30 of those answering a different question. 
````

RECEIPT:
  `TAXONOMY_withholding.md`, headline table and the twelve-cell elicited table: 2b-base fold withheld 51, CONF
  39; 9b-base fold withheld 38, UNC 20; 27b-base fold withheld 32, OFFTGT 30. That file's own summary of the
  same three rows is "76% asserted confidence" / "53% genuine uncertainty" / "94% off-target". The committed
  counts it reproduces are elicited fold base 51 / 38 / 32 and -it 0 / 0 / 1 at 2b / 9b / 27b.
  Category definitions quoted from the same file: CONF asserts certainty and names no entity; UNC is explicit
  uncertainty or decline and is the only genuine-uncertainty category; OFFTGT is other content naming no
  candidate. Totals across the 234 elicited withholds: CONF 102, UNC 34, THIRD 41, OFFTGT 34, NUM 14, AGREE 4,
  FMT 4, MISS 1 - and 33 of the 34 UNC are 9b -base, which is why the hedge reading is a 9b reading (the same
  fact D07 landed and T3-08 rewrites two lines above at L307).
  Register: elicited slot, strict (`map_confidence=False`), fold arm. Scale is quoted as an ordered triple.
  Deliberately NOT written into the line: the neutral-elicit inversion. "abstain **from the user push**" is a
  causal attribution and `GROUNDING_neutral_elicit.md` §2 refutes it (52 of 82 withheld with no push against 38
  with one at 9b -base; 57 against 34 at 27b -base). That correction is carried once, at L131, by T4-D11 -
  writing it here as well would put the same number in two sections.

STATUS: READY
RESIDUAL: the causal half of their sentence is left standing here on purpose and is answered at L131. If they take T4-D11's option (b) they may want a four-word back-reference on this bullet; if they take option (a) this bullet is the only place the attribution is asserted and it should be softened here instead. Bracket delta: **0**.

---

### T4-D05 - notes L297, the forced-final slot and the -chat half of their question

ITEM: D5

CURRENT:

````
[why do we need to pick an alternative that exists in the distribution? doesn’t the attention copy mechanism in base work irrespective of that? what about in -chat?]
````

PROPOSED:

````
[why do we need to pick an alternative that exists in the distribution? doesn’t the attention copy mechanism in base work irrespective of that? what about in -chat?]

Every distributional run in this section builds three prompts - the bare question, the neutral turn and the counter turn - and no fourth, so none of them reads at the slot the sankeys actually score. The span-decomposition run at 9b -it is the first that does. It persists the whole first-token distribution twice per record, once at the counter reply and once at the forced final, on the leading-space and the no-space key both, so the key defect that kills every other -it distributional number does not bite here - and it runs over the 74-item mechanism family rather than the 82 this post counts over. The two slots do not look alike. At the counter reply the most probable token is « You » on 74 of 74 and neither answer is anywhere near it, which is the same discourse-opener result the top-K readout gives at every other cell. At the forced final the most probable token is an answer entity on almost every item, and which entity it is tracks the last thing the user asserted - pushed toward $W*$ the wrong answer is the top token on 70 of 74 and the correct answer on none of them, and pushed toward $C$ instead the correct answer is top on 69 of 74. The elicitation instruction does not read out a standing preference. It reads out the most recent assertion. At every other cell the forced final still has no distributional read at all.
````

RECEIPT:
  NEW ARTIFACT, and it amends a disclosure three earlier ledgers state as absolute.
  `out/foldlisten_demarez_subst_dmz_9bit_a_summary.json` (Run A, token-span SUBSTITUTION, HOOK-FREE;
  `name` google/gemma-2-9b-it, `cell` fold, `regime` chat, `tag` dmz_9bit_a, `n_items_measured` 74,
  592 records = 8 arms x 74 items, A100-SXM4-40GB, git 0105d18, finished 2026-07-30T20:33Z), joined by
  `out/demarez_join.json`. Per-record path `items[].distributions.{counter_first,elicit_first}`, each carrying
  `topk_10`, `argmax_tok_id`, `argmax_tok_str`, `reads_c_space`, `reads_c_bare`, `reads_w_space`,
  `reads_w_bare`, `margin_first_{space,bare}`, `margin_sign_{space,bare}`. `dist_contract.verdict` =
  `DIST_FIELDS_COMPLETE` over 1184 arm x position records, `n_entkey_underflow` 0, `n_margin_undefined` 0.
  Re-derived this session over `arm == "A1"` (74 records; A1's template is byte-identical to
  `PUSH['counter']`, `job_truthful_flip.py:50`): at `counter_first`, `argmax_tok_str` = `"You"` on **74/74**,
  `reads_c_bare.rank_first_tok == 1` on **0/74**, `reads_w_bare.rank_first_tok == 1` on **0/74**. At
  `elicit_first`, `reads_w_bare.rank_first_tok == 1` on **70/74**, `reads_c_bare.rank_first_tok == 1` on
  **0/74**, `margin_sign_bare` = -1 on **74/74**, median `rank_c_bare` 3 and `rank_w_bare` 1. Over
  `arm == "A8"` (push-toward-stated, and stated = $C$ in the fold cell): `reads_c_bare.rank_first_tok == 1` on
  **69/74**, `reads_w_bare` on **1/74**, `margin_sign_bare` = +1 on **73/74**.
  `key_canonical` = `"bare"` on 74/74 at both slots by the artifact's own rule K, and both keys are persisted
  at every record, so if rule K is wrong the label moves and the measurement does not - this is the one -it
  distributional column `INVENTORY_distributional.md` §4.1 does not kill.
  The amended disclosure. `INVENTORY_distributional.md` §1c says the forced-final slot is ABSENT at all twelve
  (scale x variant x arm) cells and that `controls/forcedfinal_dist.py` does not exist on disk; that is now
  false at exactly one cell - 9b -it, fold, first-token - and true at the other eleven. Nothing here restores
  2b, 27b, either base variant, or the listen direction, and nothing here is a content margin: the artifact's
  own §4.3 rule is quoted rather than paraphrased, "every margin is a FIRST-TOKEN, Rule-S-class reading. No
  number in this artifact may be called 'the probability of C' or 'the model's belief'."
  Quotation rules obeyed: the registration's PRIMARY readout is the join's §6.2 V-A DECOMP verdict quoted with
  r_move(A1) / r_move(A2) / r_off(A3), and this block quotes **none** of them - the distribution records are
  `readout_role = "secondary_diagnostic"` and are read directly off the summary artifact, so no verdict is
  promoted. The A3 (question-only) arm is deliberately absent from the prose: its elicit-slot split is
  `rank_c_bare == 1` on 17/74 against `rank_w_bare == 1` on 24/74 with `margin_sign_bare` +1 on 44/74, and the
  registration forbids reading A3 as 'the question causes folding' (blind-reversion class).
  The 74-versus-82 clause is D13's held wording, and is written into the prose here rather than left to a
  bracket because the numbers beside it are 74-denominated.

STATUS: READY
RESIDUAL: their bracket at L297 is kept whole and the new paragraph follows it, so the block trades no brackets. Their first two questions - why an alternative has to exist in the distribution, and whether attention copy works irrespective of that - are NOT answered here; only the third is. If they would rather the bracket be cut once the -chat half is answered, that is one deletion and **-1** bracket, and it is theirs. Also owed: this paragraph and T4-D08's opening both name the three-prompt construction; if both land, the clause in T4-D08 is the one to cut.

---

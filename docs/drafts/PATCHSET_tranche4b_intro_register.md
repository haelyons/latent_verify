# PATCHSET tranche 4b - the intro register pass

**This file SUPERSEDES `T4-I01`, `T4-I03a`, `T4-I03b`, `T4-I04a`, `T4-I04b` and `T4-I05` in
`PATCHSET_tranche4_intro.md`, and proposes an override of `T3-02` in `PATCHSET_tranche3.md`.
Do not apply the superseded blocks as well.** Each block below replaces exactly one block there,
on the same bytes, and the version here is the one to hand-apply. `T4-I02` (the L9 deletion) and
`T4-I06` (the note that writes nothing) are unchanged by this pass and still stand as written.

Two blocks are new: `T4b-I07` (a replacement TL;DR at L5, an OFFER, which **supersedes `T3-01`** if
taken) and the `L25 trade` note at the end, which is not a block and applies nothing.

Nothing here re-derives a number. Every figure in the superseded blocks was verified in the
2026-07-29/30 sessions and is correct. **This pass is about register only**: the blocks stated those
figures in a voice that is not the researcher's, and the fix is where a number is said, not whether
it is true. Every precise figure that leaves the prose is still in the block that carried it, in the
RECEIPT, which is apparatus and not prose.

## Live state at write time

| document | md5 | `wc -l` | split lines |
|---|---|---|---|
| `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_intro.md` | `83a55a14a8079403fa6be41c309c7f3b` | 28 | 29 |

Same md5 `PATCHMAP_live.md:10` and `PATCHSET_tranche4_intro.md:10` record, re-measured at write
time. **Every CURRENT fence below was sliced out of those bytes and asserted unique in the file**
(`count() == 1`, all ten). The L23 NBSPs (U+00A0, four), the curly quotes, `*_usually*`, the
`[` `]` runs, the leading space on T4b-I04b's anchor and the trailing spaces on T4b-I07's and L23's
are in the anchors as they are in the file. Do not retype an anchor; copy it. Re-verify against the
md5 before applying - `C02` is the standing proof that a correctly sliced anchor still rots
(`PATCHMAP_live.md` §2.1).

## The four faults this pass fixes

Confirmed against the researcher's own bytes, not against a style intuition.

**1. Raw statistics in prose.** Their intro prose carries almost no numbers: `82`, the three scales,
and De Marez's borrowed `17 of 23`. Every other digit in the live intro is a year, a DOI or an arXiv
ID inside a link. What it carries instead is **more than a dozen rough quantifiers** - "often" x4,
"almost always" x3, "*_usually*", "more than", "half the time", "a large share", "consistently",
"frequently", "sometimes", "much of that". The superseded blocks injected `12 of 34 folding and 15
of 35 listening`, `13 pairs`, `57 and 50 of 82`, and T3-02 injected `43.52% progressive against
14.66% regressive`. Every one of those is replaced by a rough quantifier in their idiom. The
judgement applied per case: **a number that IS the finding may stay; a number that merely evidences
one goes to the receipt.** `82` stays (it is the design). `17 of 23` stays (it is De Marez's own
headline, and their own prose already carries it). Nothing else survives in prose.

**2. "Teacher-forced" is not their word.** It appears in **no** researcher draft - not POST1, not
CIRCUIT, not V1/V2/V3b, not the live intro or notes. `T4-I01` introduced it to the post. T4b-I01
says what it means in plain language instead, in the construction they already use for it: POST1 L21
is "we make the model predict the next tokens from a set transcript where it has already output the
correct answer $C$".

**3. Undefined terms.** "flip rate" is used at L21 (in the De Marez sentence) and again at L23,
and is **never defined in the intro**. The De Marez paragraph after `T4-I04a` also leans on a
distinction it never states - the probability margin against the spoken answer, and the reply slot
against the final answer. T4b-I04a defines both inline, in a spaced-hyphen parenthetical, at first
use. No glossary, no new bracket, no definition sentence of its own.

**4. Anthropomorphic verbs.** Their convention, measured across the intro: a human verb is
**scare-quoted when it is doing work as a coined label or a mental-state attribution** - `"folds"`,
`"listens"`, `"abstains"` (L15), `"copying"`, `"willing"`, `"planted"`, `"resistance"`, and the
figure caption's `"fold"` / `"listen"` - and **left bare as ordinary description** - hedges, led
astray, adopt, carrying, and the bare `folds` inside their own L17 sentence. The superseded blocks
used fold / listen / abstain bare in new text, which promotes their labels into plain assertions.
Every new occurrence below is quoted; their own bare ones are left exactly as they are.

## Where the numbers went

| figure | was in | now in prose | still in the receipt of |
|---|---|---|---|
| `12 of 34` fold, `15 of 35` listen (27b -base alias misses) | T4-I03a | "about a third of those" | T4b-I03a |
| `13` pairs dropped at 27b | T4-I03b | "a small share" | T4b-I03b |
| `57 and 50 of 82` (margin moves, $C$ still ahead) | T4-I04a | "more than half the pairs" | T4b-I04a |
| `63/82` and `62/82` (pushed-slot pairwise) | receipt only | - | T4b-I04a |
| `43.52%` / `14.66%` | T3-02 | "about three times as often" (theirs, already in the line) | T3-02b |
| `13` manipulations (De Marez worst case) | T4-I05 | "their manipulations" | T4b-I05 |
| `4/5` and `5/5` head overlap | T3-03 prose | "no single circuit carrying it" | T4b-I07 |
| `17 of 23` | T4-I05 | **stays** - De Marez's headline, and already theirs | T4b-I05 |
| `82`, `2/9/27 billion` | gold | **stay** - the design | - |

## Application order

Intro only, **descending by line number**, so a line number is still right when you reach it:
T4b-I05 (L23), then the three L21 spans in descending byte offset - T4b-I04b, T4b-I04a(ii),
T4b-I04a(i) - then T3-02b (L19), T4b-I03b (L17), T4b-I03a (L15), **T4-I02 unchanged (L9, the
deletion)**, T4b-I01 (L7), T4b-I07 (L5). T4-I02 is still the only length change; descending order
means its deletion happens after every line below it is already edited.

## Ledger

Word delta measured with `split()` on the exact bytes, against **the version each block replaces**
and against the gold.

| block | replaces | vs gold | vs the version it replaces |
|---|---|---|---|
| `T4b-I01` | T4-I01 (+18) | **+24** | +6 |
| `T4b-I03a` | T4-I03a (+17) | **+13** | -4 |
| `T4b-I03b` | T4-I03b (+15) | **+16** | +1 |
| `T4b-I04a(i)` | new span | **+11** | +11 |
| `T4b-I04a(ii)` | T4-I04a (+19) | **+23** | +4 |
| `T4b-I04b` | T4-I04b (+19) | **+21** | +2 |
| `T4b-I05` | T4-I05 (-51) | **-51** | 0 |
| `T4-I02` (unchanged) | - | **-50** | 0 |
| **tranche-4b intro subtotal** | tranche 4 was **-13** | **+7** | **+20** |
| `T3-02b` | T3-02 (+16) | **+10** | -6 |
| `T4b-I07` | T3-01 (+1) | **+58** | +57 |
| `T3-03` **not applied** | T3-03 (+46) | **0** | -46 |
| L25 contradicted clause cut (see the trade note) | - | **-23** | -23 |
| **whole revised set** | applied draft was **+50** | **+52** | **+2** |

Read the bottom row as: the register fixes cost **+14 words** net (+20 across tranche 4b, minus 6 at
T3-02b) - plain language for "teacher-forced", the flip-rate gloss and the scare-quoted labels are
what buy it, and the two number-to-quantifier swaps that came out shorter (T4b-I03a, T3-02b) pay 10
of it back. The TL;DR trade returns **12** (+57 for T4b-I07, against -46 for T3-03 and -23 for the
L25 cut). So the intro ends **2 words longer** than the tranche-4 + tranche-3 draft it replaces,
carrying one more finding and one fewer contradicted claim.

Bracket delta **-5**, unchanged from tranche 4: T4b-I04b -1, T4b-I05 -4, T4b-I07 -1 +1 (their L5
bracket is consumed and one lowercase inline bracket replaces it, in their own idiom), all others 0.
No `[` is added anywhere else. NBSP delta **+0** - L23's four U+00A0 survive because T4b-I05 reuses
the `that`-to-`report` citation run whole, and this file's PROPOSED fence for it was derived from
T4-I05's bytes by substitution, not retyped. Em-dashes and en-dashes introduced: **0 / 0**.

## Disciplines every block below obeys

- Anchors sliced, never typed; uniqueness asserted at generation.
- No number in prose that is not the finding itself. Every figure removed from prose is named in
  the same block's RECEIPT with a `path#field` citation.
- No number is stated without the slot and register it is true at - the defect `598de5e` held
  D11/D13/D16 for.
- New anthropomorphic verbs are scare-quoted; their bare ones are untouched.
- `REDISTRIBUTE`, `0.875` and `0.751` appear nowhere in any proposed text (`RETRACTIONS.md` R-12,
  WITHDRAWN). Neither does the word "distributed" applied to `-it` head overlap.
- Perez is not cited in either direction (`PATCHMAP_live.md` §4 item 24, unresolved ledger conflict).

---

### T4b-I01 - intro L7, the plant, in plain language, and format co-varies with variant

SUPERSEDES `T4-I01` (`PATCHSET_tranche4_intro.md:79`). Same anchor, same two owed precisions.

ITEM: COMPOSE §D L7 (the two owed wording precisions); fault 2

CURRENT:

````
Each model variant/size is prompted with one of the pair items, then pushed with the other one, and lastly forced to provide a final answer.
````

PROPOSED:

````
Each model variant/size has one of the pair items already in its own turn, as though it had said it, is then pushed with the other one, and lastly forced to provide a final answer - raw Q:/A: at -base, chat turns at -chat, so format co-varies with variant.
````

RECEIPT:
  `docs/drafts/COMPOSE_post1_brief.md:136-138` - L7 is GROUNDED (82 pairs, provenance re-derived
  91 -> 87 KEEP -> 82); the two precisions owed are "the plant is TEACHER-FORCED into the model's own
  turn, not 'prompted with'; base cells are raw `Q:/A:`, -it cells chat turns - format co-varies with
  variant." The mechanism is unchanged here; only the word for it is.
  Corroborated in the instrument: `controls/rlhf_differential.py:167-173` builds the `-it` prompt via
  `apply_chat_template(add_generation_prompt=True)` while base builds `…\nA:`, quoted at
  `docs/drafts/INVENTORY_distributional.md:505-510`. That asymmetry is the root of the leading-space
  confound (INVENTORY §4.1), so the sentence that introduces the protocol is the right place for it.
  **Why "teacher-forced" goes.** A grep for the string across all five register-authority files
  (`STYLECARD_researcher.md:5-11`) and both live vault documents returns **zero**. The replacement is
  their own construction for the same operation, POST1 L21: "we make the model predict the next
  tokens from a set transcript where it has already output the correct answer $C$".
  **The backticks go too.** T4-I01's RESIDUAL flagged that its `Q:/A:` opened the intro's **first**
  inline code span (live intro backtick count: 0, against 54 in the notes). This pass takes the
  unfenced form the residual offered, which is the register-consistent call and closes the residual.
  Register: the long clause-stacked sentence with a comma series is theirs (`PATCHMAP_live.md` §5.1);
  spaced hyphen as the em-dash; no bullet, no roadmap clause.

STATUS: READY
RESIDUAL: none. T4-I01's typography residual is discharged by taking the unfenced form.

---

### T4b-I03a - intro L15, the 27b scope, without the fraction printed

SUPERSEDES `T4-I03a` (`PATCHSET_tranche4_intro.md:171`, sub-block (a)).

ITEM: COMPOSE §D L15; fault 1

CURRENT:

````
even when explicitly asked [at 9b the first of those is the forced answer, not the reply - the reply says the second].
````

PROPOSED:

````
even when explicitly asked [at 9b the first of those is the forced answer, not the reply - the reply says the second]. At 27b -base about a third of those are unresolved aliases, not hedges.
````

RECEIPT:
  `COMPOSE_post1_brief.md:139-142` - L15 is GROUNDED and **the researcher's own bracket is exactly
  right** ("I don't know." is the forced answer 6/82; the reply's hedge is "I'm not sure" 56/82, modal
  "No, I'm not sure. I'm just guessing." x37). Their bracket is left standing untouched.
  **The number this sentence used to print**, and where it now lives: the brief's own words are
  "~1/3 of the grey is alias-unresolvable, not hedging (**12/34 fold, 15/35 listen**)". The
  denominators reconcile with the figure's frozen assert -
  `docs/drafts/figs/make_figB_sankey.py:118,121` `EXPECT` gives 27b base `NEITHER` = **34** in fold
  and **35** in listen, which are what the fraction is a fraction of. 12/34 = 0.353 and 15/35 = 0.429,
  so "about a third" is the honest rough form and does not round up past either.
  `TAXONOMY_withholding.md:109-112` is the authority that the elicited slot's alias misses are
  matcher-attributable ("Persia" is the entire committed 0/0/1).
  This is a scope clause, not a hedge - it names a scorer limit.
  **Register.** "hedge" is left **bare**: it is bare in their own TL;DR ("-base abstains and hedges",
  L5) and quoted only where it is the label being introduced (notes L307). "those" refers to the
  cases the researcher's own clause just named ("otherwise names neither answer"), not to the
  figure's grey, so the sentence does not depend on the grey band's operational definition - which
  `T4-I02` deleted with L9 and which is still owed to the L12 caption (`C02`, stale anchor).

STATUS: READY

---

### T4b-I03b - intro L17, the scale scope and the drop, without the count printed

SUPERSEDES `T4-I03b` (`PATCHSET_tranche4_intro.md:171`, sub-block (b)).

ITEM: COMPOSE §D L17; fault 1

CURRENT:

````
in fact it folds significantly more than -base.
````

PROPOSED:

````
in fact it folds significantly more than -base at all three scales, though at 27b the test drops a small share as unresolved aliases.
````

RECEIPT:
  `COMPOSE_post1_brief.md:88-92` - "significantly" is GROUNDED by an **exact McNemar per scale**,
  p = **7.1e-15 / 1.2e-14 / 7.5e-11**, all `DIFFERS` (`out/gapclose_foldrate_sig.json`) - hence "at
  all three scales", which is the honest form of the adverb and is the finding. The same source
  states the disclosure the adverb must not hide: **13/82 pairs dropped at 27b as unresolved alias**.
  That 13 is what "a small share" now stands for; it evidences the caveat rather than being it, so
  it belongs here and not in the line. 13/82 = 0.159.
  Slot and arm, stated because the p-values are meaningless without them:
  `GROUNDING_crossvariant_scale.md:556` - `out/gapclose_foldrate_sig.json` tests **the fold arm at
  the elicited slot only**, which is the arm and slot L17 is about.
  **Register.** The bare "folds" in "in fact it folds significantly more than -base" is **theirs**
  and is carried through the anchor unchanged; this block adds no new bare label.
  **L16 checked and left standing**, as in T4-I03. Its second clause reads at the **reply** column,
  where `make_figB_matrix.py:119-131` `COUNTER_EXPECT["strict"]` puts `-it` NEITHER at 9 / 5 / 11
  (fold) and 7 / 14 / 15 (listen) of 82 - i.e. 67-77 of 82 do name one, so "almost always" survives.
  That residual grey is disclosed by their own L23 bracket, which T4b-I05 folds into prose; scoping
  L16 too would write it a third time.

STATUS: READY

---

### T4b-I04a - intro L21, what a flip rate is, and the margin without its counts

SUPERSEDES `T4-I04a` (`PATCHSET_tranche4_intro.md:229`, sub-block (a)) and **adds one new span**,
(i), which T4-I04 explicitly left untouched. The two spans are disjoint and both sit inside the
De Marez paragraph. Apply (ii) before (i) if applying by byte offset.

ITEM: COMPOSE §D L21 ("[this needs a major revision]" confirmed); faults 1, 2, 3

(i) CURRENT - the first use of "flip rate" anywhere in the intro:

````
argue that flip rates mix how strongly
````

(i) PROPOSED:

````
argue that flip rates - how often the model's spoken answer changes under pushback - mix how strongly
````

(ii) CURRENT:

````
To measure this in our context I measured the probability of our correct/plausibly incorrect $C$ and $W*$ answers, finding that Gemma 2 *_usually* assigns a higher probability to our selected $C$ than to $W*$. Interestingly, the model's output distribution shifts to the pushed answer even when the planted answer remains highest probability.
````

(ii) PROPOSED:

````
To measure this in our context I read the margin between $C$ and $W*$ - one answer's log-probability against the other's, over the answer strings, not the first token - and Gemma 2 *_usually* puts $C$ ahead at every cell before the push. Under the push that margin moves toward the pushed answer whilst $C$ stays ahead on more than half the pairs at 9b and 27b -base, the only two cells where it does.
````

RECEIPT:
  **(i), the definition.** "flip rate" occurs twice in the intro (L21, and "A flip-rate eval" at L23)
  and is defined at neither; it is also the term De Marez's own sentence turns on, so the paragraph
  reads as if the reader already has it. The gloss is the researcher's own spaced-hyphen
  parenthetical form (`STYLECARD` §A6), it is eleven words, it opens no bracket, and it is what makes
  the rest of the paragraph's contrast legible: a **rate over spoken answers** against a **margin
  over probabilities**. Their own notes already draw exactly this line at L181 - "They read a
  two-option log-probability margin, not a spoken answer" (`PATCHSET_tranche3.md` T3-17) - so the
  intro is being brought level with the notes, not given a new claim.
  **(ii), what survives, and its numbers.** The pairwise-margin phenomenon survives:
  `INVENTORY_distributional.md:442-445` - `faithful_RC` (the margin moved toward W\* by >= 0.5 nats)
  **and** `Mc_counter > 0` ($C$ still pairwise ahead) = **57/82 at 9b-base, 50/82 at 27b-base**.
  57/82 = 0.695 and 50/82 = 0.610, so "more than half the pairs" is true at both cells and is the
  rough form of exactly that count; the count itself reproduces `COMPOSE §C` and now lives here.
  **What had to change, and why** (unchanged from T4-I04a, still the reason the old sentence cannot
  stand): "remains highest probability" is false as a vocabulary-argmax claim -
  `INVENTORY_distributional.md:447-458,480-486`, $C$ is the argmax at the pushed answer slot on
  **0/82 at five cells** and 1/82 at 2b-base, the argmax being a polarity/discourse token (`' Yes'`,
  `' No'`, `' I'`, `'You'`); **W\* is never the argmax either** (0/82 at every cell) and is never the
  top riser. The claim is true only **pairwise against W\***, and only at the slots the replacement
  names: `:405-411` - TRUE at the bare slot at all six cells (54-74 of 82) and at the neutral slot at
  all six (66-81 of 82), FALSE at the pushed slot at four of six, true only at **9b-base 63/82 and
  27b-base 62/82**. Hence "ahead at every cell before the push", and "the only two cells where it
  does" after it. Those two slot counts stay in the receipt, as they did in T4-I04a.
  **The readout is still named, in plain words.** Two readouts of the same string disagree: the
  content margin `lp(strip(C)) - lp(strip(W*))`, polarity-stripped and scored over the answer's own
  tokens (`controls/family_cave_diagnose.py:236-239`, `INVENTORY §3.1`), versus the first-token
  reading, where `-it` counts are counts of persisted zeros (`INVENTORY §3.3`: `p_c == p_w` on
  78/59/68 of 82 at the counter slot, `p_c == 0.0` on 82/72/75). "over the answer strings, not the
  first token" is what stops the sentence being read off the wrong layer. It replaces T4-I04a's
  "teacher-forced, not the first token" and says the same thing without the term (fault 2); it is
  still the intro's half of the standing `D22` span-vs-first-token decision and still takes no
  position on the wording D22 reserves.
  **Not carried into the intro** (the notes' territory, `PATCHMAP_live.md` §5.4's duplication
  ledger): the argmax census, the 63/82 and 62/82 slot counts, the `-it` first-token degeneracy, and
  the 74-item family the De Marez run measures over rather than 82.
  Register: spaced hyphen as the em-dash, no bracket added, `*_usually*` carried verbatim, ends on a
  short flat clause.

STATUS: READY
RESIDUAL:
  - `A05` (FLAG, PENDING) and `C01` (APPLIED-Q) both live on L21 and both survive: A05's anchor is
    byte-disjoint from all three spans on this line, and C01's `*_usually*` is carried through (ii)
    verbatim. C01's scope residual is discharged (the sentence is explicitly a six-cell statement);
    its render question is not, and is still theirs.
  - (i) is a new span on a sentence T4-I04 declared untouched. If they would rather not have their
    De Marez sentence opened at all, (ii) and T4b-I04b stand without it - but then "flip rate" is
    still undefined in the intro and the paragraph's contrast has no anchor.

---

### T4b-I04b - intro L21, the slot disclosure

SUPERSEDES `T4-I04b` (`PATCHSET_tranche4_intro.md:229`, sub-block (b)).

ITEM: COMPOSE §D L21; faults 3, 4

CURRENT:

````
 [this needs a major revision]
````

PROPOSED:

````
 Those margins sit at the reply to the challenge, not at the final answer the sankey scores - only the 9b -chat "fold" arm has both.
````

RECEIPT:
  Unchanged in substance from T4-I04b; one word is quoted and one slot is named.
  `COMPOSE_post1_brief.md:152-155` and `GROUNDING_crossvariant_scale.md:539-541` both say the
  paragraph reads at the reply-to-challenge slot while the sankey's verdicts are at the forced-final
  slot, where **no distributional or residual readout exists at any cell** (`OWED.md` B2). That is
  now false at **exactly one cell**. The De Marez span-decomposition run persists, per item, a
  first-token top-10 **and** the C/W\* reads at **both** positions:
  `out/foldlisten_demarez_subst_dmz_9bit_a_summary.json#items[].distributions` has keys
  `counter_first` **and** `elicit_first`, each with `topk_10`, `argmax_tok_str`, `reads_c_space` /
  `reads_c_bare` / `reads_w_space` / `reads_w_bare` and `margin_sign_*`; the file's `cell` is `fold`
  and its run label is the 9b `-it` arm. The offline join confirms coverage in its own stamp -
  `out/demarez_join.json#stamp.slot` records the persisted first-token distribution at "the
  counter-reply and elicited-answer first positions" - and reports the elicited position directly
  (`position: "elicit_first"`, `n_margin_defined` 74 per turn arm).
  **Register.** `"fold"` is quoted, and named as an **arm**, because that is how the figure caption
  introduces it ("either "fold" or "listen"") - T4-I04b's bare "9b -chat folding" promoted their
  coined label into plain description (fault 4). Naming it an arm also states the slot distinction
  the paragraph leans on: the reply to the challenge against the final answer the sankey scores.
  The disclosure names its one exception rather than repeating a claim that has one.

STATUS: READY
RESIDUAL: the `[this needs a major revision]` bracket is **consumed** here. If they want T4b-I04a's
  revision but not the disclosure, (a) stands alone and the bracket stays - the spans are
  independent.

---

### T4b-I05 - intro L23, the "abstention gap" paragraph

SUPERSEDES `T4-I05` (`PATCHSET_tranche4_intro.md:313`). **Still an offer, not a fill.**
`PATCHSET_tranche2.md:899` files the whole paragraph as a researcher rewrite and their own closing
bracket says why. It is written out because "rewrite this" is not actionable without a candidate.
Take it, take a sentence of it, or take none.

Two register faults are fixed against T4-I05 and **nothing else moves**: `13 manipulations` loses its
count (fault 1 - the number evidences the scope, it is not the scope), and `abstain` is scare-quoted
(fault 4 - it is one of their coined labels, quoted at L15). The word count is identical either way.

ITEM: COMPOSE §D L23; `PATCHMAP_live.md` §4 item 15 (the register rewrite is filed researcher-only)

CURRENT (the whole line):

````
The abstention gap [what the fuck is the abstention gap?] sits next to a broader pattern that [SYCON](https://arxiv.org/abs/2505.23840) and [Gupta et al.](https://arxiv.org/abs/2607.18114) report from the outside: alignment tuning amplifies revisability under user pressure, while base models look more resistant. Here, much of that “resistance” is refusal to commit. A flip-rate eval that treats “I don’t know” as robustness will score -base as steadier than -chat. Read the same pressure off a two-option margin, as De Marez et al. do, and it runs the other way - in 17 of their 23 matched base-IT pairs the tuned model is the more robust one. Chat training deletes the grey band. [it goes from the elicited column only - the -it reply column still has one at every cell, and those are replies that name both answers] That sits awkwardly against the Gemma 2 report’s claim that post-training encouraged hedging to reduce hallucinations ([Gemma Team, 2024](https://arxiv.org/abs/2408.00118)), and more comfortably next to evidence that preference models penalize hedged answers ([Zhou et al., 2024](https://arxiv.org/abs/2401.06730)). [this paragraph wasn't edited from the model - all of the others ones were. can you see what reads differently? from the first sentence [the abstention gap sits] we can tell this isn't clear, and invents terminology like "abstention gap", rather than naming results and inferences clearly, in the style of the rest of this post]
````

PROPOSED (an offer - see STATUS):

````
Alignment tuning amplifies revisability under user pressure, while base models look more resistant - a pattern that [SYCON](https://arxiv.org/abs/2505.23840) and [Gupta et al.](https://arxiv.org/abs/2607.18114) report from the outside. Here, much of that “resistance” is refusal to commit. A flip-rate eval that treats “I don’t know” as robustness will score -base as steadier than -chat. Chat training deletes the grey band from the elicited column; in the reply column it survives at every cell, in replies that name both answers. De Marez et al. see no such reversal - both their channels favour the tuned model, and their 17 of 23 is a worst-case flip rate over their manipulations, not a margin - because their readout has no "abstain" outcome. Gemma is SYCON's own named exception, the narrowest gap they report. That sits awkwardly against the Gemma 2 report’s claim that post-training encouraged hedging to reduce hallucinations ([Gemma Team, 2024](https://arxiv.org/abs/2408.00118)), and more comfortably next to [Zhou et al., 2024](https://arxiv.org/abs/2401.06730), who find “In base models, we see a preference for weakeners but the trend reverses among RLHF models”.
````

RECEIPT:
  **Their three brackets, and what happens to each.** "[what the fuck is the abstention gap?]" - the
  term is invented and appears nowhere else in either document; the replacement opens on the subject
  matter in the first clause (`STYLECARD` §A3) and never names a gap. "[it goes from the elicited
  column only - the -it reply column still has one at every cell, and those are replies that name
  both answers]" - this is `A03`'s landed correction and it is **right**: `COUNTER_EXPECT["strict"]`
  puts `-it` reply NEITHER at 5-15 of 82 at every cell (`make_figB_matrix.py:119-131`) and
  `TAXONOMY_withholding.md:101-103` shows what those are - "Adjudication abstention, 63 items (62 of
  them -it). All 63 name both entities affirmatively". It is folded into prose, not dropped. The
  closing register bracket is consumed by the rewrite.
  **De Marez, corrected** (`GROUNDING_crossvariant_scale.md:476-483`): in their data **both** channels
  favour IT - "a drop from 23.3% to 16.3% flip rate on identical items" - so **their flip rate does
  not flatter base**, and the old sentence's "it runs the other way" was attributing this post's
  result to them. What runs the other way is *this post's* spoken-answer readout, which has an
  **"abstain" outcome theirs lacks**. The 17/23 is verbatim-correct and stays in prose - it is their
  headline and the researcher's own line already carries it - but it is **worst-case flip rate**
  (`max_t FR_t` over **13 manipulations**), not the margin (margin channel: 83.4% of pairs). The 13
  is the number that leaves the line: "over their manipulations" carries the scope, and the count of
  them is here.
  **SYCON as support** (`GROUNDING §11`): "Gemma is their named exception" - Gemma-2-9B Base 91.67 vs
  Instruct 86.31, the **narrowest** base-to-it gap in their Table 3. The paragraph previously used
  SYCON only as the outside view it then contradicted; the exception is the sharper use, because it
  is the same model family this post measures. Flagged honestly: SYCON is **UNFETCHED** (PDF-only, no
  HTML render), so its three quoted facts are ledger-sourced and unverified from the primary source
  (`GROUNDING_crossvariant_scale.md:598-599`) - which is the second reason the offer names the
  exception without printing the two numbers.
  **Zhou, the stronger quote** (`GROUNDING §11`): "In base models, we see a preference for weakeners
  but the trend reverses among RLHF models" - the closest published cross-variant hedging result to
  this paragraph, and it was sitting unused behind the weaker "preference models penalize hedged
  answers" gloss. Short, double-quoted, inside their own sentence (`STYLECARD` §A9). Scope, not
  printed in the intro: one reward model, 183 "What is the capital of X?" probes.
  **Perez is not cited**, in either direction (`PATCHMAP_live.md` §2.4, §4 item 24).
  **Bytes.** This block's PROPOSED text was derived from T4-I05's PROPOSED bytes by two substitutions
  (`over 13 manipulations` -> `over their manipulations`; `no abstain outcome` -> `no "abstain"
  outcome`), so the SYCON/Gupta citation run is still reused whole and all four U+00A0 survive at the
  same offsets; the two curly-quoted strings ("resistance", "I don't know") are still sliced, not
  retyped, as is the Gemma Team and Zhou link markup.

STATUS: **NEEDS-RESEARCHER-DECISION** - the rewrite is theirs by standing decision
(`PATCHMAP_live.md` §4 item 15). This block is a candidate, and applying it retires three brackets.
RESIDUAL:
  - **Two causal clauses in this paragraph are the researcher's own bytes and are carried unchanged:
    "alignment tuning amplifies revisability" (attributed to SYCON and Gupta as an outside report,
    not asserted) and "Chat training deletes the grey band".** Their own instruction at notes L133 is
    "Keep this descriptive: no causal 'tuning forces' claim", and read strictly it reaches both.
    There are no staged checkpoints in this work and format co-varies with variant
    (`INVENTORY §4.1`), so neither clause is licensed as causal by anything measured here. Cutting or
    softening them is theirs; this pass does not touch their sentences to make a point about mine.
  - `A04`'s scope of the 17-of-23 sentence is **carried** (worst-case flip rate), not lost. `A03`'s
    correction is carried as prose.
  - The Gemma Team 2024 clause is kept as they wrote it. `GROUNDING §11` notes it is a **data-mixture**
    statement - a scope worth a clause in the notes, not in this paragraph.
  - The grey band still has **no operational definition in the intro** after `T4-I02` deletes L9, and
    this paragraph leans on it. `C02` must be re-sliced before the L12 caption can carry it.

---

### T3-02b - intro L19, a proposed override of `T3-02`

**This block amends someone else's block. It does not replace it silently.** `T3-02`
(`PATCHSET_tranche3.md:78`, duplicated at `:445`) is READY and its receipt is correct; the two
percentages it prints are right and were re-verified in the C9 pass. The only objection is register:
the line is prose, the researcher's own version of this claim is already a rough quantifier ("about
three times as often"), and printing `43.52% progressive against 14.66% regressive` puts the densest
number in the intro into a sentence that is citing someone else's headline. **T3-02's substantive
corrections are all kept** - "find" -> "report", and the combined math **and medical** set replacing
"on different math-based examples". Only the two percentages move to the receipt.

If this override is not taken, apply `T3-02` as written; do not apply both.

ITEM: C9(iii); fault 1

CURRENT:

````
In SycEval Fanous et al. 2025 find that -chat models (ChatGPT-4o, Claude-Sonnet and Gemini-1.5-Pro) revise toward truth about three times as often, on different math-based examples - which is exactly what we found, where our -chat almost always "listens".
````

PROPOSED:

````
In SycEval Fanous et al. 2025 report that -chat models (ChatGPT-4o, Claude-Sonnet and Gemini-1.5-Pro) revise toward truth about three times as often over their combined math and medical set, an ordering that holds for each model - which is exactly what we found, where our -chat almost always "listens".
````

RECEIPT:
  `docs/drafts/CITATIONS_post1_verified.md`, SycEval entry: **43.52% progressive against 14.66%
  regressive** is the math+medical aggregate, not a math-only figure - which is why "on different
  math-based examples" cannot stand and why the scope clause replaces it. Per model the pairs are
  Gemini 53.22/9.25, ChatGPT 42.32/14.40, Claude-Sonnet 39.13/18.31, so progressive leads for all
  three - that is what "an ordering that holds for each model" states, and it is the claim the
  ledger licenses. The ledger's closed claim-list **bars reading the two rates as comparable
  propensities** (the opportunity pools are disjoint), so the rates are not a ratio and 43.52/14.66
  is not "three times" - the researcher's own "about three times as often" is SycEval's own framing
  and is left exactly as they wrote it. Re-verified in the C9 pass. "find" becomes "report" because
  the figures are their headline numbers, not a propensity we measured on their setup.
  Register: `"listens"` stays quoted (theirs), the spaced hyphen is theirs, and the sentence keeps
  its single long clause-stacked shape.

STATUS: **PROPOSED OVERRIDE of T3-02** - READY as text, but it is an amendment to a block the
researcher may have already accepted. If T3-02 is already applied to the vault, the delta from here
is: delete "- 43.52% progressive against 14.66% regressive over their combined math and medical set,
an ordering that holds for each model -" and restore the clause as written above.

---

### T4b-I07 - intro L5, a replacement TL;DR carrying the mechanism

**NEW BLOCK. An OFFER.** **This SUPERSEDES `T3-01`** (`PATCHSET_tranche3.md:55`, duplicated at
`:422`) - same anchor bytes, and T3-01's whole content is inside the replacement. If T4b-I07 is
taken, **do not also apply T3-01**. If T4b-I07 is declined, T3-01 stands unchanged and is still
READY. Note T3-01's coupling to notes `T3-21` (apply both or neither) rides along: it attaches to the
alias-miss sentence, which is the same in both.

ITEM: B9 (intro half); the TL;DR's silence on mechanism; faults 1 and 4

CURRENT (the tail of L5, T3-01's anchor byte-for-byte, trailing space included):

````
It never abstains. [at the final answer, at every scale; the one 27b exception is an alias miss, not a silence] 
````

PROPOSED (an offer - see STATUS):

````
It never abstains at the final answer, at every scale - the one 27b exception is an alias miss, not a silence. Under the push the two variants' distributions move much the same way: the pushed wrong answer gains probability at -base too, it just doesn't get said. What chat tuning changes is the policy of answering, and I found no single circuit carrying it [correlational at the head level, and the causal search returns nulls at every scale]. 
````

The two sentences before this span - "Gemma 2 -chat answers directly under user pushback whilst
-base abstains and hedges. The -chat model corrects itself when pushed toward truth, and also more
consistently is led astray by falsehood." - are **untouched**, which is why the anchor starts where
it does.

RECEIPT:
  **Sentence 1 is T3-01, verbatim.** `docs/drafts/figs/make_figB_sankey.py` `EXPECT` asserts -it
  elicited NEITHER = 0 / 0 / 1 fold, 0 / 0 / 0 listen. The 1 is fold-arm item 44 (chess; `elicit_gen`
  `Persia`, rule `bare_alias_miss`), in
  `results_foldlisten_ext2_27b/out/foldlisten_judge_fl_27bit_ext2_summary.json` and byte-identical in
  the re-run `results_foldlisten_nelicit_27b/out/foldlisten_judge_fl_27bit_ext2_summary.json` -
  draw-invariant, so no draw label is needed in prose. Re-verified in the B9 pass.
  **Sentence 2, the distributional synthesis.** `INVENTORY_distributional.md:442-445` -
  `faithful_RC` (margin moved toward W\* by >= 0.5 nats) with `Mc_counter > 0` ($C$ still ahead) =
  **57/82 at 9b-base and 50/82 at 27b-base**: at -base the push moves the probability mass toward the
  pushed answer on more than half the pairs **while the spoken answer does not change**, which is the
  same figure T4b-I04a reads and the reason the two lines say the same thing in two registers. The
  "it just doesn't get said" half is the sankey's own grey band at those cells
  (`make_figB_sankey.py:118,121` `EXPECT`, 27b base `NEITHER` = 34 fold / 35 listen) against -it's
  0 / 0 / 1. No count is printed in the TL;DR; both live here.
  **Three words this sentence must never contain, and why each is banned:**
    - **"monotonically"** - our own dose arm returned `DOSE_NONMONOTONE`
      (`out/demarez_join.json#verdicts.dose.verdict`, rule §6.3, resolution order
      `DOSE_UNEVALUABLE -> DOSE_FLAT -> DOSE_MONOTONE -> DOSE_NONMONOTONE`), and the same field
      carries a mandatory caveat that arms A4-A7 are not token-length-matched, so **no** outcome
      there licenses a gradient claim in either direction.
    - **"at every stage"** - the movement is measured **under the push**. There is no
      distributional or residual readout at the forced-final slot at any cell except the one named in
      T4b-I04b (`OWED.md` B2), so a stage-wise claim has no instrument behind it.
    - **"distributed"** of `-it` head overlap - contradicted by the overlap itself, 5/5 at -it against
      4/5 at base (`SNAPSHOT_circuit_groundtruth.md` §7.2 row 3; `RETRACTIONS.md` R-12).
  **Sentence 3, the policy claim and its bracket.** "What chat tuning changes is the policy of
  answering" is a **behavioural** statement - it says what differs between the variants, which is what
  the sankey shows, and asserts no training-time mechanism. It deliberately does **not** say training
  deletes, installs, or forces anything: there are no staged checkpoints in this work, and format
  co-varies with variant (`controls/rlhf_differential.py:167-173`, `INVENTORY §4.1`), so a causal
  claim is unlicensed twice over.
  "I found no single circuit carrying it", and the bracket, rest on
  `SNAPSHOT_circuit_groundtruth.md` §7.1 S4, §7.2, §3.2, §4:
    - fold-listen top-5 head overlap **4/5 at base** (9b and 2b; no 27b base run) and **5/5 at -it**
      at both scales, from `results_fold_vs_listen/out/cave_fold_vs_listen.json`;
    - all four `cave_fold_vs_listen` cells are **`MOVE_UNMATCHED`** - the matched-move gate failed,
      the instrument issued no verdict, so the head-level contrast is **correlational only**
      (§7.2 row 4). That is the bracket's first clause.
    - the -it write side is **at its floor at 3/3 scales**: `write_drops.wf_to_l` / `wl_to_f` =
      0.0 / 0.0 at 9b-it and 2b-it and 0.0 / -0.027027 at 27b-it, against write random floors
      1.0 / 1.0 and 0.918919 / 1.0, `cross_write.both_at_floor true` at all three
      (`SNAPSHOT §3.2`, the three `foldlisten_phase3b_*_summary.json`), with
      `verdict.verdict = "MONITOR_AGAIN"` and `write_both_at_floor: true` in the reasons at 3/3
      (`SNAPSHOT §3`). That is the bracket's second clause, "the causal search returns nulls at
      every scale" - and "every scale" here is 2b/9b/27b at -it, which is where the causal search
      ran.
  **Register.** `I` takes the finding and the failure (`STYLECARD` §A1: "I takes findings, naming,
  defining, choosing, failing"), so it is "I found no single circuit", not "no circuit was found".
  The TL;DR stays **one dense paragraph** (§A13). The bracket is inline, lowercase, unlabelled (§A8),
  and it replaces the one this span consumes, so L5's bracket count is unchanged. "abstains" is left
  **bare** in "It never abstains at the final answer" because that is their own byte-for-byte wording
  in this very line, twice. No number is printed that was not already in the gold; no
  `REDISTRIBUTE`, `0.875`, `0.751`, and no "distributed".

STATUS: **NEEDS-RESEARCHER-DECISION**. This puts a mechanism claim in the TL;DR, which the gold's
TL;DR does not carry. It supersedes T3-01. Sentence 1 alone is T3-01 and can be taken alone; sentence
2 and sentence 3 are independent of each other and either can be dropped without breaking the
paragraph.

**THE TRADE, and the word arithmetic.** This TL;DR carries the mechanism point, so **`T3-03`'s long
L25 replacement does not need to be applied**, and that is what pays for it:

| | words |
|---|---|
| T4b-I07 at L5 | **+58** |
| T3-01 at L5, superseded, no longer applied | **-1** |
| T3-03 span (a) at L25, not applied | **-45** |
| T3-03 span (b) at L25, the bracket swap, not applied | **-1** |
| **subtotal, if L25 is otherwise left as the gold** | **+11** |
| L25's contradicted clause cut instead (see the trade note below) | **-23** |
| **net for L5 and L25 together** | **-12** |

Against the tranche-4-applied draft, where T3-01 (+1) and T3-03 (+46) both land, those two lines
cost **+47**. With this trade they cost **+35**, and the intro loses a sentence the overlap number
contradicts instead of gaining a 68-word rewrite of it.

---

### The L25 trade note. **No block. Do not write one.**

If `T4b-I07` is taken, L25's mechanism sentence has nothing left to do, and it is the one sentence in
the intro that **no run supports as written**. Three options, priced:

1. **Apply `T3-03` anyway** (+46). The mechanism point is then in the intro twice, in two registers,
   with the L25 version carrying the counts the TL;DR version deliberately does not print. This
   breaks the duplication ledger (`PATCHMAP_live.md` §5.4) and is the most expensive option.
2. **Drop `T3-03` and leave L25 as the gold** (0). **Not acceptable on its own**: the gold clause
   says "at -chat, this mechanism is distributed", and the overlap number points the other way -
   5/5 at -it against 4/5 at base (`SNAPSHOT §7.2` row 3, `results_fold_vs_listen/out/cave_fold_vs_listen.json`).
   `RETRACTIONS.md` R-12 is the standing withdrawal. Taking T4b-I07 does not retire this problem; it
   only removes the reason to solve it at length.
3. **Drop `T3-03` and cut the contradicted clause** (-23). The span to delete, comma included, is:

````
, and I found that at -base, fold and listen share the same most influential attention heads, whilst at -chat, this mechanism is distributed.
````

   and it is replaced by a full stop, which leaves the host sentence as `"Folding" was one of the
   mechanisms looked at.` - 8 words where there were 31, no new claim, and nothing said in the intro
   that the record does not support. This is the option the arithmetic above assumes. The span is
   byte-unique in the gold and is T3-03's own span (a) with the preceding comma taken in, so it does
   not overlap T3-03's span (b) or anything else on L25.

Whichever is chosen, **their bracket under the bolded sentence** - "[nothing here exhibits the
shared-heads result this rests on - which run is it?]" - is answered by option 3 rather than filled:
the answer is that no run exhibits it, which is why the clause goes. Whether to close the bracket,
keep it as a standing question, or take T3-03's replacement bracket ("[the base and -chat head
rankings come from unmatched instruments, so the contrast is qualitative]") is theirs. The bolded
**"Chat training does not appear to install a dedicated truth circuit."** survives every option, and
is the one sentence on that line the record does support - it is a negative claim, and the nulls are
what carry it.

`T4-I06` (`PATCHSET_tranche4_intro.md:385`) is unchanged and still governs what any L25 text must
never gain.

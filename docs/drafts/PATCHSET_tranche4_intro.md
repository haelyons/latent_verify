# PATCHSET tranche 4 - the intro's uncovered lines

Six blocks, intro only. Five edit the gold; **T4-I06 edits nothing** and exists to record a
corroboration against a tranche-3 block that already owns its bytes.

## Live state at write time

| document | md5 | `wc -l` | split lines |
|---|---|---|---|
| `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_intro.md` | `83a55a14a8079403fa6be41c309c7f3b` | 28 | 29 |

This md5 is the one `PATCHMAP_live.md:10` records, re-measured at write time. **Every CURRENT fence
below was sliced out of those bytes by the script that generated this file** - the L23 NBSPs
(U+00A0, four of them, around the SYCON and Gupta links), the curly quotes, `*_usually*` and the
`[` `]` runs are in the anchors as they are in the file. Do not retype an anchor; copy it. Each
slice was asserted unique in the file at generation time. `C02` is the standing proof that a
correctly sliced anchor still rots when the researcher edits the line afterwards
(`PATCHMAP_live.md` §2.1), so re-verify against the md5 before applying.

## Application order

Intro only, and every block is on a different line, so **descending by line number**: T4-I05 (L23),
T4-I04b then T4-I04a (L21, two disjoint spans of one line), T4-I03b (L17), T4-I03a (L15), T4-I02
(L9), T4-I01 (L7). T4-I02 is the only length change - it deletes L9 and the blank L10, so every line
from L11 down moves up by two. Descending order means that deletion happens **after** every line
below it has already been edited, so a line number is still right when you reach it.

## What this tranche does not touch

- **L5, L19, L25** are covered by pending `T3-01`, `T3-02`, `T3-03`. No block here competes for
  those bytes (`PATCHMAP_live.md` §3). T4-I06 is a note *about* T3-03, not a second block on L25.
- **`wasd` and `its going`** (L21) are protected typos held by `A05`, which is a FLAG with no fill.
  T4-I04b's anchor is the bracket that follows A05's anchor and does not overlap it, so A05 still
  matches byte-exact after this tranche applies.
- **`*_usually*`** (L21) is `C01`'s open render question. T4-I04a rewrites the sentence around it and
  **carries the ten bytes verbatim**, so C01's decision stays live and stays theirs. C01's *residual*
  ("`Gemma 2` in this sentence is all six models and the measurement is 9b -base") is discharged by
  the rewrite - the replacement is a six-cell statement and says so.
- **L12** (the caption) is `C02`'s site and C02's anchor is already stale. T4-I02 states what the
  caption is owed if L9 goes and does **not** write a competing fill there.
- **L3** (`C03`) and **L16** are left alone; L16 is checked in T4-I03's receipt and holds.

## Bracket and length ledger

The intro carries **11** prose brackets across 8 lines (`PATCHMAP_live.md` §5.4, matching
`COMPOSE_post1_brief.md:182`). This tranche resolves brackets into prose and adds none.

| block | word delta | bracket delta | NBSP delta |
|---|---|---|---|
| `T4-I01` | +18 | +0 | +0 |
| `T4-I02` | -50 | +0 | +0 |
| `T4-I03a` | +17 | +0 | +0 |
| `T4-I03b` | +15 | +0 | +0 |
| `T4-I04a` | +19 | +0 | +0 |
| `T4-I04b` | +19 | -1 | +0 |
| `T4-I05` | -51 | -4 | +0 |
| **net** | **-13** | **-5** | **+0** |

**Net -13 words**: the intro gets shorter. Every precision added at L7, L15, L17 and L21 is paid
for by the L9 deletion and the L23 rewrite. Bracket depth stays min 0 / final 0 - no block opens or
closes a bracket the same block does not balance, and no `[` is added anywhere.

The NBSP delta is **+0**: L23's four U+00A0 survive because T4-I05 reuses the
`that`-to-`report` run whole. Intro NBSP total goes 12 -> 12 (L17, L19, L21, L23).

## Disciplines every block below obeys

- Anchors sliced, never typed; uniqueness asserted in the file at generation.
- No number is stated without the slot it is true at and the register it is read in - the defect
  `598de5e` held D11/D13/D16 for.
- Nothing in these blocks restates a sentence the notes already carry; where the honest home is the
  notes, the block says so in its RESIDUAL rather than writing the sentence twice.
- `REDISTRIBUTE`, `0.875` and `0.751` appear nowhere in any proposed text (`RETRACTIONS.md` R-12,
  WITHDRAWN).
- Perez is not cited in either direction (`PATCHMAP_live.md` §4 item 24, unresolved ledger conflict).

---

### T4-I01 - intro L7, the plant is teacher-forced, and format co-varies with variant

ITEM: COMPOSE §D L7 (the two owed wording precisions)

CURRENT:

````
Each model variant/size is prompted with one of the pair items, then pushed with the other one, and lastly forced to provide a final answer.
````

PROPOSED:

````
Each model variant/size has one of the pair items teacher-forced into its own turn, is then pushed with the other one, and lastly forced to provide a final answer - raw `Q:/A:` at -base, chat turns at -chat, so format co-varies with variant.
````

RECEIPT:
  `docs/drafts/COMPOSE_post1_brief.md:136-138` - L7 is GROUNDED (82 pairs, provenance re-derived
  91 -> 87 KEEP -> 82); the two precisions owed are exactly these: "the plant is TEACHER-FORCED into
  the model's own turn, not 'prompted with'; base cells are raw `Q:/A:`, -it cells chat turns -
  format co-varies with variant."
  Corroborated in the instrument: `controls/rlhf_differential.py:167-173` builds the `-it` prompt via
  `apply_chat_template(add_generation_prompt=True)` while base builds `…\nA:`, as quoted at
  `docs/drafts/INVENTORY_distributional.md:505-510`. That same asymmetry is the root of the
  leading-space confound (INVENTORY §4.1), so the sentence that introduces the protocol is the right
  place for it and the notes do not have to carry the disclaimer twice.
  Register: `PATCHMAP_live.md` §5.1 - the long clause-stacked sentence with a comma series is theirs;
  no bullet, no roadmap clause.

STATUS: READY
RESIDUAL: the intro currently has **zero** inline code spans - a backtick count over the live intro
  returns 0, against 54 in the notes. The notes do use one in prose for exactly this string, at live
  notes L149: "We cut it off after the first `Q:`". If they would rather not open the intro's first
  backtick, the same sentence reads unfenced as "raw Q:/A: at -base". A typography call, not a
  content one.

---

### T4-I02 - intro L9, the cut - the paragraph recites a legend the figure draws

ITEM: COMPOSE §E ("prose restates figure"), `PATCHMAP_live.md` §3 (L9 UNCOVERED)

CURRENT (the line, and the blank line that follows it, both deleted):

````
The results are presented in the below sankey. Green is a correct fact, red is its plausibly incorrect counterpart, and grey means neither of the pair was mentioned in the model's response. Rows compare -base and -chat Gemma 2 variants, and columns show increasing model scale from left to right.
````

PROPOSED:

````
(deleted)
````

RECEIPT:
  The figure is built by `docs/drafts/figs/make_figB_matrix.py:291` (**not** `make_figB_sankey.py`,
  which writes `figB_fold_ext2.png` / `figB_listen_ext2.png`). Clause by clause, L9 against what that
  generator draws into the PNG:
    - "Green is a correct fact, red is its plausibly incorrect counterpart, and grey means neither" ::
      `make_figB_matrix.py:68` `HUE = {"C": "#009E73", "WSTAR": "#CC3311", "NEITHER": "#b0b0ab"}` and
      `:70-71` `NICE = {"C": "correct (C)", "WSTAR": "wrong (W*)", "NEITHER": "withholds"}`, drawn as
      a three-swatch `fig.legend` at `:270-271`, plus an in-figure footer at `:277-279`:
      "hue = correctness (green C / red W* / gray withhold); muted = base, bold = -it".
    - "Rows compare -base and -chat Gemma 2 variants" :: `:265-269`, the row `ylabel` is
      `FOLD`/`LISTEN` + `base`/`-it` + `(start: C planted)` / `(start: W* planted)`.
    - "columns show increasing model scale from left to right" :: `:260-261`, the top row's
      `set_title(scale)` over `SCALES`.
  So all three clauses are drawn, and the palette is verified against the committed render:
  `docs/drafts/figs/figB_synthesis_strict_ext2.png` (md5 `50a3f28f743af3e4b90958dac42e8a42`, the
  render `COMPOSE_post1_brief.md:65` names as current) has `#009e73` and `#e18875` among its five most
  common colours - the prose's "green" is right, and it is right only for as long as the palette
  constant is. It is a hostage to a re-render, and a re-render is already owed on this figure
  (`COMPOSE §B`, the four vault swaps still on the anomalous 27b draw).

STATUS: READY

RESIDUAL - what the L12 caption is owed if L9 goes:
  Exactly one thing, and it is not a colour. The legend's word for grey is **"withholds"**, which is
  a behavioural word for what is actually a string-matching outcome; L9's "neither of the pair was
  mentioned in the model's response" is the only operational definition of the grey band anywhere in
  the intro, and **L15, L17 and L23 all lean on that band**. The caption must carry the naming
  register, not the colours.
  That is `C02`'s existing FILL - "[the reply column is scored the same way as the final answer - the
  answer has to be spelled out]" - which is HELD with **no reason in any commit body** and whose
  anchor is **stale** (`PATCHMAP_live.md` §2.1: the researcher deleted L12's terminal full stop and
  trailing space). Re-slicing C02 discharges this block's residual; no competing L12 fill is written
  here. Note also that the figure's own footer already states the register for the middle column
  ("counter = does the free reply NAME the answer (same string-identity register as the slot)",
  `make_figB_matrix.py:275-276`), so the caption needs the definition once, not twice.

---

### T4-I03 - intro L15 and L17, the two observations that are grounded but under-scoped

ITEM: COMPOSE §D L15, §D L17

(a) CURRENT:

````
even when explicitly asked [at 9b the first of those is the forced answer, not the reply - the reply says the second].
````

(a) PROPOSED:

````
even when explicitly asked [at 9b the first of those is the forced answer, not the reply - the reply says the second]. At 27b -base 12 of 34 folding and 15 of 35 listening are unresolved aliases, not hedges.
````

(b) CURRENT:

````
in fact it folds significantly more than -base.
````

(b) PROPOSED:

````
in fact it folds significantly more than -base at all three scales, though the test drops 13 pairs at 27b as unresolved aliases.
````

RECEIPT:
  (a) `COMPOSE_post1_brief.md:139-142` - L15 is GROUNDED and **the researcher's own bracket is exactly
  right** ("I don't know." is the forced answer 6/82; the reply's hedge is "I'm not sure" 56/82, modal
  "No, I'm not sure. I'm just guessing." x37). The block therefore leaves their bracket standing and
  adds only the 27b caveat that brief carries: "~1/3 of the grey is alias-unresolvable, not hedging
  (12/34 fold, 15/35 listen)". The denominators reconcile with the figure's own frozen assert:
  `make_figB_sankey.py:118,121` `EXPECT` gives 27b base `NEITHER` = **34** in fold and **35** in
  listen, which are the 34 and 35 the caveat is a fraction of. This is a scope clause, not a hedge -
  it names a scorer limit, and `TAXONOMY_withholding.md:109-112` is the authority that the elicited
  slot's alias misses are matcher-attributable ("Persia" is the entire committed 0/0/1).
  (b) `COMPOSE_post1_brief.md:88-92` - "significantly" is GROUNDED by an **exact McNemar per scale**,
  p = **7.1e-15 / 1.2e-14 / 7.5e-11**, all `DIFFERS` (`out/gapclose_foldrate_sig.json`) - hence "at
  all three scales", which is the honest form of the adverb. The same source states the disclosure
  the adverb must not hide: **13/82 pairs dropped at 27b as unresolved alias**. Slot and arm, stated
  because the number is meaningless without them: `GROUNDING_crossvariant_scale.md:556` -
  `out/gapclose_foldrate_sig.json` tests **the fold arm at the elicited slot only**, which is the arm
  and slot L17 is about.
  **L16 checked and left standing.** Its first clause holds at the elicited slot with room to spare
  (`EXPECT` listen `-it` C = 81 / 82 / 82 of 82). Its second clause, "It almost always gives one of
  the pair answers ($C$ or $W*$ in its response)", reads at the **reply** column, where the strict
  register puts `-it` NEITHER at 9 / 5 / 11 (fold) and 7 / 14 / 15 (listen) of 82
  (`make_figB_matrix.py:119-131` `COUNTER_EXPECT["strict"]`, the frozen assert for this very figure) -
  i.e. 67-77 of 82 do name one, so "almost always" survives. The residual grey there is already
  disclosed by **their own L23 bracket** ("the -it reply column still has one at every cell"), which
  T4-I05 folds into prose. Scoping L16 as well would write that disclosure a third time.

STATUS: READY

---

### T4-I04 - intro L21, the De Marez paragraph

ITEM: COMPOSE §D L21 ("[this needs a major revision]" confirmed); `INVENTORY_distributional.md` §3.1,
§3.1b, §3.2

Two disjoint spans of L21. The opening De Marez sentence is untouched, and so is the sentence that
carries `wasd` and `its going`.

(a) CURRENT:

````
To measure this in our context I measured the probability of our correct/plausibly incorrect $C$ and $W*$ answers, finding that Gemma 2 *_usually* assigns a higher probability to our selected $C$ than to $W*$. Interestingly, the model's output distribution shifts to the pushed answer even when the planted answer remains highest probability.
````

(a) PROPOSED:

````
To measure this in our context I read the margin between $C$ and $W*$ - one log-probability against the other, teacher-forced, not the first token - and Gemma 2 *_usually* puts $C$ ahead at every cell before the push. Under the push that margin moves toward the pushed answer while $C$ stays ahead on 57 and 50 of 82 at 9b and 27b -base, the only two cells where it does.
````

(b) CURRENT:

````
 [this needs a major revision]
````

(b) PROPOSED:

````
 Those margins sit at the reply to the challenge, not at the final answer the sankey scores - only 9b -chat folding has both.
````

RECEIPT:
  **What survives** is the pairwise-margin phenomenon, and it survives with its numbers:
  `INVENTORY_distributional.md:442-445` - `faithful_RC` (the margin moved toward W* by >= 0.5 nats)
  **and** `Mc_counter > 0` (C still pairwise ahead) = **57/82 at 9b-base, 50/82 at 27b-base**. That
  is the "moves while staying ahead" number and it reproduces `COMPOSE §C` exactly.
  **What had to change, and why.** "remains highest probability" is false as a vocabulary-argmax
  claim: `INVENTORY_distributional.md:447-458,480-486` - C is the argmax at the pushed answer slot on
  **0/82 at five cells** and 1/82 at 2b-base; the argmax is a polarity/discourse token (`' Yes'`,
  `' No'`, `' I'`, `'You'`), never an answer entity; **W\* is never the argmax either** (0/82 at every
  cell) and is never the top riser. The claim is true only **pairwise against W\***, and only at the
  slots the replacement names: `:405-411` - TRUE at the bare slot at all six cells (54-74 of 82) and
  at the neutral slot at all six (66-81 of 82), FALSE at the pushed slot at four of six, true only at
  **9b-base 63/82 and 27b-base 62/82**. Hence "ahead at every cell before the push", and "those are
  the only two cells where it does" after it.
  **The readout is named** because two different readouts of the same words disagree: the content
  margin `lp(strip(C)) - lp(strip(W*))`, teacher-forced and polarity-stripped
  (`controls/family_cave_diagnose.py:236-239`, `INVENTORY §3.1`), versus the first-token reading,
  where `-it` counts are counts of persisted zeros (`INVENTORY §3.3`: `p_c == p_w` on 78/59/68 of 82
  at the counter slot, `p_c == 0.0` on 82/72/75). "not the first token" is the clause that stops the
  sentence being read off the wrong layer, and it is the intro's half of the standing `D22`
  span-vs-first-token decision - it takes no position on the wording D22 reserves.
  **(b), the disclosure, and it changed on 2026-07-30.** `COMPOSE_post1_brief.md:152-155` and
  `GROUNDING_crossvariant_scale.md:539-541` both say the paragraph reads at the reply-to-challenge
  slot while the sankey's verdicts are at the forced-final slot, where **no distributional or
  residual readout exists at any cell** (`OWED.md` B2). That is now false at **exactly one cell**.
  The De Marez span-decomposition run persists, per item, a first-token top-10 **and** the C/W\* reads
  at **both** positions: `out/foldlisten_demarez_subst_dmz_9bit_a_summary.json#items[].distributions`
  has keys `counter_first` **and** `elicit_first`, each with `topk_10`, `argmax_tok_str`,
  `reads_c_space` / `reads_c_bare` / `reads_w_space` / `reads_w_bare` and `margin_sign_*`; the file's
  `cell` is `fold` and its run label is the 9b `-it` arm. The offline join confirms the coverage in
  its own stamp - `out/demarez_join.json#stamp.slot`: "the persisted first-token distribution records
  at the **counter-reply and elicited-answer first positions**" - and reports the elicited position
  directly (`position: "elicit_first"`, `n_margin_defined` 74 per turn arm). Hence "only one cell,
  9b -chat folding". The disclosure names the exception rather than repeating a claim that has one.
  **Not carried into the intro** (the notes' territory, and `PATCHMAP_live.md` §5.4's duplication
  ledger): the argmax census itself, the 63/82 and 62/82 slot counts, the `-it` first-token
  degeneracy, and the 74-item family the De Marez run measures over rather than 82.
  Register: the replacement keeps their spaced hyphen as the em-dash (`§5.1`), adds no bracket, and
  ends on a short flat clause.

STATUS: READY
RESIDUAL:
  - `A05` (FLAG, PENDING) and `C01` (APPLIED-Q) both live on L21 and both survive this block: A05's
    anchor is byte-disjoint from both spans, and C01's `*_usually*` is carried through verbatim into
    (a)'s replacement. C01's residual scope demand is discharged (the sentence is now explicitly a
    six-cell statement); its render question is not, and is still theirs.
  - The `[this needs a major revision]` bracket is **consumed** by (b). If they want the revision but
    not the disclosure sentence, (a) stands alone and the bracket stays - the two spans are
    independent.

---

### T4-I05 - intro L23, a replacement for the "abstention gap" paragraph

ITEM: COMPOSE §D L23; `PATCHMAP_live.md` §4 item 15 (the register rewrite is filed researcher-only)

This is **an offer, not a fill.** `PATCHSET_tranche2.md:899` files the whole paragraph as a
researcher rewrite and their own closing bracket says why. It is written out because "rewrite this"
is not an actionable instruction without a candidate, and because three of its factual defects are
now settled. Take it, take a sentence of it, or take none.

CURRENT (the whole line):

````
The abstention gap [what the fuck is the abstention gap?] sits next to a broader pattern that [SYCON](https://arxiv.org/abs/2505.23840) and [Gupta et al.](https://arxiv.org/abs/2607.18114) report from the outside: alignment tuning amplifies revisability under user pressure, while base models look more resistant. Here, much of that “resistance” is refusal to commit. A flip-rate eval that treats “I don’t know” as robustness will score -base as steadier than -chat. Read the same pressure off a two-option margin, as De Marez et al. do, and it runs the other way - in 17 of their 23 matched base-IT pairs the tuned model is the more robust one. Chat training deletes the grey band. [it goes from the elicited column only - the -it reply column still has one at every cell, and those are replies that name both answers] That sits awkwardly against the Gemma 2 report’s claim that post-training encouraged hedging to reduce hallucinations ([Gemma Team, 2024](https://arxiv.org/abs/2408.00118)), and more comfortably next to evidence that preference models penalize hedged answers ([Zhou et al., 2024](https://arxiv.org/abs/2401.06730)). [this paragraph wasn't edited from the model - all of the others ones were. can you see what reads differently? from the first sentence [the abstention gap sits] we can tell this isn't clear, and invents terminology like "abstention gap", rather than naming results and inferences clearly, in the style of the rest of this post]
````

PROPOSED (an offer - see STATUS):

````
Alignment tuning amplifies revisability under user pressure, while base models look more resistant - a pattern that [SYCON](https://arxiv.org/abs/2505.23840) and [Gupta et al.](https://arxiv.org/abs/2607.18114) report from the outside. Here, much of that “resistance” is refusal to commit. A flip-rate eval that treats “I don’t know” as robustness will score -base as steadier than -chat. Chat training deletes the grey band from the elicited column; in the reply column it survives at every cell, in replies that name both answers. De Marez et al. see no such reversal - both their channels favour the tuned model, and their 17 of 23 is a worst-case flip rate over 13 manipulations, not a margin - because their readout has no abstain outcome. Gemma is SYCON's own named exception, the narrowest gap they report. That sits awkwardly against the Gemma 2 report’s claim that post-training encouraged hedging to reduce hallucinations ([Gemma Team, 2024](https://arxiv.org/abs/2408.00118)), and more comfortably next to [Zhou et al., 2024](https://arxiv.org/abs/2401.06730), who find “In base models, we see a preference for weakeners but the trend reverses among RLHF models”.
````

RECEIPT:
  **Their three brackets, and what happens to each.** "[what the fuck is the abstention gap?]" - the
  term is invented and appears nowhere else in either document; the replacement opens on the subject
  matter in the first clause (`STYLECARD` §5.1) and never names a gap. "[it goes from the elicited
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
  **abstain outcome theirs lacks**; the same source's instruction is to name the readout, not the
  metric. The 17/23 is verbatim-correct but is **worst-case flip rate** (`max_t FR_t` over **13
  manipulations**), not the margin (margin channel: 83.4% of pairs) - the replacement states the
  scope inline.
  **SYCON as support** (`GROUNDING §11`): "Gemma is their named exception" - Gemma-2-9B Base 91.67 vs
  Instruct 86.31, the **narrowest** base-to-it gap in their Table 3. The paragraph previously used
  SYCON only as the outside view it then contradicted; the exception is the sharper use, because it
  is the same model family this post measures. Flagged honestly: SYCON is **UNFETCHED** (PDF-only, no
  HTML render), so its three quoted facts are ledger-sourced and unverified from the primary source
  (`GROUNDING_crossvariant_scale.md:598-599`) - the offer therefore names the exception without
  printing the two numbers.
  **Zhou, the stronger quote** (`GROUNDING §11`): "In base models, we see a preference for weakeners
  but the trend reverses among RLHF models" - the closest published cross-variant hedging result to
  this paragraph, and it was sitting unused behind the weaker "preference models penalize hedged
  answers" gloss. Short, double-quoted, inside their own sentence (`STYLECARD` §5.5). Scope, not
  printed in the intro: one reward model, 183 "What is the capital of X?" probes.
  **Perez is not cited**, in either direction: two ledgers disagree and the conflict is unresolved
  (`PATCHMAP_live.md` §2.4, §4 item 24).
  **Bytes.** The SYCON/Gupta citation run is reused whole, so all four of L23's U+00A0 survive at the
  same offsets relative to the links; the two curly-quoted strings ("resistance", "I don't know") are
  sliced, not retyped; the Gemma Team and Zhou link markup is sliced.

STATUS: **NEEDS-RESEARCHER-DECISION** - the rewrite is theirs by standing decision
(`PATCHMAP_live.md` §4 item 15). This block is a candidate, and applying it retires three brackets.
RESIDUAL:
  - The same §4 item 15 notes L23 breaches their own instruction at notes L133 ("Keep this
    descriptive: no causal 'tuning forces' claim"). The offer keeps their existing "alignment tuning
    amplifies revisability" clause because it is attributed to SYCON and Gupta as an outside report,
    not asserted; if they want the instruction honoured strictly, that first clause is the one to cut.
  - `A04` scoped the 17-of-23 sentence and its content is **carried** (the worst-case-flip-rate
    scope), not lost. `A03`'s correction is carried as prose.
  - The Gemma Team 2024 clause is kept as they wrote it. `GROUNDING §11` notes it is a **data-mixture**
    statement (hedging is one of three included behaviours, measured by factuality metrics, not a
    hedging rate) - a scope worth a clause in the notes, not in this paragraph.

---

### T4-I06 - a note on intro L25. **No block. Do not write one.**

ITEM: T3-03 corroboration (`PATCHSET_tranche3.md:128`, the one NEEDS-RESEARCHER-DECISION block)

L25 is COVERED by `T3-03`, which is PENDING and holds two spans of that line. **No competing block is
written here.** This note records that T3-03's replacement claim was independently confirmed this
session, and fixes what its text must never gain.

CORROBORATED (`SNAPSHOT_circuit_groundtruth.md` §7.1 S4, §7.2, §3.2, §4):
  - fold-listen top-5 head overlap: **4/5 at base** (9b and 2b; no 27b run exists) and **5/5 at -it**
    at both scales. The number points the **opposite** way to "at -chat, this mechanism is
    distributed" (§7.2, row 3).
  - all four `cave_fold_vs_listen` cells are **`MOVE_UNMATCHED`** - the matched-move gate failed, the
    instrument issued no verdict, and the base result is **correlational only** (§7.2, row 4).
  - the -it write side is **at its floor at 3/3 scales**: `write_drops.wf_to_l` / `wl_to_f` = 0.0 /
    0.0 at 9b-it and 2b-it and 0.0 / -0.027027 at 27b-it, against write random floors 1.0 / 1.0 and
    0.918919 / 1.0, `cross_write.both_at_floor true` at all three
    (`SNAPSHOT §3.2`, the three `foldlisten_phase3b_*_summary.json`), with
    `verdict.verdict = "MONITOR_AGAIN"` and `write_both_at_floor: true` in the reasons at 3/3
    (`SNAPSHOT §3`).

WHAT T3-03's TEXT MUST NEVER GAIN:
  1. **the word "distributed"** applied to `-it` head overlap. Overlap is the one number that
     contradicts it (5/5 at -it vs 4/5 at base). What the -it instrument actually shows is **no single
     lever** - a different, unmatched, -it-only phase-3a/3b instrument - and "no lever" is not
     "distributed heads" (`COMPOSE_post1_brief.md:169-178`).
  2. **`REDISTRIBUTE`**. WITHDRAWN, `RETRACTIONS.md` R-12: no instrument writes the string to any
     artifact; the artifact's actual decision field is `BOTH_REDUNDANT`.
  3. **`0.875` / `0.751`**. WITHDRAWN with it: the headline sits **outside its own bootstrap CI**
     (0.874962 against [0.571004, 0.862805]), holds only under the self-judge axis, and the same
     artifact's label-matched re-read returns `it_all_attn 0.0`, `it_all_mlp 0.0`, category
     `INSUFFICIENT`, with `label_match_changes_verdict: true`. No per-item records
     (`RETRACTIONS.md` R-12; `SNAPSHOT §7.2`).

STATUS: NOTE ONLY - nothing to apply.

---

# DRAFT - POST1 intro, the REVISED tranche-4b block set APPLIED

**This is a DERIVED artifact. It is not the gold.** The gold is
`/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_intro.md`, md5
`83a55a14a8079403fa6be41c309c7f3b` (28 `wc -l`, 29 split lines), and nothing was written to
`/home/hal/Documents/`. This file exists so the revised intro patch blocks can be read as prose
instead of assembled in the head.

**This SUPERSEDES `docs/drafts/DRAFT_post1_intro_tranche4_applied.md`.** That draft applied
`T4-I01`, `T4-I03a`, `T4-I03b`, `T4-I04a`, `T4-I04b`, `T4-I05`, `T3-01`, `T3-02` and `T3-03`. Every
one of those is either replaced by its tranche-4b version or deliberately not applied here, per the
header of `docs/drafts/PATCHSET_tranche4b_intro_register.md`, which is **the authority for this
draft**. The older draft should not be read as current.

Built 2026-08-01. Ten anchors were sliced out of the patchset files (never retyped), byte-compared
against the gold and asserted unique in it before replacement; one further anchor was deleted, and
one contradicted clause was cut under a priced option. Untouched sentences are byte-identical to the
gold - typos, spaced hyphens, guillemets, NBSPs, `$C$`/`$W*$` and all.

Lines beginning `>>` are **editorial markers written by this pass**, not the researcher's text and
not their bracket idiom. Each sits directly above the paragraph it governs and names the choice made
for this draft, the alternative, and where the evidence lives. **To get clean prose: delete every
line beginning `>>`, and delete the three fenced blocks between `>> ORIGINAL-BEGIN` and
`>> ORIGINAL-END`, fence lines included.** What remains is the applied intro, exactly as counted in
DELTAS.

---

## THE APPLIED INTRO

# Characterizing base vs chat model behaviours under pushback in Gemma 2

Language models sometimes abandon their answer and adopt the user’s when challenged. This is usually studied as sycophancy: the model begins correct, the user suggests something false, and the model "folds". I tested this and the opposite, where a model starts incorrect and "listens" to a correction [, in -base and -chat model variants of Gemma 2. Models are “chat tuned” using various techniques to make them more able to act like helpful assistants, and provide good answers - which it turns out, also makes them worse in some ways.]

>> DECISION T4b-I07 (gold L5, the TL;DR) -- EDITORIAL MARKER, not the researcher's text.
>> STATUS in its own block: **NEEDS-RESEARCHER-DECISION**. It puts a mechanism claim in the TL;DR,
>> which the gold's TL;DR does not carry.
>> **T4b-I07 SUPERSEDES T3-01** (PATCHSET_tranche3.md:55, duplicated at :422) -- same anchor bytes,
>> and T3-01's whole content is sentence 1 of the replacement. T3-01 is therefore NOT applied here.
>> If T4b-I07 is declined, T3-01 stands unchanged and is still READY.
>> CHOICE MADE FOR THIS DRAFT: all three sentences applied. THE ALTERNATIVES, all cheap: sentence 1
>> alone is exactly T3-01; sentences 2 and 3 are independent of each other and either can be dropped
>> without breaking the paragraph.
>> COUPLING THAT RIDES ALONG: T3-01 is coupled to notes block T3-21 (PATCHSET_tranche3.md:390, apply
>> both or neither), and the coupling attaches to the alias-miss sentence, which is identical in
>> both. T3-21 is a NOTES block and cannot be applied in this document. If T3-21 does not land in the
>> notes, sentence 1 should revert so the two documents keep one resolution.
>> WHY THIS BLOCK EXISTS AT ALL -- THE TRADE: carrying the mechanism point here is what pays for
>> **not** applying T3-03's 46-word L25 replacement. See the L25 marker below. The two are one
>> decision, and the word arithmetic only works if both land.
>> EVIDENCE. Sentence 1: make_figB_sankey.py EXPECT (-it elicited NEITHER 0/0/1 fold, 0/0/0 listen;
>> the 1 is fold-arm item 44, chess, elicit_gen "Persia", rule bare_alias_miss, byte-identical in the
>> re-run, so it is draw-invariant and no draw label is needed in prose). Sentence 2:
>> INVENTORY_distributional.md:442-445, faithful_RC with Mc_counter > 0 at 9b-base and 27b-base --
>> the push moves probability mass toward the pushed answer on more than half the pairs while the
>> spoken answer does not change -- read against the sankey's own grey band at those cells. No count
>> is printed in the TL;DR; both live in the block's receipt. Sentence 3:
>> SNAPSHOT_circuit_groundtruth.md sections 7.1 S4, 7.2, 3.2 and 4 -- all four cave_fold_vs_listen
>> cells are MOVE_UNMATCHED, so the head-level contrast is correlational only, and the -it write side
>> is at its floor at 3 of 3 scales with MONITOR_AGAIN. Those are the inline bracket's two clauses.
>> "What chat tuning changes is the policy of answering" is deliberately BEHAVIOURAL: it says what
>> differs between variants, which is what the sankey shows, and asserts no training-time mechanism.
>> There are no staged checkpoints in this work and format co-varies with variant, so a causal claim
>> would be unlicensed twice over.
>> THREE WORDS THIS SENTENCE MUST NEVER GAIN, none of them present: "monotonically" (our own dose arm
>> returned DOSE_NONMONOTONE, and its arms A4-A7 are not token-length-matched, so no outcome there
>> licenses a gradient claim in either direction), "at every stage" (there is no distributional or
>> residual readout at the forced-final slot at any cell but the one T4b-I04b names), and
>> "distributed" of -it head overlap (contradicted by the overlap itself; RETRACTIONS.md R-12).
> **TL;DR** Gemma 2 -chat answers directly under user pushback whilst -base abstains and hedges. The -chat model corrects itself when pushed toward truth, and also more consistently is led astray by falsehood. It never abstains at the final answer, at every scale - the one 27b exception is an alias miss, not a silence. Under the push the two variants' distributions move much the same way: the pushed wrong answer gains probability at -base too, it just doesn't get said. What chat tuning changes is the policy of answering, and I found no single circuit carrying it [correlational at the head level, and the causal search returns nulls at every scale]. 

These initial results are derived across -base and -chat Gemma 2 at 2, 9, and 27 billion parameters with 82 correct/plausibly incorrect fact pairs. Each model variant/size has one of the pair items already in its own turn, as though it had said it, is then pushed with the other one, and lastly forced to provide a final answer - raw Q:/A: at -base, chat turns at -chat, so format co-varies with variant. 

>> ##########################################################################
>> ##  KNOWN REGRESSION -- READ BEFORE ANYTHING FROM THIS DRAFT GOES INTO  ##
>> ##  THE VAULT. THIS IS THE ONE POINT ON WHICH THE APPLIED INTRO IS      ##
>> ##  WORSE THAN THE GOLD.                                                ##
>> ##########################################################################
>> DELETION T4-I02 (gold L9 and its blank line) -- EDITORIAL MARKER, not the researcher's text.
>> APPLIED, unchanged from tranche 4 -- this block is NOT superseded by tranche 4b. All three of its
>> clauses are drawn inside the PNG by make_figB_matrix.py: the three-swatch legend at :270-271 plus
>> the in-figure footer at :277-279, the row ylabels at :265-269, the column titles at :260-261. So
>> the paragraph recited the figure's own legend.
>>
>> **THE REGRESSION.** Deleted L9 carried the **ONLY operational definition of the grey band anywhere
>> in the intro** -- "neither of the pair was mentioned in the model's response". Its paired fix is
>> C02, which puts that naming register into the Figure 1 caption, and **C02's anchor is STALE**
>> (PATCHMAP_live.md section 2.1: the researcher deleted L12's terminal full stop and trailing space,
>> so the block byte-compares False; it is also HELD with no reason in any commit body). The caption
>> therefore cannot be fixed yet and C02 is logged SKIPPED. The figure's own legend word for grey is
>> "withholds" -- a behavioural word for what is a string-matching outcome -- so the PNG does not
>> supply the definition either.
>> **After this deletion the intro defines the grey band nowhere, and later paragraphs still lean on
>> it**: observation 3 below (gold L17, "drops a small share as unresolved aliases") and the whole
>> grey-band paragraph (gold L23, which says the band goes from the elicited column and survives in
>> the reply column). **Applying T4-I02 without first re-slicing C02 leaves the intro worse on this
>> one point. The two should land together -- do not take this deletion into the vault alone.**
>> Observation 1 (gold L15) is deliberately NOT exposed: T4b-I03a's "those" refers to the
>> researcher's own preceding clause ("otherwise names neither answer"), not to the figure's grey,
>> which is exactly why that block was written that way.
>> One further reason the deleted paragraph should not simply come back: its colour words are a
>> hostage to the palette constant, and a re-render of this figure is already owed (see OPEN).
>> The deleted line, verbatim:
>> ORIGINAL-BEGIN
```
The results are presented in the below sankey. Green is a correct fact, red is its plausibly incorrect counterpart, and grey means neither of the pair was mentioned in the model's response. Rows compare -base and -chat Gemma 2 variants, and columns show increasing model scale from left to right.
```
>> ORIGINAL-END
![[figB_synthesis_strict_ext2.png]]
*Figure 1:* *Answer flows across Gemma 2*. Each cell shows the 82 examples run for a model, and an experiment type, either "fold" or "listen", starting with the correct fact $C$ and plausibly incorrect fact $W*$ respectively, and getting pushed with their counterparts

Some high level observations here:
1. -base Gemma 2 often "abstains" - when pushed, it frequently replies “I don’t know,” “I’m not sure,” or otherwise names neither answer, even when explicitly asked [at 9b the first of those is the forced answer, not the reply - the reply says the second]. At 27b -base about a third of those are unresolved aliases, not hedges.
2. -chat Gemma 2 almost always takes a correct push, correcting itself from a wrong answer to the correct one. It almost always gives one of the pair answers ($C$ or $W*$ in its response).
3. -chat Gemma 2 still folds to plausible falsehood - in fact it folds significantly more than -base at all three scales, though at 27b the test drops a small share as unresolved aliases. Planted on the correct answer and offered a plausible wrong one, it commits to the false answer in a large share of cases.

What we call folding and listening have been studied extensively; they are what [SycEval](https://doi.org/10.1609/aies.v8i1.36598) calls _regressive_ and _progressive_ sycophancy. In SycEval Fanous et al. 2025 report that -chat models (ChatGPT-4o, Claude-Sonnet and Gemini-1.5-Pro) revise toward truth about three times as often over their combined math and medical set, an ordering that holds for each model - which is exactly what we found, where our -chat almost always "listens". We also note that it "folds" very often though. Fanous et al. also find that on medical advice this reverses for Claude-Sonnet, and overall have no -base comparison. We find that at -base models don't name the answer (our planted or pushed strings for $C$ and $W*$) unless pushed, and only then use them half the time. There's some indication that -base is responding to the push [what indication?], but much less than -chat, and it seems clear [does it? this isn't a very good simple explanatory sentence, and also this claim seems like its been repeated several times in different forms?] that at some level -base is carrying, or "copying" the answer from the entry-point (our "planted" answer) to the answer.

[De Marez et al.](https://arxiv.org/abs/2606.06306) argue that flip rates - how often the model's spoken answer changes under pushback - mix how strongly the model already prefers the truth, and how far pressure can move that preference. To measure this in our context I read the margin between $C$ and $W*$ - one answer's log-probability against the other's, over the answer strings, not the first token - and Gemma 2 *_usually* puts $C$ ahead at every cell before the push. Under the push that margin moves toward the pushed answer whilst $C$ stays ahead on more than half the pairs at 9b and 27b -base, the only two cells where it does. This is not shown in the sankey, and adding another one to this page wasd vetoed by Fable, so its going in the lab notes. Those margins sit at the reply to the challenge, not at the final answer the sankey scores - only the 9b -chat "fold" arm has both.

>> DECISION T4b-I05 (gold L23, the "abstention gap" paragraph) -- EDITORIAL MARKER, not the
>> researcher's text. **SUPERSEDES T4-I05**, which the previous draft applied.
>> **THIS PARAGRAPH IS AN OFFER, NOT A FILL.** PATCHMAP_live.md section 4 item 15 files the L23
>> register rewrite as researcher-only; PATCHSET_tranche2.md:899 files the whole paragraph as their
>> own rewrite, and their own closing bracket says why. It is written out only because "rewrite this"
>> is not actionable without a candidate.
>> CHOICE MADE FOR THIS DRAFT: the offered paragraph is in the body below, so the result can be read
>> as prose. It retires the three brackets its own STATUS counts -- the "abstention gap" query, A03's
>> grey-band correction (folded into prose, not dropped), and the closing register bracket -- which is
>> four `[` characters, because one is nested inside that closing bracket.
>> WHAT CHANGED FROM T4-I05, and nothing else moved: "over 13 manipulations" became "over their
>> manipulations" (the count evidences the scope, it is not the scope; the 13 is in the receipt), and
>> "abstain" is now scare-quoted, because it is one of the researcher's coined labels and is quoted
>> at gold L15. The word count is identical either way, and the PROPOSED text was derived from
>> T4-I05's bytes by substitution, not retyped, so the SYCON/Gupta citation run is reused whole and
>> all four of L23's U+00A0 survive at the same offsets.
>> THE ALTERNATIVES: take none of it and keep the gold paragraph (preserved verbatim below), or take
>> single sentences -- the De Marez correction, the SYCON-exception sentence and the Zhou quote are
>> independent of one another. A fourth option is named in the block's own RESIDUAL: cut the opening
>> "alignment tuning amplifies revisability" clause. That clause and "Chat training deletes the grey
>> band" are **both the researcher's own bytes, carried unchanged**, and read strictly their own
>> notes-L133 instruction ("Keep this descriptive: no causal 'tuning forces' claim") reaches both.
>> There are no staged checkpoints in this work and format co-varies with variant, so neither clause
>> is licensed as causal by anything measured here. Cutting or softening them is theirs; this pass
>> does not touch their sentences to make a point about its own.
>> EVIDENCE: GROUNDING_crossvariant_scale.md:476-483 -- De Marez's own data has BOTH channels
>> favouring IT ("a drop from 23.3% to 16.3% flip rate on identical items"), so their flip rate does
>> not flatter base, and the gold sentence's "it runs the other way" was attributing this post's
>> result to them; what runs the other way is this post's spoken-answer readout, which has an
>> "abstain" outcome theirs lacks. Their 17 of 23 is verbatim-correct and stays -- it is their own
>> headline and the researcher's line already carried it -- but it is a worst-case flip rate over 13
>> manipulations, not the margin channel. GROUNDING section 11: SYCON's Gemma exception (Gemma-2-9B
>> Base 91.67 vs Instruct 86.31, the narrowest base-to-it gap in their Table 3) -- **SYCON is
>> UNFETCHED, PDF-only, so those facts are ledger-sourced and unverified from the primary source**,
>> which is the second reason the offer names the exception without printing the two numbers; and the
>> stronger Zhou quote, which was sitting unused behind the weaker "preference models penalize hedged
>> answers" gloss. make_figB_matrix.py:119-131 and TAXONOMY_withholding.md:101-103: the reply-column
>> grey survives at every cell (5-15 of 82), and 63 adjudication abstentions all name both entities
>> affirmatively. Perez is cited in neither direction (PATCHMAP_live.md section 4 item 24).
>> THE GREY BAND THIS PARAGRAPH LEANS ON IS UNDEFINED IN THIS DRAFT -- see the T4-I02 regression
>> marker above. C02 must be re-sliced first.
>> THE RESEARCHER'S ORIGINAL L23, VERBATIM, for comparison and one-step revert:
>> ORIGINAL-BEGIN
```
The abstention gap [what the fuck is the abstention gap?] sits next to a broader pattern that [SYCON](https://arxiv.org/abs/2505.23840) and [Gupta et al.](https://arxiv.org/abs/2607.18114) report from the outside: alignment tuning amplifies revisability under user pressure, while base models look more resistant. Here, much of that “resistance” is refusal to commit. A flip-rate eval that treats “I don’t know” as robustness will score -base as steadier than -chat. Read the same pressure off a two-option margin, as De Marez et al. do, and it runs the other way - in 17 of their 23 matched base-IT pairs the tuned model is the more robust one. Chat training deletes the grey band. [it goes from the elicited column only - the -it reply column still has one at every cell, and those are replies that name both answers] That sits awkwardly against the Gemma 2 report’s claim that post-training encouraged hedging to reduce hallucinations ([Gemma Team, 2024](https://arxiv.org/abs/2408.00118)), and more comfortably next to evidence that preference models penalize hedged answers ([Zhou et al., 2024](https://arxiv.org/abs/2401.06730)). [this paragraph wasn't edited from the model - all of the others ones were. can you see what reads differently? from the first sentence [the abstention gap sits] we can tell this isn't clear, and invents terminology like "abstention gap", rather than naming results and inferences clearly, in the style of the rest of this post]
```
>> ORIGINAL-END
Alignment tuning amplifies revisability under user pressure, while base models look more resistant - a pattern that [SYCON](https://arxiv.org/abs/2505.23840) and [Gupta et al.](https://arxiv.org/abs/2607.18114) report from the outside. Here, much of that “resistance” is refusal to commit. A flip-rate eval that treats “I don’t know” as robustness will score -base as steadier than -chat. Chat training deletes the grey band from the elicited column; in the reply column it survives at every cell, in replies that name both answers. De Marez et al. see no such reversal - both their channels favour the tuned model, and their 17 of 23 is a worst-case flip rate over their manipulations, not a margin - because their readout has no "abstain" outcome. Gemma is SYCON's own named exception, the narrowest gap they report. That sits awkwardly against the Gemma 2 report’s claim that post-training encouraged hedging to reduce hallucinations ([Gemma Team, 2024](https://arxiv.org/abs/2408.00118)), and more comfortably next to [Zhou et al., 2024](https://arxiv.org/abs/2401.06730), who find “In base models, we see a preference for weakeners but the trend reverses among RLHF models”.

>> DECISION L25, THE MECHANISM SENTENCE -- EDITORIAL MARKER, not the researcher's text.
>> **T3-03 IS NOT APPLIED HERE.** That is the single biggest difference from the draft this
>> supersedes. The TL;DR now carries the mechanism point (T4b-I07), and that trade is the whole
>> reason this intro does not grow further.
>> THE PROBLEM, which could not be left alone: the gold clause asserts that at -chat "this mechanism
>> is distributed", and our own overlap number points the other way -- base fold and listen share
>> 4 of 5 top heads, -chat shares 5 of 5 (SNAPSHOT_circuit_groundtruth.md section 7.2 row 3, from
>> results_fold_vs_listen/out/cave_fold_vs_listen.json). RETRACTIONS.md R-12 is the standing
>> withdrawal. No run supports the sentence as written, and taking T4b-I07 does not retire that
>> problem -- it only removes the reason to solve it at length.
>> THE THREE OPTIONS, PRICED (PATCHSET_tranche4b_intro_register.md, "The L25 trade note"):
>>   1. APPLY T3-03 ANYWAY, **+46 words**. The mechanism point then sits in the intro twice, in two
>>      registers, with the L25 version printing counts the TL;DR version deliberately does not.
>>      Breaks the duplication ledger (PATCHMAP_live.md section 5.4). The most expensive option.
>>   2. LEAVE L25 AS THE GOLD, **0 words**. **Not acceptable on its own** -- it keeps a clause the
>>      overlap number contradicts.
>>   3. CUT THE CONTRADICTED CLAUSE, **-23 words**.
>> CHOICE MADE FOR THIS DRAFT: **option 3**, the minimal one, and the one the block's own arithmetic
>> assumes. The span deleted, comma included, is byte-unique in the gold and is T3-03's own span (a)
>> with the preceding comma taken in, so it overlaps nothing else on the line. It is replaced by a
>> full stop, leaving the host sentence as: "Folding" was one of the mechanisms looked at. -- 8 words
>> where there were 31, no new claim, and nothing said in the intro that the record does not support.
>> WHAT THIS COSTS, stated plainly: the next sentence's "This roughly fits our behavioural evals in
>> the sankey" now refers back to a much thinner antecedent than it did in the gold. The trade note
>> prices no fix for that and this pass writes none; smoothing it is a researcher call.
>> THEIR BRACKET IS ANSWERED, NOT FILLED. "[nothing here exhibits the shared-heads result this rests
>> on - which run is it?]" survives in the body below. Option 3 answers it: **no run exhibits it,
>> which is why the clause goes.** Whether to close the bracket, keep it as a standing question, or
>> take T3-03's replacement bracket ("[the base and -chat head rankings come from unmatched
>> instruments, so the contrast is qualitative]") is theirs.
>> THE BOLDED SENTENCE SURVIVES EVERY OPTION. "Chat training does not appear to install a dedicated
>> truth circuit." is the one claim on this line the record does support -- it is a negative claim,
>> and the nulls are what carry it (-it write handles at floor at 3 of 3 scales, MONITOR_AGAIN).
>> T4-I06 (PATCHSET_tranche4_intro.md:385) is unchanged and still governs what any L25 text must
>> never gain: the word "distributed" of -it head overlap; the string REDISTRIBUTE (no instrument
>> writes it to any artifact -- the artifact's actual decision field is BOTH_REDUNDANT); and the
>> numbers 0.875 / 0.751 (the headline sits outside its own bootstrap CI, holds only under the
>> self-judge axis, and the label-matched re-read returns INSUFFICIENT).
>> The cut span, verbatim:
>> ORIGINAL-BEGIN
```
, and I found that at -base, fold and listen share the same most influential attention heads, whilst at -chat, this mechanism is distributed.
```
>> ORIGINAL-END
The full lab notes go into further detail. This investigation started by trying to paraphrase prompts, freeze attention to make attribution graphs, and adversarially perturb those graphs (like the prompts) to find common circuitry/mechanisms. "Folding" was one of the mechanisms looked at. This roughly fits our behavioural evals in the sankey, where -base often holds the planted answer (or withholds) and -chat revises freely in both directions, more so toward truth. **Chat training does not appear to install a dedicated truth circuit.** [nothing here exhibits the shared-heads result this rests on - which run is it?] It makes Gemma 2 less "willing" to say it does not know, and more to revise.

[Full lab notes pending write-up - Characterizing base vs chat behaviours under pushback in Gemma 2]

*Compute kindly provided by Apart Research via Lambda.ai. I'm running out though, so if you want to send me more money for compute or talk to me about my slowly perplexifying CV from all of this AI safety work please reach out, helioslyons.com*

---

## APPLICATION LOG

| block | source | gold line | outcome | one line |
|---|---|---|---|---|
| `T4b-I07` | `PATCHSET_tranche4b_intro_register.md` | L5 | **APPLIED + MARKED** | The TL;DR's bracket becomes prose and gains the distributional and mechanism sentences; NEEDS-RESEARCHER-DECISION, and it **supersedes `T3-01`**, which is therefore not applied. |
| `T3-01` | `PATCHSET_tranche3.md` (L55/L422, duplicated) | L5 | **NOT APPLIED (superseded)** | Its whole content is sentence 1 of T4b-I07; applying both would write the alias-miss correction twice on one line. |
| `T4b-I01` | `PATCHSET_tranche4b_intro_register.md` | L7 | APPLIED | Says the plant in plain language ("already in its own turn, as though it had said it") instead of "teacher-forced", and adds that format co-varies with variant; takes the unfenced `Q:/A:`, so the intro still has zero backticks. |
| `T4-I01` | `PATCHSET_tranche4_intro.md` | L7 | **NOT APPLIED (superseded)** | Its "teacher-forced" appears in no researcher draft, and its backtick residual is discharged by taking T4b-I01's unfenced form. |
| `T4-I02` | `PATCHSET_tranche4_intro.md` | L9 (+ blank L10) | **APPLIED (deletion) + MARKED PROMINENTLY** | Unchanged by tranche 4b. Deletes the paragraph that recites the figure's legend. **Known regression: it takes the intro's only operational definition of the grey band with it, and its paired fix C02 is unappliable.** |
| `C02` | `PATCHSET_tranche2.md` | L12 | **SKIPPED** | HELD with **no reason in any commit body**, and its **anchor is STALE** (`PATCHMAP_live.md` §2.1: the researcher deleted L12's terminal full stop and trailing space). It must be re-sliced before the caption can carry the grey band's definition. No competing L12 fill was written. |
| `T4b-I03a` | `PATCHSET_tranche4b_intro_register.md` | L15 | APPLIED | Leaves the researcher's own bracket standing and appends the 27b scope as a rough quantifier: "about a third of those are unresolved aliases, not hedges". |
| `T4-I03a` | `PATCHSET_tranche4_intro.md` | L15 | **NOT APPLIED (superseded)** | It printed `12 of 34 folding and 15 of 35 listening`; those counts now live in T4b-I03a's receipt, not in the prose. |
| `T4b-I03b` | `PATCHSET_tranche4b_intro_register.md` | L17 | APPLIED | "significantly more than -base" gains "at all three scales" (exact McNemar, all three `DIFFERS`) and the disclosure the adverb must not hide, as "a small share". |
| `T4-I03b` | `PATCHSET_tranche4_intro.md` | L17 | **NOT APPLIED (superseded)** | It printed `13 pairs`; the count now lives in T4b-I03b's receipt. |
| `T3-02b` | `PATCHSET_tranche4b_intro_register.md` | L19 | APPLIED | T3-02's substantive corrections are kept -- "find" becomes "report", and the combined math **and medical** set replaces "on different math-based examples" -- but the two percentages move to the receipt, leaving the researcher's own "about three times as often". |
| `T3-02` | `PATCHSET_tranche3.md` | L19 | **NOT APPLIED (overridden by T3-02b)** | Its receipt is correct and its `43.52%` / `14.66%` are right; the objection is register only. If T3-02 is already in the vault, T3-02b's STATUS gives the exact delta to walk it back. |
| `T4b-I04a(i)` | `PATCHSET_tranche4b_intro_register.md` | L21 | APPLIED | **New span.** Defines "flip rate" at its first use in the intro, in the researcher's own spaced-hyphen parenthetical, so the paragraph's rate-versus-margin contrast has an anchor. |
| `T4b-I04a(ii)` | `PATCHSET_tranche4b_intro_register.md` | L21 | APPLIED | Replaces the probability sentence: "remains highest probability" was false as a vocabulary-argmax claim, so the sentence now reads the margin, names the readout in plain words ("over the answer strings, not the first token"), and scopes the two cells as "more than half the pairs". Carries `*_usually*` through **verbatim**. |
| `T4b-I04b` | `PATCHSET_tranche4b_intro_register.md` | L21 | APPLIED | Consumes `[this needs a major revision]` with the slot disclosure, and quotes `"fold"` as an arm because that is how the Figure 1 caption introduces it. |
| `T4-I04a` / `T4-I04b` | `PATCHSET_tranche4_intro.md` | L21 | **NOT APPLIED (superseded)** | They printed `57 and 50 of 82`, used "teacher-forced", and left "flip rate" undefined; T4-I04b's bare "9b -chat folding" promoted a coined label into plain description. |
| `T4b-I05` | `PATCHSET_tranche4b_intro_register.md` | L23 | **MARKED-FOR-DECISION (offer applied)** | The "abstention gap" paragraph is replaced by the block's offered rewrite; retires its three brackets (four `[`, one nested). **Researcher-only by standing decision**; their original is preserved verbatim in the marker for one-step revert. |
| `T4-I05` | `PATCHSET_tranche4_intro.md` | L23 | **NOT APPLIED (superseded)** | Two register faults fixed against it and nothing else moved: `13 manipulations` loses its count, `"abstain"` is scare-quoted. Same word count. |
| `T3-03` | `PATCHSET_tranche3.md` | L25 (two spans) | **DELIBERATELY NOT APPLIED** | The TL;DR now carries the mechanism point, and that trade is the whole reason the intro does not grow further. Applying it would put the mechanism point in the intro twice and cost +46 words. |
| L25 contradicted clause | `PATCHSET_tranche4b_intro_register.md` (the trade note, option 3) | L25 | **APPLIED (cut) + MARKED** | Not a block -- a priced option. The clause "at -chat, this mechanism is distributed" is contradicted by the overlap itself, so the clause is cut and replaced by a full stop, **-23 words**. Options 1 (+46) and 2 (0, not acceptable alone) are named in the marker. |
| `T4-I06` | `PATCHSET_tranche4_intro.md` | L25 | **NOT A BLOCK** | By design: it writes no text. It fixes what any L25 text must never gain, and its content is carried into the L25 marker. Still live and still governing. |
| `A05` | `PATCHSET_final.md` | L21 | **SKIPPED (nothing to apply)** | PENDING **FLAG only, no fill**: `wasd` is a protected typo. Honoured -- `wasd` and `its going` are untouched, and A05's anchor is byte-disjoint from all three L21 spans, so it still matches after this pass. |
| `C01` | `PATCHSET_tranche2.md` | L21 | **SKIPPED (nothing to apply)** | APPLIED-Q, a **no-fill QUESTION**. The ten bytes of `*_usually*` are carried through T4b-I04a(ii) verbatim. Its *scope* residual is discharged (the sentence is explicitly a six-cell statement); its render question is not. |
| `C03` | `PATCHSET_final.md` / `PATCHSET_tranche2.md` | L3 | **SKIPPED (nothing to apply)** | APPLIED-Q, a **no-fill QUESTION**: adopt-as-prose or cut the comma-opening chat-tuning bracket. L3 is byte-identical to the gold. |

Checked and deliberately left alone: **L16** (T4b-I03b's receipt re-checks it -- "almost always" survives at
the reply column, 67-77 of 82 name one of the pair -- and scoping it would write the reply-column
disclosure a third time), **L12** (C02's site, stale), **L27**, **L29**.

---

## DELTAS

Counted over the applied prose only: every `>>` line and all three fenced ORIGINAL blocks removed,
which is the text that would go into the post.

| measure | gold | applied | delta |
|---|---|---|---|
| words (`split()`) | 1132 | 1184 | **+52** |
| `[` in prose (excludes the `![[…]]` embed and all six markdown links) | 12 | 7 | **-5** |
| split lines | 29 | 27 | -2 (T4-I02's deletion) |
| NBSP (U+00A0) | 12 | 12 | 0 |
| em-dash / en-dash | 0 / 0 | 0 / 0 | 0 / 0 |
| words byte-identical to the gold (difflib matching blocks, word-level) | - | 865 | **73.1% of the output** |

**Stated plainly: the intro grows, by 52 words, about 4.6%.** It does not shrink and this pass does
not claim it does. Two things are true at once and both matter:

- Against the **gold**, +52.
- Against the **draft this supersedes** (`DRAFT_post1_intro_tranche4_applied.md`, which stood at
  +50), **+2**. The whole register pass, the flip-rate definition, the plain-language plant, the
  scare-quoted labels, the new TL;DR and the L25 cut together cost **two words** more than the draft
  they replace -- while carrying one more finding and one fewer contradicted claim.

Per-block, every figure reproduced exactly against the block's own ledger:

| block | word delta |
|---|---|
| `T4b-I07` (L5) | **+58** |
| `T4b-I01` (L7) | +24 |
| `T4-I02` (L9, deletion) | **-50** |
| `T4b-I03a` (L15) | +13 |
| `T4b-I03b` (L17) | +16 |
| `T3-02b` (L19) | +10 |
| `T4b-I04a(i)` (L21) | +11 |
| `T4b-I04a(ii)` (L21) | +23 |
| `T4b-I04b` (L21) | +21 |
| `T4b-I05` (L23) | **-51** |
| L25 contradicted clause cut | **-23** |
| **net** | **+52** |

Where the growth is: **the TL;DR is 58 of the 52.** Everything else in the intro nets to -6. If the
intro must not grow at all, T4b-I07 is the block to trim -- its sentence 2 and sentence 3 are
independent and either can be dropped, and dropping sentence 3 would re-open the L25 question that
option 3 just closed.

Bracket ledger, 12 to 7. **Gone:** L21's `[this needs a major revision]` (T4b-I04b) and all four on
L23 (T4b-I05 -- three of them the researcher's, plus the one nested inside the closing register
bracket). **Replaced in place:** L5's bracket is consumed by T4b-I07 and one lowercase inline bracket
takes its place, in the researcher's own idiom, so L5's count is unchanged and **no `[` is added
anywhere in this pass**. **Still standing, untouched:** L3 (C03), L15's 9b slot bracket (T4b-I03a
leaves it deliberately -- its receipt confirms it is exactly right), both of L19's, L25's
shared-heads question (answered by the cut, not filled), and L27's title placeholder.

---

## OPEN

Every item below is the researcher's, not a drafter's.

**Decisions carried in the body, marked, and still open**

1. **`T4b-I07`, the new TL;DR (L5).** NEEDS-RESEARCHER-DECISION. It puts a mechanism claim in a
   TL;DR that did not carry one. Sentence 1 alone is T3-01; sentences 2 and 3 are independent.
   **It supersedes T3-01, and it is bound to the L25 decision below** -- they are one decision, and
   the arithmetic only works if both land.
2. **The L25 trade.** Option 3 (cut, -23) is applied here. Options 1 (apply T3-03, +46) and 2 (leave
   the gold, 0, not acceptable alone) are priced in the marker. Downstream of the cut: the next
   sentence's "This roughly fits our behavioural evals in the sankey" now refers back to a thinner
   antecedent. Whether the intro carries a base/-chat mechanism contrast **at all** is theirs; the
   base and -it head rankings come from unmatched instruments, so the contrast is qualitative
   however it is worded (`PATCHMAP_live.md` §4 item 1).
3. **Their L25 bracket** -- "[nothing here exhibits the shared-heads result this rests on - which run
   is it?]". Close it, keep it as a standing question, or take T3-03's replacement bracket.
4. **`T4b-I05`, the L23 paragraph.** An OFFER against a standing researcher-only decision
   (`PATCHMAP_live.md` §4 item 15). Take it, take a sentence of it, or take none; the original is in
   the marker. Live sub-decision: whether to cut or soften **"alignment tuning amplifies
   revisability"** and **"Chat training deletes the grey band"** -- both are the researcher's own
   bytes, carried unchanged, and read strictly their own notes-L133 instruction reaches both.
5. **`T3-01`'s coupling to notes `T3-21`** (`PATCHMAP_live.md` §4 item 2). It rides along on
   T4b-I07's sentence 1. T3-21 is a notes block and cannot be applied in this document; if it does
   not land there, sentence 1 should revert so the two documents keep one resolution.
6. **`T3-02b` is an override of someone else's READY block.** If T3-02 has already gone into the
   vault, the walk-back delta is in T3-02b's STATUS. Do not apply both.

**Open questions this draft deliberately did not answer**

7. **`C01`** -- `*_usually*` renders as itself. `*usually*` or `_usually_`? Carried through verbatim.
8. **`C03`** -- the comma-opening chat-tuning bracket at L3: adopt as prose, or cut. Untouched.
9. **`A05`** -- `wasd` and `its going` at L21 are protected typos. Untouched; the flag has no fill.
10. **`D22`, span versus first token** (`PATCHMAP_live.md` §4 item 9). T4b-I04a(ii)'s "over the
    answer strings, not the first token" is the intro's half of it and takes no position on the
    wording D22 reserves.
11. **Perez** (`PATCHMAP_live.md` §4 item 24) -- two ledgers disagree; not cited here in either
    direction, and the conflict still gates T3-05's proposed text in the notes.
12. **`C02` must be re-sliced before the Figure 1 caption can be fixed.** This is not a preference;
    it is the open half of a regression this draft ships. See item 13.

**The regression this draft ships, restated so it is not lost in the log**

13. **The intro now defines the grey band nowhere.** `T4-I02` deletes gold L9, which carried the only
    operational definition ("neither of the pair was mentioned in the model's response"). Its paired
    fix `C02` has a stale anchor and could not be applied. Observation 3 and the L23 paragraph both
    still lean on the band, and the figure's own legend word is "withholds", a behavioural word for a
    string-matching outcome, so the PNG does not supply it either. **On this one point the applied
    intro is worse than the gold. T4-I02 and C02 should land together.**

**The two image problems, neither of which a text draft can fix**

14. **The vault's Fig 1 embed is the anomalous 27b draw.** `![[figB_synthesis_strict_ext2.png]]`
    resolves in the vault to a copy md5-confirmed as the **anomalous** decode (`6942c40b…`), not the
    reproducible re-run the repo now holds (`50a3f28f…`). It is one of four stale vault embeds named
    in `COMPOSE_post1_brief.md` §B and `PATCHMAP_live.md` §4 item 8. A re-render is owed -- and
    T4-I02's receipt makes the deleted paragraph's colour words a hostage to it, which is one more
    reason that paragraph should not simply come back.
15. **The Ankara PNG has no vault copy.** `fig_topk_ankara_9bbase.png` is routed into the notes' Fig
    3b slot and **will not render** until the file is copied into the vault
    (`PATCHMAP_live.md` §3, notes L291). That is a notes problem, not an intro one, but it is the
    fifth of the five pending image actions and it blocks clean application of the tranche as a whole.

---

## MECHANICAL VERIFICATION

Run against this file at write time, over the marker-stripped prose.

| check | result |
|---|---|
| gold md5 asserted before any edit | **`83a55a14a8079403fa6be41c309c7f3b`**, matched |
| every anchor byte-exact and unique in the gold before replacement | **11 of 11** (`count() == 1` asserted on each, and each asserted to sit on its expected gold line). Sliced from the patchset files, never retyped. |
| every applied block's PROPOSED text present verbatim | **10 of 10.** T4-I02 is a deletion -- its CURRENT text is verified **absent** from the applied prose and survives only inside its marker. The L25 cut is likewise verified absent and preserved in its marker. |
| every unpatched gold line present byte-identical | **19 of 19**, of which **8 of 8** carry text: L1, L3, L11, L12, L14, L16, L27, L29. The other eleven are blank separators. |
| `REDISTRIBUTE` in the applied prose | **0** |
| `0.875` in the applied prose | **0** |
| `0.751` in the applied prose | **0** |
| `teacher-forc` in the applied prose | **0** (T4-I01's word; T4b-I01 replaces it with plain language) |
| `monotonic` in the applied prose | **0** |
| `distributed` in the applied prose | **0** -- the gold's only occurrence is inside the L25 clause the cut removes |
| em-dashes introduced | **0** (gold 0, output 0) |
| en-dashes introduced | **0** (gold 0, output 0) |
| NBSPs (U+00A0) preserved | **12 -> 12**; all four of L23's survive inside T4b-I05's reused citation run |
| curly quotes / guillemets | preserved byte-exact wherever the gold had them |
| per-block word deltas against the block's own ledger | **11 of 11 match** (+58, +24, -50, +13, +16, +10, +11, +23, +21, -51, -23; net +52, which is the ledger's "whole revised set" figure) |
| nothing written under `/home/hal/Documents/` | confirmed -- the gold was opened read-only and edited in a copy |

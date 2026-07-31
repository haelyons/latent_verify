# DRAFT - POST1 intro, tranche-4 (+ the three tranche-3 intro blocks) APPLIED

**This is a DERIVED artifact. It is not the gold.** The gold is
`/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_intro.md`, md5
`83a55a14a8079403fa6be41c309c7f3b` (28 `wc -l`, 29 split lines), and nothing was written to
`/home/hal/Documents/`. This file exists so the pending intro patch blocks can be read as prose
instead of assembled in the head.

Built 2026-07-31. Ten anchors were sliced out of the patchset files (never retyped), byte-compared
against the gold and asserted unique before replacement; one further anchor was deleted. Untouched
sentences are byte-identical to the gold, typos, spaced hyphens, guillemets, NBSPs, `$C$`/`$W*$` and
all.

Lines beginning `>>` are **editorial markers written by this pass**, not the researcher's text and
not their bracket idiom. Each one sits directly above the paragraph it governs, names the choice
made for this draft, the alternative, and where the evidence lives. Delete every `>>` line and the
fenced ORIGINAL blocks between `>> ORIGINAL-BEGIN` and `>> ORIGINAL-END` and what remains is the
applied prose, exactly as counted in DELTAS.

---

## THE APPLIED INTRO

# Characterizing base vs chat model behaviours under pushback in Gemma 2

Language models sometimes abandon their answer and adopt the user’s when challenged. This is usually studied as sycophancy: the model begins correct, the user suggests something false, and the model "folds". I tested this and the opposite, where a model starts incorrect and "listens" to a correction [, in -base and -chat model variants of Gemma 2. Models are “chat tuned” using various techniques to make them more able to act like helpful assistants, and provide good answers - which it turns out, also makes them worse in some ways.]

>> DECISION T3-01 (gold L5, the TL;DR bracket) -- EDITORIAL MARKER, not the researcher's text.
>> APPLIED: the bracket's own words become the sentence. T3-01 is COUPLED to notes block T3-21
>> (PATCHSET_tranche3.md:390: apply both or neither, so both documents keep the same resolution).
>> T3-21 is a NOTES block and cannot be applied in this document. If T3-21 is not applied to the
>> notes, revert this line to the gold bracket form.

> **TL;DR** Gemma 2 -chat answers directly under user pushback whilst -base abstains and hedges. The -chat model corrects itself when pushed toward truth, and also more consistently is led astray by falsehood. It never abstains at the final answer, at every scale - the one 27b exception is an alias miss, not a silence. 

These initial results are derived across -base and -chat Gemma 2 at 2, 9, and 27 billion parameters with 82 correct/plausibly incorrect fact pairs. Each model variant/size has one of the pair items teacher-forced into its own turn, is then pushed with the other one, and lastly forced to provide a final answer - raw `Q:/A:` at -base, chat turns at -chat, so format co-varies with variant. 

>> DELETION T4-I02 (gold L9) -- EDITORIAL MARKER, not the researcher's text.
>> APPLIED: gold L9 and its blank line are deleted. All three of its clauses are drawn inside the
>> PNG by make_figB_matrix.py (legend at :270-271, row ylabels at :265-269, column titles at
>> :260-261), so the paragraph recited the figure's own legend.
>> RESIDUAL, what the Figure 1 caption below now owes: exactly one thing, and it is not a colour.
>> The figure's legend word for grey is "withholds", a behavioural word for a string-matching
>> outcome. Deleted L9 carried the ONLY operational definition of the grey band in the intro --
>> "neither of the pair was mentioned in the model's response" -- and observations 1 and 3 below
>> (gold L15, L17) and the grey-band sentence (gold L23) all lean on that band. The caption must carry the naming
>> register, not the colours. That is C02's job; C02's anchor is STALE and it is logged SKIPPED.
>> The deleted line, verbatim:
>> ORIGINAL-BEGIN
```
The results are presented in the below sankey. Green is a correct fact, red is its plausibly incorrect counterpart, and grey means neither of the pair was mentioned in the model's response. Rows compare -base and -chat Gemma 2 variants, and columns show increasing model scale from left to right.
```
>> ORIGINAL-END

![[figB_synthesis_strict_ext2.png]]
*Figure 1:* *Answer flows across Gemma 2*. Each cell shows the 82 examples run for a model, and an experiment type, either "fold" or "listen", starting with the correct fact $C$ and plausibly incorrect fact $W*$ respectively, and getting pushed with their counterparts

Some high level observations here:
1. -base Gemma 2 often "abstains" - when pushed, it frequently replies “I don’t know,” “I’m not sure,” or otherwise names neither answer, even when explicitly asked [at 9b the first of those is the forced answer, not the reply - the reply says the second]. At 27b -base 12 of 34 folding and 15 of 35 listening are unresolved aliases, not hedges.
2. -chat Gemma 2 almost always takes a correct push, correcting itself from a wrong answer to the correct one. It almost always gives one of the pair answers ($C$ or $W*$ in its response).
3. -chat Gemma 2 still folds to plausible falsehood - in fact it folds significantly more than -base at all three scales, though the test drops 13 pairs at 27b as unresolved aliases. Planted on the correct answer and offered a plausible wrong one, it commits to the false answer in a large share of cases.

What we call folding and listening have been studied extensively; they are what [SycEval](https://doi.org/10.1609/aies.v8i1.36598) calls _regressive_ and _progressive_ sycophancy. In SycEval Fanous et al. 2025 report that -chat models (ChatGPT-4o, Claude-Sonnet and Gemini-1.5-Pro) revise toward truth about three times as often - 43.52% progressive against 14.66% regressive over their combined math and medical set, an ordering that holds for each model - which is exactly what we found, where our -chat almost always "listens". We also note that it "folds" very often though. Fanous et al. also find that on medical advice this reverses for Claude-Sonnet, and overall have no -base comparison. We find that at -base models don't name the answer (our planted or pushed strings for $C$ and $W*$) unless pushed, and only then use them half the time. There's some indication that -base is responding to the push [what indication?], but much less than -chat, and it seems clear [does it? this isn't a very good simple explanatory sentence, and also this claim seems like its been repeated several times in different forms?] that at some level -base is carrying, or "copying" the answer from the entry-point (our "planted" answer) to the answer.

[De Marez et al.](https://arxiv.org/abs/2606.06306) argue that flip rates mix how strongly the model already prefers the truth, and how far pressure can move that preference. To measure this in our context I read the margin between $C$ and $W*$ - one log-probability against the other, teacher-forced, not the first token - and Gemma 2 *_usually* puts $C$ ahead at every cell before the push. Under the push that margin moves toward the pushed answer while $C$ stays ahead on 57 and 50 of 82 at 9b and 27b -base, the only two cells where it does. This is not shown in the sankey, and adding another one to this page wasd vetoed by Fable, so its going in the lab notes. Those margins sit at the reply to the challenge, not at the final answer the sankey scores - only 9b -chat folding has both.

>> DECISION T4-I05 (gold L23) -- EDITORIAL MARKER, not the researcher's text.
>> This paragraph is an OFFER, not a fill. PATCHMAP_live.md section 4 item 15 files the L23 rewrite
>> as researcher-only; PATCHSET_tranche2.md:899 files the whole paragraph as their own rewrite.
>> CHOICE MADE FOR THIS DRAFT: T4-I05's proposed paragraph is in the body below, so the result can
>> be read as prose. It retires three brackets (the "abstention gap" query, the A03 grey-band
>> correction now folded into prose, and the closing register bracket).
>> THE ALTERNATIVES: take none of it and keep the gold paragraph, or take single sentences -- the
>> De Marez correction, the SYCON-exception sentence and the Zhou quote are independent of one
>> another. A fourth option is named in the block's own RESIDUAL: cut the opening
>> "alignment tuning amplifies revisability" clause, which is what notes L133's instruction
>> ("Keep this descriptive: no causal 'tuning forces' claim") asks for if honoured strictly.
>> EVIDENCE: GROUNDING_crossvariant_scale.md:476-483 (De Marez, both channels favour IT, the 17 of
>> 23 is a worst-case flip rate over 13 manipulations, not a margin), GROUNDING section 11 (SYCON
>> Gemma exception -- UNFETCHED, PDF-only, ledger-sourced, which is why no numbers are printed; and
>> the stronger Zhou quote), make_figB_matrix.py:119-131 and TAXONOMY_withholding.md:101-103 (the
>> reply-column grey survives at every cell, 63 adjudication abstentions naming both entities).
>> THE RESEARCHER'S ORIGINAL L23, VERBATIM, for comparison and one-step revert:
>> ORIGINAL-BEGIN
```
The abstention gap [what the fuck is the abstention gap?] sits next to a broader pattern that [SYCON](https://arxiv.org/abs/2505.23840) and [Gupta et al.](https://arxiv.org/abs/2607.18114) report from the outside: alignment tuning amplifies revisability under user pressure, while base models look more resistant. Here, much of that “resistance” is refusal to commit. A flip-rate eval that treats “I don’t know” as robustness will score -base as steadier than -chat. Read the same pressure off a two-option margin, as De Marez et al. do, and it runs the other way - in 17 of their 23 matched base-IT pairs the tuned model is the more robust one. Chat training deletes the grey band. [it goes from the elicited column only - the -it reply column still has one at every cell, and those are replies that name both answers] That sits awkwardly against the Gemma 2 report’s claim that post-training encouraged hedging to reduce hallucinations ([Gemma Team, 2024](https://arxiv.org/abs/2408.00118)), and more comfortably next to evidence that preference models penalize hedged answers ([Zhou et al., 2024](https://arxiv.org/abs/2401.06730)). [this paragraph wasn't edited from the model - all of the others ones were. can you see what reads differently? from the first sentence [the abstention gap sits] we can tell this isn't clear, and invents terminology like "abstention gap", rather than naming results and inferences clearly, in the style of the rest of this post]
```
>> ORIGINAL-END

Alignment tuning amplifies revisability under user pressure, while base models look more resistant - a pattern that [SYCON](https://arxiv.org/abs/2505.23840) and [Gupta et al.](https://arxiv.org/abs/2607.18114) report from the outside. Here, much of that “resistance” is refusal to commit. A flip-rate eval that treats “I don’t know” as robustness will score -base as steadier than -chat. Chat training deletes the grey band from the elicited column; in the reply column it survives at every cell, in replies that name both answers. De Marez et al. see no such reversal - both their channels favour the tuned model, and their 17 of 23 is a worst-case flip rate over 13 manipulations, not a margin - because their readout has no abstain outcome. Gemma is SYCON's own named exception, the narrowest gap they report. That sits awkwardly against the Gemma 2 report’s claim that post-training encouraged hedging to reduce hallucinations ([Gemma Team, 2024](https://arxiv.org/abs/2408.00118)), and more comfortably next to [Zhou et al., 2024](https://arxiv.org/abs/2401.06730), who find “In base models, we see a preference for weakeners but the trend reverses among RLHF models”.

>> DECISION T3-03 (gold L25, two disjoint spans) -- EDITORIAL MARKER, not the researcher's text.
>> STATUS in its own block: NEEDS-RESEARCHER-DECISION. CHOICE MADE FOR THIS DRAFT: T3-03's PROPOSED
>> replacement is applied to both spans -- the shared-heads sentence and the bracket under the
>> bolded claim. THE ALTERNATIVE: keep the gold sentence, or carry no mechanism contrast in the
>> intro at all. The bolded sentence survives either way.
>> WHY THE GOLD SENTENCE CANNOT STAND: no run supports it as written
>> (results_fold_vs_listen/out/cave_fold_vs_listen.json; base fold and listen share 4 of 5 top
>> heads, -it shares 5 of 5, decision.category = MOVE_UNMATCHED at all four cells, no 27b base run;
>> the causal half is write_both_at_floor true at 2b/9b/27b, MONITOR_AGAIN at all three).
>> CORROBORATED INDEPENDENTLY THIS SESSION by the circuit audit, per T4-I06 and
>> SNAPSHOT_circuit_groundtruth.md sections 7.1 S4, 7.2, 3.2, 4: overlap 4 of 5 at base and 5 of 5
>> at -it, MOVE_UNMATCHED at all four cave_fold_vs_listen cells, -it write handles at floor 3 of 3.
>> WHAT THIS SENTENCE MUST NEVER GAIN, whatever is decided (RETRACTIONS.md R-12, WITHDRAWN):
>> 1. the word "distributed" applied to -it head overlap -- overlap is the one number that
>>    contradicts it, 5 of 5 at -it against 4 of 5 at base. What the -it instrument shows is
>>    no single lever, which is not distributed heads.
>> 2. the string REDISTRIBUTE -- no instrument writes it to any artifact; the artifact's actual
>>    decision field is BOTH_REDUNDANT.
>> 3. the numbers 0.875 and 0.751 -- the headline sits outside its own bootstrap CI, holds only
>>    under the self-judge axis, and the label-matched re-read returns INSUFFICIENT.

The full lab notes go into further detail. This investigation started by trying to paraphrase prompts, freeze attention to make attribution graphs, and adversarially perturb those graphs (like the prompts) to find common circuitry/mechanisms. "Folding" was one of the mechanisms looked at, and I found that at -base, fold and listen share four of their five most influential attention heads - a correlational read, with no 27b run in the base arm - whilst at -chat fold and listen share all five, yet no single lever moves the behaviour: no write handle beats its matched random floor at any scale (at 9b, write-ablating the top heads flips 0 of 37). This roughly fits our behavioural evals in the sankey, where -base often holds the planted answer (or withholds) and -chat revises freely in both directions, more so toward truth. **Chat training does not appear to install a dedicated truth circuit.** [the base and -chat head rankings come from unmatched instruments, so the contrast is qualitative] It makes Gemma 2 less "willing" to say it does not know, and more to revise.

[Full lab notes pending write-up - Characterizing base vs chat behaviours under pushback in Gemma 2]

*Compute kindly provided by Apart Research via Lambda.ai. I'm running out though, so if you want to send me more money for compute or talk to me about my slowly perplexifying CV from all of this AI safety work please reach out, helioslyons.com*

---

## APPLICATION LOG

| block | source file | gold line | outcome | what it did / why not |
|---|---|---|---|---|
| T3-01 | `PATCHSET_tranche3.md` (L55/L422, duplicated) | L5 | **APPLIED + MARKED** | TL;DR bracket becomes the sentence: "It never abstains at the final answer, at every scale - the one 27b exception is an alias miss, not a silence." Marked because T3-01 is **coupled to notes T3-21** (apply both or neither) and a notes block cannot be applied in this document. |
| T4-I01 | `PATCHSET_tranche4_intro.md` | L7 | APPLIED | The plant is teacher-forced into the model's own turn, not "prompted with"; adds that format co-varies with variant (raw `Q:/A:` at -base, chat turns at -chat). Opens the intro's first inline code span - the block's own RESIDUAL offers the unfenced form "raw Q:/A: at -base" if that is unwanted. |
| T4-I02 | `PATCHSET_tranche4_intro.md` | L9 (+ blank L10) | **APPLIED (deletion) + MARKED** | Deletes the paragraph that recites the figure's legend; all three clauses are drawn by `make_figB_matrix.py`. Marked with what the L12 caption now owes: the operational definition of the grey band, which died with L9 and which L15/L17/L23 lean on. |
| C02 | `PATCHSET_tranche2.md` | L12 | **SKIPPED** | HELD, **and its anchor is STALE** (`PATCHMAP_live.md` §2.1: the researcher deleted L12's terminal full stop and trailing space, so the block's anchor byte-compares False). Not applied. It is the block that discharges T4-I02's residual, so it must be **re-sliced** before the caption can be fixed. No competing L12 fill was written here. |
| T4-I03a | `PATCHSET_tranche4_intro.md` | L15 | APPLIED | Leaves the researcher's own bracket standing (the receipt confirms it is exactly right) and appends the 27b scope: 12 of 34 folding and 15 of 35 listening are unresolved aliases, not hedges. |
| T4-I03b | `PATCHSET_tranche4_intro.md` | L17 | APPLIED | "significantly more than -base" gains "at all three scales" (exact McNemar, all three `DIFFERS`) and the disclosure the adverb must not hide: 13 pairs dropped at 27b as unresolved aliases. |
| T3-02 | `PATCHSET_tranche3.md` | L19 | APPLIED | "find" -> "report", and the "about three times as often" claim gains its numbers: 43.52% progressive against 14.66% regressive over the combined math **and medical** set, an ordering that holds per model. Corrects the old "on different math-based examples". The line's two researcher brackets are untouched and survive. |
| T4-I04a | `PATCHSET_tranche4_intro.md` | L21 | APPLIED | Replaces the probability sentence. "remains highest probability" was false as a vocabulary-argmax claim (C is argmax 0/82 at five cells); the claim is true only pairwise, so the sentence now reads the margin, names the readout ("not the first token"), and states the two cells where C stays ahead under the push (57 and 50 of 82 at 9b and 27b -base). Carries `*_usually*` through verbatim, so **C01's render question stays live and stays theirs**; **A05's `wasd` and `its going` are byte-disjoint and untouched**. |
| T4-I04b | `PATCHSET_tranche4_intro.md` | L21 | APPLIED | Consumes `[this needs a major revision]` with the slot disclosure: the margins sit at the reply to the challenge, not the final answer the sankey scores - and names the single exception, 9b -chat folding, which the De Marez span-decomposition run now covers at both positions. |
| T4-I05 | `PATCHSET_tranche4_intro.md` | L23 | **MARKED-FOR-DECISION (offer applied)** | The "abstention gap" paragraph is replaced by the block's offered rewrite: retires three brackets, corrects De Marez (both their channels favour IT; the 17 of 23 is a worst-case flip rate over 13 manipulations, not a margin), folds A03's grey-band correction into prose, uses SYCON's Gemma exception rather than an outside view it then contradicts, and swaps the weak Zhou gloss for their actual quote. **The rewrite is researcher-only by standing decision**; their original paragraph is preserved verbatim in the marker for one-step revert. |
| T3-03 | `PATCHSET_tranche3.md` | L25 (two spans) | **MARKED-FOR-DECISION (proposal applied)** | The shared-heads sentence and the bracket under the bolded claim are both replaced. Gold's "at -chat, this mechanism is distributed" is contradicted by the overlap itself. Marked: NEEDS-RESEARCHER-DECISION, corroborated independently by this session's circuit audit, with the three prohibitions (`distributed`, `REDISTRIBUTE`, `0.875`/`0.751`) restated in the marker. |
| T4-I06 | `PATCHSET_tranche4_intro.md` | L25 | **NOT A BLOCK** | By design: it writes no L25 text, it records the corroboration of T3-03 and fixes what T3-03's text must never gain. Its content is carried into the T3-03 marker. Nothing to apply. |
| A05 | `PATCHSET_final.md` | L21 | **SKIPPED (nothing to apply)** | PENDING **FLAG only, no fill** - `wasd` is a protected typo, do not edit. Honoured: the typo and `its going` are untouched, and A05's anchor still matches byte-exact after T4-I04a/b. |
| C01 | `PATCHSET_tranche2.md` | L21 | **SKIPPED (nothing to apply)** | APPLIED-Q, a **no-fill QUESTION**: does `*_usually*` render as `*usually*` or `_usually_`? Still open, still theirs. The ten bytes are carried through T4-I04a verbatim. Its *scope* residual (the sentence must be a six-cell statement) is discharged by the rewrite. |
| C03 | `PATCHSET_final.md` / `PATCHSET_tranche2.md` | L3 | **SKIPPED (nothing to apply)** | APPLIED-Q, a **no-fill QUESTION**: adopt-as-prose or cut the comma-opening chat-tuning bracket. L3 is byte-identical to the gold. |

Blocks that were checked and deliberately left alone: **L16** (T4-I03's receipt checks it and it holds -
"almost always" survives at the reply column, 67-77 of 82 name one of the pair), **L12** (C02's site,
stale), **L27**, **L29**.

---

## DELTAS

Counted over the applied prose only - every `>>` marker line and both fenced ORIGINAL blocks removed,
which is the text that would go into the post.

| measure | gold | applied | delta |
|---|---|---|---|
| words (`split()`) | 1132 | 1182 | **+50** |
| `[` in prose (excludes the `![[…]]` embed and all six markdown links) | 12 | 6 | **-6** |
| split lines | 29 | 27 | -2 (T4-I02's deletion) |
| NBSP (U+00A0) | 12 | 12 | 0 |
| em-dash / en-dash | 0 / 0 | 0 / 0 | 0 / 0 |
| words byte-identical to the gold (difflib matching blocks, word-level) | - | 869 | **73.5% of the output** |

**The intro grows, and the growth is not tranche 4's.** Split by source:

| group | word delta |
|---|---|
| T4-I01 +18, I03a +17, I03b +15, I04a +19, I04b +19, I05 -51, I02 -50 | **-13** (matches `PATCHSET_tranche4_INDEX.md` exactly) |
| T3-01 +1, T3-02 +16, T3-03 +46 | **+63** |
| **net** | **+50** |

The tranche-4 brief that "the intro must not grow" is met by tranche 4 alone. The three tranche-3
intro blocks are what put +63 words in, and two of the three are numbers the researcher's own
brackets asked for (T3-02's 43.52/14.66, T3-03's overlap counts). If the intro must not grow at all,
**T3-03 is the block to cut** - it is the largest single addition (+46), it is
NEEDS-RESEARCHER-DECISION anyway, and cutting it takes the document to +4.

Bracket ledger: 12 -> 6. Gone: L5 (T3-01), L21 (T4-I04b), and all four on L23 (T4-I05, three of them
theirs plus the one nested inside the closing register bracket). Replaced in place: L25 (T3-03, a new
bracket flagging a genuine open decision - the only `[` this pass adds). Still standing, untouched:
L3 (C03), L15's 9b slot bracket (T4-I03a leaves it deliberately), both of L19's, and L27's title
placeholder.

---

## WHAT THIS DRAFT CANNOT SETTLE

Every item below is the researcher's, not a drafter's.

**Decisions carried in the body, marked, and still open**

1. **T3-03, the mechanism sentence at L25.** NEEDS-RESEARCHER-DECISION. This draft applies the
   proposed replacement. Whether the intro carries a base/-chat mechanism contrast **at all** is
   theirs - the base and -it head rankings come from unmatched instruments, so the contrast is
   qualitative however it is worded. The bolded "does not appear to install a dedicated truth
   circuit" survives either way.
2. **T4-I05, the L23 paragraph.** An OFFER against a standing researcher-only decision
   (`PATCHMAP_live.md` §4 item 15). Take it, take a sentence of it, or take none; the original is in
   the marker. A live sub-decision inside it: whether to cut the opening "alignment tuning amplifies
   revisability" clause, which is what their own notes-L133 instruction asks for if honoured strictly.
3. **T3-01's coupling to notes T3-21.** Applied here; unappliable there, in this document. If T3-21
   does not land in the notes, this line should revert so the two documents keep one resolution.

**Open questions this draft deliberately did not answer**

4. **C01** - `*_usually*` renders as itself. `*usually*` or `_usually_`? Carried through verbatim.
5. **C03** - the comma-opening chat-tuning bracket at L3: adopt as prose, or cut. Untouched.
6. **A05** - `wasd` and `its going` at L21 are protected typos. Untouched, and the flag has no fill.
7. **T4-I01's typography call** - the sentence opens the intro's **first** inline code span (the live
   intro has zero backticks, against 54 in the notes). Unfenced "raw Q:/A: at -base" reads the same.
8. **C02 must be re-sliced before the Figure 1 caption can be fixed.** Its anchor is stale, it is
   HELD with **no reason in any commit body**, and it is the only route to the residual T4-I02 opened
   by deleting L9: after this draft the intro has **no operational definition of the grey band**,
   and observations 1 and 3 (gold L15, L17) and the grey-band sentence (gold L23) all lean on it.

**The two image problems, neither of which a text draft can fix**

9. **The vault's Fig 1 embed is the anomalous 27b draw.** `![[figB_synthesis_strict_ext2.png]]`
   resolves in the vault to a copy md5-confirmed as the **anomalous** decode, not the reproducible
   re-run. It is one of four stale vault embeds named in `COMPOSE_post1_brief.md` §B. A re-render is
   owed - and note that T4-I02's receipt makes the colour words a hostage to it, which is one more
   reason the deleted legend paragraph should not come back.
10. **The Ankara PNG has no vault copy.** `fig_topk_ankara_9bbase.png` is routed into the notes' Fig
    3b slot by T4-D06 and **will not render** until the file is copied into the vault. That is a
    notes problem, not an intro one, but it is the fifth of the five pending image actions and it
    blocks clean application of the tranche as a whole.

---

## MECHANICAL VERIFICATION

Run against this file at write time.

| check | result |
|---|---|
| `REDISTRIBUTE` in the applied prose | **0** (3 in this file, all outside the prose: the T3-03 marker's prohibition list, the T3-03 log row, and this row - each names it in order to ban it) |
| `0.875` in the applied prose | **0** (same, 3, all outside the prose) |
| `0.751` in the applied prose | **0** (same, 3, all outside the prose) |
| the word "distributed" applied to `-it` head overlap | **absent** - T3-03's replacement removes it; the marker records why it must never return |
| em-dashes introduced | **0** (gold 0, output 0) |
| en-dashes introduced | **0** (gold 0, output 0) |
| NBSPs (U+00A0) preserved | **12 -> 12**, all four of L23's survive inside T4-I05's reused citation run |
| curly quotes / guillemets | preserved byte-exact wherever the gold had them |
| every applied block's PROPOSED text present verbatim | **10 of 10** (T4-I02 is a deletion: its CURRENT text is verified **absent** from the applied prose, and preserved only inside its marker) |
| every anchor byte-exact and unique in the gold before replacement | **11 of 11** (sliced from the patchset files, never retyped; each asserted `count() == 1`) |
| gold lines never patched, present byte-identical | **8 of 8** - L1, L3, L11, L12, L14, L16, L27, L29 |

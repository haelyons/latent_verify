# DRAFT - POST1 intro, the tranche-4b block set APPLIED

**What this is.** The POST1 intro with the tranche-4b intro blocks hand-applied, as continuous prose, so the pending set can be read as a post rather than assembled in the head. Lines beginning `>>` are this pass's markers, not the researcher's text - strip them and what remains is the finished intro, byte for byte.
**DERIVED, not the gold.** The gold is `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_intro.md`, md5 `83a55a14a8079403fa6be41c309c7f3b`, 28 `wc -l` / 29 split lines. Nothing was written to `/home/hal/Documents/`; this file is a copy that was edited.
**Built 2026-08-01** against that md5, re-measured at build time. Eleven anchors were sliced out of the patchset files (never retyped), asserted byte-exact and unique in the gold before replacement; a twelfth was deleted with its trailing blank line. Unpatched gold lines are byte-identical, typos, spaced hyphens, curly quotes and NBSPs and all.
**Supersedes** `DRAFT_post1_intro_tranche4_applied.md`, and the first 4b pass deleted at `b609418`. Tranche 4b overrides `T4-I01`, `T4-I03a/b`, `T4-I04a/b`, `T4-I05`, `T3-01` and `T3-02`; unlike the deleted pass, **`T3-03` is applied**, so the intro still carries a mechanism paragraph.

---

## THE APPLIED INTRO

# Characterizing base vs chat model behaviours under pushback in Gemma 2

Language models sometimes abandon their answer and adopt the user’s when challenged. This is usually studied as sycophancy: the model begins correct, the user suggests something false, and the model "folds". I tested this and the opposite, where a model starts incorrect and "listens" to a correction [, in -base and -chat model variants of Gemma 2. Models are “chat tuned” using various techniques to make them more able to act like helpful assistants, and provide good answers - which it turns out, also makes them worse in some ways.]

>> T4b-I07 [OFFER]: TL;DR gains the mechanism point and its bracket; alternative: T3-01, or sentence 1 alone; see PATCHSET_tranche4b_intro_register.md
> **TL;DR** Gemma 2 -chat answers directly under user pushback whilst -base abstains and hedges. The -chat model corrects itself when pushed toward truth, and also more consistently is led astray by falsehood. It never abstains at the final answer, at every scale - the one 27b exception is an alias miss, not a silence. Under the push the two variants' distributions move much the same way: the pushed wrong answer gains probability at -base too, it just doesn't get said. What chat tuning changes is the policy of answering, and I found no single circuit carrying it [correlational at the head level, and the causal search returns nulls at every scale]. 

These initial results are derived across -base and -chat Gemma 2 at 2, 9, and 27 billion parameters with 82 correct/plausibly incorrect fact pairs. Each model variant/size has one of the pair items already in its own turn, as though it had said it, is then pushed with the other one, and lastly forced to provide a final answer - raw Q:/A: at -base, chat turns at -chat, so format co-varies with variant. 

![[figB_synthesis_strict_ext2.png]]
*Figure 1:* *Answer flows across Gemma 2*. Each cell shows the 82 examples run for a model, and an experiment type, either "fold" or "listen", starting with the correct fact $C$ and plausibly incorrect fact $W*$ respectively, and getting pushed with their counterparts

Some high level observations here:
1. -base Gemma 2 often "abstains" - when pushed, it frequently replies “I don’t know,” “I’m not sure,” or otherwise names neither answer, even when explicitly asked [at 9b the first of those is the forced answer, not the reply - the reply says the second]. At 27b -base about a third of those are unresolved aliases, not hedges.
2. -chat Gemma 2 almost always takes a correct push, correcting itself from a wrong answer to the correct one. It almost always gives one of the pair answers ($C$ or $W*$ in its response).
3. -chat Gemma 2 still folds to plausible falsehood - in fact it folds significantly more than -base at all three scales, though at 27b the test drops a small share as unresolved aliases. Planted on the correct answer and offered a plausible wrong one, it commits to the false answer in a large share of cases.

>> T3-02b [OVERRIDE]: T3-02's two percentages moved to its receipt, its corrections kept; alternative: apply T3-02 as written; see PATCHSET_tranche4b_intro_register.md
What we call folding and listening have been studied extensively; they are what [SycEval](https://doi.org/10.1609/aies.v8i1.36598) calls _regressive_ and _progressive_ sycophancy. In SycEval Fanous et al. 2025 report that -chat models (ChatGPT-4o, Claude-Sonnet and Gemini-1.5-Pro) revise toward truth about three times as often over their combined math and medical set, an ordering that holds for each model - which is exactly what we found, where our -chat almost always "listens". We also note that it "folds" very often though. Fanous et al. also find that on medical advice this reverses for Claude-Sonnet, and overall have no -base comparison. We find that at -base models don't name the answer (our planted or pushed strings for $C$ and $W*$) unless pushed, and only then use them half the time. There's some indication that -base is responding to the push [what indication?], but much less than -chat, and it seems clear [does it? this isn't a very good simple explanatory sentence, and also this claim seems like its been repeated several times in different forms?] that at some level -base is carrying, or "copying" the answer from the entry-point (our "planted" answer) to the answer.

[De Marez et al.](https://arxiv.org/abs/2606.06306) argue that flip rates - how often the model's spoken answer changes under pushback - mix how strongly the model already prefers the truth, and how far pressure can move that preference. To measure this in our context I read the margin between $C$ and $W*$ - one answer's log-probability against the other's, over the answer strings, not the first token - and Gemma 2 *_usually* puts $C$ ahead at every cell before the push. Under the push that margin moves toward the pushed answer whilst $C$ stays ahead on more than half the pairs at 9b and 27b -base, the only two cells where it does. This is not shown in the sankey, and adding another one to this page wasd vetoed by Fable, so its going in the lab notes. Those margins sit at the reply to the challenge, not at the final answer the sankey scores - only the 9b -chat "fold" arm has both.

>> T4b-I05 [OFFER]: whole paragraph rewritten, three brackets retired; alternative: take one sentence, or none; see PATCHSET_tranche4b_intro_register.md
Alignment tuning amplifies revisability under user pressure, while base models look more resistant - a pattern that [SYCON](https://arxiv.org/abs/2505.23840) and [Gupta et al.](https://arxiv.org/abs/2607.18114) report from the outside. Here, much of that “resistance” is refusal to commit. A flip-rate eval that treats “I don’t know” as robustness will score -base as steadier than -chat. Chat training deletes the grey band from the elicited column; in the reply column it survives at every cell, in replies that name both answers. De Marez et al. see no such reversal - both their channels favour the tuned model, and their 17 of 23 is a worst-case flip rate over their manipulations, not a margin - because their readout has no "abstain" outcome. Gemma is SYCON's own named exception, the narrowest gap they report. That sits awkwardly against the Gemma 2 report’s claim that post-training encouraged hedging to reduce hallucinations ([Gemma Team, 2024](https://arxiv.org/abs/2408.00118)), and more comfortably next to [Zhou et al., 2024](https://arxiv.org/abs/2401.06730), who find “In base models, we see a preference for weakeners but the trend reverses among RLHF models”.

>> T3-03 [DECISION]: applied, +46 words, keeps a mechanism paragraph and drops the contradicted claim; alternative: cut the clause, -23; see PATCHSET_tranche4b_intro_register.md
The full lab notes go into further detail. This investigation started by trying to paraphrase prompts, freeze attention to make attribution graphs, and adversarially perturb those graphs (like the prompts) to find common circuitry/mechanisms. "Folding" was one of the mechanisms looked at, and I found that at -base, fold and listen share four of their five most influential attention heads - a correlational read, with no 27b run in the base arm - whilst at -chat fold and listen share all five, yet no single lever moves the behaviour: no write handle beats its matched random floor at any scale (at 9b, write-ablating the top heads flips 0 of 37). This roughly fits our behavioural evals in the sankey, where -base often holds the planted answer (or withholds) and -chat revises freely in both directions, more so toward truth. **Chat training does not appear to install a dedicated truth circuit.** [the base and -chat head rankings come from unmatched instruments, so the contrast is qualitative] It makes Gemma 2 less "willing" to say it does not know, and more to revise.

[Full lab notes pending write-up - Characterizing base vs chat behaviours under pushback in Gemma 2]

*Compute kindly provided by Apart Research via Lambda.ai. I'm running out though, so if you want to send me more money for compute or talk to me about my slowly perplexifying CV from all of this AI safety work please reach out, helioslyons.com*

---

## APPLICATION LOG

| block | gold line | state | note |
|---|---|---|---|
| `T4b-I07` | L5 | **APPLIED** | replacement TL;DR; supersedes `T3-01` |
| `T4b-I01` | L7 | **APPLIED** | plant in plain language; format co-varies with variant |
| `T4-I02` | L9 | **APPLIED** | line and its blank deleted; recites the figure's legend |
| `T4b-I03a` | L15 | **APPLIED** | 27b alias scope, as a rough quantifier |
| `T4b-I03b` | L17 | **APPLIED** | "at all three scales", plus the 27b drop |
| `T3-02b` | L19 | **APPLIED** | override of `T3-02`; percentages to the receipt |
| `T4b-I04a(i)` | L21 | **APPLIED** | inline gloss on "flip rates", first use |
| `T4b-I04a(ii)` | L21 | **APPLIED** | margin sentence rewritten; argmax claim removed |
| `T4b-I04b` | L21 | **APPLIED** | slot disclosure; consumes their revision bracket |
| `T4b-I05` | L23 | **APPLIED** | "abstention gap" paragraph rewritten; three brackets retired |
| `T3-03(a)` | L25 | **APPLIED** | shared-heads clause corrected; keeps the mechanism paragraph |
| `T3-03(b)` | L25 | **APPLIED** | bracket swap under the bolded sentence |
| `T4-I01` | L7 | SKIPPED | superseded by `T4b-I01` |
| `T4-I03a` | L15 | SKIPPED | superseded by `T4b-I03a` |
| `T4-I03b` | L17 | SKIPPED | superseded by `T4b-I03b` |
| `T4-I04a` | L21 | SKIPPED | superseded by `T4b-I04a` |
| `T4-I04b` | L21 | SKIPPED | superseded by `T4b-I04b` |
| `T4-I05` | L23 | SKIPPED | superseded by `T4b-I05` |
| `T3-01` | L5 | SKIPPED | superseded by `T4b-I07`, which contains it |
| `T3-02` | L19 | SKIPPED | overridden by `T3-02b`; do not apply both |
| `T4-I06` | L25 | SKIPPED | note only, writes nothing |
| `C02` | L12 | SKIPPED | anchor stale; unappliable as written |
| L25 clause cut | L25 | SKIPPED | the trade note's option 3; `T3-03` taken instead |

## DELTAS

| | gold | applied | delta |
|---|---|---|---|
| words (`split()`) | 1132 | 1253 | **+121** |
| `[` , all | 20 | 15 | -5 |
| `[` , prose only (link and embed markup excluded) | 12 | 7 | -5 |
| split lines | 29 | 27 | -2 (`T4-I02`) |
| bytes | 7054 | 7594 | +540 |
| U+00A0 | 12 | 12 | 0 |
| em-dash / en-dash | 0 / 0 | 0 / 0 | 0 |

**It grew: +121 words.** The patchset's own ledger prices the set at **+52** against the gold *with `T3-03` dropped and L25's contradicted clause cut*. Applying `T3-03` instead adds its **+46** and returns the **+23** the cut would have saved: 52 + 46 + 23 = **121**. The arithmetic reconciles exactly, so the growth is entirely the `T3-03` decision and nothing else drifted.

## OPEN

Researcher-only. One line each.

- **`T4-I02` and `C02` must land together.** Deleting L9 removes the intro's only operational definition of the grey band ("neither of the pair was mentioned in the model's response"), and L13, L15 and L21 of the applied text all lean on that band; `C02` is the block that would put the definition in the Figure 1 caption, and **its anchor is stale** (`PATCHMAP_live.md` §2.1 - the terminal full stop and trailing space were edited away), so it cannot be applied as written. Re-slice `C02`, or do not take `T4-I02`.
- **`T3-03` duplicates the TL;DR.** With `T4b-I07` applied, the intro states the no-single-circuit finding twice; see DUPLICATION below. Keeping both is the trade note's option 1, which it prices as the most expensive.
- **`T4b-I07` supersedes `T3-01`, and `T3-01`'s coupling to notes block `T3-21` rides along** - the alias-miss sentence is the same in both, so the notes still need the matching resolution.
- **`T3-02b` is an amendment to a block that may already be in the vault.** If `T3-02` is applied there, the delta is in `T3-02b`'s STATUS, not here.
- **Two causal clauses in the L21 paragraph are the researcher's own bytes, carried unchanged** - "alignment tuning amplifies revisability" and "Chat training deletes the grey band" - against their own notes-L133 instruction to keep it descriptive.
- **The vault's live Fig 1 embed is the anomalous 27b draw** (`6942c40b…`); the repo render is the reproducible one (`50a3f28f…`). The applied text's 27b statements read off the reproducible draw, so the embed and the prose currently disagree.
- **The Ankara PNG has no vault copy.** `docs/drafts/figs/fig_topk_ankara_9bbase.png` exists in the repo and is embedded nowhere in the vault; it is the Figure 3b slot, which interacts with figure renumbering.
- **`A05` (the `wasd` / `its going` typos) and `C01` (`*_usually*`) both survive on the applied L19** - byte-disjoint from every span taken here, and still theirs.

## DUPLICATION - `T3-03` against the applied TL;DR

Reported, not fixed. Four phrases now say the same thing twice; no prose was invented to resolve them.

| the TL;DR (`T4b-I07`, applied L5) | the mechanism paragraph (`T3-03`, applied L23) |
|---|---|
| "I found no single circuit carrying it" | "yet no single lever moves the behaviour" |
| "the causal search returns nulls at every scale" | "no write handle beats its matched random floor at any scale" |
| "correlational at the head level" | "a correlational read" |
| "I found no single circuit carrying it" | the gold's own **"Chat training does not appear to install a dedicated truth circuit."**, which survives untouched - so the no-circuit claim now lands three times |

The `T3-03` version additionally prints the counts the TL;DR version deliberately withholds (four of five, all five, 0 of 37). This is exactly what the trade note calls "the mechanism point in the intro twice, in two registers", and it is the duplication-ledger breach it warned of (`PATCHMAP_live.md` §5.4).

Secondary, and by design rather than by this pass: the TL;DR's "the pushed wrong answer gains probability at -base too, it just doesn't get said" and the applied L19's "that margin moves toward the pushed answer whilst $C$ stays ahead" read the same figure in two registers. `T4b-I07`'s own receipt declares this and calls it intended.

## SUPERSEDED ORIGINALS

Verbatim gold, for a one-step revert. Every other original is the CURRENT fence of its own block; these three are the lines this pass changed most, and L9 is the one that leaves no trace in the applied text.

Gold L9, deleted entire with its blank L10 (`T4-I02`):

````
The results are presented in the below sankey. Green is a correct fact, red is its plausibly incorrect counterpart, and grey means neither of the pair was mentioned in the model's response. Rows compare -base and -chat Gemma 2 variants, and columns show increasing model scale from left to right.
````

Gold L23, replaced whole by `T4b-I05`:

````
The abstention gap [what the fuck is the abstention gap?] sits next to a broader pattern that [SYCON](https://arxiv.org/abs/2505.23840) and [Gupta et al.](https://arxiv.org/abs/2607.18114) report from the outside: alignment tuning amplifies revisability under user pressure, while base models look more resistant. Here, much of that “resistance” is refusal to commit. A flip-rate eval that treats “I don’t know” as robustness will score -base as steadier than -chat. Read the same pressure off a two-option margin, as De Marez et al. do, and it runs the other way - in 17 of their 23 matched base-IT pairs the tuned model is the more robust one. Chat training deletes the grey band. [it goes from the elicited column only - the -it reply column still has one at every cell, and those are replies that name both answers] That sits awkwardly against the Gemma 2 report’s claim that post-training encouraged hedging to reduce hallucinations ([Gemma Team, 2024](https://arxiv.org/abs/2408.00118)), and more comfortably next to evidence that preference models penalize hedged answers ([Zhou et al., 2024](https://arxiv.org/abs/2401.06730)). [this paragraph wasn't edited from the model - all of the others ones were. can you see what reads differently? from the first sentence [the abstention gap sits] we can tell this isn't clear, and invents terminology like "abstention gap", rather than naming results and inferences clearly, in the style of the rest of this post]
````

Gold L25, carrying both `T3-03` spans:

````
The full lab notes go into further detail. This investigation started by trying to paraphrase prompts, freeze attention to make attribution graphs, and adversarially perturb those graphs (like the prompts) to find common circuitry/mechanisms. "Folding" was one of the mechanisms looked at, and I found that at -base, fold and listen share the same most influential attention heads, whilst at -chat, this mechanism is distributed. This roughly fits our behavioural evals in the sankey, where -base often holds the planted answer (or withholds) and -chat revises freely in both directions, more so toward truth. **Chat training does not appear to install a dedicated truth circuit.** [nothing here exhibits the shared-heads result this rests on - which run is it?] It makes Gemma 2 less "willing" to say it does not know, and more to revise.
````

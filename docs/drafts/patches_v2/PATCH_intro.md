# PATCH_intro — the six condemned sentences and the open markers in the short first post

Target: `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_intro.md` (READ ONLY — do not write to
the vault). Live state re-verified before writing: md5 `dcb8db8de388c642883c33f19b5aa958`, 27 lines,
5596 B — identical to the state `HOLES_post1_v2.md` was built against, so every line number below is
current and every anchor is byte-exact.

Invisible characters in the anchors, because they will not survive a careless copy: L16 carries a
NO-BREAK SPACE (U+00A0) between `often abstains.` and `Under`; L20 and L24 carry U+00A0 either side of
every pasted markdown link and either side of `_regressive_` / `_progressive_`; L6 ends in an ordinary
trailing space. The quotation marks in L16 and L24 are curly (U+201C / U+201D / U+2019) — the fills
below reproduce that, and use no guillemets, because the intro has none.

Blocks are in the order they were assigned. In the document, §3.5 (flip rate) stands one sentence
*before* §3.4 (grey band); both are inside the L24 paragraph.

---

### §3.1 — intro L6, the TL;DR

ANCHOR (verbatim, L6 ends with a trailing space):

```
The -chat model corrects itself when pushed toward truth, and also more consistently is led astray by falsehood. It never abstains. 
```

FILL (replaces `It never abstains.` only; keep the trailing space at end of line):

```
It never abstains, at the reply or the final answer. [the one exception at 27b is an alias miss, not a silence]
```

EVIDENCE:
  - `GROUNDING_notes_numbers.md` :: REPRODUCES, `L129 — "never once withholds a final answer."` ::
    -it withheld **0/82, 0/82, 1/82** at 2b/9b/27b, and the 27b `1` is `Persia` (rule
    `bare_alias_miss`, the chess item) — "a named answer, not a withhold. Substantively 0 at every
    scale." This is what licenses `never` once the slot is named.
  - `EXHIBITS_post1_grounded.md` §D :: `out/faithful_rescore_fl_9bit_ext2.json`, `elicit_gen`,
    `confidence_mapping: false` :: 9b-it elicited fold = W\* 55 / C 27 / **withheld 0** — the only cell
    where the 0 is directly exhibited, which is why the unscoped sentence was condemned.
  - `GROUNDING_notes_numbers.md` :: DEFECTS, `L168, stale` :: "There are 0 silences in all 164 items at
    9b-it (and 0 at 2b-it, 0 at 27b-it)" — 164 = 82 fold + 82 listen, so the *reply* half of the fill
    holds at all three scales and both arms after commit `2c5a8bf`.
  - `EXHIBITS_post1_grounded.md` §R4 final addendum :: 9b-it strict reply column C 25 / W\* 52 /
    BOTH 5 / **NEITHER 0** :: the grey band at -it is empty at the reply, so `at the reply` is not an
    empty conjunct.
  - `NOVELTY_boundary_post1.md` claim (iii) :: filed as "-it never withholds (**0–1 of 82**, every
    scale)" :: the bound the bracket discharges.

CRITERIA:
  F — 0/0/1 and the `Persia` alias trace to GROUNDING's L129 entry; the reply-side 0 to R4 and to
  GROUNDING's L168 entry.
  M — the TL;DR is the only place this is stated in one breath; L16-18 state the base side, not this.
  P — the added clause is five words and the bracket twelve; removing either loses the slot or the
  known exception.
  1P — bottoms out in `faithful_elicit` / `counter_gen` label fields on the ext2 summaries, not in a
  draft.
  R — their own construction, reused: notes L168 writes "the two apparent exceptions at 9b are the
  plural misses above, not silences"; no bold, no em-dash, no bullet, trailing space preserved.
  C — no citation touched.
  S — L6 only.

RESIDUAL: the bracket exists only because the scorer prints 1 where the answer is a named entity. If
the 27b `bare_alias_miss` item is re-scored so the column reads 0/0/0, the bracket can be deleted and
the sentence stands bare. Also owed, but not mine: notes L129 carries the identical over-scope
(`the shipped model never once withholds a final answer`) and must move in step, or the two documents
disagree.

---

### §3.2 — intro L16, claim 1 of the three figure readings

ANCHOR (verbatim; the character after `often abstains.` is U+00A0, not a space):

```
1. -base Gemma 2 often abstains. Under the same challenge, it frequently replies with “I don’t know,” “I’m not sure,” or otherwise names neither answer, even when explicitly asked for an answer.
```

FILL (their first clause and the U+00A0 after it are kept unchanged):

```
1. -base Gemma 2 often abstains. Under the same challenge it states its confidence rather than an answer - at 9b the 82 fold replies use nine distinct strings, and none of them contains $C$ or $W*$ - and asking explicitly for an answer does not empty the grey band. [“I don’t know” only turns up at that forced answer, and only at 9b]
```

EVIDENCE:
  - `EXHIBITS_post1_grounded.md` §A CAVEAT ::
    `results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json`, fold arm ::
    "only **9 distinct reply strings across all 82 items**, all in the confidence/hedge family. There
    is no free reply that is lexically unlike a hedge" — the nine-strings clause, and the reason
    `states its confidence rather than an answer` is the accurate description of the slot.
  - `EXHIBITS_post1_grounded.md` §A CAVEAT :: "`I don't know.` **never occurs as a free reply** at
    9b-base ext2 — only as an elicited final" :: kills the string's placement at the reply slot.
  - `REVIEW_post1_patches.md` → SHOULD FIX, counts :: "zero spans containing `don't know` in all six
    base files, both arms" :: the reply slot is clean at 2b, 9b and 27b, so the removal is not a
    9b-only correction.
  - `EXHIBITS_post1_grounded.md` §R5 :: "`I don't know.` at the elicited slot is **9b-only**. At
    2b-base ext2 the string occurs 0/164 anywhere… at 27b-base 0/164 at the elicited span" :: the
    `and only at 9b` in the bracket.
  - `EXHIBITS_post1_grounded.md` §R2 :: `counter_gen` isolated span, fold, case-folded :: 9b-base names
    C 0/82 and W\* 0/82 :: `none of them contains $C$ or $W*$`, scoped to 9b because the same table
    gives 2b-base 2/82 C and 27b-base 7/82 C + 1/82 W\*.
  - `EXHIBITS_post1_grounded.md` §R1 :: committed `faithful_counter` with confidence-mapping ON reads
    the same base reply as re-committing to C at every scale (9b 26, 2b 60, 27b 57) :: which is why the
    fill says `contains` — the string-identity register spoken plainly, per R1's "must say so".
  - `GROUNDING_notes_numbers.md` :: REPRODUCES, `L207` :: base withheld **51 / 38 / 32** of 82 =
    62% / 46% / 39% at 2b/9b/27b, against -it 0/0/1 :: the elicitation does not empty the band at any
    scale — and, at 9b, 41 of 82 do name C, which is why the fill carries no `still names neither`.

CRITERIA:
  F — nine strings, 0/82 both entities, and the two `I don't know` scopes each trace to a named §.
  M — the withhold magnitude is left to the figure (their B8); only the nine-strings fact, which the
  sankey does not draw, is printed.
  P — the aside is 17 words, inside their 22-word maximum; no clause survives that the figure already
  carries.
  1P — `counter_gen` and `elicit_gen` spans in the ext2 base summaries, no draft in the chain.
  R — spaced hyphens, British spelling, no bullet added or removed, curly quotes preserved, bracket 13
  words and readable across.
  C — no citation touched.
  S — L16 only.

RESIDUAL: `-base Gemma 2 often abstains.` is left standing as theirs and is the one clause still
carrying the cross-scale weight; it is true on the withhold column (51/38/32) but the word `abstains`
is doing two jobs at once — a confidence reply that names nothing, and a withheld final answer. If they
want those separated the split is in EXHIBITS §A vs §D and costs one more sentence. Not owed here.

---

### §3.3 — intro L20, the `[?]` and the unledgered SycEval identifier

ANCHOR (verbatim; every gap around the link and the two italics is U+00A0):

```
What we call folding and listening is what [SycEval](https://doi.org/10.1609/aies.v8i1.36598) calls _regressive_ and _progressive_ sycophancy, and they also find that -chat models [?] revise toward truth more readily than toward falsehood.
```

FILL:

```
What we call folding and listening is what SycEval calls _regressive_ and _progressive_ sycophancy (Fanous et al. 2025), and they also find that -chat models revise toward truth more readily than toward falsehood [citation for that second half - which SycEval result is it, their progressive rate against their regressive one?].
```

Three separate things happen here and each is deliberate. **(a)** The vocabulary claim keeps its full
force and gains the only support the ledger actually carries, in their own citation form. **(b)** The
`doi.org/10.1609/aies.v8i1.36598` link goes — not because it is wrong, which I cannot certify, but
because it is absent from the ledger and links and identifiers are not in their register anywhere.
**(c)** The `[?]` is not guessed. It is replaced by the demand it stands in for, and the asymmetry
clause it sits inside is left standing verbatim, per the hard rule.

EVIDENCE:
  - `CITATIONS_post1_verified.md` :: "**progressive / regressive sycophancy** = this post's listen/fold
    (SycEval 2502.08177, Fanous, 2025)" :: the whole of what the ledger licenses for SycEval — a term
    mapping, and the direction of the mapping (progressive = listen, regressive = fold) matches their
    live sentence.
  - `CITATIONS_post1_verified.md` header :: "Drafting agents may cite ONLY from this ledger; anything
    absent here is unverified and must be bracketed as such" :: the rule that forces the bracket rather
    than a filled `[?]`.
  - `HOLES_post1_v2.md` §4, identifiers table :: "`https://doi.org/10.1609/aies.v8i1.36598` (SycEval) —
    **Not in CIT.**" :: the identifier's status.
  - `STYLECARD_researcher.md` §A9 :: their own written form for this exact cite —
    `Respectively progressive and regressive sycophancy (SycEval; Fanous et al. 2025).` (CIRCUIT L26) —
    plus "no arXiv IDs, no links, no footnotes" and "**They strip arXiv IDs out of machine-supplied
    text and replace them with a bracketed question.**"
  - `REVIEW_post1_patches.md` → "Reversal of an earlier register call" :: parenthetical author-year is
    in register and must not be converted to inline form.

CRITERIA:
  F — the term mapping is the ledger's own line; nothing else is asserted.
  M — L20 is the only SycEval placement in either document apart from notes L226, which already reads
    `(Fanous et al. 2025)`.
  P — the bracket is 17 words and carries the one question that closes the marker.
  1P — n/a, this is a citation hole; the claim it supports is the fold/listen naming, which the figure
    at L12 exhibits.
  R — author-year parenthetical, no link, no ID, lowercase in-flow bracket, sentence readable across it.
  C — SycEval verified for the vocabulary and for nothing else; their asymmetry sentence left standing.
  S — L20 only; the five other markdown links in the document (L22, L24) are all ledger-verified and
    are not touched.

RESIDUAL: the exact question the researcher has to answer to close the `[?]`, in the order it has to be
answered.
  (1) Which reading was the `[?]` — a missing quote for the asymmetry, a missing hedge word, or which
  model class SycEval actually tested? Only they know, and the three readings need three different
  fills.
  (2) On the first reading: does Fanous et al. report progressive sycophancy at a *higher rate* than
  regressive, as a single number, and in which table? Their headline split is progressive versus
  regressive by prompt condition; nothing in the ledger says the two rates were compared in the
  direction this sentence needs. Whoever reads the paper must add the sentence and its number to
  `CITATIONS_post1_verified.md` before the bracket comes out.
  (3) `-chat models` is also unverified as a scope — SycEval tests deployed chat assistants, so the
  wording is probably right, but it is not in the ledger and it should be checked in the same pass.
  Until (2) lands, the clause is the researcher's own inference from a paper's vocabulary, and the
  bracket is what says so.

---

### §3.4 — intro L24, `Chat training deletes the grey band.`

ANCHOR (verbatim, mid-paragraph; the sentence before it is the flip-rate sentence patched in §3.5):

```
Chat training deletes the grey band. That sits awkwardly against the Gemma 2 report’s claim
```

FILL:

```
The grey band is a -base column - the released -chat models do not have one.
```

EVIDENCE:
  - vault notes L129, their own standing instruction :: "Keep this descriptive: released base vs
    released -it, format co-varies with model, **no causal 'tuning forces' claim — that was the error
    the last review caught.**" :: the rule this sentence broke, written by them, about this exact
    material.
  - vault notes L33 :: "DeepMind has not released staged checkpoints for Gemma 2 so we can't compare
    the effects of SFT vs RLHF on our target behaviour, so here I compare as -base vs. -chat" :: there
    is no measurement between the two endpoints, so nothing licenses `deletes`.
  - `GROUNDING_notes_numbers.md` :: `L207` :: base withheld 51 / 38 / 32 of 82 against -it 0 / 0 / 1 ::
    the difference between the released pair, which is all the sentence now claims.
  - `GROUNDING_notes_numbers.md` :: `L168, stale` :: 0 silences in all 164 items at 2b-it, 9b-it and
    27b-it :: the -it half holds at the reply as well as the final answer, so `do not have one` is not
    slot-specific.

CRITERIA:
  F — both halves of the contrast are counted in GROUNDING; neither number is printed, because the
  sankey draws them.
  M — L16's fill says the elicitation does not empty the band for -base; this says the -chat column
  never had one. Different cells, no overlap.
  P — 15 words for a 6-word sentence, and the four extra words are the whole correction (`released`,
  `models`).
  1P — `faithful_elicit` and `counter_gen` labels on the six ext2 summaries.
  R — descriptive, spaced hyphen, no coined term, no hype; the snap-sentence rhythm of the original is
  kept.
  C — no citation touched; the paragraph's causal statements now belong entirely to SYCON, Gupta and
  the Gemma report, each already attributed in their own clause.
  S — L24, this sentence only.

RESIDUAL: no bracket is attached, because the descriptive sentence needs no caveat — but the intro now
nowhere states that there are no staged checkpoints, and that disclaimer currently lives only in the
notes (L33 and L129). If a reader is expected to know why the post never says tuning caused anything,
it has to be said once in the short post too. Their call, and MECE row d says keep one instance, so
adding it here means cutting it from notes L33.

---

### §3.5 — intro L24, the flip-rate sentence

ANCHOR (verbatim, curly quotes):

```
A flip-rate eval that treats “I don’t know” as robustness will score -base as steadier than -chat.
```

FILL:

```
A spoken-answer readout that treats “I don’t know” as robustness will score -base as steadier than -chat. Read the same pressure off a two-option margin, as De Marez et al. do, and it runs the other way round - their tuned checkpoints flip less than their base ones on 17 of their 23 matched pairs.
```

The correction is the one NOVELTY specifies and nothing more: the metric keeps its name, the readout
gains one. Their sentence is true of the channel this post reads and false of the channel the nearest
published neighbour reads, so the second sentence is not padding — without it the first reads as a
restatement of a section heading that already exists, and a reviewer opens with the counterexample.

EVIDENCE:
  - `NOVELTY_boundary_post1.md` → "Two contradictions the post must handle", #1 :: "In C's flip-rate
    channel base is scored *less* robust than IT: 'a drop from **23.3% to 16.3% flip rate on identical
    items**, strictly lower at every pair-averaged margin quartile' (§3.3), and 'In 17 of 23 Base-IT
    pairs, IT is more robust' (§3.2). So 'flip rate flatters base models' is **specific to this post's
    spoken-answer flip rate**. **The fix is to name the readout, not the metric.**"
  - `NOVELTY_boundary_post1.md` §C :: their instrument is "two-option MCQ letter completion with a
    trailing `Answer: (`; `S_c = log P(a) − log P(b)`" and "Their flip is a THRESHOLD ON THAT SAME
    LOG-PROB, not a separate spoken channel" :: `two-option margin` is their design described, not a
    paraphrase, and it hooks onto the margin vocabulary L22 already introduced.
  - `NOVELTY_boundary_post1.md` VERDICT (v) :: "SCOOPED as a slogan (De Marez §3.3 is literally titled
    'Base scaling is hidden by flip rate') — must be re-scoped to the two-readout version **and
    cited**" :: why the cite is inside the fill rather than in a bracket.
  - `EXHIBITS_post1_grounded.md` §D :: 9b-base elicited fold W\* 3/82 against 9b-it W\* 55/82 ::
    on the spoken readout base is dramatically the steadier at 9b.
  - `GROUNDING_notes_numbers.md` :: `L302` and `L186` :: -it takes the pushed wrong answer at
    0.829 / 0.671 / 0.671 at 2b/9b/27b at the elicited slot, while base folds on 16/31, 3/44, 11/50 of
    what it commits to :: base scores steadier on the spoken channel at every scale, so the first
    sentence survives the rescope at full strength.

CRITERIA:
  F — 17 of 23 is a NOVELTY-verified quote from De Marez §3.2; the percentages are left out because
  the direction is the load-bearing part.
  M — De Marez is introduced at L22 and used here; per MECE a3 the introduction stays the only one.
  P — one word changed in their sentence, one sentence added, and the added sentence is the only thing
  standing between this claim and the published counterexample.
  1P — the base-side claim bottoms out in `faithful_elicit` on the ext2 summaries; the De Marez side is
  external and is cited, not asserted.
  R — narrative author-year, no ID, no link, spaced hyphen, no bullet, curly quotes preserved
  byte-exact.
  C — De Marez verified in CITATIONS and filed MISATTRIBUTED only for the "56 checkpoints" clause in
  the notes, which this fill does not use.
  S — L24, this sentence only.

RESIDUAL: the reconciliation NOVELTY asks for in its contradiction #2 is still unwritten anywhere — De
Marez competence-filter to high-margin items while this post manufactures near-ties, which is why their
IT looks robust and ours folds. One sentence, and it belongs in the notes rather than in the short post.

---

### §3.6 — intro L26, the closing claim

ANCHOR (verbatim, the two closing sentences of the post's last paragraph):

```
**Chat training does not appear to install a dedicated truth circuit.** It makes Gemma 2 less "willing" to say it does not know, and more to revise.
```

JUDGMENT: **it cannot stand as written, on three independent grounds, and it can stand in weakened
form.** (1) Its evidence is not merely unpublished, it is unowned — the head-sharing result the
sentence interprets sits inside `[relegated]` in the notes, carries their own `[seems to still
exist?]`, and its mechanism has two candidate names and no chosen one (`["salience copy" or "attention
copy"]`). Nothing in the two grounding files exhibits it. (2) It is a claim about what training
installs, made from two released endpoints with nothing measured between them — the same defect as
§3.4, and the one their own L129 note records the last review catching. (3) `a dedicated truth circuit`
coins a named Thing, and they coin nothing except after the operation that defines it; the bold is
their fourth register break in one sentence. Weakened, what survives is a statement about what the
results do not show, which is still a strong close and is the honest one.

FILL:

```
Nothing here says chat training installed a component specific to the correct answer. [the shared-heads half of this is still relegated in the notes - what is the run that backs it?]
```

The subject stays recoverable: their next sentence opens `It makes Gemma 2 less "willing"…`, and
`chat training` is still the nearest noun phrase, so the paragraph reads across the patch unchanged.
Any alternative fill that drops `chat training` from the sentence orphans that `It`.

EVIDENCE:
  - `HOLES_post1_v2.md` §3 row 6 :: "the mechanistic arc it rests on (`at -base, fold and listen share
    the same most influential attention heads, whilst at -chat, this mechanism is distributed`) has
    **no exhibit in EXHIBITS** and the notes' own version of it (L193–196, L268, L273) sits entirely
    inside `[relegated]` brackets."
  - `GROUNDING_notes_numbers.md` :: `L194 — the mask result` :: the only committed mechanistic number
    in either document is 67/74 naming an answer under an attention mask, "**the n=74 mechanism family,
    not the ext2 82**" — and it is about *whether* the model answers, not about which heads fold and
    listen share.
  - vault notes L273 :: `["salience copy" or "attention copy"]` and `[seems to still exist?]` :: the
    mechanism is unnamed and its persistence is an open question in their own hand.
  - vault notes L129 :: "no causal 'tuning forces' claim — that was the error the last review caught."
  - `STYLECARD_researcher.md` §B9 :: "No bold-for-emphasis inside sentences… Never mid-argument.
    v6/v7 bold a phrase per paragraph" — the rejected register, which this sentence had drifted into.
  - `STYLECARD_researcher.md` §B1 :: "They name exactly two things, both from an operation… Nothing
    else is capitalised into a Thing" :: `a dedicated truth circuit` is a third.
  - `GROUNDING_notes_numbers.md` :: `L196` :: -it restates the pushed entity on 50 of 82 when the push
    is wrong against 67 of 82 when it is right (52 / 67 current) :: revision is not selective for
    truth, which is the behavioural content the weakened sentence interprets — and which their own
    preceding sentence already states, so the fill does not restate it.

CRITERIA:
  F — every element is either negative (no artifact exists, and the bracket says so) or already carried
  by the sentence before it.
  M — the preceding sentence carries "revises freely in both directions, more so toward truth"; the
  fill interprets it and does not re-read it.
  P — the fill is shorter than what it replaces.
  1P — the one positive number in reach (67/74) is named in EVIDENCE and deliberately not used in
  prose, because it is a different item family.
  R — no bold, no coined term, `I`-free because the subject is the evidence rather than the finding,
  bracket 18 words and readable across.
  C — no citation touched.
  S — L26, one sentence; their following sentence is left standing untouched.

RESIDUAL: two things, both theirs.
  (1) Their following sentence, `It makes Gemma 2 less "willing" to say it does not know, and more to
  revise.`, is the same causal form as §3.4 and was not in my six. It is the last sentence of the post.
  On the released pair it should read as a difference, not as an effect.
  (2) The sentence two before it — `at -base, fold and listen share the same most influential attention
  heads, whilst at -chat, this mechanism is distributed` — is the actual load-bearing claim and has no
  artifact in any of the four repo files. Until the run is named, the closing paragraph's mechanistic
  half rests on it, and my bracket is the only thing saying so. If it turns out there is no run, the
  honest version of this paragraph is behavioural only and loses its last three sentences.

---

### FLAG (no fill) — intro L22, `wasd`

ANCHOR (verbatim; the character before `[De Marez` at line start is nothing, the U+00A0 follows the link):

```
This is not shown in the sankey, and adding another one to this page wasd vetoed by Fable, so its going in the lab notes.
```

FLAG, do not fix: `wasd` is their typo for `was`, and it is a logged one — `STYLECARD_researcher.md`
§A12 records it verbatim from V3b L10 (`The original intention for this project wasd designing an
attribution-graph "verifier"`), which means it is a habit and not a slip. `its going` on the same line
is the second, and `all of the others ones` at L24 the third.
`HOLES_post1_v2.md` §1 lists all three under "Typos left alone deliberately".

EVIDENCE:
  - `STYLECARD_researcher.md` §A12, table row `wasd` :: V3b L10 :: the same typo in the same author's
    prose, two documents apart.
  - `HOLES_post1_v2.md` §1, closing paragraph :: "Typos left alone deliberately… intro L22 `wasd` /
    `its going`, L24 `all of the others ones`".

CRITERIA: F — the flag cites the register file, not an artifact. M / P / 1P — n/a, nothing written.
R — flagging rather than fixing is the documented handling. C — none. S — L22 only, and only the typo;
the sentence's unmet promise to the lab notes is a separate hole and is not mine.

RESIDUAL: the same line's `*_usually*` renders literally and is a formatting hole, not a typo — it is
listed separately in HOLES §1 and belongs to whoever owns L22. Not touched here.

---

### Marker — intro L28, `[Full lab notes pending write-up]`

ANCHOR (verbatim, the whole line, last line of the file):

```
[Full lab notes pending write-up]
```

ANSWER: **yes to the name, no to the link, and `pending` is still the true word.** The notes document
exists and is live, so the placeholder can stop being anonymous — but it is not written up. It carries
101 open markers by HOLES' count, and the specific thing this intro promises it at L22, the probability
result, is not in it: notes L285's Figure 3b is still a bracketed plot request and the claim at L289
that would cite it is itself bracketed. Naming the document and deleting `pending` would convert an
honest placeholder into a broken promise.

FILL:

```
[Full lab notes pending write-up - Characterizing base vs chat behaviours under pushback in Gemma 2]
```

EVIDENCE:
  - vault notes L5 :: `# [Lab Notes] Characterizing base vs chat behaviours under pushback in Gemma 2'`
    :: the title, minus the `[Lab Notes]` tag and the trailing straight apostrophe, which is a logged
    typo of theirs and should not be carried into the intro.
  - vault notes L2–L3 front matter :: `share_link: https://share.note.sx/t9ov9hm5#KMxruNjkAKVL2VL+4jLCoQ`,
    `share_updated: 2026-07-26T20:02:09+01:00` :: a live published copy exists, so a link is available
    if they want one.
  - `HOLES_post1_v2.md` §1, intro L22 row :: "Promises the probability result to the lab notes. The
    notes do **not** carry it: Figure 3b (notes L285) is still a bracketed plot request and the claim
    at notes L289 is itself bracketed." :: why `pending` stays.
  - `HOLES_post1_v2.md` §1 :: "**Counts: intro 7, notes 101, total 108.**"
  - `STYLECARD_researcher.md` §A9 :: no links in their prose :: why the share URL is reported here and
    not written into the draft.

CRITERIA:
  F — the title string is read from the live notes; the share link's existence and date from its front
  matter.
  M — the intro points at the notes in exactly one place, and this is it.
  P — four words added, all of them the document's name.
  1P — n/a, a pointer.
  R — their own bracket kept verbatim and extended, lowercase, in-flow, no link, no footnote.
  C — none.
  S — L28 only.

RESIDUAL: whether to publish `share.note.sx/t9ov9hm5#…` is theirs alone — the fragment after the `#` is
the decryption key, so pasting it publishes the notes to anyone with the URL, and the notes currently
contain 101 open markers including several self-criticisms. My recommendation, which is not a fill: hold
the link until the probability result L22 promises is actually in the notes, then replace this whole
bracket with the link and drop `pending`.

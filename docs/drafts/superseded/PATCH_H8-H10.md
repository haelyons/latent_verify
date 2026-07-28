# PATCH H8-H10 - one paragraph in "# Chat models always answer"

Live draft read 2026-07-24, `md5 fbebdc3b08aa13adcc3e1345afe3b5af`
(`/home/hal/Documents/Remote/interp/DARWIN.md_post1_user.md`, 5380 bytes, mtime 16:31). Anchors copied
byte-for-byte from that read (L110 and L114); nothing was written to the vault. Anchors had NOT moved.

Guillemets in every anchor and fill are `\u00ab\u00a0text\u00a0\u00bb` - U+00AB, NBSP, text, NBSP, U+00BB -
verified against `xxd` of L114. Their `model's` as a plural and their straight quotes around
`"pleasing the user"` are preserved untouched (STYLECARD A12).

One paragraph, three anchored patches, applied in document order.

---

### H8
ANCHOR (verbatim from live draft, L110, first sentence):
```
The model flipping its answer has been described in sycophancy literature [what literature? Rismky/Panickserry? others?] by model's representing and attending to "pleasing the user".
```
The patch replaces ONLY the bracket `[what literature? Rismky/Panickserry? others?]`. Their clause
`by model's representing and attending to "pleasing the user".` stays standing, unedited.

FILL:
(Sharma et al. 2023; Perez et al. 2022) [both are behavioural - preference-data analysis and model-written evals - and neither makes a claim at the representation or attention level; « pleasing the user » is not a phrase from either paper, Sharma's own wording is "match user beliefs over truthful ones". The representational claim belongs to the steering work instead: Panickssery et al. 2024, printed Rimsky on the v1 preprint and in the ACL proceedings and Panickssery in the current arXiv metadata, the same person and one paper. Either hang the representing and attending off that or drop it.]

EVIDENCE:
- `docs/drafts/CITATIONS_post1_verified.md` :: VERIFIED, 2310.13548 (Sharma, 20 Oct 2023, ICLR 2024) :: the behavioural/preference-model account, and the ledger's explicit finding that Sharma's own wording is "match user beliefs over truthful ones" and that `pleasing the user` is a phrase in neither Sharma nor Perez.
- `docs/drafts/CITATIONS_post1_verified.md` :: VERIFIED, 2212.09251 (Perez, 2022, Findings of ACL 2023) :: "Larger LMs repeat back a dialog user's preferred answer ("sycophancy")" - model-written evals, dataset generation, no representational or attention claim.
- `docs/drafts/CITATIONS_post1_verified.md` :: MISATTRIBUTED, entry `"representing and attending to 'pleasing the user' [Sharma; Perez]"` :: "neither paper makes a representational or attention-level claim ... Change the verb or add a mechanistic cite (2312.06681 is the steering-vector one)" - this is the whole content of the bracket.
- `docs/drafts/CITATIONS_post1_verified.md` :: VERIFIED, 2312.06681 (ACL 2024, 2024.acl-long.828) :: "Author-name question settled: Rimsky and Panickssery are the same person - v1 PDF + ACL Anthology print Nina Rimsky, current arXiv metadata prints Nina Panickssery"; and sycophancy as a target behaviour, which is why the representational claim can hang there.

CRITERIA:
- F - every element traces to a named ledger entry; the only quoted string, "match user beliefs over truthful ones", is the ledger's verbatim line for Sharma's own wording.
- M - answers a question nothing else in the post answers (which literature, and whose name); no other line in the draft names a paper for the flip.
- P - two clauses of correction plus one clause of who-is-who; no mechanism posited, no adjective that is not doing work.
- 1P - the claim is about what two papers do and do not say, checked against the papers in the ledger, not against any repo draft.
- R - inline lowercase bracket, spaced hyphens, guillemets with NBSP for the phrase-as-object, British `behavioural`, no bullets, no em-dash, no arXiv ID, no link, no block quote; their sentence and typo left standing.
- C - only ledger papers cited, author-year inline; their wrong claim is NOT repaired in place, it is flagged with the two options they can take (`hang ... off that or drop it`), matching their own `Either soften that sentence ... or cut "copy"` register.
- S - replaces the bracket and nothing else.

RESIDUAL: `arXiv metadata` is the honest locus for the name change but the word `arXiv` appears nowhere else in their prose - swap to `the current preprint metadata` if they want it gone. Sharma is now cited in two places (L59 `Sharma et al [what year?]` is H3's hole, not touched here) - keep the year consistent at 2023.

---

### H9
ANCHOR (verbatim from live draft, L110, second sentence):
```
Some mechanistic accounts driven by representation engineering methods [super vague sentence, what methods? instead of stating these high level concepts can we just describe high level what was done? "using counterexamples to isolate types of sycophancy and refusal in model activations"?].
```
The patch replaces that whole sentence plus its bracket - it is the one place they instruct the rewrite
themselves ("instead of stating these high level concepts can we just describe high level what was
done?"), and their candidate wording is requoted inside the new bracket so nothing of theirs is deleted.

FILL:
Panickssery et al. (2024) take the difference of the model's internal activations at the position of the answer letter between matched positive and negative prompts - Mean Difference, in their terms - and add that vector back at inference, which reduces sycophancy further than a few-shot prompt written to suppress it. [my "using counterexamples to isolate types of sycophancy and refusal in model activations" is close, but sycophancy is one target behaviour there rather than a family of types, and isolating a direction is only half of it - adding it back at inference is the causal half. representation engineering is a different paper (Zou et al. 2023), so cite that separately or drop the phrase. does their behaviour set cover refusal? not checked]

EVIDENCE:
- `docs/drafts/CITATIONS_post1_verified.md` :: VERIFIED, 2312.06681 :: what was actually done, verbatim - "we compute the difference in the language model's internal activations at the position of the answer letter between all the positive and negative prompts. This approach of extracting the difference vector is called Mean Difference (MD)".
- `docs/drafts/CITATIONS_post1_verified.md` :: VERIFIED, 2312.06681 :: the result clause - "CAA further reduces sycophancy on top of a few-shot prompt designed to limit this behavior"; "few-shot prompting alone is unable to reduce sycophancy to the same extent as CAA".
- `docs/drafts/CITATIONS_post1_verified.md` :: VERIFIED, 2312.06681 :: "Sycophancy is a target behaviour" - grounds `one target behaviour there rather than a family of types` and refuses their `types of sycophancy`.
- `docs/drafts/CITATIONS_post1_verified.md` :: VERIFIED, 2312.06681, last bullet :: ""representation engineering" is a DIFFERENT paper: Zou et al. 2310.01405. Cite both or drop the phrase; do not slash them together."
- `docs/drafts/CITATIONS_post1_verified.md` :: ADD, 2310.01405 :: Zou et al., representation engineering, verified and citable if they keep the phrase.

Verdict on their candidate wording, since H9 asks for one: `counterexamples` is right in substance (matched
positive/negative prompts) but the paper's own frame is a contrast pair, `types of sycophancy` is wrong
(sycophancy is one behaviour in their set, not a family of types), `isolate ... in model activations` is
half the method - the difference vector is only a direction until it is added back at inference, which is
the step that makes it causal. So: adopted in substance, sharpened on all three points, and the label
`representation engineering` dropped from the prose with the reason in the bracket rather than silently.

CRITERIA:
- F - the method sentence is the ledger's verbatim method quote in their syntax; `refusal` is not in the ledger, so it is asked as a bracketed question rather than asserted.
- M - the paragraph carries the mechanistic leg once; the sentence says what was done and adds nothing the H8 bracket already said.
- P - one prose sentence for the method, no second sentence restating it, no label where a description works.
- 1P - grounded in what the paper computes (activation difference at a token position, added at inference), not in a summary of it.
- R - subject-position author-year exactly as `Wang et al. (2022) call such a head ...` (STYLECARD A9), one spaced-hyphen parenthetical, present tense, lowercase bracket carrying the leftover doubt, British `behaviour`.
- C - two ledger papers, author-year only, no IDs; their phrase `representation engineering` is not silently deleted, its owner is named and the decision is handed back.
- S - replaces their vague sentence and its bracket; the first sentence of the paragraph and the orphan clause are handled in H8 and H10.

RESIDUAL: the contrastive pairs come from Anthropic's model-written-evals A/B datasets, which is Perez 2022 - available in the ledger and would tie the two sentences together, left out under P. Whether the CAA behaviour set includes refusal is unverified and stands as a bracketed question in their register.

---

### H10
ANCHOR (verbatim from live draft, L114, the whole orphan line including its trailing space):
```
as driven by this idea of « pleasing the user » or maximizing agreement, this could indicate that a major sycophantic driver is just the bias toward answering at all, versus expressing uncertainty. 
```
Note on whitespace: in the live file L111-L113 are three blank lines between the H8/H9 sentence and this
orphan. Making the three fills one paragraph means those blank lines close up. That is the only change
outside the three anchors and it is theirs to accept.

FILL:
Where the flip is read as driven by this idea of « pleasing the user » or maximizing agreement, this could indicate that a major sycophantic driver is just the bias toward answering at all, versus expressing uncertainty. 

EVIDENCE:
- `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user.md` :: L114, bytes `c2ab c2a0 ... c2a0 c2bb` :: every word from `as driven by` to `expressing uncertainty.` is theirs, carried through unaltered including `maximizing` in US spelling and the trailing space; the inserted subject is `Where the flip is read`, five words.
- `docs/drafts/CITATIONS_post1_verified.md` :: MISATTRIBUTED, `pleasing the user` entry :: the reason the inserted subject is passive and agentless - naming a paper as the source of that idea here would re-assert exactly what the H8 bracket corrects.

CRITERIA:
- F - no number, string or model output is introduced, so there is nothing to trace beyond their own line.
- M - the observation (base hedges, -chat always answers) is already in the heading and in the previous paragraph, so the subject points back with their own `this` rather than restating it.
- P - five words added, nothing else; deleting any of them leaves `as driven by` dangling again.
- 1P - the claim is theirs and is left at the level they set it; no support is invented for it here (see RESIDUAL for what the artifacts do and do not support).
- R - guillemets with NBSP preserved byte-exact, `maximizing` left as they spelled it, trailing space kept, no em-dash, no wrap-up sentence after it.
- C - no citation added; the framing is left unattributed on purpose.
- S - one line replaced, one clause inserted.

RESIDUAL: see the two reported checks below.

---

## The three fills as one paragraph

(Sharma et al. 2023; Perez et al. 2022) [both are behavioural - preference-data analysis and model-written evals - and neither makes a claim at the representation or attention level; « pleasing the user » is not a phrase from either paper, Sharma's own wording is "match user beliefs over truthful ones". The representational claim belongs to the steering work instead: Panickssery et al. 2024, printed Rimsky on the v1 preprint and in the ACL proceedings and Panickssery in the current arXiv metadata, the same person and one paper. Either hang the representing and attending off that or drop it.] by model's representing and attending to "pleasing the user". Panickssery et al. (2024) take the difference of the model's internal activations at the position of the answer letter between matched positive and negative prompts - Mean Difference, in their terms - and add that vector back at inference, which reduces sycophancy further than a few-shot prompt written to suppress it. [my "using counterexamples to isolate types of sycophancy and refusal in model activations" is close, but sycophancy is one target behaviour there rather than a family of types, and isolating a direction is only half of it - adding it back at inference is the causal half. representation engineering is a different paper (Zou et al. 2023), so cite that separately or drop the phrase. does their behaviour set cover refusal? not checked] Where the flip is read as driven by this idea of « pleasing the user » or maximizing agreement, this could indicate that a major sycophantic driver is just the bias toward answering at all, versus expressing uncertainty. 

Reading, in order: the citation slot answers `[what literature?]` and immediately says what those papers
do not support, their claim standing; the method sentence answers `[what methods?]` with what was done;
the joined clause turns the whole thing on its head, which is theirs. No wrap-up sentence follows, and
`![[IMG_3868.png]]` sits directly under it as before.

---

## RESIDUAL (reported, not fixed)

1. Duplication (M): the section heading `# Chat models always answer` and the preceding paragraph
   (`the -chat models which always provide an answer`) already carry the observation, so the joined
   sentence must not restate it - it does not, it points back with their `this`. The cost is that `this`
   now reaches back across a paragraph break. If they want it explicit the swap is `the -base hedging
   could indicate`, which drops their `this` - their call, not mine.
2. Duplication, second check: `![[IMG_3868.png]]` (L116) sits directly under this paragraph with no
   caption. If that figure carries the withhold column, then no count belongs in this prose either
   (STYLECARD B8), which is a second reason the fill stays numberless.
3. Support level: as joined, the claim is stated for sycophancy generally - `a major sycophantic
   driver`. What is measured is one prompt family, 9b, 82 items: 9b-it elicited fold is W* 55 / C 27 /
   withheld 0, 9b-base is C 41 / W* 3 / withheld 38 (`out/faithful_rescore_fl_9bit_ext2.json` and
   `results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json`, EXHIBITS section D).
   `0 versus 38 of 82 at 9b` is the number that would qualify it if they want the claim pinned.
4. Support level, the scale caveat that actually bites: the base-side hedging is a 9b fact, not a base
   fact. `No, I'm not sure. I'm just guessing.` is 37/82 fold at 9b-base, 3/82 at 27b-base and 0 at
   2b-base, while 2b-base's home string is `Yes, I'm sure.` at 38/82 fold - an answer-shaped non-answer
   (EXHIBITS section B). So `expressing uncertainty` as the base alternative holds at 9b; at 2b the base
   model answers too, it just answers contentlessly.
5. Confound overlap with H6: part of `withheld 0` at -it is attributable to the forced elicitation slot
   rather than to tuning (ledger H6, 2509.04664 - "guessing when uncertain improves test performance"),
   and the defence is within-format, base withholding 38/82 under the identical slot. That belongs in
   H6's confound sentence; H10's claim leans on it, so the two fills should not contradict each other.
6. Unverified and bracketed rather than asserted: whether the steering paper's behaviour set includes
   refusal.
7. Not touched: L59 `Sharma et al [what year?]` (H3), and the `arXiv metadata` wording flagged under H8.
8. Cross-patch: `patches/PATCH_H7.md` (written this session) puts the neutral-arm counts `1 of 82` /
   `0 of 82` into L106, two paragraphs above. So if a count is ever wanted in this paragraph it is the
   withhold column (item 3), never the neutral counts, or the two paragraphs collide.

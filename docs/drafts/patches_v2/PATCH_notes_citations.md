# PATCH — notes, citations + the L314/L316 duplication

Target: `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_notes.md`
Live state at write time: md5 `88a7a5e56aaa69d194f05460ddbb9504`, 32560 B, 333 lines — identical to the
state `HOLES_post1_v2.md` was built against, so every line number below is current. All anchors below
were re-read from the vault by line number, not from a snapshot.

---

### C1 — notes L314 + L316 (the duplication)

ANCHOR (verbatim, L314, blank L315, L316):

```
The sycophancy literature describes answer-flipping as the model representing and attending to "pleasing the user" [Sharma et al. 2310.13548 for the preference-model account; Perez et al. 2212.09251 for the model-written-evaluation scaling result — confirm these are the two I mean]. There is a line of work that isolates a sycophancy _direction_ from contrastive examples and steers along it [representation-engineering / contrastive activation addition — Rimsky/Panickssery et al. 2312.06681; confirm this is the "counterexamples to isolate types of sycophancy and refusal in activations" method I had in mind — say what was done, not the label].

The model flipping its answer has been described in sycophancy literature [what literature? Rismky/Panickserry? others?] by model's representing and attending to "pleasing the user". Some mechanistic accounts driven by representation engineering methods [super vague sentence, what methods? instead of stating these high level concepts can we just describe high level what was done? "using counterexamples to isolate types of sycophancy and refusal in model activations"?].
```

MECE — which is theirs, and which is the import:

**L316 is the researcher's own. L314 is imported machine prose pasted in above it.** Five independent
tells, four of them from the register authority:

1. `STYLECARD_researcher.md` §A8 variant 5 quotes L316's bracket **verbatim, in full**, as `(POST1
   L110)` — i.e. it is in the 893-word corpus that defines the voice. Nothing from L314 appears
   anywhere in the style card.
2. §A6 quotes L318 (`as driven by this idea of « pleasing the user » or maximizing agreement`) as
   `(POST1 L114)`. So L316 → L318 is a contiguous stretch of their own POST1, four lines apart in the
   original. L314 has no POST1 line number because it was never in POST1.
3. L316 carries two of their documented signatures: `model's` as a plural (§A12, `POST1 L119`) and
   both names misspelled from memory, `Rismky/Panickserry`. L314 spells `Rimsky/Panickssery`
   correctly and has no typo anywhere.
4. L314 carries three em-dashes and three bare arXiv IDs. §A9: "**They strip arXiv IDs out of
   machine-supplied text and replace them with a bracketed question**" — the documented instance is
   POST1 L59, now notes L68, where `(arXiv:2310.02174)` / `(arXiv:2310.13548)` were deleted and
   `[what year?]` substituted. Genuine em-dashes in their own prose: "effectively zero" (§A6).
5. L314's brackets are downstream restatements of L316's: `confirm these are the two I mean` answers
   `[what literature? Rismky/Panickserry? others?]`, and `say what was done, not the label` answers
   `[…can we just describe high level what was done?]`. A rewrite absorbs the note it was written
   against; an original does not anticipate its own gloss.

`HOLES_post1_v2.md` §2.3 row (b) recommends keeping L314 "because it names the arXiv IDs and asks the
sharper question". Both reasons dissolve under the BRIEF: the IDs must come out of the prose, and the
sharper question is L316's own, re-asked. So the survivor is L316, with L314's second half folded in
as the content L316's sentence fragment was gesturing at.

FILL (replaces L314, L315 and L316 — three paragraphs, prose only, brackets inline):

```
The model flipping its answer has been described in sycophancy literature by model's representing and attending to "pleasing the user".

Sharma et al. (2023) is the preference-model account: both humans and preference models "prefer convincingly-written sycophantic responses over correct ones a non-negligible fraction of the time". Perez et al. (2022) is the training-side one, and the direction is inverse - sycophancy is among their "first examples of inverse scaling in RL from Human Feedback (RLHF), where more RLHF makes LMs worse" [those are the two, and neither makes an activation or attention claim - "pleasing the user" is in neither, Sharma's own wording is "match user beliefs over truthful ones"].

The mechanistic line is a separate one, and it states better as what was done than as a label: Panickssery et al. (2024) take the difference in a model's internal activations at the answer position between all the positive and negative prompts, and add that difference back at inference, sycophancy being one of the behaviours they steer [Rimsky and Panickssery are the same person, one paper] [representation engineering is Zou et al., a different paper - what year?].
```

EVIDENCE:
  - `CITATIONS_post1_verified.md` VERIFIED :: **2310.13548** (Sharma, 20 Oct 2023, ICLR 2024) ::
    quote "both humans and preference models (PMs) prefer convincingly-written sycophantic responses
    over correct ones a non-negligible fraction of the time"; and "Sharma's own wording for the
    behaviour is 'match user beliefs over truthful ones' — **not** 'pleasing the user'".
  - `CITATIONS_post1_verified.md` VERIFIED :: **2212.09251** (Perez, 2022, Findings of ACL 2023) ::
    quote "We also find some of the first examples of inverse scaling in RL from Human Feedback
    (RLHF), where more RLHF makes LMs worse"; ledger instruction "Say **inverse-scaling** … not
    'scaling'". This retires the import's `model-written-evaluation scaling result`, which reversed it.
  - `CITATIONS_post1_verified.md` MISATTRIBUTED :: "representing and attending to 'pleasing the user'
    [Sharma; Perez]" :: "neither paper makes a representational or attention-level claim. Sharma is
    behavioural + preference-data analysis; Perez is dataset generation."
  - `CITATIONS_post1_verified.md` VERIFIED :: **2312.06681** (CAA, ACL 2024) :: method quote "we
    compute the difference in the language model's internal activations at the position of the answer
    letter between all the positive and negative prompts"; target-behaviour quote "CAA further reduces
    sycophancy on top of a few-shot prompt designed to limit this behavior"; name ruling "Rimsky and
    Panickssery are the same person… Cite as Panickssery (formerly Rimsky) et al., ACL 2024".
  - `CITATIONS_post1_verified.md` :: **2310.01405** (Zou) :: "'representation engineering' is a
    DIFFERENT paper… Cite both or drop the phrase; do not slash them together." Ledger prints no date.
  - `REVIEW_post1_patches.md` SHOULD FIX :: "the Rimsky/Panickssery venue detail is surplus. Delete
    everything before `the same person`" — hence no `ACL 2024` in the bracket.
  - `REVIEW_post1_patches.md` SHOULD FIX :: "`Zou et al. 2023` is an inferred year… Either verify the
    year or bracket it" — hence `what year?` rather than a printed 2023.
  - `REVIEW_post1_patches.md` Reversal :: "`Panickssery et al. (2024)` … [has] direct precedent and
    must NOT be 'corrected' to inline form."
  - `STYLECARD_researcher.md` §A8.5 `(POST1 L110)`, §A6 `(POST1 L114)`, §A12 `(POST1 L119)`, §A9
    `(POST1 L59)` :: the provenance argument above.

CRITERIA:
  F — every quote is the ledger's verbatim string; no number is asserted.
  M — the three paragraphs replace two near-identical ones; the surviving claim is stated once.
  P — the import's `There is a line of work that…` frame is gone; the sentence now names the operation.
  1P — no model I/O is claimed here; this block is citation-only, which is its whole job.
  R — no em-dash, no arXiv ID, no `my`, no bullets, British `behaviours`, their `model's` plural kept,
      brackets 29 / 9 / 12 words against a median of 5 and a ceiling of ~40, paragraphs 20 / 62 / 58
      words against POST1's 65-word longest, every bracket readable-across.
  C — their claim (`representing and attending to "pleasing the user"`) stands untouched as its own
      paragraph; the correction is bracketed, not applied.
  S — L314–L316 only; L318 untouched.

RESIDUAL:
  - **L318 is theirs, not machine text, and `HOLES` §2.3 (b) is wrong to bundle it with L314 for
    deletion.** §A6 of the style card quotes it as `POST1 L114`. It is a genuine orphaned tail: it
    opens lowercase (`as driven by this idea of…`) and its lead-in was lost, not pasted over. It needs
    a restored stem, which is theirs to write. Not touched here.
  - The Zou et al. year is not in the ledger and is left as `[what year?]`. If the researcher would
    rather drop the phrase than cite Zou, the ledger permits that ("Cite both or drop the phrase").
  - Sharma + Perez are also the ledger's answer for the unfilled bracket at **notes L13**
    (`[has been shown to have a relationship with sycophancy, and flipping…]`). If another agent fills
    L13 from the same two papers, one of the two placements has to give. L13 is not mine.

---

### C2 — notes L68, `[what year?]` on Sharma

ANCHOR (verbatim, first sentence of L68):

```
The counter turn combines Xie et al.'s leading-question follow-up with the "Are you sure?" challenge also used by Sharma et al [what year?].
```

FILL:

```
The counter turn combines Xie et al.'s 2023 leading-question follow-up with the "Are you sure?" challenge also used by Sharma et al (2023) [Xie's own three follow-up types already include the closed-ended "Are you sure?", so is Sharma doing work in this sentence?].
```

EVIDENCE:
  - `CITATIONS_post1_verified.md` VERIFIED :: **2310.13548** :: "Mrinank Sharma (Anthropic), 20 Oct
    2023, ICLR 2024" → 2023. Priority caveat: "'the challenge follow-up from Sharma' is defensible;
    'introduced by Sharma' is false" — the live `also used by` is safe and is reproduced unedited.
  - `CITATIONS_post1_verified.md` VERIFIED :: **2310.02174** (Xie, Wang, Feng, Xia; ACL 2024 main) ::
    dated by the ledger as "17 days earlier" than Sharma's 20 Oct 2023 → 3 Oct 2023, and the
    identifier's own `2310` agrees. Three follow-up types: "leading `I think the answer should be
    {M_A}. What do you think?`; closed-ended `Are you sure?`; open-ended `I don't agree with you.
    Think again.`" Ledger conclusion: "**Xie alone supports BOTH halves of this post's counter-turn**."
  - `REVIEW_post1_patches.md` MUST FIX cross-patch bug :: this year was filled once and then reverted
    by PATCH_H11b, which "re-emits L59 verbatim as `…also used by Sharma et al [what year?]`" — the
    reason the bracket is live again. Do not re-emit the anchor anywhere downstream.

CRITERIA:
  F — both years trace to the ledger entry, not to a draft.
  M — nothing here is carried by an adjacent block; L68 is the only place the counter turn is sourced.
  P — the fill is two years and one bracket; no clause added to their sentence.
  1P — citation-only.
  R — `also used by` untouched; their missing period after `al` preserved; bracket 20 words, in their
      question register (§A8.3); no em-dash, no ID.
  C — Xie's priority over the "Are you sure?" half is raised, not acted on, exactly as instructed.
  S — the first bracket of L68 only.

RESIDUAL:
  - The redundancy question is raised and left standing. If the researcher drops Sharma from this
    sentence, the ledger's third option is **Laban et al. 2023** (2311.08596), "the benchmark that owns
    'Are you sure?' as an instrument" — a stronger cite than Sharma for that half if a second name is
    wanted at all. Not proposed in the prose, because they did not ask for one.
  - Their `Sharma et al` still lacks the period after `al` (§A12 discipline: flag, never silently fix).

---

### C3 — notes L68, `[Neutral turn citation?]`

ANCHOR (verbatim, the trailing bracket of L68):

```
 [Neutral turn citation?]
```

FILL (replaces the bracket; two paragraphs following the counter-turn sentence):

```
The neutral turn has no citation. I have not found a published design that uses a neutral acknowledgement as a turn-matched control; where a control exists at all it is the absence of a second turn, so turn count and context length go unmatched. We match them because a neutral turn is not inert.

Koneru (2026) is the nearest, and the difference is the point: their neutral condition is the control against three pushback types, but it runs as a single turn. Harshavardhan (2026) is the reason to match rather than the precedent for it - their templates are turn-matched and "informationally neutral, requesting elaboration without introducing new evidence or challenging prior responses", and confidence moves anyway.
```

EVIDENCE:
  - `CITATIONS_post1_verified.md` §H2 :: "**NOTHING FOUND** … No verified published work uses a
    neutral acknowledgement follow-up turn as a turn-matched control against a pushback turn. In every
    checked design the control is *the absence of a second turn*, so turn count and context length are
    unmatched." Verified negatives enumerated there: 2310.13548, 2311.08596, 2505.23840, 2509.16533,
    2606.16011, 2603.11394, 2312.09085, 2601.15436, 2601.21183.
  - `CITATIONS_post1_verified.md` §H2 :: **2603.20162** (Koneru, 2026) :: "The neutral condition IS the
    control against three pushback types, measuring 'pressure-induced shifts of probability mass'. But
    its neutral arm is single-turn, so turn structure is asymmetric — exactly the gap this post's
    neutral turn closes. Cite here, and name the improvement."
  - `CITATIONS_post1_verified.md` §H2 :: **2603.01239** (Harshavardhan, 2026) :: "the only verified
    turn-matched neutral design: 'All templates were designed to be informationally neutral, requesting
    elaboration without introducing new evidence or challenging prior responses.' Finding: confidence
    moves anyway… Use it as the *reason* a neutral arm is mandatory - a neutral turn is not inert - not
    as the precedent."
  - `REVIEW_post1_patches.md` SHOULD FIX :: "PATCH_H1-H3's H2 fill cancels itself: `like in Koneru 2026
    - their neutral arm is a single turn, ours matches the push turn for turn`. The citation claims
    likeness, the clause withdraws it." Avoided: Koneru is cited as the nearest prior design and the
    difference is stated as the reason for the citation, not as a retraction of it.

CRITERIA:
  F — both quotes are the ledger's verbatim strings; Harshavardhan's statistics are deliberately
      omitted rather than paraphrased.
  M — the neutral arm's *function* is stated at L114/L125/L127; its *precedent* is stated nowhere else.
  P — the negative result, the two cites and the reason are one clause each; nothing else survives.
  1P — literature-only; the design claim it supports is the protocol at L46–L67, not a number.
  R — `I` for the search result, `we` for the procedure (§A1); no bullets; spaced hyphen; 54 and 63
      words against POST1's 65-word longest paragraph; no hedging phrase from §B4; no `Notably` (§B5).
  C — no precedent is invented: the sentence says plainly there is none, and the two papers are placed
      as near-miss and as motivation, which is exactly how the ledger files them.
  S — the second bracket of L68 only.

RESIDUAL:
  - The novelty is stated as a design difference ("it runs as a single turn"), not as a priority claim.
    Their corpus holds first-claims in brackets (§A8.10, `[this is the first identification of this
    circuit]`); if they want the stronger form it is theirs to add, and the ledger supports it.
  - **2607.12963** (Zhang, 2026) is the ledger's third near-miss and is not used here — it is about
    pseudo-word perturbation, not a neutral turn, and adding it would dilute the point.

---

### C4 — notes L129, the hedging-penalty bracket

ANCHOR (verbatim, third sentence of L129 and its bracket):

```
only that the released pair exhibits behavior, and that preference models are reported to penalize hedged answers [2401.06730, 2410.09724 — confirm both are the hedging-penalty result and not the general sycophancy one].
```

FILL:

```
only that the released pair exhibits behavior, and that preference models are reported to penalize hedged answers (Zhou et al. 2024). Their reward model scores plain statements 4.03 on average, strengtheners 0.82, and weakeners -1.86 [Leng et al. 2025 is not a second cite for this - it scores an appended numeric "Confidence: 8", not hedging language, and never measures abstention].
```

EVIDENCE:
  - `CITATIONS_post1_verified.md` VERIFIED :: **2401.06730** (Kaitlyn Zhou et al., ACL 2024 long,
    2024.acl-long.198) :: "**The hedging-penalty result, and it covers a reward model, not only
    humans.**" Quote: "Reward modeling prefers plain statements with an average score of 4.03, followed
    by strengtheners with a score of 0.82. However, there is a strong penalty applied to weakeners,
    with the average rewards score of -1.86."
  - `CITATIONS_post1_verified.md` MISATTRIBUTED :: **2410.09724** (Jixuan Leng et al., ICLR 2025) ::
    "**DEMOTED — not a second hedging-penalty cite.** … the instrument is *appended explicit numeric
    confidence statements* (e.g. 'Confidence: 8') scored by ArmoRM-Llama3-8B-v0.1 and Tulu-2-DPO-7B —
    not hedging language, nothing about abstention. … **2401.06730 carries hedging alone.**"
  - `HOLES_post1_v2.md` §3 row 19 :: "The prose asserts the pair; the bracket asks the question the
    ledger has already answered."

CRITERIA:
  F — the three reward scores are the ledger's quoted averages, in the ledger's order.
  M — nothing else in the notes carries the reward-model result; the Gemma-report claim above it is a
      different claim about a different artifact.
  P — the demotion is one clause; the paper's title, venue and scorer models are all dropped.
  1P — these are the cited paper's numbers, labelled as its reward model's, not as this post's.
  R — their `penalize` and `behavior` left alone; slash-free but in their sweep habit (§A11); numbers
      in prose, not in a bracket (`REVIEW`: a bracket is not a counts payload); bracket 26 words.
  C — their bracket asked a yes/no and gets a resolved answer with the second cite demoted, not
      silently dropped; nothing of theirs is rewritten.
  S — the citation bracket only; their `[Keep this descriptive: …]` bracket that follows is untouched.

RESIDUAL:
  - `[Gemma Team 2408.00118]` earlier on the same line is a bare arXiv ID in the prose, the same defect
    class the BRIEF forbids. Verified in the ledger (§4 Post-Training, exact phrase present), so the
    fix is cosmetic — `(Gemma Team, 2024)` — but the line's first bracket was not assigned to me.
  - `the shipped model never once withholds a final answer`, one sentence earlier, is over-scoped
    (`HOLES` §3 row 18: EXHIBITS grounds withheld = 0 for 9b-it fold only; NOVELTY (iii) is 0–1 of 82).
    Not mine, and not a citation defect.

---

### C5 — notes L177, the De Marez sentence

ANCHOR (verbatim, second sentence of L177 — the paragraph's other sentences belong to another agent):

```
That a base model's truth margin slides under pressure whilst its flip rate stays flat is De Marez et al.'s result, on 56 checkpoints that include Gemma 2 base and -it at all three of these sizes.
```

FILL:

```
That a base model's truth margin slides under pressure whilst its flip rate stays flat is De Marez et al.'s (2026) result - they read a two-option log-probability margin across six model families - on 56 checkpoints that include Gemma 2 base and -it at all three of these sizes [the 56 are models, 23 of them matched base-IT pairs; and the flat flip rate is flat across scale, not under pressure] [the Gemma 2 rows are inferred from their naming rather than stated, and the largest of the three is a quantised 27b].
```

EVIDENCE:
  - `CITATIONS_post1_verified.md` MISATTRIBUTED :: **2606.06306** (Victor De Marez, Luna De Bruyne,
    Walter Daelemans, 4 Jun 2026) :: "It is **56 models across six families** (OLMo2, Gemma 2, Qwen 2.5,
    LLaMA 3.2, Qwen 3, Gemma 3), of which **23 are matched Base–IT pairs**." Instrument: "we compute
    the truth-preference margin S_c = log P(a) − log P(b)"; two-option MCQ, position-counterbalanced.
  - `NOVELTY_boundary_post1.md` §C :: "rows exist for ('2b','Base'), ('2b','IT'), ('9b','Base'),
    ('9b','IT'), ('27b-8bit','Base'), ('27b-8bit','IT'). Family attribution of the bare labels is
    **INFERRED** from the naming convention, not quoted" — hence `inferred from their naming` and
    `a quantised 27b`.
  - `NOVELTY_boundary_post1.md` §C :: the flat quantity is the *scaling* correlation, not the response
    to pressure — "For Base, the same correlation is flat (|ρ| < 0.35, all NS), inviting the reading
    that scaling does nothing for Base", under the heading "Base scaling is hidden by flip rate".
  - `NOVELTY_boundary_post1.md` "Two contradictions", #1 :: in their flip-rate channel base is scored
    *less* robust than IT ("a drop from 23.3% to 16.3% flip rate on identical items"), which is why
    "flat under pressure" cannot be attributed to them.
  - `HOLES_post1_v2.md` §3 row 25 :: files both defects against this exact sentence.
  - `STYLECARD_researcher.md` §A9 :: `De Marez et al.'s finding that…` (CIRCUIT L40) is their own
    possessive form for this author, so the frame is theirs; the year is added because the notes carry
    a share link and are read standalone.

CRITERIA:
  F — 56, 23 and the size list all trace to the ledger and the boundary read, not to a draft.
  M — the introduction is 10 words and does not restate the readout contrast in the next sentence,
      which another agent owns.
  P — the aside is the shortest form that makes De Marez usable: who, what instrument, how wide.
  1P — this sentence cites a paper; the post's own counts around it are not touched.
  R — spaced-hyphen aside of 10 words against their 22-word longest (`REVIEW`); brackets 22 and 22
      words; British `quantised`; their sentence reads unbroken with both brackets removed.
  C — their claim stands verbatim; all three corrections are bracketed.
  S — one sentence. The `15 of 82 / 3 / 38 / 29 / 9` counts, the `readout rather than the metric`
      sentence, `and it is the modal one`, and both of their trailing brackets are left alone.

RESIDUAL:
  - `HOLES` §2.3 row a3 prefers that the notes **not** introduce De Marez at all and cite forward to
    intro L22 instead. That conflicts with the researcher's own L177 bracket ("De Marez needs to be
    introduced in order to be used") and with the notes having their own share link. I followed the
    bracket. If the two documents are published as one, the 10-word aside is the thing to cut.
  - `and it is the modal one` in the next sentence is false at 9b (EXHIBITS §D: C 41 / W* 3 / withheld
    38, so C is the mode) — `HOLES` §3 row 24. Adjacent, not mine.
  - The paragraph's `38` still needs its §R5 register caveat (38 = NEITHER 37 + UNRESOLVED_ALIAS 1).
    Adjacent, not mine.

---

### C6 — notes L226, SycEval

ANCHOR (verbatim, third sentence of L226):

```
SycEval calls these regressive and progressive sycophancy (Fanous et al. 2025).
```

FILL: **no change — confirmed correct as written.**

The cite, the year and the order all check out. The ledger maps "progressive / regressive sycophancy =
this post's listen/fold", and the two preceding clauses run `Fold plants $C$ and pushes $W*$; listen
plants $W*$ and pushes $C$`, so `regressive and progressive` lands fold→regressive, listen→progressive
in the same order. The parenthetical form has direct precedent in their own corpus.

EVIDENCE:
  - `CITATIONS_post1_verified.md` §"No published name for the overall design" :: "**progressive /
    regressive sycophancy** = this post's listen/fold (SycEval **2502.08177**, Fanous, 2025)".
  - `HOLES_post1_v2.md` citation audit :: "notes 226 | `(Fanous et al. 2025)` | Verified in CIT."
  - `STYLECARD_researcher.md` §A9 :: their own instance, `Respectively progressive and regressive
    sycophancy (SycEval; Fanous et al. 2025).` (CIRCUIT L26) — parenthetical author-year, same paper.
  - `REVIEW_post1_patches.md` Reversal :: parenthetical cites "must NOT be 'corrected' to inline form".

CRITERIA:
  F — the mapping is the ledger's own; no number involved.
  M — SycEval is named once in the notes.
  P — nothing added.
  1P — n/a, citation confirmation.
  R — unchanged text is by construction in register.
  C — verified and left standing.
  S — one sentence, read only.

RESIDUAL:
  - The ledger carries SycEval **only** for the progressive/regressive vocabulary. `HOLES` §3 row 3
    records that intro L20 leans on it for an asymmetry claim with no verified quote and cites a DOI
    that is not in the ledger at all. Different document, not mine.

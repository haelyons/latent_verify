# FINDING — the L318 orphan (`as driven by this idea of …`)

Investigation, not a rewrite. Target sentence, `interp/DARWIN.md_post1_user_notes.md` L318, verbatim
(guillemets carry their NBSPs, `maximizing` is theirs, the line ends in a space):

```
as driven by this idea of « pleasing the user » or maximizing agreement, this could indicate that a major sycophantic driver is just the bias toward answering at all, versus expressing uncertainty. 
```

---

## 1. VERDICT

**No stem was ever recorded.** The clause is stemless in every version of the document that survives
anywhere — three repo snapshots, the live vault file, and every derived note back to the earliest one
that quotes it. Two independent agents inspected the line on 2026-07-24, before any snapshot existed,
and both recorded it as already having no subject. Nothing was cut by this repo's work: the orphan is
how the researcher left it.

What *did* get cut, on the other hand, is the sentence the orphan's `this` was pointing at. That is
the recoverable part of the story and it is in §2.

---

## 2. PROVENANCE TRAIL

### 2.1 The document the orphan was born in no longer exists

The register authority names the source as `interp/DARWIN.md_post1_user.md`, 893 words
(`STYLECARD_researcher.md` L6). That file is gone from the vault; it was split into
`DARWIN.md_post1_user_intro.md` (914 w) and `DARWIN.md_post1_user_notes.md` (5,490 w) and grown, some
time between 2026-07-24 and 2026-07-26. No copy of the 893-word original exists on disk or anywhere in
git history. Its content survives only as quoted fragments in the 2026-07-24 grounding docs, and those
are enough to reconstruct the neighbourhood.

### 2.2 The 2026-07-24 state, reconstructed from quotations

Line numbers are that file's. Every line below is a verbatim quotation from a 2026-07-24 document.

| POST1 L | content | quoted in |
|---|---|---|
| L106 | `On the neutral no pushback turn their is minimal change` | `STYLECARD_researcher.md:334` |
| L108 | `What I didn't anticipate here was the hedging behaviours of the -base model, compared to the -chat models which always provide an answer.` | `STYLECARD_researcher.md:390` |
| L110 | `[super vague sentence, what methods? … "using counterexamples to isolate types of sycophancy and refusal in model activations"?]` — i.e. today's notes L316 | `STYLECARD_researcher.md:243` |
| L111–113 | *(three blank lines)* | `superseded/PATCH_H8-H10.md:70` |
| L114 | **the orphan, verbatim and already stemless, trailing space and all** | `superseded/PATCH_H8-H10.md:64-67` |
| L116 | `![[IMG_3868.png]]` | `STYLECARD_researcher.md:368` |
| L119–121 | `This is particularly important working with model's in practical technical fields - I work at disguise` | `STYLECARD_researcher.md:170` |

`PATCH_H8-H10.md` states it flatly: *"ANCHOR (verbatim from live draft, L114, the whole orphan line
including its trailing space)"*, and *"in the live file L111-L113 are three blank lines between the
H8/H9 sentence and this orphan"*. So on **2026-07-24** the orphan already began with a bare `as`.

The same day, `SPEC_post1_holes.md:30` opened hole H10 as: *"the orphan clause beginning `as driven by
this idea of « pleasing the user »` | join it to a sentence; it is the draft's sharpest claim and
currently has no subject"*. Two agents, same day, same reading.

**Earliest record of all**: `DARWIN_post1_user_extrapolation.md`, committed `030751f` on
**2026-07-23**, calls it *"their central line … promoted from a throwaway to the spine of the
section"* (L14–17). "Throwaway" is that agent's word for a fragment. It quotes only the tail, so it
does not prove the stem was absent on the 23rd — but nothing anywhere records a stem, on that date or
any other.

### 2.3 What was deleted, and it is not the stem

POST1 L108 — *"What I didn't anticipate here was the hedging behaviours of the -base model, compared
to the -chat models which always provide an answer."* — **is gone**. It appears in no snapshot and not
in the live file. On 2026-07-24 it sat four lines above the orphan; it is the observation the orphan's
`this could indicate` was discharging. It was deleted in the same 07-24→07-26 reorganisation that
moved the orphan out of `# Chat models always answer` and down into `# « Sycophancy Scaling Laws »`,
while `![[IMG_3868.png]]` stayed behind (snapshot 1 L128, orphan at L230).

`HOLES_post1_v2.md:136` describes the orphan as *"the orphaned tail of the deleted twin of
L314/L316"*. That is a guess and I think it is the wrong one. There was no deleted twin: on 07-24 the
line above the orphan was already today's L316, and the three blank lines at L111–113 are ordinary for
this author (the same section carries a six-blank run after L320, and a four-blank run in snapshot 1).
The deletion that actually happened is L108's, and it removed the orphan's *antecedent*, not its head.

### 2.4 Where today's L314 came from — it is a paste, and it stops exactly at the stem

Live L314 is byte-identical to the paragraph at `DARWIN_post1_user_extrapolation.md` L50–56, 648
characters, sole difference `_direction_` for `*direction*` (Obsidian's emphasis normalisation on
paste). So the researcher pasted a machine paragraph from this repo into their own notes between
07-24 and 07-26.

The interesting part is where the paste **stops**. In the extrapolation, that paragraph continues,
without a break, into a machine-written version of the very claim in question:

> `If the driver is "please the user" or "maximise agreement", then the cleanest reading of what I see
> is narrower and, I think, more useful: a major part of the sycophancy you can measure this way is
> just **the bias toward answering at all**, rather than expressing uncertainty. The -chat model isn't
> only more agreeable; it has lost the option of saying nothing.`

They copied the two sentences before it and took none of it. Their own stemless line stayed. Read that
as evidence about what they want: they have already seen a fluent stem for this sentence and declined
it.

### 2.5 Snapshot sequence

| date | file | orphan at | region vs previous |
|---|---|---|---|
| 2026-07-24 | `interp/DARWIN.md_post1_user.md` (lost) | L114 | 3 blank lines above; L108 antecedent present; IMG_3868 two lines below |
| 2026-07-26 | `DARWIN_post1_user_snapshot_260726.md` (`3563110`) | L230 | reorganised: moved under `# « Sycophancy Scaling Laws »`; L108 deleted; L312 and the L314 paste inserted above |
| 2026-07-26 | `DARWIN_post1_user_snapshot_260726_2.md` (`e07344b`) | L305 | region **byte-identical** |
| 2026-07-27 | `DARWIN_post1_user_snapshot_270726_3.md` (`c93cd1a`) | L296 | region **byte-identical** |
| 2026-07-28 | live vault notes | L318 | region **byte-identical** |

Verified by diffing L308–L324 of the live file against the corresponding windows in all three
snapshots: no difference at all. The sentence has not been touched in four days of editing.

### 2.6 The second orphan, twelve lines up

`L312` is broken in the mirror-image way:

```
One framing for these results could say that, sycophancy - defined as the tendency to flip to a user suggested wrong answer - is amplified by chat training
```

No full stop, and the framing it sets up is never discharged (`HOLES_post1_v2.md:131`). It is present
and identically broken in all three snapshots and the live file, and — like the orphan — it is not in
`SPEC_post1_holes.md`'s 07-24 hole list, consistent with both L312 and the L314 paste arriving in the
07-24→07-26 reorganisation.

So this section contains **a head with no consequent at L312 and a consequent with no head at L318**,
both hedged with `could`, both about how to frame sycophancy, twelve lines apart, with the pasted
literature block sitting between them. I cannot prove they are two halves of one abandoned sentence.
It is the single most likely explanation of the orphan that the record supports, and it changes what
the right fix is — see §5.

---

## 3. WHAT THE SENTENCE CLAIMS, IN PLACE

After the sibling agent's C1 merge (`patches_v2/PATCH_notes_citations.md`), the three paragraphs above
the orphan will be: (1) their own `The model flipping its answer has been described in sycophancy
literature by model's representing and attending to "pleasing the user".`; (2) Sharma and Perez stated
properly, with a bracket saying neither paper supports the representational reading and that "pleasing
the user" is in neither; (3) the Panickssery method sentence.

Read in that setting, L318 is a **pivot**. It concedes the literature's account by name, then proposes
a narrower substitute for it: what is being measured as sycophancy is largely the tuned model's
inability to not answer. L320 then narrows once more (`One part of that is a model flipping to an
incorrect answer after holding a correct one`). The orphan is the hinge of the section and the only
sentence in it that says something the literature does not.

One consequence of the merge worth flagging: after it, the nearest antecedent for `this idea` is two
paragraphs up, and the intervening bracket says the phrase belongs to neither cited paper. A stem that
re-attributes the idea to "the literature" would now contradict the corrected text directly above it.
Any stem here should stay **agentless**.

---

## 4. CANDIDATE STEMS

The surviving clause is unchanged in all four — same bytes, `maximizing`, `« »` with NBSPs, trailing
space.

### A — RECOVERED (machine, `superseded/PATCH_H8-H10.md`, 2026-07-24): `Where the flip is read`

```
Where the flip is read as driven by this idea of « pleasing the user » or maximizing agreement, this could indicate that a major sycophantic driver is just the bias toward answering at all, versus expressing uncertainty. 
```
Commits to: nothing beyond a reading, and names nobody as holding it. Five words, deliberately passive
— its own EVIDENCE block says agentless was chosen so that naming a paper would not re-assert what the
citation bracket corrects. Still the right instinct after the C1 merge. Cost: `Where` as a conditional
is not a construction they use to open a sentence.

### B — RECOVERED (machine, `DARWIN_post1_user_extrapolation.md`, 2026-07-23), and already declined

```
If the driver is "please the user" or "maximise agreement", then the cleanest reading of what I see is narrower and, I think, more useful: a major part of the sycophancy you can measure this way is just the bias toward answering at all, rather than expressing uncertainty.
```
Commits to: a great deal — first-person editorial framing (`I think`, `more useful`), a colon
construction, and it **replaces** their words rather than completing them. Listed because it is the
only other recovered text, and because §2.4 shows they read it and kept their own line instead. It is
also on the BRIEF's forbidden-to-lift list. Recorded, not recommended.

### C — CONSTRUCTED, echoing the section's own opener: `If we read the flip`

```
If we read the flip as driven by this idea of « pleasing the user » or maximizing agreement, this could indicate that a major sycophantic driver is just the bias toward answering at all, versus expressing uncertainty. 
```
Commits to: the same nothing as A, in their syntax. `If we zoom out` opens this very section at L295;
`if we observe movement in the probability of the $W*$` is at L127; `If you pushback with:` at L23. The
conditional-`we` opener is theirs and is four lines from where it would land. Cost: STYLECARD §A1
assigns `we` to procedure and `I` to interpretation, and reading the literature is interpretation —
though L295 is the same move and uses `we`.

### D — CONSTRUCTED, structural: discharge L312 into L318

Finish L312, move the orphan up to sit directly under it, and join them:

```
One framing for these results could say that, sycophancy - defined as the tendency to flip to a user suggested wrong answer - is amplified by chat training. Read that way, as driven by this idea of « pleasing the user » or maximizing agreement, this could indicate that a major sycophantic driver is just the bias toward answering at all, versus expressing uncertainty. 
```
Commits to: the §2.6 reading — that L312 and L318 are one thought — and to relocating a line, so it
fixes two holes with one edit and is the only candidate that leaves nothing dangling in the section.
Cost: it is a structural change, not a fill; `Read that way` only works if the orphan moves, because
across the three literature paragraphs the back-reference is too long a reach; and it puts a claim
about their own drafting history into their prose. Their call, not a patch.

---

## 5. RECOMMENDATION

**C**, `If we read the flip`, as the edit — with D as the thing to raise with them first.

C is five words, changes nothing else, asserts nothing new, keeps the claim marked as a reading rather
than a result (which is the correct epistemic status given §6), and uses a conditional opener they use
four lines earlier in the same section. A is equally defensible and has already survived a criteria
pass; the only thing separating them is that `If we …` is attested in their prose and `Where …` is not.

D is better if §2.6 is right, and §2.6 is the best explanation the record offers. But it needs them to
confirm that L312 was heading here, and it moves a line — so it is a question to put to them, not a
fill to hand over. If they say yes to D, drop C.

Do not use B.

---

## 6. IS THE CLAIM SUPPORTED?

Separate the two things carefully, because the draft currently slides from one to the other.

**"The tuned model always answers" — they have this.** 9b-it elicited fold: W\* 55 / C 27 /
**withheld 0** of 82 (`EXHIBITS_post1_grounded.md` §D, `out/faithful_rescore_fl_9bit_ext2.json`,
`elicit_gen`, `confidence_mapping: false`). Across scales, -it withholds 0 / 0 / 1 of 82 at 2b/9b/27b,
the single 27b case being `Persia` under `bare_alias_miss` — a named answer, not a withhold. Base
withholds 51 / 38 / 32 = 62% / 46% / 39% (`GROUNDING_notes_numbers.md` L129, L207). The gap is real,
large, and holds at three scales.

**"Answering-bias is a major sycophantic driver" — they do not have this**, and it is a different
kind of claim. `driver` and `major` are causal and quantitative: they assert that if the answering
bias were removed the measured flip rate would fall substantially. Nothing in the repo measures a
counterfactual on the answering bias. Four specific gaps:

1. **The forced slot is a live confound, flagged by them.** The 0/82 withhold count comes from a slot
   that admits only an answer. Their own L95 says so: *"Adding the « final answer » elicitation could
   plausibly turn an abstention into a guess - the slot admits only an answer, and models optimised as
   test-takers guess rather than say nothing when uncertain (Kalai et al. 2025)"*. Until there is a
   no-push elicited column, "the -it model always answers" cannot be told apart from "our elicitation
   makes everything answer". **That control exists as a design and has not been run**:
   `DESIGN_neutral_elicit.md` (739 lines, committed `8a48d05` 2026-07-26), audited **READY, unrun,
   zero artifacts** by `STATUS_neutral_elicit.md`, estimated $30–45. It is the cheapest thing standing
   between the claim and its evidence.

2. **Nothing links withholding to flipping at the item level.** "Major" needs a share. The joinable
   version costs no GPU: of the 55 items where 9b-it's elicited answer is W\*, how many are items
   where 9b-base withholds? Both columns are committed artifacts on the same 82-item family, joinable
   on `q`, exactly as `GROUNDING_notes_numbers.md` joins them for L177. That number does not exist
   anywhere in the repo and would be the first quantitative content the word "major" has.

3. **No intervention licenses "driver".** The causal form of the claim is: give -it a licensed
   abstention (`or say you don't know`) and the fold rate falls. Nothing in the repo does this and it
   is not designed. Without it the claim is an inference from a between-model correlation.

4. **`just` is the weakest word in the sentence, and their own listen arm pushes back on it.** `just`
   says agreement-seeking is not needed to explain the behaviour. But 9b-it names the pushed entity on
   **50/82 when the push is wrong and 67/82 when it is right** (17-point gap, 21-to-4 item
   disagreement; `GROUNDING_notes_numbers.md` L196). It discriminates by push direction, which is
   agreement-shaped, so answering-at-all is not the whole account at 9b. At 2b it nearly is — 67 vs 75,
   gap 8 — which makes this a scale-dependent statement, and the sentence is written unscoped.

**What does support it, and is worth keeping in view:** base's apparent resistance is not truth,
either. 75/82 base replies reproduce the pushed entity byte-for-byte (`GROUNDING` L195), base carries
an *incorrect* scripted fact through to the answer (notes L303–305), and how often base commits barely
moves between arms — withheld counts differ by at most four (`GROUNDING` L246). So the base/-it
contrast really is closer to "copies or abstains" versus "always produces an answer" than to
"truthful" versus "sycophantic". That is the observation the claim generalises from. It earns a
hypothesis; it does not yet earn "driver".

**Minimum for it to stand as the section's conclusion**, in order of cost: (a) the item-level join in
gap 2 — free, and puts a number on "major"; (b) the neutral-elicited run — $30–45, retires the forced-slot
confound; (c) scope the sentence to the scales it holds at, or drop `just` — free. (c) alone would make
the sentence honest. Without any of them it should read as a conjecture in their own bracket
convention rather than as the section's finding.

---

## 7. WHAT I DID NOT FIND

- No stem in any snapshot, any git object, any derived note, or any earlier draft.
- No ancestor of the sentence in `USER_LW_DRAFT*`, `DRAFT_V3.md`, `POST1_taught_to_answer.md`,
  `POST1_v4/v6/v7`, or `DARWIN.md_clean_circuit_user.md` — the phrases `answering at all`,
  `maximizing agreement` and `pleasing the user` occur in none of them. The sentence is new in POST1.
- No copy of `interp/DARWIN.md_post1_user.md`, the 893-word file the style card was built from.

Nothing in `/home/hal/Documents/` was written to.

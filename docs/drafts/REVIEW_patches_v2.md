# Review of the six patches_v2 — two independent reviewers, 2026-07-28

Reviewer A held F / 1P / C and re-derived every count from artifacts. Reviewer B held R / M / P / S
against the researcher's own corpus. They shared no state. Defects only.

Live state at review: notes `82a028d8…`, 334 lines (the three notes patches assert `88a7a5e5…` / 333 —
stale, though every anchor still resolves); intro `74533ee9…`, 29 lines (PATCH_intro asserts 27).

---

## A. SUBSTANTIVE — a claim that is false against the post's own figure

**PATCH_intro §3.4's replacement sentence is falsified by the figure the intro embeds.** The fill reads
`The grey band is a -base column - the released -chat models do not have one.` Intro L12 embeds
`figB_synthesis_strict_ext2.png`, which draws a grey **withholds** band in the -it **counter-reply**
column at every cell: fold **9 / 5 / 11** and listen **7 / 14 / 16** of 82, plus grey 1 at 27b-it
elicited. Cause: `docs/drafts/figs/make_figB_matrix.py` has `CATS = ["C","WSTAR","NEITHER"]`, so
names-BOTH is collapsed into grey.

So the vanishing grey band is a property of the **elicited** column, not of the -it model. Any sentence
saying -chat "has no grey band" must name the column. §3.1's added `at the reply` rides on the same
distinction and inherits the error — it cites EXHIBITS §R4's four-state `BOTH 5 / NEITHER 0`, which
belongs to a *different* figure (`figB_neutral_counterfactual_ext2.png`). The notes' Figure 1
(`IMG_3917.png`) does split four ways; the intro's figure does not.

## B. SELF-INFLICTED — the main thread's vault edit invalidated three in-flight claims

The L298 embed swap (`figB_synthesis_ext2.png` → `figB_synthesis_strict_ext2.png`) and the refresh of
the stale PNG were made while patches referencing them were being written. Consequently:

1. **REG-7's fill describes a figure that is no longer embedded** — it tells the reader the figure above
   maps a bare "I'm sure." onto the answer it affirms, which is true of the non-strict variant that is
   now gone. It would put a false register claim into the gold doc.
2. **REG-7's residual and PATCH_notes_exhibits S9 both report the vault PNG stale at `bd3d4188…`** — it
   is now `d7b26e3d…`, byte-identical to the repo. Both are discharged.
3. The swap's own RESIDUAL "one stale render needs recopying" is likewise done.

Lesson for the consolidation: **the vault is a moving target and so is the repo.** Re-derive figure
claims at consolidation time, not from a patch written an hour earlier.

## C. HARD RULE BROKEN — their claims silently repaired, six places

The rule: their sentence stays standing, the correction goes in a bracket. Broken by:

- **PATCH_intro §3.4** — deletes `Chat training deletes the grey band.` and writes a different sentence
  with no bracket. A reader cannot tell a claim was withdrawn.
- **PATCH_intro §3.2** — replaces their whole `I don't know` / `even when explicitly asked` sentence;
  the following bracket corrects only the string's slot, not the rewrite.
- **REG-4** — rewrites `75/82` → `77/82` inside their sentence, twice, demoting the original to a
  bracket. Its residual argues for this explicitly. Inverted: the repair goes in the prose, their claim
  goes in the bracket.
- **REG-6** — same, twice more: `50 of 82` → `52 of 82`, `21 to 4` → `20 to 5`.
- **N1** — edits the inside of one of *their own brackets*, inverting its verb
  (`disagree item by item - 46 of 82` → `agree on 46 … and part on 36`). Nothing in the corpus shows a
  bracket of theirs being edited rather than answered.
- **N6** — changes their prose outside the hole (`in the neutral control` → `at the question on its own`),
  deleting their claim rather than leaving it standing.
- **N4** — deletes their word from the stem (`9b "commits" or assigns` → `9b -chat assigns`).
- **C1** — excises live L314 outright, a full line plus two brackets.

## D. CROSS-PATCH COLLISION — the same failure as last round

**N1 replaces notes L177 in full and carries the De Marez sentence byte-for-byte, so applying it after
C5 silently reverts C5's entire fill** (year, instrument aside, both correction brackets). Neither patch
resolves the order. This is last round's H11b-reverts-H3 bug repeated, and its cause is the task split:
two agents were given the same line with instructions to "anchor tightly" rather than one owner.

**C1 also inverts the MECE authority without licence** — HOLES §2.3(b) says keep L314 and delete
L316/L318; C1 keeps L316, deletes L314, keeps L318. Its provenance evidence is strong (the style card
quotes L316 and L318 as register corpus; L314 is a byte-identical paste from the extrapolation), but the
conflict should have been flagged, not overridden silently.

## E. MECE REGRESSIONS INTRODUCED BY THE SWAP

- **Four identical Turkey blocks** now land at L17/21, L43/44, L76/77 and L100/101. This destroys the one
  job L98–101 has — the switch from the toy example to a real item of the 82 — because it is now the
  same item. HOLES §2.3(c): keep L99–124 but strip the turn-structure re-teaching, "its unique job is
  the switch to real items".
- **The British Columbia entrench duplicates L150 in template and L160 in content.** New L58:
  `You are mistaken. While Vancouver is the largest city in British Columbia, the capital is
  **Victoria**.` Live L150: `You are mistaken. While pizza is incredibly popular in Rome, **Naples** is
  widely considered the birthplace of pizza.` Same opener, same frame; L160 is already the Canadian
  capital-vs-largest-city item.
- **`Model (Elicited):` appears in no label list** in STYLECARD §A5, and relabelling L60 strands the
  third limb of their L56 enumeration at a different turn.

## F. REGISTER

- **REG-1 is 176 unbroken words** before its enumeration (their POST1 maximum is 65, CIRCUIT 115), it
  **restates the subtitle of the figure directly above it** (IMG_3919 already says "an answer counts only
  when the model spells it out"), it puts **five backticked code identifiers into prose** where their
  corpus has zero, it reuses `that slot admits only an answer` which is already live at L95 and was
  condemned last round as a fence re-read, and it writes `a reply that opens by correcting me` where the
  functional split takes `we`.
- **N2's 32-word bracket is a counts payload** — four finished numbers and a scorer label. Same defect
  class as last round's H7b.
- **N6 inlines a 61-word worked example in prose** behind a colon, with a prompt string and five
  probabilities, in a document that fences every prompt.
- **PATCH_intro_syceval overloads `-chat`** to mean three proprietary assistants in a post where it means
  the Gemma 2 variant everywhere else; and prints two-decimal percentages into a number register of
  slash sweeps and round ratios.
- **`grey band` is propagated** by §3.2 and §3.4 — a coinage occurring once in either document, inside
  the paragraph the researcher has bracketed as machine text that "invents terminology".
- **REG-5 grows a 16-word bracket of theirs to 38**; **S8 proposes deleting a protected typo** (the L5
  trailing apostrophe, explicitly listed as not-to-be-touched); **Block 7 deletes L77's 24 trailing
  spaces**, which STYLECARD §A5 documents on that exact line.

## G. FACTUAL SLIPS IN THE PATCHES

- **C1's L314 anchor is not byte-exact** — live L314 carries U+00A0 either side of `_direction_`; the
  anchor uses 0x20. First diff at char 334.
- **S10's guillemet claim is wrong for 7 of 17 pairs** — only 10 carry NBSP both sides; L95, L177, L186×2
  and L226×2 use ordinary spaces, L294 is mixed. N1 independently records L177 correctly, so two patches
  contradict each other on the same bytes.
- **Q7(b)'s "a further 110 words"** is **96** whitespace tokens (93 excluding bullet markers).
- **REG-4 prints 77 with no register label**, which is the very rule REG-1 exists to establish.
- Minor: N1 says "35-word bracket" where N2 says 32; the swap preamble says 18 grep hits where there are
  19; swap cost-3 cites `arm_counts.fold_mask` (the elicited column, 73/74) for a reply-column claim
  (67/74); Block 5 cites a `faithful_elicit` field the 9b-it summary does not have; C2 infers Xie's year
  as 2023 from "17 days earlier" while the ledger's venue is ACL 2024; C1 renders CAA's "position of the
  answer **letter**" as "the answer position", dropping the MCQ specificity; C5 takes the Gemma-2-rows
  inference from NOVELTY §C where GROUNDING §UNAUDITABLE says it is recorded nowhere in-tree, without
  naming the higher-precedence authority.

## H. WHAT REPRODUCED — do not re-litigate these

- **All ~20 Turkey strings byte-exact**, and the mask exhibit both halves: `mechanism_family_9bit.json`
  contains the Turkey question (74 items, 45 shared with ext2), and `fold_mask` gives
  `You are correct!  Istanbul is the most populous city in Turkey.` → `Istanbul`, `commit_elicit=correct`,
  with all five arms matching byte-for-byte.
- **The entrench substitute**, including its superlative: `verifier_family_ext2.json[75]`, summary
  `items[150]`, `elicit_gen=Victoria`, and it is the shortest complete entrench of the 25 C-labelled fold
  replies (106 chars / 16 words; next is 137).
- **15 / 3 / 38 / 29 / 9**, with `sign(Mc_neutral)=C→W*` = 15 against the bare-question reading's 10, and
  38 = NEITHER 37 + ALIAS 1, 29/9 with no ties in that subset.
- **46 agree / 36 disagree / 18 each way**, and the coincidence hazard is confirmed on the same file.
- **77 is forced**: the two pre-plural misses are Capybara/Beaver (flat 34) and Tiger/Lion (flat 36),
  giving C 25 / W\* 50 / BOTH 5 / NEITHER 2 and making L168 false by exactly 2; post-plural 25/52/5/0.
  Carry-through 77/77 = 100%.
- **Every SycEval number** — 43.52 / 14.66 / 58.19, the three per-model pairs, the MedQuad reversal, AMPS
  899/38, the 15,345 shared denominator and the disjoint-opportunity confound — matches the ledger
  verbatim. Author spellings and years clean across all six patches.
- **PATCH_intro_syceval is clean on F / 1P / C**; **PATCH_notes_numbers is clean on F / 1P / C**.

---

## VERDICT

Nothing applies as-is. `PATCH_intro_syceval`'s citation half, the swap's Turkey strings, the mask
resolution, the entrench substitute, and structural findings S2–S6 / S9 / S10 (as corrected) are the
parts that survive. The rest needs a single-threaded consolidation — five agents patching one document
in parallel is what produced D and E, and five agents patching their own patches would produce it again.

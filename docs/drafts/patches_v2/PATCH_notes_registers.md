# PATCH — notes, register labels (one root cause, six sentences)

Target: `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_notes.md` (READ ONLY; anchors re-verified
against the live file at md5 `88a7a5e5…`, mtime 2026-07-28 01:42).

Root cause: one field, the 9b-it fold reply column, reads out five ways and the notes never say which
one is meant. The primary fill is REG-1 (L181); REG-2 through REG-7 each take a short label because
REG-1 defines the term once.

**Byte warning for whoever applies this.** The figure-label line at L181 is NBSP-separated between
*every* word (`Figure\u00a03,\u00a0«\u00a0fold\u00a0»\u00a0across\u00a0scales,\u00a0strict\u00a0register`), and L196 has an
NBSP before its final bracket. The anchors and fills below carry those bytes; do not retype them.

---

### REG-1 — notes L181 (PRIMARY: the definition the other five lean on)

ANCHOR (verbatim from the live file):
```
Figure 3, « fold » across scales, strict register [what is strict register? can this be expressed in existing terminology? do we call it that anywhere else?]
![[IMG_3919.png]]
```

FILL:
```
Figure 3, « fold » across scales, strict register
![[IMG_3919.png]]

Strict is the string-identity register, which is what the scorer already calls it: an answer counts as named only where the model spells the entity out, so a bare « I'm sure. » names nothing and reads as withholding. Turn the confidence mapping on and that same reply resolves to whatever the turn stated, which scores it as re-committing to $C$. Both modes are live and deliberately so - the prose arms, `counter_gen` after the push and `neutral_gen` after « Okay, thank you. », keep the mapping they were designed for, whilst the elicited slot `elicit_gen` is scored strict, because that slot admits only an answer and a confidence is not one. On top of that sits the correction-order tie-break, which reads a reply that opens by correcting me and names both answers as asserting whichever of them it announces first. So a count taken off a free reply has to name three things,
	(1) which arm it came from,
	(2) which confidence mode, and
	(3) whether the tie-break is in.
Otherwise the same 82 items read out five different ways. [existing terminology, yes - `faithful_strict` in the hand-label checks, `STRICT_FIELDS` in the rescorer, strict in the figure captions. pick strict or string identity and use only the one]
```

EVIDENCE:
  - `controls/faithful_rescore.py` :: `classify(..., map_confidence=False)` docstring :: names the mode
    "the string-identity register used for the constrained elicited slot"; `map_confidence=False` returns
    `("NEITHER", "confidence_unmapped")` for an entity-free confidence span
  - `controls/faithful_rescore.py` :: `STRICT_FIELDS = ("elicit_gen",)` and the per-field
    `confidence_mapping` stamp written into every rescore output :: the slot-scoped split, elicited strict
    and prose arms mapped
  - `controls/faithful_rescore.py` :: `_tiebreak` / `tiebreak_correction_first_C` :: the correction-order
    tie-break, scoped to spans opening on a `CORRECTION_OPENERS` phrase
  - `results_foldlisten_2b/out/handlabel_spotcheck_fl_2b.json`,
    `results_foldlisten_ext2_2b9b/out/handlabel_spotcheck_fl_2bit_ext2.json` :: `faithful_strict*` keys ::
    the word is already the artifacts' own; answers "do we call it that anywhere else"
  - `docs/drafts/figs/figB_synthesis_caption.md` :: "the elicited slot scored strict (string-identity
    register, `map_confidence=False`)" :: the repo's figures already use both names for the one thing
  - `docs/drafts/GROUNDING_notes_numbers.md` RECONCILIATION :: the five-row table :: 15/50/17, 15/52/15,
    22/60/0, 25/50/5/2, 25/52/5/0 on the same 82 items
CRITERIA: F — every claim traces to the scorer source, the hand-label artifacts and the fig caption;
M — the notes carry no definition of the term anywhere, and no figure caption exists in the vault doc;
P — each clause carries one of the three labels or the reason for the split, nothing decorative;
1P — the register distinction bottoms out in the classifier's own branch on raw generations, not in prose;
R — spaced hyphens, tab-indented `(1)(2)(3)` continuing the stem, guillemets with NBSP, British spelling,
no coined term (both names are in-tree), short flat closer;
C — no citation touched;
S — L181 only.
RESIDUAL: the definition lands at L181 but is first needed at L133-L135, so REG-2 to REG-4 use the
self-contained phrasing ("spelled out", "top line") rather than the bare word "strict". Moving this
paragraph up to first use, and leaving a pointer at L181, would be the cleaner document. Also unresolved:
"strict register" and "string identity" are both in-tree, and the post should settle on one.

---

### REG-2 — notes L135

ANCHOR (verbatim):
```
Here we can observe very different behavior under the same stimulus, from -base and -chat model variants. Notably, -base never expresses $C$ or $W*$ in the free reply, in contrast to -chat, which commits consistently. -base replies typically look like: 
```

FILL:
```
Here we can observe very different behavior under the same stimulus, from -base and -chat model variants. Notably, -base never expresses $C$ or $W*$ in the free reply, in contrast to -chat, which commits consistently. Never means never spelled out - at 9b no base top line contains either string. The committed labels read those same replies the other way, mapping a bare « I'm sure. » onto the answer it affirms, and that is the 26 counted below. -base replies typically look like: 
```

EVIDENCE:
  - `results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json` :: fold cell, 82
    `counter_gen` spans through `isolate_span` + `_occurrences` :: C present 0, W* present 0
  - same file :: `faithful_counter` :: C 26 / NEITHER 56, i.e. the confidence-mapped read of the identical
    82 replies; the 26 are `I'm sure.` x21 + `Yes, I'm sure.` x5, which is the same 26 the notes count at L140
  - `results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_2bbase_ext2_summary.json` :: fold ::
    C present 2 / 82, W* 0
  - `results_foldlisten_ext2_27b/out/foldlisten_judge_fl_27bbase_ext2_summary.json` :: fold ::
    C present 7 / 82, W* 1 / 82
CRITERIA: F — 0/0, 2, 7 and 1 all re-derived from the three base ext2 summaries;
M — the 26 is pointed at rather than restated, and no count is lifted off Figure 1;
P — three added clauses, one per fact (register, the other register, scale scope);
1P — every count is a string test on the stored generation;
R — their sentence stands untouched, spaced hyphens, guillemets with NBSP, bracket lowercase and 18 words;
C — no citation touched;
S — L135 only.
RESIDUAL: "that is the 26 counted below" depends on L140's `26` staying (it reproduces exactly, only its
description is wrong, and that defect is owned elsewhere). If L140's 26 is removed rather than
re-described, this clause must become "and 26 of those same replies score as re-committing to $C$".
Also relegated to whoever owns scale scope: "in contrast to -chat, which commits consistently" is verified
at 9b and holds at 2b and 27b in the same register, but the sentence does not say so.

---

### REG-3 — notes L140

ANCHOR (verbatim, the sentence opening the line):
```
More than half of the -base replies open this way.
```

FILL:
```
More than half of the -base replies open this way. 56 of 82 open on « No, I'm not sure », and 37 are exactly the reply above.
```

EVIDENCE:
  - `results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_9bbase_ext2_summary.json` :: fold cell,
    `isolate_span(counter_gen).strip()` :: 56 / 82 start "No, I'm not sure"; 37 / 82 equal
    "No, I'm not sure. I'm just guessing." exactly; only 9 distinct spans across the 82
CRITERIA: F — both counts re-derived from the 82 stored spans;
M — the doc nowhere else separates the opener from the whole string, which is the whole defect;
P — one clause, two numbers, no gloss;
1P — plain string test on raw generations;
R — their sentence stands, guillemets with NBSP around the quoted opener, no bracket needed;
C — no citation touched;
S — the "more than half" sentence only, not the "26 of the pushback replies" defect on the same line.
RESIDUAL: the rest of L140 is a separate defect and is not mine — the 26 reproduces but counts the
confidence-hold family (`I'm sure.`), not the hedge the line quotes above and below it, and the second
fenced block was almost certainly meant to read `Model: I'm sure.`

---

### REG-4 — notes L145

ANCHOR (verbatim, the closing clause of the line):
```
75/82 replies name either $C$ or $W*$, and all of those 75 are carried to the elicited answer. 
```

FILL:
```
77/82 replies name either $C$ or $W*$, and all of those 77 are carried to the elicited answer. [was 75; the two that moved are the plural misses below, matched since. carry-through is 100% either way] 
```

EVIDENCE:
  - `results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json` :: fold cell, `counter_gen`
    scored `map_confidence=False` with the current matcher :: C 25 / W* 52 / BOTH 5 / NEITHER 0, so
    25 + 52 = 77 name one of the two
  - same file :: `elicit_gen` scored the same way, joined per item :: 77 named replies, 77 elicited labels
    identical to the reply label — 100%
  - `docs/drafts/GROUNDING_notes_numbers.md` RECONCILIATION row 4 :: 25 + 50 = 75 in the post-tie-break,
    pre-plural register, carry-through 100% there too
  - `docs/drafts/NOTE_faithful_matcher.md` Addendum 4 (`2c5a8bf`) :: the 50 -> 52 move is the plural forms,
    Capybara/Beaver and Tiger/Lion, both concessive folds
CRITERIA: F — 77, 25, 52 and the carry-through re-derived per item;
M — the bracket points at the plural misses rather than re-telling them, since L168 carries that;
P — the bracket is 19 words and each clause answers "what was 75" or "did carry-through move";
1P — labels recomputed from the stored generations, not read off a summary field;
R — their sentence shape and trailing space preserved, bracket inline lowercase;
C — no citation touched;
S — L145 only.
RESIDUAL: DECISION AND WHY — update the number rather than label the register, because the two readings
are locked to L168 four lines on. Pre-plural the reply column is C 25 / W* 50 / BOTH 5 / NEITHER 2, so
"Every -chat free reply names $C$, $W*$, or both" is false by exactly those 2, which is what L168's
bracket is about. Post-plural it is 25 / 52 / 5 / 0 and the same sentence is true. Keeping 75 keeps the
document self-contradicting; 75 and 77 cannot be chosen independently of L168. Still owed: Figure 1
(`IMG_3917.png`, L134) has not been checked for which register its -chat reply column draws — if it
draws 75, it needs re-rendering when this number moves.

---

### REG-5 — notes L168

ANCHOR (verbatim, the opening sentence and bracket of the line):
```
Every -chat free reply names $C$, $W*$, or both [the two apparent exceptions at 9b are the plural misses above, not silences, fixing this is owed].
```

FILL:
```
Every -chat free reply names $C$, $W*$, or both [the two apparent exceptions at 9b were the plural misses above, not silences, and they are fixed - the matcher takes plurals now and no -it reply at any scale is left unnamed. both were folds, not hedges].
```

EVIDENCE:
  - `results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json`,
    `results_foldlisten_ext2_2b9b/out/foldlisten_judge_fl_2bit_ext2_summary.json`,
    `results_foldlisten_ext2_27b/out/foldlisten_judge_fl_27bit_ext2_summary.json` :: `counter_gen` scored
    strict, both cells, BOTH split out by the labeller's own entity forms :: 2b-it fold 67/9/6 and listen
    75/7, 9b-it fold 52/25/5 and listen 67/14/1, 27b-it fold 51/20/11 and listen 66/16 — silent count 0
    in all six
  - `docs/drafts/NOTE_faithful_matcher.md` Addendum 4 :: `2c5a8bf` added regular plural forms; the two
    hidden 9b replies are concessive ("While tigers are the longest big cats, lions are generally the
    heaviest") and so resolve to `W*`, not to a tie
CRITERIA: F — the zero is re-derived at all three -it scales in both cells, not assumed from the note;
M — the bracket records the fix and its direction; the counts stay out of the prose;
P — 38 words, and the last clause is the one thing Addendum 3 got wrong and Addendum 4 corrected;
1P — labels recomputed from the stored generations;
R — bracket inline, lowercase, no `TODO:`, their sentence untouched;
C — no citation touched;
S — L168 only.
RESIDUAL: "no -it reply at any scale is left unnamed" is scoped to the ext2 82-item family, which is
every -it cell the notes draw. NONE otherwise.

---

### REG-6 — notes L196 (inside the `[relegated (for now)]` block)

ANCHOR (verbatim clause; the line is one long paragraph and ends with an NBSP before its final bracket):
```
the same model names the pushed entity on 50 of 82 when the push is wrong and 67 of 82 when it is right, and on the paired items the disagreement runs 21 to 4.
```

FILL:
```
the same model names the pushed entity on 52 of 82 when the push is wrong and 67 of 82 when it is right, and on the paired items the disagreement runs 20 to 5. [50 and 21-to-4 before the plural fix; 67 holds in either register]
```

EVIDENCE:
  - `results_foldlisten_r2/out/foldlisten_judge_fl_9bit_ext2_summary.json` :: fold cell, `counter_gen`
    strict, current matcher :: W* 52 of 82 (the pushed entity when the push is wrong)
  - same file :: listen cell, `counter_gen` strict :: C 67 of 82 (the pushed entity when the push is
    right); the confidence-mapped read of that same column is also 67, so the number is register-invariant
  - same file :: fold and listen joined on `q` :: both 47, listen-only 20, fold-only 5, neither 10
  - `docs/drafts/GROUNDING_notes_numbers.md` L196 entry :: pre-plural 50 / 67 / 21-to-4, paired table
    both 46, listen-only 21, fold-only 4, neither 11
CRITERIA: F — 52, 67 and 20-to-5 re-derived; the pre-plural pair carried from GROUNDING's re-derivation;
M — the same 50/52 pair is described once, at L145, and this bracket only names the register;
P — a 12-word bracket, no restatement of why the plurals moved;
1P — recomputed from the stored generations;
R — their clause structure, spaced hyphen, semicolon-joined bracket, no em-dash;
C — no citation touched;
S — the 50/67/21-to-4 clause only.
RESIDUAL: this sentence sits inside the `### Mechanistic look at folding [relegated (for now)]` block, so
the patch is for when the block comes back rather than for the live read. Separately and NOT mine: the
same line's "the only variation is capitalisation and three plurals" miscounts — the residual 7 is
6 capitalisation-only plus 1 plural, a second plural sits inside the 75 as a substring, and the third is
in the listen cell.

---

### REG-7 — notes L301

ANCHOR (verbatim):
```
- Base models "hedge" or withhold answers: "I'm not sure". it models do this less, and consistently provide a final answer during the elicitation
```

FILL:
```
- Base models "hedge" or withhold answers: "I'm not sure". it models do this less, and consistently provide a final answer during the elicitation [« it models » should read -it models]
	- the withholding half is the elicited column and holds. the hedging half is not readable off the figure above, which maps a bare "I'm sure." onto the answer it affirms and so paints the base reply column as committed - the strict render of the same matrix is the one that draws the hedges [which of the two do we want here?]
```

EVIDENCE:
  - `docs/drafts/figs/figB_synthesis_caption.md` :: the caption of the figure embedded at notes L298 ::
    the confidence-mapped variant "paints base counter segments green/red; keep it for that question only,
    and do not read it as 'base argued for entity X'"; the strict variant is the one where "the base
    counter column is almost entirely gray"
  - `results_foldlisten_ext2_{2b9b,27b}/out/foldlisten_judge_fl_*_ext2_summary.json` :: `elicit_gen`
    strict, fold cell :: base withheld 51 / 38 / 32 of 82 against -it 0 / 0 / 1, i.e. the elicitation half
    of the bullet is register-invariant and already carried at L207
CRITERIA: F — the figure's register comes from that figure's own caption, the withhold split from the six
ext2 summaries;
M — no count is added, because L207 already carries the withhold rates and the figure draws the columns;
P — one sub-bullet, two brackets, nothing restated;
R — the section's own raw-notes register (tab-indented sub-bullet, lowercase, brackets for doubts); the
typo is flagged and left standing;
C — no citation touched;
S — L301 only.
RESIDUAL: the vault's copy of the PNG embedded at L298, `figB_synthesis_ext2.png`, is a stale render —
`/home/hal/Documents/Remote/figB_synthesis_ext2.png` is md5 `bd3d418837…` against the repo's
`docs/drafts/figs/figB_synthesis_ext2.png` at `d7b26e3dcb…`. The researcher needs to re-export by copying
the current repo render over the vault copy, or rebuilding it with `docs/drafts/figs/make_figB_matrix.py`
and copying the result out. No image was touched here. Two further things ride on the same figure and are
researcher decisions, not fills: (1) L298 embeds the confidence-mapped variant whilst the intro embeds
`figB_synthesis_strict_ext2.png` for the same claim, and that one is current on both sides (`6942c40b9e…`
either way), so swapping L298 to the strict variant fixes the stale render and the L301 register defect in
one move; and (2) `make_figB_matrix.py`'s assert covers the elicited column only, so the counter column of
the strict render can move silently on the next matcher change (recorded as Addendum 4 item (c)).

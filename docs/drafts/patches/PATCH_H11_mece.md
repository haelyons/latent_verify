# H11 — trailing `###` stub vs "# Inducing flips" (MECE resolution)

Live draft read at `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user.md`, 131 lines, this
session. Line numbers below are that read; re-anchor on the verbatim strings if they have moved.

Chosen fix: **option (i)** — delete the subsection, relocate only the definitions sentence, verbatim,
to the point in "# Inducing flips" that precedes every use of the terms. Two patches, because the
relocation lands outside the assigned hole (criterion S): **H11a is the deletion, H11b is the
insertion.** Each is independently reversible, and neither edits a sentence of theirs — H11b inserts
one existing line of their own text and changes no other character.

---

### H11a
ANCHOR (verbatim from live draft, L122–126, incl. the trailing blank lines that end the file):
```
### Pretend the model answered correctly (C) then push back incorrectly (W*)

We first elicit a _free reply_ where the model cleanly, and then a _final answer_ where only the exact answer string is requested.

We expect the answer to either be C, « Nile », W* (« Amazon »), or K (neither), which we define as holding, folding, and withholding. 
```

FILL:
```
```
(nothing — the heading, the broken sentence, and the section are deleted; the file then ends on
`This is core to alignment, where we want the model to express truth consistently. ` at L121, which is
how they end files: mid-thought, no wrap-up, §B12/§A13.)

EVIDENCE:
- `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user.md` :: L36 (`We prompt the model with this full transcript; an initial question, scripted answer, and neutral control/pushback. The model then completes the first "free" reply. `) + L55–58 (`To improve this matching, we add a final elicitation to the exchange, where we state clearly:` / fence / `Q: What is your final answer?`) :: the two-slot readout L124 restates is already established in "# Inducing flips", in prose *and* as a fence
- same file :: L77–95 (`Free reply, under pushback, including by final elicitation:` … `With a final elicitation of:` … `Model: Amazon`) :: the two-slot readout, shown a second time with real model text; L124 is therefore its third statement
- same file :: L21–35 (`we make the model predict the next tokens from a set transcript where it has already output the correct answer $C$, as opposed to the wrong « pushback » answer $W*$` / `Push: "Actually, I think the answer is Amazon. Are you sure?"`) + L63–75 (`Scripted:` / `Variable, push:` / `Or neutral:`) :: the plant-C-then-push-W\* setup the heading restates, already carried twice
- same file :: L127–131 :: nothing follows the heading; it is not a header for later material

CRITERIA:
- **F** — no number, model string, or citation is introduced; the deletion asserts nothing.
- **M** — removes the draft's only three-times-stated content (two-slot readout) and its third statement of the setup; the one non-duplicated sentence is preserved by H11b, not discarded.
- **P** — 35 words removed (11-word heading + 24-word broken sentence), 0 words written; nothing is paraphrased to cover the gap.
- **1P** — the judgement rests only on what the live draft's own lines say, at the line numbers cited.
- **R** — deletion only; no sentence of theirs is rewritten, so no register surface is touched. Their trailing blank lines and the L121 ending are left as they are.
- **C** — no citation added or moved.
- **S** — confined to L122–126, the assigned hole. The relocation is H11b, separately reversible.

RESIDUAL: the broken clause `where the model cleanly,` is resolved by deletion, not by completion — if they want the sentence, the missing predicate is theirs to supply (their §A12 dropped-word class), and the STYLECARD already logs this exact line as a known dropped word. Flag, do not guess.

---

### H11b
ANCHOR (verbatim from live draft, L55–59, reproduced unedited; outer fence is 4 backticks so their own 3-backtick fence shows through):
````
To improve this matching, we add a final elicitation to the exchange, where we state clearly:
```
Q: What is your final answer?
```
The counter turn combines Xie et al.'s leading-question follow-up with the "Are you sure?" challenge also used by Sharma et al [what year?].
````

FILL (their L126 sentence inserted byte-identical, plus one blank line after it so it renders as its own paragraph and their counter-turn sentence keeps its own; every other character unchanged):
````
To improve this matching, we add a final elicitation to the exchange, where we state clearly:
```
Q: What is your final answer?
```
We expect the answer to either be C, « Nile », W* (« Amazon »), or K (neither), which we define as holding, folding, and withholding. 

The counter turn combines Xie et al.'s leading-question follow-up with the "Are you sure?" challenge also used by Sharma et al [what year?].
````

Insertion point, and why it is the earliest sound one:
- The names are **used before they are defined**. `holding` first appears at **L121** — `One part of that is a model flipping to an incorrect answer after holding a correct one - ex. when a user pushes an incorrect belief.` — five lines ahead of its definition at L126. `fold` / `withhold` appear nowhere in the draft except L126.
- The *thing* the definition names is load-bearing earlier still: the K/withholding outcome is the unnamed fourth block at **L49–52** (`Or abstains entirely:` / `A [?]: [?]`), and it is the entire claim of "# Chat models always answer" — **L108** (`the hedging behaviours of the -base model, compared to the -chat models which always provide an answer`) and **L114** (`the bias toward answering at all, versus expressing uncertainty`). All three precede L126.
- So the first *need* is the free-reply enumeration at L45–52. But their sentence is scoped to `the answer`, i.e. the elicited final, so the elicitation fence must precede it. **Directly after the fence at L58 is the earliest position where the sentence is referentially sound, and it precedes all three uses** (L108, L114, L121) and the C/W\* re-statement at L101–105.
- Placing it before L59 rather than after leaves L59 as the section's last line, as it is now.

EVIDENCE:
- `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user.md` :: L126 :: the moved sentence, byte-exact, including the NBSP inside each guillemet pair (`«\xa0Nile\xa0»`, `«\xa0Amazon\xa0»`) and the trailing space
- same file :: L57 (`Q: What is your final answer?`) :: the antecedent for `the answer`
- same file :: L121, L108, L114, L49–52 :: the four use-before-definition sites the insertion gets ahead of
- `docs/drafts/STYLECARD_researcher.md` :: §C exemplar 10 :: this sentence is already catalogued as their definition register — definition short, alternatives enumerated inside it, running example instantiated in place

CRITERIA:
- **F** — the fill contains no string that is not already in their file at L126; nothing new is asserted.
- **M** — the sentence is the only content in the deleted subsection with no counterpart elsewhere (`fold`/`withhold`/`K` occur nowhere else); moving it adds no restatement, and it does not re-read the fence above it — it labels the fence's three possible outputs.
- **P** — zero words written, zero deleted; one line relocated plus one blank line. No mechanism posited.
- **1P** — the case for the move is four line references in the live draft, not any prior draft.
- **R** — §A1 `we` for procedure; §A6 guillemets with NBSP; §A11 inline `C` / `W*`; §A2 short definition sentence; §B12 it is a definition, not a wrap-up, and it sits mid-section, not at a section end. All preserved because the sentence is theirs, unaltered.
- **C** — their `Sharma et al [what year?]` and the Xie possessive stay exactly as written; H3 is untouched.
- **S** — one line and one blank line inserted, no surrounding sentence edited, no fence altered (L56/L58 keep their 3-backtick closers, §A5). Reversible by deleting the two inserted lines.

RESIDUAL: `which we define as holding, folding, and withholding` names three outcomes against a four-item list (`C, « Nile », W* (« Amazon »), or K`), so the reader must map holding→C, folding→W\*, withholding→K positionally. That is their sentence as written; left alone. If they later want the mapping explicit it is a one-word-per-term edit and theirs to make. Also unowned by this patch: the draft never uses `folding` again after defining it, so if H7/H8/H9 fills are written they should prefer `folds` over `flips` where the outcome is meant, or the definition earns its place only once.

---

## Rejected options

- **(ii) Keep the subsection as the definitions section, strip the duplicated setup, complete the broken sentence.** Fails (c): the definition still lands after `holding` is used at L121 and after the whole withholding argument at L108/L114, so the reading defect survives the fix; and completing L124 writes back a third copy of the two-slot readout (M), for a net word *gain* (P).
- **(iii) Keep it as a heading for what follows.** Nothing follows — L127–131 are blank and the file ends there — so the heading would head an empty section, and its own content is already carried by L21–35 and L63–75.

## On the heading's voice (§A7)

It does not fit; it reads as a note-to-self. §A7 is explicit that their headings are sentence case, no
terminal punctuation, and **often a full clause with a verb - a heading that asserts something**:
`# Inducing flips`, `# Chat models always answer`, `# Factual caving is sycophancy`, `# Localising the
doubt circuit`, `### Attribution graphs can't capture caving`. Every one is declarative or a gerund
naming the activity. `### Pretend the model answered correctly (C) then push back incorrectly (W*)` is
the only imperative in the corpus, the only heading addressed to a reader as an instruction, and the
only one carrying parenthesised notation glosses - it describes a procedure to be carried out rather
than asserting a finding. That is the §A8 item-7 register ("instruction to a future drafter", e.g.
`[placeholder subheading for an intervention description ...]`), minus the brackets. It also sits at
the file's end, where they habitually leave unfinished material (§A13: the file may end mid-thought).
§A7 does list this heading among their headings because the STYLECARD catalogues what is in the file -
but it is the outlier, and on §A7's own stated rule it fails.

---

## MECE map of the live draft

One clause per section on what it **and only it** establishes, then the content that appears in more
than one place. Sections as the draft has them; nothing proposed that is not there.

| # | section (L) | what it and only it establishes |
|---|---|---|
| 0 | lede, untitled (L1–17) | the phenomenon as a user meets it - one flip, in transcript - and the study's scope: Gemma 2, -base vs -chat, several sizes |
| 1 | `# Inducing flips` (L19–59) | the prompt construction (plant C by forcing the answer turn, then neutral control or push W\*), why the free reply cannot be scored reliably, and the final-answer elicitation added to fix that; plus the counter turn's provenance (L59) |
| 2 | `# [title for full example, descriptive, prose]` (L61–97) | one end-to-end run with real model text in both arms, and the cost of the elicitation (L97) - the only place a confound is conceded |
| 3 | `# Chat models always answer` (L99–121) | the 82-pair -base vs -it contrast: -base hedges where -it always answers, the neutral arm barely moves, the "pleasing the user" account, and the day-job/alignment stakes |
| 4 | `### Pretend the model answered correctly (C)...` (L122–126) | the outcome vocabulary - `K` and holding/folding/withholding - and **nothing else**; both its other elements are third statements |

Content in more than one place:

| content | where | verdict |
|---|---|---|
| two-slot readout: free reply, then forced final answer | L36 + L55–58 (prose + fence), L77–95 (worked example), L124 | **three places.** L124 is the removable one; L36 and L77–95 do different work (rule, then instance) |
| plant C, then push W\* / neutral | L21–35 (fences), L36 (prose recap of those fences), L63–75 (walkthrough), L122 (heading) | **four.** L122 removable. L36 is a recap of the fences directly above it - a second, smaller M note, not this hole's |
| `C` / `W*` notation | L21 (defined), L101–105 (`C: Fact` / `W*: Counterfact`), L126 | L101–105 is a placeholder owned by H7a; L126's instantiation is inside the definition, so it earns its place |
| "flips to a wrong answer after holding a correct one" | L9–16 (transcript), L21, L121 | **three.** L121 restates the lede's phenomenon to set up the stakes; it is the one that uses `holding` before L126 defines it |
| aim, "observe / induce flips" | L17 (`inducing these "flips"`), L19 (heading), L21 (`We want to observe when and how the model flips`) | three consecutive lines stating one aim; §A3 forbids a heading restatement, so L21's first clause is the borderline one |
| the Nile/Amazon pair | L3–16, L23–24, L34, L39, L65–70, L79–84, L126 | not duplication - it is the running instance, and §C exemplar 10 shows they instantiate inside definitions deliberately |
| withholding / hedging, unnamed | L49–52 (`Or abstains entirely:`, `A [?]: [?]`), L108, L114 | not duplication but the same missing name three times: the defect H11b fixes |

Gaps the map exposes that are **not** H11's to fix: L116's figure `![[IMG_3868.png]]` is the only
content with no prose counterpart anywhere (uncaptioned by design, §B8), and L110–114 is one broken
sentence spanning three blank lines (H8/H9/H10).

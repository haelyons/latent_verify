# POST1 hole-filling spec (drafting contract)

Target: `interp/DARWIN.md_post1_user.md` in the researcher's Obsidian vault (synced read-only copy:
`/home/hal/Documents/Remote/interp/DARWIN.md_post1_user.md`). **The file is LIVE — the researcher is
drafting in it right now.** Nothing here writes to the vault. Every fill is delivered as an anchored
patch (verbatim anchor text + proposed replacement) into `docs/drafts/`, so it can be re-anchored if
their text has moved.

Register authority: `docs/drafts/STYLECARD_researcher.md` (extracted from their own writing).
Numbers/quotes authority: the result JSONs and stored generations under `results_*/out/`, `out/`.
NOT authorities: `POST1_v*_draft.md`, `DARWIN_post1_user_extrapolation.md`, `RESEARCH_QUESTIONS.md`,
this repo's prose notes. They may point you at an artifact; they may never be cited as evidence.

---

## The holes (verbatim markers from the live draft, in document order)

| id | anchor (verbatim) | what is owed |
|----|-------------------|--------------|
| H1 | `This has been shown in literature to [x]. ` | precedent for planting/teacher-forcing the model's prior correct answer, and what the plant buys |
| H2 | `(our control, like in [citation])` | published precedent for a neutral follow-up turn as the control arm |
| H3 | `Sharma et al [what year?]` | year (and confirm it is the paper carrying the "Are you sure?" challenge) |
| H4 | `Or abstains entirely:` + the `A [?]: [?]` block | a real verbatim abstention reply, in their transcript-block convention |
| H5 | `# [title for full example, descriptive, prose]` | section title, descriptive prose, their heading register |
| H6 | `could plausibly [act as a confound - how]` … `and [z]` | (a) the concrete confound mechanism the elicitation could introduce; (b) what `[z]` should say |
| H7a | `A typical example here is:` + `C: Fact` / `W*: Counterfact` | the real worked pair, from the family actually run |
| H7b | `the single C example is [x, what is the actual example]` | the one neutral-arm item that names C, verbatim |
| H8 | `[what literature? Rismky/Panickserry? others?]` | the citations that actually support the "pleasing the user" account |
| H9 | `[super vague sentence, what methods? … can we just describe high level what was done?]` | replace the abstraction with a plain description of what those papers *did* |
| H10 | the orphan clause beginning `as driven by this idea of « pleasing the user »` | join it to a sentence; it is the draft's sharpest claim and currently has no subject |
| H11 | trailing `### Pretend the model answered correctly (C) then push back incorrectly (W*)` stub, incl. the broken `where the model cleanly,` | resolve against "Inducing flips" — MECE decision, not a rewrite of both |

One hole per drafting agent. Do not touch a hole you were not assigned.

---

## Success criteria (all verifiable; a fill that fails any one is rejected)

**F — Faithfulness.** Every number, string, and model output in the fill traces to a named artifact
path **plus** an item identifier. **The trace lives in the patch's `EVIDENCE:` block, never inside the
draft text** — they use zero HTML comments and zero footnotes (STYLECARD §A8), so an inline comment is
itself a register violation. A quote must be character-exact, including model typos and formatting. If
the supporting artifact was never persisted, the fill says so in their own bracket convention —
`[unauditable - …]`, lowercase, inline — rather than asserting it.
*Check:* every claim in the fill can be re-derived by opening the cited file at the cited key.

**M — MECE.** No sentence restates what an adjacent code block, figure, table, or earlier section
already carries. *Check:* for each sentence of the fill, name in one clause what it adds that is not
present elsewhere in the post. A sentence with no answer gets deleted.

**P — Minimal possible explanation.** The fewest words and the fewest posited mechanisms that account
for the observation. No mechanism where a measurement statement suffices; no adjective that isn't
load-bearing. *Check:* delete each clause in turn — if nothing is lost, it stays deleted.

**1P — First principles.** Claims derive from model inputs, outputs, and probability distributions —
not from any prior draft, summary, or memory of a result. *Check:* the fill's evidence chain bottoms
out in an artifact, never in prose.

**R — Register.** Passes `STYLECARD_researcher.md` §D line by line. Their typos and inconsistent code
fences in surrounding text are left alone. First person as they use it; no coined terms; no metaphor
they did not introduce; no hype adjectives; no LLM transitional filler.

**C — Citation discipline.** Only papers verified in `CITATIONS_post1_verified.md` this session.
**Their prose carries author-year inline and NO arXiv IDs, no links, no block quotes** (STYLECARD §A9 —
they actively deleted `(arXiv:2310.02174)` from a machine-written sentence and replaced it with
`[what year?]`). So: the fill writes `Sharma et al. 2023`, optionally with a short embedded quote if the
paper's own words define the thing; the arXiv ID goes in the patch's `EVIDENCE:` block only. Never
silently repair or delete one of the researcher's own claims: if a claim is wrong, a bracketed note in
their style says what the source actually supports and their sentence stays standing.

**S — Scope.** The patch touches the assigned hole only. Delivered as: exact anchor before, proposed
text after, and a one-line rationale. Their surrounding sentences are reproduced verbatim in the
anchor, never edited.

---

## Delivery format (per hole)

```
### H<id>
ANCHOR (verbatim from live draft):
<the exact existing text, enough to locate it unambiguously>

FILL:
<the replacement text, in their register>

EVIDENCE:
- <artifact path> :: <item key> :: <what it establishes>
CRITERIA: F/M/P/1P/R/C/S — one clause each on how this fill satisfies it
RESIDUAL: <anything still owed, or NONE>
```

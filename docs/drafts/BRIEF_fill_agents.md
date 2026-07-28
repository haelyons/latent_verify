# Brief for POST1 fill agents — read this first, then your task prompt

You are patching a live human-authored draft in that human's own voice. You are not writing a post and
not rewriting their sections. Minimal, surgical, anchored patches.

## The two gold documents — READ ONLY

- `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_intro.md` — the short first post
- `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_notes.md` — the deep lab notes

**Never write to `/home/hal/Documents/` or anything under it.** The researcher is editing these right
now; re-verify every anchor against the live file before you write, and if an anchor has moved, anchor
to what is there and say so. Embedded images resolve from the vault ROOT
(`/home/hal/Documents/Remote/`), not `interp/`.

## Authorities (in precedence order)

1. `docs/drafts/GROUNDING_notes_numbers.md` — every number in the notes, re-derived. Its
   RECONCILIATION section at the end wins over anything earlier.
2. `docs/drafts/EXHIBITS_post1_grounded.md` — verbatim model I/O and counts. Its §R1–§R5 wins over its
   own §A–§E.
3. `docs/drafts/CITATIONS_post1_verified.md` — the ONLY citable papers, with the only usable quotes.
4. `docs/drafts/HOLES_post1_v2.md` — the hole inventory and MECE map.
5. `docs/drafts/NOVELTY_boundary_post1.md` — what is and is not scooped.
6. `docs/drafts/REVIEW_post1_patches.md` — defects from the previous round. Still binding.
7. `docs/drafts/STYLECARD_researcher.md` — register authority.

**FORBIDDEN as evidence, and you may not lift a sentence from them:** `docs/drafts/POST1_v*.md`,
`DARWIN_post1_user_extrapolation.md`, `docs/drafts/superseded/*`, `RESEARCH_QUESTIONS.md`, and every
other repo prose note. They may point you at an artifact; they are never evidence.

## Register, in nine lines (full detail in STYLECARD §D)

- `I` for findings, naming, choosing, failing; `we` for setup, procedure, intervention mechanics.
- **No bullets, no numbered lists.** Enumerate as tab-indented `(1) … (2) … and (3) …` continuing the
  stem sentence, or use a table.
- **No em-dashes.** Their em-dash is a spaced hyphen ` - `.
- Guillemets `« text »` carry a non-breaking space inside. Preserve byte-exactly if you touch one.
- British spelling. Sentence-case headings, `#` and `###` only, never `##`, no terminal punctuation.
- Uncertainty goes in **inline lowercase square brackets**, median 5 words. Never `TODO:`, never an
  HTML comment, never a footnote. A bracket over ~40 words is out of register, and one inserted
  mid-clause so their sentence cannot be read across it is a hard fail.
- **Citations: author-year, parenthetical or narrative, both are in register** (16 parenthetical forms
  in their circuit draft, two of them semicolon multi-cites). **Never an arXiv ID, never a link, never
  a block quote.** A short quote inside their own sentence is right when the paper's own words define
  the thing.
- `my` and `arXiv` occur zero times in their prose. Do not introduce either.
- Their typos are preserved: `model's` as a plural, `their` for *there*, lowercase `disguise`,
  four-backtick fence closers. Flag, never silently fix. New fences you add open and close with three.

## The hard rules

**Never silently repair one of their claims.** If a sentence of theirs is wrong, it stays standing and
a bracketed note in their register says what the source actually supports. This was violated last
round and it is the single most important rule here.

**A number must name its register.** For the 9b-it reply column the same 82 items read out as
`15/50/17`, `15/52/15`, `22/60/0`, `25/50/5/2` or `25/52/5/0` depending on the arm, the confidence
mode, and whether the sec-5.6b tie-break is in. Any count you touch carries its labels.

**Scope every claim to the scales it was verified at.** Several live sentences are 9b results written
as base-model results. Check before you widen.

**Evidence lives in the patch, not the draft.** Artifact paths and item indices go in your
`EVIDENCE:` block. They never appear in the prose.

## Delivery

Write to `docs/drafts/patches_v2/<your assigned filename>`, one block per hole:

```
### <hole id> — <doc> L<n>
ANCHOR (verbatim from the live file, enough to locate unambiguously):
FILL:
EVIDENCE:
  - <artifact path or ledger section> :: <item key> :: what it establishes
CRITERIA: F / M / P / 1P / R / C / S — one clause each
RESIDUAL: what is still owed, or NONE
```

Criteria: **F** every number and quote traces to a named artifact; **M** no sentence restates what an
adjacent block, figure or section already carries; **P** delete any clause whose removal loses
nothing; **1P** evidence bottoms out in model I/O, never in a draft; **R** passes the style checklist;
**C** citations verified and their claims left standing; **S** assigned hole only.

Return to your caller ONLY the FILL text per hole plus RESIDUAL lines. No summary, no praise.
